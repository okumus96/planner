import torch
import torch.nn as nn
import torch.nn.functional as F
from GameFormer.predictor import GameFormer # Orijinal modelinize dokunmadan çağırın
from .predictor_modules import *

import torch
import torch.nn as nn
from .predictor_modules import CrossTransformer, FutureEncoder

class ModeSelector(nn.Module):
    def __init__(self, dim=256, num_lat=5, num_lon=12, feature_dim=6):
        super(ModeSelector, self).__init__()
        self.dim = dim
        self.num_lat = num_lat
        self.num_lon = num_lon

        # 1. Enlemsel Rota Kodlayıcı (PointNet) -> [N_r x D_m] to [D]
        # feature_dim = 6 (x, y, yaw, curvature, v_max, occupancy)
        self.lat_encoder = nn.Sequential(
            nn.Linear(feature_dim, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, dim)
        )

        # 2. Boylamsal Mod için Embedding ARTIK YOK. (Makaleye göre deterministik hesaplanacak)

        # 3. 2D'den D'ye Düşüren Linear Katman (Alignment Layer)
        self.mode_proj = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.ReLU()
        )

        # 4. Komşu top-1 future encoder (decoder argmax trajectorylerini D-dim'a çevirir).
        # GF predictor_modules'taki FutureEncoder ile aynı mimari; agirliklari burada
        # ayri (mode_selector ile birlikte ogreniliyor).
        self.neighbor_future_encoder = FutureEncoder()

        # 5. Çapraz Dikkat Mekanizması ve Skorlama
        self.query_encoder = CrossTransformer(dim=dim)
        self.score_mlp = nn.Sequential(nn.Linear(dim, 64), nn.ELU(), nn.Linear(64, 1))

    def forward(self, scene_encoding, c_lat, scene_mask=None,
                neighbor_top1_futures=None, neighbor_current_states=None,
                neighbor_valid=None):
        """
        scene_encoding: [B, S, D]    GF encoder full fused scene
        c_lat:          [B, N_lat, N_r, feature_dim]
        scene_mask:     [B, S]       key_padding_mask from encoder
        neighbor_top1_futures: [B, N_nbr, T, 2]  argmax-mod predicted xy per neighbor
        neighbor_current_states: [B, N_nbr, >=5] (x, y, heading, vx, vy)
        neighbor_valid: [B, N_nbr]   bool, True = valid neighbor (else token masked)
        """
        B = scene_encoding.shape[0]
        device = scene_encoding.device

        # --- A. LATERAL MODE HAZIRLIĞI ---
        # c_lat girdisi: [B, N_lat, N_r, feature_dim] -> Örn: [B, 5, 50, 6]
        lat_feat = self.lat_encoder(c_lat) # [B, N_lat, N_r, D]
        # N_r noktası boyunca Max-Pooling (PointNet aggregation)
        lat_feat = torch.max(lat_feat, dim=2)[0] # [B, N_lat, D]

        # --- B. LONGITUDINAL MODE HAZIRLIĞI ---
        # c_lon_j = j / N_lon formülü
        j_vals = torch.arange(self.num_lon, dtype=torch.float32, device=device) # [0, 1, ..., 11]
        c_lon_scalar = j_vals / (self.num_lon - 1) # [0.0, ..., 1.0] - egitim/inference ile tutarli

        # Skaler değeri D boyutuna kopyala -> [N_lon, D]
        lon_feat = c_lon_scalar.unsqueeze(1).repeat(1, self.dim)
        # Batch boyutuna genişlet -> [B, N_lon, D]
        lon_feat = lon_feat.unsqueeze(0).expand(B, -1, -1)

        # --- C. BİRLEŞTİRME (COMBINATION: N_lat x N_lon x 2D) ---
        # Her iki tensörü de [B, N_lat, N_lon, D] yapısına genişlet
        lat_expanded = lat_feat.unsqueeze(2).expand(-1, -1, self.num_lon, -1)
        lon_expanded = lon_feat.unsqueeze(1).expand(-1, self.num_lat, -1, -1)

        # Son boyutta (dim=-1) birleştir (Concatenation) -> [B, N_lat, N_lon, 2D]
        combined_modes = torch.cat([lat_expanded, lon_expanded], dim=-1)

        # --- D. HİZALAMA (ALIGNMENT: 2D -> D) ---
        # Lineer katmandan geçir -> [B, N_lat, N_lon, D]
        aligned_modes = self.mode_proj(combined_modes)

        # Transformer Query'si için düzleştir (Flatten) -> [B, 60, D]
        mode_queries = aligned_modes.view(B, self.num_lat * self.num_lon, self.dim)

        # --- E. CONTEXT INSASI ---
        # Default: sadece scene_encoding (CarPlanner-style).
        # Eger komsu top-1 future'lar verilirse, FutureEncoder ile [B, N_nbr, D]'ye
        # encode edip scene_encoding'in SONUNA append ederiz; key_padding_mask de uzar.
        if neighbor_top1_futures is not None:
            # FutureEncoder [B, N, M, T, 2] bekliyor -> M=1 ekle, sonra sik
            fut_in = neighbor_top1_futures.unsqueeze(2)                               # [B, N_nbr, 1, T, 2]
            future_emb = self.neighbor_future_encoder(fut_in, neighbor_current_states).squeeze(2)  # [B, N_nbr, D]

            N_nbr = future_emb.shape[1]
            context = torch.cat([scene_encoding, future_emb], dim=1)                  # [B, S+N_nbr, D]

            # Mask uzat: gecersiz komsulara karsilik gelen future tokenlari maskelenir
            if neighbor_valid is None:
                neighbor_valid = torch.ones(B, N_nbr, dtype=torch.bool, device=device)
            future_mask = ~neighbor_valid                                             # [B, N_nbr]

            if scene_mask is None:
                S = scene_encoding.shape[1]
                full_mask = torch.cat(
                    [torch.zeros(B, S, dtype=torch.bool, device=device), future_mask],
                    dim=1,
                )
            else:
                full_mask = torch.cat([scene_mask, future_mask], dim=1)               # [B, S+N_nbr]
        else:
            context = scene_encoding
            full_mask = scene_mask

        # --- F. FUSION VE SKORLAMA ---
        mode_features = self.query_encoder(mode_queries, context, context, mask=full_mask)  # [B, 60, D]
        
        # Her mod için olasılık skoru hesapla
        mode_scores = self.score_mlp(mode_features).squeeze(-1) # [B, 60]

        # --- LOGIT MASKING SADECE INFERENCE'TA UYGULANIR ---
        # Training'de mask kapalı: agin "padded c_lat -> dusuk score" implicit kuralini
        # ogrenmesine izin veriyoruz (smooth structural learning, daha az hesitation).
        # Inference'ta (eval mode) mask aktif: arac asla padded bir lateral rotaya
        # gitmesin diye safety net.
        if not self.training:
            lat_valid_mask = (torch.abs(c_lat).sum(dim=(2, 3)) > 1e-4)              # [B, 5]
            mode_valid_mask = lat_valid_mask.unsqueeze(2).expand(-1, -1, self.num_lon)  # [B, 5, 12]
            mode_valid_mask = mode_valid_mask.reshape(B, self.num_lat * self.num_lon)   # [B, 60]
            mode_scores = mode_scores.masked_fill(~mode_valid_mask, -1e9)

        return mode_scores, mode_features

    
