import torch
import torch.nn as nn
import torch.nn.functional as F
from GameFormer.predictor import GameFormer # Orijinal modelinize dokunmadan çağırın
from .predictor_modules import *

import torch
import torch.nn as nn
from .predictor_modules import CrossTransformer # Kendi import yolunuza göre ayarlayın

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
        
        # 4. Çapraz Dikkat Mekanizması ve Skorlama
        self.query_encoder = CrossTransformer(dim=dim)
        self.score_mlp = nn.Sequential(nn.Linear(dim, 64), nn.ELU(), nn.Linear(64, 1))

    def forward(self, env_encoding, c_lat, scene_encoding=None, scene_mask=None):
        B = env_encoding.shape[0]
        device = env_encoding.device
        
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

        # --- E. FUSION VE SKORLAMA ---
        # Context: encoder'dan gelen sahne (harita+ajan gecmisi) ile decoder'dan gelen
        # interaction encoding'i birlestirir -> [B, N_scene+6, D]
        if scene_encoding is not None:
            env_mask = torch.zeros(B, env_encoding.shape[1], dtype=torch.bool, device=device)
            context = torch.cat([scene_encoding, env_encoding], dim=1)
            context_mask = torch.cat([scene_mask, env_mask], dim=1) if scene_mask is not None else None
        else:
            context = env_encoding
            context_mask = None
        mode_features = self.query_encoder(mode_queries, context, context, mask=context_mask) # [B, 60, D]
        
        # Her mod için olasılık skoru hesapla
        mode_scores = self.score_mlp(mode_features).squeeze(-1) # [B, 60]

        # --- YENİ: SIFIR (PADDED) ROTALARI MASKELEME (LOGIT MASKING) ---
        # 1. Hangi yanal (lateral) rotaların geçerli olduğunu bul.
        # c_lat shape: [B, N_lat, N_r, feature_dim]. Eğer rota tamamen 0 ise toplamı < 1e-4 olur.
        lat_valid_mask = (torch.abs(c_lat).sum(dim=(2, 3)) > 1e-4) # Shape: [B, 5] (Bool)
        
        # 2. Bu 5'lik maskeyi 12 hız moduyla genişletip 60'lık maskeye (1D) çevir.
        mode_valid_mask = lat_valid_mask.unsqueeze(2).expand(-1, -1, self.num_lon) # [B, 5, 12]
        mode_valid_mask = mode_valid_mask.reshape(B, self.num_lat * self.num_lon) # [B, 60]
        
        # 3. Geçersiz (sıfır olan) rotaların skorunu -1e9 (Eksi Sonsuz) yap.
        mode_scores = mode_scores.masked_fill(~mode_valid_mask, -1e9)

        return mode_scores, mode_features

    
