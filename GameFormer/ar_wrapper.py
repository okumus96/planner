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

    def forward(self, scene_encoding, c_lat, scene_mask=None):
        # CarPlanner Sec. 3.3.2: selector sadece s0 (encoder output) ile çalışır.
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

        # --- E. FUSION VE SKORLAMA ---
        # Context sadece s0 encoder output'u (CarPlanner-style minimal)
        mode_features = self.query_encoder(mode_queries, scene_encoding, scene_encoding, mask=scene_mask) # [B, 60, D]
        
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

    
