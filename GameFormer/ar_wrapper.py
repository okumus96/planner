import torch
import torch.nn as nn
from GameFormer.predictor import GameFormer
from .predictor_modules import *


PREDICATE_NAMES = [
    'follow',               # 0: aynı şeritte önümde, benzer hız — takip
    'yield_to',             # 1: önümde, belirgin yavaş — yol ver
    'blocked_by',           # 2: önümde neredeyse durmuş — acil yavaşla
    'cut_in',               # 3: yan şeritten benim şeridime giriyor
    'merge_conflict',       # 4: şerit değiştirmek istediğim yerde
    'intersection_conflict',# 5: kavşakta yolum kesişiyor
    'overtake_target',      # 6: yavaş, geçebilirim
    'vulnerable_user',      # 7: yaya/bisikletli — özel dikkat
    'no_match',             # 8: hiçbir kurala uymayan ama sahnede var olan ajan
]
NUM_PREDICATES = len(PREDICATE_NAMES)


@torch.no_grad()
def compute_tau_hat(current_states, decoder_outputs):
    """
    GameFormer'ın tahminlerinden τ̂ (predicate ground-truth) üretir.
    Hiçbir harici veriye ihtiyaç yok — sadece current_states ve decoder çıktıları.

    current_states  : [B, N, 5]  — [x, y, heading, vx, vy],  0=ego, 1:=neighbors
    decoder_outputs : dict       — level_K_interactions [B,N,M,T,4], level_K_scores [B,N,M]

    Returns tau_hat : [B, N_neigh, K]  float {0,1}
    """
    device = current_states.device
    B = current_states.shape[0]

    level_K = max(int(k.split('_')[1]) for k in decoder_outputs if 'interactions' in k)

    # N_neigh'i decoder'dan al — encoder 20 komşu görse de decoder sadece 10 işler
    N_neigh = decoder_outputs[f'level_{level_K}_scores'].shape[1] - 1

    ego   = current_states[:, 0]              # [B, 5]
    neigh = current_states[:, 1:N_neigh + 1]  # [B, N_neigh, 5]

    # --- Geometrik özellikler ---
    ego_heading = ego[:, 2]                                                         # [B]
    cos_h = torch.cos(ego_heading).unsqueeze(1)                                     # [B,1]
    sin_h = torch.sin(ego_heading).unsqueeze(1)                                     # [B,1]

    rel = neigh[:, :, :2] - ego[:, None, :2]                                       # [B,N,2]
    lon = rel[:, :, 0] * cos_h + rel[:, :, 1] * sin_h                             # [B,N]  ileri=+
    lat = -rel[:, :, 0] * sin_h + rel[:, :, 1] * cos_h                            # [B,N]  sol=+
    dist = torch.norm(rel, dim=-1)                                                  # [B,N]

    ego_spd   = torch.hypot(ego[:, 3], ego[:, 4]).unsqueeze(1)                     # [B,1]
    neigh_spd = torch.hypot(neigh[:, :, 3], neigh[:, :, 4])                       # [B,N]

    neigh_heading = torch.atan2(neigh[:, :, 4],
                                neigh[:, :, 3].clamp(min=1e-6))                    # [B,N]
    hdiff = torch.abs(torch.atan2(
        torch.sin(neigh_heading - ego_heading.unsqueeze(1)),
        torch.cos(neigh_heading - ego_heading.unsqueeze(1))
    ))                                                                              # [B,N]

    # Komşunun tahmin edilen geleceği (en yüksek skorlu mod)
    fut_traj   = decoder_outputs[f'level_{level_K}_interactions'][:, 1:, :, :, :2]  # [B,N,M,T,2]
    fut_scores = decoder_outputs[f'level_{level_K}_scores'][:, 1:]                  # [B,N,M]
    best_m     = fut_scores.argmax(dim=-1)                                           # [B,N]
    bi = torch.arange(B, device=device).unsqueeze(1).expand(B, N_neigh)
    ni = torch.arange(N_neigh, device=device).unsqueeze(0).expand(B, N_neigh)
    best_fut   = fut_traj[bi, ni, best_m]                                           # [B,N,T,2]

    fut_spd_end = torch.norm(best_fut[:, :, -1] - best_fut[:, :, -2], dim=-1) / 0.1

    # Var olmayan (padding) komşuları maskele
    exists = (neigh_spd > 0.05) | (dist < 60.0)                                    # [B,N]

    tau = torch.zeros(B, N_neigh, NUM_PREDICATES, device=device)

    # 0 follow: aynı şeritte önde, benzer hız (ego takip ediyor)
    tau[:, :, 0] = ((lon > 2.0) & (lon < 40.0) &
                    (lat.abs() < 2.5) &
                    (neigh_spd >= ego_spd - 0.5) &
                    (neigh_spd > 1.0) & exists).float()

    # 1 yield_to: önde, belirgin yavaş ama henüz durmamış
    tau[:, :, 1] = ((lon > 2.0) & (lon < 35.0) &
                    (lat.abs() < 2.5) &
                    (neigh_spd < ego_spd - 1.5) &
                    (neigh_spd > 0.5) & exists).float()

    # 2 blocked_by: önde, neredeyse ya da tamamen durmuş
    tau[:, :, 2] = ((lon > 1.0) & (lon < 25.0) &
                    (lat.abs() < 2.5) &
                    (neigh_spd < 0.5) & exists).float()

    # 3 cut_in: yan şeritte, ego'nun şeridine lateral hareketle yaklaşıyor
    # gelecekte lateral mesafesi azalıyor
    fut_lat_end = (-(best_fut[:, :, -1, 0] - ego[:, None, 0]) * sin_h +
                   (best_fut[:, :, -1, 1] - ego[:, None, 1]) * cos_h)
    tau[:, :, 3] = ((lat.abs() > 1.0) & (lat.abs() < 4.5) &
                    (lon > -2.0) & (lon < 20.0) &
                    (fut_lat_end.abs() < lat.abs() - 0.5) & exists).float()

    # 4 merge_conflict: ego'nun merge hedefi olan şeritte, boylamsal yakın
    tau[:, :, 4] = ((lat.abs() > 1.5) & (lat.abs() < 5.5) &
                    (lon > -8.0) & (lon < 30.0) &
                    (fut_spd_end > 0.5) & exists).float()

    # 5 intersection_conflict: yönler belirgin çakışıyor, yakın, hareket ediyor
    tau[:, :, 5] = ((hdiff > 0.5) & (dist < 25.0) &
                    (neigh_spd > 1.0) & exists).float()

    # 6 overtake_target: aynı yönde ileride ama belirgin yavaş, yan şerit boşluğu var
    tau[:, :, 6] = ((lon > 5.0) & (lon < 50.0) &
                    (lat.abs() < 3.0) &
                    (neigh_spd < ego_spd - 2.0) &
                    (fut_spd_end < ego_spd * 0.7) & exists).float()

    # 7 vulnerable_user: çok yavaş veya durmuş, yakın mesafede (40m'ye genişletildi)
    tau[:, :, 7] = ((neigh_spd < 2.0) & (dist < 40.0) & exists).float()

    # 8 no_match: var olan ama hiçbir kurala girmeyen ajan — type_emb sıfır kalmasın
    tau[:, :, 8] = ((tau[:, :, :8].sum(dim=-1) == 0) & exists).float()

    return tau                                                                      # [B, N_neigh, K]


class SceneGraphModeSelector(nn.Module):
    def __init__(self, dim=256, num_lat=5, num_lon=12, num_neighbors=10, feature_dim=6):
        super().__init__()
        self.dim = dim
        self.num_lat = num_lat
        self.num_lon = num_lon
        self.num_neighbors = num_neighbors

        # past+scene ctx ve future embedding'i D'ye project et
        self.agent_proj = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
        )

        # Her komşu tüm sahneyi sorgular → ilişki öğrenir
        self.relation_encoder = CrossTransformer(dim=dim)

        # Kenar kararları
        self.relevance_head = nn.Linear(dim, 1)
        self.edge_type_head = nn.Linear(dim, NUM_PREDICATES)

        # Edge-type embedding: softmax çıktısını D'ye çevirir
        self.type_embedding = nn.Embedding(NUM_PREDICATES, dim)

        # Lateral rota encoder (PointNet tarzı)
        self.lat_encoder = nn.Sequential(
            nn.Linear(feature_dim, 64), nn.ReLU(),
            nn.Linear(64, 128),        nn.ReLU(),
            nn.Linear(128, dim),
        )

        # Lat+lon → tek mode query
        self.mode_proj = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
        )

        # Mode query'leri sahne bağlamıyla birleştirir
        self.mode_encoder = CrossTransformer(dim=dim)

        self.score_mlp = nn.Sequential(
            nn.Linear(dim, 64), nn.ELU(), nn.Linear(64, 1)
        )

    def forward(self, encoding, mask, decoder_outputs, neighbor_content,
                c_lat_candidates, current_states):
        """
        encoding         : [B, S, D]
        mask             : [B, S]
        decoder_outputs  : dict
        neighbor_content : [B, N, M, D]
        c_lat_candidates : [B, N_lat, N_pts, F]
        current_states   : [B, N+1, 5]  — ego + komşular, τ̂ için
        """
        B = encoding.shape[0]
        device = encoding.device

        level_K = max(int(k.split('_')[1]) for k in decoder_outputs if 'interactions' in k)
        N_neigh = neighbor_content.shape[1]

        # --- A. τ̂: geometric edge type (rule-based, no grad) ---
        tau_hat = compute_tau_hat(current_states, decoder_outputs)            # [B, N, K]

        # --- B. PER-AGENT TEMPORAL EMBEDDING ---
        ctx_emb = encoding[:, 1:N_neigh + 1, :]                              # [B, N, D]

        fut_scores = decoder_outputs[f'level_{level_K}_scores'][:, 1:N_neigh + 1]
        weights    = fut_scores.softmax(dim=-1).unsqueeze(-1)                 # [B, N, M, 1]
        fut_emb    = (weights * neighbor_content).sum(dim=2)                  # [B, N, D]

        h_j = self.agent_proj(torch.cat([ctx_emb, fut_emb], dim=-1))         # [B, N, D]

        # --- C. SCENE-AWARE RELEVANCE ---
        # Her komşu tüm sahneyi sorgular → sadece "önemli mi?" öğrenilir
        out       = self.relation_encoder(h_j, encoding, encoding, mask=mask) # [B, N, D]
        relevance = torch.sigmoid(self.relevance_head(out)).squeeze(-1)       # [B, N]

        # --- D. AGGREGATION ---
        # τ̂ edge TYPE'ı belirler, relevance önem ağırlığını belirler
        type_emb  = tau_hat @ self.type_embedding.weight                      # [B, N, D]
        weighted  = relevance.unsqueeze(-1) * type_emb * h_j                 # [B, N, D]
        scene_ctx = weighted.sum(dim=1, keepdim=True)                         # [B, 1, D]

        ego_emb      = encoding[:, 0:1, :]                                    # [B, 1, D]
        mode_context = torch.cat([ego_emb, scene_ctx], dim=1)                 # [B, 2, D]

        # --- E. MODE QUERIES ---
        lat_feat   = self.lat_encoder(c_lat_candidates)                       # [B, N_lat, N_pts, D]
        lat_feat   = torch.max(lat_feat, dim=2)[0]                            # [B, N_lat, D]

        j_vals     = torch.arange(self.num_lon, dtype=torch.float32, device=device)
        lon_scalar = (j_vals / (self.num_lon - 1)).unsqueeze(1).repeat(1, self.dim)
        lon_feat   = lon_scalar.unsqueeze(0).expand(B, -1, -1)               # [B, N_lon, D]

        lat_exp = lat_feat.unsqueeze(2).expand(-1, -1, self.num_lon, -1)
        lon_exp = lon_feat.unsqueeze(1).expand(-1, self.num_lat, -1, -1)
        mode_queries = self.mode_proj(
            torch.cat([lat_exp, lon_exp], dim=-1)
        ).view(B, self.num_lat * self.num_lon, self.dim)                      # [B, 60, D]

        lat_valid  = (torch.abs(c_lat_candidates).sum(dim=(2, 3)) > 1e-4)    # [B, N_lat]
        mode_valid = lat_valid.unsqueeze(2).expand(-1, -1, self.num_lon).reshape(B, -1)

        # --- F. MODE SCORING ---
        mode_features = self.mode_encoder(mode_queries, mode_context, mode_context)
        mode_scores   = self.score_mlp(mode_features).squeeze(-1)             # [B, 60]
        mode_scores   = mode_scores.masked_fill(~mode_valid, -1e9)

        return mode_scores, mode_features, relevance, tau_hat


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

    
