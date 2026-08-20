import torch
import logging
import glob
import random
import numpy as np
from torch.utils.data import Dataset
from torch.nn import functional as F


def initLogging(log_file: str, level: str = "INFO"):
    logging.basicConfig(filename=log_file, filemode='w',
                        level=getattr(logging, level, None),
                        format='[%(levelname)s %(asctime)s] %(message)s',
                        datefmt='%m-%d %H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler())


def set_seed(CUR_SEED):
    random.seed(CUR_SEED)
    np.random.seed(CUR_SEED)
    torch.manual_seed(CUR_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sort_candidates_by_lateral(c_lat_ego, c_lat_global=None):
    """Aday lateral rotalari UZAMSAL sol->sag sirala (ordinal + frame'ler arasi kararli).

    Adaylar ego-frame'de uretilir: feature[...,0]=ileri (rel_x), [...,1]=yanal (rel_y).
    Her rotanin gecerli noktalarinin ortalama yanal offset'i (rel_y) anahtardir:
    buyuk +y = sol, kucuk/-y = sag. AZALAN rel_y -> sol->sag siralama. Boylece indeks
    BFS/ego-mesafesi yerine GEOMETRIYE baglanir; hem ordinal olur (Gaussian lat label
    artik anlamli) hem de frame'ler arasi KARARLI olur (ayni fiziksel serit ~ ayni indeks
    -> jitter ve indeks-cache riski azalir).

    Padded (tamamen sifir) rotalar SONA atilir. c_lat_global verilirse AYNI permutasyon
    uygulanir (planlamada kullanilan global rota ile skorlanan ego rota ayni indekste kalsin).

    c_lat_ego:    np.ndarray [N_lat, T, F]   (ego-frame adaylar)
    c_lat_global: np.ndarray [N_lat, T, F]   (opsiyonel; global-frame karsiliklari)
    Doner: c_lat_global None ise ego_sirali; degilse (ego_sirali, global_sirali).
    """
    point_valid = np.abs(c_lat_ego).sum(axis=-1) > 1e-4              # [N, T]
    route_valid = point_valid.any(axis=-1)                           # [N]
    cnt = point_valid.sum(axis=-1).clip(min=1)                       # [N]
    lat_off = (c_lat_ego[..., 1] * point_valid).sum(axis=-1) / cnt   # [N] ortalama yanal offset
    # Azalan rel_y icin negatif anahtar; padded rotalar +inf ile sona
    key = np.where(route_valid, -lat_off, np.inf)
    order = np.argsort(key, kind='stable')                          # esitlikte orijinal sira korunur

    ego_sorted = c_lat_ego[order]
    if c_lat_global is None:
        return ego_sorted
    return ego_sorted, c_lat_global[order]


class DrivingData(Dataset):
    def __init__(self, data_dir, n_neighbors):
        self.data_list = glob.glob(data_dir)
        self._n_neighbors = n_neighbors

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = np.load(self.data_list[idx])
        ego = data['ego_agent_past']
        neighbors = data['neighbor_agents_past']
        route_lanes = data['route_lanes']
        map_lanes = data['lanes']
        map_crosswalks = data['crosswalks']
        ego_future_gt = data['ego_agent_future']
        neighbors_future_gt = data['neighbor_agents_future'][:self._n_neighbors]
        c_lat_candidates = data['c_lat_candidates']
        c_lat_candidates_global = data['c_lat_candidates_global'] if 'c_lat_candidates_global' in data else data['c_lat_candidates']

        # Karar etiketleri (dod_meta / H): HAM aday sirasiyla, sort'tan ONCE hesaplanmali —
        # koridor-goreli lane-change tespiti aday 0 = ego'nun seridi varsayimina dayanir;
        # asagidaki lateral sort aday 0'i "en soldaki" yapar ve etiketi bozardi (geometrik
        # fallback'in kavisli yollarda %14.8 yanlis-lateral urettigi olculdu). GT-turevi ->
        # model-bagimsiz, cache damgasi gerekmez.
        from .decision_labels import decision_labels_single
        dec_lon, dec_lat = decision_labels_single(ego_future_gt, c_lat_candidates)

        # Adaylari uzamsal sol->sag sirala (ordinal + kararli indeks). Etiket
        # (get_expert_mode_index) geometri-argmin tabanli oldugundan bu re-sort'tan
        # bagimsiz dogru kalir; sadece indeks numarasi degisir. plannerv2 inference'ta
        # AYNI helper'i kullanir -> train/inference tutarli.
        c_lat_candidates, c_lat_candidates_global = sort_candidates_by_lateral(
            c_lat_candidates, c_lat_candidates_global
        )

        # Predicate kanallari (extract_channels.py'nin damgaladigi anahtarlar; channels branch).
        # Training GF-varyantini okur (train/deploy tutarliligi). Damgasiz eski dosyada sifir-fallback:
        # hicbir kanal yanmaz -> gating acikken o ornek maskeye ajan sokmaz, eski davranis bozulmaz.
        from .channels import NUM_CHANNELS, NUM_EVIDENCE, NUM_MAP_CHANNELS, NUM_MAP_EVIDENCE
        n = self._n_neighbors
        if 'channel_active_gf' in data:
            ch_active = data['channel_active_gf'][:n]
            ch_evidence = data['channel_evidence_gf'][:n]
            mch_active = data['map_channel_active']
            mch_evidence = data['map_channel_evidence']
        else:
            S = map_lanes.shape[0] + map_crosswalks.shape[0] + route_lanes.shape[0]
            ch_active = np.zeros((n, NUM_CHANNELS), dtype=bool)
            ch_evidence = np.zeros((n, NUM_EVIDENCE), dtype=np.float32)
            mch_active = np.zeros((S, NUM_MAP_CHANNELS), dtype=bool)
            mch_evidence = np.zeros((S, NUM_MAP_EVIDENCE), dtype=np.float32)

        return (ego, neighbors, map_lanes, map_crosswalks, route_lanes, ego_future_gt,
                neighbors_future_gt, c_lat_candidates, c_lat_candidates_global,
                ch_active, ch_evidence, mch_active, mch_evidence,
                np.int64(dec_lon), np.int64(dec_lat))


def imitation_loss(gmm, scores, ground_truth):
    B, N = gmm.shape[0], gmm.shape[1]
    distance = torch.norm(gmm[:, :, :, :, :2] - ground_truth[:, :, None, :, :2], dim=-1)
    best_mode = torch.argmin(distance.mean(-1), dim=-1)

    mu = gmm[..., :2]
    best_mode_mu = mu[torch.arange(B)[:, None, None], torch.arange(N)[None, :, None], best_mode[:, :, None]]
    best_mode_mu = best_mode_mu.squeeze(2)
    dx = ground_truth[..., 0] - best_mode_mu[..., 0]
    dy = ground_truth[..., 1] - best_mode_mu[..., 1]

    cov = gmm[..., 2:]
    best_mode_cov = cov[torch.arange(B)[:, None, None], torch.arange(N)[None, :, None], best_mode[:, :, None]]
    best_mode_cov = best_mode_cov.squeeze(2)
    log_std_x = torch.clamp(best_mode_cov[..., 0], -2, 2)
    log_std_y = torch.clamp(best_mode_cov[..., 1], -2, 2)
    std_x = torch.exp(log_std_x)
    std_y = torch.exp(log_std_y)

    gmm_loss = log_std_x + log_std_y + 0.5 * (torch.square(dx/std_x) + torch.square(dy/std_y))
    gmm_loss = torch.mean(gmm_loss)

    score_loss = F.cross_entropy(scores.permute(0, 2, 1), best_mode, label_smoothing=0.2, reduction='none')
    score_loss = score_loss * torch.ne(ground_truth[:, :, 0, 0], 0)
    score_loss = torch.mean(score_loss)
    
    loss = gmm_loss + score_loss

    return loss, best_mode_mu, best_mode


def level_k_loss(outputs, ego_future, neighbors_future, neighbors_future_valid):
    loss: torch.tensor = 0
    levels = len(outputs.keys()) // 2 
    gt_future = torch.cat([ego_future[:, None], neighbors_future], dim=1)

    for k in range(levels):
        trajectories = outputs[f'level_{k}_interactions']
        scores = outputs[f'level_{k}_scores']
        predictions = trajectories[:, 1:] * neighbors_future_valid[:, :, None, :, 0, None]
        plan = trajectories[:, :1]
        trajectories = torch.cat([plan, predictions], dim=1)
        il_loss, future, best_mode = imitation_loss(trajectories, scores, gt_future)
        loss += il_loss 

    return loss, future


def planning_loss(plan, ego_future):
    loss = F.smooth_l1_loss(plan, ego_future)
    loss += F.smooth_l1_loss(plan[:, -1], ego_future[:, -1])

    return loss

def motion_metrics(plan_trajectory, prediction_trajectories, ego_future, neighbors_future, neighbors_future_valid):
    prediction_trajectories = prediction_trajectories * neighbors_future_valid
    plan_distance = torch.norm(plan_trajectory[:, :, :2] - ego_future[:, :, :2], dim=-1)
    prediction_distance = torch.norm(prediction_trajectories[:, :, :, :2] - neighbors_future[:, :, :, :2], dim=-1)
    heading_error = torch.abs(torch.fmod(plan_trajectory[:, :, 2] - ego_future[:, :, 2] + np.pi, 2 * np.pi) - np.pi)

    # planning
    plannerADE = torch.mean(plan_distance)
    plannerFDE = torch.mean(plan_distance[:, -1])
    plannerAHE = torch.mean(heading_error)
    plannerFHE = torch.mean(heading_error[:, -1])
    
    # prediction
    predictorADE = torch.mean(prediction_distance, dim=-1)
    predictorADE = torch.masked_select(predictorADE, neighbors_future_valid[:, :, 0, 0])
    predictorADE = torch.mean(predictorADE)
    predictorFDE = prediction_distance[:, :, -1]
    predictorFDE = torch.masked_select(predictorFDE, neighbors_future_valid[:, :, 0, 0])
    predictorFDE = torch.mean(predictorFDE)

    return plannerADE.item(), plannerFDE.item(), plannerAHE.item(), plannerFHE.item(), predictorADE.item(), predictorFDE.item()

def get_expert_mode_index(ego_future, c_lat_candidates, num_lat=5, num_lon=12, max_speed=15.0):
    """
    CarPlanner (Section 3.4) — Mode Assignment.

    Lateral GT:
        "Ego'nun 8s sonraki konumu hangi rotanin UZERINDE?"
        Her rota icin: min_t  ||route_point_t - gt_endpoint||
        En kucuk min-mesafeli rota = pozitif lateral mod.

        NOT: Eski versiyon route ENDPOINT'i (sabit ~120m'deki son nokta) ile
        gt endpoint'ini karsilastiriyordu — ego hizina bagli olarak boylamsal
        fark yanal farki bastiriyordu. Yeni versiyon hizdan bagimsiz: ego'nun
        8s sonraki konumu hangi rotanin uzerindeyse o rota dogru cevap.

    Longitudinal GT: ortalama hiza en yakin bucket (degismedi).

    ego_future:        [B, 80, 2]
    c_lat_candidates:  [B, 5, T, 6]
    """
    NUM_LON = 12
    B = ego_future.shape[0]
    device = ego_future.device

    # 1. C_LAT DOGRU CEVAP (min distance from gt endpoint to ANY point on each route)
    gt_endpoint = ego_future[:, -1, :2]                                  # [B, 2]
    gt_exp = gt_endpoint.unsqueeze(1).unsqueeze(1)                       # [B, 1, 1, 2]
    diffs = c_lat_candidates[..., :2] - gt_exp                           # [B, 5, T, 2]
    dists_all = torch.norm(diffs, dim=-1)                                # [B, 5, T]

    # Padded nokta maskelemesi: tum 6 ozellik tam sifir olan noktalar dolgu sayilir
    # (gercek nokta ego frame'de (0,0)'a yakin olsa bile curvature/v_max gibi alanlari non-zero)
    point_valid = (torch.abs(c_lat_candidates).sum(dim=-1) > 1e-4)       # [B, 5, T]
    dists_all = dists_all.masked_fill(~point_valid, float('inf'))

    min_dist = dists_all.min(dim=-1).values                              # [B, 5]

    # Tamamen padded rotalari ele
    lat_valid = (torch.abs(c_lat_candidates).sum(dim=(2, 3)) > 1e-4)     # [B, 5]
    min_dist = min_dist.masked_fill(~lat_valid, float('inf'))

    best_lat = torch.argmin(min_dist, dim=-1)                            # [B]
    
    # 2. C_LON DOĞRU CEVABINI BUL (Ortalama hıza en yakın dilim)
    # 8 saniyede alınan toplam mesafe / 8.0 = Ortalama Hız
    total_dist = torch.norm(ego_future[:, -1, :2] - ego_future[:, 0, :2], dim=-1)
    avg_speed = total_dist / 8.0 # [B]
    
    speed_bins = torch.linspace(0, max_speed, num_lon, device=device) # [12]
    # Uzmanın hızının, bizim 12 dilimden hangisine en yakın olduğunu bul
    speed_diffs = torch.abs(speed_bins.unsqueeze(0) - avg_speed.unsqueeze(1)) # [B, 12]
    best_lon = torch.argmin(speed_diffs, dim=-1) # [B]
    
    # 3. İNDEKSİ BİRLEŞTİR (0 ile 59 arası tek bir indeks)
    gt_mode_idx = best_lat * num_lon + best_lon # [B]
    
    expert_lat_idx = gt_mode_idx // NUM_LON
    expert_lon_idx = gt_mode_idx % NUM_LON
    
    return gt_mode_idx, expert_lat_idx, expert_lon_idx