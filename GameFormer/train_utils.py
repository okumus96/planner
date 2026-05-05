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

        return ego, neighbors, map_lanes, map_crosswalks, route_lanes, ego_future_gt, neighbors_future_gt, c_lat_candidates, c_lat_candidates_global


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
    CarPlanner (Section 3.4): Eğitim (Loss) için Uzmanın seçtiği C_lat ve C_lon'u bularak 
    60 mod içerisindeki doğru cevabın (0-59) indeksini döndürür.
    ego_future: [B, 80, 2] (Uzmanın gerçek 8 saniyelik yörüngesi)
    c_lat_candidates: [B, 5, 50, 3] (BFS rotaları)
    """
    NUM_LON =12
    B = ego_future.shape[0]
    device = ego_future.device
    
    # 1. C_LAT DOĞRU CEVABINI BUL (En son noktaya en yakın rota)
    gt_endpoint = ego_future[:, -1, :2] # Uzmanın 8 saniye sonraki son noktası [B, 2]
    route_endpoints = c_lat_candidates[:, :, -1, :2] # BFS rotalarının son noktaları [B, 5, 2]

    # Mesafe hesabı (L2 Norm)
    dists = torch.norm(route_endpoints - gt_endpoint.unsqueeze(1), dim=-1) # [B, 5]

    # Sifir-padded kandidatleri mesafe hesabindan cikar (origin'e yakin gorunup yanlis secilmesin)
    lat_valid = (torch.abs(c_lat_candidates).sum(dim=(2, 3)) > 1e-4)  # [B, 5]
    dists = dists.masked_fill(~lat_valid, float('inf'))

    best_lat = torch.argmin(dists, dim=-1) # [B] (Hangi rota en yakınsa onun indeksi)
    
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