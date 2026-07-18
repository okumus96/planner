"""
Faz 3 (on-the-fly torch): 60 aday modun (lat x lon) maliyetini, refinement.py'nin KENDI cost
fonksiyonlariyla (elle cost yazmadan) batched hesaplar. Cikan maliyet, mode_selector'a
"cost-aware" soft hedef olarak verilir (imitation CE'ye EK; CE kalir, entropy kalkar).

Tasarim:
  - Mod hiz profili: lon bucket -> sabit hedef hiz v; ds[B,M,N] = v (clamp>=0).
  - refinement.acceleration, refinement.jerk -> comfort (sadece ego_state'in hiz/ivme alani lazim).
  - refinement.speed_target -> progress / hiz-limiti uyumu (c_lat v_max -> arc-length 0.1m grid'ine
    resample edilip speed_limit olarak verilir; refinement kendi s-index'leme mantigini kullanir).
  - safety: refinement.safety occupancy grid'i ister (numpy/per-sample occupancy_adpter); on-the-fly
    torch'a uymadigi icin v1'de YOK (gerekirse offline precompute ile eklenir).

refinement fonksiyonlari theseus optim_vars/aux_vars (.tensor) bekler -> hafif _V sarmalayici ile
DOGRUDAN onlari cagiririz (matematigi tekrar yazmayiz).
"""
import torch

from Planner.refinement import acceleration, jerk, speed_target  # refinement'in KENDI cost fn'leri

DT = 0.1
HORIZON = 80                 # 8s @ 10Hz
MAX_SPEED = 15.0
MAX_LEN_M = 120.0
GRID_N = int(MAX_LEN_M / 0.1)  # 1200 (speed_limit'in 0.1m grid uzunlugu)


class _V:
    """theseus Variable yerine: refinement fn'lerinin bekledigi .tensor arayuzu."""
    __slots__ = ('tensor',)
    def __init__(self, t):
        self.tensor = t


def ego_state_from_past(ego_past):
    """ego_agent_past [B,Th,>=2] -> ego_state [B,7] (refinement formati: idx3=hiz, idx5=ivme).
    Hiz/ivme POZISYON farkindan turetilir -> ozellik indekslerine bagimli degil."""
    pos = ego_past[..., :2]
    v_last = (pos[:, -1] - pos[:, -2]).norm(dim=-1) / DT      # [B]
    v_prev = (pos[:, -2] - pos[:, -3]).norm(dim=-1) / DT
    acc = (v_last - v_prev) / DT
    st = torch.zeros(ego_past.shape[0], 7, device=ego_past.device, dtype=ego_past.dtype)
    st[:, 3] = v_last
    st[:, 5] = acc
    return st


def dense_speed_limit(c_lat, grid_n=GRID_N):
    """c_lat [B,L,T,6] -> [B,L,grid_n]: v_max (kanal 4) arc-length 0.1m grid'ine resample.
    Arc-length c_lat'in KENDI xy noktalarindan (kumulatif mesafe) hesaplanir -> spacing'e bagimsiz."""
    B, L, T, _ = c_lat.shape
    xy = c_lat[..., :2]
    vmax = c_lat[..., 4]
    valid_pt = c_lat.abs().sum(-1) > 1e-4                                  # [B,L,T]
    seg = (xy[:, :, 1:] - xy[:, :, :-1]).norm(dim=-1) * valid_pt[:, :, 1:].float()
    s = torch.cat([torch.zeros(B, L, 1, device=c_lat.device, dtype=c_lat.dtype),
                   seg.cumsum(-1)], dim=-1)                                # [B,L,T] arc-length
    grid = torch.arange(grid_n, device=c_lat.device).to(c_lat.dtype) * 0.1  # [grid_n] (m)
    sf = s.reshape(B * L, T).contiguous()
    idx = torch.searchsorted(sf, grid[None].expand(B * L, grid_n).contiguous())
    idx = idx.clamp(0, T - 1)
    out = torch.gather(vmax.reshape(B * L, T), 1, idx).reshape(B, L, grid_n)
    out = out * valid_pt.any(-1)[..., None].to(out.dtype)                  # padded route -> 0 limit
    return out


def compute_mode_costs(c_lat, ego_past, num_lat=5, num_lon=12, max_speed=MAX_SPEED,
                       horizon=HORIZON, w_accel=0.1, w_jerk=1.0, w_speed=0.3):
    """60 mod icin maliyet [B,M] + gecerlilik [B,M].

    c_lat:    [B, num_lat, T, 6]  (x,y,yaw,curv,vmax,occ) ego-frame
    ego_past: [B, Th, >=2]        ego gecmisi (hiz/ivme icin)
    Agirliklar refinement.RefinementPlanner.gains'i yansitir (accel=0.1, jerk=1.0, speed=0.3).
    """
    B = c_lat.shape[0]
    device = c_lat.device
    M = num_lat * num_lon

    ego_state = ego_state_from_past(ego_past)                              # [B,7]

    lon = torch.linspace(0, max_speed, num_lon, device=device)            # [num_lon]
    v_mode = lon.repeat(num_lat)                                          # [M] (mode=lat*num_lon+lon)
    ds = v_mode[None, :, None].expand(B, M, horizon).clamp(min=0)         # [B,M,N] sabit hedef hiz

    lat_of_mode = torch.arange(num_lat, device=device).repeat_interleave(num_lon)  # [M]
    sl = dense_speed_limit(c_lat)                                         # [B,L,grid]
    sl_mode = sl[:, lat_of_mode]                                         # [B,M,grid]

    BM = B * M
    ds_f = ds.reshape(BM, horizon)
    es_f = ego_state[:, None].expand(B, M, 7).reshape(BM, 7)
    sl_f = sl_mode.reshape(BM, -1)
    s_f = torch.cumsum(ds_f * DT, dim=-1)                                 # [BM,N] arc-length(t)

    # --- refinement'in KENDI fonksiyonlari (matematik tekrar yazilmadan) ---
    acc = acceleration([_V(ds_f)], [_V(es_f)])                           # [BM,N]
    jrk = jerk([_V(ds_f)], [_V(es_f)])                                   # [BM,N]
    spt = speed_target([_V(ds_f)], [_V(sl_f), _V(s_f)])                  # [BM,N]

    cost = (w_accel * acc.pow(2).mean(-1)
            + w_jerk * jrk.pow(2).mean(-1)
            + w_speed * spt.pow(2).mean(-1))                             # [BM]
    cost = cost.reshape(B, M)

    mode_valid = (c_lat.abs().sum((2, 3)) > 1e-4)[:, lat_of_mode]        # [B,M]
    return cost, mode_valid


def cost_soft_target(cost, mode_valid, temperature=1.0):
    """Maliyet [B,M] -> softmax(-cost/T) soft hedef [B,M]. Gecersiz modlar maskelenir.

    NaN guvenligi: bir ornekte HIC gecerli mod yoksa (tum lateral rotalar padded), tum modlar
    -inf olur ve softmax NaN verir. Bu satirlari SIFIR hedefe dusururuz -> soft-CE'ye 0 katki
    (imitation yolunun build_gaussian_soft_target'taki zarif davranisiyla ayni).
    """
    cost = torch.nan_to_num(cost, nan=0.0, posinf=1e6, neginf=0.0)        # defensif
    has_valid = mode_valid.any(dim=-1, keepdim=True)                      # [B,1]
    neg = (-cost / temperature).masked_fill(~mode_valid, float('-inf'))
    neg = neg.masked_fill(~has_valid, 0.0)                                # all-invalid satir -> guvenli
    tgt = torch.softmax(neg, dim=-1)
    return tgt * has_valid.to(tgt.dtype)                                  # all-invalid -> all-zero hedef
