"""Predicate-tabanli iliski kanallari (A'nin yapisal gating katmani).

compute_channels(...): her (ego, komsu-j) cifti icin hangi iliski kanallarinin YANDIGINI (active)
ve o ciftin surekli kanit sayilarini (evidence) dondurur. Kurallar SADECE yapiyi kurar
("bu edge var, bu tipte"); onem/agirlik atamaz -- o, attention'in isi.

Kanal tanimlari arkadasin KG'sindeki predicate'lerin frame-seviyesi karsiliklaridir
(nuPlan_Predicates_KG): hasSpatialMapRelation (same/adjacent_left/right/successor/predecessor),
onObservedCollisionCourseWith/hasObservedTTCTo, near/veryNear. Esikler mumkun oldugunca oradaki
sabitlerden alinmistir (same-flow 0.45 rad; near 5 m / veryNear 2 m; CPA horizon 8 s, clearance 2 m;
overtake side-by-side 1.25 m, rel-speed 0.3 m/s, clearance 1.0 m).

ego->agent FINAL set (R=10, kullanici onayi 2026-08-15): interaction ailesi TAM KADRO --
follows / merges (yon evidence'ta) / overtakes anticipated formda; connected-succ/pred atildi
(koridor projeksiyonu zaten same_lane_*'a katliyor), shared_traffic_control reddedildi,
near tek isim (veryNear alt kumesi d_fs evidence'inda).

Frame konvansiyonu: her sey t anindaki EGO frame'inde (x ileri, y sol). Girdiler data_process /
observation_adapter ciktilariyla ayni.
"""
import math

import torch

# --- Kanal indeksleri (R = 10, ego->agent FINAL seti; isimler = KG predicate adlari) ---
CH_SAME_LANE_AHEAD = 0      # same_lane_ahead (relevance reason; same_map + path>0)
CH_SAME_LANE_BEHIND = 1     # same_lane_behind
CH_ADJACENT_LEFT = 2        # adjacent_left (_pair_map_relation)
CH_ADJACENT_RIGHT = 3       # adjacent_right
CH_COLLISION_COURSE = 4     # onObservedCollisionCourseWith (GF-future uzerinde)
CH_SHARES_INTERSECTION = 5  # sharesIntersectionWith (v1 geometrik proxy, v2 map_api)
CH_NEAR = 6                 # near (d_fs <= 5, fallback; veryNear alt kumesi evidence'ta)
CH_FOLLOWS = 7              # follows (frame-level zarf: gap/headway + teklik + 1 s sureklilik)
CH_MERGES = 8               # merges (anticipated; np:merges katalog ust kavrami --
                            #   yon = evidence'taki giris-ds isareti, bolmek tek satir)
CH_OVERTAKES = 9            # overtakes (anticipated, in-progress)
CH_VRU = 10                 # vulnerable_road_user_near_ego_path (KG reason; DAR sinir: 12 m
                            #   KG VRU-kritik esigi ∧ yaklasiyor -- yayanin 5-12 m olu bolgesini
                            #   kapatan konum-tabanli kanal; 2026-08-15 miss-kirilimi %42 ped)
NUM_CHANNELS = 11

CHANNEL_NAMES = [
    "same_lane_ahead", "same_lane_behind", "adjacent_left", "adjacent_right",
    "onObservedCollisionCourseWith", "sharesIntersectionWith", "near",
    "follows", "merges", "overtakes", "vulnerable_road_user_near_ego_path",
]

# --- Kanit indeksleri (E = 9) ---
EV_DS = 0            # koridor boyunca isaretli boyuna mesafe (s_j - s_ego) [m]
EV_DLAT = 1          # koridora isaretli lateral (sol +) [m]
EV_DFS = 2           # kaba serbest mesafe (merkez - yaricaplar) [m]
EV_CLOSING = 3       # kapanma hizi (+ = yaklasiyor) [m/s], KG hasClosingSpeedTo isareti
EV_TTC = 4           # zaman-hizali ilk yakin-gecis zamani [s]; yoksa HORIZON_S
EV_T_ENTRY = 5       # predicted koridor-giris zamani [s]; yoksa HORIZON_S
EV_DTHETA_FLOW = 6   # |wrap(theta_j - koridor tanjanti)| [rad]
EV_VLAT = 7          # koridora dogru lateral hiz (+ = koridora yaklasiyor) [m/s]
EV_DS_ENTRY = 8      # merges: ongorulen koridor-giris noktasinin ego'ya gore ds'i
                     #   (+ = onume girecek ~mergesInFrontOf, - = arkama ~mergesBehind)
NUM_EVIDENCE = 9

# --- Esikler (kaynak: nuPlan_Predicates_KG sabitleri; * isaretliler step-1'de kalibre edilecek) ---
LANE_W = 3.5                 # * varsayilan serit genisligi [m]
# KG'nin DIRECTIONAL esik ailesi (relevance_logic.py:46-49, "Directional semantics should be
# stricter than pair relevance") -- kanallarimiz directional semantics oldugu icin dogru katman bu.
# (Onceki 100/35/45 degerleri onlarin RELEVANCE-ELEME katmaniydi; olculdu: 8a76'da -38 m'deki
# ajan adjacent yaniyordu.)
AHEAD_MAX_M = 80.0           # KG directional_same_lane_ahead_m
BEHIND_MAX_M = 25.0          # KG directional_same_lane_behind_m
ADJ_MAX_M = 35.0             # KG directional_adjacent_m (merkez mesafe)
SAME_FLOW_RAD = 0.45         # KG same-flow (hasTravelDirectionDifferenceTo) esigi -- AJANLAR
MAP_DIR_RAD = 0.60           # KG map/motion agreement esigi (hasEffectiveTravelHeading) -- HARITA
                             # elemani yon testi: kavsakta kivrilan koridora karsi 0.45 cok siki
                             # (olculdu: dtheta 0.55'lik adjacent kopuklugu), 0.60 KG'nin harita
                             # basligi icin kendi sabiti
NEAR_M = 5.0                 # KG near ust siniri
CPA_CLEARANCE_M = 1.0        # KG critical_cpa_clearance_m. (2.0 = onlarin RELEVANCE eleme
                             # marji cpa_clearance_margin_m'di; kanal semantigi icin kritik esik
                             # dogrusu -- 2.0, yan seritten/karsidan NORMAL gecisleri de (lateral
                             # ~3.5 m) collide sayiyordu; olculdu: f4d2'de 37 m arkadaki adjacent)
HORIZON_S = 8.0              # KG cpa_horizon_s == GF future ufku
DT = 0.1
CROSSING_LOOKAHEAD_M = 60.0  # * path-crossing icin koridor ileri penceresi
CROSSING_MIN_ANGLE = math.radians(30.0)  # *
MERGE_VLAT_MIN = 0.2         # * lane-change onset lateral hiz esigi
OT_SIDE_MIN_LAT = 1.25       # KG overtake side-by-side lateral esigi
OT_REL_SPEED_MIN = 0.3       # KG overtake pozitif rel-speed kaniti
OT_CLEAR_M = 1.0             # KG overtake clearance
OT_WINDOW_DS_M = 10.0        # * "su an yanimda" boyuna penceresi
OT_PAST_ADVANCE_M = 1.0      # * gecmis 2 s'de minimum relatif ilerleme
# follows (KG follows tanimindan, frame-level):
FOL_HEADWAY_S = 5.0          # KG maximum_follow_headway_s (g/v <= 5)
FOL_MAX_GAP_M = 80.0         # KG moving-follow g ust siniri
FOL_QUEUE_VS = 2.0           # KG kuyruk modu: v_ego <= 2
FOL_QUEUE_VO = 4.0           # KG kuyruk modu: v_lider <= 4
FOL_QUEUE_GAP_M = 12.0       # KG maximum_queue_follow_gap_m
FOL_MARGIN_M = 0.50          # KG en-yakin-lider ikinci-aday marji
FOL_PERSIST_FRAMES = 11      # KG minimum_follow_duration_s = 1.0 s @ 10 Hz
EGO_HALF_LEN_M = 2.31        # pacifica yari-boy (bumper gap icin)
VRU_CRIT_M = 12.0            # KG VRU-kritik merkez mesafesi (relevance_logic.py:267);
                             # 35 m'lik vulnerable_road_user_distance_m ELEME degeri, kanala dar olan alinir


def _wrap(a):
    return torch.atan2(torch.sin(a), torch.cos(a))


def select_ego_corridor(ref_path):
    """ref_path [B,R,P,>=3] -> [B] EGO'NUN SERIDINDEN baslayan aday indeksi.

    "Aday 0 = ego'nun seridi" VARSAYIMI KALDIRILDI (2026-08-18): aday 0 ham npz sirasinda
    ~%95 ego seridiydi ama garanti degildi; loader'in lateral sort'u gibi yeniden siralamalar
    altinda tamamen bozuluyordu (viz'de gorulen "koridor baska seritten basliyor" bunun
    sonucuydu). Kural: her adayi ego konumuna (ego-frame orijini) projeksiyon yap;
    |d_lat| <= LANE_W/2 ve |heading farki| <= MAP_DIR_RAD saglayanlardan en kucuk |d_lat|'lisi
    secilir; hicbiri saglamazsa (ego donus ortasinda / rota disi) min-|d_lat| fallback.
    Secim SIRA-BAGIMSIZ -> ham npz sirasi, lateral-sortlu loader sirasi ve deployment
    lattice sirasi ayni sonucu verir."""
    xy = ref_path[..., :2].float()                                  # [B,R,P,2]
    yaw = ref_path[..., 2].float()
    valid = xy.abs().sum(-1) > 1e-6                                 # [B,R,P]
    d = xy.norm(dim=-1).masked_fill(~valid, 1e9)                    # orijine mesafe
    idx = d.argmin(dim=-1)                                          # [B,R] en yakin nokta
    g = lambda t: torch.gather(t, 2, idx.unsqueeze(-1)).squeeze(-1)
    y0, px, py = g(yaw), g(xy[..., 0]), g(xy[..., 1])
    dlat = (torch.sin(y0) * px - torch.cos(y0) * py).abs()          # |lateral ofset(ego)|
    hd = torch.atan2(torch.sin(y0), torch.cos(y0)).abs()            # ego heading = 0
    cand_ok = valid.any(-1)
    inlane = cand_ok & (dlat <= LANE_W / 2) & (hd <= MAP_DIR_RAD)
    # Beraberlik kirici (SIRA-BAGIMSIZ): ayni start seridini paylasan adaylar ego'da ayni
    # |d_lat|'a sahiptir (duz-git vs don, ikisi de ego seridinden baslar) -> argmin tek basina
    # siraya bagli kalirdi. Ikincil anahtar: ilk 20 m'de ego heading'ine ortalama hizalanma
    # ("su anki koridor" semantigi); ucuncul: daha uzun aday. Katsayilar olcek ayirici
    # (dlat metre duzeyinde baskin, hizalanma beraberlikte belirleyici).
    seg = (xy[:, :, 1:] - xy[:, :, :-1]).norm(dim=-1)               # [B,R,P-1]
    cum = torch.cat([torch.zeros_like(seg[..., :1]), seg.cumsum(-1)], dim=-1)  # [B,R,P]
    near20 = valid & (cum <= 20.0)
    hd_all = torch.atan2(torch.sin(yaw), torch.cos(yaw)).abs()      # [B,R,P]
    hd20 = ((hd_all * near20).sum(-1) / near20.sum(-1).clamp(min=1))
    smax = torch.where(valid, cum, torch.zeros_like(cum)).amax(-1)  # [B,R] toplam uzunluk
    score = dlat + 0.3 * hd20 + 1e-3 * (200.0 - smax.clamp(max=200.0))
    pick_fb = score.masked_fill(~cand_ok, 1e9).argmin(dim=-1)       # fallback: en iyi skor
    pick_in = score.masked_fill(~inlane, 1e9).argmin(dim=-1)
    return torch.where(inlane.any(-1), pick_in, pick_fb)            # [B]


def _corridor_arrays(ref_path):
    """ref_path [B,R,P,>=3] -> EGO-SERIDI adayinin (xy [B,P,2], yaw [B,P], cum_s, valid).
    Aday secimi select_ego_corridor ile (sabit indeks 0 DEGIL)."""
    sel = select_ego_corridor(ref_path)                             # [B]
    r0 = ref_path[torch.arange(ref_path.shape[0], device=ref_path.device), sel]
    xy = r0[..., :2].float()
    yaw = r0[..., 2].float()
    valid = xy.abs().sum(-1) > 1e-6
    seg = (xy[:, 1:] - xy[:, :-1]).norm(dim=-1)
    cum = torch.cat([torch.zeros_like(seg[:, :1]), seg.cumsum(dim=1)], dim=1)  # [B,P]
    return xy, yaw, cum, valid


def _project(points, cxy, cyaw, ccum, cvalid):
    """points [B,K,2] -> koridora projeksiyon: s [B,K], d_lat [B,K] (sol +), tan_yaw [B,K],
    on_start [B,K] (en yakin nokta koridorun ILK noktasi -> arkada olabilir)."""
    d = (points[:, :, None, :] - cxy[:, None, :, :]).norm(dim=-1)          # [B,K,P]
    d = d.masked_fill(~cvalid[:, None, :], 1e9)
    idx = d.argmin(dim=-1)                                                 # [B,K]
    s = torch.gather(ccum, 1, idx)
    yawk = torch.gather(cyaw, 1, idx)
    cx = torch.gather(cxy[..., 0], 1, idx)
    cy = torch.gather(cxy[..., 1], 1, idx)
    dx = points[..., 0] - cx
    dy = points[..., 1] - cy
    d_lat = -torch.sin(yawk) * dx + torch.cos(yawk) * dy
    on_start = idx == 0
    return s, d_lat, yawk, on_start


def _arc_walk(cxy, ccum, cvalid, s_query):
    """Koridor uzerinde arc-length s_query [B,T] konumlari [B,T,2] (lineer interp)."""
    B, P = ccum.shape
    smax = torch.where(cvalid, ccum, torch.zeros_like(ccum)).max(dim=1, keepdim=True).values
    sq = s_query.clamp(min=0.0)
    sq = torch.minimum(sq, smax.expand_as(sq))
    idx = torch.searchsorted(ccum.contiguous(), sq.contiguous(), right=True).clamp(1, P - 1)
    s1 = torch.gather(ccum, 1, idx - 1)
    s2 = torch.gather(ccum, 1, idx)
    w = ((sq - s1) / (s2 - s1).clamp(min=1e-6)).unsqueeze(-1)
    p1 = torch.gather(cxy, 1, (idx - 1).unsqueeze(-1).expand(-1, -1, 2))
    p2 = torch.gather(cxy, 1, idx.unsqueeze(-1).expand(-1, -1, 2))
    return p1 + w * (p2 - p1)


@torch.no_grad()
def compute_channels(neighbor_agents_past, ego_agent_past, neighbor_futures, ref_path,
                     neighbor_valid=None):
    """
    neighbor_agents_past [B,N,21,11] : (x,y,theta,vx,vy,yawrate,L,W,tip3) -- ego frame @ t
    ego_agent_past       [B,21,7]    : (x,y,theta,vx,vy,ax,ay)
    neighbor_futures     [B,N,T,2]   : GF top-1 (runtime) veya GT (offline) -- ayni fonksiyon
    ref_path             [B,R,P,>=3] : c_lat_candidates; aday 0 = ego'nun seridinin yolu
    neighbor_valid       [B,N] bool  : yoksa son-frame konumu != 0'dan turetilir

    Doner: active [B,N,NUM_CHANNELS] bool, evidence [B,N,NUM_EVIDENCE] float
    """
    B, N, _, _ = neighbor_agents_past.shape
    dev = neighbor_agents_past.dtype
    cur = neighbor_agents_past[:, :, -1]                       # [B,N,11]
    pos = cur[..., 0:2]
    theta = cur[..., 2]
    vel = cur[..., 3:5]
    L = cur[..., 6].clamp(min=0.5)
    W = cur[..., 7].clamp(min=0.3)
    if neighbor_valid is None:
        neighbor_valid = cur[..., :2].abs().sum(-1) > 1e-6

    ego_cur = ego_agent_past[:, -1]
    ego_v = ego_cur[..., 3:5].norm(dim=-1)                      # [B]
    r_ego = 0.5 * math.hypot(4.6, 2.0)                          # pacifica kaba yaricap
    r_j = 0.5 * torch.sqrt(L ** 2 + W ** 2)

    cxy, cyaw, ccum, cvalid = _corridor_arrays(ref_path)
    has_corr = cvalid.any(dim=1)                                # [B]

    # --- ortak buyuklukler ---
    s_j, d_lat, tan_yaw, on_start = _project(pos, cxy, cyaw, ccum, cvalid)
    # koridor ego'dan basladigi icin s_ego ~= 0; arkadaki ajanlar start'a yapisir -> ego-frame fallback
    behind_mode = on_start & (pos[..., 0] < 0)
    ds = torch.where(behind_mode, pos[..., 0], s_j)             # arkada: boyuna = ego-frame x
    d_lat_eff = torch.where(behind_mode, pos[..., 1], d_lat)
    tan_eff = torch.where(behind_mode, torch.zeros_like(tan_yaw), tan_yaw)

    dtheta_flow = _wrap(theta - tan_eff).abs()
    same_flow = dtheta_flow <= SAME_FLOW_RAD
    d_center = pos.norm(dim=-1)
    d_fs = d_center - r_ego - r_j
    # KG hasClosingSpeedTo: -(r . dv)/|r|, dv = v_j - v_ego (+ = kapaniyor)
    ego_vv = ego_cur[..., 3:5].unsqueeze(1)
    closing = -((pos * (vel - ego_vv)).sum(-1) / d_center.clamp(min=1e-3))
    v_lat_corr = -torch.sin(tan_eff) * vel[..., 0] + torch.cos(tan_eff) * vel[..., 1]
    v_lat_toward = -torch.sign(d_lat_eff) * v_lat_corr          # + = koridora yaklasiyor

    # --- zaman-hizali gelecek buyuklukleri ---
    T = neighbor_futures.shape[2]
    tgrid = torch.arange(1, T + 1, device=neighbor_agents_past.device).float() * DT   # [T]
    ego_sweep = _arc_walk(cxy, ccum, cvalid, ego_v[:, None] * tgrid[None, :])          # [B,T,2]
    fut = neighbor_futures[..., :2].float()                                            # [B,N,T,2]
    d_align = (fut - ego_sweep[:, None]).norm(dim=-1)                                  # [B,N,T]
    # gelecegin koridora laterali (giris tespiti) -- projeksiyonu topluca yap
    fs, fdlat, _, _ = _project(fut.reshape(B, N * T, 2), cxy, cyaw, ccum, cvalid)
    fs = fs.view(B, N, T)
    fdlat = fdlat.view(B, N, T)

    # --- tip guard'lari (KG tanimlarindan transcribe: follows/changesLane -> VehicleLike,
    # overtakes -> MotorVehicle). Tip one-hot: [...,8:11] = [vehicle, pedestrian, bicycle].
    # Serit-topoloji kanallari serit KULLANAN aktorler icindir: vehicle + bicycle (veh_like).
    # overtaking: yalnizca motorlu arac (KG MotorVehicle). Yayalar collision-course /
    # path-crossing / proximity kanallarindan girer -- onlarin dogru kanallari bunlar.
    tip = cur[..., 8:11].argmax(-1)                             # 0=veh, 1=ped, 2=bic
    veh_like = tip != 1
    motor = tip == 0

    # --- kanallar ---
    active = torch.zeros(B, N, NUM_CHANNELS, dtype=torch.bool, device=pos.device)
    inlane = d_lat_eff.abs() <= 0.5 * LANE_W
    adjL = (d_lat_eff > 0.5 * LANE_W) & (d_lat_eff <= 1.5 * LANE_W)
    adjR = (d_lat_eff < -0.5 * LANE_W) & (d_lat_eff >= -1.5 * LANE_W)

    active[..., CH_SAME_LANE_AHEAD] = inlane & same_flow & (ds > 0) & (ds <= AHEAD_MAX_M) & veh_like
    active[..., CH_SAME_LANE_BEHIND] = inlane & same_flow & (ds < 0) & (ds >= -BEHIND_MAX_M) & veh_like
    # arkaya BOYUNA kisit (kullanici istegi 2026-08-15): duz-cizgi sanal projeksiyon cok geriyi
    # de sokuyordu (35 m merkez kapagi tek basina -34 m'yi geciriyor). Deger KG'den:
    # directional_same_lane_behind_m = 25 -- geriye-dogru ilgi menzili semantigi ayni.
    adj_lon_ok = ds >= -BEHIND_MAX_M
    active[..., CH_ADJACENT_LEFT] = adjL & same_flow & (d_center <= ADJ_MAX_M) & adj_lon_ok & veh_like
    active[..., CH_ADJACENT_RIGHT] = adjR & same_flow & (d_center <= ADJ_MAX_M) & adj_lon_ok & veh_like

    # collision-course: zaman-hizali mesafe clearance'in altina iniyor (KG CPA sabitleri).
    # Yaricap = yarim-GENISLIK (yarim-diyagonal degil): diyagonal, yandan gecen normal trafigi
    # (3.5 m lateral) bile "çakışma"ya ceviriyordu. d_fs (proximity) diyagonal kalir.
    r_w = 1.0 + 0.5 * W                                                                # [B,N]
    clear = d_align - r_w[..., None]
    hit = clear <= CPA_CLEARANCE_M                                                     # [B,N,T]
    ttc = torch.where(hit.any(-1),
                      tgrid[None, None, :].expand_as(hit).masked_fill(~hit, HORIZON_S).min(-1).values,
                      torch.full_like(d_center, HORIZON_S))
    # YON sarti (gorsel denetim 2026-08-31): kanal yonsuzdu -- ego'yu ARKADAN takip eden
    # aracin rotasi ego'nunkiyle ortustugu icin de yaniyordu (yanan girdilerin %17.4'u
    # arkadaki ajanda, M_cas kutlesinin %40-42'si arkaya kaciyordu). Ego'nun KARAR vermesi
    # gereken catisma ya ONUNDE (yetisiyor) ya da KESISEN bir akistadir; arkadan gelen
    # ayni-yonlu trafik ego'nun freninin sebebi degildir.
    cc_directional = (ds > 0) | (dtheta_flow > CROSSING_MIN_ANGLE)
    active[..., CH_COLLISION_COURSE] = hit.any(-1) & (closing > 0) & cc_directional

    # sharesIntersectionWith (v1 geometrik proxy): predicted yol koridoru ILERIDE kesiyor
    fut_head = torch.cat([fut[:, :, 1:] - fut[:, :, :-1], fut[:, :, -1:] - fut[:, :, -2:-1]], dim=2)
    fut_ang = torch.atan2(fut_head[..., 1], fut_head[..., 0])
    on_corr = (fdlat.abs() <= 0.5 * LANE_W) & (fs > 0) & (fs <= CROSSING_LOOKAHEAD_M)
    # kesisme acisi: o noktadaki hareket yonu vs koridor tanjanti (yaklasik: en yakin nokta tanjanti)
    cross_ang = _wrap(fut_ang - tan_eff[..., None]).abs()
    crossing_pt = on_corr & (cross_ang > CROSSING_MIN_ANGLE) & (cross_ang < math.pi - CROSSING_MIN_ANGLE)
    active[..., CH_SHARES_INTERSECTION] = crossing_pt.any(-1)

    # merges (anticipated): adjacent ∧ koridora dogru lateral hiz ∧ future koridora giriyor.
    # Yon (onume mi arkama mi) evidence'taki ds_entry isaretinde -- np:mergesInFrontOf/Behind
    # ayrimina bolmek istenirse tek satir: sign(ds_entry).
    t_entry_hit = (fdlat.abs() <= 0.5 * LANE_W) & (fs > 0)
    t_entry = torch.where(t_entry_hit.any(-1),
                          tgrid[None, None, :].expand_as(t_entry_hit).masked_fill(~t_entry_hit, HORIZON_S).min(-1).values,
                          torch.full_like(d_center, HORIZON_S))
    idx_entry = t_entry_hit.float().argmax(-1)                                          # ilk giris indeksi
    fs_entry = torch.gather(fs, 2, idx_entry.unsqueeze(-1)).squeeze(-1)
    ds_entry = torch.where(t_entry_hit.any(-1), fs_entry - ego_v[:, None] * t_entry,
                           torch.zeros_like(d_center))
    active[..., CH_MERGES] = (adjL | adjR) & (v_lat_toward >= MERGE_VLAT_MIN) & t_entry_hit.any(-1) & veh_like

    # gecmis relatif boyuna konum (follows sureklilik + overtakes sira-degisimi icin ortak)
    past_rel = neighbor_agents_past[..., 0] - ego_agent_past[..., 0].unsqueeze(1)      # [B,N,21] ego-frame x

    # follows (frame-level KG zarfi): ayni koridorda onumde ∧ bumper-gap/headway zarfi
    # (moving VEYA kuyruk) ∧ EN YAKIN lider (ikinciyle marj > 0.50) ∧ 1 s "onde" surekliligi.
    # Sapma: KG'nin 1 s KOSUL-surekliligi yerine 1 s "onde kalma" proxy'si (2 s pencereden).
    g = ds - 0.5 * L - EGO_HALF_LEN_M                                                   # kaba bumper gap
    v_o = vel.norm(dim=-1)
    ego_vN = ego_v[:, None]
    moving_f = (ego_vN > 0.30) & (g > 0) & (g <= FOL_MAX_GAP_M) & (g <= FOL_HEADWAY_S * ego_vN)
    queue_f = (ego_vN <= FOL_QUEUE_VS) & (v_o <= FOL_QUEUE_VO) & (g > 0) & (g <= FOL_QUEUE_GAP_M)
    fol_cand = inlane & same_flow & (ds > 0) & motor & (moving_f | queue_f) & neighbor_valid
    ds_c = ds.masked_fill(~fol_cand, 1e9)
    best_v, best_i = ds_c.min(dim=-1, keepdim=True)
    second_v = ds_c.scatter(1, best_i, torch.full_like(best_v, 1e9)).min(dim=-1, keepdim=True).values
    unique_leader = (second_v - best_v) > FOL_MARGIN_M
    ahead_1s = (past_rel[..., -FOL_PERSIST_FRAMES:] > 0).all(-1)
    active[..., CH_FOLLOWS] = fol_cand & (ds_c == best_v) & unique_leader & ahead_1s

    # overtakes (anticipated): gecmiste relatif ilerliyordu ∧ su an yanimda ∧ future'da onume geciyor
    past_adv = past_rel[..., -1] - past_rel[..., 0]                                     # 2 s'lik ilerleme
    ds_fut = fs - (ego_v[:, None, None] * tgrid[None, None, :])
    v_long_rel = ((vel - ego_vv) * torch.stack([torch.cos(tan_eff), torch.sin(tan_eff)], -1)).sum(-1)
    # SIRA-DEGISIMI sarti (validation kalibrasyonu: bu olmadan "yan seritte zaten onde ve hizli
    # akan" normal trafik overtaking sayiliyordu, %4.1 fire): 2 s once en fazla ~1 m onde olmali.
    was_not_ahead = past_rel[..., 0] <= OT_PAST_ADVANCE_M
    # TAMAMLANMA sarti (gorsel denetim 2026-08-31): KG np:overtakes gecisi TAMAMLANMIS sayar
    # (gec VE serite geri don). Eski hal yalnizca 'onume gecer' istiyordu, bu yuzden yanindan
    # gecip KENDI seridinde kalan arac da overtake etiketi aliyordu. Simdi ayni zaman
    # adiminda hem onde (ds_fut >= OT_CLEAR_M) hem de EGO'NUN seridinde (|fdlat| yarim serit)
    # olmasi gerekiyor -- yani onume kesip giriyor.
    ot_completes = ((ds_fut >= OT_CLEAR_M) & (fdlat.abs() <= 0.5 * LANE_W)).any(-1)
    active[..., CH_OVERTAKES] = ((past_adv >= OT_PAST_ADVANCE_M)
                                  & was_not_ahead
                                  & (ds.abs() <= OT_WINDOW_DS_M)
                                  & (d_lat_eff.abs() >= OT_SIDE_MIN_LAT)
                                  & (v_long_rel >= OT_REL_SPEED_MIN)
                                  & ot_completes
                                  & motor)                       # KG: overtakes -> MotorVehicle

    # vulnerable_road_user_near_ego_path: yaya/bisiklet ∧ <= 12 m ∧ YAKLASIYOR.
    # Konum-tabanli (future'siz) -- yayanin serit-kanali muadili; near'dan ONCE atanir
    # ki none_yet dogru hesaplansin.
    active[..., CH_VRU] = (tip != 0) & (d_center <= VRU_CRIT_M) & (closing > 0)

    # proximity-fallback: yakin ama hicbir sey yanmadi (coverage sigortasi)
    none_yet = ~active.any(-1)
    active[..., CH_NEAR] = none_yet & (d_fs <= NEAR_M)

    # koridor yoksa (ref_path bos) topoloji/gelecek kanallari guvenilmez -> yalniz proximity kalir
    no_corr = ~has_corr
    if no_corr.any():
        keep = torch.zeros_like(active)
        keep[..., CH_NEAR] = d_fs <= NEAR_M
        keep[..., CH_VRU] = (tip != 0) & (d_center <= VRU_CRIT_M) & (closing > 0)  # koridor istemez
        active[no_corr] = keep[no_corr]

    active = active & neighbor_valid.unsqueeze(-1)

    evidence = torch.stack([
        ds, d_lat_eff, d_fs, closing, ttc, t_entry, dtheta_flow, v_lat_toward, ds_entry,
    ], dim=-1).to(dev)
    evidence = evidence * neighbor_valid.unsqueeze(-1)

    return active, evidence


# ======================== R2: ego <- harita elemani kanallari ========================
# Isimler KG'nin kendi predicate/reason adlari (kural: kanal id = np: kavraminin adi).
# npz-only v1'de map_api olmadigi icin iki slot REZERVE (R1'deki connected-* gibi):
#   sharesIntersectionWith (intersection poligonu npz'de yok) ve crosswalk-on-path.
# data_process (map_api'li) surumu ayni isimlerle birebir KG mantigini kosacak.
MCH_IN_LANE = 0            # np:inLane / np:hasPrimaryLane (ego bu elemanin icinde)
MCH_ADJ_LEFT = 1           # adjacent_left  (_pair_map_relation degeri; geometrik proxy)
MCH_ADJ_RIGHT = 2          # adjacent_right
MCH_SUCCESSOR = 3          # successor (ego'nun gececegi yol uzerinde, ileride; geometrik proxy)
MCH_SHARES_INT = 4         # np:inIntersection -- REZERVE, map_api ister. (sharesIntersectionWith
                           # AJAN-cifti predicate'i; harita elemani icin dogru kavram inIntersection/
                           # hasPrimaryMapIntersection bilesimi: eleman ∈ ego'nun kavsagI)
MCH_ROUTE = 5              # ego_route_corridor (relevance_logic adi; route tokenlari yapisal)
MCH_TRAFFIC = 6            # traffic_control (shared_traffic_control ailesi; TL kaydi + path-ilişkili)
MCH_NEAR = 7               # np:near fallback
NUM_MAP_CHANNELS = 8
MAP_CHANNEL_NAMES = ["inLane", "adjacent_left", "adjacent_right", "successor",
                     "inIntersection", "ego_route_corridor", "traffic_control", "near"]
NUM_MAP_EVIDENCE = 8       # [min_dist_ego, d_lat_corr, s_nearest, dtheta, tl_onehot(4)]


@torch.no_grad()
def compute_map_channels(map_lanes, map_crosswalks, route_lanes, ref_path):
    """R2: her harita elemani (token sirasi MODELLE AYNI: 40 lane + 5 crosswalk + 10 route) icin
    active [B,S,NUM_MAP_CHANNELS] ve evidence [B,S,NUM_MAP_EVIDENCE].
    map_lanes [B,L,P,7] (x,y,heading + TL one-hot 4), map_crosswalks [B,C,P2,3],
    route_lanes [B,R,P3,3], ref_path [B,5,1200,>=3]. Ego = origin (ego frame)."""
    B = map_lanes.shape[0]
    dev = map_lanes.device

    def _elems(t):
        xy = t[..., :2].float()
        hd = t[..., 2].float()
        valid = xy.abs().sum(-1) > 1e-6                     # [B,E,P]
        return xy, hd, valid

    lx, lh, lv = _elems(map_lanes)
    cx, ch_, cv = _elems(map_crosswalks)
    rx, rh, rv = _elems(route_lanes)
    tl = map_lanes[..., 3:7].float()                        # [B,L,P,4]

    # P boyutlari farkli (50/30/50) -- ortak P'ye pad et
    Pmax = max(lx.shape[2], cx.shape[2], rx.shape[2])

    def _pad(x, val=0.0):
        if x.shape[2] == Pmax:
            return x
        pad_shape = list(x.shape)
        pad_shape[2] = Pmax - x.shape[2]
        return torch.cat([x, torch.full(pad_shape, val, dtype=x.dtype, device=x.device)], dim=2)

    exy = torch.cat([_pad(lx), _pad(cx), _pad(rx)], dim=1)          # [B,S,Pmax,2]
    ehd = torch.cat([_pad(lh), _pad(ch_), _pad(rh)], dim=1)         # [B,S,Pmax]
    ev_ = torch.cat([_pad(lv.float()), _pad(cv.float()), _pad(rv.float())], dim=1) > 0.5
    S = exy.shape[1]
    L = lx.shape[1]
    C = cx.shape[1]
    elem_valid = ev_.any(-1)                                        # [B,S]
    is_route_tok = torch.zeros(B, S, dtype=torch.bool, device=dev)
    is_route_tok[:, L + C:] = True
    is_cross_tok = torch.zeros(B, S, dtype=torch.bool, device=dev)
    is_cross_tok[:, L:L + C] = True

    cxy, cyaw, ccum, cvalid = _corridor_arrays(ref_path)

    # tum eleman noktalarini koridora projeksiyonla
    flat = exy.reshape(B, S * Pmax, 2)
    fs, fdlat, fyaw, _ = _project(flat, cxy, cyaw, ccum, cvalid)
    fs = fs.view(B, S, Pmax)
    fdlat = fdlat.view(B, S, Pmax)
    big = torch.tensor(1e9, device=dev)
    fdlat_m = torch.where(ev_, fdlat, big)                          # gecersiz nokta etkisiz

    # ego'ya (origin) mesafeler
    d_ego = torch.where(ev_, exy.norm(dim=-1), big)                 # [B,S,Pmax]
    min_d_ego, min_idx = d_ego.min(dim=-1)                          # [B,S]
    hd_near = torch.gather(ehd, 2, min_idx.unsqueeze(-1)).squeeze(-1)

    # inLane: ego merkezine en yakin nokta band icinde ∧ heading AYNI YONDE (abs YOK --
    # abs zit yonu de kabul ediyordu; KG'de serit yonu tektir)
    in_band_ego = min_d_ego <= 0.5 * LANE_W
    aligned_ego = torch.cos(hd_near) > math.cos(MAP_DIR_RAD)           # harita yon esigi 0.60
    inlane = in_band_ego & aligned_ego & ~is_cross_tok

    # koridor-iliskili istatistikler (yalniz koridor bandindaki ileri kisim)
    on_corr_pt = (fdlat_m.abs() <= 0.5 * LANE_W) & (fs > 1.0)
    frac_on_corr = (on_corr_pt & ev_).float().sum(-1) / ev_.float().sum(-1).clamp(min=1.0)
    successor = (frac_on_corr > 0.3) & ~inlane & ~is_cross_tok

    # adjacent: noktalarinin medyan laterali bantta ∧ AYNI YON (abs YOK -- karsi yon serit
    # adjacent DEGILDIR; KG _pair_map_relation adjacency'si ayni-yon roadblock seritleri)
    # ∧ tamamen 35 m'den geride degil (R1'deki BEHIND_MAX ile ayni mantik; onceden kisit yoktu
    # ve tamamen arkadaki paralel seritler koridor baslangicina yapisip adjacent yaniyordu).
    med_dlat = torch.where(ev_, fdlat, torch.zeros_like(fdlat)).sum(-1) / ev_.float().sum(-1).clamp(min=1.0)
    tan_near = torch.gather(fyaw.view(B, S, Pmax), 2, min_idx.unsqueeze(-1)).squeeze(-1)
    par = torch.cos(hd_near - tan_near) > math.cos(MAP_DIR_RAD)
    # adjacent SERIT ego'ya ULASMALI (en ileri nokta x > 0): tamamen arkada kalmis parca
    # planlamaya girmez -- 35 m lookbehind AJAN kuralidir, harita elemanina uygulanmaz.
    # (Yanindaki seridin sana degen kismi zaten ayri elemandir ve o yanar.)
    reaches_ego = torch.where(ev_, exy[..., 0], torch.tensor(-1e9, device=dev)).amax(-1) > 0.0
    adjL = (med_dlat > 0.5 * LANE_W) & (med_dlat <= 1.5 * LANE_W) & par & reaches_ego & ~is_cross_tok
    adjR = (med_dlat < -0.5 * LANE_W) & (med_dlat >= -1.5 * LANE_W) & par & reaches_ego & ~is_cross_tok

    # traffic_control: GERCEK sinyal durumu olan lane ∧ path-iliskili.
    # nuPlan one-hot [green, yellow, red, UNKNOWN]: sinyalsiz her segment unknown=[0,0,0,1]
    # tasir (LaneSegmentTrafficLightData) -> yalniz ilk 3 dim'e bak, yoksa her lane yanar.
    has_tl = torch.zeros(B, S, dtype=torch.bool, device=dev)
    has_tl[:, :L] = tl[..., :3].abs().sum(-1).amax(-1) > 1e-6
    tl_onehot = torch.zeros(B, S, 4, device=dev)
    tl_onehot[:, :L] = tl.amax(dim=2)                               # eleman-bazina max one-hot

    # route: EGO'YA ULASMAYAN (tamamen geride kalan) token yanmaz -- kullanici karari 2026-08-15:
    # lookbehind = 0, "icinde oldugumuz route'dan baslanir". (KG'nin rank-lookbehind=1 kurali
    # AJAN relevansi icindir; harita elemani icin geride kalan route parcasi plana girmez.)
    route_fire = is_route_tok & reaches_ego

    active = torch.zeros(B, S, NUM_MAP_CHANNELS, dtype=torch.bool, device=dev)
    active[..., MCH_IN_LANE] = inlane
    active[..., MCH_ADJ_LEFT] = adjL
    active[..., MCH_ADJ_RIGHT] = adjR
    active[..., MCH_SUCCESSOR] = successor
    active[..., MCH_ROUTE] = route_fire
    active[..., MCH_TRAFFIC] = has_tl & (inlane | successor | is_route_tok)
    none_yet = ~active.any(-1)
    active[..., MCH_NEAR] = none_yet & (min_d_ego <= 20.0)
    active = active & elem_valid.unsqueeze(-1)

    s_near = torch.gather(fs, 2, min_idx.unsqueeze(-1)).squeeze(-1)
    evidence = torch.cat([
        min_d_ego.unsqueeze(-1), med_dlat.unsqueeze(-1), s_near.unsqueeze(-1),
        _wrap(hd_near - tan_near).abs().unsqueeze(-1), tl_onehot,
    ], dim=-1)
    evidence = evidence * elem_valid.unsqueeze(-1)
    return active, evidence
