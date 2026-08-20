"""Factored karar etiketleri (longitudinal x lateral) — DOD'un zengin taksonomisi.

Taksonomi nuReasoning'in (Huang et al., 2026, Appendix B.2 Fig. S5) meta-action setinden
uyarlandi: 9 longitudinal + 7 lateral sinif. GT ego gelecegi tek kaynak -> etiketler %100
model-bagimsiz (no-GF-labels kisiti korunur).

Tasarim kararlari (2026-08-17, kullanici ile):
  - Longitudinal etiket ILK 4 s'den (40 frame) okunur: "su anki hiz karari" — 8 s'lik karma
    profillerde (yavasla->hizlan) tek etiket muglaklasirdi.
  - Lateral etiket TAM 8 s'den okunur: donus/serit-degistirme niyeti ufkun ilerisinde olabilir.
  - quickly/gently ayrimi extractor'da KORUNUR; egitimde birlestirme stats sonrasi config karari.
  - nuReasoning'in "remain_stopped => no_lateral" kurali UYGULANMAZ (bilincli sapma): onlarin
    iki aksiyonu AYNI pencereden okunur, bizde pencereler ayrik (lon 4 s / lat 8 s). Bizde
    remain_stopped x turn_left = "durmus, donmek uzere" — korunmasiz donus sahnelerinin tam
    imzasi ve DOD kosullamasi icin en degerli kombinasyon (cross-tab bunu 8 sahnede yakaladi;
    kural onlari no_lateral'e ezip donus niyetini siliyordu). Gercekten park halindeki sahneler
    MIN_ARC kontrolunden dogal olarak no_lateral'e duser.
  - U-turn ayri sinif degil -> turn_left/right'a katlanir (nuReasoning'de de yok; mevcut
    maneuver_labels sag U-turn'u zaten turning_right'a dusuruyordu).

Lane-change tespiti KORIDOR-goreli yapilir (koridor = select_ego_corridor'un sectigi,
ego'nun SERIDINDEN baslayan aday; 2026-08-18'den once sabit aday-0 varsayimi vardi): koridora
gore lateral ofsetin ~serit genisligi kadar degismesi = mantiksal koridor degisimi — KG'nin
np:changesLane predicate'inin (logical corridor semantigi) frame-level karsiligi. Secim
sira-bagimsizdir (ham/sortlu fark etmez). Ref path yoksa geometrik fallback
(ego-frame y yer degistirme + heading-geri-donme guard'i).

Siniflar:
  LON: 0 remain_stopped, 1 stop_quickly, 2 stop_gently, 3 slow_quickly, 4 slow_gently,
       5 accel_quickly, 6 accel_gently, 7 maintain, 8 reverse
  LAT: 0 turn_left, 1 turn_right, 2 lane_change_left, 3 lane_change_right,
       4 inlane_left, 5 inlane_right, 6 no_lateral
"""
import numpy as np
import torch

from .channels import _corridor_arrays, _project

LON_CLASSES = ['remain_stopped', 'stop_quickly', 'stop_gently', 'slow_quickly', 'slow_gently',
               'accel_quickly', 'accel_gently', 'maintain', 'reverse']
LAT_CLASSES = ['turn_left', 'turn_right', 'lane_change_left', 'lane_change_right',
               'inlane_left', 'inlane_right', 'no_lateral']
NUM_LON = len(LON_CLASSES)
NUM_LAT = len(LAT_CLASSES)

# Ters-frekans CE agirliklari (dod_meta egitimi): w_c = N/(K*n_c), cap 10.0, bos sinif -> 1.0.
# Sayimlar 20k train orneklemi (eval_decisions.py --limit 20000, decision_stats_train20k.json):
# lon [4499, 73, 383, 238, 1237, 1928, 5328, 6314, 0], lat [6130, 2955, 235, 291, 523, 616, 9250].
# Duz CE ile psi_lon her sahneye 'maintain' der ve nadir/guvenlik-kritik siniflari (sert fren,
# lane change) bedavaya yok sayardi -- 5-sinif DOD'da gerek yoktu (denge ~2:1), burada 80:1.
LON_CE_WEIGHT = [0.49, 10.0, 5.80, 9.34, 1.80, 1.15, 0.42, 0.35, 1.0]
LAT_CE_WEIGHT = [0.47, 0.97, 10.0, 9.82, 5.46, 4.64, 0.31]

# lon_merge=1: quickly/gently katlama (psi_lon confusion matrisi ayrimi ogrenemezse) -- 9 -> 6.
# Etiket LON_MERGE_MAP ile remap edilir, psi_lon/embedding 6 boyutlu kurulur.
LON_MERGE_MAP = [0, 1, 1, 2, 2, 3, 3, 4, 5]
LON_MERGED_CLASSES = ['remain_stopped', 'stop', 'slow', 'accel', 'maintain', 'reverse']
NUM_LON_MERGED = len(LON_MERGED_CLASSES)
LON_MERGED_CE_WEIGHT = [0.74, 7.31, 2.26, 0.46, 0.53, 1.0]   # birlesik sayimlardan ayni formul

DT = 0.1                 # [s] frame araligi
LON_WINDOW = 40          # longitudinal pencere: ilk 4 s
V_STOP = 0.5             # [m/s] altinda "duruk" sayilir
V_MOVING = 1.0           # [m/s] remain_stopped icin pencere boyunca asilmamasi gereken hiz
DV_BAND = 1.0            # [m/s] maintain bandi: |v_end - v0| < DV_BAND
A_HARD = 1.5             # [m/s^2] quickly/gently ayrimi (0.5 s duzlestirilmis tepe ivme)
A_SMOOTH_W = 5           # ivme duzlestirme penceresi (frame) = 0.5 s
REV_X = -0.5             # [m] pencere sonunda geri net yer degistirme -> reverse
LC_DLAT = 2.0            # [m] koridor-goreli |delta d_lat| >= -> lane change (serit ~3.5 m)
INLANE_DLAT = 0.6        # [m] koridor-goreli |delta d_lat| >= -> in-lane kayma
MIN_ARC = 3.0            # [m] lateral siniflar icin asgari kat edilen yol (durukta lateral yok)
# turn esikleri: train_planner._maneuver_one ile AYNI (get_decision.py'a sadik)
TURN_C_LO, TURN_C_HI = 0.03, 0.18
TURN_HDIFF = 0.2


def _resample_arc(xy, n):
    """xy [M,2] -> yay-uzunlugu boyunca uniform n nokta (train_planner._resample_arc ile ayni)."""
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] < 1e-6:
        return np.repeat(xy[:1], n, axis=0)
    cum = cum / cum[-1]
    t = np.linspace(0.0, 1.0, n)
    return np.stack([np.interp(t, cum, xy[:, 0]), np.interp(t, cum, xy[:, 1])], axis=1)


def _turn_class(xy, yaw):
    """Tam-ufuk donus tespiti (train_planner._maneuver_one'in donus dali; U-turn katlanir).
    Doner: 'left' | 'right' | None."""
    valid = ~np.all(xy == 0, axis=1)
    xy, yaw = xy[valid], yaw[valid]
    if len(xy) < 2:
        return None
    length = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
    if length < MIN_ARC:
        return None
    pts = _resample_arc(xy, max(int(length), 2))
    tan = np.diff(pts, axis=0)
    tan = tan / np.clip(np.linalg.norm(tan, axis=1, keepdims=True), 1e-8, None)
    if len(tan) < 2:
        return None
    ang = np.arccos(np.clip((tan[:-1] * tan[1:]).sum(1), -1.0, 1.0))
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    curv = ang / np.clip(seg[:-1], 1e-8, None)
    sign = np.sign(np.cross(tan[:-1], tan[1:]))
    i = int(np.argmax(curv))
    c = round(float(curv[i]), 2)
    s = float(sign[i])
    diff = round(float(abs(yaw[0] - yaw[-1])), 2)
    turning = (TURN_C_LO < c < TURN_C_HI and diff > TURN_HDIFF) or (0.1 < c < TURN_C_HI)
    uturn = c >= TURN_C_HI
    if not (turning or uturn):
        return None
    if s == 1.0:
        return 'left'
    if s == -1.0:
        return 'right'
    return None


def _lon_one(xy, yaw):
    """Ilk LON_WINDOW frame'den longitudinal sinif (0..8)."""
    w = xy[:LON_WINDOW]
    valid = ~np.all(w == 0, axis=1)
    # gecersiz kuyruk (senaryo sonu 0-pad): son gecerli noktaya kadar kes
    if valid.sum() < 2:
        return LON_CLASSES.index('remain_stopped')
    w = w[valid]
    v = np.linalg.norm(np.diff(w, axis=0), axis=1) / DT                     # [T-1]
    v0 = float(v[:5].mean()) if len(v) >= 5 else float(v.mean())
    v_end = float(v[-5:].mean()) if len(v) >= 5 else float(v.mean())
    # reverse: ego-frame x (ileri ekseni) net geri gidiyorsa
    if w[-1, 0] < REV_X and v.max() > V_STOP:
        return LON_CLASSES.index('reverse')
    if v.max() < V_MOVING and v0 < V_STOP and v_end < V_STOP:
        return LON_CLASSES.index('remain_stopped')
    # duzlestirilmis ivme (0.5 s pencere) -> tepe buyuklugu quickly/gently ayrimi icin
    a = np.diff(v) / DT
    if len(a) >= A_SMOOTH_W:
        kern = np.ones(A_SMOOTH_W) / A_SMOOTH_W
        a = np.convolve(a, kern, mode='valid')
    hard = bool(np.abs(a).max() >= A_HARD) if len(a) else False
    dv = v_end - v0
    if v0 >= V_MOVING and v_end < V_STOP:
        return LON_CLASSES.index('stop_quickly' if hard else 'stop_gently')
    if dv <= -DV_BAND:
        return LON_CLASSES.index('slow_quickly' if hard else 'slow_gently')
    if dv >= DV_BAND:
        return LON_CLASSES.index('accel_quickly' if hard else 'accel_gently')
    return LON_CLASSES.index('maintain')


def _lat_one(xy, yaw, dlat=None):
    """Tam ufuktan lateral sinif (0..6). dlat: koridor-goreli lateral ofset [P] (None -> fallback)."""
    turn = _turn_class(xy, yaw)
    if turn is not None:
        return LAT_CLASSES.index('turn_left' if turn == 'left' else 'turn_right')
    valid = ~np.all(xy == 0, axis=1)
    if valid.sum() < 2:
        return LAT_CLASSES.index('no_lateral')
    length = float(np.linalg.norm(np.diff(xy[valid], axis=0), axis=1).sum())
    if length < MIN_ARC:
        return LAT_CLASSES.index('no_lateral')
    if dlat is not None:
        dl = dlat[valid]
        delta = float(np.median(dl[-max(len(dl) // 8, 1):]) - np.median(dl[:max(len(dl) // 8, 1)]))
    else:
        # fallback: ego-frame y yer degistirme; heading-geri-donme guard'i kavisli yolu
        # (surekli donen heading) serit degisiminden ayirir
        if abs(float(yaw[valid][-1] - yaw[valid][0])) > TURN_HDIFF:
            return LAT_CLASSES.index('no_lateral')
        delta = float(xy[valid][-1, 1] - xy[valid][0, 1])
    if delta >= LC_DLAT:
        return LAT_CLASSES.index('lane_change_left')
    if delta <= -LC_DLAT:
        return LAT_CLASSES.index('lane_change_right')
    if delta >= INLANE_DLAT:
        return LAT_CLASSES.index('inlane_left')
    if delta <= -INLANE_DLAT:
        return LAT_CLASSES.index('inlane_right')
    return LAT_CLASSES.index('no_lateral')


def decision_labels(ego_future, ref_path=None):
    """ego_future [B,80,3] (x,y,heading; ego-frame) [+ ref_path [B,R,P,>=3], aday 0 = ego seridi]
    -> (lon [B], lat [B]) LongTensor, ayni cihazda."""
    ef = ego_future.detach().cpu().numpy()
    B = ef.shape[0]
    dlat_np = None
    if ref_path is not None:
        cxy, cyaw, ccum, cvalid = _corridor_arrays(ref_path)
        pts = ego_future[..., :2].float().to(ref_path.device)               # [B,80,2]
        _, d_lat, _, _ = _project(pts, cxy, cyaw, ccum, cvalid)             # [B,80]
        has_corr = cvalid.any(dim=1).cpu().numpy()                          # koridor bos -> fallback
        dlat_np = d_lat.detach().cpu().numpy()
    lon, lat = [], []
    for b in range(B):
        lo = _lon_one(ef[b, :, :2], ef[b, :, 2])
        dl = dlat_np[b] if (dlat_np is not None and has_corr[b]) else None
        la = _lat_one(ef[b, :, :2], ef[b, :, 2], dl)
        lon.append(lo)
        lat.append(la)
    dev = ego_future.device
    return (torch.tensor(lon, dtype=torch.long, device=dev),
            torch.tensor(lat, dtype=torch.long, device=dev))


def decision_labels_single(ego_future_np, ref_path_np):
    """Loader (DrivingData.__getitem__) icin tek-ornek sarmalayici: [80,3] + [R,P,6] numpy
    -> (lon, lat) int. Koridor adayi select_ego_corridor ile SIRA-BAGIMSIZ secilir (ham veya
    lateral-sortlu c_lat ayni sonucu verir). Koridor modu sart: geometrik fallback'in kavisli
    yollarda %14.8 yanlis-lateral urettigi olculdu."""
    ef = torch.from_numpy(np.ascontiguousarray(ego_future_np)).float().unsqueeze(0)
    rp = torch.from_numpy(np.ascontiguousarray(ref_path_np)).float().unsqueeze(0)
    lon, lat = decision_labels(ef, rp)
    return int(lon[0]), int(lat[0])
