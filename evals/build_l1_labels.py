"""L1 ETIKETLERI — etkileşim karari katmaninin denetim hedefi.

Tasarim: L0 (yapisal, anlik) -> L1 (etkilesim karari, gelecek gerektirir) -> L2 (manevra).
Bu script L1'in GT etiketlerini uretir. Model, GF tahminleriyle hesaplanmis GURULTULU
kanallardan (channel_active_gf) bu TEMIZ etiketleri tahmin etmeyi ogrenecek -- yani
CHANNELS_AUDIT'in "weak-evidence channels" problemi (IoU %37-67) girdi gurultusu olmaktan
cikip denetim sinyaline donusuyor.

AJAN L1 (6 sinif, oncelik sirasiyla tek etikete indirgenir):
  np:yieldingTo        KG intent.py:101 kurali, GT gelecekler uzerinde
  np:waitingFor        KG intent.py:121
  np:mergesInFrontOf   channel_active_gt[CH_MERGES]
  np:overtakes         channel_active_gt[CH_OVERTAKES]
  np:follows           channel_active_gt[CH_FOLLOWS]
  none

HARITA L1 (2 sinif):
  np:keepsLane         ego'nun GT gelecegi bu serit elemanini takip ediyor
  none
  (np:waitsAtRedLight / np:crossesStopLine CIKARILDI -- olculdu: islenmis veride KIRMIZI
   isik yok, 1118 sahnede RED one-hot'i sifir. data_process.py kirmizi durumu yalnizca rota
   dolulugunda kullaniyor, serit kodlamasina gecmemis.)

Kosum:
  python evals/build_l1_labels.py --valid_set <dir> --out l1_labels.npz
"""
import argparse
import collections
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GameFormer.channels import CH_FOLLOWS, CH_MERGES, CH_OVERTAKES

# --- KG esikleri (predicate_logic.py) ---
DT = 0.1
YIELD_HOR = 60            # future_horizon_s = 6.0
STOPPED = 0.30            # stopped_speed_mps
DECEL = 0.30              # acceleration_threshold_mps2
ARRIVAL_TOL = 1.5         # arrival_time_tolerance_s
PERSIST = 10              # minimum_intent_duration_s = 1.0 @ 10 Hz
WAIT_BLOCK_S = 3.0        # max(minimum_intent_duration_s*2, 3.0)
# KG kuralindan SAPMA (bizim sikilastirmamiz): catisma noktasinda yon farki bu esigin
# altindaysa bu bir KESISME degil, AYNI YONDE TAKIP'tir. KG'nin _common_conflict_region'i
# tampon-kesisimi kullandigi icin paralel yollarda da bolge uretiyor ve onde giden araci
# yieldingTo isaretliyor (olculdu: 12 gorsel panelin 3'u boyle -- sahne 303, 400 aynı
# seritte onde giden araca 'yol veriyor' dedi). 30 derece, KG'nin kendi
# CROSSING_MIN_ANGLE'iyla ayni buyukluk.
CROSS_MIN_RAD = 0.52      # 30 derece
EGO_W = 2.297
# --- keepsLane ---
LANE_LAT_TOL = 1.75       # yarim serit
LANE_COVER = 0.5          # ufkun en az bu orani bu elemanin yakininda gecmeli

AG_CLASSES = ['none', 'np:follows', 'np:yieldingTo', 'np:waitingFor',
              'np:mergesInFrontOf', 'np:overtakes']
MP_CLASSES = ['none', 'np:keepsLane']
# tek etikete indirgeme onceligi: acik KARAR > manevra > varsayilan kisit
AG_PRIORITY = ['np:yieldingTo', 'np:waitingFor', 'np:mergesInFrontOf',
               'np:overtakes', 'np:follows']


def _arrival(path, pt, rad):
    d = np.linalg.norm(path - pt, axis=-1)
    m = d <= rad
    return float(np.argmax(m)) * DT if m.any() else np.inf


def _heading(path, i):
    a = max(0, i - 3); b = min(len(path) - 1, i + 3)
    d = path[b] - path[a]
    return np.arctan2(d[1], d[0])


def intent_labels(ego_xy, ego_v, nb_last, nb_fut):
    """KG intent.py kuralini ego=subject icin uygular. Doner: (yield_set, wait_set)."""
    ys, ws = set(), set()
    for j in range(nb_last.shape[0]):
        if np.abs(nb_last[j]).sum() == 0:
            continue
        obj = np.concatenate([nb_last[j:j + 1, :2], nb_fut[j, :, :2]])
        if (np.abs(obj).sum(-1) > 1e-6).sum() < YIELD_HOR // 2:
            continue
        W = max(float(nb_last[j, 7]), 1.0)
        thr, rad = (EGO_W + W) / 2.0, max(EGO_W, W)
        y_streak = y_best = w_streak = w_best = 0
        # tam mesafe matrisi BIR kez; k pencereleri onun dilimleri (11x hizlanma)
        DF = np.linalg.norm(ego_xy[:, None, :] - obj[None, :, :], axis=-1)
        for k in range(PERSIST + 1):
            sp, op = ego_xy[k:k + YIELD_HOR], obj[k:k + YIELD_HOR]
            if len(sp) < 5 or len(op) < 5:
                break
            D = DF[k:k + YIELD_HOR, k:k + YIELD_HOR]
            if D.min() > thr:                       # ortak catisma bolgesi yok
                y_streak = w_streak = 0
                continue
            a, b = np.unravel_index(D.argmin(), D.shape)
            dth = _heading(sp, a) - _heading(op, b)
            if abs(np.arctan2(np.sin(dth), np.cos(dth))) < CROSS_MIN_RAD:
                y_streak = w_streak = 0                 # ayni yonde takip -> kesisme degil
                continue
            pt = 0.5 * (sp[a] + op[b])
            sa, oa = _arrival(sp, pt, rad), _arrival(op, pt, rad)
            acc = (ego_v[min(k + 5, len(ego_v) - 1)] - ego_v[k]) / 0.5
            decel, stopped = acc <= -DECEL, ego_v[k] <= STOPPED
            clears = np.isfinite(oa) and (not np.isfinite(sa) or oa + ARRIVAL_TOL < sa)
            y_streak = y_streak + 1 if ((decel or stopped) and clears) else 0
            y_best = max(y_best, y_streak)
            blocks = np.isfinite(oa) and oa <= WAIT_BLOCK_S
            w_streak = w_streak + 1 if (stopped and blocks) else 0
            w_best = max(w_best, w_streak)
        if y_best >= PERSIST - 2:
            ys.add(j)
        if w_best >= PERSIST - 2:
            ws.add(j)
    return ys, ws


def keeps_lane(ego_xy, lanes):
    """Ego'nun GT gelecegi hangi serit elemanini takip ediyor -> [L] bool."""
    out = np.zeros(lanes.shape[0], dtype=bool)
    for i in range(lanes.shape[0]):
        pl = lanes[i, :, :2]
        v = np.abs(pl).sum(-1) > 1e-6
        if v.sum() < 2:
            continue
        d = np.linalg.norm(ego_xy[:, None, :] - pl[None, v, :], axis=-1).min(1)
        out[i] = (d <= LANE_LAT_TOL).mean() >= LANE_COVER
    return out


def main(a):
    files = sorted(glob.glob(a.valid_set + "/*.npz"))
    print(f"[data] {len(files)} sahne")
    N, S = 10, None
    AG, MP, FILES = [], [], []
    multi = collections.Counter()
    for f in files:
        d = np.load(f)
        ego_xy = np.concatenate([d['ego_agent_past'][-1:, :2], d['ego_agent_future'][:, :2]])
        ego_v = np.concatenate([[0.], np.linalg.norm(ego_xy[1:] - ego_xy[:-1], axis=-1) / DT])
        ego_v[0] = ego_v[1]
        nb_last, nb_fut = d['neighbor_agents_past'][:N, -1], d['neighbor_agents_future'][:N]
        gt = d['channel_active_gt'][:N]
        ys, ws = intent_labels(ego_xy, ego_v, nb_last, nb_fut)

        ag = np.zeros(N, dtype=np.int64)
        for j in range(N):
            if np.abs(d['neighbor_agents_past'][j]).sum() == 0:
                continue
            fired = []
            if j in ys: fired.append('np:yieldingTo')
            if j in ws: fired.append('np:waitingFor')
            if gt[j, CH_MERGES]: fired.append('np:mergesInFrontOf')
            if gt[j, CH_OVERTAKES]: fired.append('np:overtakes')
            if gt[j, CH_FOLLOWS]: fired.append('np:follows')
            if len(fired) > 1:
                multi[tuple(sorted(fired))] += 1
            for c in AG_PRIORITY:                       # oncelikle tek etikete indirge
                if c in fired:
                    ag[j] = AG_CLASSES.index(c)
                    break
        lanes = d['lanes']
        kl = keeps_lane(ego_xy, lanes)
        nS = lanes.shape[0] + d['crosswalks'].shape[0] + d['route_lanes'].shape[0]
        S = nS if S is None else S
        mp = np.zeros(nS, dtype=np.int64)
        mp[:lanes.shape[0]] = kl.astype(np.int64)       # yalniz serit elemanlari
        AG.append(ag); MP.append(mp); FILES.append(os.path.basename(f))

    AG, MP = np.stack(AG), np.stack(MP)
    print(f"\n=== AJAN L1 ({AG.size} ajan-yuvasi, {int((AG >= 0).sum())} kayit) ===")
    valid = AG.reshape(-1)
    cnt = collections.Counter(valid.tolist())
    for k in range(len(AG_CLASSES)):
        print(f"  {AG_CLASSES[k]:<22s} {cnt[k]:7d}  %{100 * cnt[k] / len(valid):5.2f}")
    print(f"  en az bir non-none ajani olan sahne: "
          f"{int((AG > 0).any(1).sum())}/{len(AG)}  (%{100 * (AG > 0).any(1).mean():.1f})")
    print(f"\n=== HARITA L1 ({MP.shape[1]} eleman) ===")
    m = MP.reshape(-1)
    for k in range(len(MP_CLASSES)):
        print(f"  {MP_CLASSES[k]:<22s} {int((m == k).sum()):7d}  %{100 * (m == k).mean():5.2f}")
    print(f"  sahne basina keepsLane eleman sayisi: med "
          f"{np.median((MP == 1).sum(1)):.0f}, max {int((MP == 1).sum(1).max())}")
    print(f"  hic keepsLane olmayan sahne: {int(((MP == 1).sum(1) == 0).sum())}/{len(MP)}")
    if multi:
        print(f"\n=== COKLU ETIKET (oncelikle tek'e indirgendi) ===")
        for k, v in multi.most_common(8):
            print(f"  {' + '.join(x.replace('np:', '') for x in k):<48s} {v}")
    np.savez_compressed(a.out, agent=AG, map=MP, files=np.array(FILES),
                        ag_classes=np.array(AG_CLASSES), mp_classes=np.array(MP_CLASSES))
    print(f"\n[done] -> {a.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--valid_set", required=True)
    p.add_argument("--out", default="/home/lt-hta-ai4/GameFormer-Planner/results_label_identity/l1_labels.npz")
    main(p.parse_args())
