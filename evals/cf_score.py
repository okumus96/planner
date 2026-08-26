"""CF SKORLAYICI (nuReasoning-tarzi, deterministik, model-free): zorlanmis-karar planina
SAFE / SUBOPTIMAL / UNSAFE verdicti + gerekce. VLM yok, annotation yok — girdiler planin
kendisi, GF komsu tahminleri ve harita polyline'lari; planner hizinda, offline analiz icin.

Kurallar (oncelik sirasi):
  UNSAFE-collision : herhangi bir komsunun tahmini future'ina zaman-hizali mesafe < COLL_M
  UNSAFE-offroad   : plan noktasi TUM serit/route polyline'larindan > OFFROAD_M, ust uste
                     >= OFFROAD_STEPS adim
  SUBOPTIMAL       : |ivme| tepe (0.5 s duzlestirilmis) > A_MAX  VEYA  ilerleme < PROG_FRAC x taban
  SAFE             : hicbiri
"""
import numpy as np

COLL_M = 2.0            # [m] merkez-merkez carpisma esigi (arac ayak izi yaklastirmasi)
OFFROAD_M = 2.5         # [m] en yakin serit/route merkez cizgisine yanal uzaklik esigi
OFFROAD_STEPS = 10      # ust uste adim (1.0 s)
A_MAX = 3.0             # [m/s^2] konfor tepe ivme
PROG_FRAC = 0.5         # taban ilerlemenin orani
DT = 0.1


def _arc(xy):
    return float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())


def score_plan(plan, futs, valid, road_pts, base_arc=None):
    """plan [80,2] ego-frame; futs [N,80,2] (GF top-1, ego-frame); valid [N] bool;
    road_pts [P,2] (serit+route polyline noktalari, ego-frame); base_arc: taban plan yay boyu
    (None -> ilerleme kurali atlanir). Doner: (verdict, gerekce)."""
    # 1. collision (time-aligned). Reason: STRUCTURED dict (viz predicate formatlar).
    T = plan.shape[0]
    for j in np.where(valid)[0]:
        f = futs[j]
        fm = np.abs(f).sum(-1) > 1e-3
        if fm.sum() < 2:
            continue
        d = np.linalg.norm(plan[fm] - f[fm], axis=1)
        if d.min() < COLL_M:
            t = float(np.argmin(d) * DT)
            return "UNSAFE", {"pred": "collide", "agent": int(j), "t": t}
    # 2. off-road
    if road_pts is not None and len(road_pts) > 2:
        dmin = np.linalg.norm(plan[:, None, :] - road_pts[None, :, :], axis=-1).min(axis=1)
        off = dmin > OFFROAD_M
        run = 0
        for t in range(T):
            run = run + 1 if off[t] else 0
            if run >= OFFROAD_STEPS:
                return "UNSAFE", {"pred": "offroad", "t": (t - OFFROAD_STEPS + 1) * DT}
    # 3. comfort / progress
    v = np.linalg.norm(np.diff(plan, axis=0), axis=1) / DT
    a = np.diff(v) / DT
    if len(a) >= 5:
        a = np.convolve(a, np.ones(5) / 5, mode="valid")
    if len(a) and np.abs(a).max() > A_MAX:
        return "SUBOPTIMAL", {"pred": "harsh_accel", "a": float(np.abs(a).max())}
    if base_arc is not None and base_arc > 3.0 and _arc(plan) < PROG_FRAC * base_arc:
        return "SUBOPTIMAL", {"pred": "low_progress"}
    return "SAFE", None
