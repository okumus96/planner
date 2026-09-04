"""NEDENSEL-GEREKLILIK SETI — GT'den dogrulanan, ONUMDEKI ARAC sahneleri.

Tavani %100 olan bir set: "bu ajani silmek karari degistirmeli". Model hic kullanilmaz --
ne kanallar karar verir, ne psi, ne GF tahminleri; yalniz kayitli gelecekler, harita ve
kanal EVIDENCE'i (saf geometrik olcumler).

FILTRELER -- sert, istisna yok:
  F1  uzman frenledi        decision_lon in SLOW
  F2  ego hareketli         v0 >= 3 m/s
  F3  yol DUZ               ego'nun 8 s'de gidecegi TUM mesafe boyunca |dyaw| <= 0.20 rad
  F4  isik yok              map_channel_active[MCH_TRAFFIC] hicbir yerde yanmiyor
  F5  ajan ONDE             ds > 0
  F6  ajan YAKIN            ds <= 40 m ve d_fs > 0 (t=0'da temas halinde degil)
  F7  ajan YAKLASIYOR       closing > 0
  F8  ADJACENT DEGIL        adjacent_left / adjacent_right YANMIYOR
  F9  TEK aday              F5-F8'i saglayan tam olarak bir ajan
  F10 gercekten carpardi    sabit-hiz ego, o ajanla SERIT-ICI boyuna bosluk <= -2 m
"""
import argparse
import glob
import json
import os
import sys
import collections

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GameFormer.channels import (select_ego_corridor, MCH_TRAFFIC, CHANNEL_NAMES,
                                 CH_ADJACENT_LEFT, CH_ADJACENT_RIGHT,
                                 EV_DS, EV_DLAT, EV_DFS, EV_CLOSING)
from GameFormer.decision_labels import decision_labels_single, LON_CLASSES

SLOW = [LON_CLASSES.index(c) for c in
        ('stop_quickly', 'stop_gently', 'slow_quickly', 'slow_gently')]
DT, HORIZON = 0.1, 80
V_MIN, CURV_MAX, DS_MAX = 3.0, 0.20, 40.0
LAT_TOL, GAP_MAX, EGO_HALF_L = 1.0, -2.0, 2.31
DLAT_MAX = 1.75          # F5b: ajan EGO'NUN SERIDINDE (yarim serit genisligi)


def project(cor_xy, pts):
    d = np.linalg.norm(pts[:, None, :] - cor_xy[None, :, :], axis=-1)
    k = d.argmin(1)
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(cor_xy[1:] - cor_xy[:-1], axis=-1))])
    t = np.zeros_like(cor_xy); t[:-1] = cor_xy[1:] - cor_xy[:-1]; t[-1] = t[-2]
    t /= np.maximum(np.linalg.norm(t, axis=-1, keepdims=True), 1e-6)
    n = np.stack([-t[:, 1], t[:, 0]], -1)
    return arc[k], ((pts - cor_xy[k]) * n[k]).sum(-1)


def main(a):
    files = sorted(glob.glob(a.valid_set + "/*.npz"))
    print(f"[data] {len(files)} sahne")
    keep, c = [], dict.fromkeys(['total', 'f1', 'f2', 'f3', 'f4', 'f9', 'f10'], 0)
    for si, f in enumerate(files):
        d = np.load(f); c['total'] += 1
        lon, _ = decision_labels_single(d['ego_agent_future'], d['c_lat_candidates'])
        if int(lon) not in SLOW:
            continue
        c['f1'] += 1
        v0 = float(np.linalg.norm(d['ego_agent_past'][-1, 3:5]))
        if v0 < V_MIN:
            continue
        c['f2'] += 1

        rp = torch.from_numpy(d['c_lat_candidates']).float()[None]
        cor = d['c_lat_candidates'][int(select_ego_corridor(rp)[0])]
        xy, yaw = cor[..., :2], cor[..., 2]
        val = np.abs(xy).sum(-1) > 1e-6
        if val.sum() < 5:
            continue
        cor_xy, cor_yaw = xy[val], yaw[val]
        arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(cor_xy[1:] - cor_xy[:-1], axis=-1))])
        reach = v0 * DT * HORIZON
        if arc[-1] < reach * 0.8:
            continue
        m = arc <= reach
        dth = np.abs(np.arctan2(np.sin(cor_yaw[m] - cor_yaw[0]), np.cos(cor_yaw[m] - cor_yaw[0])))
        if dth.max() > CURV_MAX:
            continue
        c['f3'] += 1
        if bool(d['map_channel_active'][:, MCH_TRAFFIC].any()):
            continue
        c['f4'] += 1

        nb, nf = d['neighbor_agents_past'][:10], d['neighbor_agents_future'][:10]
        ch, ev = d['channel_active_gf'][:10], d['channel_evidence_gf'][:10]
        cand = [j for j in range(10)
                if np.abs(nb[j]).sum() != 0
                and ev[j, EV_DS] > 0 and ev[j, EV_DS] <= DS_MAX and ev[j, EV_DFS] > 0
                and abs(ev[j, EV_DLAT]) <= DLAT_MAX
                and ev[j, EV_CLOSING] > 0
                and not ch[j, CH_ADJACENT_LEFT] and not ch[j, CH_ADJACENT_RIGHT]]
        if len(cand) != 1:
            continue
        c['f9'] += 1
        j = cand[0]

        want = v0 * DT * np.arange(1, HORIZON + 1)
        ecv = cor_xy[np.clip(np.searchsorted(arc, want), 0, len(cor_xy) - 1)]
        s_ego, _ = project(cor_xy, ecv)
        fj = nf[j, :HORIZON, :2]; ok = np.abs(fj).sum(-1) > 1e-6
        if ok.sum() < 10:
            continue
        L, W = max(float(nb[j, -1, 6]), 1.0), max(float(nb[j, -1, 7]), 1.0)
        s_j, lat_j = project(cor_xy, fj[ok])
        in_lane = np.abs(lat_j) <= LAT_TOL + 0.5 * W
        if not in_lane.any():
            continue
        gap = float((s_j[in_lane] - s_ego[ok][in_lane] - (EGO_HALF_L + 0.5 * L)).min())
        if gap > GAP_MAX:
            continue
        c['f10'] += 1
        keep.append(dict(scene=si, file=os.path.basename(f), agent=j,
                         lon_gt=LON_CLASSES[int(lon)], v0=round(v0, 2),
                         curv=round(float(dth.max()), 3), gap=round(gap, 2), clearance=round(gap, 2),
                         ds=round(float(ev[j, EV_DS]), 1), dfs=round(float(ev[j, EV_DFS]), 1), dlat=round(float(ev[j, EV_DLAT]), 2),
                         closing=round(float(ev[j, EV_CLOSING]), 2),
                         rels=[CHANNEL_NAMES[k] for k in range(len(CHANNEL_NAMES)) if ch[j, k]]))

    print("\n=== HUNI ===")
    for k, l in [('total', 'tum sahneler'), ('f1', '+ uzman frenledi'),
                 ('f2', f'+ ego hareketli (v0 >= {V_MIN})'),
                 ('f3', f'+ yol DUZ (gidilen tum mesafe boyunca <= {CURV_MAX} rad)'),
                 ('f4', '+ trafik isigi yok'),
                 ('f9', f'+ TEK aday: SERITIMDE (|dlat|<={DLAT_MAX}) & onde & <={DS_MAX:.0f}m & yaklasan & adj degil'),
                 ('f10', f'+ sabit-hiz ego serit-ici carpardi (bosluk <= {GAP_MAX} m)')]:
        print(f"  {l:<56s} {c[k]:5d}")
    if keep:
        for f_, l in [('ds', 'ds [m]'), ('dfs', 'd_fs [m]'), ('closing', 'closing [m/s]'), ('dlat', 'dlat [m]'),
                      ('gap', 'carpisma boslugu [m]'), ('v0', 'ego v0 [m/s]'), ('curv', 'egrilik [rad]')]:
            v = np.array([r[f_] for r in keep])
            print(f"  {l:<22s} min {v.min():7.2f} | med {np.median(v):7.2f} | max {v.max():7.2f}")
        cc = collections.Counter()
        for r in keep:
            cc.update(r['rels'] or ['(HICBIRI)'])
        print("  yakan kanallar:")
        for k, v in cc.most_common():
            print(f"    {k:<38s} {v:3d}/{len(keep)}")
    json.dump(keep, open(a.out, 'w'), indent=1)
    print(f"\n[done] {len(keep)} sahne -> {a.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--valid_set", required=True)
    p.add_argument("--out", default="/home/lt-hta-ai4/GameFormer-Planner/results_label_identity/causal_set.json")
    main(p.parse_args())
