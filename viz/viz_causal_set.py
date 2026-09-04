"""GT-dogrulanmis nedensel set icin TESHIS BEV'i.

Amac: build_causal_set.py'nin sucladigi ajan ("ego frene basmasaydi TAM buna carpardi")
predicate katmani tarafindan NASIL etiketlenmis, ve etiket yanlissa NEDEN yanlis.

Her panelde:
  - siyah kutu + KESIKLI siyah  : ego ve frene BASMASAYDI izleyecegi sabit-hiz yolu
  - siyah DUZ ince              : ego'nun GERCEK (kayitli) gelecegi -- frene basmis hali
  - KIRMIZI kutu + kirmizi iz   : GT'nin sucladigi ajan ve gercek gelecegi
  - gri kutular                 : diger ajanlar
  - kalin cizgi                 : secilen ego koridoru
  - baslik                      : clearance, v0, ajanin YAKAN kanallari + ds/dlat kaniti

Kosum:
  python viz/viz_causal_set.py --valid_set <dir> --set_json causal_set.json \
      --only adjacent_right --out viz_out/causal_set_adjR.png
"""
import argparse
import json
import math
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GameFormer.channels import (select_ego_corridor, CHANNEL_NAMES, LANE_W,
                                 EV_DS, EV_DLAT, EV_DFS, EV_TTC, EV_CLOSING, EV_DTHETA_FLOW)

DT, HORIZON = 0.1, 80


def box(ax, x, y, th, L, W, **kw):
    c, s = math.cos(th), math.sin(th)
    cx, cy = x - (L / 2) * c + (W / 2) * s, y - (L / 2) * s - (W / 2) * c
    ax.add_patch(Rectangle((cx, cy), L, W, angle=math.degrees(th), **kw))


def const_vel(cor, v0):
    xy = cor[..., :2]
    xy = xy[np.abs(xy).sum(-1) > 1e-6]
    if len(xy) < 2:
        return None
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(xy[1:] - xy[:-1], axis=-1))])
    want = v0 * DT * np.arange(1, HORIZON + 1)
    if arc[-1] < want[-1] * 0.5:
        return None
    return xy[np.clip(np.searchsorted(arc, want), 0, len(xy) - 1)]


def main(a):
    import glob
    sel = json.load(open(a.set_json))
    fs = {os.path.basename(f): f for f in glob.glob(a.valid_set + "/*.npz")}
    picks = []
    for r in sel:
        d = np.load(fs[r['file']])
        ch = d['channel_active_gf'][r['agent']]
        rels = [CHANNEL_NAMES[k] for k in range(len(CHANNEL_NAMES)) if ch[k]]
        if a.only and a.only not in rels:
            continue
        picks.append((r, rels))
    print(f"[set] {len(sel)} sahne, filtre '{a.only or 'yok'}' -> {len(picks)} panel")
    if not picks:
        return
    n = len(picks)
    ncol = min(3, n); nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.2 * ncol, 6.4 * nrow), squeeze=False)
    for ax in axes.ravel():
        ax.axis('off')

    for i, (r, rels) in enumerate(picks):
        ax = axes[i // ncol][i % ncol]; ax.axis('on')
        d = np.load(fs[r['file']])
        # harita
        for pl in d['lanes'][..., :2]:
            v = np.abs(pl).sum(-1) > 1e-6
            if v.sum() > 1:
                ax.plot(pl[v, 0], pl[v, 1], color='0.85', lw=1.0, zorder=1)
        for pl in d['route_lanes'][..., :2]:
            v = np.abs(pl).sum(-1) > 1e-6
            if v.sum() > 1:
                ax.plot(pl[v, 0], pl[v, 1], color='#ff9ec7', lw=1.6, zorder=2)
        # secilen koridor
        rp = torch.from_numpy(d['c_lat_candidates']).float()[None]
        cor = d['c_lat_candidates'][int(select_ego_corridor(rp)[0])]
        cv = np.abs(cor[..., :2]).sum(-1) > 1e-6
        ax.plot(cor[cv, 0], cor[cv, 1], color='#6a3d9a', lw=2.2, zorder=3, label='ego koridoru')
        # ego
        ego = d['ego_agent_past'][-1]
        v0 = float(np.linalg.norm(ego[3:5]))
        box(ax, 0, 0, 0, 4.62, 2.1, fc='k', ec='k', zorder=6)
        ef = d['ego_agent_future'][:, :2]
        ax.plot(ef[:, 0], ef[:, 1], color='k', lw=1.5, zorder=5, label='ego GERCEK (fren yapti)')
        ecv = const_vel(cor, v0)
        if ecv is not None:
            ax.plot(ecv[:, 0], ecv[:, 1], color='k', lw=2.2, ls='--', zorder=5,
                    label='ego FREN YAPMASAYDI')
        # ajanlar
        nb, nf = d['neighbor_agents_past'][:10], d['neighbor_agents_future'][:10]
        for j in range(10):
            if np.abs(nb[j]).sum() == 0:
                continue
            s = nb[j, -1]
            tgt = (j == r['agent'])
            box(ax, s[0], s[1], s[2], max(s[6], 1.0), max(s[7], 1.0),
                fc=('#d62728' if tgt else '0.75'), ec='k', lw=1.0 if tgt else 0.4,
                zorder=6 if tgt else 4, alpha=1.0 if tgt else 0.7)
            fj = nf[j, :, :2]; ok = np.abs(fj).sum(-1) > 1e-6
            if ok.sum() > 1:
                ax.plot(fj[ok, 0], fj[ok, 1], color=('#d62728' if tgt else '0.8'),
                        lw=1.8 if tgt else 0.8, zorder=5 if tgt else 3)
        ev = d['channel_evidence_gf'][r['agent']]
        ax.set_title(
            f"clearance {r['clearance']:+.1f} m   v0 {v0:.1f} m/s   [{r['lon_gt']}]\n"
            f"YAKAN: {', '.join(rels) or 'HICBIRI'}\n"
            f"ds {ev[EV_DS]:+.1f}  dlat {ev[EV_DLAT]:+.1f} (serit {LANE_W/2:.2f}-{1.5*LANE_W:.2f})  "
            f"d_fs {ev[EV_DFS]:.1f}  TTC {ev[EV_TTC]:.1f}  closing {ev[EV_CLOSING]:+.1f}  "
            f"dth {ev[EV_DTHETA_FLOW]:.2f}", fontsize=8.5)
        ax.set_aspect('equal')
        ax.set_xlim(-25, 75); ax.set_ylim(-25, 25)
        ax.tick_params(labelsize=6)
        if i == 0:
            ax.legend(fontsize=7, loc='upper left')
    plt.tight_layout()
    plt.savefig(a.out, dpi=130, bbox_inches='tight')
    print(f"[saved] {a.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--valid_set", required=True)
    p.add_argument("--set_json",
                   default="/home/lt-hta-ai4/GameFormer-Planner/results_label_identity/causal_set.json")
    p.add_argument("--only", type=str, default="", help="sadece bu kanali yakan sahneler")
    p.add_argument("--out", default="viz_out/causal_set.png")
    main(p.parse_args())
