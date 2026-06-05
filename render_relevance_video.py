"""relevance_data.pkl -> 2 PANELLI senaryo VIDEOSU (mp4).

run_nuplan_test.py --debug ile her senaryo klasorune relevance_data.pkl yazilir (per-adim
GAT onem verisi + attention). Bu script o veriden, her kare iki panel olacak sekilde video uretir:

  SOL  (harita): harita oldugu gibi + 0-1 ONEM HEATMAP'i (her eleman onemine gore renkli).
                 Sadece normalize onemi > esik (varsayilan 0.65) olanlarin ID'si haritada yazilir.
  SAG  (graf):   sadece esigi gecen dugumler + ego; aralarindaki ATTENTION baglantilari
                 (0-1, ego-ajan 1.0 gibi) cizgi kalinligi+sayi ile; her dugumun ICINDE id.
                 Soldaki id'lerle birebir eslesir.

ffmpeg gerekmez (OpenCV ile mp4).

Kullanim:
  python render_relevance_video.py --data testing_log/<exp>/.../debug_plots/scenario_01
  python render_relevance_video.py --data testing_log/<exp>/.../debug_plots   # tum senaryolar
  python render_relevance_video.py --data .../relevance_data.pkl --fps 8 --threshold 0.65 --radius 60
"""
import argparse
import glob
import os
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
import numpy as np
import cv2

from GameFormer.relevance_graph import (
    NODE_TYPE_EGO, NODE_TYPE_VEHICLE, NODE_TYPE_PEDESTRIAN, NODE_TYPE_BICYCLE,
    NODE_TYPE_LANE, NODE_TYPE_CROSSWALK, NODE_TYPE_ROUTE,
)

CMAP = 'YlOrRd'   # 0 -> acik sari, 1 -> koyu kirmizi (koyu = onemli)
TYPE_MARKER = {
    NODE_TYPE_VEHICLE: 'o', NODE_TYPE_PEDESTRIAN: 'P', NODE_TYPE_BICYCLE: 'X',
    NODE_TYPE_LANE: 's', NODE_TYPE_CROSSWALK: 'D', NODE_TYPE_ROUTE: '^',
}


def _norm_importance(rec):
    imp = rec['importance'].astype('float32')
    valid = rec['valid']
    vmax = max(imp[valid].max(), 1e-6) if valid.any() else 1.0
    norm = np.where(valid, imp / vmax, 0.0)
    return norm


def draw_bev_heatmap(ax, rec, threshold, radius):
    """SOL panel: harita + 0-1 onem heatmap; sadece >esik olanlarin id'si."""
    valid = rec['valid']; nt = rec['node_types']; pose = rec['node_pose']; sl = rec['slices']
    norm = _norm_importance(rec)
    cmo = plt.get_cmap(CMAP)

    # 1) Once tum haritayi soluk gri ciz (harita "oldugu gibi" gorunur)
    for arr_key in ('lanes_xy', 'route_xy', 'cw_xy'):
        for poly in rec[arr_key]:
            m = np.abs(poly).sum(-1) > 1e-6
            if m.any():
                ax.plot(poly[m, 0], poly[m, 1], '-', color='0.85', linewidth=0.6, zorder=0)

    # 2) Uzerine heatmap: her eleman polyline'ini onemine gore renklendir
    def _heat(arr_key, sl_key):
        if sl_key not in sl:
            return
        arr = rec[arr_key]; start = sl[sl_key][0]
        for e in range(arr.shape[0]):
            node = start + e
            if node >= len(valid) or not valid[node]:
                continue
            m = np.abs(arr[e]).sum(-1) > 1e-6
            if not m.any():
                continue
            w = float(norm[node])
            ax.plot(arr[e][m, 0], arr[e][m, 1], '-', color=cmo(w),
                    linewidth=1.0 + 4.0 * w, alpha=0.35 + 0.65 * w, zorder=2 + int(8 * w))
    _heat('lanes_xy', 'lane'); _heat('route_xy', 'route'); _heat('cw_xy', 'crosswalk')

    # 3) Ajanlar
    a0, a1 = sl['agent']
    for v in range(a0, a1):
        if not valid[v]:
            continue
        x, y = pose[v, 0], pose[v, 1]
        if nt[v] == NODE_TYPE_EGO:
            ax.scatter(x, y, c='deepskyblue', s=260, marker='*', edgecolors='k', linewidths=1.0, zorder=20)
        else:
            ax.scatter(x, y, c=[norm[v]], cmap=CMAP, vmin=0, vmax=1,
                       s=120 + 420 * norm[v], marker=TYPE_MARKER.get(int(nt[v]), 'o'),
                       edgecolors='k', linewidths=0.9, zorder=21)

    # 4) SADECE >esik olanlarin id'si (ego dahil referans icin)
    ax.annotate('0', (pose[a0, 0], pose[a0, 1]), fontsize=8, fontweight='bold', color='blue',
                zorder=30, ha='center', va='center')
    for v in np.where(valid & (norm > threshold))[0]:
        if nt[v] == NODE_TYPE_EGO:
            continue
        ax.annotate(str(int(v)), (pose[v, 0], pose[v, 1]), fontsize=8, fontweight='bold',
                    color='black', zorder=30, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='red', alpha=0.9, lw=0.7))

    sm = cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(0, 1)); sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label='onem (0-1, koyu=YUKSEK)')

    ax.set_xlim(-radius, radius); ax.set_ylim(-radius, radius)
    ax.set_aspect('equal'); ax.set_xlabel('x (m, ego frame)'); ax.set_ylabel('y (m, ego frame)')
    ax.set_title(f'Harita onem heatmap (id yazili = onem>{threshold:.2f})')


def draw_graph(ax, rec, threshold):
    """SAG panel: sadece >esik dugumler + ego; attention baglantilari (0-1) + node ici id."""
    valid = rec['valid']; nt = rec['node_types']; sl = rec['slices']
    norm = _norm_importance(rec)
    A = rec.get('attention')
    if A is not None:
        A = A.astype('float32')

    ego = [int(v) for v in range(*sl['agent']) if valid[v] and nt[v] == NODE_TYPE_EGO]
    passed = [int(v) for v in np.where(valid & (norm > threshold))[0] if nt[v] != NODE_TYPE_EGO]
    shown = (ego[:1] + passed)

    # cember duzeni: ego merkezde, gecenler cevrede
    layout = {}
    if ego:
        layout[ego[0]] = (0.0, 0.0)
    for k, n in enumerate(passed):
        ang = np.pi / 2 - 2 * np.pi * k / max(len(passed), 1)
        layout[n] = (float(np.cos(ang)), float(np.sin(ang)))

    # kenarlar: shown dugumler arasi attention (0-1 normalize), sayi ile
    if A is not None and len(shown) >= 2:
        S = np.maximum(A, A.T); np.fill_diagonal(S, 0.0)
        sub = S[np.ix_(shown, shown)]
        smax = max(sub.max(), 1e-6)
        for a in range(len(shown)):
            for b in range(a + 1, len(shown)):
                i, j = shown[a], shown[b]
                w = S[i, j] / smax
                if w < 0.05:
                    continue
                (x1, y1), (x2, y2) = layout[i], layout[j]
                ax.plot([x1, x2], [y1, y2], '-', color='steelblue',
                        linewidth=0.8 + 5.0 * w, alpha=0.3 + 0.6 * w, zorder=2)
                mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                ax.annotate(f"{w:.2f}", (mx, my), fontsize=8, color='navy', zorder=8,
                            ha='center', va='center',
                            bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='steelblue', alpha=0.85, lw=0.5))

    # dugumler: marker=tip, renk=onem, ICINE id
    for v in shown:
        x, y = layout[int(v)]
        if nt[v] == NODE_TYPE_EGO:
            ax.scatter(x, y, c='deepskyblue', s=950, marker='*', edgecolors='k', linewidths=1.2, zorder=6)
        else:
            ax.scatter(x, y, c=[norm[v]], cmap=CMAP, vmin=0, vmax=1, s=900,
                       marker=TYPE_MARKER.get(int(nt[v]), 'o'), edgecolors='k', linewidths=1.0, zorder=5)
        ax.annotate(str(int(v)), (x, y), fontsize=11, fontweight='bold', color='black', zorder=9,
                    ha='center', va='center', bbox=dict(boxstyle='circle,pad=0.1', fc='white', ec='none', alpha=0.6))

    legend_handles = [
        Line2D([0], [0], marker='*', color='w', markerfacecolor='deepskyblue', markeredgecolor='k', markersize=15, label='ego'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markeredgecolor='k', markersize=10, label='arac'),
        Line2D([0], [0], marker='P', color='w', markerfacecolor='orange', markeredgecolor='k', markersize=10, label='yaya'),
        Line2D([0], [0], marker='X', color='w', markerfacecolor='orange', markeredgecolor='k', markersize=10, label='bisiklet'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='orange', markeredgecolor='k', markersize=10, label='lane'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='orange', markeredgecolor='k', markersize=9, label='crosswalk'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='orange', markeredgecolor='k', markersize=10, label='route'),
        Line2D([0], [0], color='steelblue', lw=4, label='kenar = attention (0-1)'),
    ]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=8, framealpha=0.9)
    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4); ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(f'Iliski grafigi: onem>{threshold:.2f} dugumler (node ici=id)')


def _find_pkls(path):
    if path.endswith('.pkl'):
        return [path]
    return sorted(glob.glob(os.path.join(path, '**', 'relevance_data.pkl'), recursive=True))


def render_one(pkl_path, fps, threshold, radius, dpi):
    with open(pkl_path, 'rb') as f:
        blob = pickle.load(f)
    records = blob['records']
    thr = threshold if threshold is not None else blob.get('threshold', 0.65)
    if not records:
        print(f"  [atlandi] kayit yok: {pkl_path}")
        return
    if 'attention' not in records[0]:
        print("  [uyari] kayitlarda 'attention' yok -> sag grafikte kenar cizilemez. "
              "Yeni bir --debug kosusu gerek (attention artik kaydediliyor).")

    out = os.path.join(os.path.dirname(pkl_path), 'relevance.mp4')
    writer = None
    for rec in records:
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(20, 10), dpi=dpi)
        draw_bev_heatmap(axL, rec, thr, radius)
        draw_graph(axR, rec, thr)
        it = rec.get('iteration', '?')
        sel = ''
        if rec.get('lat_idx') is not None:
            v = (rec.get('lon_idx') or 0) / 11.0 * 15.0
            sel = f'  |  secilen lat={rec["lat_idx"]} v={v:.1f} m/s'
        fig.suptitle(f'Iter {it}{sel}', fontsize=13)

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        bgr = cv2.cvtColor(buf[..., :3], cv2.COLOR_RGB2BGR)
        plt.close(fig)

        if writer is None:
            h, w = bgr.shape[:2]
            writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        writer.write(bgr)

    if writer is not None:
        writer.release()
        print(f"  yazildi: {out}  ({len(records)} kare, {fps} fps, esik={thr})")


def main():
    ap = argparse.ArgumentParser(description='relevance_data.pkl -> 2 panelli senaryo videosu')
    ap.add_argument('--data', required=True, help='relevance_data.pkl, senaryo klasoru ya da ust klasor')
    ap.add_argument('--fps', type=float, default=8.0)
    ap.add_argument('--threshold', type=float, default=None, help='normalize onem esigi (varsayilan: pkl icindeki)')
    ap.add_argument('--radius', type=float, default=60.0, help='harita panel yariçapi [m]')
    ap.add_argument('--dpi', type=int, default=100)
    args = ap.parse_args()

    pkls = _find_pkls(args.data)
    if not pkls:
        print(f"relevance_data.pkl bulunamadi: {args.data}\n-> run_nuplan_test.py'yi --debug ile calistirdin mi?")
        return
    print(f"{len(pkls)} senaryo verisi bulundu")
    for p in pkls:
        print(f"[{os.path.dirname(p)}]")
        render_one(p, args.fps, args.threshold, args.radius, args.dpi)


if __name__ == '__main__':
    main()
