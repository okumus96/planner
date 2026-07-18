"""relevance_data.pkl -> 2-PANEL scenario VIDEO (mp4).

run_nuplan_test.py --debug writes a relevance_data.pkl into each scenario folder
(per-step GAT importance data + attention). From that data this script renders a video
with two panels per frame:

  LEFT  (map):   the map as-is + a 0-1 IMPORTANCE HEATMAP (every element coloured by its
                 importance). Only elements with normalized importance > threshold get their
                 id printed on the map; everything else is still drawn as heatmap, just unlabelled.
  RIGHT (graph): ONLY nodes/edges above threshold. An edge is the normalized attention (0-1)
                 between two nodes; edges below the edge-threshold are NOT drawn, and a node
                 with no surviving edge is NOT drawn. Node id is printed inside the marker and
                 matches the left-panel ids one-to-one.

No ffmpeg required (mp4 via OpenCV).

Usage:
  python render_relevance_video.py --data testing_log/<exp>/.../debug_plots/scenario_01
  python render_relevance_video.py --data testing_log/<exp>/.../debug_plots   # all scenarios
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

CMAP = 'YlOrRd'   # 0 -> light yellow, 1 -> dark red (dark = important)
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
    """LEFT panel: map + 0-1 importance heatmap; only elements > threshold get an id label."""
    valid = rec['valid']; nt = rec['node_types']; pose = rec['node_pose']; sl = rec['slices']
    norm = _norm_importance(rec)
    cmo = plt.get_cmap(CMAP)

    # 1) First draw the whole map faint grey (map shown "as-is")
    for arr_key in ('lanes_xy', 'route_xy', 'cw_xy'):
        for poly in rec[arr_key]:
            m = np.abs(poly).sum(-1) > 1e-6
            if m.any():
                ax.plot(poly[m, 0], poly[m, 1], '-', color='0.85', linewidth=0.6, zorder=0)

    # 2) Heatmap on top: colour each element polyline by its importance
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

    # 3) Agents
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

    # 4) Only elements > threshold get an id label (ego always labelled for reference)
    ax.annotate('0', (pose[a0, 0], pose[a0, 1]), fontsize=8, fontweight='bold', color='blue',
                zorder=30, ha='center', va='center')
    for v in np.where(valid & (norm > threshold))[0]:
        if nt[v] == NODE_TYPE_EGO:
            continue
        ax.annotate(str(int(v)), (pose[v, 0], pose[v, 1]), fontsize=8, fontweight='bold',
                    color='black', zorder=30, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='red', alpha=0.9, lw=0.7))

    sm = cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(0, 1)); sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label='importance (0-1, dark = HIGH)')

    ax.set_xlim(-radius, radius); ax.set_ylim(-radius, radius)
    ax.set_aspect('equal'); ax.set_xlabel('x (m, ego frame)'); ax.set_ylabel('y (m, ego frame)')
    ax.set_title(f'Map importance heatmap (labelled id = importance > {threshold:.2f})')


def draw_graph(ax, rec, threshold, edge_threshold):
    """RIGHT panel: only nodes/edges above threshold.

    Candidate nodes = ego + (importance > threshold), i.e. the same set whose ids are
    labelled on the left. Among those we compute the symmetric, normalized attention; edges
    below edge_threshold are dropped, and any node with no surviving edge is dropped too.
    """
    valid = rec['valid']; nt = rec['node_types']; sl = rec['slices']
    norm = _norm_importance(rec)
    A = rec.get('attention')

    ego = [int(v) for v in range(*sl['agent']) if valid[v] and nt[v] == NODE_TYPE_EGO]
    passed = [int(v) for v in np.where(valid & (norm > threshold))[0] if nt[v] != NODE_TYPE_EGO]
    candidates = ego[:1] + passed

    # Symmetric, normalized attention among candidates; keep only edges >= edge_threshold
    edges = []
    if A is not None and len(candidates) >= 2:
        A = A.astype('float32')
        S = np.maximum(A, A.T); np.fill_diagonal(S, 0.0)
        sub = S[np.ix_(candidates, candidates)]
        smax = max(sub.max(), 1e-6)
        for a in range(len(candidates)):
            for b in range(a + 1, len(candidates)):
                i, j = candidates[a], candidates[b]
                w = float(S[i, j] / smax)
                if w >= edge_threshold:
                    edges.append((i, j, w))

    # A node is shown only if it is incident to at least one surviving edge
    connected = {n for e in edges for n in (e[0], e[1])}
    shown = [n for n in candidates if n in connected]

    # Circular layout: ego in the centre (if shown), the rest around it
    layout = {}
    ring = [n for n in shown if nt[n] != NODE_TYPE_EGO]
    for n in shown:
        if nt[n] == NODE_TYPE_EGO:
            layout[n] = (0.0, 0.0)
    for k, n in enumerate(ring):
        ang = np.pi / 2 - 2 * np.pi * k / max(len(ring), 1)
        layout[n] = (float(np.cos(ang)), float(np.sin(ang)))

    # Edges: thickness + number both from the same normalized weight w
    for i, j, w in edges:
        (x1, y1), (x2, y2) = layout[i], layout[j]
        ax.plot([x1, x2], [y1, y2], '-', color='steelblue',
                linewidth=0.8 + 5.0 * w, alpha=0.3 + 0.6 * w, zorder=2)
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.annotate(f"{w:.2f}", (mx, my), fontsize=8, color='navy', zorder=8,
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='steelblue', alpha=0.85, lw=0.5))

    # Nodes: marker = type, colour = importance, id inside
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
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markeredgecolor='k', markersize=10, label='vehicle'),
        Line2D([0], [0], marker='P', color='w', markerfacecolor='orange', markeredgecolor='k', markersize=10, label='pedestrian'),
        Line2D([0], [0], marker='X', color='w', markerfacecolor='orange', markeredgecolor='k', markersize=10, label='bicycle'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='orange', markeredgecolor='k', markersize=10, label='lane'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='orange', markeredgecolor='k', markersize=9, label='crosswalk'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='orange', markeredgecolor='k', markersize=10, label='route'),
        Line2D([0], [0], color='steelblue', lw=4, label='edge = attention (0-1)'),
    ]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=8, framealpha=0.9)
    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4); ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(f'Relevance graph: importance > {threshold:.2f} & edge > {edge_threshold:.2f}')


def _find_pkls(path):
    if path.endswith('.pkl'):
        return [path]
    return sorted(glob.glob(os.path.join(path, '**', 'relevance_data.pkl'), recursive=True))


def render_one(pkl_path, fps, threshold, edge_threshold, radius, dpi):
    with open(pkl_path, 'rb') as f:
        blob = pickle.load(f)
    records = blob['records']
    thr = threshold if threshold is not None else blob.get('threshold', 0.65)
    edge_thr = edge_threshold if edge_threshold is not None else thr
    if not records:
        print(f"  [skipped] no records: {pkl_path}")
        return
    if 'attention' not in records[0]:
        print("  [warning] records have no 'attention' -> right graph cannot draw edges. "
              "A new --debug run is needed (attention is saved now).")

    out = os.path.join(os.path.dirname(pkl_path), 'relevance.mp4')
    writer = None
    for rec in records:
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(20, 10), dpi=dpi)
        draw_bev_heatmap(axL, rec, thr, radius)
        draw_graph(axR, rec, thr, edge_thr)
        it = rec.get('iteration', '?')
        sel = ''
        if rec.get('lat_idx') is not None:
            v = (rec.get('lon_idx') or 0) / 11.0 * 15.0
            sel = f'  |  selected lat={rec["lat_idx"]} v={v:.1f} m/s'
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
        print(f"  written: {out}  ({len(records)} frames, {fps} fps, threshold={thr}, edge={edge_thr})")


def main():
    ap = argparse.ArgumentParser(description='relevance_data.pkl -> 2-panel scenario video')
    ap.add_argument('--data', required=True, help='relevance_data.pkl, a scenario folder, or a parent folder')
    ap.add_argument('--fps', type=float, default=8.0)
    ap.add_argument('--threshold', type=float, default=None, help='normalized importance threshold (default: value inside the pkl)')
    ap.add_argument('--edge-threshold', type=float, default=None, help='normalized attention edge threshold (default: same as --threshold)')
    ap.add_argument('--radius', type=float, default=60.0, help='map panel radius [m]')
    ap.add_argument('--dpi', type=int, default=100)
    args = ap.parse_args()

    pkls = _find_pkls(args.data)
    if not pkls:
        print(f"relevance_data.pkl not found: {args.data}\n-> did you run run_nuplan_test.py with --debug?")
        return
    print(f"{len(pkls)} scenario data file(s) found")
    for p in pkls:
        print(f"[{os.path.dirname(p)}]")
        render_one(p, args.fps, args.threshold, args.edge_threshold, args.radius, args.dpi)


if __name__ == '__main__':
    main()
