"""Yayin kalitesinde BEV figuru — KG run_frame stilinde (makale icin).

Stil kaynagi: nuPlan_Predicates_KG notebook cell 8-11 (_MAP_LAYER_STYLE + "edge legend'i
plotun ALTINA yazdir, asla icine degil" felsefesi) + eval_channels R2 paleti (kullanici onayli).

Duzen kararlari (2026-08-18, kullanici ile):
  - Serit BANTLARI (KG LANE #dfe8f3 / kenar #8aa4c0), crosswalk #fff4b8, "proper lane" gorunumu.
  - Elemanlarin ustune HAM SAYI YAZILMAZ — heatmap yeter; kanal dokumu panel ALTINDA tablo satiri.
  - Harita iliskileri (ego->map kanallari) RENKLE gosterilir: inLane yesil, adjacent mor,
    successor mavi, route pembe (R2 paleti); yogunluk = M_cas_map (alpha).
  - Eski viz_causal tarzi okunur ortak legend altta.
  - Ego-merkezli, burun YUKARI; olcek cubugu; karar (DOD) karti sol ustte.

Kullanim:
  python viz_paper.py --causal_path <ckpt> [model bayraklari] --auto 6 --out viz_out/paper
  python viz_paper.py ... --tokens us-ma-boston_xxx,sg-one-north_yyy
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, RegularPolygon, Patch
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.transforms import Affine2D
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
import numpy as np
import torch
from torch.utils.data import DataLoader

from GameFormer.predictor import GameFormer
from GameFormer.causal_graph import CausalPlanner
from GameFormer.train_utils import DrivingData
from GameFormer.decision_labels import LON_CLASSES, LAT_CLASSES, LON_MERGED_CLASSES
from train_planner import read_batch, extract_neighbor_top1_futures, freeze_gameformer

# ---- KG _MAP_LAYER_STYLE (notebook cell 8) ----
LANE_FACE, LANE_EDGE = "#dfe8f3", "#8aa4c0"
CROSS_FACE, CROSS_EDGE = "#fff4b8", "#b59b31"
C_PLAN = "#1565C0"
C_EGO = "#212121"
C_FADED, C_FADED_EDGE = "#ececec", "#cccccc"
CMAP_M = LinearSegmentedColormap.from_list(
    "mcas_paper", mpl.colormaps["Reds"](np.linspace(0.18, 0.95, 256)))

# ---- harita kanal renkleri (eval_channels R2 paleti; channels.py MCH_* sirasi) ----
MAP_CH_COLOR = {0: "#66bb6a",   # inLane        (yesil)
                1: "#ce93d8",   # adjacent_left (mor)
                2: "#ce93d8",   # adjacent_right
                3: "#64b5f6",   # successor     (mavi)
                4: "#c9a55a",   # inIntersection (rezerve)
                5: "#f06292",   # ego_route_corridor (pembe)
                6: "#b93b3b",   # traffic_control
                7: "#9e9e9e"}   # near (fallback, gri)
MAP_CH_PRIORITY = [0, 1, 2, 3, 6, 5, 4, 7]   # dominant kanal secimi (near en son)
CH_NAMES = ["ahead", "behind", "adj-L", "adj-R", "collide", "intersect", "near",
            "follows", "merges", "overtakes", "VRU"]
LON_PRETTY = {'remain_stopped': 'remain stopped', 'stop_quickly': 'hard stop',
              'stop_gently': 'gentle stop', 'slow_quickly': 'slow down hard',
              'slow_gently': 'slow down gently', 'accel_quickly': 'accelerate hard',
              'accel_gently': 'accelerate gently', 'maintain': 'maintain speed',
              'reverse': 'reverse', 'stop': 'stop', 'slow': 'slow down', 'accel': 'accelerate'}
LAT_PRETTY = {'turn_left': 'turn left', 'turn_right': 'turn right',
              'lane_change_left': 'lane change left', 'lane_change_right': 'lane change right',
              'inlane_left': 'nudge left', 'inlane_right': 'nudge right', 'no_lateral': 'keep lane'}


def _rot(xy):
    """Ego-frame (x ileri) -> figur (burun YUKARI): (x,y) -> (-y, x)."""
    out = np.empty_like(xy)
    out[..., 0] = -xy[..., 1]
    out[..., 1] = xy[..., 0]
    return out


def _box(ax, x, y, heading, L, W, face, edge, lw, z, alpha=1.0):
    L = L if L > 0.5 else 4.5
    W = W if W > 0.3 else 2.0
    r = Rectangle((-L / 2, -W / 2), L, W, facecolor=face, edgecolor=edge,
                  lw=lw, alpha=alpha, zorder=z, joinstyle="round")
    r.set_transform(Affine2D().rotate(heading + np.pi / 2).translate(x, y) + ax.transData)
    ax.add_patch(r)


def _agent(ax, x, y, heading, L, W, atype, face, edge, lw, z, alpha=1.0):
    if atype == 1:
        ax.add_patch(Circle((x, y), 1.2, facecolor=face, edgecolor=edge, lw=lw, zorder=z, alpha=alpha))
    elif atype == 2:
        ax.add_patch(RegularPolygon((x, y), numVertices=3, radius=1.5, orientation=heading,
                                    facecolor=face, edgecolor=edge, lw=lw, zorder=z, alpha=alpha))
    else:
        _box(ax, x, y, heading, L, W, face, edge, lw, z, alpha)


def _fading_line(ax, pts, color, lw, z, a0=0.9, a1=0.15):
    if len(pts) < 2:
        return
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    alphas = np.linspace(a0, a1, len(segs))
    lc = LineCollection(segs, colors=[mpl.colors.to_rgba(color, a) for a in alphas],
                        linewidths=lw, zorder=z, capstyle="round")
    ax.add_collection(lc)


def _dominant_map_channel(mch_row, mty_row):
    """Yanan kanallar icinden dominanti sec: typed agirlik varsa argmax, yoksa oncelik sirasi."""
    fired = [c for c in range(len(mch_row)) if mch_row[c]]
    if not fired:
        return None
    if mty_row is not None and float(mty_row[fired].sum()) > 1e-6:
        return int(max(fired, key=lambda c: float(mty_row[c])))
    for c in MAP_CH_PRIORITY:
        if c in fired:
            return c
    return fired[0]


def render_scene(ax, s, topk=2, reach=35.0):
    """KG run_frame stili tek sahne. Kanal dokumunu STRING olarak dondurur (panel alti satir)."""
    ax.set_facecolor("white")
    ylo, yhi = -reach * 0.6, reach * 1.25

    # --- 1. serit bantlari (KG LANE stili) — taban katman ---
    L, C, _ = s["map_counts"]
    Mmap = s.get("M_cas_map")
    mch = s.get("mch_active")
    mty = s.get("M_cas_map_typed")
    mmax = max(float(Mmap.max()), 1e-6) if Mmap is not None else 1.0
    for i, poly in enumerate(s["map_polys"]):
        m = np.abs(poly).sum(-1) > 1e-3
        if m.sum() < 2:
            continue
        p = _rot(poly[m])
        if i < L:            # serit: bant + ince kenar
            ax.plot(p[:, 0], p[:, 1], color=LANE_FACE, lw=7.5, alpha=0.85,
                    zorder=0.4, solid_capstyle="round")
            ax.plot(p[:, 0], p[:, 1], color=LANE_EDGE, lw=0.5, alpha=0.55, zorder=0.5)
        elif i < L + C:      # crosswalk
            ax.plot(p[:, 0], p[:, 1], color=CROSS_FACE, lw=6.0, alpha=0.9,
                    zorder=0.6, solid_capstyle="round")
            ax.plot(p[:, 0], p[:, 1], color=CROSS_EDGE, lw=0.5, alpha=0.6, zorder=0.65)
        # route tokenlari taban katmanda cizilmez -- iliski boyamasi (asagida) gosterir

    # --- 2. harita ILISKI boyamasi: yanan ego->map kanali RENK, M_cas_map YOGUNLUK ---
    if mch is not None:
        for i, poly in enumerate(s["map_polys"]):
            ch = _dominant_map_channel(mch[i], mty[i] if mty is not None else None)
            if ch is None:
                continue
            m = np.abs(poly).sum(-1) > 1e-3
            if m.sum() < 2:
                continue
            p = _rot(poly[m])
            w = float(Mmap[i]) / mmax if Mmap is not None else 0.5
            ax.plot(p[:, 0], p[:, 1], color=MAP_CH_COLOR[ch], lw=3.4,
                    alpha=0.28 + 0.62 * w, zorder=0.9, solid_capstyle="round")

    # --- 3. ajanlar: heatmap dolgu, SAYI YOK ---
    M = s["M_cas"]
    valid = s["valid"]
    gated = s["gated_valid"] if s.get("gated_valid") is not None else valid
    order = np.argsort(-M)
    causal_idx = [j for j in order if valid[j] and gated[j] and M[j] > 0.02][:max(topk, 1)]
    m_max = max(float(M[valid].max()), 1e-6) if valid.any() else 1.0
    letters = "ABCDE"
    for j in range(s["N"]):
        if not valid[j]:
            continue
        x, y = _rot(s["pos"][j, :2][None])[0]
        hd = s["pos"][j, 2]
        if not (-reach - 6 < x < reach + 6 and ylo - 6 < y < yhi + 6):
            continue
        if j in causal_idx:
            face = CMAP_M(float(M[j]) / m_max)
            _agent(ax, x, y, hd, s["dims"][j, 0], s["dims"][j, 1], int(s["types"][j]),
                   face, "black", 1.3, z=4)
            fut = s["fut"][j]
            fm = np.abs(fut).sum(-1) > 1e-3
            if fm.sum() > 1:
                _fading_line(ax, _rot(fut[fm]), mpl.colors.to_hex(face), 1.9, z=3)
            ax.text(x, y, letters[causal_idx.index(j)], color="white", fontsize=6.4,
                    ha="center", va="center", fontweight="bold", zorder=7)
        elif gated[j]:       # kanal yanmis ama dusuk agirlik
            _agent(ax, x, y, hd, s["dims"][j, 0], s["dims"][j, 1], int(s["types"][j]),
                   CMAP_M(float(M[j]) / m_max) if M[j] > 0.005 else "white",
                   "#8a8a8a", 0.8, z=2)
        else:                # gate-disi: yapisal olarak softmax'a giremez
            _agent(ax, x, y, hd, s["dims"][j, 0], s["dims"][j, 1], int(s["types"][j]),
                   C_FADED, C_FADED_EDGE, 0.7, z=1.5)

    # --- 4. ego + plan ---
    _box(ax, 0, 0, 0.0, 4.8, 2.0, C_EGO, "black", 1.2, z=5)
    plan = _rot(s["ego_plan"][:, :2])
    ax.plot(plan[:, 0], plan[:, 1], color=C_PLAN, lw=2.4, zorder=6, solid_capstyle="round")

    # --- 5. karar karti (sol ust) ---
    if s.get("dod"):
        lon_name, lon_p, lat_name, lat_p = s["dod"]
        txt = (f"$\\bf{{decision}}$  {LON_PRETTY.get(lon_name, lon_name)} ({lon_p:.2f})"
               f" + {LAT_PRETTY.get(lat_name, lat_name)} ({lat_p:.2f})")
        ax.text(0.03, 0.978, txt, transform=ax.transAxes, fontsize=7.6, ha="left", va="top",
                zorder=11, bbox=dict(boxstyle="round,pad=0.38", facecolor="white",
                                     edgecolor="#bbbbbb", lw=0.8, alpha=0.95))

    # --- 6. olcek cubugu (sag alt; karar kartiyla cakismaz) ---
    ax.plot([reach - 12, reach - 2], [ylo + 2.6, ylo + 2.6], color="#555555", lw=1.5, zorder=11)
    ax.text(reach - 7, ylo + 3.8, "10 m", fontsize=6.4, ha="center", color="#555555", zorder=11)

    ax.set_xlim(-reach, reach)
    ax.set_ylim(ylo, yhi)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#999999"); sp.set_linewidth(0.8)

    # --- panel alti kanal dokumu (KG: "edge legend plotun ALTINA") ---
    rows = []
    for rank, j in enumerate(causal_idx[:topk]):
        w2 = lambda v: f"{v:.2f}".lstrip("0")
        parts = []
        if s.get("M_typed") is not None and s.get("ch_active") is not None:
            fired = [c for c in range(s["ch_active"].shape[-1]) if s["ch_active"][j, c]]
            fired.sort(key=lambda c: -float(s["M_typed"][j, c]))
            parts = [f"{CH_NAMES[c]} {w2(float(s['M_typed'][j, c]))}" for c in fired[:3]
                     if s["M_typed"][j, c] >= 0.005]
        elif s.get("ch_active") is not None:
            parts = [CH_NAMES[c] for c in range(s["ch_active"].shape[-1]) if s["ch_active"][j, c]][:3]
        rows.append(f"$\\bf{{{letters[rank]}}}$ ego$\\rightarrow$agent: " + " · ".join(parts))
    return rows


def build_legend(fig, dod_meta):
    handles = [
        Patch(facecolor=C_EGO, edgecolor="black", label="Ego"),
        Line2D([0], [0], color=C_PLAN, lw=2.4, label="Ego plan"),
        Patch(facecolor=CMAP_M(0.85), edgecolor="black", lw=1.3, label="Causal agent (color = $M^{cas}$)"),
        Line2D([0], [0], color=CMAP_M(0.85), lw=1.9, label="Predicted future (fading = time)"),
        Patch(facecolor=C_FADED, edgecolor=C_FADED_EDGE, label="Gated-out agent (no predicate fired)"),
        Patch(facecolor=LANE_FACE, edgecolor=LANE_EDGE, label="Lane"),
        Patch(facecolor=CROSS_FACE, edgecolor=CROSS_EDGE, label="Crosswalk"),
        Line2D([0], [0], color=MAP_CH_COLOR[0], lw=3.4, label="inLane"),
        Line2D([0], [0], color=MAP_CH_COLOR[1], lw=3.4, label="adjacent lane"),
        Line2D([0], [0], color=MAP_CH_COLOR[3], lw=3.4, label="successor"),
        Line2D([0], [0], color=MAP_CH_COLOR[5], lw=3.4, label="route corridor"),
        Line2D([0], [0], marker="o", ls="None", markerfacecolor="0.75", markeredgecolor="0.4",
               markersize=9, label="Pedestrian"),
        Line2D([0], [0], marker="^", ls="None", markerfacecolor="0.75", markeredgecolor="0.4",
               markersize=10, label="Bicycle"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=7, fontsize=7.8,
               frameon=True, borderpad=0.6, handletextpad=0.5, columnspacing=1.1,
               bbox_to_anchor=(0.5, -0.02))


@torch.no_grad()
def collect(gameformer, causal, loader, num_neighbors, device, want_tokens, auto_n):
    scenes, Na = [], num_neighbors + 1
    for batch, files in loader:
        inputs, ego_future, _, ref_path = read_batch(batch, device)
        enc = gameformer.encoder(inputs)
        top1, nbr_states, _ = extract_neighbor_top1_futures(gameformer, enc, num_neighbors)
        out = causal(enc, inputs, num_agents=Na, neighbor_futures=top1, neighbor_states=nbr_states)
        actors = enc["actors"][:, :, -1]
        traj = out["traj"][:, 0, :, :, :2]
        best = out["score"][:, 0].argmax(-1)
        dims = inputs["neighbor_agents_past"][:, :, -1, 6:8]
        types = inputs["neighbor_agents_past"][:, :, -1, 8:11].argmax(-1)
        map_src = [inputs["map_lanes"][:, :, :, :2], inputs["map_crosswalks"][:, :, :, :2],
                   inputs["route_lanes"][:, :, :, :2]]
        M_typed = out.get("M_cas_typed")
        M_map_typed = out.get("M_cas_map_typed")
        dods = []
        if out.get("psi_lon_cas") is not None:
            lon_names = (LON_MERGED_CLASSES if out["psi_lon_cas"].shape[-1] == len(LON_MERGED_CLASSES)
                         else LON_CLASSES)
            p1 = torch.softmax(out["psi_lon_cas"], -1); p2 = torch.softmax(out["psi_lat_cas"], -1)
            i1, i2 = p1.argmax(-1), p2.argmax(-1)
            for b in range(i1.shape[0]):
                dods.append((lon_names[int(i1[b])], float(p1[b, i1[b]]),
                             LAT_CLASSES[int(i2[b])], float(p2[b, i2[b]])))
        else:
            dods = [None] * actors.shape[0]

        for b in range(actors.shape[0]):
            token = os.path.basename(files[b]).replace(".npz", "")
            v = out["nbr_valid"][b].cpu().numpy()
            mca = out["M_cas"][b].cpu().numpy()
            gv = out["gated_valid"][b].cpu().numpy() if out.get("gated_valid") is not None else v
            n_gated = int((v & gv).sum())
            interesting = (n_gated >= 2 and 0.25 <= float(mca[v].max() if v.any() else 0) <= 0.98)
            if want_tokens:
                if token not in want_tokens:
                    continue
            elif not interesting:
                continue
            scenes.append({
                "token": token, "N": num_neighbors,
                "M_cas": mca, "valid": v, "gated_valid": gv,
                "M_typed": (M_typed[b].cpu().numpy() if M_typed is not None else None),
                "ch_active": (out["ch_active"][b].cpu().numpy()
                              if out.get("ch_active") is not None else None),
                "M_cas_map": (out["M_cas_map"][b].cpu().numpy()
                              if out.get("M_cas_map") is not None else None),
                "mch_active": (out["mch_active"][b].cpu().numpy()
                               if out.get("mch_active") is not None else None),
                "M_cas_map_typed": (M_map_typed[b].cpu().numpy()
                                    if M_map_typed is not None else None),
                "pos": actors[b, 1:Na, :3].cpu().numpy(),
                "dims": dims[b].cpu().numpy(), "types": types[b].cpu().numpy(),
                "ego_plan": traj[b, best[b]].cpu().numpy(),
                "fut": top1[b].cpu().numpy(),
                "map_polys": [arr[b, e].cpu().numpy() for arr in map_src
                              for e in range(arr.shape[1])],
                "map_counts": (inputs["map_lanes"].shape[1], inputs["map_crosswalks"].shape[1],
                               inputs["route_lanes"].shape[1]),
                "dod": dods[b],
            })
            if want_tokens is None and len(scenes) >= max(auto_n * 8, 40):
                return scenes
        if want_tokens and len(scenes) >= len(want_tokens):
            return scenes
    return scenes


class _NamedDataset(DrivingData):
    def __getitem__(self, idx):
        return super().__getitem__(idx), self.data_list[idx]


def _collate(batch):
    items = [b[0] for b in batch]
    files = [b[1] for b in batch]
    return torch.utils.data.default_collate(items), files


def main(args):
    dev = args.device
    gameformer = GameFormer(encoder_layers=3, decoder_levels=2, neighbors=args.num_neighbors)
    gameformer.load_state_dict(torch.load(args.pretrained_path, map_location=dev))
    gameformer = gameformer.to(dev); freeze_gameformer(gameformer)
    causal = CausalPlanner(layers=args.graph_layers, modes=args.modes, nbr_enrich=args.nbr_enrich,
                           gate=args.gate, ego_residual=args.ego_residual,
                           gate_channels=args.gate_channels, typed_kv=args.typed_kv,
                           channel_evidence=args.channel_evidence, gate_trust=args.gate_trust, rel_bottleneck=args.rel_bottleneck,
                           dod_meta=args.dod_meta, num_lon=(6 if args.lon_merge else 9)).to(dev)
    missing, unexpected = causal.load_state_dict(torch.load(args.causal_path, map_location=dev),
                                                 strict=False)
    if missing or unexpected:
        print(f"[load] missing={list(missing)}  unexpected={list(unexpected)}")
    causal.eval()

    ds = _NamedDataset(args.valid_set + "/*.npz", args.num_neighbors)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4,
                        collate_fn=_collate)
    want = set(args.tokens.split(",")) if args.tokens else None
    scenes = collect(gameformer, causal, loader, args.num_neighbors, dev, want, args.auto)
    print(f"collected {len(scenes)} candidate scenes")

    if want is None:
        def _pick_diverse(cands, n):
            pref = sorted(cands, key=lambda s: (
                0 if (s["dod"] and s["dod"][2] != "no_lateral") else 1,
                -float(s["M_cas"][s["valid"]].max() if s["valid"].any() else 0)))
            seen, out = set(), []
            for s in pref:
                k = (s["dod"][2], s["dod"][0]) if s["dod"] else ("?", "?")
                if k in seen:
                    continue
                seen.add(k)
                out.append(s)
                if len(out) >= n:
                    return out
            for s in pref:
                if s not in out:
                    out.append(s)
                    if len(out) >= n:
                        break
            return out
        scenes = _pick_diverse(scenes, args.auto)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n = min(len(scenes), args.auto if not want else len(scenes))

    def _caption(ax, rows):
        # KG felsefesi: dokum plotun ALTINA — sola yasli, ajan basina bir satir (tasma yok)
        ax.text(0.0, -0.035, "\n".join(rows), transform=ax.transAxes, ha="left", va="top",
                fontsize=6.9, linespacing=1.45)

    # tekil sahneler
    for i in range(n):
        fig, ax = plt.subplots(figsize=(3.6, 3.8))
        rows = render_scene(ax, scenes[i], topk=args.topk)
        _caption(ax, rows)
        build_legend(fig, args.dod_meta)
        fig.tight_layout(rect=(0, 0.12, 1, 1))
        p = f"{args.out}_{scenes[i]['token']}.png"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("saved:", p)

    # 1x3 makale seridi
    if n >= 3:
        fig, axes = plt.subplots(1, 3, figsize=(10.8, 4.6))
        for ax, s in zip(axes, scenes[:3]):
            rows = render_scene(ax, s, topk=args.topk)
            _caption(ax, rows)
        build_legend(fig, args.dod_meta)
        fig.tight_layout(rect=(0, 0.12, 1, 1))
        for ext in ("png", "pdf"):
            fig.savefig(f"{args.out}_strip.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"saved: {args.out}_strip.png/.pdf")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Publication-quality BEV figure (KG run_frame style)")
    p.add_argument("--pretrained_path", required=True)
    p.add_argument("--causal_path", required=True)
    p.add_argument("--valid_set", required=True)
    p.add_argument("--tokens", type=str, default=None)
    p.add_argument("--auto", type=int, default=6)
    p.add_argument("--topk", type=int, default=2)
    p.add_argument("--num_neighbors", type=int, default=10)
    p.add_argument("--graph_layers", type=int, default=1)
    p.add_argument("--nbr_enrich", type=int, default=0)
    p.add_argument("--gate_channels", type=int, default=0)
    p.add_argument("--typed_kv", type=int, default=0)
    p.add_argument("--channel_evidence", type=int, default=0)
    p.add_argument("--gate_trust", type=str, default="all", choices=["all", "reliable"])
    p.add_argument("--rel_bottleneck", type=int, default=0)
    p.add_argument("--dod_meta", type=int, default=0)
    p.add_argument("--lon_merge", type=int, default=0)
    p.add_argument("--ego_residual", type=int, default=1)
    p.add_argument("--gate", type=str, default="softmax", choices=["softmax", "sigmoid"])
    p.add_argument("--modes", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--out", type=str, default="viz_out/paper")
    p.add_argument("--device", type=str, default="cuda")
    main(p.parse_args())
