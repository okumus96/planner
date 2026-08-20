"""
Causal graph BEV visualization. For each scene the ego + neighbor agents are drawn and colored by
M_cas (agents), and the map polylines (lanes/crosswalks/routes) are colored by M_cas_map. The palette
is SEQUENTIAL: light = non-causal ... dark red = causal.

Selection (which agents / map elements count as "causal") uses ONE rule applied IDENTICALLY to agents
and to map:
  --select topn      --topn N     : top-N by score for agents AND top-N for map (scale-independent).
  --select threshold --threshold T : score > T for both (note: map scores ~0.2 << agent scores ~0.5,
                                     because each relation has its OWN softmax over a different count,
                                     so use a low T or prefer topn).
Selected agents get a bold black outline + M_cas label + predicted-future line (blue). The selected map
elements get an M_cas_map label.

Example:
  python viz_causal.py --pretrained_path training_log/normal/model_epoch_19_valADE_1.6487.pth \
    --causal_path training_log/causal_map/causal_epoch_16_minADE_0.7396.pth \
    --valid_set /home/lt-hta-ai4/ssd1/nuplan/processed_data/validation \
    --select topn --topn 1 --scenario turn --out viz_out/causal_map.png
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, RegularPolygon, Patch
from matplotlib.lines import Line2D
from matplotlib.transforms import Affine2D
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
import numpy as np
import torch
from torch.utils.data import DataLoader

from GameFormer.predictor import GameFormer
from GameFormer.causal_graph import CausalPlanner
from GameFormer.train_utils import DrivingData
from train_planner import read_batch, extract_neighbor_top1_futures, freeze_gameformer

# Sequential palette: LIGHT (non-causal) -> DARK RED (causal). Truncated so the low end stays visible
# (red contrasts better than blue against the faint map polylines).
CMAP = LinearSegmentedColormap.from_list("mcas_seq", mpl.colormaps["Reds"](np.linspace(0.12, 1.0, 256)))
FUTURE_COLOR = "#1565C0"    # blue: causal agent's predicted future (contrast vs red map)
# agent type idx: neighbor_agents_past[..., 8:11] one-hot argmax -> 0=vehicle, 1=pedestrian, 2=bicycle


def select_mask(values, valid, mode, thr, topn):
    """Return a boolean 'selected' mask over elements, applying the SAME rule to agents and map.
    mode='topn' -> top-`topn` valid elements by value; mode='threshold' -> value > thr among valid;
    mode='none' -> hicbir eleman secilmez (sadece renklendirme, kutu/etiket/gelecek-cizgisi YOK)."""
    sel = np.zeros(values.shape, dtype=bool)
    if mode == "none":
        return sel
    idx = np.where(valid)[0]
    if len(idx) == 0:
        return sel
    if mode == "topn":
        order = idx[np.argsort(values[idx])[::-1]]
        sel[order[:max(topn, 0)]] = True
    else:  # threshold (absolute), same number for agents and map
        sel[idx[values[idx] > thr]] = True
    return sel


def _draw_car(ax, x, y, heading, length, width, facecolor, edgecolor, lw, alpha=1.0, z=5):
    length = length if length > 0.5 else 4.5
    width = width if width > 0.3 else 2.0
    rect = Rectangle((-length / 2, -width / 2), length, width,
                     facecolor=facecolor, edgecolor=edgecolor, lw=lw, alpha=alpha, zorder=z)
    rect.set_transform(Affine2D().rotate(heading).translate(x, y) + ax.transData)
    ax.add_patch(rect)


def _draw_agent(ax, x, y, heading, length, width, atype, facecolor, edgecolor, lw, z=5):
    """Shape = agent type (vehicle=rect, pedestrian=circle, bicycle=triangle); color = M_cas."""
    if atype == 1:      # pedestrian
        ax.add_patch(Circle((x, y), 0.9, facecolor=facecolor, edgecolor=edgecolor, lw=lw, zorder=z))
    elif atype == 2:    # bicycle
        ax.add_patch(RegularPolygon((x, y), numVertices=3, radius=1.3, orientation=heading - np.pi / 2,
                                    facecolor=facecolor, edgecolor=edgecolor, lw=lw, zorder=z))
    else:               # vehicle
        _draw_car(ax, x, y, heading, length, width, facecolor, edgecolor, lw, z=z)


def _plot_map_causal(ax, polys, mcas, valid):
    """Draw map polylines colored by M_cas_map (light->dark red, per-scene scaled).
    Map is COLORED ONLY (no selection / no labels) -- selection is agents-only."""
    vm = mcas[valid]
    vmax = float(vm.max()) if vm.size else 1.0
    vmax = max(vmax, 1e-6)
    order = np.argsort(mcas)                                 # low first -> causal drawn on top
    for e in order:
        if not valid[e]:
            continue
        poly = polys[e]
        msk = np.abs(poly).sum(-1) > 1e-3
        if msk.sum() < 2:
            continue
        r = float(mcas[e]) / vmax
        ax.plot(poly[msk, 0], poly[msk, 1], color=CMAP(r), lw=0.8 + 3.2 * r,
                zorder=1 + 2.0 * r, alpha=0.9, solid_capstyle="round")


def ego_maneuver(ego_future):
    """ego_future [80,3] (x,y,heading, ego-frame) -> maneuver dict (turn dir + start/stop)."""
    xy = ego_future[:, :2]
    hd = ego_future[:, 2]
    dheading = float(np.arctan2(np.sin(hd[-1] - hd[0]), np.cos(hd[-1] - hd[0])))
    v = np.linalg.norm(np.diff(xy, axis=0), axis=-1) / 0.1
    v0, vend = float(v[:5].mean()), float(v[-5:].mean())
    turn = "left" if dheading > 0.35 else ("right" if dheading < -0.35 else "straight")
    starting = (v0 < 1.5) and (vend > v0 + 2.0)
    stopping = (v0 > 3.0) and (vend < 1.0)
    return {"turn": turn, "dheading": dheading, "v0": v0, "vend": vend,
            "starting": starting, "stopping": stopping}


def scenario_match(man, scenario):
    t, st, sp = man["turn"], man["starting"], man["stopping"]
    return {
        "any": True, "straight": t == "straight", "left_turn": t == "left", "right_turn": t == "right",
        "turn": t != "straight", "starting": st, "stopping": sp,
        "starting_left_turn": st and t == "left", "starting_right_turn": st and t == "right",
        "stopping_straight": sp and t == "straight",
    }.get(scenario, True)


CH_SHORT = ["ahead", "behind", "adjL", "adjR", "collide", "intersect", "near",
            "follows", "merges", "overtakes", "vru"]   # CHANNEL_NAMES kisa etiketleri
# DOD manevra siniflari (train_planner._MANEUVER ile AYNI indeks sirasi)
MANEUVER_SHORT = ["stationary", "straight", "turn-L", "turn-R", "U-turn-L"]
# Harita arka plan paleti (eval_channels._draw / KG notebook stili ile ayni tonlar)
BG_LANE, BG_LANE_EDGE = "#e8e8e8", "#c4c4c4"
BG_CROSSWALK, BG_CROSSWALK_EDGE = "#fff4b8", "#b59b31"
BG_ROUTE = "#f06292"


def _plot_map_background(ax, s):
    """Tum harita geometrisini SOLUK arka plan bandi olarak ciz (serit gri, crosswalk sari,
    rota pembe) -- M_cas_map renklendirmesi ustte kalir. Gate'li modellerde harita elemanlarinin
    cogu M~0 aldigindan _plot_map_causal tek basina 'bos' gorunuyordu; baglam bu katmandan gelir."""
    ax.set_facecolor("#fafafa")
    L, C, _ = s["map_counts"]
    for i, poly in enumerate(s["map_polys"]):
        msk = np.abs(poly).sum(-1) > 1e-3
        if msk.sum() < 2:
            continue
        p = poly[msk]
        if i < L:            # serit: genis gri bant + ince kenar
            ax.plot(p[:, 0], p[:, 1], color=BG_LANE, lw=6.0, alpha=0.6,
                    zorder=0.3, solid_capstyle="round")
            ax.plot(p[:, 0], p[:, 1], color=BG_LANE_EDGE, lw=0.5, alpha=0.5, zorder=0.4)
        elif i < L + C:      # crosswalk: sari bant
            ax.plot(p[:, 0], p[:, 1], color=BG_CROSSWALK, lw=5.0, alpha=0.85,
                    zorder=0.5, solid_capstyle="round")
            ax.plot(p[:, 0], p[:, 1], color=BG_CROSSWALK_EDGE, lw=0.5, alpha=0.6, zorder=0.55)
        else:                # rota seridi: pembe (KG R2_ROUTE tonu)
            ax.plot(p[:, 0], p[:, 1], color=BG_ROUTE, lw=2.6, alpha=0.5,
                    zorder=0.6, solid_capstyle="round")


def plot_scene(ax, s, mode, thr, topn, branch="cas", show_channels=False):
    """s: dict of numpy arrays for one sample. branch='cas'|'cfd' -> only affects label text
    (mask/data used was already selected upstream in collect_scenes)."""
    tag_word = "causal" if branch == "cas" else "confound"
    _plot_map_background(ax, s)                                           # baglam: soluk bantlar
    _plot_map_causal(ax, s["map_polys"], s["map_mcas"], s["map_valid"])   # map: colored only

    # ego (idx 0) -> origin, black
    _draw_car(ax, s["ego"][0], s["ego"][1], s["ego"][2], 4.8, 2.0,
              facecolor="#222222", edgecolor="black", lw=1.0, z=6)
    ax.plot(s["ego_plan"][:, 0], s["ego_plan"][:, 1], "--", color="#2ca02c", lw=1.8, zorder=7)
    # DOD karari: head'in kullandigi b* (modelin TAHMINI, GT degil). dod_meta'da "lon + lat" cifti.
    # Panel kose etiketi (ego uzerine yazinca komsu ajan etiketleriyle cakisiyordu).
    if s.get("dod_text"):
        ax.text(0.02, 0.97, s["dod_text"], transform=ax.transAxes,
                color="#1b5e20", fontsize=7, ha="left", va="top", fontweight="bold", zorder=9,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.2))

    # L_conflict'in koridoru: lattice planner referans yolu (c_lat_candidates[0]).
    # KOYU mor = reach icinde kalan parca (cezanin gercekten kullandigi), ACIK mor = otesi.
    rp = s["ref_path"]
    rp = rp[np.abs(rp).sum(-1) > 1e-6]
    if len(rp) > 1:
        inside = np.linalg.norm(rp, axis=-1) <= s["reach"]
        ax.plot(rp[~inside, 0], rp[~inside, 1], ".", color="#9C7BB8", ms=1.2, zorder=3, alpha=0.7)
        ax.plot(rp[inside, 0], rp[inside, 1], "-", color="#6A1B9A", lw=2.2, zorder=4, alpha=0.9)

    agent_sel = select_mask(s["M_cas"][:s["N"]], s["valid"], mode, thr, topn)
    # HARITA ile AYNI normalizasyon: renk = m / (sahne-ici max M_cas), boyle harita ile
    # kiyaslanabilir olur (onceden ham/mutlak degerle boyaniyordu -> harita hep daha "kesin"
    # gorunuyordu, gercekte agent/map peak'leri farkli olcekte -- bkz peak/uniform oranlari).
    m_valid = s["M_cas"][s["valid"]]
    m_vmax = float(m_valid.max()) if m_valid.size else 1.0
    m_vmax = max(m_vmax, 1e-6)
    xs, ys = [s["ego"][0]], [s["ego"][1]]
    for j in range(s["N"]):
        if not s["valid"][j]:
            continue
        x, y, hd = s["pos"][j, 0], s["pos"][j, 1], s["pos"][j, 2]
        m = float(s["M_cas"][j])                 # HAM deger (etiket icin)
        r = m / m_vmax                            # NORMALIZE deger (renk icin)
        causal = bool(agent_sel[j])
        _draw_agent(ax, x, y, hd, s["dims"][j, 0], s["dims"][j, 1], int(s["types"][j]),
                    facecolor=CMAP(r), edgecolor=("black" if causal else "0.45"),
                    lw=(2.4 if causal else 0.8), z=(8 if causal else 5))
        if causal:
            ax.text(x, y + 2.2, f"{m:.2f}", color="black", fontsize=7, ha="center",
                    fontweight="bold", zorder=9)
            fut = s["fut"][j]                            # causal agent's predicted future (ego-frame, absolute)
            fm = np.abs(fut).sum(-1) > 1e-3
            if fm.sum() > 1:
                ax.plot(fut[fm, 0], fut[fm, 1], "-", color=FUTURE_COLOR, lw=1.6, alpha=0.95, zorder=6)
        if show_channels and s.get("ch_active") is not None:
            fired = [c for c in range(s["ch_active"].shape[-1]) if s["ch_active"][j, c]]
            if s.get("M_typed") is not None and fired:
                # typed model: kanal-basina M_cas agirligi (softmax (ajan,kanal) girisleri uzerinde;
                # ajanin per-agent degeri bunlarin toplami). Agirliga gore buyukten kucuge.
                fired.sort(key=lambda c: -float(s["M_typed"][j, c]))
                lbl = ",".join(f"{CH_SHORT[c]}:{float(s['M_typed'][j, c]):.2f}" for c in fired)
            else:
                lbl = ",".join(CH_SHORT[c] for c in fired)
            ax.text(x, y - 2.6, lbl if fired else "-",
                    color=("black" if causal else "0.35"), fontsize=5.5, ha="center",
                    fontweight=("bold" if causal else "normal"), zorder=9)
        xs.append(x); ys.append(y)

    r = 40
    ax.set_xlim(min(xs) - 10, max(xs) + 10)
    ax.set_ylim(min(ys) - 10, max(ys) + 10)
    ax.set_xlim(np.clip(ax.get_xlim(), -r, r))
    ax.set_ylim(np.clip(ax.get_ylim(), -r, r))
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    man = s.get("man", {})
    tag = man.get("turn", "?")
    if man.get("starting"):
        tag = "start+" + tag
    if man.get("stopping"):
        tag = "stop+" + tag
    ax.set_title(f"[{tag}]  {tag_word} agents: {int(agent_sel.sum())}  |  "
                 f"maxA={s['M_cas'][s['valid']].max():.2f}  maxMap={s['map_mcas'][s['map_valid']].max():.2f}",
                 fontsize=8)


@torch.no_grad()
def collect_scenes(gameformer, causal, loader, num_neighbors, num_scenes, device,
                   scenario, mode, thr, topn, branch="cas", ref_idx=0):
    """branch='cas' (varsayilan) -> M_cas/M_cas_map + f_cas'ten uretilen plan (ANA, deploy edilen).
    branch='cfd' -> M_cfd/M_cfd_map + f_cfd'den uretilen plan (traj_cfd/score_cfd, tani/ablasyon amacli,
    hicbir ajan silinmez)."""
    scenes = []
    Na = num_neighbors + 1
    mask_key = "M_cas" if branch == "cas" else "M_cfd"
    map_key = "M_cas_map" if branch == "cas" else "M_cfd_map"
    for batch in loader:
        inputs, ego_future, _, ref_path = read_batch(batch, device)
        enc = gameformer.encoder(inputs)
        top1_fut, nbr_states, _ = extract_neighbor_top1_futures(gameformer, enc, num_neighbors)
        out = causal(enc, inputs, num_agents=Na, neighbor_futures=top1_fut, neighbor_states=nbr_states,
                     also_cfd_plan=(branch == "cfd"))

        M_cas = out[mask_key]; valid = out["nbr_valid"]
        M_cas_map = out[map_key]; map_valid = out["map_valid"]
        # typed (ajan,kanal) maskesi [B,N,R] (typed_kv=0 ise None); dal ile tutarli anahtar
        M_typed = out.get("M_cas_typed" if branch == "cas" else "M_cfd_typed")
        # DOD karari: head'in query'sine giden b* (+softmax guveni). dod_meta'da factored (lon x lat).
        if out.get("psi_cas") is not None:
            psi_prob = torch.softmax(out["psi_cas"], dim=-1)             # [B,K]
            dod_idx = psi_prob.argmax(-1)                                # [B]
            dod_texts = [f"DOD: {MANEUVER_SHORT[int(dod_idx[b])]} ({float(psi_prob[b, dod_idx[b]]):.2f})"
                         for b in range(dod_idx.shape[0])]
        else:
            from GameFormer.decision_labels import LON_CLASSES, LAT_CLASSES, LON_MERGED_CLASSES
            lon_names = LON_MERGED_CLASSES if out["psi_lon_cas"].shape[-1] == len(LON_MERGED_CLASSES) else LON_CLASSES
            p1 = torch.softmax(out["psi_lon_cas"], dim=-1)
            p2 = torch.softmax(out["psi_lat_cas"], dim=-1)
            i1, i2 = p1.argmax(-1), p2.argmax(-1)
            dod_texts = [f"DOD: {lon_names[int(i1[b])]} ({float(p1[b, i1[b]]):.2f}) + "
                         f"{LAT_CLASSES[int(i2[b])]} ({float(p2[b, i2[b]]):.2f})"
                         for b in range(i1.shape[0])]
        actors = enc["actors"][:, :, -1]
        traj_key, score_key = ("traj_cfd", "score_cfd") if branch == "cfd" else ("traj", "score")
        traj = out[traj_key][:, 0, :, :, :2]
        best = out[score_key][:, 0].argmax(-1)
        if ref_idx < 0:
            # otomatik: ego'nun SERIDINDEN baslayan aday (2026-08-18; loader'in lateral sort'u
            # yuzunden sabit indeks yanlis seridi cizebiliyordu -- "koridor baska seritten
            # basliyor" gorunumunun sebebi buydu)
            from GameFormer.channels import select_ego_corridor
            _sel = select_ego_corridor(ref_path)
            ref0 = ref_path[torch.arange(ref_path.shape[0], device=ref_path.device), _sel][..., :2]
        else:
            ref0 = ref_path[:, ref_idx, :, :2]                   # [B,R,2] manuel aday secimi
        ego_spd = torch.norm(actors[:, 0, 3:5], dim=-1)     # [B] anlik ego hizi
        reach = (ego_spd * 0.1 * 80).clamp(min=10.0)        # L_conflict'teki reach ile AYNI formul
        dims = inputs["neighbor_agents_past"][:, :, -1, 6:8]
        types = inputs["neighbor_agents_past"][:, :, -1, 8:11].argmax(-1)

        # flat list of map polylines (order matches M_cas_map: lanes, crosswalks, routes)
        map_src = [inputs["map_lanes"][:, :, :, :2], inputs["map_crosswalks"][:, :, :, :2],
                   inputs["route_lanes"][:, :, :, :2]]

        B = M_cas.shape[0]
        for b in range(B):
            v = valid[b]
            if v.sum() < 2:
                continue
            man = ego_maneuver(ego_future[b].cpu().numpy())
            if scenario != "any" and not scenario_match(man, scenario):
                continue
            mca = M_cas[b].cpu().numpy(); vv = v.cpu().numpy()
            if scenario == "any" and mode != "none":
                # 'any' -> keep interactive scenes: at least one SELECTED agent
                # (mode='none' -> secim yok, bu filtre atlanir; TUM sahneler tutulur)
                if not select_mask(mca[:num_neighbors], vv, mode, thr, topn).any():
                    continue
            polys = [arr[b, e].cpu().numpy() for arr in map_src for e in range(arr.shape[1])]
            scenes.append({
                "man": man, "N": num_neighbors,
                "M_cas": mca, "valid": vv,
                # yanan predicate kanallari [N,R] (channels branch; kanal yoksa None)
                "ch_active": (out["ch_active"][b].cpu().numpy()
                              if out.get("ch_active") is not None else None),
                # typed (ajan,kanal) M agirliklari [N,R] (typed_kv=0 -> None; per-agent M = satir toplami)
                "M_typed": (M_typed[b].cpu().numpy() if M_typed is not None else None),
                # DOD: hazir-formatli etiket metni (5-sinif VEYA dod_meta lon+lat) -- modelin tahmini
                "dod_text": dod_texts[b],
                # arka plan stillemesi icin poly tip sayilari (lanes, crosswalks, routes)
                "map_counts": (inputs["map_lanes"].shape[1], inputs["map_crosswalks"].shape[1],
                               inputs["route_lanes"].shape[1]),
                "ego": actors[b, 0, :3].cpu().numpy(),
                "pos": actors[b, 1:Na, :3].cpu().numpy(),
                "dims": dims[b].cpu().numpy(), "types": types[b].cpu().numpy(),
                "ego_plan": traj[b, best[b]].cpu().numpy(),
                "ref_path": ref0[b].cpu().numpy(), "reach": float(reach[b]),
                "fut": top1_fut[b].cpu().numpy(),
                "map_polys": polys,
                "map_mcas": M_cas_map[b].cpu().numpy(),
                "map_valid": map_valid[b].cpu().numpy(),
            })
            if len(scenes) >= num_scenes:
                return scenes
    return scenes


def main(args):
    dev = args.device
    gameformer = GameFormer(encoder_layers=args.encoder_layers, decoder_levels=args.decoder_levels,
                            neighbors=args.num_neighbors)
    gameformer.load_state_dict(torch.load(args.pretrained_path, map_location=dev))
    gameformer = gameformer.to(dev); freeze_gameformer(gameformer)

    causal = CausalPlanner(layers=args.graph_layers, modes=args.modes, nbr_enrich=args.nbr_enrich, gate=args.gate, ego_residual=args.ego_residual, gate_channels=args.gate_channels, typed_kv=args.typed_kv, channel_evidence=args.channel_evidence, gate_trust=args.gate_trust, dod_meta=args.dod_meta, num_lon=(6 if args.lon_merge else 9)).to(dev)
    # strict=False: eski checkpoint'ler (ornegin Step 2 oncesi) cfd_recon agirliklarini icermez.
    # O modul viz yolunda HIC kullanilmaz, o yuzden eksik olmasi zararsiz -- ama ne eksikse yazdir,
    # sessizce gercek bir uyumsuzlugu yutmayalim.
    missing, unexpected = causal.load_state_dict(torch.load(args.causal_path, map_location=dev), strict=False)
    if missing or unexpected:
        print(f"[load] missing={list(missing)}  unexpected={list(unexpected)}")
    causal.eval()

    valid_set = DrivingData(args.valid_set + "/*.npz", args.num_neighbors)
    loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    scenes = collect_scenes(gameformer, causal, loader, args.num_neighbors, args.num_scenes, dev,
                            args.scenario, args.select, args.threshold, args.topn, branch=args.branch,
                            ref_idx=args.ref_idx)
    if args.select == "none":
        sel_desc = "yok (secim kapali, sadece normalize renklendirme)"
    elif args.select == "topn":
        sel_desc = f"topn={args.topn}"
    else:
        sel_desc = f"threshold={args.threshold}"
    print(f"collected {len(scenes)} scenes (scenario={args.scenario}, branch={args.branch}, "
          f"select={args.select} {sel_desc}).")

    n = len(scenes)
    cols = min(3, n) if n else 1
    rows = int(np.ceil(n / cols)) if n else 1
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 4.2), squeeze=False)
    for i in range(rows * cols):
        ax = axes[i // cols][i % cols]
        if i < n:
            plot_scene(ax, scenes[i], args.select, args.threshold, args.topn, branch=args.branch,
                       show_channels=bool(args.show_channels))
        else:
            ax.axis("off")
    sm = mpl.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(0, 1))
    mask_name = "M_cas" if args.branch == "cas" else "M_cfd"
    tag_word = "causal" if args.branch == "cas" else "confound"
    fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01,
                 label=f"{mask_name}   SAHNE-ICI GORELI (light = dusuk  ->  dark = o sahnedeki en yuksek); "
                       f"agent ve map AYNI olcekte, gercek/mutlak deger icin etiket sayisina bak")

    handles = [
        Patch(facecolor="#222222", edgecolor="black", label="Ego"),
        Line2D([0], [0], color="#2ca02c", lw=1.8, ls="--",
               label=f"Ego plan (pred, from f_{args.branch})"),
    ]
    if args.select != "none":
        sel_label = f"top-{args.topn}" if args.select == "topn" else f"M > {args.threshold:g}"
        handles += [
            Patch(facecolor=CMAP(0.9), edgecolor="black", lw=2.2, label=f"{tag_word.capitalize()} agent ({sel_label})"),
            Line2D([0], [0], color=FUTURE_COLOR, lw=1.6, label=f"{tag_word.capitalize()} agent future traj."),
        ]
    handles += [
        Line2D([0], [0], color=CMAP(0.95), lw=3.2, label=f"Map polyline (sahne-ici en yuksek {mask_name}_map)"),
        Line2D([0], [0], color=CMAP(0.25), lw=1.2, label="Map polyline (sahne-ici dusuk)"),
        Line2D([0], [0], color=BG_LANE, lw=6.0, label="Lane (arka plan)"),
        Line2D([0], [0], color=BG_CROSSWALK, lw=5.0, label="Crosswalk (arka plan)"),
        Line2D([0], [0], color=BG_ROUTE, lw=2.6, label="Route (arka plan)"),
        Patch(facecolor="0.7", edgecolor="0.4", label="Type: vehicle (rect)"),
        Line2D([0], [0], marker="o", ls="None", markerfacecolor="0.7", markeredgecolor="0.4",
               markersize=11, label="Type: pedestrian (circle)"),
        Line2D([0], [0], marker="^", ls="None", markerfacecolor="0.7", markeredgecolor="0.4",
               markersize=12, label="Type: bicycle (triangle)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8.5,
               frameon=True, bbox_to_anchor=(0.5, -0.04))

    fig.suptitle(f"{tag_word.capitalize()} graph — agents ({mask_name}) + map ({mask_name}_map)   "
                 f"(scenario={args.scenario}, select={args.select} {sel_desc})  |  "
                 f"{os.path.basename(args.causal_path)}", fontsize=11)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="BEV visualization of the causal graph (agents + map)")
    p.add_argument("--pretrained_path", required=True)
    p.add_argument("--causal_path", required=True)
    p.add_argument("--valid_set", required=True)
    p.add_argument("--branch", type=str, default="cas", choices=["cas", "cfd"],
                   help="'cas' (varsayilan) = M_cas/M_cas_map + ANA plan (f_cas'ten, deploy edilen). "
                        "'cfd' = M_cfd/M_cfd_map + f_cfd'den uretilen plan (tani/ablasyon, hicbir ajan silinmez).")
    p.add_argument("--select", type=str, default="topn", choices=["topn", "threshold", "none"],
                   help="selection rule applied IDENTICALLY to agents and map. "
                        "'none' = hicbir eleman secilmez/kutulanmaz, sadece normalize renklendirme.")
    p.add_argument("--topn", type=int, default=1, help="for --select topn: top-N per relation")
    p.add_argument("--threshold", type=float, default=0.5, help="for --select threshold: M > T (same for both)")
    p.add_argument("--scenario", type=str, default="any",
                   choices=["any", "straight", "left_turn", "right_turn", "turn", "starting",
                            "stopping", "starting_left_turn", "starting_right_turn", "stopping_straight"],
                   help="scene filter by ego maneuver (derived from the GT future)")
    p.add_argument("--num_scenes", type=int, default=9)
    p.add_argument("--num_neighbors", type=int, default=10)
    p.add_argument("--encoder_layers", type=int, default=3)
    p.add_argument("--decoder_levels", type=int, default=2)
    p.add_argument("--graph_layers", type=int, default=1)
    p.add_argument("--nbr_enrich", type=int, default=0)
    p.add_argument("--ref_idx", type=int, default=-1,
                   help="-1 (varsayilan) = ego'nun seridinden baslayan aday OTOMATIK secilir "
                        "(select_ego_corridor); >=0 = manuel aday indeksi")
    p.add_argument("--show_channels", type=int, default=0,
                   help="1 = her ajanin altina yanan predicate kanallarini yaz (ahead/near/vru...)")
    p.add_argument("--gate_channels", type=int, default=0)
    p.add_argument("--typed_kv", type=int, default=0)
    p.add_argument("--channel_evidence", type=int, default=0)
    p.add_argument("--gate_trust", type=str, default="all", choices=["all", "reliable"])
    p.add_argument("--dod_meta", type=int, default=0,
                   help="ckpt hangi degerle egitildiyse o: factored (lon x lat) meta-aksiyon DOD'u")
    p.add_argument("--lon_merge", type=int, default=0,
                   help="ckpt lon_merge=1 ile egitildiyse o (6 lon sinifi)")
    p.add_argument("--ego_residual", type=int, default=1,
                   help="checkpoint hangi degerle EGITILDIYSE o verilmeli (0 = h_ego residual yok)")
    p.add_argument("--gate", type=str, default="softmax", choices=["softmax", "sigmoid"],
                   help="checkpoint hangi kapi moduyla EGITILDIYSE o verilmeli (aksi halde maske yanlis hesaplanir)")
    p.add_argument("--modes", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--out", type=str, default="causal_bev.png")
    p.add_argument("--device", type=str, default="cuda")
    main(p.parse_args())
