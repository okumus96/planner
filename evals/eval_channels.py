"""Step 1: kanal aktivasyonlarinin OFFLINE dogrulamasi (training yok, model degisikligi yok).

Uc mod:
  --selftest              : sentetik, bilinen-cevapli sahnelerle unit test (veri gerekmez)
  --data <npz_dir>        : istatistikler -- fire oranlari, multi-fire histogrami (tek-kanal
                            hipotezi), coverage (yakin/kapanan ama sifir kanalli ajan), GT-vs-GF
                            kanal uyumu (--pretrained_path verilirse)
  --viz N                 : ilk N sahneyi BEV olarak cizer (viz_causal paletiyle ayni mantik:
                            ajan rengi = yanan kanal; koridor cizgisi; GF future'lari)

Ornek:
  python eval_channels.py --selftest
  python eval_channels.py --data /path/to/test14_random_reduced_npz \
      --pretrained_path training_log/normal/model_epoch_19_valADE_1.6487.pth --viz 9
"""
import argparse
import glob
import json
import math
import os

import numpy as np
import torch

from GameFormer.channels import (compute_channels, CHANNEL_NAMES, NUM_CHANNELS,
                                 CH_SAME_LANE_AHEAD, CH_ADJACENT_RIGHT, CH_SHARES_INTERSECTION,
                                 CH_MERGES, CH_NEAR, CH_FOLLOWS, EV_DFS, EV_CLOSING, NEAR_M,
                                 compute_map_channels, MAP_CHANNEL_NAMES, NUM_MAP_CHANNELS,
                                 MCH_IN_LANE, MCH_ADJ_LEFT, MCH_ADJ_RIGHT, MCH_SUCCESSOR,
                                 MCH_ROUTE, MCH_TRAFFIC, MCH_NEAR)


def _load_npz(path, num_neighbors):
    d = np.load(path, allow_pickle=True)
    out = {
        "neighbor_agents_past": torch.from_numpy(d["neighbor_agents_past"][:num_neighbors]).float()[None],
        "ego_agent_past": torch.from_numpy(d["ego_agent_past"]).float()[None],
        "gt_futures": torch.from_numpy(d["neighbor_agents_future"][:num_neighbors, :, :2]).float()[None],
        "ref_path": torch.from_numpy(d["c_lat_candidates"]).float()[None],
        "map_lanes": torch.from_numpy(d["lanes"]).float()[None],
        "map_crosswalks": torch.from_numpy(d["crosswalks"]).float()[None],
        "route_lanes": torch.from_numpy(d["route_lanes"]).float()[None],
        "token": str(d["token"]),
    }
    return out


@torch.no_grad()
def _gf_futures(gameformer, sample, num_neighbors, device):
    """viz_causal ile ayni yol: frozen encoder+decoder -> top-1 komsu gelecekleri."""
    from train_planner import extract_neighbor_top1_futures
    inputs = {
        "ego_agent_past": sample["ego_agent_past"].to(device),
        "neighbor_agents_past": sample["neighbor_agents_past"].to(device),
    }
    # encoder harita girdilerini de ister -- npz'den tamamla
    return extract_neighbor_top1_futures(gameformer, gameformer.encoder(inputs), num_neighbors)


def run_dataset(args):
    files = sorted(glob.glob(os.path.join(args.data, "*.npz")))
    assert files, f"npz bulunamadi: {args.data}"
    if args.shuffle:
        import random as _rnd
        _rnd.Random(args.seed).shuffle(files)     # tekrarlanabilir rastgele ornekleme
    do_r1 = args.mode in ("all", "r1")
    do_r2 = args.mode in ("all", "r2")
    print(f"{len(files)} senaryo bulundu (mode={args.mode}).")

    gameformer = None
    if args.pretrained_path and do_r1:
        from GameFormer.predictor import GameFormer
        gameformer = GameFormer(encoder_layers=args.encoder_layers,
                                decoder_levels=args.decoder_levels, neighbors=args.num_neighbors)
        gameformer.load_state_dict(torch.load(args.pretrained_path, map_location=args.device))
        gameformer.to(args.device).eval()

    fire = np.zeros(NUM_CHANNELS, dtype=np.int64)      # kanal basina yanan pair sayisi
    agree = np.zeros(NUM_CHANNELS, dtype=np.int64)     # GT-vs-GF ayni karar
    both = np.zeros(NUM_CHANNELS, dtype=np.int64)      # karsilastirilan pair sayisi
    inter = np.zeros(NUM_CHANNELS, dtype=np.int64)     # GT ∧ GF yanan (pozitif-IoU pay)
    union = np.zeros(NUM_CHANNELS, dtype=np.int64)     # GT ∨ GF yanan (pozitif-IoU payda)
    multi = np.zeros(6, dtype=np.int64)                # 0,1,2,3,4,5+ kanal yanan pair histogrami
    n_pairs = 0
    cover_miss = 0                                     # yakin/kapanan ama sifir kanal
    cover_all = 0
    miss_tip = np.zeros(3, dtype=np.int64)             # kacanlar: veh/ped/bic
    miss_dir = np.zeros(3, dtype=np.int64)             # kacanlar: ayni-yon / kesen / karsi-yon
    # R2 (harita) istatistikleri
    mfire = np.zeros(NUM_MAP_CHANNELS, dtype=np.int64)
    mmulti = np.zeros(4, dtype=np.int64)               # 0,1,2,3+ kanal yanan eleman
    n_elems = 0
    viz_scenes = []

    for f in files:
        s = _load_npz(f, args.num_neighbors)
        valid = s["neighbor_agents_past"][0, :, -1, :2].abs().sum(-1) > 1e-6
        act = torch.zeros(1, s["neighbor_agents_past"].shape[1], NUM_CHANNELS, dtype=torch.bool)
        if do_r1:
            act_gt, ev = compute_channels(s["neighbor_agents_past"], s["ego_agent_past"],
                                          s["gt_futures"], s["ref_path"])
            act = act_gt
            if gameformer is not None:
                npast = s["neighbor_agents_past"].to(args.device)
                inputs = {"ego_agent_past": s["ego_agent_past"].to(args.device),
                          "neighbor_agents_past": npast}
                # harita anahtarlarini npz'den ekle (encoder icin gerekli)
                d = np.load(f, allow_pickle=True)
                for k_npz, k_in in (("lanes", "map_lanes"), ("crosswalks", "map_crosswalks"),
                                    ("route_lanes", "route_lanes")):
                    inputs[k_in] = torch.from_numpy(d[k_npz]).float()[None].to(args.device)
                from train_planner import extract_neighbor_top1_futures
                top1, _, _ = extract_neighbor_top1_futures(gameformer, gameformer.encoder(inputs),
                                                           args.num_neighbors)
                act_gf, _ = compute_channels(s["neighbor_agents_past"], s["ego_agent_past"],
                                             top1.cpu(), s["ref_path"])
                m = valid[None, :, None].expand_as(act_gt)
                agree += ((act_gt == act_gf) & m).sum((0, 1)).numpy()
                both += m.sum((0, 1)).numpy()
                inter += (act_gt & act_gf & m).sum((0, 1)).numpy()
                union += ((act_gt | act_gf) & m).sum((0, 1)).numpy()
                act = act_gf   # istatistigi runtime kosuluyla (GF) raporla

            a = act[0][valid]                          # [n_valid, R]
            e = ev[0][valid]
            n_pairs += len(a)
            fire += a.sum(0).numpy()
            cnt = a.sum(-1).clamp(max=5).numpy()
            for c in cnt:
                multi[int(c)] += 1
            # coverage: "onemli gorunen" ama sifir kanalli. Onceki tanim (closing>0.5 tek basina)
            # 8 s ufkunda erisilemeyen uzak nesneleri de "onemli" sayiyordu (%33.7 sahte alarm);
            # simdiki: yakin VEYA (kapaniyor ∧ sabit-hiz TTC ≤ 8 s).
            ttc_cv = e[:, EV_DFS] / e[:, EV_CLOSING].clamp(min=1e-3)
            important = (e[:, EV_DFS] <= NEAR_M) | ((e[:, EV_CLOSING] > 0.5) & (ttc_cv <= 8.0))
            cover_all += int(important.sum())
            miss = important & (a.sum(-1) == 0)
            cover_miss += int(miss.sum())
            # kacanlarin profili: tip (veh/ped/bic) ve yon (ayni <0.45 / kesen / karsi >2.70)
            tipv = s["neighbor_agents_past"][0, :, -1, 8:11].argmax(-1)[valid]
            dthv = e[:, 6]
            for k in range(3):
                miss_tip[k] += int((miss & (tipv == k)).sum())
            miss_dir[0] += int((miss & (dthv <= 0.45)).sum())
            miss_dir[1] += int((miss & (dthv > 0.45) & (dthv < 2.70)).sum())
            miss_dir[2] += int((miss & (dthv >= 2.70)).sum())
            if args.inspect and args.inspect in s["token"]:
                short = ["ahead", "behind", "adjL", "adjR", "collide", "intersect", "near",
                         "follows", "merges", "overtakes", "vru"]
                tipn = ["veh", "ped", "bic"]
                cur = s["neighbor_agents_past"][0, :, -1]
                print(f"\n--- INSPECT {s['token']} (ajan-basina ego->agent karari) ---")
                print(f"{'idx':>4} {'tip':>4} {'kanallar':<24} {'x':>7} {'y':>7} {'d_fs':>6} "
                      f"{'ds':>7} {'d_lat':>7} {'closing':>8} {'ttc':>5} {'dtheta':>7}")
                for j in range(act.shape[1]):
                    if not valid[j]:
                        continue
                    chs = ",".join(short[c] for c in range(NUM_CHANNELS) if act[0, j, c]) or "-"
                    ej = ev[0, j]
                    print(f"{j:>4} {tipn[int(cur[j, 8:11].argmax())]:>4} {chs:<24} "
                          f"{float(cur[j, 0]):>7.1f} {float(cur[j, 1]):>7.1f} {float(ej[EV_DFS]):>6.1f} "
                          f"{float(ej[0]):>7.1f} {float(ej[1]):>7.1f} {float(ej[EV_CLOSING]):>8.2f} "
                          f"{float(ej[4]):>5.1f} {float(ej[6]):>7.2f}")

        # R2: harita kanallari (future gerektirmez; token sirasi modelle ayni)
        if do_r2:
            mact, mev = compute_map_channels(s["map_lanes"], s["map_crosswalks"],
                                             s["route_lanes"], s["ref_path"])
            ma = mact[0]                               # gecerlilik compute icinde maskelendi
            n_elems += int(ma.shape[0])
            mfire += ma.sum(0).numpy()
            mcnt = ma.sum(-1).clamp(max=3).numpy()
            for c in mcnt:
                mmulti[int(c)] += 1
            if args.inspect and args.inspect in s["token"]:
                Lc = s["map_lanes"].shape[1]
                Cc = s["map_crosswalks"].shape[1]
                me = mev[0]
                print(f"\n--- INSPECT {s['token']} (eleman-basina R2 karari) ---")
                print(f"{'idx':>4} {'tur':>6} {'kanallar':<28} {'min_d':>7} {'med_dlat':>9} "
                      f"{'s_near':>7} {'dtheta':>7}")
                for k in range(ma.shape[0]):
                    if float(me[k, 0]) == 0.0 and not ma[k].any():
                        continue                        # gecersiz eleman
                    if float(me[k, 0]) > 45.0 and not ma[k].any():
                        continue                        # uzak ve sessiz -> atla
                    kind = "lane" if k < Lc else ("cwalk" if k < Lc + Cc else "route")
                    chs = ",".join(MAP_CHANNEL_NAMES[c] for c in range(NUM_MAP_CHANNELS)
                                   if ma[k, c]) or "-"
                    print(f"{k:>4} {kind:>6} {chs:<28} {float(me[k, 0]):>7.1f} "
                          f"{float(me[k, 1]):>9.1f} {float(me[k, 2]):>7.1f} {float(me[k, 3]):>7.2f}")
        else:
            S = (s["map_lanes"].shape[1] + s["map_crosswalks"].shape[1]
                 + s["route_lanes"].shape[1])
            ma = torch.zeros(S, NUM_MAP_CHANNELS, dtype=torch.bool)

        if args.viz and len(viz_scenes) < args.viz:
            # cizilen future = KARARI VEREN future (GF kullanildiysa GF; yoksa GT) --
            # aksi halde GT cizgisi duz giderken etiket collide diyebilir (olculdu: f4d2 j=6)
            fut_used = (top1.detach().cpu()[0].numpy() if (do_r1 and gameformer is not None)
                        else s["gt_futures"][0].numpy())
            viz_scenes.append((s, act[0].numpy(), valid.numpy(), ma.numpy(), fut_used))

    if do_r1:
        print("\n=== R1 KANAL FIRE ORANLARI (pair basina) ===")
        for i, name in enumerate(CHANNEL_NAMES):
            print(f"  {name:20s} {fire[i]:6d}  ({100.0 * fire[i] / max(n_pairs, 1):5.1f}%)")
        print(f"\n=== R1 MULTI-FIRE HISTOGRAMI ({n_pairs} pair) ===")
        for k in range(6):
            lab = f"{k}" if k < 5 else "5+"
            print(f"  {lab:>2s} kanal: {multi[k]:6d}  ({100.0 * multi[k] / max(n_pairs, 1):5.1f}%)")
        print(f"\n=== R1 COVERAGE ===\n  onemli-gorunen {cover_all} pair icinde sifir-kanal: "
              f"{cover_miss} ({100.0 * cover_miss / max(cover_all, 1):.2f}%)  <- 0'a yakin olmali")
        cm = max(cover_miss, 1)
        print(f"  kacanlarin tipi : veh {miss_tip[0]} ({100.0 * miss_tip[0] / cm:.0f}%)  "
              f"ped {miss_tip[1]} ({100.0 * miss_tip[1] / cm:.0f}%)  "
              f"bic {miss_tip[2]} ({100.0 * miss_tip[2] / cm:.0f}%)")
        print(f"  kacanlarin yonu : ayni {miss_dir[0]} ({100.0 * miss_dir[0] / cm:.0f}%)  "
              f"kesen {miss_dir[1]} ({100.0 * miss_dir[1] / cm:.0f}%)  "
              f"karsi {miss_dir[2]} ({100.0 * miss_dir[2] / cm:.0f}%)")
        if gameformer is not None:
            print("\n=== R1 GT-vs-GF KANAL UYUMU (ham | pozitif-IoU) ===")
            for i, name in enumerate(CHANNEL_NAMES):
                iou = 100.0 * inter[i] / union[i] if union[i] else float("nan")
                print(f"  {name:20s} {100.0 * agree[i] / max(both[i], 1):6.2f}%  |  IoU {iou:6.2f}%")
    if do_r2:
        print(f"\n=== R2 HARITA KANALLARI ({n_elems} eleman) ===")
        for i, name in enumerate(MAP_CHANNEL_NAMES):
            print(f"  {name:24s} {mfire[i]:7d}  ({100.0 * mfire[i] / max(n_elems, 1):5.1f}%)")
        for k in range(4):
            lab = f"{k}" if k < 3 else "3+"
            print(f"  {lab:>2s} kanal yanan eleman: {mmulti[k]:7d}  ({100.0 * mmulti[k] / max(n_elems, 1):5.1f}%)")

    if args.viz:
        _draw(viz_scenes, args.out, mode=args.mode)
    if args.json:
        payload = {"mode": args.mode}
        if do_r1:
            payload.update({"n_pairs": int(n_pairs), "fire": fire.tolist(),
                            "channel_names": CHANNEL_NAMES, "multi": multi.tolist(),
                            "coverage_miss": int(cover_miss), "coverage_all": int(cover_all),
                            "agree": agree.tolist(), "compared": both.tolist(),
                            "iou_inter": inter.tolist(), "iou_union": union.tolist()})
        if do_r2:
            payload.update({"r2_n_elems": int(n_elems), "r2_fire": mfire.tolist(),
                            "r2_channel_names": MAP_CHANNEL_NAMES,
                            "r2_multi": mmulti.tolist()})
        with open(args.json, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"JSON: {args.json}")


def _draw(scenes, out, mode="all"):
    """KG video pipeline'inin frame stili (notebook cell 9/10 paleti): serit bantlari #dfe8f3,
    crosswalk #fff4b8, route #a58abf; ajan sekli tipe gore (rect/circle/triangle), rengi kanala gore
    (kanal renkleri KG iliski paletine eslendi: follows-mavi, merge-turuncu, overtake-kirmizi...)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle, RegularPolygon, Patch
    from matplotlib.lines import Line2D
    from matplotlib.transforms import Affine2D
    # ego->agent kanal renkleri (R=10) -- KG iliski paleti (cell 10) uzerine eslenmis
    colors = ["#1565c0",  # same_lane_ahead  (mavi)
              "#64b5f6",  # same_lane_behind (acik mavi)
              "#6a1b9a",  # adjacent_left    (KG moru)
              "#8e24aa",  # adjacent_right
              "#b93b3b",  # onObservedCollisionCourseWith (KG stop-line kirmizisi)
              "#c9a55a",  # sharesIntersectionWith (KG intersection tonu)
              "#9e9e9e",  # near             (KG unknown grisi)
              "#0d47a1",  # follows          (KG follows koyu mavisi)
              "#fb8c00",  # merges           (KG merge turuncusu)
              "#e53935",  # overtakes        (KG overtake kirmizisi)
              "#00838f"]  # vulnerable_road_user_near_ego_path (teal)
    short = ["ahead", "behind", "adjL", "adjR", "collide", "intersect", "near",
             "follows", "merges", "overtakes", "vru"]
    # R2: taban serit NOTR GRI (kanal renkleri okunabilsin diye maviden cekildi);
    # her kanal AYRI renk, oncelik: inLane > successor > adjacent > route
    LANE_FACE, LANE_EDGE = "#e8e8e8", "#c4c4c4"
    R2_INLANE = "#66bb6a"         # yesil: ego'nun seridi
    R2_SUCC = "#64b5f6"           # mavi: gececegi yol
    R2_ADJ = "#ce93d8"            # mor: komsu serit
    R2_ROUTE = "#f06292"          # pembe: rota seridi (yesil/mavi/mor/teal ile karismayan tek bos ton)
    CW_FACE, CW_EDGE = "#fff4b8", "#b59b31"
    ROUTE_C = "#7b1fa2"
    TL_EDGE = "#b93b3b"
    EGO_C = "#212121"             # KG ego rengi

    show_r1 = mode in ("all", "r1")
    show_r2 = mode in ("all", "r2")
    n = len(scenes)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.6, rows * 4.6), squeeze=False)
    for i, (s, act, valid, mact, fut_used) in enumerate(scenes):
        ax = axes[i // cols][i % cols]
        ax.set_facecolor("#fafafa")

        def _poly(t):
            xy = t.numpy()
            return [p[np.abs(p[:, :2]).sum(-1) > 1e-6][:, :2] for p in xy]

        lanes = _poly(s["map_lanes"][0])
        cwalks = _poly(s["map_crosswalks"][0])
        routes = _poly(s["route_lanes"][0])
        L = len(lanes)
        C = len(cwalks)
        # seritler: genis bant (KG lane dolgusu) + ince centerline; R2 yanan seritler koyu ton
        for k, p in enumerate(lanes):
            if len(p) < 2:
                continue
            face, alpha = LANE_FACE, 0.55
            if show_r2:
                if mact[k, MCH_IN_LANE]:
                    face, alpha = R2_INLANE, 0.8
                elif mact[k, MCH_SUCCESSOR]:
                    face, alpha = R2_SUCC, 0.8
                elif mact[k, MCH_ADJ_LEFT] or mact[k, MCH_ADJ_RIGHT]:
                    face, alpha = R2_ADJ, 0.75
                elif mact[k, MCH_ROUTE]:
                    face, alpha = R2_ROUTE, 0.75
            ax.plot(p[:, 0], p[:, 1], color=face, lw=8, alpha=alpha,
                    solid_capstyle="round", zorder=0.6)
            ax.plot(p[:, 0], p[:, 1], color=LANE_EDGE, lw=0.6, alpha=0.6, zorder=0.7)
            if show_r2 and mact[k, MCH_TRAFFIC]:
                ax.plot(p[:, 0], p[:, 1], color=TL_EDGE, lw=1.1, ls="--", alpha=0.9, zorder=0.8)
        if show_r2:                                     # r1 modunda crosswalk cizilmez (ajan odagi)
            for k, p in enumerate(cwalks):
                if len(p) < 2:
                    continue
                ax.plot(p[:, 0], p[:, 1], color=CW_FACE, lw=8, alpha=0.8, solid_capstyle="round", zorder=0.9)
                ax.plot(p[:, 0], p[:, 1], color=CW_EDGE, lw=0.8, alpha=0.8, zorder=1.0)
        for p in routes:
            p = p[p[:, 0] > 0.0]            # ego'nun gerisindeki route parcasi cizilmez (lookbehind=0)
            if len(p) < 2:
                continue
            # route tokenlari = ego_route_corridor kanalinin kendisi -> PEMBE (legend'la tutarli;
            # onceden koyu mor cizilip adjacent moruyla karisiyordu)
            ax.plot(p[:, 0], p[:, 1], color=R2_ROUTE, lw=1.6, ls="--", alpha=0.9, zorder=1.1)
        rp = s["ref_path"][0, 0, :, :2].numpy()
        rp = rp[np.abs(rp).sum(-1) > 1e-6]
        ax.plot(rp[:, 0], rp[:, 1], color="#616161", lw=1.6, ls=":", alpha=0.9, zorder=1.2)

        cur = s["neighbor_agents_past"][0, :, -1].numpy()
        fut = fut_used
        rect = Rectangle((-2.3, -1.0), 4.6, 2.0, facecolor=EGO_C, edgecolor="black", lw=0.8, zorder=6)
        rect.set_transform(Affine2D().rotate(0).translate(0, 0) + ax.transData)
        ax.add_patch(rect)
        ax.plot([0.0, 2.1], [0.0, 0.0], color="white", lw=1.5, zorder=7)   # heading cizgisi (one dogru)
        for j in range(cur.shape[0]):
            if not valid[j]:
                continue
            ch = np.where(act[j])[0] if show_r1 else np.array([], dtype=int)
            fc = colors[ch[0]] if len(ch) else "#eeeeee"
            ec = "#212121" if len(ch) else "#4d4d4d"
            atype = int(np.argmax(cur[j, 8:11]))
            x, y, hd = cur[j, 0], cur[j, 1], cur[j, 2]
            if atype == 1:      # yaya: daire (KG sekil kurali)
                ax.add_patch(Circle((x, y), 0.9, facecolor=fc, edgecolor=ec, lw=0.7, zorder=5))
            elif atype == 2:    # bisiklet: ucgen
                ax.add_patch(RegularPolygon((x, y), 3, radius=1.1, orientation=hd,
                                            facecolor=fc, edgecolor=ec, lw=0.7, zorder=5))
            else:
                r = Rectangle((-cur[j, 6] / 2, -cur[j, 7] / 2), max(cur[j, 6], 1.0),
                              max(cur[j, 7], 0.6), facecolor=fc, edgecolor=ec, lw=0.7, zorder=5)
                r.set_transform(Affine2D().rotate(hd).translate(x, y) + ax.transData)
                ax.add_patch(r)
            if len(ch):
                ax.plot(fut[j, :, 0], fut[j, :, 1], color=fc, lw=1.1, alpha=0.85, zorder=4)
                ax.text(x, y + 1.7, ",".join(short[c] for c in ch),
                        fontsize=6, ha="center", color="#212121", zorder=7)
        ax.set_xlim(-40, 80)
        ax.set_ylim(-40, 40)
        ax.set_aspect("equal")
        ax.set_title(s["token"][:12], fontsize=8)
    for i in range(n, rows * cols):
        axes[i // cols][i % cols].axis("off")
    handles = []
    if show_r1:
        seen = set()
        for k in range(NUM_CHANNELS):
            if colors[k] in seen:
                continue
            seen.add(colors[k])
            # legend = viz tag'iyle AYNI kisa ad + tam KG adi (tag/legend uyumsuzlugu duzeltmesi)
            handles.append(Patch(facecolor=colors[k], label=f"{short[k]} = {CHANNEL_NAMES[k]}"))
    if show_r2:
        handles += [
            Patch(facecolor=R2_INLANE, label="inLane"),
            Patch(facecolor=R2_SUCC, label="successor"),
            Patch(facecolor=R2_ADJ, label="adjacent_L/R"),
            Line2D([0], [0], color=R2_ROUTE, ls="--", label="ego_route_corridor (route)"),
            Line2D([0], [0], color=TL_EDGE, ls="--", label="traffic_control"),
        ]
    else:
        handles += [Line2D([0], [0], color=R2_ROUTE, ls="--", label="route")]
    handles += [Patch(facecolor=LANE_FACE, label="lane")]
    if show_r2:
        handles += [Patch(facecolor=CW_FACE, label="crosswalk")]
    handles += [
        Line2D([0], [0], color="#616161", ls=":", label="ref path"),
        Patch(facecolor=EGO_C, label="ego"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=7)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Viz: {out}")


def selftest():
    """Bilinen-cevapli sentetik sahneler. Koridor: +x boyunca duz, 120 m."""
    P = 1200
    ref = torch.zeros(1, 5, P, 6)
    ref[0, 0, :, 0] = torch.arange(P) * 0.1          # x: 0..120
    # yaw = 0, y = 0

    def agent(x, y, th, vx, vy, past_x0=None, tip=0):
        a = torch.zeros(21, 11)
        a[:, 0] = x if past_x0 is None else torch.linspace(past_x0, x, 21)
        a[:, 1] = y
        a[:, 2] = th
        a[:, 3] = vx
        a[:, 4] = vy
        a[:, 6] = 4.5 if tip == 0 else 0.8
        a[:, 7] = 2.0 if tip == 0 else 0.8
        a[:, 8 + tip] = 1.0                       # one-hot: 8=veh, 9=ped, 10=bic
        return a

    ego = torch.zeros(1, 21, 7)
    ego[0, :, 3] = 5.0                                # 5 m/s ileri

    # sahne: 6 ajan
    nbr = torch.zeros(1, 6, 21, 11)
    nbr[0, 0] = agent(20.0, 0.0, 0.0, 5.0, 0.0)                    # onde ayni serit -> corridor-ahead
    nbr[0, 1] = agent(15.0, -9.0, 0.0, 0.0, 0.0)                   # sagda uzak park -> hicbir sey
    nbr[0, 2] = agent(12.0, -3.2, 0.15, 5.0, 0.8)                  # sagdan sokulan -> adjacent-right + merging
    nbr[0, 3] = agent(30.0, -20.0, math.pi / 2, 0.0, 6.0)          # kesen -> path-crossing (+collision)
    # YAYA, arac olsaydi adjacent+overtaking ATESLEYECEK kinematikte -> tip guard testi
    nbr[0, 4] = agent(2.0, -3.2, 0.0, 6.0, 0.0, past_x0=-1.5, tip=1)
    # YAKLASAN yaya, 9.2 m'de (5-12 m olu bolgesi; gecis mesafesi ~4 m -> collision degil)
    nbr[0, 5] = agent(8.0, 4.5, -2.9, -1.2, -0.3, tip=1)

    T = 80
    fut = torch.zeros(1, 6, T, 2)
    t = torch.arange(1, T + 1).float() * 0.1
    fut[0, 0, :, 0] = 20.0 + 5.0 * t                               # duz devam
    fut[0, 1, :, 0] = 15.0
    fut[0, 1, :, 1] = -9.0                                         # sabit
    fut[0, 2, :, 0] = 12.0 + 5.0 * t
    fut[0, 2, :, 1] = torch.clamp(-3.2 + 0.8 * t, max=0.0)         # serite giriyor
    fut[0, 3, :, 0] = 30.0
    fut[0, 3, :, 1] = -20.0 + 6.0 * t                              # dik kesiyor
    fut[0, 4, :, 0] = 2.0 + 6.0 * t
    fut[0, 4, :, 1] = -3.2                                         # yaya duz gidiyor
    fut[0, 5, :, 0] = 8.0 - 1.2 * t
    fut[0, 5, :, 1] = 4.5 - 0.3 * t                                # yaklasan yaya

    act, ev = compute_channels(nbr, ego, fut, ref)
    a = act[0]
    ok = True

    def check(cond, msg):
        nonlocal ok
        status = "OK " if cond else "FAIL"
        if not cond:
            ok = False
        print(f"  [{status}] {msg}")

    check(bool(a[0, CH_SAME_LANE_AHEAD]) and bool(a[0, CH_FOLLOWS]) and a[0].sum() == 2,
          "lead: same_lane_ahead + follows (lider zarfi: g=15.4, headway 3.1 s)")
    check(a[1].sum() == 0, "uzak park: hicbir kanal")
    check(bool(a[2, CH_ADJACENT_RIGHT]) and bool(a[2, CH_MERGES]), "sokulan: adjacent_right + merges")
    check(bool(a[3, CH_SHARES_INTERSECTION]), "kesen: sharesIntersectionWith")
    from GameFormer.channels import CH_OVERTAKES, CH_ADJACENT_LEFT, EV_DS_ENTRY
    ped_lane_ch = bool(a[4, CH_OVERTAKES]) or bool(a[4, CH_ADJACENT_RIGHT]) or bool(a[4, CH_ADJACENT_LEFT])
    check((not ped_lane_ch) and bool(a[4, CH_NEAR]),
          "yaya (arac-kinematikli): serit kanallari YOK, sadece near (KG tip guard'i)")
    check(float(ev[0, 2, EV_DS_ENTRY]) > 0,
          "sokulan: giris-ds POZITIF (onume girecek ~ mergesInFrontOf yonu)")
    from GameFormer.channels import CH_VRU
    check(bool(a[5, CH_VRU]) and a[5].sum() == 1,
          "yaklasan yaya (9.2 m, olu bolge): SADECE vulnerable_road_user_near_ego_path")

    # ---- R2 selftest: duz koridor, 4 lane + 1 crosswalk + 1 route tokeni ----
    Pl = 50
    xs = torch.arange(Pl).float()
    lanes = torch.zeros(1, 6, Pl, 7)
    lanes[0, 0, :, 0] = xs - 10.0                     # ego'nun seridi (ego icinde) -> inLane
    lanes[0, 1, :, 0] = xs + 45.0                     # ileride koridor ustunde -> successor
    lanes[0, 1, :, 3] = 1.0                           # TL kaydi var -> traffic_control da yanmali
    lanes[0, 2, :, 0] = xs - 10.0
    lanes[0, 2, :, 1] = 3.5                           # sol paralel -> adjacent_left
    lanes[0, 3, :, 0] = xs - 10.0
    lanes[0, 3, :, 1] = 25.0                          # uzak paralel (>20 m) -> hicbiri (near bile degil)
    lanes[0, 4, :, 0] = 10.0 - xs                     # KARSI YON seridi (heading pi), y=3.5
    lanes[0, 4, :, 1] = 3.5
    lanes[0, 4, :, 2] = math.pi                       # -> adjacent YANMAMALI (ayni-yon sarti)
    lanes[0, 5, :, 0] = -20.0 + xs * 0.36             # geride, egoya ULASMIYOR (-20..-2), y=3.5
    lanes[0, 5, :, 1] = 3.5                           # -> adjacent YANMAMALI (reaches_ego kurali)
    cwalks = torch.zeros(1, 1, 30, 3)
    cwalks[0, 0, :, 0] = 12.0
    cwalks[0, 0, :, 1] = torch.linspace(-4, 4, 30)    # yolu kesen crosswalk (v1'de near'a duser)
    cwalks[0, 0, :, 2] = math.pi / 2
    routes = torch.zeros(1, 2, Pl, 3)
    routes[0, 0, :, 0] = xs                            # route tokeni -> ego_route_corridor
    routes[0, 1, :, 0] = -90.0 + xs * 0.5              # TAMAMEN geride (-90..-65) -> yanmamali
    mact, _ = compute_map_channels(lanes, cwalks, routes, ref)
    m = mact[0]
    check(bool(m[0, MCH_IN_LANE]), "R2 lane0: inLane")
    check(bool(m[1, MCH_SUCCESSOR]) and bool(m[1, MCH_TRAFFIC]), "R2 lane1: successor + traffic_control")
    check(bool(m[2, MCH_ADJ_LEFT]), "R2 lane2: adjacent_left")
    check(m[3].sum() == 0, "R2 lane3 (25 m lateral): hicbir kanal")
    check(not (bool(m[4, MCH_ADJ_LEFT]) or bool(m[4, MCH_ADJ_RIGHT])),
          "R2 KARSI YON seridi: adjacent YANMAZ (ayni-yon sarti)")
    check(not (bool(m[5, MCH_ADJ_LEFT]) or bool(m[5, MCH_ADJ_RIGHT])),
          "R2 geride kalan (egoya ulasmayan) paralel serit: adjacent YANMAZ (reaches_ego)")
    check(bool(m[7, MCH_ROUTE]), "R2 route tokeni: ego_route_corridor")
    check(not bool(m[8, MCH_ROUTE]), "R2 geride kalan route tokeni: YANMAZ (lookbehind=0, reaches_ego)")
    print("SELFTEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--data", type=str, default=None, help="npz dizini (test14_random_reduced islenmis)")
    p.add_argument("--pretrained_path", type=str, default=None, help="verilirse GT-vs-GF uyumu da olculur")
    p.add_argument("--num_neighbors", type=int, default=10)
    p.add_argument("--encoder_layers", type=int, default=3)
    p.add_argument("--decoder_levels", type=int, default=2)
    p.add_argument("--viz", type=int, default=0)
    p.add_argument("--inspect", type=str, default=None,
                   help="senaryo token on-eki: o sahnede eleman-basina R2 kararlari + kanitlari yazdirilir")
    p.add_argument("--mode", type=str, default="all", choices=["all", "r1", "r2"],
                   help="r1 = sadece ajan kanallari, r2 = sadece harita kanallari; "
                        "istatistik + JSON + viz + legend HEPSI bu moda uyar")
    p.add_argument("--shuffle", action="store_true", help="sahneleri rastgele sirala (viz orneklemesi icin)")
    p.add_argument("--seed", type=int, default=0, help="--shuffle tohumu; farkli sayfalar icin degistir")
    p.add_argument("--out", type=str, default="channels_bev.png")
    p.add_argument("--json", type=str, default=None)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()
    if args.selftest:
        raise SystemExit(selftest())
    assert args.data, "--data veya --selftest gerekli"
    run_dataset(args)
