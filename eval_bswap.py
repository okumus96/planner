"""b*-swap: karar slotu YUK TASIYOR mu? (H'nin falsification deneyi)

Iki olcum:
  1. AGREEMENT: modelin urettigi plani decision_labels() ile YENIDEN etiketle, ilan ettigi
     b* = (lon, lat) ile karsilastir -> "planner ilan ettigi kararla %X uyumlu"
     ("dekoratif karar" itirazinin cevabi).
  2. SWAP/COMPLIANCE: f_cas ve ego_clean SABIT, head'e ZORLANMIS karar ver:
     head(f_cas, ego_clean, (lon', lat')). Zorlanmis plani yeniden etiketle ->
     compliance = relabel == zorlanan sinif (yalniz zorlanan != ilan edilen sahnelerde).
     Ek: doz-yon kontrolleri (stop/slow zorla -> uc hiz dusuyor mu; turn zorla -> heading
     dogru yone donuyor mu) + Delta-plan buyuklugu.

Aile-duzeyi compliance ana okuma: hareket halindeki ego'ya "remain_stopped" zorlansa en iyi
ihtimalle stop_* olarak etiketlenir (4 s'de fiziksel durus) -> tam-sinif compliance yapisal
olarak cezali; aile {stop-ish, slow, accel, maintain, reverse} adil olcek.

Kosum (v2 ckpt, cuda:1):
  python eval_bswap.py --pretrained_path training_log/normal/model_epoch_19_valADE_1.6487.pth \
    --causal_path training_log/dodmeta_typed_noresid_v2/causal_epoch_13_minADE_0.7253.pth \
    --valid_set /home/lt-hta-ai4/ssd1/nuplan/processed_data/validation \
    --graph_layers 1 --nbr_enrich 2 --ego_residual 0 --gate_channels 1 --typed_kv 1 \
    --dod_meta 1 --device cuda:1
"""
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from GameFormer.predictor import GameFormer
from GameFormer.causal_graph import CausalPlanner
from GameFormer.train_utils import DrivingData
from GameFormer.decision_labels import decision_labels, LON_CLASSES, LAT_CLASSES, NUM_LON, NUM_LAT
from train_planner import read_batch, extract_neighbor_top1_futures, freeze_gameformer

# aile katlamasi: hareketli ego'ya zorlanmis "dur" ailesi tek grupta adil olculur
LON_FAMILY = {0: 'stop-ish', 1: 'stop-ish', 2: 'stop-ish', 3: 'slow', 4: 'slow',
              5: 'accel', 6: 'accel', 7: 'maintain', 8: 'reverse'}
LAT_FAMILY = {0: 'turn_left', 1: 'turn_right', 2: 'left-ish', 3: 'right-ish',
              4: 'left-ish', 5: 'right-ish', 6: 'no_lateral'}


def plan_with_heading(xy):
    """[B,80,2] tahmin xy -> [B,80,3] (x,y,heading); heading ardisik farklardan
    (deployment'taki plan donusumuyle ayni yontem)."""
    d = xy[:, 1:] - xy[:, :-1]
    hd = torch.atan2(d[..., 1], d[..., 0])
    hd = torch.cat([hd[:, :1], hd], dim=1)
    return torch.cat([xy, hd.unsqueeze(-1)], dim=-1)


def end_speed(xy):
    """[B,80,2] -> [B] lon penceresi sonunda (3.5-4.0 s) ortalama hiz."""
    v = (xy[:, 1:] - xy[:, :-1]).norm(dim=-1) / 0.1
    return v[:, 34:40].mean(dim=1)


def net_heading(xy):
    """[B,80,2] -> [B] net yon degisimi (ilk->son gecerli segment)."""
    d = xy[:, 1:] - xy[:, :-1]
    hd = torch.atan2(d[..., 1], d[..., 0])
    return torch.atan2(torch.sin(hd[:, -1] - hd[:, 0]), torch.cos(hd[:, -1] - hd[:, 0]))


@torch.no_grad()
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
    miss, unexp = causal.load_state_dict(torch.load(args.causal_path, map_location=dev), strict=False)
    if miss or unexp:
        print(f"[load] missing={list(miss)}  unexpected={list(unexp)}")
    causal.eval()
    assert causal.dod_meta, "b*-swap dod_meta ckpt ister (factored karar slotu)"

    ds = DrivingData(args.valid_set + "/*.npz", args.num_neighbors)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    n = 0
    agree_lon = agree_lat = agree_joint = 0
    # compliance sayaclari: [zorlanan][yeniden-etiket]
    C_lon = np.zeros((NUM_LON, NUM_LON), dtype=int)
    C_lat = np.zeros((NUM_LAT, NUM_LAT), dtype=int)
    lon_forced_n = np.zeros(NUM_LON, dtype=int)
    lat_forced_n = np.zeros(NUM_LAT, dtype=int)
    dv = {c: [] for c in range(NUM_LON)}         # zorlanan lon -> uc-hiz farki (forced - base)
    dhd = {c: [] for c in range(NUM_LAT)}        # zorlanan lat -> net heading (forced)
    dade = {('lon', c): [] for c in range(NUM_LON)}
    dade.update({('lat', c): [] for c in range(NUM_LAT)})

    for batch in loader:
        inputs, ego_future, _, ref_path = read_batch(batch, dev)
        enc = gameformer.encoder(inputs)
        top1, nbr_states, _ = extract_neighbor_top1_futures(gameformer, enc, args.num_neighbors)
        out = causal(enc, inputs, num_agents=args.num_neighbors + 1,
                     neighbor_futures=top1, neighbor_states=nbr_states)
        B = ego_future.shape[0]
        f_cas, ego_clean = out['f_cas'], out['ego_clean']
        b_lon = out['psi_lon_cas'].argmax(-1)                        # [B]
        b_lat = out['psi_lat_cas'].argmax(-1)

        traj0 = out['traj'][:, 0]                                    # [B,M,80,4]
        best0 = out['score'][:, 0].argmax(-1)
        plan0 = traj0[torch.arange(B), best0][..., :2]               # [B,80,2]

        # --- 1. agreement: ilan edilen karar vs planin yeniden etiketi ---
        rl_lon, rl_lat = decision_labels(plan_with_heading(plan0).cpu(), ref_path.cpu())
        rl_lon, rl_lat = rl_lon.to(dev), rl_lat.to(dev)
        agree_lon += int((rl_lon == b_lon).sum())
        agree_lat += int((rl_lat == b_lat).sum())
        agree_joint += int(((rl_lon == b_lon) & (rl_lat == b_lat)).sum())
        v0 = end_speed(plan0)

        # --- 2. swap: her lon sinifini zorla (lat = ilan edilen), sonra her lat sinifini ---
        for c in range(NUM_LON):
            fc = torch.full_like(b_lon, c)
            trajF, scoreF = causal.head(f_cas, ego_clean, (fc, b_lat))
            bestF = scoreF[:, 0].argmax(-1)
            planF = trajF[:, 0][torch.arange(B), bestF][..., :2]
            rlF, _ = decision_labels(plan_with_heading(planF).cpu(), ref_path.cpu())
            mask = (b_lon != c)                                      # yalniz gercek zorlamalar
            idx = mask.nonzero().flatten()
            for i in idx.tolist():
                C_lon[c, int(rlF[i])] += 1
            lon_forced_n[c] += int(mask.sum())
            dv[c] += (end_speed(planF) - v0)[mask].tolist()
            dade[('lon', c)] += (planF - plan0).norm(dim=-1).mean(-1)[mask].tolist()
        for c in range(NUM_LAT):
            fc = torch.full_like(b_lat, c)
            trajF, scoreF = causal.head(f_cas, ego_clean, (b_lon, fc))
            bestF = scoreF[:, 0].argmax(-1)
            planF = trajF[:, 0][torch.arange(B), bestF][..., :2]
            _, rlF = decision_labels(plan_with_heading(planF).cpu(), ref_path.cpu())
            mask = (b_lat != c)
            idx = mask.nonzero().flatten()
            for i in idx.tolist():
                C_lat[c, int(rlF[i])] += 1
            lat_forced_n[c] += int(mask.sum())
            dhd[c] += net_heading(planF)[mask].tolist()
            dade[('lat', c)] += (planF - plan0).norm(dim=-1).mean(-1)[mask].tolist()

        n += B
        if args.limit and n >= args.limit:
            break

    print(f"\n=== b*-swap — {args.causal_path.split('/')[-2]} ({n} sahne) ===")
    print("\n--- 1. AGREEMENT (ilan edilen karar vs planin yeniden etiketi) ---")
    print(f"  lon: {100.0 * agree_lon / n:.1f}%   lat: {100.0 * agree_lat / n:.1f}%   "
          f"joint: {100.0 * agree_joint / n:.1f}%")

    def fam_comp(C, forced_n, classes, family):
        rows = []
        for c in range(len(classes)):
            tot = forced_n[c]
            if tot == 0:
                continue
            exact = C[c, c] / tot
            fam = sum(C[c, r] for r in range(len(classes)) if family[r] == family[c]) / tot
            rows.append((classes[c], tot, exact, fam))
        return rows

    print("\n--- 2. COMPLIANCE (zorlanan sinif -> zorlanmis planin etiketi; zorlanan != ilan) ---")
    print(f"  {'zorlanan (lon)':16s} {'n':>6s} {'tam':>7s} {'aile':>7s}   ek: uc-hiz farki (m/s), Δplan (m)")
    for name, tot, ex, fa in fam_comp(C_lon, lon_forced_n, LON_CLASSES, LON_FAMILY):
        c = LON_CLASSES.index(name)
        mdv = np.mean(dv[c]) if dv[c] else 0.0
        mad = np.mean(dade[('lon', c)]) if dade[('lon', c)] else 0.0
        print(f"  {name:16s} {tot:6d} {100*ex:6.1f}% {100*fa:6.1f}%   Δv_end={mdv:+.2f}  Δplan={mad:.2f}")
    print(f"\n  {'zorlanan (lat)':16s} {'n':>6s} {'tam':>7s} {'aile':>7s}   ek: net heading (rad), Δplan (m)")
    for name, tot, ex, fa in fam_comp(C_lat, lat_forced_n, LAT_CLASSES, LAT_FAMILY):
        c = LAT_CLASSES.index(name)
        mhd = np.mean(dhd[c]) if dhd[c] else 0.0
        mad = np.mean(dade[('lat', c)]) if dade[('lat', c)] else 0.0
        print(f"  {name:16s} {tot:6d} {100*ex:6.1f}% {100*fa:6.1f}%   hd={mhd:+.2f}  Δplan={mad:.2f}")

    # doz-yon ozetleri
    slow_ok = np.mean([x < 0 for c in (1, 2, 3, 4) for x in dv[c]]) if any(dv[c] for c in (1, 2, 3, 4)) else 0
    acc_ok = np.mean([x > 0 for c in (5, 6) for x in dv[c]]) if any(dv[c] for c in (5, 6)) else 0
    tl_ok = np.mean([x > 0 for x in dhd[0]]) if dhd[0] else 0
    tr_ok = np.mean([x < 0 for x in dhd[1]]) if dhd[1] else 0
    print("\n--- 3. DOZ-YON (zorlamaya SEMANTIK dogru tepki orani) ---")
    print(f"  stop/slow zorla -> hiz dusuyor : {100*slow_ok:.1f}%")
    print(f"  accel zorla     -> hiz artiyor : {100*acc_ok:.1f}%")
    print(f"  turn_left zorla -> sola donuyor: {100*tl_ok:.1f}%")
    print(f"  turn_right zorla-> saga donuyor: {100*tr_ok:.1f}%")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="b*-swap: karar slotunun yuk-tasima olcumu")
    p.add_argument("--pretrained_path", required=True)
    p.add_argument("--causal_path", required=True)
    p.add_argument("--valid_set", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--num_neighbors", type=int, default=10)
    p.add_argument("--graph_layers", type=int, default=1)
    p.add_argument("--nbr_enrich", type=int, default=0)
    p.add_argument("--gate_channels", type=int, default=0)
    p.add_argument("--typed_kv", type=int, default=0)
    p.add_argument("--channel_evidence", type=int, default=0)
    p.add_argument("--gate_trust", type=str, default="all", choices=["all", "reliable"])
    p.add_argument("--rel_bottleneck", type=int, default=0)
    p.add_argument("--dod_meta", type=int, default=1)
    p.add_argument("--lon_merge", type=int, default=0)
    p.add_argument("--ego_residual", type=int, default=1)
    p.add_argument("--gate", type=str, default="softmax", choices=["softmax", "sigmoid"])
    p.add_argument("--modes", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", type=str, default="cuda")
    main(p.parse_args())
