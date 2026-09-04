"""P(b* | REJIM): predicate rejimleri karar dagilimini ayirt ediyor mu? (okuma-amacli, egitimsiz)

NEDEN: out_fc_cas'i rejim-anahtarli expert'lere bolmek ancak rejimler FARKLI karar dagilimlari
tasiyorsa ise yarar (lat_moe'nin calismasinin sebebi de buydu: her expert kendi sinifinin
sahnelerini gorup FARKLI bir cikti dagilimi ogrendi). Dagilimlar ayni cikarsa expert'lerin
uzmanlasacak bir seyi yok -> ayni matrise yakinsarlar -> bosuna egitim.

REJIM: sahnenin nedensel dikkat kutlesi ILISKI GRUPLARINA toplanir, argmax = rejim.
  lead        = same_lane_ahead + follows
  conflict    = collision_course + sharesIntersection + vru
  lateral     = adjacent_left + adjacent_right + overtakes + merges
  background  = same_lane_behind + near

OLCUM: her rejim icin lon ve lat karar dagilimi AYRI AYRI; hem MODELIN ilan ettigi b* hem de
GT etiketi. Ozet istatistik = toplam varyasyon (TV) uzakligi:
  TV(p,q) = 0.5 * sum |p_i - q_i|;  0 = ayni dagilim, 1 = tamamen ayrik.
Rejim-marjinal TV kucukse (< ~0.05) rejim karar hakkinda hicbir sey soylemiyor demektir.
"""
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from GameFormer.predictor import GameFormer
from GameFormer.causal_graph import CausalPlanner
from GameFormer.train_utils import DrivingData
from GameFormer.channels import CHANNEL_NAMES
from GameFormer.decision_labels import (NUM_LON4, NUM_LAT5V, NUM_LON5, NUM_LAT5,
                                        LON4_MAP, LAT5V_MAP, LON4_CLASSES, LAT5V_CLASSES)
from train_planner import read_batch, extract_neighbor_top1_futures, freeze_gameformer

REGIMES = {
    "lead":       ["same_lane_ahead", "follows"],
    "conflict":   ["onObservedCollisionCourseWith", "sharesIntersectionWith",
                   "vulnerable_road_user_near_ego_path"],
    "lateral":    ["adjacent_left", "adjacent_right", "overtakes", "merges"],
    "background": ["same_lane_behind", "near"],
}
RNAMES = list(REGIMES)


def _tv(p, q):
    return 0.5 * float(np.abs(p - q).sum())


def _report(title, counts, cls_names):
    """counts [n_regime, n_class] -> satir-normalize dagilim tablosu + TV ozetleri."""
    tot = counts.sum(1, keepdims=True)
    P = counts / np.maximum(tot, 1)
    marg = counts.sum(0) / max(counts.sum(), 1)
    w = (counts.sum(1) / max(counts.sum(), 1))
    print(f"\n--- {title} ---")
    hdr = "  " + f"{'rejim':12s}" + f"{'n':>7s}" + "".join(f"{c[:11]:>13s}" for c in cls_names) + f"{'TV(marj)':>10s}"
    print(hdr)
    for i, rn in enumerate(RNAMES):
        if tot[i, 0] == 0:
            print(f"  {rn:12s}{0:7d}   (bu rejimde sahne yok)")
            continue
        row = "".join(f"{100*P[i, j]:12.1f}%" for j in range(len(cls_names)))
        print(f"  {rn:12s}{int(tot[i,0]):7d}{row}{_tv(P[i], marg):10.3f}")
    row = "".join(f"{100*marg[j]:12.1f}%" for j in range(len(cls_names)))
    print(f"  {'MARJINAL':12s}{int(counts.sum()):7d}{row}")
    ok = [i for i in range(len(RNAMES)) if tot[i, 0] > 0]
    pw = [(_tv(P[i], P[j]), RNAMES[i], RNAMES[j]) for a, i in enumerate(ok) for j in ok[a+1:]]
    wtv = sum(w[i] * _tv(P[i], marg) for i in ok)
    if pw:
        m = max(pw)
        print(f"  agirlikli ort. TV(rejim, marjinal) = {wtv:.3f}   |   "
              f"en buyuk ikili TV = {m[0]:.3f} ({m[1]} vs {m[2]})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_path", required=True)
    p.add_argument("--causal_path", required=True)
    p.add_argument("--valid_set", required=True)
    p.add_argument("--num_neighbors", type=int, default=10)
    p.add_argument("--encoder_layers", type=int, default=3)
    p.add_argument("--decoder_levels", type=int, default=2)
    p.add_argument("--graph_layers", type=int, default=1)
    p.add_argument("--nbr_enrich", type=int, default=0)
    p.add_argument("--ego_residual", type=int, default=1)
    p.add_argument("--gate", type=str, default="softmax")
    p.add_argument("--gate_channels", type=int, default=0)
    p.add_argument("--typed_kv", type=int, default=0)
    p.add_argument("--channel_evidence", type=int, default=0)
    p.add_argument("--gate_trust", type=str, default="all")
    p.add_argument("--dod_meta", type=int, default=1)
    p.add_argument("--dec_moe", type=int, default=0)
    p.add_argument("--lat_moe", type=int, default=0)
    p.add_argument("--lon_merge", type=int, default=0)
    p.add_argument("--modes", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--regime_src", type=str, default="mass", choices=["mass", "count"],
                   help="rejim: dikkat KUTLESI argmax'i (router'in okuyacagi) veya ham YANMA sayisi")
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()
    assert args.lat_moe, "bu script 4x5 (lat_moe) ckpt icin yazildi"
    dev = torch.device(args.device)

    gameformer = GameFormer(encoder_layers=args.encoder_layers, decoder_levels=args.decoder_levels,
                            modalities=args.modes, neighbors=args.num_neighbors).to(dev)
    gameformer.load_state_dict(torch.load(args.pretrained_path, map_location=dev))
    freeze_gameformer(gameformer); gameformer.eval()
    causal = CausalPlanner(layers=args.graph_layers, modes=args.modes, nbr_enrich=args.nbr_enrich,
                           gate=args.gate, ego_residual=args.ego_residual,
                           gate_channels=args.gate_channels, typed_kv=args.typed_kv,
                           channel_evidence=args.channel_evidence, gate_trust=args.gate_trust,
                           dod_meta=args.dod_meta, dec_moe=args.dec_moe, lat_moe=args.lat_moe,
                           num_lon=NUM_LON4, num_lat=NUM_LAT5V).to(dev)
    miss, unexp = causal.load_state_dict(torch.load(args.causal_path, map_location=dev), strict=False)
    print(f"[load] missing={list(miss) or 'NONE'}  unexpected={list(unexp) or 'NONE'}")
    causal.eval()

    gidx = [[CHANNEL_NAMES.index(n) for n in v] for v in REGIMES.values()]
    lon_map = torch.tensor(LON4_MAP, dtype=torch.long, device=dev)
    lat_map = torch.tensor(LAT5V_MAP, dtype=torch.long, device=dev)
    Cm_lon = np.zeros((len(RNAMES), NUM_LON4), dtype=np.int64)
    Cm_lat = np.zeros((len(RNAMES), NUM_LAT5V), dtype=np.int64)
    Cg_lon = np.zeros((len(RNAMES), NUM_LON4), dtype=np.int64)
    Cg_lat = np.zeros((len(RNAMES), NUM_LAT5V), dtype=np.int64)
    n_skip = 0
    P_all, F_all, GL_all, GT_all = [], [], [], []

    ds = DrivingData(args.valid_set + "/*.npz", args.num_neighbors)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    with torch.no_grad():
        for batch in loader:
            inputs, ego_future, _, ref_path = read_batch(batch, dev)
            enc = gameformer.encoder(inputs)
            top1, nbr_states, _ = extract_neighbor_top1_futures(gameformer, enc, args.num_neighbors)
            out = causal(enc, inputs, num_agents=args.num_neighbors + 1,
                         neighbor_futures=top1, neighbor_states=nbr_states, ref_path=ref_path)
            if args.regime_src == "mass":
                prof = out["M_cas_typed"].sum(1)                        # [B,R] ajanlar uzerinden
            else:
                prof = out["ch_active"].float().sum(1)                  # [B,R] ham yanma sayisi
            grp = torch.stack([prof[:, ix].sum(-1) for ix in gidx], dim=-1)   # [B,4]
            alive = grp.sum(-1) > 1e-6
            n_skip += int((~alive).sum())
            reg = grp.argmax(-1)                                        # [B]
            m_lon = out["psi_lon_cas"].argmax(-1)
            m_lat = out["psi_lat_cas"].argmax(-1)
            g_lon = lon_map[inputs["decision_lon"]]
            g_lat = lat_map[inputs["decision_lat"]]
            P_all.append(prof.float().cpu()); F_all.append(out["f_cas"].float().cpu())
            GL_all.append(g_lon.cpu()); GT_all.append(g_lat.cpu())
            for b in range(reg.shape[0]):
                if not alive[b]:
                    continue
                r = int(reg[b])
                Cm_lon[r, int(m_lon[b])] += 1; Cm_lat[r, int(m_lat[b])] += 1
                Cg_lon[r, int(g_lon[b])] += 1; Cg_lat[r, int(g_lat[b])] += 1

    # --- EK OLCUM: profil (11 sayi) ve f_cas (256 sayi) karari NE KADAR ongoruyor? ---
    # Rejim argmax'i KABA bir ozet (4 kova). Profilin kendisi surekli ve 11 boyutlu; karar
    # sinyali tasiyip tasimadigini dogrudan probe ile olcelim. Taban = en sik sinifin orani.
    import torch.nn as nn

    def _probe(X, y, ncls, name):
        g = torch.Generator().manual_seed(0)
        perm = torch.randperm(X.shape[0], generator=g).to(X.device)
        ntr = int(0.8 * perm.numel()); tr, te = perm[:ntr], perm[ntr:]
        mu, sd = X[tr].mean(0), X[tr].std(0).clamp(min=1e-6)
        Xtr, Xte, ytr, yte = (X[tr]-mu)/sd, (X[te]-mu)/sd, y[tr], y[te]
        lin = nn.Linear(X.shape[1], ncls).to(X.device)
        opt = torch.optim.Adam(lin.parameters(), lr=1e-2, weight_decay=1e-4)
        for _ in range(400):
            opt.zero_grad(); nn.functional.cross_entropy(lin(Xtr), ytr).backward(); opt.step()
        with torch.no_grad():
            acc = (lin(Xte).argmax(-1) == yte).float().mean().item()
        base = torch.bincount(yte, minlength=ncls).max().item() / yte.numel()
        print(f"  {name:32s} acc={acc:6.3f}  taban={base:.3f}  fark={acc-base:+.3f}")

    PR = torch.cat(P_all).to(dev); FC = torch.cat(F_all).to(dev)
    GL = torch.cat(GL_all).to(dev); GT_ = torch.cat(GT_all).to(dev)
    print(f"\n=== PROBE: karar ONGORULEBILIR mi? (n={PR.shape[0]}, %80/%20) ===")
    _probe(PR, GL, NUM_LON4,  "profil(11) -> GT lon")
    _probe(FC, GL, NUM_LON4,  "f_cas(256) -> GT lon")
    _probe(PR, GT_, NUM_LAT5V, "profil(11) -> GT lat")
    _probe(FC, GT_, NUM_LAT5V, "f_cas(256) -> GT lat")

    print(f"\nrejim kaynagi = {args.regime_src};  hicbir iliski yanmayan (atlanan) sahne = {n_skip}")
    _report("MODEL b*  |  LON", Cm_lon, LON4_CLASSES)
    _report("MODEL b*  |  LAT", Cm_lat, LAT5V_CLASSES)
    _report("GT etiket |  LON", Cg_lon, LON4_CLASSES)
    _report("GT etiket |  LAT", Cg_lat, LAT5V_CLASSES)


if __name__ == "__main__":
    main()
