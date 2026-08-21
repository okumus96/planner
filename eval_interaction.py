"""Step A -- M_cas'in SECTIGI ajan gercekten ego ile ETKILESIYOR mu?

RemoveNonCausal'in yapisal zaafi (bkz MASK_LOSS_INVESTIGATION.md): f_cas konveks bir
kombinasyon oldugu icin M_cas~0 olan ajan zaten toplamda YOK -> onu cikarmak plani
degistirmez. Yani Delta_low~0 keskin her maske icin GARANTI, secimin dogrulugunu
kanitlamaz. Bu script modelin kendi kafasindan tamamen BAGIMSIZ bir olcut kullanir:
SADECE GT gelecekler.

Secilen ajan j icin:
  d0     : t=0'da ego'ya mesafe (m)
  dmin   : min_t || ego_gt(t) - nbr_gt(j,t) ||  -- AYNI ANDA en yakin yaklasma (m)
  tmin   : o anin zamani (s)
  path   : min_{t,s} || ego_gt(t) - nbr_gt(j,s) ||  -- yollar kesisiyor mu (zamandan bagimsiz)

UC REFERANS ile kiyaslanir (kritik nokta):
  top1   : M_cas'in secdigi ajan
  random : rastgele gecerli ajan            -> SANS seviyesi
  near   : t=0'da EN YAKIN ajan             -> saf mesafe sezgisi

Okuma:
  top1(dmin) ~ random(dmin)  -> maske etkilesim hakkinda HICBIR sey bilmiyor
  top1(dmin) ~ near(dmin)    -> maske sadece mesafeyi tekrar ediyor
  near < top1 < random       -> arada; mesafeden iyi ama etkilesimi tam yakalamiyor
  top1 < near                -> maske mesafenin OTESINDE etkilesim biliyor  (istenen)
"""

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from GameFormer.predictor import GameFormer
from GameFormer.causal_graph import CausalPlanner
from GameFormer.train_utils import DrivingData
from train_planner import read_batch, extract_neighbor_top1_futures, freeze_gameformer


def _pairwise_min(ego_xy, nbr_xy, nbr_t_valid):
    """ego_xy [T,2], nbr_xy [T,2], nbr_t_valid [T] -> (dmin_sametime, tmin, dmin_path).
    dmin_sametime: ayni t'de en yakin yaklasma. dmin_path: zamandan bagimsiz yol yakinligi."""
    v = nbr_t_valid
    if v.sum() < 2:
        return np.nan, np.nan, np.nan
    e, n = ego_xy[v], nbr_xy[v]
    same = np.linalg.norm(e - n, axis=-1)                       # [Tv]
    i = int(np.argmin(same))
    # yollar arasi en kucuk mesafe (her ego noktasi x her komsu noktasi)
    cross = np.linalg.norm(ego_xy[:, None, :] - n[None, :, :], axis=-1)
    return float(same[i]), float(np.where(v)[0][i] * 0.1), float(cross.min())


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description="Interaction check for the selected causal agent")
    p.add_argument("--pretrained_path", required=True)
    p.add_argument("--causal_path", required=True)
    p.add_argument("--valid_set", required=True)
    p.add_argument("--num_neighbors", type=int, default=10)
    p.add_argument("--encoder_layers", type=int, default=3)
    p.add_argument("--decoder_levels", type=int, default=2)
    p.add_argument("--graph_layers", type=int, default=1)
    p.add_argument("--nbr_enrich", type=int, default=0)
    p.add_argument("--gate_channels", type=int, default=0)
    p.add_argument("--typed_kv", type=int, default=0)
    p.add_argument("--channel_evidence", type=int, default=0)
    p.add_argument("--gate_trust", type=str, default="all", choices=["all", "reliable"])
    p.add_argument("--rel_bottleneck", type=int, default=0)
    p.add_argument("--dod_meta", type=int, default=0)
    p.add_argument("--lon_merge", type=int, default=0)
    p.add_argument("--ego_residual", type=int, default=1,
                   help="checkpoint hangi degerle EGITILDIYSE o verilmeli (0 = h_ego residual yok)")
    p.add_argument("--gate", type=str, default="softmax", choices=["softmax", "sigmoid"])
    p.add_argument("--modes", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_batches", type=int, default=0, help="0 = tum validation")
    p.add_argument("--conflict_thr", type=float, default=5.0, help="dmin bu esigin altinda ise 'etkilesiyor'")
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    dev = torch.device(args.device)
    gameformer = GameFormer(encoder_layers=args.encoder_layers, decoder_levels=args.decoder_levels,
                            neighbors=args.num_neighbors).to(dev)
    gameformer.load_state_dict(torch.load(args.pretrained_path, map_location=dev))
    freeze_gameformer(gameformer)

    causal = CausalPlanner(layers=args.graph_layers, modes=args.modes,
                           nbr_enrich=args.nbr_enrich, gate=args.gate, ego_residual=args.ego_residual, gate_channels=args.gate_channels, typed_kv=args.typed_kv, channel_evidence=args.channel_evidence, gate_trust=args.gate_trust, rel_bottleneck=args.rel_bottleneck, dod_meta=args.dod_meta, num_lon=(6 if args.lon_merge else 9)).to(dev)
    missing, unexpected = causal.load_state_dict(torch.load(args.causal_path, map_location=dev), strict=False)
    if missing or unexpected:
        print(f"[load] missing={list(missing)}  unexpected={list(unexpected)}")
    causal.eval()

    loader = DataLoader(DrivingData(args.valid_set + "/*.npz", args.num_neighbors),
                        batch_size=args.batch_size, shuffle=False, num_workers=4)
    rng = np.random.default_rng(0)
    N = args.num_neighbors
    acc = {k: {m: [] for m in ("d0", "dmin", "tmin", "path", "behind")} for k in ("top1", "random", "near")}
    mcas_sel = []

    for bi, batch in enumerate(loader):
        if args.num_batches and bi >= args.num_batches:
            break
        inputs, ego_future, nbr_future, _ = read_batch(batch, dev)
        enc = gameformer.encoder(inputs)
        top1_fut, nbr_states, _ = extract_neighbor_top1_futures(gameformer, enc, args.num_neighbors)
        out = causal(enc, inputs, num_agents=args.num_neighbors + 1,
                     neighbor_futures=top1_fut, neighbor_states=nbr_states)

        M = out['M_cas'][:, :N].cpu().numpy()
        valid = out['nbr_valid'][:, :N].cpu().numpy()
        pos0 = enc['actors'][:, 1:1 + N, -1, :2].cpu().numpy()          # [B,N,2] ego-frame, t=0
        ego_gt = ego_future[:, :, :2].cpu().numpy()                      # [B,80,2]
        nbr_gt = nbr_future[:, :N, :, :2].cpu().numpy()                  # [B,N,80,2]
        t_valid = (np.abs(nbr_gt).sum(-1) > 1e-6)                        # [B,N,80]

        for b in range(M.shape[0]):
            vj = np.where(valid[b] & (t_valid[b].sum(-1) >= 2))[0]       # GT gelecegi olan gecerli komsular
            if len(vj) < 2:
                continue
            rng_d = np.linalg.norm(pos0[b, vj], axis=-1)
            picks = {
                "top1": int(vj[np.argmax(M[b, vj])]),
                "random": int(rng.choice(vj)),
                "near": int(vj[np.argmin(rng_d)]),
            }
            mcas_sel.append(float(M[b, picks["top1"]]))
            for k, j in picks.items():
                dmin, tmin, path = _pairwise_min(ego_gt[b], nbr_gt[b, j], t_valid[b, j])
                if np.isnan(dmin):
                    continue
                acc[k]["d0"].append(float(np.linalg.norm(pos0[b, j])))
                # ego-frame'de +x ileri: x<0 -> ajan ego'nun ARKASINDA
                acc[k]["behind"].append(float(pos0[b, j, 0] < 0))
                acc[k]["dmin"].append(dmin)
                acc[k]["tmin"].append(tmin)
                acc[k]["path"].append(path)

    n = len(acc["top1"]["dmin"])
    name = args.causal_path.split('/')[-2]
    print(f"\n=== Interaction check — {name} ({n} sahne) ===")
    print(f"  secilen ajanin ortalama M_cas: {np.mean(mcas_sel):.3f}\n")
    hdr = f"{'':10s}{'d0 (m)':>18s}{'dmin (m)':>18s}{'path (m)':>18s}{'tmin (s)':>10s}{'dmin<%.0fm':>10s}{'ARKADA':>9s}" % args.conflict_thr
    print(hdr); print("-" * len(hdr))
    for k in ("top1", "near", "random"):
        a = acc[k]
        frac = float(np.mean(np.array(a["dmin"]) < args.conflict_thr))
        print(f"{k:10s}"
              f"{np.mean(a['d0']):9.2f} (med {np.median(a['d0']):5.1f})"
              f"{np.mean(a['dmin']):9.2f} (med {np.median(a['dmin']):5.1f})"
              f"{np.mean(a['path']):9.2f} (med {np.median(a['path']):5.1f})"
              f"{np.mean(a['tmin']):10.2f}{frac:10.3f}{np.mean(a['behind']):9.3f}")

    t, r, nr = np.array(acc["top1"]["dmin"]), np.array(acc["random"]["dmin"]), np.array(acc["near"]["dmin"])
    print(f"\n  top1 vs random : {np.mean(r) / max(np.mean(t), 1e-6):.2f}x daha yakin "
          f"(1.0 = sans seviyesi, maske etkilesim bilmiyor)")
    print(f"  top1 vs near   : {np.mean(nr) / max(np.mean(t), 1e-6):.2f}x  "
          f"(<1.0 = maske mesafeden KOTU, ~1.0 = mesafeyi tekrar ediyor, >1.0 = mesafenin OTESI)")
    print(f"  top1 == near secim orani: "
          f"{np.mean(np.abs(np.array(acc['top1']['d0']) - np.array(acc['near']['d0'])) < 1e-6):.3f}")


if __name__ == "__main__":
    main()
