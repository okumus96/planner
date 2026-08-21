"""Sahne CF — kenar-duzeyi mudahale: do(relation = off).

Typed causal graph'ta ego->ajan kenarlari (ajan, iliski) ciftleri. Bir iliskiyi kapatmak,
o tipteki TUM girisleri softmax'tan cikarmak demek: ajan sahnede kalir (baska iliskileri
varsa), o iliski uzerinden bilgi akisi kesilir. Literaturdeki removal testleri DUGUM siler;
bu KENAR siler -- ancak kenarlarin kimligi (tipi) oldugu icin mumkun.

Olculen (yalniz iliskinin YANDIGI sahnelerde):
  - karar degisim orani (lon / lat)
  - fren -> fren-degil gecis orani (semantik yon: fren iliskisini kaldirinca fren dusmeli)
  - uc-hiz farki (forced - base), + = model hizlaniyor
  - Delta plan (L2)
Kontrol: alakasiz iliskiler (ornegin overtakes) ~0 vermeli.
Ayrisim: 'saf kenar' = kaldirilan iliskiyi tasiyan ajanlarin HEPSI cok-iliskili (ajan sahnede
kalir); 'dugume donusen' = en az bir ajan tek-iliskiliydi (o ajan grafikten tamamen duser).
"""
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from GameFormer.predictor import GameFormer
from GameFormer.causal_graph import CausalPlanner
from GameFormer.train_utils import DrivingData
from GameFormer.channels import CHANNEL_NAMES
from GameFormer.decision_labels import LON_CLASSES
from train_planner import read_batch, extract_neighbor_top1_futures, freeze_gameformer

BRAKE = {LON_CLASSES.index(c) for c in
         ['stop_quickly', 'stop_gently', 'slow_quickly', 'slow_gently', 'remain_stopped']}


def best_plan(out, B):
    traj = out['traj'][:, 0]
    best = out['score'][:, 0].argmax(-1)
    return traj[torch.arange(B, device=traj.device), best][..., :2]


def end_speed(xy):
    v = (xy[:, 1:] - xy[:, :-1]).norm(dim=-1) / 0.1
    return v[:, 34:40].mean(1)


@torch.no_grad()
def main(a):
    dev = a.device
    gf = GameFormer(encoder_layers=3, decoder_levels=2, neighbors=a.num_neighbors)
    gf.load_state_dict(torch.load(a.pretrained_path, map_location=dev))
    gf = gf.to(dev); freeze_gameformer(gf)
    m = CausalPlanner(layers=a.graph_layers, modes=a.modes, nbr_enrich=a.nbr_enrich,
                      ego_residual=a.ego_residual, gate_channels=1, typed_kv=a.typed_kv,
                      dod_meta=a.dod_meta, num_lon=(6 if a.lon_merge else 9),
                      rel_bottleneck=a.rel_bottleneck).to(dev)
    miss, unexp = m.load_state_dict(torch.load(a.causal_path, map_location=dev), strict=False)
    if miss or unexp:
        print(f"[load] missing={list(miss)} unexpected={list(unexp)}")
    m.eval()
    ld = DataLoader(DrivingData(a.valid_set + "/*.npz", a.num_neighbors),
                    batch_size=a.batch_size, shuffle=False, num_workers=4)

    R = len(CHANNEL_NAMES)
    acc = {r: dict(n=0, flip_lon=0, flip_lat=0, unbrake=0, n_brake=0, dv=[], dp=[],
                   n_pure=0, flip_pure=0, hi_n=0, lo_n=0, hi_flip=0, lo_flip=0,
                   hi_dv=[], lo_dv=[], hi_dp=[], lo_dp=[], share=[]) for r in range(R)}
    for batch in ld:
        inp, ef, _, rp = read_batch(batch, dev)
        if "channel_active" not in inp:
            raise SystemExit("cache'te kanal yok — extract_channels calistirilmali")
        enc = gf.encoder(inp)                                   # kanallardan BAGIMSIZ, bir kez
        t1, ns, _ = extract_neighbor_top1_futures(gf, enc, a.num_neighbors)
        base = m(enc, inp, num_agents=a.num_neighbors + 1, neighbor_futures=t1, neighbor_states=ns)
        B = ef.shape[0]
        p0 = best_plan(base, B); v0 = end_speed(p0)
        l0 = base['psi_lon_cas'].argmax(-1); a0 = base['psi_lat_cas'].argmax(-1)
        ch = inp["channel_active"]                               # [B,N,R] bool
        nv = base['nbr_valid']
        mt = base['M_cas_typed']                                 # [B,N,R]
        share = mt.sum(1) / mt.sum((1, 2), keepdim=True).squeeze(1).clamp(min=1e-6)  # [B,R]
        for r in range(R):
            has_r = ch[..., r] & nv                              # [B,N]
            fired = has_r.any(-1)                                # [B]
            if not fired.any():
                continue
            # saf kenar: r'yi tasiyan ajanlarin hepsi cok-iliskili -> ajan sahnede kalir
            multi = (ch.sum(-1) > 1)
            pure = fired & (~(has_r & ~multi)).all(-1)
            ch2 = ch.clone(); ch2[..., r] = False
            inp2 = dict(inp); inp2["channel_active"] = ch2
            o2 = m(enc, inp2, num_agents=a.num_neighbors + 1, neighbor_futures=t1, neighbor_states=ns)
            p2 = best_plan(o2, B); v2 = end_speed(p2)
            l2 = o2['psi_lon_cas'].argmax(-1); a2 = o2['psi_lat_cas'].argmax(-1)
            f = fired
            sr = share[:, r]
            thr = sr[fired].median() if fired.sum() > 1 else sr.max()
            hi = fired & (sr >= thr); lo = fired & (sr < thr)
            d = acc[r]
            d['hi_n'] += int(hi.sum()); d['lo_n'] += int(lo.sum())
            d['hi_flip'] += int(((l2 != l0) & hi).sum()); d['lo_flip'] += int(((l2 != l0) & lo).sum())
            d['hi_dv'] += (v2 - v0)[hi].tolist(); d['lo_dv'] += (v2 - v0)[lo].tolist()
            d['hi_dp'] += (p2 - p0).norm(dim=-1).mean(-1)[hi].tolist()
            d['lo_dp'] += (p2 - p0).norm(dim=-1).mean(-1)[lo].tolist()
            d['share'] += sr[fired].tolist()
            d['n'] += int(f.sum())
            d['flip_lon'] += int(((l2 != l0) & f).sum())
            d['flip_lat'] += int(((a2 != a0) & f).sum())
            br = f & torch.tensor([int(x) in BRAKE for x in l0.tolist()], device=dev)
            d['n_brake'] += int(br.sum())
            unbr = br & torch.tensor([int(x) not in BRAKE for x in l2.tolist()], device=dev)
            d['unbrake'] += int(unbr.sum())
            d['dv'] += (v2 - v0)[f].tolist()
            d['dp'] += (p2 - p0).norm(dim=-1).mean(-1)[f].tolist()
            d['n_pure'] += int(pure.sum())
            d['flip_pure'] += int(((l2 != l0) & pure).sum())

    print(f"\n=== Sahne CF — KUTLE-KOSULLU doz-cevap — {a.causal_path.split('/')[-2]} ===")
    print("  (her iliski icin: o iliskinin causal kutle payi YUKSEK vs DUSUK sahneler)")
    print(f"{'iliski':26s}{'pay%':>7s}{'n_hi':>6s}{'flip_hi':>9s}{'dv_hi':>8s}{'dp_hi':>7s}"
          f"{'n_lo':>7s}{'flip_lo':>9s}{'dv_lo':>8s}{'dp_lo':>7s}")
    for r in range(R):
        d = acc[r]
        if d['hi_n'] == 0:
            continue
        print(f"{CHANNEL_NAMES[r][:25]:26s}{100*np.mean(d['share']):>7.1f}{d['hi_n']:>6d}"
              f"{100*d['hi_flip']/max(d['hi_n'],1):>8.1f}%{np.mean(d['hi_dv']):>+8.3f}"
              f"{np.mean(d['hi_dp']):>7.3f}{d['lo_n']:>7d}"
              f"{100*d['lo_flip']/max(d['lo_n'],1):>8.1f}%{np.mean(d['lo_dv']):>+8.3f}"
              f"{np.mean(d['lo_dp']):>7.3f}")
    print()
    print(f"{'iliski':26s}{'n':>6s}{'flipLON':>9s}{'flipLAT':>9s}{'unbrake':>9s}"
          f"{'dv_end':>9s}{'dplan':>8s}{'n_pure':>8s}{'flip_pure':>10s}")
    for r in range(R):
        d = acc[r]
        if d['n'] == 0:
            continue
        ub = 100 * d['unbrake'] / max(d['n_brake'], 1)
        print(f"{CHANNEL_NAMES[r][:25]:26s}{d['n']:>6d}"
              f"{100*d['flip_lon']/d['n']:>8.1f}%{100*d['flip_lat']/d['n']:>8.1f}%"
              f"{ub:>8.1f}%{np.mean(d['dv']):>+9.3f}{np.mean(d['dp']):>8.3f}"
              f"{d['n_pure']:>8d}{100*d['flip_pure']/max(d['n_pure'],1):>9.1f}%")
    print("\nflipLON/LAT: karar degisim orani | unbrake: fren kararindan cikma orani "
          "| dv_end: +ise model hizlaniyor | n_pure/flip_pure: saf kenar-silme alt kumesi")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_path", required=True)
    p.add_argument("--causal_path", required=True)
    p.add_argument("--valid_set", required=True)
    p.add_argument("--num_neighbors", type=int, default=10)
    p.add_argument("--graph_layers", type=int, default=1)
    p.add_argument("--nbr_enrich", type=int, default=2)
    p.add_argument("--typed_kv", type=int, default=1)
    p.add_argument("--dod_meta", type=int, default=1)
    p.add_argument("--rel_bottleneck", type=int, default=0)
    p.add_argument("--lon_merge", type=int, default=0)
    p.add_argument("--ego_residual", type=int, default=0)
    p.add_argument("--modes", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", type=str, default="cuda:1")
    main(p.parse_args())
