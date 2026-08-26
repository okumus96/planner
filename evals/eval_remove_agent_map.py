"""RemoveNonCausal — AJAN + HARITA birlesik. Hem komsu ajanlari hem harita polygonlarini cikarip
plan kaymasi (Delta) ile M_cas / M_cas_map korelasyonuna bakar. "Genel" guvenilirlik: yuksek-M bir
eleman (ajan VEYA harita) cikarilinca plan cok kaymali, dusuk-M cikarilinca kaymamali.

Ajan cikar = encoder mask'i kapat. Harita cikar = ilgili polyline'i sifirla (disentangler'da invalid).
"""
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from GameFormer.predictor import GameFormer
from GameFormer.causal_graph import CausalPlanner
from GameFormer.train_utils import DrivingData
from train_planner import read_batch, extract_neighbor_top1_futures, freeze_gameformer


@torch.no_grad()
def plan_of(causal, enc, inputs, Na, top1, states, best=None):
    out = causal(enc, inputs, num_agents=Na, neighbor_futures=top1, neighbor_states=states)
    traj = out['traj'][:, 0, :, :, :2]                    # [B,K,80,2]
    if best is None:
        best = out['score'][:, 0].argmax(-1)
    B = traj.shape[0]
    plan = traj[torch.arange(B, device=traj.device), best]
    return out, plan, best


@torch.no_grad()
def delta_remove(causal, enc, inputs, Na, top1, states, plan_base, best, kind, idx, rows, L, C):
    """kind='agent'|'map'; idx [B] cikarilacak eleman; rows [B] bool (hangi ornekler)."""
    B = plan_base.shape[0]
    if kind == 'agent':
        enc2 = dict(enc); m2 = enc['mask'].clone()
        rr = rows.nonzero().flatten()
        m2[rr, idx[rr] + 1] = True                        # komsu j -> sutun 1+j
        enc2['mask'] = m2
        out = causal(enc2, inputs, num_agents=Na, neighbor_futures=top1, neighbor_states=states)
    else:
        inp2 = dict(inputs)
        lanes = inputs['map_lanes'].clone(); cw = inputs['map_crosswalks'].clone(); rt = inputs['route_lanes'].clone()
        for r in rows.nonzero().flatten().tolist():
            e = int(idx[r].item())
            if e < L:            lanes[r, e] = 0
            elif e < L + C:      cw[r, e - L] = 0
            else:                rt[r, e - L - C] = 0
        inp2['map_lanes'] = lanes; inp2['map_crosswalks'] = cw; inp2['route_lanes'] = rt
        out = causal(enc, inp2, num_agents=Na, neighbor_futures=top1, neighbor_states=states)
    plan_i = out['traj'][:, 0, :, :, :2][torch.arange(B, device=plan_base.device), best]
    return torch.norm(plan_i - plan_base, dim=-1).mean(-1)  # [B] ADE


def main(args):
    dev = args.device
    gf = GameFormer(encoder_layers=args.encoder_layers, decoder_levels=args.decoder_levels, neighbors=args.num_neighbors)
    gf.load_state_dict(torch.load(args.pretrained_path, map_location=dev)); gf = gf.to(dev); freeze_gameformer(gf)
    causal = CausalPlanner(layers=args.graph_layers, modes=args.modes).to(dev)
    causal.load_state_dict(torch.load(args.causal_path, map_location=dev)); causal.eval()
    loader = DataLoader(DrivingData(args.valid_set + "/*.npz", args.num_neighbors), batch_size=args.batch_size,
                        shuffle=False, num_workers=4)

    pool = {'agent': ([], []), 'map': ([], [])}           # (M, Delta)
    Na = args.num_neighbors + 1
    for bi, batch in enumerate(loader):
        if bi >= args.num_batches:
            break
        inputs, _, _, _ = read_batch(batch, dev)
        B = inputs['ego_agent_past'].shape[0]
        enc = gf.encoder(inputs)
        top1, states, _ = extract_neighbor_top1_futures(gf, enc, args.num_neighbors)
        out, plan_base, best = plan_of(causal, enc, inputs, Na, top1, states)

        L = inputs['map_lanes'].shape[1]; C = inputs['map_crosswalks'].shape[1]
        specs = [
            ('agent', out['M_cas'],  out['nbr_valid']),
            ('map',   out['M_cas_map'], out['map_valid']),
        ]
        for kind, M, valid in specs:
            n_valid = valid.sum(-1)
            usable = n_valid >= 2
            big = M.masked_fill(~valid, -1e9); small = M.masked_fill(~valid, 1e9)
            rnd = torch.rand_like(M).masked_fill(~valid, -1.0)
            for sel in (big.argmax(-1), small.argmin(-1), rnd.argmax(-1)):
                d = delta_remove(causal, enc, inputs, Na, top1, states, plan_base, best, kind, sel, usable, L, C)
                mval = M.gather(1, sel[:, None]).squeeze(1)
                pool[kind][0].append(mval[usable].cpu().numpy())
                pool[kind][1].append(d[usable].cpu().numpy())

    def cat(x): return np.concatenate(x) if x else np.array([0.0])
    ma, da = cat(pool['agent'][0]), cat(pool['agent'][1])
    mm, dm = cat(pool['map'][0]), cat(pool['map'][1])
    mc, dc = np.concatenate([ma, mm]), np.concatenate([da, dm])

    def corr(m, d): return float(np.corrcoef(m, d)[0, 1]) if m.std() > 0 else 0.0
    print(f"\n=== RemoveNonCausal (AJAN + HARITA) — {args.causal_path.split('/')[-1]} ===")
    print(f"  AJAN  : corr(M_cas, Delta)      = {corr(ma, da):+.3f}   (n={len(ma)}, mean Delta high-uzeri hesaplanabilir)")
    print(f"  HARITA: corr(M_cas_map, Delta)  = {corr(mm, dm):+.3f}   (n={len(mm)})")
    print(f"  GENEL : corr(M, Delta) havuz    = {corr(mc, dc):+.3f}   (n={len(mc)}, ajan+harita birlikte)")
    print("\n  M araligina gore ortalama plan kaymasi (Delta, m):")
    for name, m, d in [('AJAN', ma, da), ('HARITA', mm, dm)]:
        print(f"   [{name}]")
        for lo, hi in [(0.0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 1.01)]:
            s = (m >= lo) & (m < hi)
            tag = f"M in [{lo:.1f},{hi:.1f})"
            print(f"     {tag:16s} n={int(s.sum()):5d}  Delta={d[s].mean():.4f}" if s.sum() else f"     {tag:16s} n=0")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="RemoveNonCausal for agents + map")
    p.add_argument("--pretrained_path", required=True)
    p.add_argument("--causal_path", required=True)
    p.add_argument("--valid_set", required=True)
    p.add_argument("--num_neighbors", type=int, default=10)
    p.add_argument("--encoder_layers", type=int, default=3)
    p.add_argument("--decoder_levels", type=int, default=2)
    p.add_argument("--graph_layers", type=int, default=3)
    p.add_argument("--modes", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_batches", type=int, default=15)
    p.add_argument("--device", type=str, default="cuda")
    main(p.parse_args())
