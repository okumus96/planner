"""L1 TEST-TIME INTERVENTION — CBM'in kendi olcutu.

Soru: konsept katmanina ELLE mudahale edince karar VE yorunge degisiyor mu?
Predicate maskesine degil, L1'in KENDISINE mudahale ediyoruz:

    resp_ag[j]'yi zorla sinif c yap  ->  ozet yeniden kurulur  ->  psi  ->  b*  ->  trajectory

Bu, "iz gercek mi yoksa yan cikti mi" sorusunun dogrudan testi. Iz gercekse konsepti
degistirmek plani degistirmeli; yan cikti ise plan aynen kalir (b*'in yok sayilmasi gibi).

Olculenler (hedef ajan = ahead yakanlar icinde en yuksek M_cas kutlesi):
  psi flip      : ilan edilen lon karari degisti mi
  dv_end        : plan uc-hizi farki [m/s]  (+ = hizlandi)
  dplan         : plan L2 farki [m]
Karsilastirma: ayni sahnede PREDICATE maskesine mudahale (eski test) ile yan yana.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader
from GameFormer.predictor import GameFormer
from GameFormer.causal_graph import CausalPlanner
from GameFormer.train_utils import DrivingData
from GameFormer.channels import CH_FOLLOWS, CH_SAME_LANE_AHEAD, CH_ADJACENT_RIGHT
from GameFormer.decision_labels import NUM_LON4, NUM_LAT5V, LON4_CLASSES
from train_planner import read_batch, extract_neighbor_top1_futures, freeze_gameformer

AHEAD = [CH_FOLLOWS, CH_SAME_LANE_AHEAD]
L1_NAMES = ['none', 'follows', 'yieldingTo', 'waitingFor', 'mergesInFrontOf', 'overtakes']


def end_speed(xy):
    v = (xy[:, 1:] - xy[:, :-1]).norm(dim=-1) / 0.1
    return v[:, 34:40].mean(1)


def best_plan(traj, score, B, dev):
    b = score[:, 0].argmax(-1)
    return traj[:, 0][torch.arange(B, device=dev), b][..., :2]


@torch.no_grad()
def run(m, enc, inp, t1, ns, force=None):
    """force = (agent_idx [B], class [B]) -> o ajanin L1 sinifi ZORLANIR."""
    dis = m.disentangler(enc['agent_tokens'][:, :11].detach(), ~enc['mask'][:, :11],
                         enc['actors'][:, :11, -1].detach(),
                         torch.zeros(enc['mask'].shape[0], 11, dtype=torch.long,
                                     device=enc['mask'].device), inp,
                         neighbor_futures=t1, neighbor_states=ns)
    B, dev = dis['f_cas'].shape[0], dis['f_cas'].device
    l1_ag = m.l1_head_ag(dis['src_cas_ag'])
    l1_mp = m.l1_head_mp(dis['src_cas_mp'])
    p_ag = torch.softmax(l1_ag, -1)
    if force is not None:
        j, c = force
        oh = torch.zeros_like(p_ag[torch.arange(B, device=dev), j])
        oh[torch.arange(B, device=dev), c] = 1.0            # SERT mudahale: tek-sicak
        p_ag = p_ag.clone()
        p_ag[torch.arange(B, device=dev), j] = oh
    va = (dis['gated_valid'].float() * dis['M_cas'])[..., None]
    vm = (dis['gated_map_valid'].float() * dis['M_cas_map'])[..., None]
    summ = torch.cat([(p_ag * va).sum(1), (torch.softmax(l1_mp, -1) * vm).sum(1)], -1)
    z = summ if m.l1_bottleneck else torch.cat([dis['f_cas'], summ], -1)
    lon, lat = m.psi_lon(z), m.psi_lat(z)
    b_star = (lon.argmax(-1), lat.argmax(-1))
    traj, score = m.head(dis['f_cas'], dis['ego_clean'], b_star)
    plan = best_plan(traj, score, B, dev)
    return dict(lon=lon.argmax(-1), P=torch.softmax(lon.float(), -1), plan=plan,
                v=end_speed(plan), l1=l1_ag, mt=dis['M_cas_typed'], nv=dis['gated_valid'])


@torch.no_grad()
def main(a):
    dev = a.device
    gf = GameFormer(encoder_layers=3, decoder_levels=2, neighbors=10)
    gf.load_state_dict(torch.load(a.pretrained_path, map_location=dev))
    gf = gf.to(dev); freeze_gameformer(gf)
    m = CausalPlanner(layers=1, modes=6, nbr_enrich=2, ego_residual=0, gate_channels=1,
                      typed_kv=1, dod_meta=1, lat_moe=1, num_lon=NUM_LON4, num_lat=NUM_LAT5V,
                      l1=1, l1_bottleneck=a.l1_bottleneck, num_l1_ag=6).to(dev)
    miss, unexp = m.load_state_dict(torch.load(a.causal_path, map_location=dev), strict=False)
    if miss or unexp:
        raise SystemExit(f"[load] missing={len(miss)} unexpected={len(unexp)}")
    m.eval()
    ld = DataLoader(DrivingData(a.valid_set + "/*.npz", 10), batch_size=32,
                    shuffle=False, num_workers=4)
    rows = []
    for batch in ld:
        inp, ef, nf, rp = read_batch(batch, dev)
        B = ef.shape[0]; ar = torch.arange(B, device=dev)
        enc = gf.encoder(inp)
        t1, ns, _ = extract_neighbor_top1_futures(gf, enc, 10)
        base = run(m, enc, inp, t1, ns)
        ch0 = inp['channel_active'][:, :10].bool()
        nv = base['nv']
        ahead = (ch0[..., AHEAD].any(-1) & nv)
        has = ahead.any(-1)
        tgt = base['mt'].sum(-1).masked_fill(~ahead, -1.0).argmax(-1)
        outs = {}
        # --- L1 mudahalesi: hedefin sinifini zorla ---
        for c, nm in enumerate(L1_NAMES):
            outs[f'l1_{nm}'] = run(m, enc, inp, t1, ns, force=(tgt, torch.full_like(tgt, c)))
        # --- karsilastirma: PREDICATE maskesine mudahale (eski test) ---
        for nm, rel in (('pred_follows', CH_FOLLOWS), ('pred_adjR', CH_ADJACENT_RIGHT)):
            ch = ch0.clone(); ch[ar, tgt, :] = False; ch[ar, tgt, rel] = True
            i2 = dict(inp); i2['channel_active'] = ch
            outs[nm] = run(m, enc, i2, t1, ns)
        for i in range(B):
            if not bool(has[i]):
                continue
            r = dict(base_lon=LON4_CLASSES[int(base['lon'][i])],
                     base_l1=L1_NAMES[int(base['l1'][i, int(tgt[i])].argmax())])
            for nm, o in outs.items():
                r[nm] = dict(flip=bool(o['lon'][i] != base['lon'][i]),
                             lon=LON4_CLASSES[int(o['lon'][i])],
                             dv=float(o['v'][i] - base['v'][i]),
                             dplan=float((o['plan'][i] - base['plan'][i]).norm(dim=-1).mean()))
            rows.append(r)
    n = len(rows)
    print(f"\n=== L1 TEST-TIME INTERVENTION — {a.tag}  (n={n} hedefli sahne) ===")
    print(f"{'mudahale':<28s}{'psi flip':>10s}{'dv_end':>10s}{'dplan':>9s}")
    print("-" * 58)
    for nm in [f'l1_{x}' for x in L1_NAMES] + ['pred_follows', 'pred_adjR']:
        d = [r[nm] for r in rows]
        lab = ('L1 -> ' + nm[3:]) if nm.startswith('l1_') else ('predicate -> ' + nm[5:])
        print(f"{lab:<28s}{100 * np.mean([x['flip'] for x in d]):>9.1f}%"
              f"{np.mean([x['dv'] for x in d]):>+10.3f}{np.mean([x['dplan'] for x in d]):>9.3f}")
    import collections
    print(f"\n  taban L1 sinifi: " + "  ".join(
        f"{k} {v}" for k, v in collections.Counter(r['base_l1'] for r in rows).most_common()))
    if a.out:
        json.dump(rows, open(a.out, 'w'))
        print(f"  -> {a.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_path", required=True)
    p.add_argument("--causal_path", required=True)
    p.add_argument("--valid_set", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--l1_bottleneck", type=int, default=0)
    p.add_argument("--out", type=str, default="")
    p.add_argument("--device", default="cuda:1")
    main(p.parse_args())
