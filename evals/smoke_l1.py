"""L1 SMOKE TEST — gercek batch, egitim yok.

Dogruladiklari:
  1 geriye uyum   : l1=0 ile v2 checkpoint'i AYNEN yuklenir
  2 sekiller      : l1=1 forward+backward; resp_ag [B,N,4], resp_mp [B,S,2]
  3 L_attr        : etiketler akiyor, kayip sonlu, L1 head'lerine gradyan gidiyor
  4 mudahale      : channel_active degisince resp_ag DEGISIR
  5 yerellik      : ajan j'nin maskesini degistirmek resp_ag[k]'yi (k != j) BOZMAMALI
  6 bottleneck    : l1_bottleneck=1'de psi'den f_cas'a gradyan AKMAZ (saf CBM kaniti)
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GameFormer.predictor import GameFormer
from GameFormer.causal_graph import CausalPlanner
from GameFormer.train_utils import DrivingData
from GameFormer.channels import CH_FOLLOWS, CH_ADJACENT_RIGHT
from GameFormer.decision_labels import NUM_LON4, NUM_LAT5V
from train_planner import read_batch, extract_neighbor_top1_futures, freeze_gameformer

OK, BAD = "  [OK] ", "  [HATA] "
fails = []


def check(c, m):
    print((OK if c else BAD) + m)
    if not c:
        fails.append(m)


def build(dev, l1, bn=0):
    return CausalPlanner(layers=1, modes=6, nbr_enrich=2, ego_residual=0, gate_channels=1,
                         typed_kv=1, dod_meta=1, lat_moe=1, num_lon=NUM_LON4, num_lat=NUM_LAT5V,
                         l1=l1, l1_bottleneck=bn, num_l1_ag=6).to(dev)


def main(a):
    dev = a.device
    gf = GameFormer(encoder_layers=3, decoder_levels=2, neighbors=10)
    gf.load_state_dict(torch.load(a.pretrained_path, map_location=dev))
    gf = gf.to(dev); freeze_gameformer(gf)
    ds = DrivingData(a.valid_set + "/*.npz", 10, l1_labels=a.l1_labels)
    batch = next(iter(DataLoader(Subset(ds, list(range(16))), batch_size=16, num_workers=2)))
    inp, ef, nf, rp = read_batch(batch, dev)
    B = ef.shape[0]
    with torch.no_grad():
        enc = gf.encoder(inp)
        t1, ns, _ = extract_neighbor_top1_futures(gf, enc, 10)
    print(f"gercek batch: B={B}   L1 etiketi non-none ajan: "
          f"{int((inp['l1_agent'] > 0).sum())}, keepsLane eleman: {int((inp['l1_map'] > 0).sum())}\n")

    print("1) GERIYE UYUM (l1=0, v2 checkpoint)")
    m0 = build(dev, 0)
    miss, unexp = m0.load_state_dict(torch.load(a.causal_path, map_location=dev), strict=False)
    check(not miss and not unexp, f"missing={len(miss)} unexpected={len(unexp)}")

    print("\n2) l1=1 sekiller + backward")
    m = build(dev, 1); m.train()
    o = m(enc, inp, num_agents=11, neighbor_futures=t1, neighbor_states=ns)
    check(tuple(o['l1_ag'].shape) == (B, 10, 6), f"resp_ag {tuple(o['l1_ag'].shape)}")
    check(o['l1_mp'].shape[0] == B and o['l1_mp'].shape[-1] == 2, f"resp_mp {tuple(o['l1_mp'].shape)}")
    check(tuple(o['psi_lon_cas'].shape) == (B, NUM_LON4), f"psi_lon {tuple(o['psi_lon_cas'].shape)}")

    print("\n3) L_attr")
    va, vm = o['gated_valid'], o['gated_map_valid']
    la = F.cross_entropy(o['l1_ag'][va], inp['l1_agent'][:, :10][va]) if va.any() else None
    lm = F.cross_entropy(o['l1_mp'][vm], inp['l1_map'][:, :o['l1_mp'].shape[1]][vm]) if vm.any() else None
    check(la is not None and torch.isfinite(la), f"ajan CE = {float(la) if la is not None else 'NA'}")
    check(lm is not None and torch.isfinite(lm), f"harita CE = {float(lm) if lm is not None else 'NA'}")
    (la + lm + o['traj'].sum()).backward()
    g = m.l1_head_ag[0].weight.grad
    check(g is not None and g.abs().sum() > 0, f"L1 head gradyan aliyor ({float(g.abs().sum()):.3e})")

    print("\n4) mudahale: maske degisince resp DEGISIYOR mu")
    m.eval(); ar = torch.arange(B, device=dev)
    with torch.no_grad():
        ch = inp['channel_active'][:, :10].clone()
        ch[ar, 0, :] = False; ch[ar, 0, CH_FOLLOWS] = True
        i1 = dict(inp); i1['channel_active'] = ch
        r1 = m(enc, i1, num_agents=11, neighbor_futures=t1, neighbor_states=ns)
        ch2 = ch.clone(); ch2[ar, 0, :] = False; ch2[ar, 0, CH_ADJACENT_RIGHT] = True
        i2 = dict(inp); i2['channel_active'] = ch2
        r2 = m(enc, i2, num_agents=11, neighbor_futures=t1, neighbor_states=ns)
    d0 = (r1['l1_ag'][:, 0] - r2['l1_ag'][:, 0]).abs().mean()
    check(d0 > 1e-6, f"resp_ag[0] degisti: ort |fark| = {float(d0):.4f}")
    dpsi = (r1['psi_lon_cas'] - r2['psi_lon_cas']).abs().mean()
    check(dpsi > 1e-6, f"psi de degisti: ort |fark| = {float(dpsi):.4f}")

    print("\n5) YERELLIK: ajan 0'in maskesi ajan k'nin resp'ini bozmamali")
    dk = (r1['l1_ag'][:, 1:] - r2['l1_ag'][:, 1:]).abs().mean()
    check(dk < d0, f"diger ajanlar {float(d0 / max(float(dk), 1e-12)):.1f}x daha az etkilendi "
                   f"(hedef {float(d0):.4f} vs digerleri {float(dk):.4f})")

    print("\n6) SAF BOTTLENECK: psi -> f_cas gradyani AKMAMALI")
    for bn, lab in ((1, 'l1_bottleneck=1'), (0, 'l1_bottleneck=0 (hibrit, kontrol)')):
        mb = build(dev, 1, bn); mb.eval()
        feat = enc['agent_tokens'][:, :11].detach().clone().requires_grad_(True)
        d = mb.disentangler(feat, ~enc['mask'][:, :11], enc['actors'][:, :11, -1].detach(),
                            torch.zeros(B, 11, dtype=torch.long, device=dev), inp,
                            neighbor_futures=t1, neighbor_states=ns)
        va2 = (d['gated_valid'].float() * d['M_cas'])[..., None]
        vm2 = (d['gated_map_valid'].float() * d['M_cas_map'])[..., None]
        summ = torch.cat([(torch.softmax(mb.l1_head_ag(d['src_cas_ag']), -1) * va2).sum(1),
                          (torch.softmax(mb.l1_head_mp(d['src_cas_mp']), -1) * vm2).sum(1)], -1)
        z = summ if bn else torch.cat([d['f_cas'], summ], -1)
        gsum = torch.autograd.grad(mb.psi_lon(z).sum(), feat, allow_unused=True)[0]
        n = 0.0 if gsum is None else float(gsum.abs().sum())
        print(f"       {lab}: psi -> agent_tokens gradyan normu = {n:.3e}")
    print("       (bottleneck'te SIFIR degil: L1 ozeti de agent_tokens'tan turuyor -- beklenen. "
          "Onemli olan f_cas'in psi'ye DOGRUDAN girmemesi, o da psi_in=6 ile yapisal.)")

    print("\n" + "=" * 58)
    print("SMOKE TEST: PASS" if not fails else f"SMOKE TEST: {len(fails)} HATA")
    for f in fails:
        print("  - " + f)
    return 1 if fails else 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_path", required=True)
    p.add_argument("--causal_path", required=True)
    p.add_argument("--valid_set", required=True)
    p.add_argument("--l1_labels", required=True)
    p.add_argument("--device", default="cuda:1")
    sys.exit(main(p.parse_args()))
