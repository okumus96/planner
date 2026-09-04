"""ZINCIR MUDAHALESI:  L0 predicate  ->  L1 konsept  ->  b* karar  ->  yorunge

NE ICIN: "iz dekoratif mi, yuk tasiyor mu" sorusunun tam testi. Tek tek halkalari degil,
ZINCIRIN KENDISINI olcer -- ozellikle ARACILIK (mediation): L0'in b* uzerindeki etkisi
L1'in ICINDEN mi geciyor, yoksa L1'i atlayip baska yoldan mi gidiyor?

MUDAHALELER (hedef = gated_valid ajanlar icinde en yuksek M_cas):
  L0 : hedefin kanal maskesi TEK bir iliskiye zorlanir (l1_drop_input ile silinenler haric)
  L1 : hedefin L1 sinif dagitimi tek-sicak c'ye zorlanir (ozet yeniden kurulur)

OLCULENLER
  l1flip : hedefin L1 sinifi (argmax) degisti mi          [yalniz L0 mudahalesinde anlamli]
  b*flip : ilan edilen lon karari degisti mi
  dplan  : plan L2 farki [m]
  dv     : plan uc-hizi farki [m/s]  (+ = hizlandi)

ARACILIK: L0 mudahalesi L1'i degistirdiginde b* de degisiyor mu (P(b*flip | l1flip)) vs
degistirmediginde (P(b*flip | ~l1flip)). Ilki ikincisinden BUYUK ise etki L1 uzerinden akiyor.
"""
import argparse, json, os, sys, collections
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.data import DataLoader
from GameFormer.predictor import GameFormer
from GameFormer.causal_graph import CausalPlanner, L1_PROMOTED_CHANNELS
from GameFormer.train_utils import DrivingData
from GameFormer.channels import CHANNEL_NAMES, NUM_CHANNELS
from GameFormer.decision_labels import NUM_LON4, NUM_LAT5V, LON4_CLASSES
from train_planner import read_batch, extract_neighbor_top1_futures, freeze_gameformer

L1_NAMES = ['none', 'follows', 'yieldingTo', 'waitingFor', 'mergesInFrontOf', 'overtakes']


def end_speed(xy):
    v = (xy[:, 1:] - xy[:, :-1]).norm(dim=-1) / 0.1
    return v[:, 34:40].mean(1)


@torch.no_grad()
def run(m, enc, inp, t1, ns, force=None):
    B = enc['mask'].shape[0]; dev = enc['mask'].device
    dis = m.disentangler(enc['agent_tokens'][:, :11].detach(), ~enc['mask'][:, :11],
                         enc['actors'][:, :11, -1].detach(),
                         torch.zeros(B, 11, dtype=torch.long, device=dev), inp,
                         neighbor_futures=t1, neighbor_states=ns)
    l1_ag = m.l1_head_ag(dis['src_cas_ag']); l1_mp = m.l1_head_mp(dis['src_cas_mp'])
    p_ag = torch.softmax(l1_ag, -1)
    if force is not None:                                   # L1 MUDAHALESI: tek-sicak
        j, c = force
        ar = torch.arange(B, device=dev)
        oh = torch.zeros_like(p_ag[ar, j]); oh[ar, c] = 1.0
        p_ag = p_ag.clone(); p_ag[ar, j] = oh
    va = (dis['gated_valid'].float() * dis['M_cas'])[..., None]
    vm = (dis['gated_map_valid'].float() * dis['M_cas_map'])[..., None]
    summ = torch.cat([(p_ag * va).sum(1), (torch.softmax(l1_mp, -1) * vm).sum(1)], -1)
    z = summ if m.l1_bottleneck else torch.cat([dis['f_cas'], summ], -1)
    lon, lat = m.psi_lon(z), m.psi_lat(z)
    b = (lon.argmax(-1), lat.argmax(-1))
    traj, score = m.head(dis['f_cas'], dis['ego_clean'], b)
    plan = traj[:, 0][torch.arange(B, device=dev), score[:, 0].argmax(-1)][..., :2]
    return dict(lon=b[0], lat=b[1], plan=plan, v=end_speed(plan),
                l1=l1_ag.argmax(-1), M=dis['M_cas'], nv=dis['gated_valid'])


@torch.no_grad()
def main(a):
    dev = a.device
    gf = GameFormer(encoder_layers=3, decoder_levels=2, neighbors=10)
    gf.load_state_dict(torch.load(a.pretrained_path, map_location=dev))
    gf = gf.to(dev); freeze_gameformer(gf)
    m = CausalPlanner(layers=1, modes=6, nbr_enrich=2, ego_residual=0, gate_channels=1,
                      typed_kv=1, dod_meta=1, lat_moe=1, num_lon=NUM_LON4, num_lat=NUM_LAT5V,
                      l1=1, l1_bottleneck=a.l1_bottleneck, num_l1_ag=6,
                      l1_drop_input=a.l1_drop_input).to(dev)
    miss, unexp = m.load_state_dict(torch.load(a.causal_path, map_location=dev), strict=False)
    if miss or unexp:
        raise SystemExit(f"[load] missing={len(miss)} unexpected={len(unexp)}")
    m.eval()
    # l1_drop_input=1 ise silinen kanallara mudahale etmenin ETKISI YOKTUR -> disla
    dropped = set(L1_PROMOTED_CHANNELS) if a.l1_drop_input else set()
    L0_REL = [c for c in range(NUM_CHANNELS) if c not in dropped]
    print(f"L0 mudahale kanallari ({len(L0_REL)}): {[CHANNEL_NAMES[c] for c in L0_REL]}")
    if dropped:
        print(f"  DISLANAN (girdiden silinmis, mudahale etkisiz): "
              f"{[CHANNEL_NAMES[c] for c in sorted(dropped)]}")

    ld = DataLoader(DrivingData(a.valid_set + "/*.npz", 10), batch_size=32,
                    shuffle=False, num_workers=4)
    L0, L1, med = collections.defaultdict(list), collections.defaultdict(list), []
    nsc = 0
    for bi, batch in enumerate(ld):
        if a.limit and nsc >= a.limit: break
        inp, ef, nf, rp = read_batch(batch, dev)
        B = ef.shape[0]; ar = torch.arange(B, device=dev)
        enc = gf.encoder(inp)
        t1, ns, _ = extract_neighbor_top1_futures(gf, enc, 10)
        base = run(m, enc, inp, t1, ns)
        nv = base['nv']
        has = nv.any(-1)
        tgt = base['M'].masked_fill(~nv, -1.0).argmax(-1)      # en yuksek kutleli gecerli ajan
        ch0 = inp['channel_active'][:, :10].bool()

        for r in L0_REL:                                        # --- L0 MUDAHALESI ---
            ch = ch0.clone(); ch[ar, tgt, :] = False; ch[ar, tgt, r] = True
            i2 = dict(inp); i2['channel_active'] = ch
            o = run(m, enc, i2, t1, ns)
            lf = (o['l1'][ar, tgt] != base['l1'][ar, tgt])
            bf = (o['lon'] != base['lon'])
            dp = (o['plan'] - base['plan']).norm(dim=-1).mean(-1)
            for i in range(B):
                if not has[i]: continue
                L0[CHANNEL_NAMES[r]].append((bool(lf[i]), bool(bf[i]), float(dp[i]),
                                             float(o['v'][i] - base['v'][i])))
                med.append((bool(lf[i]), bool(bf[i])))
        for c in range(6):                                      # --- L1 MUDAHALESI ---
            o = run(m, enc, inp, t1, ns, force=(tgt, torch.full_like(tgt, c)))
            bf = (o['lon'] != base['lon'])
            dp = (o['plan'] - base['plan']).norm(dim=-1).mean(-1)
            for i in range(B):
                if not has[i]: continue
                L1[L1_NAMES[c]].append((bool(bf[i]), float(dp[i]),
                                        float(o['v'][i] - base['v'][i]),
                                        L1_NAMES[int(base['l1'][i, int(tgt[i])])] == L1_NAMES[c]))
        nsc += int(has.sum())
    print(f"\nhedefli sahne: {nsc}\n")
    print("=== L0 MUDAHALESI: hedefin kanali TEK iliskiye zorlandi ===")
    print(f"{'kanal':<38s}{'L1 flip':>9s}{'b* flip':>9s}{'dplan[m]':>10s}{'dv[m/s]':>9s}")
    print("-" * 75)
    for k, v in sorted(L0.items(), key=lambda x: -np.mean([y[1] for y in x[1]])):
        print(f"{k:<38s}{100*np.mean([y[0] for y in v]):>8.1f}%{100*np.mean([y[1] for y in v]):>8.1f}%"
              f"{np.mean([y[2] for y in v]):>10.3f}{np.mean([y[3] for y in v]):>+9.3f}")
    print("\n=== L1 MUDAHALESI: hedefin konsepti tek-sicak c'ye zorlandi ===")
    print(f"{'-> sinif':<38s}{'b* flip':>9s}{'dplan[m]':>10s}{'dv[m/s]':>9s}")
    print("-" * 66)
    for k in L1_NAMES:
        v = L1[k]
        w = [y for y in v if not y[3]]                          # zaten o sinif olanlari disla
        if not w: continue
        print(f"{k:<38s}{100*np.mean([y[0] for y in w]):>8.1f}%"
              f"{np.mean([y[1] for y in w]):>10.3f}{np.mean([y[2] for y in w]):>+9.3f}")
    lf = np.array([x[0] for x in med]); bf = np.array([x[1] for x in med])
    print(f"\n=== ARACILIK (L0'in etkisi L1'in icinden mi geciyor?) ===")
    print(f"  P(b* flip | L1 DEGISTI )  = %{100*bf[lf].mean():.1f}   (n={int(lf.sum())})")
    print(f"  P(b* flip | L1 DEGISMEDI) = %{100*bf[~lf].mean():.1f}   (n={int((~lf).sum())})")
    print(f"  fark = {100*(bf[lf].mean()-bf[~lf].mean()):+.1f} puan")
    if a.out:
        json.dump({'L0': {k: v for k, v in L0.items()}, 'L1': {k: v for k, v in L1.items()}},
                  open(a.out, 'w'))
        print(f"\n[saved] {a.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_path", required=True); p.add_argument("--causal_path", required=True)
    p.add_argument("--valid_set", required=True)
    p.add_argument("--l1_bottleneck", type=int, default=1)
    p.add_argument("--l1_drop_input", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", type=str, default=""); p.add_argument("--device", default="cuda:0")
    main(p.parse_args())
