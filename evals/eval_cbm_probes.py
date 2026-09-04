"""CBM IC DENETIMI -- "L1 dogru seyi yapiyor ve gerekli mi?" dort test.

A YETERLILIK  : 8 sayilik ozet, 256-boyutlu f_cas kadar karar bilgisi tasiyor mu?
                Iki lineer prob (ayni egitim/test bolunmesi): f_cas->karar vs summ->karar.
                Fark = bottleneck'in BILGI BEDELI. Kucukse "8 sayi yetiyor".
B SIZINTI     : ozetten KONSEPT OLMAYAN sey okunabiliyor mu (ego hizi, en yakin ajan mesafesi)?
                Kontrol: f_cas'in RASTGELE 8-boyutlu projeksiyonu (ayni kapasite, sifir anlam).
                Ozet kontrolden belirgin iyiyse "konsept" adi altinda ham bilgi kaciyor demektir.
C MUDAHALE DOGRULUGU : model konsepti YANLIS bildigi sahnelerde, konsepti GT'ye zorlayinca
                karar DAHA DOGRU oluyor mu? CBM'in altin testi -- simdiye kadar yalnizca
                "degisiyor mu" (duyarlilik) olctuk, "duzeliyor mu" degil.
D GEREKLILIK  : ozet batch icinde KARISTIRILINCA karar/plan cokuyor mu? Cokmezse L1 surus icin
                dekoratif, yalnizca rapor katmani.
"""
import argparse, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.data import DataLoader, Subset
from GameFormer.predictor import GameFormer
from GameFormer.causal_graph import CausalPlanner
from GameFormer.train_utils import DrivingData
from GameFormer.decision_labels import NUM_LON4, NUM_LAT5V, LON4_MAP, LAT5V_MAP, LON4_CLASSES
from train_planner import read_batch, extract_neighbor_top1_futures, freeze_gameformer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, r2_score

L1N = ['none','follows','yieldingTo','waitingFor','mergesInFrontOf','overtakes']


def summarize(m, d, force=None):
    """L1 ozetini kur; force=(idx,cls) verilirse o ajanin sinifi tek-sicak zorlanir."""
    B = d['f_cas'].shape[0]; dev = d['f_cas'].device
    p = torch.softmax(m.l1_head_ag(d['src_cas_ag']), -1)
    if force is not None:
        j, c = force; ar = torch.arange(B, device=dev)
        oh = torch.zeros_like(p[ar, j]); oh[ar, c] = 1.0
        p = p.clone(); p[ar, j] = oh
    va = (d['gated_valid'].float() * d['M_cas'])[..., None]
    vm = (d['gated_map_valid'].float() * d['M_cas_map'])[..., None]
    return torch.cat([(p * va).sum(1),
                      (torch.softmax(m.l1_head_mp(d['src_cas_mp']), -1) * vm).sum(1)], -1)


def decide(m, d, summ):
    z = summ if m.l1_bottleneck else torch.cat([d['f_cas'], summ], -1)
    lon, lat = m.psi_lon(z), m.psi_lat(z)
    b = (lon.argmax(-1), lat.argmax(-1))
    traj, sc = m.head(d['f_cas'], d['ego_clean'], b)
    B = summ.shape[0]; ar = torch.arange(B, device=summ.device)
    return b[0], traj[:, 0][ar, sc[:, 0].argmax(-1)][..., :2]


@torch.no_grad()
def main(a):
    dev = a.device
    gf = GameFormer(encoder_layers=3, decoder_levels=2, neighbors=10)
    gf.load_state_dict(torch.load(a.pretrained_path, map_location=dev))
    gf = gf.to(dev); freeze_gameformer(gf)
    m = CausalPlanner(layers=1, modes=6, nbr_enrich=2, ego_residual=0, gate_channels=1,
                      typed_kv=1, dod_meta=1, lat_moe=1, num_lon=NUM_LON4, num_lat=NUM_LAT5V,
                      l1=1, l1_bottleneck=1, num_l1_ag=6, l1_drop_input=a.l1_drop_input).to(dev)
    m.load_state_dict(torch.load(a.causal_path, map_location=dev), strict=False); m.eval()
    ds = DrivingData(a.valid_set + "/*.npz", 10, l1_labels=a.l1_labels)
    n = min(a.limit or len(ds), len(ds))
    L4 = torch.tensor(LON4_MAP, device=dev)

    F, S, Y, V0, DMIN, NAG = [], [], [], [], [], []
    corr_base, corr_fix, nfix = 0, 0, 0
    d_lon, d_plan, shuf_acc, base_acc = [], [], 0, 0
    for b in DataLoader(Subset(ds, list(range(n))), batch_size=64, num_workers=4):
        inp, ef, nf, rp = read_batch(b, dev)
        B = ef.shape[0]; ar = torch.arange(B, device=dev)
        enc = gf.encoder(inp); t1, ns, _ = extract_neighbor_top1_futures(gf, enc, 10)
        d = m.disentangler(enc['agent_tokens'][:, :11].detach(), ~enc['mask'][:, :11],
                           enc['actors'][:, :11, -1].detach(),
                           torch.zeros(B, 11, dtype=torch.long, device=dev), inp,
                           neighbor_futures=t1, neighbor_states=ns)
        summ = summarize(m, d)
        lon_gt = L4[inp['decision_lon']]
        lon0, plan0 = decide(m, d, summ)
        F.append(d['f_cas'].cpu()); S.append(summ.cpu()); Y.append(lon_gt.cpu())
        # B icin konsept-disi hedefler
        V0.append(inp['ego_agent_past'][:, -1, 3:5].norm(dim=-1).cpu())
        pos = inp['neighbor_agents_past'][:, :10, -1, :2]
        val = d['gated_valid']
        DMIN.append(pos.norm(dim=-1).masked_fill(~val, 1e3).min(1).values.clamp(max=100).cpu())
        NAG.append(val.float().sum(1).cpu())
        # --- C: konsepti GT'ye zorla ---
        y = inp['l1_agent'][:, :10]
        pred = m.l1_head_ag(d['src_cas_ag']).argmax(-1)
        tgt = d['M_cas'].masked_fill(~val, -1.0).argmax(-1)
        wrong = val[ar, tgt] & (pred[ar, tgt] != y[ar, tgt])
        if wrong.any():
            lonF, _ = decide(m, d, summarize(m, d, force=(tgt, y[ar, tgt])))
            corr_base += int((lon0[wrong] == lon_gt[wrong]).sum())
            corr_fix += int((lonF[wrong] == lon_gt[wrong]).sum())
            nfix += int(wrong.sum())
        # --- D: ozeti batch icinde karistir ---
        perm = torch.randperm(B, device=dev)
        lonS, planS = decide(m, d, summ[perm])
        base_acc += int((lon0 == lon_gt).sum()); shuf_acc += int((lonS == lon_gt).sum())
        d_lon.append((lonS != lon0).float().cpu())
        d_plan.append((planS - plan0).norm(dim=-1).mean(-1).cpu())

    F = torch.cat(F).numpy(); S = torch.cat(S).numpy(); Y = torch.cat(Y).numpy()
    N = len(Y); tr = np.zeros(N, bool); tr[:int(.7 * N)] = True
    rng = np.random.default_rng(0); rng.shuffle(tr)
    print(f"\n{'='*64}\nA) YETERLILIK -- 8 sayi 256 boyut kadar karar bilgisi tasiyor mu\n{'='*64}")
    for lab, X in (('f_cas (256 boyut)', F), ('summ  (8 sayi)', S)):
        clf = LogisticRegression(max_iter=2000, multi_class='multinomial').fit(X[tr], Y[tr])
        acc = balanced_accuracy_score(Y[~tr], clf.predict(X[~tr]))
        print(f"   {lab:<22s} dengeli dogruluk = {acc:.4f}")
    print(f"\n{'='*64}\nB) SIZINTI -- ozetten KONSEPT OLMAYAN sey okunabiliyor mu (R^2)\n{'='*64}")
    R = rng.standard_normal((F.shape[1], 8)) / np.sqrt(F.shape[1])
    FR = F @ R                                    # ayni kapasite, sifir anlam -> KONTROL
    tgts = [('ego hizi [m/s]', torch.cat(V0).numpy()),
            ('en yakin ajan mesafesi [m]', torch.cat(DMIN).numpy()),
            ('grafikteki ajan sayisi', torch.cat(NAG).numpy())]
    print(f"   {'hedef':<28s}{'summ(8)':>10s}{'rastgele(8)':>13s}{'f_cas(256)':>12s}")
    for nm, t in tgts:
        r = []
        for X in (S, FR, F):
            r.append(r2_score(t[~tr], Ridge(alpha=1.0).fit(X[tr], t[tr]).predict(X[~tr])))
        print(f"   {nm:<28s}{r[0]:>10.3f}{r[1]:>13.3f}{r[2]:>12.3f}")
    print(f"\n{'='*64}\nC) MUDAHALE DOGRULUGU -- yanlis konsepti GT'ye zorlayinca karar duzeliyor mu\n{'='*64}")
    print(f"   konsepti YANLIS bilinen sahne: {nfix}")
    if nfix:
        print(f"   karar dogrulugu  mudahale ONCESI = %{100*corr_base/nfix:.1f}")
        print(f"   karar dogrulugu  mudahale SONRASI = %{100*corr_fix/nfix:.1f}"
              f"   ({100*(corr_fix-corr_base)/nfix:+.1f} puan)")
    print(f"\n{'='*64}\nD) GEREKLILIK -- ozet karistirilinca ne cokuyor\n{'='*64}")
    print(f"   karar dogrulugu  normal     = %{100*base_acc/N:.1f}")
    print(f"   karar dogrulugu  KARISTIRIK = %{100*shuf_acc/N:.1f}"
          f"   ({100*(shuf_acc-base_acc)/N:+.1f} puan)")
    print(f"   karar degisme orani         = %{100*torch.cat(d_lon).mean():.1f}")
    print(f"   plan L2 kaymasi             = {torch.cat(d_plan).mean():.3f} m")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_path", required=True); p.add_argument("--causal_path", required=True)
    p.add_argument("--valid_set", required=True); p.add_argument("--l1_labels", required=True)
    p.add_argument("--l1_drop_input", type=int, default=1)
    p.add_argument("--limit", type=int, default=0); p.add_argument("--device", default="cuda:0")
    main(p.parse_args())
