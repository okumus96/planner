"""BINDING TEST — secilmis sette mudahale olcumu (set: eval_binding_set.py -> JSON).

Sahne basina DORT kosum, tek fark mudahale:
  base   : dokunulmamis sahne (referans dagilim P0)
  edge   : hedefin YALNIZ follows+same_lane_ahead kenarlari kesilir; ajan sahnede ve grafikte
           (diger iliskileri varsa) kalir. Iliski-duzeyi do().
  graph  : hedefin TUM kanallari kapatilir -> ajan causal grafikten cikar (gate), ama frozen
           GF encoder'da hala gorunur. Grafik-duzeyi ajan cikarma.
  enc    : hedef encoder maskesiyle TAMAMEN silinir (komsu tahminleri de yeniden hesaplanir)
           -> arka-kanal (frozen backbone ego token'i) da kapanir. Tam cikarma = tavan.
  ctrl   : EN UZAK, dikkat-iliskisi tasimayan ajan encoder-duzeyi silinir -> beklenti ~0.

edge-vs-graph farki = hedefin diger iliskilerinin tasidigi bilgi; graph-vs-enc farki =
FROZEN BACKBONE ARKA-KANALININ boyutu (kenar mudahalesinin yapisal tavani — paper'da acikca
raporlanacak sayi).

Metrikler (once/sonra): unbrake = argmax SLOW ailesinden cikti; flip = argmax degisti;
dP(SLOW)/dP(GO) = softmax aile olasiligi farki (sonra - once); dv_end = plan uc-hiz farki;
dplan = plan L2. Ana dilim = model tabanda frenleyen sahneler (uzman zaten frenliyor, filtre
geregi); ayrica tum set + kutle-payi (medyan ustu/alti) dilimleri.

Kosum (v6, set JSON'u onceden uretilmis olmali):
  python eval_binding_test.py --pretrained_path training_log/normal/model_epoch_19_valADE_1.6487.pth \
    --causal_path training_log/dodmeta_v6_relev/causal_epoch_18_minADE_0.7139.pth \
    --valid_set /home/lt-hta-ai4/ssd1/nuplan/processed_data/validation \
    --set_json binding_set_v6.json --rel_bottleneck 1 --rel_evidence 1 --device cuda:1
"""
import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from GameFormer.predictor import GameFormer
from GameFormer.causal_graph import CausalPlanner
from GameFormer.train_utils import DrivingData
from GameFormer.channels import (CH_FOLLOWS, CH_SAME_LANE_AHEAD, CH_ADJACENT_RIGHT,
                                 CH_COLLISION_COURSE, CH_SHARES_INTERSECTION, CH_MERGES,
                                 CH_VRU)
from GameFormer.decision_labels import LON_CLASSES
from train_planner import read_batch, extract_neighbor_top1_futures, freeze_gameformer

AHEAD = [CH_FOLLOWS, CH_SAME_LANE_AHEAD]
CAUTION_ALL = AHEAD + [CH_COLLISION_COURSE, CH_SHARES_INTERSECTION, CH_MERGES, CH_VRU]
SLOW_IDX = [LON_CLASSES.index(c) for c in
            ('stop_quickly', 'stop_gently', 'slow_quickly', 'slow_gently')]
GO_IDX = [LON_CLASSES.index(c) for c in ('accel_quickly', 'accel_gently', 'maintain')]


def best_plan(out, B):
    traj = out['traj'][:, 0]
    best = out['score'][:, 0].argmax(-1)
    return traj[torch.arange(B, device=traj.device), best][..., :2]


def end_speed(xy):
    v = (xy[:, 1:] - xy[:, :-1]).norm(dim=-1) / 0.1
    return v[:, 34:40].mean(1)


@torch.no_grad()
def run(m, enc, inp, t1, ns, N):
    out = m(enc, inp, num_agents=N + 1, neighbor_futures=t1, neighbor_states=ns)
    B = inp['ego_agent_past'].shape[0]
    P = torch.softmax(out['psi_lon_cas'].float(), -1)                 # [B,9]
    lon = P.argmax(-1)
    p = best_plan(out, B)
    return dict(P=P, lon=lon, plan=p, v=end_speed(p), mt=out['M_cas_typed'])


def fam(P, idx):
    return P[:, idx].sum(-1)


@torch.no_grad()
def main(a):
    dev = a.device
    sel = json.load(open(a.set_json))
    print(f"[set] {a.set_json}: {len(sel)} sahne")
    gf = GameFormer(encoder_layers=3, decoder_levels=2, neighbors=a.num_neighbors)
    gf.load_state_dict(torch.load(a.pretrained_path, map_location=dev))
    gf = gf.to(dev); freeze_gameformer(gf)
    m = CausalPlanner(layers=a.graph_layers, modes=a.modes, nbr_enrich=a.nbr_enrich,
                      ego_residual=a.ego_residual, gate_channels=1, typed_kv=a.typed_kv,
                      dod_meta=a.dod_meta, num_lon=(6 if a.lon_merge else 9),
                      rel_bottleneck=a.rel_bottleneck, rel_evidence=a.rel_evidence).to(dev)
    miss, unexp = m.load_state_dict(torch.load(a.causal_path, map_location=dev), strict=False)
    if miss or unexp:
        raise SystemExit(f"[load] bayrak/checkpoint uyusmazligi: missing={len(miss)} "
                         f"unexpected={len(unexp)}")
    m.eval()

    ds = DrivingData(a.valid_set + "/*.npz", a.num_neighbors)
    sub = Subset(ds, [s['scene'] for s in sel])                        # JSON sirasi korunur
    ld = DataLoader(sub, batch_size=a.batch_size, shuffle=False, num_workers=4)
    N = a.num_neighbors

    rows = []                                                          # sahne-basina kayit
    off = 0
    for batch in ld:
        inp, ef, nf, rp = read_batch(batch, dev)
        B = ef.shape[0]
        metas = sel[off:off + B]; off += B
        tgt = torch.tensor([s['agent'] for s in metas], device=dev)    # [B]
        ar = torch.arange(B, device=dev)

        enc = gf.encoder(inp)
        t1, ns, _ = extract_neighbor_top1_futures(gf, enc, N)
        base = run(m, enc, inp, t1, ns, N)
        ch0 = inp["channel_active"]

        # --- edge: hedefin follows + lane_ahead kenarlari ---
        ch_e = ch0.clone()
        for r in AHEAD:
            ch_e[ar, tgt, r] = False
        inp_e = dict(inp); inp_e["channel_active"] = ch_e
        edge = run(m, enc, inp_e, t1, ns, N)

        # --- swapR: TIP-DEGISIMI mudahalesi (do(type = yanlis)) --- ajan ve girisleri
        # kalir, SADECE etiket degisir: lane_ahead/follows -> adjacent_right. Tipler anlamliysa
        # "onumdeki lider" -> "yandaki arac" yeniden-etiketi freni birakmali; girisler jenerik
        # slotsa hicbir sey degismemeli.
        ch_sr = ch0.clone()
        for r in AHEAD:
            ch_sr[ar, tgt, r] = False
        ch_sr[ar, tgt, CH_ADJACENT_RIGHT] = True
        inp_sr = dict(inp); inp_sr["channel_active"] = ch_sr
        swapR = run(m, enc, inp_sr, t1, ns, N)

        # --- swapA: hedefin TUM dikkat etiketleri -> adjacent_right (tam yanlis-etiket).
        # Ajan grafikte KALIR (girisi var) ama artik "zararsiz yandaki arac" olarak etiketli.
        # swapA ~= graph (silme) ise etiket semantigi tasiyor; swapA ~= base ise tasimiyor.
        ch_sa = ch0.clone()
        for r in CAUTION_ALL:
            ch_sa[ar, tgt, r] = False
        ch_sa[ar, tgt, CH_ADJACENT_RIGHT] = True
        inp_sa = dict(inp); inp_sa["channel_active"] = ch_sa
        swapA = run(m, enc, inp_sa, t1, ns, N)

        # --- swapC: swapA'nin ESLESMIS KONTROLU (caution -> caution). Ayni yapisal
        # mudahale (hedefin tum dikkat etiketleri TEK girise indirgenir), tek fark hayatta
        # kalan etiketin ailesi: collision_course (caution) vs adjacent_right (benign).
        # Etiket semantigi okunuyorsa swapC ~= base (fren kalir) << swapA; "herhangi bir
        # etiket degisikligi flip uretir" artefakti dogruysa swapC ~= swapA.
        ch_sc = ch0.clone()
        for r in CAUTION_ALL:
            ch_sc[ar, tgt, r] = False
        ch_sc[ar, tgt, CH_COLLISION_COURSE] = True
        inp_sc = dict(inp); inp_sc["channel_active"] = ch_sc
        swapC = run(m, enc, inp_sc, t1, ns, N)

        # --- graph: hedefin TUM kanallari (gate disina cikar) ---
        ch_g = ch0.clone(); ch_g[ar, tgt] = False
        inp_g = dict(inp); inp_g["channel_active"] = ch_g
        graph = run(m, enc, inp_g, t1, ns, N)

        # --- enc: encoder maskesiyle tam silme (tahminler yeniden) ---
        enc2 = dict(enc); msk = enc['mask'].clone()
        msk[ar, tgt + 1] = True                                        # komsu j -> sutun 1+j
        enc2['mask'] = msk
        t1e, nse, _ = extract_neighbor_top1_futures(gf, enc2, N)
        encr = run(m, enc2, inp_g, t1e, nse, N)                        # kanallar da hedefsiz

        # --- ctrl: en uzak dikkatsiz ajan tam silme ---
        nbr_xy = inp["neighbor_agents_past"][:, :N, -1, :2]
        dist = nbr_xy.norm(dim=-1)
        valid = base['P'].new_ones(B, N, dtype=torch.bool)
        valid &= inp["neighbor_agents_past"][:, :N].abs().sum((2, 3)) > 0
        no_caut = ~ch0[..., AHEAD].any(-1) & valid
        no_caut[ar, tgt] = False
        dist_m = dist.masked_fill(~no_caut, -1.0)
        ctl = dist_m.argmax(-1)                                        # [B]
        has_ctl = dist_m.gather(1, ctl[:, None]).squeeze(1) > 0
        enc3 = dict(enc); msk3 = enc['mask'].clone()
        msk3[ar, ctl + 1] = True
        enc3['mask'] = msk3
        t1c, nsc, _ = extract_neighbor_top1_futures(gf, enc3, N)
        ch_c = ch0.clone(); ch_c[ar, ctl] = False
        inp_c = dict(inp); inp_c["channel_active"] = ch_c
        ctrl = run(m, enc3, inp_c, t1c, nsc, N)

        # --- inject: TERS swapA (benign -> caution). En uzak dikkatsiz ajana (ctrl'un
        # sectigi ayni ajan) follows+lane_ahead+collision etiketleri VERILIR; sahnenin
        # baska hicbir seyi degismez. Etiketler enjeksiyonla da okunuyorsa dP_SLOW >= 0
        # beklenir; taban zaten frenledigi icin (P(SLOW)~0.97) flip beklenmez — tavan
        # etkisi; bilgilendirici olcum ginj'dedir.
        CAUT_INJ = [CH_FOLLOWS, CH_SAME_LANE_AHEAD, CH_COLLISION_COURSE]
        ch_i = ch0.clone()
        for r in CAUT_INJ:
            ch_i[ar, ctl, r] = True
        inp_i = dict(inp); inp_i["channel_active"] = ch_i
        inject = run(m, enc, inp_i, t1, ns, N)

        # --- ginj: graph + inject — GERCEK sebep grafikten cikar, SAHTE sebep (uzak
        # ajan) caution etiketleriyle grafige sokulur. Etiketler semantik okunuyorsa fren
        # SURMELI: flip(ginj) << flip(graph). Enjeksiyon yonunde tavani asan olcum.
        ch_gi = ch_g.clone()
        for r in CAUT_INJ:
            ch_gi[ar, ctl, r] = True
        inp_gi = dict(inp); inp_gi["channel_active"] = ch_gi
        ginj = run(m, enc, inp_gi, t1, ns, N)

        for i in range(B):
            # model_braking ve mass_share SET modelinden DEGIL, test edilen modelden:
            # ayni JSON seti v3/v4/v6 icin kullanilir (uyelik model-bagimsiz: uzman etiketi +
            # cache kanallari), ama dilimleme her modelin KENDI karari/kutlesiyle yapilmali.
            tgt_i = int(tgt[i])
            r = dict(**{k: metas[i][k] for k in ('scene', 'agent', 'lon_gt',
                                                 'other_caution_agents', 'traffic_light')},
                     lon_model=LON_CLASSES[int(base['lon'][i])],
                     model_braking=bool(int(base['lon'][i]) in SLOW_IDX),
                     mass_share=float(base['mt'][i, tgt_i].sum()
                                      / base['mt'][i].sum().clamp(min=1e-9)),
                     base_lon=LON_CLASSES[int(base['lon'][i])],
                     base_slow=float(fam(base['P'], SLOW_IDX)[i]),
                     has_ctrl=bool(has_ctl[i]))
            for nm, o in (('edge', edge), ('swapR', swapR), ('swapA', swapA),
                          ('swapC', swapC), ('graph', graph), ('enc', encr),
                          ('ctrl', ctrl), ('inject', inject), ('ginj', ginj)):
                if nm in ('ctrl', 'inject', 'ginj') and not has_ctl[i]:
                    continue
                r[nm] = dict(
                    lon=LON_CLASSES[int(o['lon'][i])],
                    flip=bool(o['lon'][i] != base['lon'][i]),
                    unbrake=bool((int(base['lon'][i]) in SLOW_IDX)
                                 and (int(o['lon'][i]) not in SLOW_IDX)),
                    dslow=float(fam(o['P'], SLOW_IDX)[i] - fam(base['P'], SLOW_IDX)[i]),
                    dgo=float(fam(o['P'], GO_IDX)[i] - fam(base['P'], GO_IDX)[i]),
                    dv=float(o['v'][i] - base['v'][i]),
                    dplan=float((o['plan'][i] - base['plan'][i]).norm(dim=-1).mean()))
            rows.append(r)

    # ---- rapor ----
    def agg(rs, key):
        d = [r[key] for r in rs if key in r]
        if not d:
            return None
        n = len(d)
        return dict(n=n,
                    flip=100 * sum(x['flip'] for x in d) / n,
                    unbrake=100 * sum(x['unbrake'] for x in d) / n,
                    dslow=np.mean([x['dslow'] for x in d]),
                    dgo=np.mean([x['dgo'] for x in d]),
                    dv=np.mean([x['dv'] for x in d]),
                    dplan=np.mean([x['dplan'] for x in d]))

    def table(rs, tag):
        print(f"\n--- {tag} (n={len(rs)}) ---")
        print(f"{'mudahale':8s}{'n':>5s}{'flip%':>8s}{'unbrake%':>10s}{'dP_SLOW':>9s}"
              f"{'dP_GO':>8s}{'dv_end':>8s}{'dplan':>7s}")
        for key, nm in (('edge', 'edge'), ('swapR', 'swapR'), ('swapA', 'swapA'),
                        ('swapC', 'swapC'), ('graph', 'graph'), ('enc', 'enc'),
                        ('ctrl', 'ctrl'), ('inject', 'inject'), ('ginj', 'ginj')):
            g = agg(rs, key)
            if g is None:
                continue
            print(f"{nm:8s}{g['n']:>5d}{g['flip']:>7.1f}%{g['unbrake']:>9.1f}%"
                  f"{g['dslow']:>+9.3f}{g['dgo']:>+8.3f}{g['dv']:>+8.3f}{g['dplan']:>7.3f}")

    print("\n=== BINDING TEST — " + a.causal_path.split('/')[-2] + " ===")
    print("beklenti: edge/graph/enc -> unbrake YUKSEK, dP_SLOW NEGATIF, dv_end POZITIF; "
          "ctrl -> ~0\n(enc = tavan; graph-enc farki = frozen-backbone arka-kanali)")
    braking = [r for r in rows if r['model_braking']]
    table(braking, "ANA DILIM: model tabanda frenliyor")
    table(rows, "TUM SET")
    med = float(np.median([r['mass_share'] for r in rows]))
    table([r for r in braking if r['mass_share'] >= med], f"frenliyor & kutle payi >= {med:.2f}")
    table([r for r in braking if r['mass_share'] < med], f"frenliyor & kutle payi < {med:.2f}")
    clean = [r for r in braking if not r['other_caution_agents'] and not r['traffic_light']]
    table(clean, "EN TEMIZ: frenliyor & 2. sebep yok & isik yok")
    # REAKTIF dilim — MODEL-BAGIMSIZ on-kosul (set filtresi adayi, 2026-08-24 analizi):
    # uzman ACIL frenlemis (lon_gt stop_quickly/slow_quickly). Kuyruk/durumsal duraklamalar
    # (nazik stop, sabit kuyruk lideri) elenir; ajan-kaynakli REAKTIF frenler kalir.
    URG = ('stop_quickly', 'slow_quickly')
    table([r for r in braking if r['lon_gt'] in URG], "REAKTIF: uzman ACIL frenlemis")
    table([r for r in braking if r['lon_gt'] not in URG], "DURUMSAL: uzman nazik frenlemis")
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)
        print(f"\n  -> sahne-basina kayit: {a.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_path", required=True)
    p.add_argument("--causal_path", required=True)
    p.add_argument("--valid_set", required=True)
    p.add_argument("--set_json", type=str, default="binding_set_v6.json")
    p.add_argument("--out", type=str, default="binding_test_results.json")
    p.add_argument("--rel_bottleneck", type=int, default=0)
    p.add_argument("--rel_evidence", type=int, default=0)
    p.add_argument("--num_neighbors", type=int, default=10)
    p.add_argument("--graph_layers", type=int, default=1)
    p.add_argument("--nbr_enrich", type=int, default=2)
    p.add_argument("--typed_kv", type=int, default=1)
    p.add_argument("--dod_meta", type=int, default=1)
    p.add_argument("--lon_merge", type=int, default=0)
    p.add_argument("--ego_residual", type=int, default=0)
    p.add_argument("--modes", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", type=str, default="cuda:1")
    main(p.parse_args())
