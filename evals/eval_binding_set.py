"""BINDING SET secici — v3 (kullanici karari 2026-08-24): GT-temelli secim.

SET v3 = iki filtre (AND), ikisi de basit:
  1. GT: UZMAN FRENLEMIS — npz'deki decision_lon etiketi SLOW ailesinde
     (stop_quickly/stop_gently/slow_quickly/slow_gently). Etiket insan surucunun KAYITLI
     gelecegin'den turetilir (H egitim etiketi) — model/tahmin/kanal isin icinde degil.
     "Frenlemek icin gercek bir sebep vardi"nin zemin geregi bu.
  2. TAM BIR ajan follows VEYA same_lane_ahead yakiyor (model-tarafi kanallar — kenar grafikte
     VAR, yani kesilebilir). Bu iki kanal SIMDIKI kinematikten hesaplanir (tahmin yok) ve isaret
     geregi ONDE olmayi gerektirir -> collision'in "arkadaki araci yakalama" ve tahmin-artefakti
     problemleri yapisal olarak imkansiz.

collision_course secim kanali OLARAK BIRAKILDI (v2 denemesi: tahmin artefaktlari — park halinde
yan arac ds=0'da "carpisiyor", paralel serit yakaliyor, arkadaki araci yakaliyor; UNRELIABLE_
CHANNELS'in varlik sebebi). Hedefin GT-gelecekle dogrulanmis collision durumu TESHIS olarak
raporlanir.

Teshisler (filtre DEGIL; dilimleme icin): modelin kendi karari (uzmanla uyum), sahnede trafik
isigi, diger ajanlarda dikkat iliskisi (collision/merges/VRU), hedefte GT-dogrulanmis collision,
hedefin dikkat-kutle payi, ttc/closing/ds.

Viz: kutu = SIMDIKI kare (gecmis iz yok — sifir-dolgu artefakti). KESIKLI = tahmin (kirmizi:
hedefin GF tahmini, siyah: ego'nun GF plani). DUZ INCE = gercek gelecek (koyu kirmizi: hedefin
GT'si, gri: uzmanin surdugu yol). Uzman yolu frenlemeyi gozle dogrulatir.

Kosum (v6):
  python eval_binding_set.py --pretrained_path training_log/normal/model_epoch_19_valADE_1.6487.pth \
    --causal_path training_log/dodmeta_v6_relev/causal_epoch_18_minADE_0.7139.pth \
    --valid_set /home/lt-hta-ai4/ssd1/nuplan/processed_data/validation \
    --rel_bottleneck 1 --rel_evidence 1 --device cuda:1
Sadece viz (GPU/model gerekmez):
  python eval_binding_set.py --from_json binding_set_v6.json --pretrained_path x --causal_path x \
    --valid_set /home/lt-hta-ai4/ssd1/nuplan/processed_data/validation
"""
import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from GameFormer.predictor import GameFormer
from GameFormer.causal_graph import CausalPlanner
from GameFormer.train_utils import DrivingData
from GameFormer.channels import (compute_channels, CHANNEL_NAMES, CH_COLLISION_COURSE,
                                 CH_FOLLOWS, CH_SAME_LANE_AHEAD, CH_MERGES, CH_VRU,
                                 EV_TTC, EV_CLOSING, EV_DS, MCH_TRAFFIC)
from GameFormer.decision_labels import LON_CLASSES
from train_planner import read_batch, extract_neighbor_top1_futures, freeze_gameformer

AHEAD = [CH_FOLLOWS, CH_SAME_LANE_AHEAD]          # secim kanallari: onde + tahminsiz
OTHER_CAUTION = [CH_COLLISION_COURSE, CH_MERGES, CH_VRU]   # teshis: diger ajanlarda 2. sebep
SLOW_IDX = [LON_CLASSES.index(c) for c in
            ('stop_quickly', 'stop_gently', 'slow_quickly', 'slow_gently')]


@torch.no_grad()
def main(a):
    dev = a.device
    if a.from_json:
        sel = json.load(open(a.from_json))
        print(f"[from_json] {a.from_json}: {len(sel)} sahne — secim atlandi, sadece viz")
        _viz(_spread(sel, 9), a)
        return
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
                         f"unexpected={len(unexp)} — bayraklari kontrol et")
    m.eval()
    ld = DataLoader(DrivingData(a.valid_set + "/*.npz", a.num_neighbors),
                    batch_size=a.batch_size, shuffle=False, num_workers=4)

    n_all = n_f1 = 0
    sel = []
    scene_base = 0
    N = a.num_neighbors
    for batch in ld:
        inp, ef, nf, rp = read_batch(batch, dev)
        if "decision_lon" not in inp:
            raise SystemExit("cache'te decision_lon yok — extract_channels/decision etiketi sart")
        enc = gf.encoder(inp)
        t1, ns, _, ego_plan = extract_neighbor_top1_futures(gf, enc, N, return_ego=True)
        out = m(enc, inp, num_agents=N + 1, neighbor_futures=t1, neighbor_states=ns)
        B = ef.shape[0]; n_all += B
        lon_gt = inp["decision_lon"]                                         # [B] UZMAN etiketi
        lon_md = out['psi_lon_cas'].argmax(-1)                               # [B] modelin karari
        ch = inp["channel_active"] & out['nbr_valid'][..., None]             # [B,N,R] model-tarafi
        mch = inp["map_channel_active"]
        evid = inp["channel_evidence"]
        mt = out['M_cas_typed']
        # TESHIS: GT-gelecekle kanallar (yalniz gelecege-bagimli kanallar degisir)
        ch_gt, _ = compute_channels(inp["neighbor_agents_past"][:, :N],
                                    inp["ego_agent_past"], nf[:, :N, :, :2], rp)

        f1 = torch.zeros(B, dtype=torch.bool, device=dev)                    # uzman frenlemis
        for k in SLOW_IDX:
            f1 |= lon_gt == k
        n_f1 += int(f1.sum())
        is_ahead = ch[..., AHEAD].any(-1)                                    # [B,N]
        f2 = is_ahead.sum(-1) == 1                                           # tam BIR onde-kisit
        for b in (f1 & f2).nonzero().flatten().tolist():
            j = int(is_ahead[b].float().argmax())
            others = ch[b].clone(); others[j] = False
            sel.append(dict(
                scene=scene_base + b, agent=j,
                lon_gt=LON_CLASSES[int(lon_gt[b])], lon_model=LON_CLASSES[int(lon_md[b])],
                model_braking=bool(sum(int(lon_md[b]) == k for k in SLOW_IDX)),
                traffic_light=bool(mch[b, :, MCH_TRAFFIC].any()),
                other_caution_agents=int(others[:, OTHER_CAUTION].any(-1).sum()),
                gt_collision=bool(ch_gt[b, j, CH_COLLISION_COURSE]),
                all_rels=[CHANNEL_NAMES[r] for r in range(ch.shape[-1]) if bool(ch[b, j, r])],
                ttc=round(float(evid[b, j, EV_TTC]), 2),
                closing=round(float(evid[b, j, EV_CLOSING]), 2),
                ds=round(float(evid[b, j, EV_DS]), 1),
                mass_share=round(float(mt[b, j].sum() / mt[b].sum().clamp(min=1e-9)), 4),
                fut=[[round(float(x), 1) for x in p] for p in t1[b, j, 7::8, :2]],
                gt_fut=[[round(float(x), 1) for x in p] for p in nf[b, j, 7::8, :2]],
                ego_fut=[[round(float(x), 1) for x in p] for p in ego_plan[b, 7::8, :2]],
                ego_gt=[[round(float(x), 1) for x in p] for p in ef[b, 7::8, :2]]))
        scene_base += B

    # ---- rapor ----
    print("\n=== BINDING SET v3 — GT-temelli secim ===")
    print(f"  tum sahneler                                : {n_all}")
    print(f"  F1  uzman frenlemis (decision_lon SLOW)     : {n_f1}")
    print(f"  F1&F2  + tam BIR follows/lane_ahead ajani   : {len(sel)}   <== SET")
    nb = sum(s['model_braking'] for s in sel)
    ntl = sum(s['traffic_light'] for s in sel)
    noc = sum(s['other_caution_agents'] > 0 for s in sel)
    ngc = sum(s['gt_collision'] for s in sel)
    print(f"\n  teshisler (filtre DEGIL):")
    print(f"    model de frenliyor (uzmanla ayni aile)    : {nb}/{len(sel)}")
    print(f"    sahnede trafik isigi kaydi                : {ntl}/{len(sel)}")
    print(f"    diger ajanda collision/merges/VRU         : {noc}/{len(sel)}")
    print(f"    hedefte GT-dogrulanmis collision          : {ngc}/{len(sel)}")
    from collections import Counter
    print("    uzman karari :", dict(Counter(s['lon_gt'] for s in sel).most_common()))
    print("    model karari :", dict(Counter(s['lon_model'] for s in sel).most_common(6)))
    ms = np.array([s['mass_share'] for s in sel]) if sel else np.array([0.0])
    print(f"    hedefin kutle payi: medyan={np.median(ms):.3f} q25={np.percentile(ms,25):.3f} "
          f"q75={np.percentile(ms,75):.3f} | pay<0.1: {int((ms<0.1).sum())} sahne")
    if a.json:
        json.dump(sel, open(a.json, "w"), indent=1)
        print(f"  -> set kaydedildi: {a.json} ({len(sel)} sahne)")
    if a.viz and sel:
        _viz(_spread(sel, 9), a)


def _spread(sel, k):
    """Sete yayilmis k ornek (ilk k yerine) — cesitlilik goz kontrolu icin."""
    if len(sel) <= k:
        return sel
    idx = np.linspace(0, len(sel) - 1, k).astype(int)
    return [sel[i] for i in idx]


def _rect(ax, x, y, th, L, W, color, z, lw=1.2, fill=True):
    """Yonlu arac kutusu (merkez x,y; yon th)."""
    import matplotlib.patches as mp
    c, s_ = np.cos(th), np.sin(th)
    dx, dy = -L / 2, -W / 2
    corner = (x + dx * c - dy * s_, y + dx * s_ + dy * c)
    r = mp.Rectangle(corner, L, W, angle=np.degrees(th),
                     facecolor=(color if fill else 'none'), edgecolor=color,
                     alpha=0.85 if fill else 1.0, lw=lw, zorder=z)
    ax.add_patch(r)


def _viz(samples, a):
    """Goz kontrolu BEV (v3). SIYAH kutu = ego, KIRMIZI kutu = follows/lane_ahead yakan TEK ajan,
    GRI = digerleri. Duz ok = simdiki hiz. KESIKLI = tahmin (GF). DUZ INCE cizgi = GERCEK gelecek
    (koyu kirmizi: hedef GT, gri: uzmanin surdugu yol — frenleme burada gorunur)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import matplotlib.patches as mpatches
    ds_ = DrivingData(a.valid_set + "/*.npz", a.num_neighbors)
    fig, axes = plt.subplots(3, 3, figsize=(16.5, 16))
    for ax, s in zip(axes.flat, samples):
        b = ds_[s['scene']]
        ego = np.asarray(b[0]); nbr = np.asarray(b[1]); lanes = np.asarray(b[2])
        for L in lanes:
            pts = L[:, :2][np.abs(L[:, :2]).sum(-1) > 1e-6]
            if len(pts):
                ax.plot(pts[:, 0], pts[:, 1], color='0.9', lw=1.0, zorder=0)
        for j in range(min(nbr.shape[0], a.num_neighbors)):
            cur = nbr[j, -1]
            if np.abs(cur[:2]).sum() < 1e-6:
                continue
            x, y, th = cur[0], cur[1], cur[2]
            vx, vy = cur[3], cur[4]
            Lw, Ww = max(cur[6], 1.0), max(cur[7], 0.6)
            tgt = (j == s['agent'])
            col, z = ('crimson', 5) if tgt else ('0.55', 2)
            _rect(ax, x, y, th, Lw, Ww, col, z)
            sp = float(np.hypot(vx, vy))
            if sp > 0.3:
                ax.annotate('', xy=(x + vx, y + vy), xytext=(x, y), zorder=z + 1,
                            arrowprops=dict(arrowstyle='-|>', color=col, lw=1.6))
            if tgt:
                fut = np.asarray(s['fut']); gtf = np.asarray(s['gt_fut'])
                ax.plot(fut[:, 0], fut[:, 1], '--', color='crimson', lw=1.4, zorder=4)
                ax.plot(gtf[:, 0], gtf[:, 1], '-', color='darkred', lw=1.3, zorder=4)
                rels = '+'.join(r.replace('same_lane_ahead', 'lane_ahead')
                                 .replace('onObservedCollisionCourseWith', 'collision')
                                 .replace('vulnerable_road_user_near_ego_path', 'VRU')
                                for r in s['all_rels'])
                ax.annotate(f"{rels}\nds={s['ds']}m closing={s['closing']}m/s hiz={sp:.1f}m/s"
                            + ("\nGT-collision DOGRULANDI" if s['gt_collision'] else "")
                            + (f"\n2.sebep: {s['other_caution_agents']} ajan"
                               if s['other_caution_agents'] else ""),
                            xy=(x, y), xytext=(x + 3, y + 5.5), fontsize=8, color='crimson',
                            zorder=9, bbox=dict(fc='white', ec='crimson', alpha=0.88, lw=0.8))
        ex, ey, eth = ego[-1, 0], ego[-1, 1], ego[-1, 2]
        evx, evy = ego[-1, 3], ego[-1, 4]
        _rect(ax, ex, ey, eth, 4.6, 1.9, 'black', 6)
        efut = np.asarray(s['ego_fut']); egt = np.asarray(s['ego_gt'])
        ax.plot(efut[:, 0], efut[:, 1], '--', color='black', lw=1.4, zorder=4)
        ax.plot(egt[:, 0], egt[:, 1], '-', color='0.35', lw=1.6, zorder=4)
        esp = float(np.hypot(evx, evy))
        if esp > 0.3:
            ax.annotate('', xy=(ex + evx, ey + evy), xytext=(ex, ey), zorder=7,
                        arrowprops=dict(arrowstyle='-|>', color='black', lw=1.8))
        ax.annotate(f"EGO {esp:.1f}m/s", xy=(ex, ey), xytext=(ex - 13, ey - 6.5), fontsize=8.5,
                    zorder=9, bbox=dict(fc='white', ec='black', alpha=0.85, lw=0.8))
        ax.set_title(f"#{s['scene']}  |  uzman: {s['lon_gt']}  |  model: {s['lon_model']}"
                     f"{'  |  ISIK' if s['traffic_light'] else ''}"
                     f"  |  pay: {s['mass_share']:.2f}", fontsize=9.5)
        ax.set_aspect('equal'); ax.set_xlim(-18, 62); ax.set_ylim(-30, 30)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes.flat[len(samples):]:
        ax.axis('off')
    handles = [
        mpatches.Patch(color='black', label='EGO (kesikli = GF plani, duz gri = UZMANIN yolu)'),
        mpatches.Patch(color='crimson', label='HEDEF — follows/lane_ahead yakan TEK ajan '
                                              '(kesikli = tahmini, duz koyu = GERCEK gelecegi)'),
        mpatches.Patch(color='0.55', label='diger ajanlar'),
        Line2D([0], [0], color='0.9', lw=1.2, label='serit merkez cizgileri'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=2, fontsize=10.5, frameon=True)
    fig.suptitle('BINDING SET v3 — uzman FRENLEMIS + tam BIR follows/lane_ahead ajani\n'
                 '(gri duz cizgi kisaliyor mu = uzman gercekten yavasliyor; '
                 'kirmizi hedef gercekten onde ve engel mi?)', fontsize=13)
    fig.tight_layout(rect=[0, 0.045, 1, 0.955])
    fig.savefig(a.viz, dpi=130)
    print(f"  -> BEV ornekleri: {a.viz}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_path", required=True)
    p.add_argument("--causal_path", required=True)
    p.add_argument("--valid_set", required=True)
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
    p.add_argument("--json", type=str, default="binding_set_v6.json")
    p.add_argument("--from_json", type=str, default=None,
                   help="kayitli seti oku, modeli HIC kosturmadan sadece viz uret")
    p.add_argument("--viz", type=str, default="viz_out/binding_set.png")
    p.add_argument("--device", type=str, default="cuda:1")
    main(p.parse_args())
