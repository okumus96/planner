"""b*-swap: karar slotu YUK TASIYOR mu? (H'nin falsification deneyi)

Iki olcum:
  1. AGREEMENT: modelin urettigi plani decision_labels() ile YENIDEN etiketle, ilan ettigi
     b* = (lon, lat) ile karsilastir -> "planner ilan ettigi kararla %X uyumlu"
     ("dekoratif karar" itirazinin cevabi).
  2. SWAP/COMPLIANCE: f_cas ve ego_clean SABIT, head'e ZORLANMIS karar ver:
     head(f_cas, ego_clean, (lon', lat')). Zorlanmis plani yeniden etiketle ->
     compliance = relabel == zorlanan sinif (yalniz zorlanan != ilan edilen sahnelerde).
     Ek: doz-yon kontrolleri (stop/slow zorla -> uc hiz dusuyor mu; turn zorla -> heading
     dogru yone donuyor mu) + Delta-plan buyuklugu.

Aile-duzeyi compliance ana okuma: hareket halindeki ego'ya "remain_stopped" zorlansa en iyi
ihtimalle stop_* olarak etiketlenir (4 s'de fiziksel durus) -> tam-sinif compliance yapisal
olarak cezali; aile {stop-ish, slow, accel, maintain, reverse} adil olcek.

Kosum (v2 ckpt, cuda:1):
  python eval_bswap.py --pretrained_path training_log/normal/model_epoch_19_valADE_1.6487.pth \
    --causal_path training_log/dodmeta_typed_noresid_v2/causal_epoch_13_minADE_0.7253.pth \
    --valid_set /home/lt-hta-ai4/ssd1/nuplan/processed_data/validation \
    --graph_layers 1 --nbr_enrich 2 --ego_residual 0 --gate_channels 1 --typed_kv 1 \
    --dod_meta 1 --device cuda:1
"""
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from GameFormer.predictor import GameFormer
from GameFormer.causal_graph import CausalPlanner
from GameFormer.train_utils import DrivingData
from GameFormer.decision_labels import (decision_labels, LON_CLASSES, LAT_CLASSES, NUM_LON, NUM_LAT,
                                        LON5_MAP, LAT5_MAP, LON5_CLASSES, LAT5_CLASSES,
                                        NUM_LON5, NUM_LAT5,
                                        LON4_MAP, LAT5V_MAP, LON4_CLASSES, LAT5V_CLASSES,
                                        NUM_LON4, NUM_LAT5V)
from train_planner import read_batch, extract_neighbor_top1_futures, freeze_gameformer

# aile katlamasi: hareketli ego'ya zorlanmis "dur" ailesi tek grupta adil olculur
LON_FAMILY = {0: 'stop-ish', 1: 'stop-ish', 2: 'stop-ish', 3: 'slow', 4: 'slow',
              5: 'accel', 6: 'accel', 7: 'maintain', 8: 'reverse'}
LAT_FAMILY = {0: 'turn_left', 1: 'turn_right', 2: 'left-ish', 3: 'right-ish',
              4: 'left-ish', 5: 'right-ish', 6: 'no_lateral'}
# dec_moe (5x5 sozluk) karsiliklari: relabel 9x7'den LON5_MAP/LAT5_MAP ile katlanir
LON5_FAM = {0: 'stop-ish', 1: 'stop-ish', 2: 'slow', 3: 'accel', 4: 'maintain'}
LAT5_FAM = {0: 'turn_left', 1: 'turn_right', 2: 'left-ish', 3: 'right-ish', 4: 'no_lateral'}

# --- 4x5 GORUNUM (kullanici sozlugu, 2026-08-25): training'siz degerlendirme katlamasi ---
# lon 4: stop{remain_stopped,stop_q,stop_g} / slow / accel / maintain (reverse DISLANIR);
# lat 5: turn_left / turn_right / to_left{lc_l,inlane_l} / to_right{lc_r,inlane_r} / none.
# Sinif = aile -> ayri aile katmani kalmaz. Zorlama modelin KENDI sozlugunde yapilir,
# SAYIM bu katlanmis uzayda. Not: bu "4x5 ile egitilmis modelin sayilari" degil,
# "4x5 degerlendirme gorunumu"dur.
LON4_NAMES = ['stop', 'slow', 'accel', 'maintain']
LAT5V_NAMES = ['turn_left', 'turn_right', 'to_left', 'to_right', 'none']
LON9_FOLD = [0, 0, 0, 1, 1, 2, 2, 3, -1]      # 9-sinif -> 4 (reverse -> -1 = disla)
LAT7_FOLD = [0, 1, 2, 3, 2, 3, 4]             # 7-sinif -> 5
LON5_FOLD = [0, 0, 1, 2, 3]                   # moe 5-sinif -> 4
LAT5_FOLD = [0, 1, 2, 3, 4]                   # moe 5-sinif -> 5 (birebir)


def plan_with_heading(xy):
    """[B,80,2] tahmin xy -> [B,80,3] (x,y,heading); heading ardisik farklardan
    (deployment'taki plan donusumuyle ayni yontem)."""
    d = xy[:, 1:] - xy[:, :-1]
    hd = torch.atan2(d[..., 1], d[..., 0])
    hd = torch.cat([hd[:, :1], hd], dim=1)
    return torch.cat([xy, hd.unsqueeze(-1)], dim=-1)


def end_speed(xy):
    """[B,80,2] -> [B] lon penceresi sonunda (3.5-4.0 s) ortalama hiz."""
    v = (xy[:, 1:] - xy[:, :-1]).norm(dim=-1) / 0.1
    return v[:, 34:40].mean(dim=1)


def start_speed(xy):
    """[B,80,2] -> [B] plan basinda (0-0.5 s) ortalama hiz — fizibilite icin."""
    v = (xy[:, 1:] - xy[:, :-1]).norm(dim=-1) / 0.1
    return v[:, :5].mean(dim=1)


def lon_feasible(name, v0):
    """Zorlanan lon sinifi bu sahnede KINEMATIK olarak uyulabilir mi? (fizibilite tavani)
    remain_stopped: zaten durukken (v0<0.5) — hareketliyken 4 s'de relabel ancak stop_* olur.
    stop_*: yavaslayacak hiz olmali (v0>=1.0). slow_*: -1 m/s bandi icin v0>=2.0.
    accel/maintain: her zaman. reverse: sozlukten cikarilmis sayilir (hicbir sahnede)."""
    if name == 'remain_stopped':
        return v0 < 0.5
    if name.startswith('stop'):
        return v0 >= 1.0
    if name.startswith('slow'):
        return v0 >= 2.0
    if name == 'reverse':
        return torch.zeros_like(v0, dtype=torch.bool)
    return torch.ones_like(v0, dtype=torch.bool)


def net_heading(xy):
    """[B,80,2] -> [B] net yon degisimi (ilk->son gecerli segment)."""
    d = xy[:, 1:] - xy[:, :-1]
    hd = torch.atan2(d[..., 1], d[..., 0])
    return torch.atan2(torch.sin(hd[:, -1] - hd[:, 0]), torch.cos(hd[:, -1] - hd[:, 0]))


@torch.no_grad()
def main(args):
    dev = args.device
    gameformer = GameFormer(encoder_layers=3, decoder_levels=2, neighbors=args.num_neighbors)
    gameformer.load_state_dict(torch.load(args.pretrained_path, map_location=dev))
    gameformer = gameformer.to(dev); freeze_gameformer(gameformer)
    causal = CausalPlanner(layers=args.graph_layers, modes=args.modes, nbr_enrich=args.nbr_enrich,
                           gate=args.gate, ego_residual=args.ego_residual,
                           gate_channels=args.gate_channels, typed_kv=args.typed_kv,
                           channel_evidence=args.channel_evidence, gate_trust=args.gate_trust,
                           dod_meta=args.dod_meta, dec_moe=args.dec_moe, lat_moe=args.lat_moe,
                           num_lon=(NUM_LON4 if args.lat_moe else NUM_LON5 if args.dec_moe
                                    else 6 if args.lon_merge else 9),
                           num_lat=(NUM_LAT5V if args.lat_moe
                                    else NUM_LAT5 if args.dec_moe else 7)).to(dev)
    miss, unexp = causal.load_state_dict(torch.load(args.causal_path, map_location=dev), strict=False)
    if miss or unexp:
        print(f"[load] missing={list(miss)}  unexpected={list(unexp)}")
    causal.eval()
    assert causal.dod_meta, "b*-swap dod_meta ckpt ister (factored karar slotu)"

    # sinif uzayi: dec_moe'de model 5x5 konusur; decision_labels() 9x7 dondurur -> relabel'lar
    # LON5_MAP/LAT5_MAP ile katlanip modelin uzayinda karsilastirilir. Zorlanan karar head'e
    # girince aile-dallari OTOMATIK degisir (head, lon/lat ailesini b'dan turetir) -- yani bu
    # test dec_moe'de tam olarak "salter cevirince baska devre calisiyor mu"yu olcer.
    if args.lat_moe:
        # lat_moe: model 4x5 konusur; 4x5 gorunumle BIREBIR ayni uzay -> fold = kimlik.
        n_lon, n_lat = NUM_LON4, NUM_LAT5V
        cls_lon, cls_lat = LON4_CLASSES, LAT5V_CLASSES
        fam_lon = {i: LON4_CLASSES[i] for i in range(NUM_LON4)}
        fam_lat = {i: LAT5V_CLASSES[i] for i in range(NUM_LAT5V)}
        lon_map = torch.tensor(LON4_MAP, dtype=torch.long)
        lat_map = torch.tensor(LAT5V_MAP, dtype=torch.long)
        slow_cls, acc_cls = (0, 1), (2,)          # stop, slow -> hiz dussun; accel -> artsin
        lon_fold = torch.arange(NUM_LON4)
        lat_fold = torch.arange(NUM_LAT5V)
    elif args.dec_moe:
        n_lon, n_lat = NUM_LON5, NUM_LAT5
        cls_lon, cls_lat = LON5_CLASSES, LAT5_CLASSES
        fam_lon, fam_lat = LON5_FAM, LAT5_FAM
        lon_map = torch.tensor(LON5_MAP, dtype=torch.long)
        lat_map = torch.tensor(LAT5_MAP, dtype=torch.long)
        slow_cls, acc_cls = (1, 2), (3,)          # stop, slow -> hiz dussun; accel -> artsin
        lon_fold = torch.tensor(LON5_FOLD, dtype=torch.long)
        lat_fold = torch.tensor(LAT5_FOLD, dtype=torch.long)
    else:
        n_lon, n_lat = NUM_LON, NUM_LAT
        cls_lon, cls_lat = LON_CLASSES, LAT_CLASSES
        fam_lon, fam_lat = LON_FAMILY, LAT_FAMILY
        lon_map = lat_map = None
        slow_cls, acc_cls = (1, 2, 3, 4), (5, 6)
        lon_fold = torch.tensor(LON9_FOLD, dtype=torch.long)
        lat_fold = torch.tensor(LAT7_FOLD, dtype=torch.long)

    ds = DrivingData(args.valid_set + "/*.npz", args.num_neighbors)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    n = 0
    agree_lon = agree_lat = agree_joint = 0
    # compliance sayaclari: [zorlanan][yeniden-etiket]
    C_lon = np.zeros((n_lon, n_lon), dtype=int)
    C_lat = np.zeros((n_lat, n_lat), dtype=int)
    lon_forced_n = np.zeros(n_lon, dtype=int)
    lat_forced_n = np.zeros(n_lat, dtype=int)
    dv = {c: [] for c in range(n_lon)}           # zorlanan lon -> uc-hiz farki (forced - base)
    dhd = {c: [] for c in range(n_lat)}          # zorlanan lat -> net heading (forced)
    dade = {('lon', c): [] for c in range(n_lon)}
    dade.update({('lat', c): [] for c in range(n_lat)})

    # --- fizibilite + any-mode sayaclari (teshis 2026-08-25) ---
    # fizibilite: zorlama kinematik olarak uyulabilir sahnelerde mi olculuyor?
    # any-mode: 6 moddan HERHANGI biri uyuyor mu? (best-mode >> any-mode farki = MOD SECICI
    # suclu: decoder itaatkar plan uretiyor ama skor basi baglam-tercihli modu seciyor.)
    fam_lon_match = {c: torch.tensor([fam_lon[r] == fam_lon[c] for r in range(n_lon)])
                     for c in range(n_lon)}
    fam_lat_match = {c: torch.tensor([fam_lat[r] == fam_lat[c] for r in range(n_lat)])
                     for c in range(n_lat)}
    feas_n = np.zeros(n_lon, dtype=int)          # fizibil zorlama sayisi
    feas_fam = np.zeros(n_lon, dtype=int)        # fizibil & best-mode aile uyumu
    any_fam_lon = np.zeros(n_lon, dtype=int)     # any-mode aile uyumu (tum zorlamalar)
    anyfeas_fam = np.zeros(n_lon, dtype=int)     # fizibil & any-mode aile uyumu
    any_fam_lat = np.zeros(n_lat, dtype=int)     # lat: any-mode aile uyumu

    # --- 4x5 gorunum sayaclari: zorlanan AILE bazinda havuzlanir ---
    n4 = np.zeros(4, dtype=int); arg4 = np.zeros(4, dtype=int); cc4 = np.zeros(4, dtype=int)
    n5 = np.zeros(5, dtype=int); arg5 = np.zeros(5, dtype=int); cc5 = np.zeros(5, dtype=int)
    # 4x5 agreement: argmax-secimli plan vs KARAR-TUTARLI secimli plan (taban, zorlamasiz)
    ag4_lon = ag4_lat = ag4_joint = 0            # argmax secim
    cc_ag_lon = cc_ag_lat = cc_ag_joint = 0      # karar-tutarli secim (any-mode, ilan ile)

    for batch in loader:
        inputs, ego_future, _, ref_path = read_batch(batch, dev)
        enc = gameformer.encoder(inputs)
        top1, nbr_states, _ = extract_neighbor_top1_futures(gameformer, enc, args.num_neighbors)
        out = causal(enc, inputs, num_agents=args.num_neighbors + 1,
                     neighbor_futures=top1, neighbor_states=nbr_states)
        B = ego_future.shape[0]
        f_cas, ego_clean = out['f_cas'], out['ego_clean']
        b_lon = out['psi_lon_cas'].argmax(-1)                        # [B]
        b_lat = out['psi_lat_cas'].argmax(-1)

        traj0 = out['traj'][:, 0]                                    # [B,M,80,4]
        best0 = out['score'][:, 0].argmax(-1)
        plan0 = traj0[torch.arange(B), best0][..., :2]               # [B,80,2]

        # --- 1. agreement: ilan edilen karar vs planin yeniden etiketi ---
        rl_lon, rl_lat = decision_labels(plan_with_heading(plan0).cpu(), ref_path.cpu())
        if lon_map is not None:                                      # dec_moe: 9x7 -> 5x5
            rl_lon, rl_lat = lon_map[rl_lon], lat_map[rl_lat]
        rl_lon, rl_lat = rl_lon.to(dev), rl_lat.to(dev)
        agree_lon += int((rl_lon == b_lon).sum())
        agree_lat += int((rl_lat == b_lat).sum())
        agree_joint += int(((rl_lon == b_lon) & (rl_lat == b_lat)).sum())
        v0 = end_speed(plan0)
        vstart = start_speed(plan0)                                  # fizibilite icin
        M = out['traj'].shape[2]
        ref_rep = ref_path.cpu().repeat_interleave(M, dim=0)         # any-mode relabel icin

        # --- 4x5 agreement (taban): argmax-secim ve karar-tutarli secim ---
        fb_lon = lon_fold[b_lon.cpu()]                               # [B] ilanin 4'lu ailesi
        fb_lat = lat_fold[b_lat.cpu()]
        fr_lon = lon_fold[rl_lon.cpu()]                              # argmax planin relabel ailesi
        fr_lat = lat_fold[rl_lat.cpu()]
        ok_l, ok_t = (fr_lon == fb_lon), (fr_lat == fb_lat)
        ag4_lon += int(ok_l.sum()); ag4_lat += int(ok_t.sum())
        ag4_joint += int((ok_l & ok_t).sum())
        allm0 = out['traj'][:, 0][..., :2].reshape(B * M, 80, 2)     # taban 6 mod
        r0l, r0t = decision_labels(plan_with_heading(allm0).cpu(), ref_rep)
        if lon_map is not None:
            r0l, r0t = lon_map[r0l], lat_map[r0t]
        f0l = lon_fold[r0l].view(B, M); f0t = lat_fold[r0t].view(B, M)
        okA_l = (f0l == fb_lon[:, None]); okA_t = (f0t == fb_lat[:, None])
        cc_ag_lon += int(okA_l.any(1).sum()); cc_ag_lat += int(okA_t.any(1).sum())
        cc_ag_joint += int((okA_l & okA_t).any(1).sum())             # AYNI mod ikisine de uymali

        # --- 2. swap: her lon sinifini zorla (lat = ilan edilen), sonra her lat sinifini ---
        for c in range(n_lon):
            fc = torch.full_like(b_lon, c)
            trajF, scoreF = causal.head(f_cas, ego_clean, (fc, b_lat))
            bestF = scoreF[:, 0].argmax(-1)
            planF = trajF[:, 0][torch.arange(B), bestF][..., :2]
            rlF, _ = decision_labels(plan_with_heading(planF).cpu(), ref_path.cpu())
            if lon_map is not None:
                rlF = lon_map[rlF]
            mask = (b_lon != c)                                      # yalniz gercek zorlamalar
            idx = mask.nonzero().flatten()
            for i in idx.tolist():
                C_lon[c, int(rlF[i])] += 1
            lon_forced_n[c] += int(mask.sum())
            dv[c] += (end_speed(planF) - v0)[mask].tolist()
            dade[('lon', c)] += (planF - plan0).norm(dim=-1).mean(-1)[mask].tolist()

            # --- fizibilite + any-mode (lon) ---
            fmask = fam_lon_match[c]
            best_ok = fmask[rlF.cpu()]                               # [B] best-mode aile uyumu
            allm = trajF[:, 0][..., :2].reshape(B * M, 80, 2)        # [B*M,80,2] tum modlar
            rlA, _ = decision_labels(plan_with_heading(allm).cpu(), ref_rep)
            if lon_map is not None:
                rlA = lon_map[rlA]
            any_ok = fmask[rlA].view(B, M).any(dim=1)                # [B] herhangi bir mod uyar mi
            feas = lon_feasible(cls_lon[c], vstart).cpu() & mask.cpu()
            feas_n[c] += int(feas.sum())
            feas_fam[c] += int((best_ok & feas).sum())
            any_fam_lon[c] += int((any_ok & mask.cpu()).sum())
            anyfeas_fam[c] += int((any_ok & feas).sum())

            # --- 4x5 gorunum (lon): zorlanan aile bazinda havuzla ---
            f4 = int(lon_fold[c])
            if f4 >= 0:
                m4 = (lon_fold[b_lon.cpu()] != f4) & mask.cpu()      # ilan zaten ailedeyse zorlama degil
                arg_ok4 = (lon_fold[rlF.cpu()] == f4)
                any_ok4 = (lon_fold[rlA].view(B, M) == f4).any(dim=1)
                n4[f4] += int(m4.sum())
                arg4[f4] += int((arg_ok4 & m4).sum())
                cc4[f4] += int((any_ok4 & m4).sum())
        for c in range(n_lat):
            fc = torch.full_like(b_lat, c)
            trajF, scoreF = causal.head(f_cas, ego_clean, (b_lon, fc))
            bestF = scoreF[:, 0].argmax(-1)
            planF = trajF[:, 0][torch.arange(B), bestF][..., :2]
            _, rlF = decision_labels(plan_with_heading(planF).cpu(), ref_path.cpu())
            if lat_map is not None:
                rlF = lat_map[rlF]
            mask = (b_lat != c)
            idx = mask.nonzero().flatten()
            for i in idx.tolist():
                C_lat[c, int(rlF[i])] += 1
            lat_forced_n[c] += int(mask.sum())
            dhd[c] += net_heading(planF)[mask].tolist()
            dade[('lat', c)] += (planF - plan0).norm(dim=-1).mean(-1)[mask].tolist()

            # --- any-mode (lat; fizibilite yok — yol geometrisi kinematikten okunamaz) ---
            allm = trajF[:, 0][..., :2].reshape(B * M, 80, 2)
            _, rlA = decision_labels(plan_with_heading(allm).cpu(), ref_rep)
            if lat_map is not None:
                rlA = lat_map[rlA]
            any_ok = fam_lat_match[c][rlA].view(B, M).any(dim=1)
            any_fam_lat[c] += int((any_ok & mask.cpu()).sum())

            # --- 4x5 gorunum (lat) ---
            f5 = int(lat_fold[c])
            m5 = (lat_fold[b_lat.cpu()] != f5) & mask.cpu()
            arg_ok5 = (lat_fold[rlF.cpu()] == f5)
            any_ok5 = (lat_fold[rlA].view(B, M) == f5).any(dim=1)
            n5[f5] += int(m5.sum())
            arg5[f5] += int((arg_ok5 & m5).sum())
            cc5[f5] += int((any_ok5 & m5).sum())

        n += B
        if args.limit and n >= args.limit:
            break

    print(f"\n=== b*-swap — {args.causal_path.split('/')[-2]} ({n} sahne) ===")
    print("\n--- 1. AGREEMENT (ilan edilen karar vs planin yeniden etiketi) ---")
    print(f"  lon: {100.0 * agree_lon / n:.1f}%   lat: {100.0 * agree_lat / n:.1f}%   "
          f"joint: {100.0 * agree_joint / n:.1f}%")

    def fam_comp(C, forced_n, classes, family):
        rows = []
        for c in range(len(classes)):
            tot = forced_n[c]
            if tot == 0:
                continue
            exact = C[c, c] / tot
            fam = sum(C[c, r] for r in range(len(classes)) if family[r] == family[c]) / tot
            rows.append((classes[c], tot, exact, fam))
        return rows

    print("\n--- 2. COMPLIANCE (zorlanan sinif -> zorlanmis planin etiketi; zorlanan != ilan) ---")
    print(f"  {'zorlanan (lon)':16s} {'n':>6s} {'tam':>7s} {'aile':>7s}   ek: uc-hiz farki (m/s), Δplan (m)")
    for name, tot, ex, fa in fam_comp(C_lon, lon_forced_n, cls_lon, fam_lon):
        c = cls_lon.index(name)
        mdv = np.mean(dv[c]) if dv[c] else 0.0
        mad = np.mean(dade[('lon', c)]) if dade[('lon', c)] else 0.0
        print(f"  {name:16s} {tot:6d} {100*ex:6.1f}% {100*fa:6.1f}%   Δv_end={mdv:+.2f}  Δplan={mad:.2f}")
    print(f"\n  {'zorlanan (lat)':16s} {'n':>6s} {'tam':>7s} {'aile':>7s}   ek: net heading (rad), Δplan (m)")
    for name, tot, ex, fa in fam_comp(C_lat, lat_forced_n, cls_lat, fam_lat):
        c = cls_lat.index(name)
        mhd = np.mean(dhd[c]) if dhd[c] else 0.0
        mad = np.mean(dade[('lat', c)]) if dade[('lat', c)] else 0.0
        print(f"  {name:16s} {tot:6d} {100*ex:6.1f}% {100*fa:6.1f}%   hd={mhd:+.2f}  Δplan={mad:.2f}")

    # doz-yon ozetleri
    slow_ok = np.mean([x < 0 for c in slow_cls for x in dv[c]]) if any(dv[c] for c in slow_cls) else 0
    acc_ok = np.mean([x > 0 for c in acc_cls for x in dv[c]]) if any(dv[c] for c in acc_cls) else 0
    tl_ok = np.mean([x > 0 for x in dhd[0]]) if dhd[0] else 0
    tr_ok = np.mean([x < 0 for x in dhd[1]]) if dhd[1] else 0
    print("\n--- 3. DOZ-YON (zorlamaya SEMANTIK dogru tepki orani) ---")
    print(f"  stop/slow zorla -> hiz dusuyor : {100*slow_ok:.1f}%")
    print(f"  accel zorla     -> hiz artiyor : {100*acc_ok:.1f}%")
    print(f"  turn_left zorla -> sola donuyor: {100*tl_ok:.1f}%")
    print(f"  turn_right zorla-> saga donuyor: {100*tr_ok:.1f}%")

    print("\n=== 4. 4x5 GORUNUM (sinif=aile; reverse dislandi; training'siz katlama) ===")
    print("  argmax = skorun sectigi mod | karar-tutarli = ilana/zorlamaya uyan modlar icinden"
          " en yuksek skorlu (uyan yoksa argmax'a duser; uyum orani = any-mode)")
    print("\n--- 4a. AGREEMENT (taban plan, ilan ile ayni ailede mi) ---")
    print(f"  argmax secim       : lon {100*ag4_lon/n:5.1f}%  lat {100*ag4_lat/n:5.1f}%  "
          f"joint {100*ag4_joint/n:5.1f}%")
    print(f"  karar-tutarli secim: lon {100*cc_ag_lon/n:5.1f}%  lat {100*cc_ag_lat/n:5.1f}%  "
          f"joint {100*cc_ag_joint/n:5.1f}%")
    print("\n--- 4b. COMPLIANCE (zorlanan aile -> plan o ailede mi) ---")
    print(f"  {'zorlanan (lon)':16s} {'n':>6s} {'argmax':>8s} {'karar-tutarli':>14s}")
    for f in range(4):
        if n4[f] == 0:
            continue
        print(f"  {LON4_NAMES[f]:16s} {n4[f]:6d} {100*arg4[f]/n4[f]:7.1f}% {100*cc4[f]/n4[f]:13.1f}%")
    print(f"\n  {'zorlanan (lat)':16s} {'n':>6s} {'argmax':>8s} {'karar-tutarli':>14s}")
    for f in range(5):
        if n5[f] == 0:
            continue
        print(f"  {LAT5V_NAMES[f]:16s} {n5[f]:6d} {100*arg5[f]/n5[f]:7.1f}% {100*cc5[f]/n5[f]:13.1f}%")

    if args.feas:
        # DIPNOT (oncelik disi, kullanici karari 2026-08-25): fizibilite dilimi — zorlama yalniz
        # kinematik uyulabilir sahnelerde sayilir. Detay: plan.md par.11.
        print("\n--- [dipnot] FIZIBILITE + ANY-MODE (lon; NATIVE aile uyumu) ---")
        print(f"  {'zorlanan':16s} {'n':>6s} {'n_feas':>7s} {'aile|feas':>10s} {'aile(any)':>10s}"
              f" {'any&feas':>9s}")
        for c in range(n_lon):
            if lon_forced_n[c] == 0 or cls_lon[c] == 'reverse':
                continue
            nf = feas_n[c]
            pf = 100 * feas_fam[c] / nf if nf else 0.0
            pa = 100 * any_fam_lon[c] / lon_forced_n[c]
            paf = 100 * anyfeas_fam[c] / nf if nf else 0.0
            print(f"  {cls_lon[c]:16s} {lon_forced_n[c]:6d} {nf:7d} {pf:9.1f}% {pa:9.1f}% {paf:8.1f}%")
        print(f"\n  {'zorlanan (lat)':16s} {'n':>6s} {'aile(best)':>11s} {'aile(any)':>10s}")
        for c in range(n_lat):
            if lat_forced_n[c] == 0:
                continue
            pb = (100 * sum(C_lat[c, r] for r in range(n_lat) if fam_lat[r] == fam_lat[c])
                  / lat_forced_n[c])
            pa = 100 * any_fam_lat[c] / lat_forced_n[c]
            print(f"  {cls_lat[c]:16s} {lat_forced_n[c]:6d} {pb:10.1f}% {pa:9.1f}%")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="b*-swap: karar slotunun yuk-tasima olcumu")
    p.add_argument("--pretrained_path", required=True)
    p.add_argument("--causal_path", required=True)
    p.add_argument("--valid_set", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--num_neighbors", type=int, default=10)
    p.add_argument("--graph_layers", type=int, default=1)
    p.add_argument("--nbr_enrich", type=int, default=0)
    p.add_argument("--gate_channels", type=int, default=0)
    p.add_argument("--typed_kv", type=int, default=0)
    p.add_argument("--channel_evidence", type=int, default=0)
    p.add_argument("--gate_trust", type=str, default="all", choices=["all", "reliable"])
    p.add_argument("--dod_meta", type=int, default=1)
    p.add_argument("--dec_moe", type=int, default=0)
    p.add_argument("--lat_moe", type=int, default=0)
    p.add_argument("--feas", type=int, default=0,
                   help="dipnot: fizibilite dilimli native tabloyu da bas (oncelik disi)")
    p.add_argument("--lon_merge", type=int, default=0)
    p.add_argument("--ego_residual", type=int, default=1)
    p.add_argument("--gate", type=str, default="softmax", choices=["softmax", "sigmoid"])
    p.add_argument("--modes", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", type=str, default="cuda")
    main(p.parse_args())
