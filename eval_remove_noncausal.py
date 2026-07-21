"""
RemoveNonCausal — causal graph icin kabul testi (post-hoc dogrulama).

Fikir (CausalAgents / CRiTIC): eger M_cas gercekten "ego kararina etki eden" ajanlari
buluyorsa, sahneden:
  - DUSUK-M_cas (non-causal) bir ajani cikarmak -> plan neredeyse degismemeli  (Delta ~ 0)
  - YUKSEK-M_cas (causal) bir ajani cikarmak    -> plan belirgin degismeli      (Delta buyuk)
  - RASTGELE bir ajani cikarmak                  -> arada bir yerde

Ayrim orani = Delta_high / Delta_low. Buyukse graph guvenilir. (Strict causal-only head
sayesinde non-causal cikarimin etkisiz olmasi kismen garanti; ASIL sinav Delta_high'in
buyuk olmasi = head f_cas'i GERCEKTEN kullaniyor ve M_cas etkili ajani dogru siraliyor.)

"Ajani cikar" = disentangler'da o komsunun validity'sini kapat (mask=True) -> M_cas=0,
f_cas'tan tamamen dislanir. ego_clean + harita degismez, yani plan yalnizca f_cas uzerinden
degisir. Delta = baseline plan ile mudahale plani arasindaki ADE (baseline'in en iyi modu sabit).

EK (branch-swap, refiner'siz + closed-loop'suz): ajan CIKARMADAN, sadece f_cas yerine f_cfd'den
plan uretilirse (Delta_branch) ham trajectory ne kadar degisiyor? Bu, viz'de gozlemlenen "confound
graph farkli bir yola kayiyor" izleniminin gercek buyuklugunu, refiner'in ve pure-decoder kalitesinin
karismadigi (offline ADE, aci-dongude simulasyon YOK) buyuk-N bir olcumle test eder:
  Delta_branch >> Delta_low, ~Delta_high'a yakin -> f_cfd gercekten davranis-degistirici (refiner
    onceki closed-loop testlerdeki farki maskeliyor olabilir).
  Delta_branch ~ Delta_low  -> iki dal ham seviyede bile ayni plani uretiyor, sorun refiner degil,
    graph'in/loss'un kendisinde.

EK (DISTANCE-MATCHED KONTROL, ROAR — Hooker et al. NeurIPS'19 elestirisine cevap): "high-M_cas ajani
cikarinca plan degisiyor" bulgusu, o ajan basitce EN YAKIN oldugu icin de ayni sonucu verebilir (ki bu
durumda M_cas sadece mesafeyi tekrar ediyor demektir, causal bilgi degil). Kontrol: high_j ile AYNI
mesafe+goreli-hiz'a sahip ama FARKLI (dusuk) M_cas'li bir komsu ('matched_j') seçilip O cikarilir.
  Delta_matched << Delta_high (matched_j high_j ile GEOMETRIK olarak esdeger olmasina ragmen) ->
    M_cas mesafenin OTESINDE bilgi tasiyor -> iddia savunulabilir.
  Delta_matched ~ Delta_high -> etki byk olcude mesafeden geliyor, M_cas mesafeyi tekrar ediyor.
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
def _plan_and_mcas(causal, enc_out, inputs, Na, top1_fut, nbr_states):
    out = causal(enc_out, inputs, num_agents=Na, neighbor_futures=top1_fut, neighbor_states=nbr_states,
                also_cfd_plan=True)
    traj_xy = out['traj'][:, 0, :, :, :2]                 # [B, K, 80, 2]
    best = out['score'][:, 0].argmax(-1)                  # [B]  en yuksek skorlu mod (cas)
    traj_cfd_xy = out['traj_cfd'][:, 0, :, :, :2]         # [B, K, 80, 2]
    best_cfd = out['score_cfd'][:, 0].argmax(-1)          # [B]  en yuksek skorlu mod (cfd)
    return out, traj_xy, best, traj_cfd_xy, best_cfd


def _pick_distance_matched(rng, spd, valid, high_j, speed_weight=0.5):
    """high_j'ye (mesafe, goreli hiz) olarak EN YAKIN, high_j'nin KENDISI OLMAYAN gecerli komsuyu sec.
    rng/spd [B,N] metre / m/s. Doner: matched_j [B], match_gap [B] (eslesme kalitesi, kucuk=iyi)."""
    range_high = rng.gather(1, high_j[:, None])            # [B,1]
    speed_high = spd.gather(1, high_j[:, None])             # [B,1]
    score = (rng - range_high).abs() + speed_weight * (spd - speed_high).abs()   # [B,N]
    self_mask = torch.zeros_like(valid)
    self_mask.scatter_(1, high_j[:, None], True)
    score = score.masked_fill(~valid | self_mask, 1e9)
    matched_j = score.argmin(-1)                             # [B]
    match_gap = score.gather(1, matched_j[:, None]).squeeze(1)
    return matched_j, match_gap


@torch.no_grad()
def _run_removed(causal, enc_out, inputs, Na, top1_fut, nbr_states, remove_col):
    """remove_col [B] = full-mask sutun indeksi (komsu j -> 1+j); <0 = cikarma yok."""
    enc2 = dict(enc_out)
    mask2 = enc_out['mask'].clone()
    sel = remove_col >= 0
    if sel.any():
        rows = torch.arange(mask2.shape[0], device=mask2.device)[sel]
        mask2[rows, remove_col[sel]] = True
    enc2['mask'] = mask2
    out = causal(enc2, inputs, num_agents=Na, neighbor_futures=top1_fut, neighbor_states=nbr_states)
    return out['traj'][:, 0, :, :, :2], out['score'][:, 0]   # [B,K,80,2], [B,K]


def main(args):
    dev = args.device
    gameformer = GameFormer(encoder_layers=args.encoder_layers, decoder_levels=args.decoder_levels,
                            neighbors=args.num_neighbors)
    gameformer.load_state_dict(torch.load(args.pretrained_path, map_location=dev))
    gameformer = gameformer.to(dev)
    freeze_gameformer(gameformer)

    causal = CausalPlanner(layers=args.graph_layers, modes=args.modes).to(dev)
    causal.load_state_dict(torch.load(args.causal_path, map_location=dev))
    causal.eval()

    valid_set = DrivingData(args.valid_set + "/*.npz", args.num_neighbors)
    loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    stats = {k: [] for k in ['high', 'low', 'rand', 'matched']}
    switch = {k: [] for k in ['high', 'low', 'rand', 'matched']}
    mcas_high, mcas_low, n_samples = [], [], 0
    pooled_m, pooled_d = [], []   # (removed-agent M_cas, plan shift) tum cikarimlar -> kalibrasyon
    branch_d, branch_d_agree, branch_d_disagree = [], [], []  # Delta_branch (cas vs cfd, cikarma YOK)
    mcas_matched, range_high_l, range_matched_l = [], [], []  # distance-matched teshis

    for bi, batch in enumerate(loader):
        if bi >= args.num_batches:
            break
        inputs, ego_future, _, _ = read_batch(batch, dev)
        B = inputs['ego_agent_past'].shape[0]
        Na = args.num_neighbors + 1

        enc_out = gameformer.encoder(inputs)
        top1_fut, nbr_states, _ = extract_neighbor_top1_futures(gameformer, enc_out, args.num_neighbors)

        out, traj_base, best, traj_cfd, best_cfd = _plan_and_mcas(causal, enc_out, inputs, Na, top1_fut, nbr_states)
        M_cas = out['M_cas']                      # [B, N]
        M_cfd = out['M_cfd']                      # [B, N]
        nbr_valid = out['nbr_valid']              # [B, N] bool
        n_valid = nbr_valid.sum(-1)               # [B]
        usable = n_valid >= 2                      # en az 2 gecerli komsu (cikarinca sahne kalsin)

        # baseline plan = baseline'in en iyi modu
        plan_base = traj_base[torch.arange(B, device=dev), best]        # [B, 80, 2]
        plan_cfd = traj_cfd[torch.arange(B, device=dev), best_cfd]      # [B, 80, 2] (f_cfd'nin KENDI en iyi modu)

        # --- Delta_branch: AJAN CIKARMADAN, sadece f_cas yerine f_cfd'den plan uretilirse ---
        d_branch = torch.norm(plan_cfd - plan_base, dim=-1).mean(-1)     # [B] ADE (m)
        big0 = M_cas.masked_fill(~nbr_valid, -1e9)
        smallf0 = M_cfd.masked_fill(~nbr_valid, -1e9)
        top1_cas = big0.argmax(-1); top1_cfd = smallf0.argmax(-1)
        agree = (top1_cas == top1_cfd) & usable                          # ayni ajani en onemli buluyorlar mi
        disagree = (top1_cas != top1_cfd) & usable
        branch_d.append(d_branch[usable].detach().cpu().numpy())
        if agree.any():
            branch_d_agree.append(d_branch[agree].detach().cpu().numpy())
        if disagree.any():
            branch_d_disagree.append(d_branch[disagree].detach().cpu().numpy())

        # sec: en yuksek / en dusuk M_cas (gecerliler arasinda) + rastgele gecerli
        big = M_cas.masked_fill(~nbr_valid, -1e9)
        small = M_cas.masked_fill(~nbr_valid, 1e9)
        high_j = big.argmax(-1)                    # [B] komsu idx
        low_j = small.argmin(-1)                   # [B]
        rnd = torch.rand_like(M_cas).masked_fill(~nbr_valid, -1.0)
        rand_j = rnd.argmax(-1)                     # rastgele gecerli komsu

        # DISTANCE-MATCHED: high_j ile ayni mesafe+goreli-hiz'da, FARKLI (dusuk) M_cas'li komsu
        pos = enc_out['actors'][:, 1:Na, -1]        # [B, N, 5] = x, y, heading, vx, vy (ego-frame)
        rng = torch.norm(pos[..., :2], dim=-1)       # [B, N] mesafe (m)
        spd = torch.norm(pos[..., 3:5], dim=-1)      # [B, N] goreli hiz buyuklugu (m/s)
        matched_j, _ = _pick_distance_matched(rng, spd, nbr_valid, high_j)
        mcas_matched.append(M_cas.gather(1, matched_j[:, None]).squeeze(1)[usable].detach().cpu().numpy())
        range_high_l.append(rng.gather(1, high_j[:, None]).squeeze(1)[usable].detach().cpu().numpy())
        range_matched_l.append(rng.gather(1, matched_j[:, None]).squeeze(1)[usable].detach().cpu().numpy())

        for name, jj in [('high', high_j), ('low', low_j), ('rand', rand_j), ('matched', matched_j)]:
            col = torch.where(usable, jj + 1, torch.full_like(jj, -1))   # komsu j -> sutun 1+j
            traj_i, score_i = _run_removed(causal, enc_out, inputs, Na, top1_fut, nbr_states, col)
            plan_i = traj_i[torch.arange(B, device=dev), best]           # ayni (baseline) modu karsilastir
            delta = torch.norm(plan_i - plan_base, dim=-1).mean(-1)      # [B] ADE (m)
            best_i = score_i.argmax(-1)
            sw = (best_i != best)
            stats[name].append(delta[usable].detach().cpu().numpy())
            switch[name].append(sw[usable].detach().cpu().numpy())
            m_removed = M_cas.gather(1, jj[:, None]).squeeze(1)          # cikarilan ajanin M_cas'i
            pooled_m.append(m_removed[usable].detach().cpu().numpy())
            pooled_d.append(delta[usable].detach().cpu().numpy())

        mcas_high.append(big.gather(1, high_j[:, None])[usable].detach().cpu().numpy())
        mcas_low.append(small.gather(1, low_j[:, None])[usable].detach().cpu().numpy())
        n_samples += int(usable.sum())

    def cat(d):
        return np.concatenate(d) if len(d) else np.array([0.0])

    dh, dl, dr = cat(stats['high']), cat(stats['low']), cat(stats['rand'])
    dm = cat(stats['matched'])
    db = cat(branch_d)
    db_agree = cat(branch_d_agree) if branch_d_agree else np.array([0.0])
    db_disagree = cat(branch_d_disagree) if branch_d_disagree else np.array([0.0])
    print(f"\n=== RemoveNonCausal — {n_samples} samples ({args.causal_path.split('/')[-1]}) ===")
    print(f"  removed agent mean M_cas:  high={cat(mcas_high).mean():.3f}   low={cat(mcas_low).mean():.3f}")
    print(f"\n  plan shift Delta (m, on baseline best-mode):")
    print(f"    remove HIGH-M_cas (causal)     : {dh.mean():.4f}  (median {np.median(dh):.4f})")
    print(f"    remove RANDOM agent            : {dr.mean():.4f}  (median {np.median(dr):.4f})")
    print(f"    remove LOW-M_cas (non-causal)  : {dl.mean():.4f}  (median {np.median(dl):.4f})")
    print(f"    remove DISTANCE-MATCHED (low-M_cas, high_j ile ayni mesafe/hiz) : "
          f"{dm.mean():.4f}  (median {np.median(dm):.4f})")
    ratio = dh.mean() / max(dl.mean(), 1e-6)
    print(f"\n  ratio Delta_high / Delta_low = {ratio:.1f}x   (buyuk = graph guvenilir)")

    print(f"\n  --- DISTANCE-MATCHED KONTROL (ROAR / Hooker et al. NeurIPS'19 elestirisine cevap) ---")
    rh, rm = cat(range_high_l), cat(range_matched_l)
    ratio_matched = dh.mean() / max(dm.mean(), 1e-6)
    print(f"    esleme kalitesi: mean |range_high - range_matched| = {np.abs(rh-rm).mean():.2f} m   "
          f"(kucuk = matched_j GERCEKTEN high_j ile ayni mesafede)")
    print(f"    mean M_cas: high={cat(mcas_high).mean():.3f}  matched={cat(mcas_matched).mean():.3f}   "
          f"(matched'in DUSUK olmasi -> ayni mesafede farkli M_cas'e sahip bir cift bulduk)")
    print(f"    ratio Delta_high / Delta_matched = {ratio_matched:.1f}x   "
          f"(esdeger mesafede bile buyukse -> M_cas mesafenin OTESINDE bilgi tasiyor)")
    if ratio_matched > 2.0:
        print(f"    -> GECTI: ayni geometriye ragmen Delta_high >> Delta_matched, "
              f"'sadece mesafe' itirazi CURUTULDU")
    else:
        print(f"    -> ZAYIF: Delta_high/Delta_matched dusuk, etki buyuk olcude MESAFEDEN geliyor olabilir")

    print(f"\n  --- Delta_branch (AJAN CIKARMADAN, f_cas yerine f_cfd'den plan; refiner YOK, sim YOK) ---")
    print(f"    Delta_branch (cas vs cfd, TUMU)         : {db.mean():.4f}  (median {np.median(db):.4f})  n={len(db)}")
    print(f"    Delta_branch | top1(cas)==top1(cfd)     : {db_agree.mean():.4f}  n={len(db_agree)}")
    print(f"    Delta_branch | top1(cas)!=top1(cfd)     : {db_disagree.mean():.4f}  n={len(db_disagree)}")
    print(f"    kiyas: Delta_high={dh.mean():.4f}  Delta_low={dl.mean():.4f}")
    if db.mean() >= dh.mean() * 0.5:
        print(f"    -> Delta_branch, Delta_high'a YAKIN/BUYUK: cfd gercekten davranis-degistirici "
              f"(refiner onceki closed-loop farki maskeliyor olabilir)")
    else:
        print(f"    -> Delta_branch kucuk (Delta_low'a yakin): iki dal HAM seviyede de benzer plan uretiyor "
              f"(sorun refiner degil, graph/loss'un kendisinde)")
    print(f"  argmax-mode switch rate:  high={cat(switch['high']).mean():.3f}  "
          f"rand={cat(switch['rand']).mean():.3f}  low={cat(switch['low']).mean():.3f}")

    # --- Kalibrasyon: cikarilan ajanin M_cas'i, plan kaymasini ONGORUYOR mu? ---
    pm, pd = cat(pooled_m), cat(pooled_d)
    corr = float(np.corrcoef(pm, pd)[0, 1]) if pm.std() > 0 else 0.0
    print(f"\n  KALIBRASYON  corr(M_cas_removed, Delta) = {corr:+.3f}   (pozitif = M_cas etkiyi ongoruyor)")
    print("  cikarilan ajanin M_cas'ina gore ortalama plan kaymasi:")
    for lo, hi in [(0.0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 1.01)]:
        sel = (pm >= lo) & (pm < hi)
        tag = f"M_cas in [{lo:.1f},{hi:.1f})"
        if sel.sum() > 0:
            print(f"    {tag:22s} n={int(sel.sum()):5d}  Delta={pd[sel].mean():.4f}")
        else:
            print(f"    {tag:22s} n=    0")

    # --- Etkilesimli alt-kume: gercekten causal bir ajan OLAN sahneler (M_cas_high > 0.5) ---
    mh = cat(mcas_high).reshape(-1)
    inter = mh > 0.5
    if inter.sum() > 0:
        print(f"\n  ETKILESIMLI sahneler (removed high-agent M_cas>0.5, n={int(inter.sum())}):")
        print(f"    Delta_high(confident causal) = {dh[inter].mean():.4f}   vs   Delta_low(overall) = {dl.mean():.4f}"
              f"   -> {dh[inter].mean()/max(dl.mean(),1e-6):.1f}x")

    strong = (corr > 0.15) and (ratio > 2.0)
    print(f"\n  VERDICT: {'PASS' if strong else 'WEAK'}  "
          f"(PASS: corr>0.15 ve Delta_high/Delta_low>2x)\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="RemoveNonCausal acceptance test for the causal agent graph")
    p.add_argument("--pretrained_path", required=True, help="frozen GameFormer checkpoint")
    p.add_argument("--causal_path", required=True, help="trained CausalPlanner checkpoint")
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
