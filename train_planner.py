import argparse
import csv
import logging
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from GameFormer.predictor import GameFormer
from GameFormer.causal_graph import CausalPlanner
from GameFormer.train_utils import DrivingData, imitation_loss, initLogging, set_seed


def extract_neighbor_top1_futures(gameformer, encoder_outputs, num_neighbors, return_ego=False):
    """Run frozen decoder, take argmax-mode trajectory per neighbor.
    Returns (top1_futures [B,N,T,2], current_states [B,N,5], valid [B,N]).
    return_ego=True ise ayrica ego'nun top-1 plani [B,T,2] doner (inter index 0). Bu, conflict
    koridoru icin ALTERNATIF kaynak: --ego_corridor gf. Varsayilan kaynak lattice planner referans
    yolu (npz'deki c_lat_candidates) -- GF ciktisina bagimlilik istemedigimiz icin."""
    decoder_outputs, _ = gameformer.decoder(encoder_outputs)
    last_k = max(int(k.split('_')[1]) for k in decoder_outputs if 'interactions' in k)
    inter = decoder_outputs[f'level_{last_k}_interactions']        # [B, N+1, M, T, 4]
    scores = decoder_outputs[f'level_{last_k}_scores']             # [B, N+1, M]

    nbr_inter = inter[:, 1:1 + num_neighbors]                       # [B, N, M, T, 4]
    nbr_scores = scores[:, 1:1 + num_neighbors]                     # [B, N, M]
    best_mod = nbr_scores.argmax(-1)                                # [B, N]
    B, N, M, T, _ = nbr_inter.shape
    g = best_mod.view(B, N, 1, 1, 1).expand(-1, -1, 1, T, 2)
    top1_futures = torch.gather(nbr_inter[..., :2], 2, g).squeeze(2)  # [B, N, T, 2]

    current_states = encoder_outputs['actors'][:, 1:1 + num_neighbors, -1]  # [B, N, 5]
    nbr_valid = ~encoder_outputs['mask'][:, 1:1 + num_neighbors]            # [B, N]

    if return_ego:
        ego_best = scores[:, 0].argmax(-1)                                  # [B]
        g0 = ego_best.view(B, 1, 1, 1).expand(-1, 1, T, 2)
        ego_plan = torch.gather(inter[:, 0, ..., :2], 1, g0).squeeze(1)      # [B, T, 2]
        return top1_futures, current_states, nbr_valid, ego_plan
    return top1_futures, current_states, nbr_valid


def read_batch(batch, device):
    inputs = {
        "ego_agent_past": batch[0].to(device).float(),
        "neighbor_agents_past": batch[1].to(device).float(),
        "map_lanes": batch[2].to(device).float(),
        "map_crosswalks": batch[3].to(device).float(),
        "route_lanes": batch[4].to(device).float(),
    }
    ego_future = batch[5].to(device).float()
    neighbors_future = batch[6].to(device).float()
    c_lat_candidates = batch[7].to(device).float()
    # Predicate kanallari (DrivingData 9-12; channels branch) -- inputs SOZLUGUNUN icinde
    # tasinir ki read_batch imzasi ve cagri yerleri degismesin (model inputs'u zaten aliyor).
    if len(batch) > 12:
        inputs["channel_active"] = batch[9].to(device).bool()
        inputs["channel_evidence"] = batch[10].to(device).float()
        inputs["map_channel_active"] = batch[11].to(device).bool()
        inputs["map_channel_evidence"] = batch[12].to(device).float()
    # Karar etiketleri (dod_meta / H; DrivingData 13-14) -- model girdisi DEGIL, loss hedefi;
    # imza degismesin diye inputs sozlugunde tasinir (model bilinmeyen anahtari okumaz).
    if len(batch) > 14:
        inputs["decision_lon"] = batch[13].to(device).long()
        inputs["decision_lat"] = batch[14].to(device).long()
    # L1 etiketleri (build_l1_labels.py); model girdisi DEGIL, L_attr hedefi
    if len(batch) > 16:
        inputs["l1_agent"] = batch[15].to(device).long()
        inputs["l1_map"] = batch[16].to(device).long()
    return inputs, ego_future, neighbors_future, c_lat_candidates


def freeze_gameformer(gameformer):
    gameformer.eval()
    for parameter in gameformer.parameters():
        parameter.requires_grad = False


# --- CP-tarzi 5-sinif manevra etiketi (get_decision.py port) ---
# psi'nin STABIL/anlamli hedefi: m* (WTA kazanan mod, egitim boyunca KAYAR, ogrenilemez) YERINE
# GT ego-future'dan turetilen SABIT manevra etiketi. Bedava (GT'den), per-agent label YOK.
_MANEUVER = {'stationary': 0, 'straight': 1, 'turning_left': 2, 'turning_right': 3, 'U-turn_left': 4}
NUM_MANEUVERS = 5

# --- dod_meta (H): factored (lon x lat) meta-aksiyon DOD'u (decision_labels.py) ---
# CE agirliklari ve lon_merge remap'i; main() bayraklara gore gunceller (varsayilan tam 9 sinif).
from GameFormer.decision_labels import (LON_CE_WEIGHT, LAT_CE_WEIGHT, LON_MERGE_MAP,
                                        LON_MERGED_CE_WEIGHT, NUM_LON, NUM_LAT, NUM_LON_MERGED,
                                        LON5_MAP, LAT5_MAP, LON5_CE_WEIGHT, LAT5_CE_WEIGHT,
                                        NUM_LON5, NUM_LAT5,
                                        LON4_MAP, LAT5V_MAP, LON4_CE_WEIGHT, LAT5V_CE_WEIGHT,
                                        NUM_LON4, NUM_LAT5V,
                                        LAT5L_MAP, LAT5L_CE_WEIGHT, NUM_LAT5L)
_LON_W = torch.tensor(LON_CE_WEIGHT, dtype=torch.float32)
_LAT_W = torch.tensor(LAT_CE_WEIGHT, dtype=torch.float32)
_LON_REMAP = None      # lon_merge=1 -> LongTensor[9], etiketleri 6 sinifa katlar
_LAT_REMAP = None      # dec_moe=1 -> LongTensor[7], etiketleri 5 sinifa katlar


def _resample_arc(xy, n):
    """xy [M,2] -> yay-uzunlugu boyunca uniform n nokta (get_decision.resample_line ile ayni, np.interp)."""
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] < 1e-6:
        return np.repeat(xy[:1], n, axis=0)
    cum = cum / cum[-1]
    t = np.linspace(0.0, 1.0, n)
    return np.stack([np.interp(t, cum, xy[:, 0]), np.interp(t, cum, xy[:, 1])], axis=1)


def _maneuver_one(xy, yaw):
    """xy [M,2], yaw [M] (GT ego gelecek, ego-frame) -> 0..4 manevra sinifi (get_decision.py'a sadik).
    Esikler: straight=0.03, turning=0.18 egrilik; heading-farki 0.2; yay-uzunlugu 3m (altinda stationary)."""
    valid = ~np.all(xy == 0, axis=1)
    xy, yaw = xy[valid], yaw[valid]
    if len(xy) < 2:
        return _MANEUVER['stationary']
    length = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
    if length < 3.0:
        return _MANEUVER['stationary']
    pts = _resample_arc(xy, int(length))                     # ~1m aralik -> dusuk-hiz egrilik gurultusunu keser
    tan = np.diff(pts, axis=0)
    tan = tan / np.clip(np.linalg.norm(tan, axis=1, keepdims=True), 1e-8, None)
    ang = np.arccos(np.clip((tan[:-1] * tan[1:]).sum(1), -1.0, 1.0))
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    curv = ang / np.clip(seg[:-1], 1e-8, None)
    sign = np.sign(np.cross(tan[:-1], tan[1:]))
    i = int(np.argmax(curv))
    c = round(float(curv[i]), 2)
    s = float(sign[i])
    diff = round(float(abs(yaw[0] - yaw[-1])), 2)
    turning = (0.03 < c < 0.18 and diff > 0.2) or (0.1 < c < 0.18)
    uturn = c >= 0.18
    if turning or uturn:
        if s == 1.0:
            ctx = 'U-turn_left' if uturn else 'turning_left'
        elif s == -1.0:
            ctx = 'turning_right'          # get_decision: sag U-turn -> turning'e dusurulur -> turning_right
        else:
            ctx = 'straight'
    else:
        ctx = 'straight'
    return _MANEUVER[ctx]


def maneuver_labels(ego_future):
    """ego_future [B,80,3] (x,y,heading) -> LongTensor [B] manevra etiketi (0..4), ayni cihazda."""
    ef = ego_future.detach().cpu().numpy()
    labs = [_maneuver_one(ef[b, :, :2], ef[b, :, 2]) for b in range(ef.shape[0])]
    return torch.tensor(labs, dtype=torch.long, device=ego_future.device)


# L1 ajan sinif agirliklari -- olculen dagilimin TERSI, ust sinir 40 (evals/build_l1_labels.py,
# validation): none %93.21, follows %3.21, yieldingTo %0.06, waitingFor %0.09,
# mergesInFrontOf %0.64, overtakes %2.79.
# yieldingTo/waitingFor cok seyrek (7 ve 10 ornek): ham ters-frekans 417/278 olurdu, egitimi
# dagitir. 40'ta sinirlandi -- sozlukte KALIYORLAR (iz onlari ifade edebilsin diye) ama
# ogrenilmeleri beklenmiyor; olculecek, raporlanacak.
L1_AG_WEIGHT = [0.27, 7.8, 40.0, 40.0, 39.0, 9.0]

# L_mass hedef onceligi: AG_CLASSES = ['none','follows','yieldingTo','waitingFor',
# 'mergesInFrontOf','overtakes'] -> indeksler. Sira build_l1_labels.py'deki AG_PRIORITY ile
# AYNI: acik KARAR (yield/wait) > manevra (merge/overtake) > varsayilan kisit (follows).
# En yuksek oncelikli sinifa sahip ajan(lar) kutlenin gitmesi GEREKEN yerdir.
L1_MASS_PRIORITY = [2, 3, 4, 5, 1]


def causal_loss_and_metrics(out, ego_future, lambda_kld, lambda_ci, lambda_mask, lambda_recon=0.0,
                            neighbors_future=None, lambda_nbr=0.0, lambda_budget=0.0, budget_rho=0.2, budget_lo=0.0,
                            lambda_bc=0.0, lambda_peak=0.0, peak_tau=0.5,
                            lambda_conflict=0.0, conflict_terms='all', dec_labels=None,
                            lambda_attr=0.0, l1_labels=None, lambda_mass=0.0):
    """Causal-Planner'a SADIK loss (ref: ~/Causal-Planner lightning_trainer.py). Backdoor/GRL YOK.

    CP'nin toplami:
        loss = <traj lossleri> + decision_loss + 0.2*scenario_loss
             + 0.5 * decision_causal_inference_loss
             + 0.5 * soft_mask_loss
    Bizdeki karsiliklari (scenario_loss PLUTO'ya ozel, bizde yok):
        L_TRAJ                  <- CP: agent_reg_loss + agent_cls_loss (bizde GMM WTA)
        lambda_kld  * L_KLD     <- CP: decision_loss = CE(decision_causal, m*)          [1.0]
        lambda_ci   * L_CI      <- CP: KL(uniform||p_cfd) + 0.1*(-H)                    [0.5]
        lambda_mask * L_MASK    <- CP: (comp+excl+norm)_other + (comp+excl+norm)_g2a    [0.5]

    Onceki (sadik OLMAYAN) halinden farklar:
      - comp/excl artik MSE (onceden L1 / ham ortalama)
      - normalization_loss EKLENDI (softmax yuzunden ~0 cikar; CP'de de oyle)
      - ajan+harita TOPLANIYOR (onceden 0.5x ortalama) -> adv ~0.87 yerine ~1.6 (CP ile ayni olcek)
      - confound dali icin KL ana terim, entropy 0.1 agirlikli yardimci (onceden entropy tek basinaydi)
    Doner (loss, metrics_dict)."""
    gt = ego_future[:, None]                                       # [B, 1, 80, 3]
    l_traj, _, best_mode = imitation_loss(out['traj'], out['score'], gt)
    dod_meta = out.get('psi_lon_cas') is not None

    def _uniform_ci(logits):
        """KL(uniform || p) + entropi -- CP decision_causal_inference_loss'un cekirdegi."""
        lp = F.log_softmax(logits, dim=-1)
        u = torch.ones_like(logits) / logits.shape[-1]
        kl = F.kl_div(input=lp, target=u, reduction='batchmean', log_target=False)
        e = -(lp.exp() * lp).sum(-1).mean()
        return kl, e

    if dod_meta:
        # --- dod_meta (H): factored (lon x lat) karar, AGIRLIKLI CE (80:1 dengesizlik) ---
        # Etiketler loader'dan (GT-turevi, ham-aday-sirali koridorla; decision_labels.py).
        # Iki head'in terimleri ORTALANIR ki lambda_kld/lambda_ci olcekleri 5-sinif kosularla
        # karsilastirilabilir kalsin.
        lon_star, lat_star = dec_labels
        if _LON_REMAP is not None:                                 # lon_merge: 9->6 / dec_moe: 9->5
            lon_star = _LON_REMAP.to(lon_star.device)[lon_star]
        if _LAT_REMAP is not None:                                 # dec_moe: 7 -> 5 sinif
            lat_star = _LAT_REMAP.to(lat_star.device)[lat_star]
        w_lon = _LON_W.to(out['psi_lon_cas'].device)
        w_lat = _LAT_W.to(out['psi_lat_cas'].device)
        l_kld = 0.5 * (F.cross_entropy(out['psi_lon_cas'], lon_star, weight=w_lon)
                       + F.cross_entropy(out['psi_lat_cas'], lat_star, weight=w_lat))
        kl1, e1 = _uniform_ci(out['psi_lon_cfd'])
        kl2, e2 = _uniform_ci(out['psi_lat_cfd'])
        l_kl, ent = 0.5 * (kl1 + kl2), 0.5 * (e1 + e2)
        l_ci = l_kl + 0.1 * (-ent)
        m_star = None
    else:
        m_star = maneuver_labels(ego_future)                      # [B] SABIT GT-manevra etiketi (0..4)

        # --- decision_loss (CP): causal dal GT manevrasini bilsin (m* = kararsiz WTA mod DEGIL) ---
        l_kld = F.cross_entropy(out['psi_cas'], m_star)

        # --- decision_causal_inference_loss (CP): KL(uniform || p_cfd) + 0.1*(-H) ---
        l_kl, ent = _uniform_ci(out['psi_cfd'])
        l_ci = l_kl + 0.1 * (-ent)

    # --- soft_mask_loss (CP): her iliski icin comp+excl+norm, sonra iliskiler TOPLANIR ---
    def _soft_mask(M_cas, M_cfd, valid):
        """CP: complementarity_loss + exclusivity_loss + normalization_loss (hepsi MSE)."""
        vs = valid                                                  # [B, n] bool
        comp = F.mse_loss((M_cas + M_cfd)[vs], torch.ones_like(M_cas[vs]))          # -> 1
        excl = F.mse_loss((M_cas * M_cfd)[vs], torch.zeros_like(M_cas[vs]))         # -> 0
        rows = vs.any(dim=-1)                                       # [B] en az 1 gecerli
        cs, fs = M_cas.sum(-1)[rows], M_cfd.sum(-1)[rows]
        ones = torch.ones_like(cs)
        norm = F.mse_loss(cs, ones) + F.mse_loss(fs, ones)          # softmax -> ~0 (CP'de de no-op)
        return comp, excl, norm

    comp_ag, excl_ag, norm_ag = _soft_mask(out['M_cas'], out['M_cfd'], out['nbr_valid'])
    comp_mp, excl_mp, norm_mp = _soft_mask(out['M_cas_map'], out['M_cfd_map'], out['map_valid'])
    l_comp, l_excl, l_norm = comp_ag + comp_mp, excl_ag + excl_mp, norm_ag + norm_mp
    # STEP 3: norm loss'tan CIKARILDI. Softmax altinda zaten tam 0 (no-op); sigmoid altinda ise
    # comp ile CELISIR -- comp Sum(g_cfd)=N-Sum(g_cas) derken norm ikisinin de 1 olmasini ister,
    # bu ancak N=2 icin tutarli. Metrik olarak loglanmaya devam ediyor.
    l_mask = l_comp + l_excl                                        # CP: other_soft_mask + g2a_soft_mask

    # STEP 3 (opsiyonel emniyet supabi): sigmoid'in dejenere optimumu g_cas=1 her yerde
    # (comp=0, excl=0 -> "her sey nedensel, hicbir sey confound"). Hinge, ESITLIK degil: az almak
    # bedava, cok almak pahali. Varsayilan 0 -- once recon<->L_traj rekabetinin yetip yetmedigine bakilir.
    nvb_b = out['nbr_valid']
    g_mean = (out['M_cas'] * nvb_b).sum() / nvb_b.sum().clamp(min=1)
    # IKI YONLU: tavan dejenere 'her sey nedensel'i keser, TABAN 'hicbir sey nedensel'i keser.
    # step3'te taban yoktu -> g_cas 0.05'e cokup ajan dali bosaldi (viz: cfd her yerde ~0.97).
    l_budget = F.relu(g_mean - budget_rho) + F.relu(budget_lo - g_mean)

    # --- STEP 2: f_cfd collapse fix. [f_cas(pathway-dropout'lu); f_cfd] gate'siz tam-sahne ozetini
    # (f_all) geri kurmali. Hedef DETACH -> ana toplama/mesaj projeksiyonlari carpitilmaz; gradyan
    # yalniz cfd_recon + f_cfd (+ birakilan orneklerde f_cas) uzerine akar. lambda_recon=0 -> KAPALI.
    l_recon = F.mse_loss(out['recon_pred'], out['f_all'].detach())

    # --- STEP 2b: CP'nin others_reg_loss'unun CONFOUND dala uygulanmis hali. f_cfd TEK BASINA tum
    # komsularin GT gelecegini tasimali. Ajan-BASINA hedef -> uniform maske optimal degil (recon'un
    # zaafi buydu). Agirliksiz: M_cfd ile carpsak model "kolay ajani sec"e kacardi. lambda_nbr=0 -> KAPALI.
    l_nbr = torch.zeros((), device=out['traj'].device)
    if neighbors_future is not None:
        N = out['nbr_pred'].shape[1]
        gt_nbr = neighbors_future[:, :N, :, :2]                              # [B,N,T,2]
        vmask = torch.ne(gt_nbr, 0).any(-1) & out['nbr_valid'][:, :N, None]  # [B,N,T]
        per = F.smooth_l1_loss(out['nbr_pred'], gt_nbr, reduction='none').mean(-1)   # [B,N,T]
        l_nbr = (per * vmask).sum() / vmask.sum().clamp(min=1)

    # --- YOL A: comp/excl'in yerine, softmax parametrizasyonuna UYAN iki terim ---
    # (1) BHATTACHARYYA ORTUSMESI. excl = MSE(M_cas*M_cfd, 0) iki kucuk olasiligi carpip
    # kareliyor -> uniform'da 1/N^4 (N=10 icin 1e-4) cikip "cozuldu" der; oysa uniform
    # AZAMI ortusmedir. BC = sum_j sqrt(M_cas*M_cfd) ayni durumda tam 1.0 verir ve N'den
    # BAGIMSIZDIR (N=4 de 40 da 1.0). Olculdu: E(uniform) BC=1.000 excl=0.0039,
    # C(farkli tepeler) BC=0.512 excl=0.0009 -> BC 0.49 oynuyor, excl 0.003.
    # NOT: excl kor DEGIL, sadece DUZ bolgede kor -- ayni ajanda cift tepeyi iyi yakalar
    # (B: excl=0.131). Bu yuzden BC onun YERINE degil, TAMAMLAYICISI olarak dusunulmeli.
    eps = 1e-12
    def _bc(M_cas, M_cfd, valid):
        vf = valid.float()
        return (((M_cas.clamp(min=eps) * M_cfd.clamp(min=eps)).sqrt() * vf).sum(-1)).mean()
    l_bc = _bc(out['M_cas'], out['M_cfd'], out['nbr_valid']) \
         + _bc(out['M_cas_map'], out['M_cfd_map'], out['map_valid'])

    # (2) ENTROPI HINGE (secicilik). M_cas_ent zaten log(n_valid)'e BOLUNMUS: 1=uniform, 0=tek-tepe.
    # relu(H - tau): tau'nun ALTI bedava -> "2-3 ajan onemli" cezalanmaz, "10 ajan esit" cezalanir.
    # Duz -H olsaydi her sahneyi tek-tepeye zorlardik. Hedef HEAD-ORTALAMASI entropi (M_cas_ent),
    # head-basina (M_cas_headent) DEGIL: teslim ettigimiz maske head ortalamasi, ve ortalamayi
    # cezalamak head'leri hem keskinlesmeye HEM ANLASMAYA zorlar (olculdu: ent 0.908 vs headent 0.513).
    l_peak = F.relu(out['M_cas_ent'] - peak_tau).mean() + F.relu(out['M_cas_map_ent'] - peak_tau).mean()

    # --- L_conflict: M_cas'in BEKLENEN conflict-mesafesini cezalandir -> attention'i ego yoluyla
    # cakisan ajanlara iter. penalty = 3 mesafe feature'inin ortalamasi (uzak = yuksek).
    # M_cas softmax oldugu icin Sum_j M_cas_j * penalty_j = E_{M_cas}[penalty].
    # conflict DETACHED -> gradyan YALNIZ M_cas uzerinden akar, feature'lar sabit hedef.
    # OLCULDU (gat-conflict kosulari, eval_interaction.py): feature'i yalnizca GIRDIYE koymak
    # top1-vs-random'i 1.24 -> 1.26x yapiyor (etkisiz); bu LOSS 1.65x yapiyor. Yani modelin
    # etkilesimi kullanmasi icin ifade edebilmesi YETMIYOR, acikca soylenmesi gerekiyor.
    l_conflict = torch.zeros((), device=out['traj'].device)
    if lambda_conflict > 0.0 and out.get('conflict') is not None:
        # HANGI TERIMLER: olculdu (viraj-guvenli lider tanimiyla, lider = ref path'e <2m ve ileride):
        #   [route, aligned, reach]  lider secme 0.565   arkada 0.392   <- 'all'
        #   [route, reach]                       0.698          0.381
        #   [reach] tek                          0.706          0.360   <- 'reach'
        #   [route] tek                          0.294          0.503
        #   [aligned] tek                        0.385          0.401
        # d_ego_aligned ZARARLI: ikisi de koridorun ustundeyken route/reach ayirt etmez, karar ona
        # kalir, o da takip mesafesini KORUYAN lideri cezalandirir (her t'de 25 m uzakta sayilir),
        # arkadan geleni odullendirir (ego'nun yay noktalarindan gecer). CLS'te de gorunur: yay
        # yurumesi bu terimi keskinlestirdi ve 0.8126 -> 0.794 (taban 10 ve 30'da AYNI).
        # d_route zayif: route_lanes 10 ayri serit, yan seritleri odullendiriyor.
        pen_idx = [2] if conflict_terms == 'reach' else [0, 1, 2]
        penalty = out['conflict'][..., pen_idx].mean(-1)            # [B,N] uzak = yuksek
        l_conflict = (out['M_cas'] * penalty).sum(-1).mean()

    nv_f = out['nbr_valid'].float()
    # --- L_attr: L1 (etkilesim karari) denetimi ---
    # M_cas'i SEMANTIK bir hedefe baglar. Olculdu (2026-08-30): M_cas'in kisitlayan ajani
    # kisitlamayanin ustune koyma AUC'si 0.710, ham geometri 0.975 -- maske dogru ajani
    # buluyor ama siralamasi zayif. Bu kayip ona ilk kez "kim ve NE ILISKISI" der.
    # Girdi kanallari GF-tabanli (gurultulu), hedef GT-tabanli (temiz): CHANNELS_AUDIT'in
    # weak-evidence problemi boylece girdi gurultusu olmaktan cikip denetim sinyaline doner.
    l_attr = torch.zeros((), device=out['traj'].device)
    if lambda_attr > 0 and l1_labels is not None and out.get('l1_ag') is not None:
        y_ag, y_mp = l1_labels
        va, vm = out['gated_valid'], out['gated_map_valid']
        la, lm = out['l1_ag'], out['l1_mp']
        w_ag = torch.tensor(L1_AG_WEIGHT, device=la.device, dtype=la.dtype)
        if va.any():
            l_attr = l_attr + F.cross_entropy(la[va], y_ag[:, :la.shape[1]][va], weight=w_ag)
        if vm.any():
            l_attr = l_attr + F.cross_entropy(lm[vm], y_mp[:, :lm.shape[1]][vm])

    # --- L_mass: M_cas'a DOGRUDAN denetim (L_attr'in ikinci terimi) ---
    # L_attr yalnizca "ajan j NE yapiyor" sorusunu denetler, "HANGI ajan onemli"yi denetlemez.
    # Iz (0.33 * follows(ego, agent1)) sinif-dogru ama ajan-yanlis olabilir -- olculdu:
    # M_cas AUC 0.743, ayni ayrimi ham geometri 0.975 ile yapiyor. Bu terim kutleyi GT'nin
    # nedensel dedigi ajan(lar)a itiyor: hedef q = en yuksek oncelikli non-none sinifa sahip
    # gecerli ajanlar uzerinde duzgun dagilim; kayip -sum_j q_j log p_j, p = gecerli ajanlar
    # uzerinde YENIDEN NORMALIZE edilmis M_cas (M_cas ham halde ~0.92'ye toplaniyor, log'u
    # normalize etmeden almak toplam kutleyi de yukari iterdi -- istemedigimiz bir yan etki).
    l_mass = torch.zeros((), device=out['traj'].device)
    if lambda_mass > 0 and l1_labels is not None and out.get('M_cas') is not None:
        y_ag = l1_labels[0]
        Mc, va = out['M_cas'], out['gated_valid']
        y_ag = y_ag[:, :Mc.shape[1]]
        rank = torch.zeros_like(Mc)
        for r, c in enumerate(L1_MASS_PRIORITY):                 # yuksek puan = daha oncelikli
            rank = torch.where(y_ag == c, float(len(L1_MASS_PRIORITY) - r), rank)
        rank = rank * va.float()                                 # grafikte olmayan ajan hedef olamaz
        best = rank.max(-1, keepdim=True).values
        has = (best.squeeze(-1) > 0)
        if has.any():
            q = ((rank == best) & (rank > 0)).float()
            q = q / q.sum(-1, keepdim=True).clamp(min=1.0)
            p = Mc * va.float()
            p = p / p.sum(-1, keepdim=True).clamp(min=1e-6)
            l_mass = -(q[has] * torch.log(p[has] + 1e-8)).sum(-1).mean()
    loss = (l_traj + lambda_kld * l_kld + lambda_ci * l_ci + lambda_mask * l_mask
            + lambda_attr * l_attr + lambda_mass * l_mass
            + lambda_recon * l_recon + lambda_nbr * l_nbr + lambda_budget * l_budget + lambda_bc * l_bc + lambda_peak * l_peak
            + lambda_conflict * l_conflict)

    with torch.no_grad():
        traj_xy = out['traj'][:, 0, :, :, :2]                     # [B, M, 80, 2]
        gt_xy = gt[:, 0, None, :, :2]                             # [B, 1, 80, 2]
        ade = torch.norm(traj_xy - gt_xy, dim=-1).mean(-1)        # [B, M]
        best = ade.argmin(-1)                                     # [B]
        minade = ade.gather(1, best[:, None]).mean().item()
        fde = torch.norm(traj_xy[:, :, -1] - gt_xy[:, :, -1], dim=-1)  # [B, M]
        minfde = fde.gather(1, best[:, None]).mean().item()

        if dod_meta:
            # iki factored head'in ortalama dogrulugu (yuksek olmali / ~sans olmali)
            cas_acc = 0.5 * ((out['psi_lon_cas'].argmax(-1) == lon_star).float().mean()
                             + (out['psi_lat_cas'].argmax(-1) == lat_star).float().mean()).item()
            cfd_acc = 0.5 * ((out['psi_lon_cfd'].argmax(-1) == lon_star).float().mean()
                             + (out['psi_lat_cfd'].argmax(-1) == lat_star).float().mean()).item()
        else:
            cas_acc = (out['psi_cas'].argmax(-1) == m_star).float().mean().item()   # yuksek olmali
            cfd_acc = (out['psi_cfd'].argmax(-1) == m_star).float().mean().item()   # ~1/K (bilgisiz olmali)
        nvb = out['nbr_valid']
        # M_cas komsular uzerinde dagilim -> mean yaniltici, PEAK'e bak.
        mcas_peak = out['M_cas'].masked_fill(~nvb, 0.0).max(-1).values.mean().item()   # en causal ajan
        mcas_map_peak = out['M_cas_map'].masked_fill(~out['map_valid'], 0.0).max(-1).values.mean().item()  # en causal harita
        q_bar = ((out['M_cas'] * nv_f).sum() / nv_f.sum().clamp(min=1.0)).item()       # marjinal E[M_cas]
        # M_cfd'yi DOGRUDAN gozleyen metrik (adv/excl yapisal olarak olu, ent/cfdacc dolayli).
        mcfd_peak = out['M_cfd'].masked_fill(~nvb, 0.0).max(-1).values.mean().item()   # en confounding ajan
        # COLLAPSE MONITORU: f_cfd'nin BATCH boyunca varyansi. Loss f_cfd'nin bir sey tasimasini
        # ZORUNLU KILMIYOR (tek gorevi "manevra hakkinda bilgisiz ol" -> sabit vektor bunu saglar).
        # Simdilik canli, ama masklari birbirinden iten bir terim eklenince cokme riski dogar.
        # fcfd_var / fcas_var oraninin 0'a gitmesi = f_cfd sabitlesiyor.
        fcfd_var = out['f_cfd'].var(dim=0).mean().item()
        fcas_var = out['f_cas'].var(dim=0).mean().item()
        # UNIFORM REFERANSI: n_valid komsuya esit dagilsa peak tam 1/n_valid olurdu. Peak'i buna gore oku:
        #   peak ~= unif  -> dagilim duz, M sekillenMIYOR (olu)
        #   peak >> unif  -> dagilim tepe yapiyor, M gercekten seciyor
        n_valid = nvb.sum(-1).clamp(min=1).float()                                     # [B]
        unif = (1.0 / n_valid).mean().item()

        # ELESTIRI #3 (causal_graph.py _attend): rapor edilen M_cas = head-ortalamasi, ama toplama
        # her head'in KENDI agirligini kullanir. mcas_ent = entropy(mean-head M_cas)/log(n_valid)
        # (mcas_peak'in dayandigi ayni dagilim, NORMALIZE -- ham nats degil, [0,1] araliginda);
        # mcas_headent = mean(entropy(M_cas_h))/log(n_valid) (her head KENDI icinde ne kadar keskin).
        # mcas_ent >> mcas_headent ise head'ler FARKLI komsulara tepe yapiyor ve mean/mcas_peak bunu
        # gizleyip duz gosteriyor demektir. Ayni tesihs harita (g2a) icin de tutuluyor: peak/uniform
        # =7.33x iddiasinin head-anlasmazligi artefakti olup olmadigini gorebilmek icin.
        # L1 metrikleri: genel dogruluk, non-none GERI CAGIRMA (asil onemli olan -- %94 none
        # oldugu icin duz dogruluk hep yuksek gorunur), harita dogrulugu.
        l1_acc = l1_rec = l1_macc = 0.0
        if out.get('l1_ag') is not None and dec_labels is not None and l1_labels is not None:
            ya, ym = l1_labels
            if ya is not None:
                va2, vm2 = out['gated_valid'], out['gated_map_valid']
                if va2.any():
                    pa = out['l1_ag'].argmax(-1)[va2]; ta = ya[:, :out['l1_ag'].shape[1]][va2]
                    l1_acc = (pa == ta).float().mean().item()
                    nz = ta > 0
                    l1_rec = (pa[nz] == ta[nz]).float().mean().item() if nz.any() else 0.0
                if vm2.any():
                    pm = out['l1_mp'].argmax(-1)[vm2]; tm = ym[:, :out['l1_mp'].shape[1]][vm2]
                    l1_macc = (pm == tm).float().mean().item()
        mcas_ent = out['M_cas_ent'].mean().item()
        mcas_headent = out['M_cas_headent'].mean().item()
        mcfd_ent = out['M_cfd_ent'].mean().item()
        mcfd_headent = out['M_cfd_headent'].mean().item()
        mcas_map_ent = out['M_cas_map_ent'].mean().item()
        mcas_map_headent = out['M_cas_map_headent'].mean().item()
        mcfd_map_ent = out['M_cfd_map_ent'].mean().item()
        mcfd_map_headent = out['M_cfd_map_headent'].mean().item()

        # katman-basina cos(f_cas, h_ego) -- bypass (+h_ego residual) + GRU-fix
        # etkilesimi: h_ego zinciri neredeyse identity olabilir. ~1'e yakinsa gate (M_cas-agirlikli
        # toplama) o katmanda marjinal/olu demektir, M_cas gorsellestirmesi dekoratif olur.
        gate_cos_per_layer = out['gate_cos'].mean(0)          # [L] -- batch uzerinden ortalama, katman ayri
        gate_cos_last = gate_cos_per_layer[-1].item()          # SON katman -- head'e giden f_cas'a en yakin

        # ENTROPI MAKASI (CP lightning_trainer.py:232-236 ile ayni mantik): causal dal PEAKED
        # (dusuk entropi, decision_loss ile denetleniyor), confound dal UNIFORM (yuksek entropi).
        # Ayrisma = aradaki makas. casent'i olcmeden makasi goremiyorduk; asil izlenecek sayi bu.
        if dod_meta:
            lp1 = F.log_softmax(out['psi_lon_cas'], dim=-1)
            lp2 = F.log_softmax(out['psi_lat_cas'], dim=-1)
            casent = 0.5 * (-(lp1.exp() * lp1).sum(-1).mean()
                            - (lp2.exp() * lp2).sum(-1).mean()).item()
        else:
            log_p_cas = F.log_softmax(out['psi_cas'], dim=-1)
            casent = -(log_p_cas.exp() * log_p_cas).sum(-1).mean().item()
        entgap = ent.item() - casent                     # BUYUK olmali (cfd tavanda, cas dipte)

    metrics = {
        'loss': loss.item(), 'traj': l_traj.item(), 'kld': l_kld.item(),
        'kl': l_kl.item(), 'ent': ent.item(), 'casent': casent, 'entgap': entgap,
        'ci': l_ci.item(), 'mask': l_mask.item(),
        'comp': l_comp.item(), 'excl': l_excl.item(), 'norm': l_norm.item(),
        'recon': l_recon.item(), 'nbr': l_nbr.item(), 'budget': l_budget.item(),
        'bc': l_bc.item(), 'peak_hinge': l_peak.item(), 'conflict': l_conflict.item(),
        # joint_softmax: ajan ailesinin softmax kutlesi. Ayri softmax'ta tanim geregi 1.0 (olculemez).
        # Baslangicta ~0.12 (girdi sayisi orani) -- egitimde yukselmezse harita ajanlari ezmis demektir.
        'agmass': (out['agent_mass'].mean().item() if out.get('agent_mass') is not None else 0.0),
        # STEP 3 uyelik metrikleri: sigmoid'de M artik bir dagilim degil, 'ajan iceride mi' skoru.
        # gcas_mean = sahnenin ne kadari nedensel ilan edildi; frac05 = kac ajan g>0.5.
        'gcas_mean': g_mean.item(),
        'gcfd_mean': ((out['M_cfd'] * nvb_b).sum() / nvb_b.sum().clamp(min=1)).item(),
        'gcas_frac05': (((out['M_cas'] > 0.5) & nvb_b).sum() / nvb_b.sum().clamp(min=1)).item(),
        'minADE': minade, 'minFDE': minfde, 'casacc': cas_acc, 'cfdacc': cfd_acc,
        'attr': l_attr.item(), 'mass': l_mass.item(), 'l1acc': l1_acc, 'l1rec': l1_rec, 'l1macc': l1_macc,
        'mcas_peak': mcas_peak, 'mcas_map_peak': mcas_map_peak, 'qbar': q_bar,
        'mcfd_peak': mcfd_peak, 'unif': unif,
        'fcfd_var': fcfd_var, 'fcas_var': fcas_var,
        'mcas_ent': mcas_ent, 'mcas_headent': mcas_headent,
        'mcfd_ent': mcfd_ent, 'mcfd_headent': mcfd_headent,
        'mcas_map_ent': mcas_map_ent, 'mcas_map_headent': mcas_map_headent,
        'mcfd_map_ent': mcfd_map_ent, 'mcfd_map_headent': mcfd_map_headent,
        'gate_cos_last': gate_cos_last,
    }
    for i, v in enumerate(gate_cos_per_layer.tolist()):
        metrics[f'gate_cos_l{i}'] = v      # katman-basina, CSV'de tam detay
    return loss, metrics


def _run_epoch(data_loader, gameformer, causal, device, num_neighbors,
               lambda_kld, lambda_ci, lambda_mask, lambda_recon=0.0, lambda_nbr=0.0,
               lambda_budget=0.0, budget_rho=0.2, budget_lo=0.0,
               lambda_bc=0.0, lambda_peak=0.0, peak_tau=0.5, lambda_conflict=0.0, conflict_terms='all',
               ego_corridor='refpath', optimizer=None, desc="Training", lambda_attr=0.0,
               lambda_mass=0.0):
    train = optimizer is not None
    causal.train() if train else causal.eval()
    gameformer.eval()
    agg = defaultdict(list)

    with tqdm(data_loader, desc=desc, unit="batch") as data_epoch:
        for batch in data_epoch:
            inputs, ego_future, neighbors_future, ref_path = read_batch(batch, device)

            with torch.no_grad():
                encoder_outputs = gameformer.encoder(inputs)
                if ego_corridor == 'gf':
                    # ALTERNATIF koridor: frozen GameFormer'in kendi ego plani (GT DEGIL, tahmin).
                    # Ref path'ten farki: donusleri ve hizi modelin gordugu gibi tasir, ama conflict
                    # feature'i frozen backbone'un kalitesine baglar.
                    top1_fut, nbr_states, _, ego_plan = extract_neighbor_top1_futures(
                        gameformer, encoder_outputs, num_neighbors=num_neighbors, return_ego=True)
                    ref_path = ego_plan[:, None]                             # [B,1,T,2]
                else:
                    top1_fut, nbr_states, _ = extract_neighbor_top1_futures(
                        gameformer, encoder_outputs, num_neighbors=num_neighbors
                    )

            with torch.set_grad_enabled(train):
                out = causal(encoder_outputs, inputs, num_agents=num_neighbors + 1,
                             neighbor_futures=top1_fut, neighbor_states=nbr_states, ref_path=ref_path)
                loss, metrics = causal_loss_and_metrics(out, ego_future, lambda_kld,
                                                         lambda_ci, lambda_mask, lambda_recon,
                                                         neighbors_future, lambda_nbr,
                                                         lambda_budget, budget_rho, budget_lo,
                                                         lambda_bc, lambda_peak, peak_tau,
                                                         lambda_conflict, conflict_terms,
                                                         dec_labels=(inputs.get('decision_lon'),
                                                                     inputs.get('decision_lat')),
                                                         lambda_attr=lambda_attr,
                                                         lambda_mass=lambda_mass,
                                                         l1_labels=((inputs['l1_agent'], inputs['l1_map'])
                                                                    if 'l1_agent' in inputs else None))

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(causal.parameters(), 5.0)
                optimizer.step()

            for key, val in metrics.items():
                agg[key].append(val)
            data_epoch.set_postfix(
                loss=f"{np.mean(agg['loss']):.3f}", minADE=f"{np.mean(agg['minADE']):.3f}",
                casacc=f"{np.mean(agg['casacc']):.3f}", cfdacc=f"{np.mean(agg['cfdacc']):.3f}",
                peak=f"{np.mean(agg['mcas_peak']):.3f}", cfdpk=f"{np.mean(agg['mcfd_peak']):.3f}",
                l1rec=f"{np.mean(agg['l1rec']):.3f}",
                entgap=f"{np.mean(agg['entgap']):.3f}",
                unif=f"{np.mean(agg['unif']):.3f}",
            )

    return {key: float(np.mean(val)) for key, val in agg.items()}


def model_training(args):
    # dod_meta + lon_merge: CE agirliklarini ve etiket remap'ini birlesik siniflara gecir
    # (loss fonksiyonu modul-global _LON_W/_LON_REMAP uzerinden okur).
    global _LON_W, _LON_REMAP, _LAT_W, _LAT_REMAP
    if args.dod_meta and args.lon_merge:
        _LON_W = torch.tensor(LON_MERGED_CE_WEIGHT, dtype=torch.float32)
        _LON_REMAP = torch.tensor(LON_MERGE_MAP, dtype=torch.long)
    if args.dec_moe:
        # dec_moe: HER IKI eksen 5 sinifa katlanir (LON5/LAT5); lon_merge ile birlesmez.
        assert args.dod_meta and not args.lon_merge, "dec_moe: dod_meta=1, lon_merge=0 gerektirir"
        _LON_W = torch.tensor(LON5_CE_WEIGHT, dtype=torch.float32)
        _LON_REMAP = torch.tensor(LON5_MAP, dtype=torch.long)
        _LAT_W = torch.tensor(LAT5_CE_WEIGHT, dtype=torch.float32)
        _LAT_REMAP = torch.tensor(LAT5_MAP, dtype=torch.long)
    if args.dod_tf:
        # dod_tf: ham 9x7 etiket embedding'e gider -> lon_merge (6-sinif embedding) ile uyumsuz.
        # Sozluk/loss AYNEN 9x7 kalir: agirlik/remap degismez.
        assert args.dod_meta and not args.lon_merge and not args.dec_moe, \
            "dod_tf: dod_meta=1, lon_merge=0, dec_moe=0 gerektirir"
    if args.lat_moe:
        # lat_moe: 4x5 GERCEK sozluk + lat-sinifi basina GMM expert.
        # lat_moe=1: LAT5V (to_*, v3_latmoe); lat_moe=2: LAT5L (v4 — LC birinci-sinif).
        assert args.dod_meta and not (args.lon_merge or args.dec_moe or args.dod_tf), \
            "lat_moe: dod_meta=1; lon_merge/dec_moe/dod_tf ile birlesmez"
        _LON_W = torch.tensor(LON4_CE_WEIGHT, dtype=torch.float32)
        _LON_REMAP = torch.tensor(LON4_MAP, dtype=torch.long)
        if int(args.lat_moe) >= 2:
            _LAT_W = torch.tensor(LAT5L_CE_WEIGHT, dtype=torch.float32)
            _LAT_REMAP = torch.tensor(LAT5L_MAP, dtype=torch.long)
        else:
            _LAT_W = torch.tensor(LAT5V_CE_WEIGHT, dtype=torch.float32)
            _LAT_REMAP = torch.tensor(LAT5V_MAP, dtype=torch.long)

    log_path = f"./training_log/{args.name}/"
    os.makedirs(log_path, exist_ok=True)
    initLogging(log_file=log_path + "train.log")

    logging.info("------------- {} -------------".format(args.name))
    logging.info("Batch size: {}".format(args.batch_size))
    logging.info("Learning rate: {}".format(args.learning_rate))
    logging.info("Use device: {}".format(args.device))
    # TAM KONFIGURASYON. Eski kosularda sadece batch/lr/device loglaniyordu; lambda degerleri
    # hicbir yerde kayitli degildi, bu yuzden "bu kosu hangi lambda_mask ile egitildi" sorusunu
    # tahmin ederek cevaplamak zorunda kaldik. Artik checkpoint <-> konfigurasyon bagi kopmuyor.
    logging.info("Config: {}".format({k: v for k, v in sorted(vars(args).items())}))

    set_seed(args.seed)

    gameformer = GameFormer(
        encoder_layers=args.encoder_layers,
        decoder_levels=args.decoder_levels,
        neighbors=args.num_neighbors,
    )
    gameformer.load_state_dict(torch.load(args.pretrained_path, map_location=args.device))
    gameformer = gameformer.to(args.device)
    freeze_gameformer(gameformer)

    causal = CausalPlanner(layers=args.graph_layers, modes=args.modes, dropout=args.dropout,
                           # lambda_recon=0 iken pathway-dropout'u da kapat: torch.rand cagrilmasin,
                           # yoksa RNG akisi kayar ve eski kosularla birebir kiyas bozulur.
                           recon_drop=(args.recon_drop if args.lambda_recon > 0 else 0.0),
                           num_neighbors=args.num_neighbors, gate=args.gate,
                           conflict_feats=args.conflict_feats, conflict_bias=args.conflict_bias,
                           compute_conflict=(args.compute_conflict or args.lambda_conflict > 0),
                           aligned_mode=args.aligned_mode, ego_residual=args.ego_residual,
                           joint_softmax=args.joint_softmax,
                           nbr_enrich=args.nbr_enrich,
                           gate_channels=args.gate_channels, typed_kv=args.typed_kv,
                           channel_evidence=args.channel_evidence,
                           gate_trust=args.gate_trust,
                           l1_drop_input=args.l1_drop_input,
                           dod_meta=args.dod_meta, dec_moe=args.dec_moe, dod_tf=args.dod_tf,
                           lat_moe=args.lat_moe,
                           l1=args.l1, l1_bottleneck=args.l1_bottleneck,
                           num_l1_ag=args.num_l1_ag, num_l1_mp=args.num_l1_mp,
                           num_lon=(NUM_LON4 if args.lat_moe else NUM_LON5 if args.dec_moe
                                    else NUM_LON_MERGED if args.lon_merge else NUM_LON),
                           num_lat=(NUM_LAT5V if args.lat_moe
                                    else NUM_LAT5 if args.dec_moe else NUM_LAT)).to(args.device)

    optimizer = optim.AdamW(causal.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 12, 14, 16, 18], gamma=0.5)

    train_set = DrivingData(args.train_set + "/*.npz", args.num_neighbors,
                            l1_labels=(args.l1_labels or None))
    valid_set = DrivingData(args.valid_set + "/*.npz", args.num_neighbors,
                            l1_labels=(args.l1_valid_labels or args.l1_labels or None))

    # --- dup_boost (v4, kullanici karari 2026-08-26): DETERMINISTIK COGALTMA ---
    # CIL soyunun dengeli-ornekleme pratiginin ilimli hali: LC sahneleri ve dinamik stop/slow
    # sahneleri indeks listesinde dup_boost kopya, gerisi 1 (iadeli-cekilis sampler DEGIL:
    # her sahne her epoch garantili gorulur). Etiket kaynagi: evals/build_label_cache.py
    # ciktisi (ham 9x7). CE agirliklari cogaltilmis EFEKTIF dagilimdan yeniden hesaplanir
    # (sampler + ters-frekans cifte sayimi onlenir).
    if args.dup_boost > 0:
        cache = np.load(args.label_cache)
        lon_raw = torch.from_numpy(cache['lon'].astype(np.int64))
        lat_raw = torch.from_numpy(cache['lat'].astype(np.int64))
        assert len(lon_raw) == len(train_set), \
            f"label cache {len(lon_raw)} != train_set {len(train_set)} — cache'i yeniden uret"
        boost = ((lat_raw == 2) | (lat_raw == 3)                      # lane_change_l/r (ham)
                 | ((lon_raw >= 1) & (lon_raw <= 4)))                 # stop_q/g, slow_q/g (ham)
        idx = torch.arange(len(train_set))
        dup = idx[boost].repeat(args.dup_boost - 1)
        idx = torch.cat([idx, dup])
        logging.info(f"dup_boost x{args.dup_boost}: {int(boost.sum())} sahne cogaltildi "
                     f"({100.0*float(boost.float().mean()):.1f}%), epoch {len(train_set)} -> {len(idx)}")
        train_set = torch.utils.data.Subset(train_set, idx.tolist())
        # efektif CE agirliklari (remap'li sinif uzayinda, w = N/(K*n), cap 10, bos sinif 1)
        def _eff_w(raw, remap, k):
            lab = remap[raw] if remap is not None else raw
            lab = torch.cat([lab, (remap[raw] if remap is not None else raw)[boost].repeat(
                args.dup_boost - 1)])
            cnt = torch.bincount(lab, minlength=k).float()
            w = len(lab) / (k * cnt.clamp(min=1.0))
            w = torch.where(cnt > 0, w.clamp(max=10.0), torch.ones_like(w))
            return w.float()
        _LON_W = _eff_w(lon_raw, _LON_REMAP, len(_LON_W))
        _LAT_W = _eff_w(lat_raw, _LAT_REMAP, len(_LAT_W))
        logging.info(f"efektif CE agirliklari: lon={[round(float(x),2) for x in _LON_W]} "
                     f"lat={[round(float(x),2) for x in _LAT_W]}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=os.cpu_count())
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=os.cpu_count())
    logging.info("Dataset Prepared: {} train data, {} validation data\n".format(len(train_set), len(valid_set)))

    for epoch in range(args.train_epochs):
        logging.info(f"Epoch {epoch + 1}/{args.train_epochs}")
        # scheduled sampling: p(tahminle kosulla) 0 -> ss_max lineer rampa (yalniz TF varyantlari okur)
        causal.ss_p = args.ss_max * epoch / max(1, args.train_epochs - 1)
        train_m = _run_epoch(train_loader, gameformer, causal, args.device, args.num_neighbors,
                             args.lambda_kld, args.lambda_ci, args.lambda_mask, args.lambda_recon, args.lambda_nbr,
                             args.lambda_budget, args.budget_rho, args.budget_lo,
                             args.lambda_bc, args.lambda_peak, args.peak_tau, args.lambda_conflict, args.conflict_terms,
                             args.ego_corridor, optimizer=optimizer, desc="Training",
                             lambda_attr=args.lambda_attr, lambda_mass=args.lambda_mass)
        val_m = _run_epoch(valid_loader, gameformer, causal, args.device, args.num_neighbors,
                           args.lambda_kld, args.lambda_ci, args.lambda_mask, args.lambda_recon, args.lambda_nbr,
                           args.lambda_budget, args.budget_rho, args.budget_lo,
                           args.lambda_bc, args.lambda_peak, args.peak_tau, args.lambda_conflict, args.conflict_terms,
                           args.ego_corridor, optimizer=None, desc="Validation",
                           lambda_attr=args.lambda_attr, lambda_mass=args.lambda_mass)

        log = {"epoch": epoch + 1, "lr": optimizer.param_groups[0]["lr"]}
        log.update({f"train-{k}": v for k, v in train_m.items()})
        log.update({f"val-{k}": v for k, v in val_m.items()})

        log_file = f"./training_log/{args.name}/train_log.csv"
        write_header = epoch == 0
        with open(log_file, "w" if write_header else "a", newline="") as csv_file:
            writer = csv.writer(csv_file)
            if write_header:
                writer.writerow(log.keys())
            writer.writerow(log.values())

        logging.info(
            f"train: minADE={train_m['minADE']:.3f} casacc={train_m['casacc']:.3f} "
            f"cfdacc={train_m['cfdacc']:.3f} l1acc={train_m['l1acc']:.3f} l1rec={train_m['l1rec']:.3f} "
            f"l1macc={train_m['l1macc']:.3f} peak={train_m['mcas_peak']:.3f} cfdpk={train_m['mcfd_peak']:.3f} "
            f"hgap={train_m['mcas_ent'] - train_m['mcas_headent']:.3f} "
            f"hgap_mp={train_m['mcas_map_ent'] - train_m['mcas_map_headent']:.3f} "
            f"gcos={train_m['gate_cos_last']:.3f} "
            f"{('agmass=%.3f ' % train_m['agmass']) if train_m.get('agmass') else ''}| "
            f"val: minADE={val_m['minADE']:.3f} casacc={val_m['casacc']:.3f} "
            f"cfdacc={val_m['cfdacc']:.3f} peak={val_m['mcas_peak']:.3f} cfdpk={val_m['mcfd_peak']:.3f} "
            f"hgap={val_m['mcas_ent'] - val_m['mcas_headent']:.3f} "
            f"hgap_mp={val_m['mcas_map_ent'] - val_m['mcas_map_headent']:.3f} "
            f"gcos={val_m['gate_cos_last']:.3f} "
            f"{('agmass=%.3f ' % val_m['agmass']) if val_m.get('agmass') else ''}"
            f"cfdvar={val_m['fcfd_var'] / max(val_m['fcas_var'], 1e-8):.3f}"   # ->0 = f_cfd cokuyor
        )

        scheduler.step()

        torch.save(
            causal.state_dict(),
            f"training_log/{args.name}/causal_epoch_{epoch + 1}_minADE_{val_m['minADE']:.4f}.pth",
        )
        logging.info(f"CausalPlanner saved in training_log/{args.name}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training the ego-centric causal agent graph")
    parser.add_argument("--name", type=str, help='log name (default: "Exp1")', default="Exp1")
    parser.add_argument("--train_set", type=str, help="path to train data")
    parser.add_argument("--valid_set", type=str, help="path to validation data")
    parser.add_argument("--seed", type=int, help="fix random seed", default=3407)
    parser.add_argument("--encoder_layers", type=int, help="number of encoding layers", default=3)
    parser.add_argument("--decoder_levels", type=int, help="levels of reasoning", default=2)
    parser.add_argument("--num_neighbors", type=int, help="number of neighbor agents", default=10)
    parser.add_argument("--train_epochs", type=int, help="epochs of training", default=20)
    parser.add_argument("--batch_size", type=int, help="batch size (default: 32)", default=32)
    parser.add_argument("--learning_rate", type=float, help="learning rate (default: 1e-4)", default=1e-4)
    parser.add_argument("--weight_decay", type=float, help="AdamW weight decay (default: 0.01)", default=0.01)
    parser.add_argument("--dropout", type=float, help="CausalPlanner dropout (default: 0.1)", default=0.1)
    parser.add_argument("--device", type=str, help="run on which device (default: cuda)", default="cuda")
    parser.add_argument("--pretrained_path", type=str, help="Path to frozen GameFormer model", required=True)
    parser.add_argument("--graph_layers", type=int, help="number of ego-causal disentangler layers", default=1)
    parser.add_argument("--nbr_enrich", type=int, help="neighbor->map enrichment layers before split (0 = KAPALI)", default=0)
    parser.add_argument("--modes", type=int, help="number of trajectory head modes K", default=6)
    # Agirliklar Causal-Planner lightning_trainer.py:263-265 ile ayni:
    #   loss = <traj> + 1.0*decision_loss + 0.5*decision_causal_inference_loss + 0.5*soft_mask_loss
    parser.add_argument("--lambda_kld", type=float, default=1.0,
                        help="CP decision_loss: CE(psi_cas, m*) [CP=1.0]")
    parser.add_argument("--lambda_ci", type=float, default=0.5,
                        help="CP decision_causal_inference_loss: KL(uniform||p_cfd)+0.1*(-H) [CP=0.5]")
    parser.add_argument("--conflict_terms", type=str, default="all", choices=["all", "reach"],
                        help="L_conflict cezasindaki terimler: all = [d_route, d_ego_aligned, d_ego_reach] "
                             "| reach = yalniz d_ego_reach (olculdu: lider secme 0.565 -> 0.706)")
    parser.add_argument("--gate_channels", type=int, default=0,
                        help="channels: yapisal gating -- hicbir kanali yanmayan ajan/eleman softmax'a giremez")
    parser.add_argument("--typed_kv", type=int, default=0,
                        help="channels: (ajan,kanal) girdileri + kanal-basina K/V (gating'i yapisal olarak icerir)")
    parser.add_argument("--channel_evidence", type=int, default=0,
                        help="channels: 9/8-dim kanal kaniti edge'e concat")
    parser.add_argument("--l1_drop_input", type=int, default=0,
                        help="L1'e terfi eden kanallari (follows/merges/overtakes) L0 girdisinden sil. "
                             "l1=1 ile birlikte kullan: aksi halde girdi->hedef kopyadir.")
    parser.add_argument("--gate_trust", type=str, default="all", choices=["all", "reliable"],
                        help="reliable: zayif-IoU kanallar (collision/intersect/merges) GATE kararina sayilmaz")
    parser.add_argument("--dod_meta", type=int, default=0,
                        help="H: DOD karar slotu 5-sinif geometrik manevra YERINE factored (lon x lat) "
                             "meta-aksiyon (nuReasoning taksonomisi, decision_labels.py; lon ilk 4s / "
                             "lat tam 8s, koridor-goreli LC). psi -> psi_lon+psi_lat, agirlikli CE.")
    parser.add_argument("--lon_merge", type=int, default=0,
                        help="dod_meta ile: quickly/gently katla (9 -> 6 lon sinifi). psi_lon confusion "
                             "matrisi ayrimi ogrenemezse acilir; extractor'a dokunmaz.")
    parser.add_argument("--l1", type=int, default=0,
                        help="1 = L1 (etkilesim karari) katmani: ajan/eleman basina sinif "
                             "{none, follows, mergesInFrontOf, overtakes} / {none, keepsLane}. "
                             "Etiket build_l1_labels.py'den (GT kanallari + KG intent kurali).")
    parser.add_argument("--l1_bottleneck", type=int, default=0,
                        help="1 = psi ARTIK f_cas okumaz, yalniz L1 ozetini okur (saf CBM). "
                             "0 = hibrit (f_cas + L1 ozeti), performans sigortasi.")
    parser.add_argument("--num_l1_ag", type=int, default=6)
    parser.add_argument("--num_l1_mp", type=int, default=2)
    parser.add_argument("--lambda_mass", type=float, default=0.0,
                        help="M_cas'a DOGRUDAN denetim. L_attr sinifi denetler, bu AJANI denetler: "
                             "kutle GT'nin nedensel dedigi ajana itilir. --l1_labels ile birlikte kullan.")
    parser.add_argument("--lambda_attr", type=float, default=0.0,
                        help="L_attr agirligi (L1 denetimi). 0 = kapali.")
    parser.add_argument("--l1_valid_labels", type=str, default="",
                        help="validation icin ayri L1 etiket dosyasi")
    parser.add_argument("--l1_labels", type=str, default="",
                        help="build_l1_labels.py ciktisi (npz). Bos = L1 etiketi yok.")
    parser.add_argument("--lat_moe", type=int, default=0,
                        help="4x5 GERCEK sozluk + CA sonrasi LAT SINIFI basina GMM expert (5). "
                             "1 = LAT5V (to_*, v3_latmoe sozlugu) | 2 = LAT5L (v4: LC birinci-"
                             "sinif, inlane->none). Lon dallanmaz (embedding+TF); --ss_max ile "
                             "scheduled sampling. dod_meta gerektirir.")
    parser.add_argument("--dup_boost", type=int, default=0,
                        help="deterministik cogaltma katsayisi (0=kapali): LC + dinamik stop/slow "
                             "sahneleri indekste bu kadar kopya, gerisi 1. --label_cache ister; "
                             "CE agirliklari efektif dagilimdan yeniden hesaplanir.")
    parser.add_argument("--label_cache", type=str, default="",
                        help="evals/build_label_cache.py ciktisi (npz: lon[N], lat[N] ham 9x7).")
    parser.add_argument("--ss_max", type=float, default=0.0,
                        help="scheduled sampling tavani: egitimde GT yerine TAHMINLE kosullama "
                             "olasiligi 0'dan bu degere lineer rampa (TF varyantlari icin; 0 = saf TF).")
    parser.add_argument("--dod_tf", type=int, default=0,
                        help="YALNIZ teacher forcing (dec_moe ablasyon basamagi): egitimde karar "
                             "embedding'ine psi'nin tahmini yerine GT (lon,lat) verilir; sozluk "
                             "9x7 AYNEN, dal yok. dod_meta gerektirir; lon_merge/dec_moe ile birlesmez.")
    parser.add_argument("--dec_moe", type=int, default=0,
                        help="karar-kapili expert decoder (hard MoE / CIL-branch, FAKTORLU cift-eksen): "
                             "sozluk 5x5'e katlanir (LON5/LAT5); lat AILESI (left/right/none) q_enh "
                             "dalini, lon AILESI (brake/hold/cruise) predictor dalini secer. Egitimde "
                             "GT karar (teacher forcing/oracle routing), inference'ta b*. Amac: karari "
                             "yapisal baglayici kilmak (b*-swap compliance teshisinin mimari cozumu). "
                             "dod_meta=1 gerektirir, lon_merge ile birlesmez.")
    parser.add_argument("--joint_softmax", type=int, default=0,
                        help="1 = ajan ve harita girdileri ORTAK softmax paydasini paylasir "
                             "(Sum_ajan M_cas serbest kalir) | 0 = ayri softmax (varsayilan). "
                             "typed_kv + gate_channels gerektirir.")
    parser.add_argument("--ego_residual", type=int, default=1,
                        help="1 = f_cas = LN(out_fc + h_ego) (CP Eq 6, varsayilan) | 0 = residual YOK, "
                             "ego bilgisi f_cas'a yalniz self_fea uzerinden girer")
    parser.add_argument("--aligned_mode", type=str, default="straight", choices=["straight", "arc"],
                        help="d_ego_aligned'da ego'nun t anindaki yeri: straight = [hiz*t, 0] duz cizgi "
                             "(A/B/D kosulari) | arc = ref path uzerinde yay yurumesi (E/F; CLS -0.019)")
    parser.add_argument("--ego_corridor", type=str, default="refpath", choices=["refpath", "gf"],
                        help="conflict koridorunun kaynagi: refpath = lattice planner referans yolu "
                             "(npz c_lat_candidates, GF'den bagimsiz) | gf = frozen GameFormer'in ego plani")
    parser.add_argument("--lambda_conflict", type=float, default=0.0,
                        help="L_conflict: E_{M_cas}[conflict-mesafe] cezasi (0 = KAPALI). Feature'lari "
                             "girdiye eklemeden de calisir -- compute_conflict yeter")
    parser.add_argument("--conflict_feats", type=int, default=0,
                        help="conflict feature'larini AJAN edge'ine ekle (girdi olarak). Tek basina etkisiz olcuruldu")
    parser.add_argument("--conflict_bias", type=int, default=0,
                        help="causal logit'ten softplus(w)*conflict-mesafe cikar (explicit prior)")
    parser.add_argument("--compute_conflict", type=int, default=0,
                        help="conflict feature'larini yalnizca LOSS icin hesapla (girdiyi degistirmeden)")
    parser.add_argument("--lambda_bc", type=float, default=0.0,
                        help="YOL A: Bhattacharyya ortusme cezasi (0 = KAPALI). excl'in duz bolgede kor oldugu yeri gorur")
    parser.add_argument("--lambda_peak", type=float, default=0.0,
                        help="YOL A: relu(H(M_cas)/log n - tau) secicilik hinge'i (0 = KAPALI)")
    parser.add_argument("--peak_tau", type=float, default=0.5,
                        help="YOL A: entropi hinge esigi; ALTI bedava (1=uniform, 0=tek-tepe)")
    parser.add_argument("--gate", type=str, default="softmax", choices=["softmax", "sigmoid"],
                        help="STEP 3: komsu maskesi -- softmax (dagilim, toplam=1) | sigmoid (bagimsiz uyelik)")
    parser.add_argument("--lambda_budget", type=float, default=0.0,
                        help="STEP 3: relu(mean(g_cas)-rho) hinge agirligi (0 = KAPALI)")
    parser.add_argument("--budget_rho", type=float, default=0.2,
                        help="STEP 3: butce bandinin UST siniri")
    parser.add_argument("--budget_lo", type=float, default=0.0,
                        help="STEP 3: butce bandinin ALT siniri (0 = taban yok, tek yonlu)")
    parser.add_argument("--lambda_nbr", type=float, default=0.0,
                        help="STEP 2b: f_cfd->komsu-gelecek tahmini agirligi (0 = KAPALI)")
    parser.add_argument("--lambda_recon", type=float, default=0.0,
                        help="STEP 2: [f_cas;f_cfd]->f_all rekonstruksiyon agirligi (0 = KAPALI)")
    parser.add_argument("--recon_drop", type=float, default=0.5,
                        help="STEP 2: recon head'inde f_cas'in sifirlanma olasiligi (pathway dropout)")
    parser.add_argument("--lambda_mask", type=float, default=0.5,
                        help="CP soft_mask_loss: (comp+excl+norm) ajan + harita TOPLAMI [CP=0.5]")
    args = parser.parse_args()

    model_training(args)
