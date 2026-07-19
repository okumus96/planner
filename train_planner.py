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
from GameFormer.train_utils import DrivingData, initLogging, set_seed


def extract_neighbor_top1_futures(gameformer, encoder_outputs, num_neighbors):
    """Run frozen decoder, take argmax-mode trajectory per neighbor.
    Returns (top1_futures [B,N,T,2], current_states [B,N,5], valid [B,N])."""
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
    return inputs, ego_future, neighbors_future, c_lat_candidates


def freeze_gameformer(gameformer):
    gameformer.eval()
    for parameter in gameformer.parameters():
        parameter.requires_grad = False


def traj_reg_loss(traj, scores, gt):
    """CP-tarzi DETERMINISTIK trajectory loss: (x, y, cos, sin) uzerinde WTA smooth-L1.

    traj:   [B, N, M, 80, 4] = (x, y, cos_h, sin_h)   (CausalEgoHead ciktisi)
    scores: [B, N, M]
    gt:     [B, N, 80, 3]    = (x, y, heading)
    Eski GMM-NLL (mu, log_sig) YERINE: heading artik BIRINCIL cikti (cos/sin), 4 kanal birden
    smooth-L1 ile denetlenir -> inference'ta sonlu-fark heading hack'i gerekmez. Mod secimi
    (WTA) hala xy-ADE uzerinden. Doner (loss, best_mode) -- best_mode [B, N]."""
    B, N = traj.shape[0], traj.shape[1]
    dist = torch.norm(traj[..., :2] - gt[:, :, None, :, :2], dim=-1)         # [B, N, M, 80]
    best_mode = torch.argmin(dist.mean(-1), dim=-1)                          # [B, N]
    best_traj = traj[torch.arange(B)[:, None, None], torch.arange(N)[None, :, None],
                     best_mode[:, :, None]].squeeze(2)                       # [B, N, 80, 4]

    gt_h = gt[..., 2]
    gt_4 = torch.cat([gt[..., :2],
                      torch.stack([gt_h.cos(), gt_h.sin()], dim=-1)], dim=-1)  # [B, N, 80, 4]
    reg_loss = F.smooth_l1_loss(best_traj, gt_4)

    score_loss = F.cross_entropy(scores.permute(0, 2, 1), best_mode,
                                 label_smoothing=0.2, reduction='none')        # [B, N]
    score_loss = (score_loss * torch.ne(gt[:, :, 0, 0], 0)).mean()

    return reg_loss + score_loss, best_mode


# --- CP-tarzi 5-sinif manevra etiketi (get_decision.py port, SHAPELY YOK; length = segment normlari) ---
# GT gelecek yorungesinin GEOMETRISINDEN uretilir (yay uzunlugu + max egrilik + isaret + heading farki).
# psi'nin STABIL/anlamli hedefi -> m* (WTA kazanan mod, ogrenilemez) YERINE. Bedava (GT'den), etiket YOK.
_MANEUVER = {'stationary': 0, 'straight': 1, 'turning_left': 2, 'turning_right': 3, 'U-turn_left': 4}
NUM_MANEUVERS = 5


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
    # identifiy() ile ayni siniflandirma
    turning = (0.03 < c < 0.18 and diff > 0.2) or (0.1 < c < 0.18)
    uturn = c >= 0.18
    if turning or uturn:
        if s == 1.0:
            ctx = 'U-turn_left' if uturn else 'turning_left'
        elif s == -1.0:
            ctx = 'turning_right'          # get_decision: sag U-turn -> turning'e dusurulur -> turning_right
        else:
            ctx = 'straight'               # isaretsiz (nadir) -> yon yok
    else:
        ctx = 'straight'
    return _MANEUVER[ctx]


def maneuver_labels(ego_future):
    """ego_future [B,80,3] (x,y,heading) -> LongTensor [B] manevra etiketi (0..4), ayni cihazda."""
    ef = ego_future.detach().cpu().numpy()
    labs = [_maneuver_one(ef[b, :, :2], ef[b, :, 2]) for b in range(ef.shape[0])]
    return torch.tensor(labs, dtype=torch.long, device=ego_future.device)


# --- StatePerturbation POC (kapalı-döngü drift toparlama augmentation'ı) ---
# Ego-frame sahneyi (gecmis + komsular + harita + GT gelecek) rastgele SE(2) ile kaydirir, ama ego'nun
# ANLIK adimini planlama origin'inde (0,0,heading 0) SABIT tutar -> ego "yoldan sapmis" gorunur, hedef
# (kaydirilmis GT gelecek) "geri don" der. Frozen encoder + egitilen head toparlamayi ogrenir.
# SADECE girdi/target'a dokunur: disentangler, psi, manevra etiketi (SE(2)-degismez) etkilenmez.
def _rot_trans(t, cos, sin, tx, ty, dtheta, has_vel):
    """t [B, *mid, C] -> per-ornek SE(2): kanal(0,1)=xy rot+trans, (2)=heading +dtheta,
    (3,4)=vx,vy rot (has_vel). Sadece GECERLI satir (abs-sum>eps); padding (0) DEGISMEZ."""
    B, C = t.shape[0], t.shape[-1]
    view = (B,) + (1,) * (t.dim() - 2)
    co, si = cos.view(view), sin.view(view)
    TX, TY, DT = tx.view(view), ty.view(view), dtheta.view(view)
    out = t.clone()
    valid = t.abs().sum(-1) > 1e-6                                   # [B, *mid]
    x, y = t[..., 0], t[..., 1]
    out[..., 0] = torch.where(valid, co * x - si * y + TX, x)
    out[..., 1] = torch.where(valid, si * x + co * y + TY, y)
    if C > 2:
        h = t[..., 2]
        hr = torch.atan2(torch.sin(h + DT), torch.cos(h + DT))
        out[..., 2] = torch.where(valid, hr, h)
    if has_vel and C > 4:
        vx, vy = t[..., 3], t[..., 4]
        out[..., 3] = torch.where(valid, co * vx - si * vy, vx)
        out[..., 4] = torch.where(valid, si * vx + co * vy, vy)
    return out


def perturb_batch(inputs, ego_future, prob=0.5, max_lat=0.75, max_lon=1.0, max_yaw=0.35,
                  max_dvel=1.0, collision_thresh=2.5):
    """FULL StatePerturbation (CP augment()'ina sadik). Per-ornek olasilik `prob` ile ego-frame sahneyi
    (+ GT gelecek) ayni SE(2) ile saptirir, ego anlik adimini origin'de sabitler. POC'a eklenenler:
      - COLLISION-SAFETY: transform sonrasi ego-origin bir komsu ile cakisirsa (< collision_thresh m) o
        ornekte perturbation IPTAL edilir -> carpisma-icine-toparlama ogretilmez.
      - HIZ PERTURBATION: ego anlik forward hizi (vx) ±max_dvel saptirilir (clamp>=0) -> hiz-toparlama.
    Doner (inputs, ego_future)."""
    ep = inputs["ego_agent_past"]
    B, dev = ep.shape[0], ep.device
    do = (torch.rand(B, device=dev) < prob).float()
    dtheta = (torch.rand(B, device=dev) * 2 - 1) * max_yaw * do
    tx = (torch.rand(B, device=dev) * 2 - 1) * max_lon * do          # boylamsal (x)
    ty = (torch.rand(B, device=dev) * 2 - 1) * max_lat * do          # yanal (y)
    dvel = (torch.rand(B, device=dev) * 2 - 1) * max_dvel * do       # forward hiz sapmasi

    # --- collision-safety: niyet edilen delta ile komsu ANLIK merkezlerini transform et; ego-origin'e
    #     collision_thresh'ten yakin bir gecerli komsu varsa o ornekte perturbation'i sifirla (identity).
    c0, s0 = torch.cos(dtheta).view(B, 1), torch.sin(dtheta).view(B, 1)
    nb = inputs["neighbor_agents_past"][:, :, -1, :]                 # [B, N, C] son adim
    nvalid = nb.abs().sum(-1) > 1e-6
    nx, ny = nb[..., 0], nb[..., 1]
    nxr = c0 * nx - s0 * ny + tx.view(B, 1)
    nyr = s0 * nx + c0 * ny + ty.view(B, 1)
    ndist = torch.sqrt(nxr * nxr + nyr * nyr + 1e-9)
    unsafe = ((ndist < collision_thresh) & nvalid).any(-1)          # [B]
    safe = (~unsafe).float()
    dtheta, tx, ty, dvel = dtheta * safe, tx * safe, ty * safe, dvel * safe
    cos, sin = torch.cos(dtheta), torch.sin(dtheta)

    # --- SE(2): sahne (gecmis + komsu + harita) + GT gelecek AYNI transform ---
    inputs["ego_agent_past"] = _rot_trans(ep, cos, sin, tx, ty, dtheta, has_vel=True)
    inputs["neighbor_agents_past"] = _rot_trans(inputs["neighbor_agents_past"], cos, sin, tx, ty, dtheta, has_vel=True)
    inputs["map_lanes"] = _rot_trans(inputs["map_lanes"], cos, sin, tx, ty, dtheta, has_vel=False)
    inputs["map_crosswalks"] = _rot_trans(inputs["map_crosswalks"], cos, sin, tx, ty, dtheta, has_vel=False)
    inputs["route_lanes"] = _rot_trans(inputs["route_lanes"], cos, sin, tx, ty, dtheta, has_vel=False)
    ego_future = _rot_trans(ego_future, cos, sin, tx, ty, dtheta, has_vel=False)

    # ego ANLIK adim -> planlama origin'i (pos=0, heading=0); forward hiz (vx) dvel ile saptirilir (>=0).
    inputs["ego_agent_past"][:, -1, 0:3] = 0.0
    inputs["ego_agent_past"][:, -1, 3] = (inputs["ego_agent_past"][:, -1, 3] + dvel).clamp(min=0.0)
    return inputs, ego_future


def causal_loss_and_metrics(out, ego_future, lambda_kld, lambda_ci, lambda_mask):
    """Causal-Planner'a SADIK loss (ref: ~/Causal-Planner lightning_trainer.py). Backdoor/GRL YOK.

    CP'nin toplami:
        loss = <traj lossleri> + decision_loss + 0.2*scenario_loss
             + 0.5 * decision_causal_inference_loss
             + 0.5 * soft_mask_loss
    Bizdeki karsiliklari (scenario_loss PLUTO'ya ozel, bizde yok):
        L_TRAJ                  <- CP: agent_reg_loss + agent_cls_loss (bizde 4-kanal WTA smooth-L1: x,y,cos,sin)
        lambda_kld  * L_KLD     <- CP: decision_loss = CE(decision_causal, 5-sinif manevra)  [1.0]
        lambda_ci   * L_CI      <- CP: KL(uniform||p_cfd) + 0.1*(-H)                    [0.5]
        lambda_mask * L_MASK    <- CP: (comp+excl+norm)_other + (comp+excl+norm)_g2a    [0.5]

    Onceki (sadik OLMAYAN) halinden farklar:
      - comp/excl artik MSE (onceden L1 / ham ortalama)
      - normalization_loss EKLENDI (softmax yuzunden ~0 cikar; CP'de de oyle)
      - ajan+harita TOPLANIYOR (onceden 0.5x ortalama) -> adv ~0.87 yerine ~1.6 (CP ile ayni olcek)
      - confound dali icin KL ana terim, entropy 0.1 agirlikli yardimci (onceden entropy tek basinaydi)
    Doner (loss, metrics_dict)."""
    gt = ego_future[:, None]                                       # [B, 1, 80, 3] (x, y, heading)
    l_traj, _ = traj_reg_loss(out['traj'], out['score'], gt)
    dlab = maneuver_labels(ego_future)                            # [B] 5-sinif manevra (m* DEGIL -> stabil hedef)
    K = out['psi_cas'].shape[-1]                                   # = NUM_MANEUVERS (5)

    # --- decision_loss (CP): causal dal ego'nun MANEVRASINI bilsin (get_decision.py 5-sinif) ---
    l_kld = F.cross_entropy(out['psi_cas'], dlab)

    # --- decision_causal_inference_loss (CP): KL(uniform || p_cfd) + 0.1*(-H) ---
    log_p_cfd = F.log_softmax(out['psi_cfd'], dim=-1)
    target_uniform = torch.ones_like(out['psi_cfd']) / K
    l_kl = F.kl_div(input=log_p_cfd, target=target_uniform, reduction='batchmean', log_target=False)
    ent = -(log_p_cfd.exp() * log_p_cfd).sum(-1).mean()            # H(p_cfd), tavan = ln K
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
    l_mask = l_comp + l_excl + l_norm                               # CP: other_soft_mask + g2a_soft_mask

    nv_f = out['nbr_valid'].float()
    loss = l_traj + lambda_kld * l_kld + lambda_ci * l_ci + lambda_mask * l_mask

    with torch.no_grad():
        traj_xy = out['traj'][:, 0, :, :, :2]                     # [B, M, 80, 2]
        gt_xy = gt[:, 0, None, :, :2]                             # [B, 1, 80, 2]
        ade = torch.norm(traj_xy - gt_xy, dim=-1).mean(-1)        # [B, M]
        best = ade.argmin(-1)                                     # [B]
        minade = ade.gather(1, best[:, None]).mean().item()
        fde = torch.norm(traj_xy[:, :, -1] - gt_xy[:, :, -1], dim=-1)  # [B, M]
        minfde = fde.gather(1, best[:, None]).mean().item()

        cas_acc = (out['psi_cas'].argmax(-1) == dlab).float().mean().item()   # yuksek olmali (manevra bilinir)
        cfd_acc = (out['psi_cfd'].argmax(-1) == dlab).float().mean().item()   # ~1/K (bilgisiz olmali)
        nvb = out['nbr_valid']
        # M_cas komsular uzerinde dagilim -> mean yaniltici, PEAK'e bak.
        mcas_peak = out['M_cas'].masked_fill(~nvb, 0.0).max(-1).values.mean().item()   # en causal ajan
        mcas_map_peak = out['M_cas_map'].masked_fill(~out['map_valid'], 0.0).max(-1).values.mean().item()  # en causal harita
        q_bar = ((out['M_cas'] * nv_f).sum() / nv_f.sum().clamp(min=1.0)).item()       # marjinal E[M_cas]
        # M_cfd'yi DOGRUDAN gozleyen metrik (adv/excl yapisal olarak olu, ent/cfdacc dolayli).
        mcfd_peak = out['M_cfd'].masked_fill(~nvb, 0.0).max(-1).values.mean().item()   # en confounding ajan
        # UNIFORM REFERANSI: n_valid komsuya esit dagilsa peak tam 1/n_valid olurdu. Peak'i buna gore oku:
        #   peak ~= unif  -> dagilim duz, M sekillenMIYOR (olu)
        #   peak >> unif  -> dagilim tepe yapiyor, M gercekten seciyor
        n_valid = nvb.sum(-1).clamp(min=1).float()                                     # [B]
        unif = (1.0 / n_valid).mean().item()

        # ENTROPI MAKASI (CP lightning_trainer.py:232-236 ile ayni mantik): causal dal PEAKED
        # (dusuk entropi, decision_loss ile denetleniyor), confound dal UNIFORM (yuksek entropi).
        # Ayrisma = aradaki makas. casent'i olcmeden makasi goremiyorduk; asil izlenecek sayi bu.
        log_p_cas = F.log_softmax(out['psi_cas'], dim=-1)
        casent = -(log_p_cas.exp() * log_p_cas).sum(-1).mean().item()
        entgap = ent.item() - casent                     # BUYUK olmali (cfd tavanda, cas dipte)

    metrics = {
        'loss': loss.item(), 'traj': l_traj.item(), 'kld': l_kld.item(),
        'kl': l_kl.item(), 'ent': ent.item(), 'casent': casent, 'entgap': entgap,
        'ci': l_ci.item(), 'mask': l_mask.item(),
        'comp': l_comp.item(), 'excl': l_excl.item(), 'norm': l_norm.item(),
        'minADE': minade, 'minFDE': minfde, 'casacc': cas_acc, 'cfdacc': cfd_acc,
        'mcas_peak': mcas_peak, 'mcas_map_peak': mcas_map_peak, 'qbar': q_bar,
        'mcfd_peak': mcfd_peak, 'unif': unif,
    }
    return loss, metrics


def _run_epoch(data_loader, gameformer, causal, device, num_neighbors,
               lambda_kld, lambda_ci, lambda_mask,
               optimizer=None, desc="Training", perturb_prob=0.0,
               perturb_dvel=1.0, perturb_collision_thresh=2.5):
    train = optimizer is not None
    causal.train() if train else causal.eval()
    gameformer.eval()
    agg = defaultdict(list)

    with tqdm(data_loader, desc=desc, unit="batch") as data_epoch:
        for batch in data_epoch:
            inputs, ego_future, _, _ = read_batch(batch, device)
            # StatePerturbation: SADECE egitimde (val temiz olcum). Girdi+GT'yi ayni SE(2) ile saptirir.
            # ABLASYON: perturb_dvel=0 -> hiz perturbation KAPALI; perturb_collision_thresh=0 -> safety KAPALI.
            if train and perturb_prob > 0.0:
                inputs, ego_future = perturb_batch(inputs, ego_future, prob=perturb_prob,
                                                   max_dvel=perturb_dvel,
                                                   collision_thresh=perturb_collision_thresh)

            with torch.no_grad():
                encoder_outputs = gameformer.encoder(inputs)
                top1_fut, nbr_states, _ = extract_neighbor_top1_futures(
                    gameformer, encoder_outputs, num_neighbors=num_neighbors
                )

            with torch.set_grad_enabled(train):
                out = causal(encoder_outputs, inputs, num_agents=num_neighbors + 1,
                             neighbor_futures=top1_fut, neighbor_states=nbr_states)
                loss, metrics = causal_loss_and_metrics(out, ego_future, lambda_kld,
                                                         lambda_ci, lambda_mask)

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
                entgap=f"{np.mean(agg['entgap']):.3f}",
                unif=f"{np.mean(agg['unif']):.3f}",
            )

    return {key: float(np.mean(val)) for key, val in agg.items()}


def model_training(args):
    log_path = f"./training_log/{args.name}/"
    os.makedirs(log_path, exist_ok=True)
    initLogging(log_file=log_path + "train.log")

    logging.info("------------- {} -------------".format(args.name))
    logging.info("Batch size: {}".format(args.batch_size))
    logging.info("Learning rate: {}".format(args.learning_rate))
    logging.info("Use device: {}".format(args.device))

    set_seed(args.seed)

    gameformer = GameFormer(
        encoder_layers=args.encoder_layers,
        decoder_levels=args.decoder_levels,
        neighbors=args.num_neighbors,
    )
    gameformer.load_state_dict(torch.load(args.pretrained_path, map_location=args.device))
    gameformer = gameformer.to(args.device)
    freeze_gameformer(gameformer)

    causal = CausalPlanner(layers=args.graph_layers, modes=args.modes,
).to(args.device)

    optimizer = optim.AdamW(causal.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 12, 14, 16, 18], gamma=0.5)

    train_set = DrivingData(args.train_set + "/*.npz", args.num_neighbors)
    valid_set = DrivingData(args.valid_set + "/*.npz", args.num_neighbors)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=os.cpu_count())
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=os.cpu_count())
    logging.info("Dataset Prepared: {} train data, {} validation data\n".format(len(train_set), len(valid_set)))

    for epoch in range(args.train_epochs):
        logging.info(f"Epoch {epoch + 1}/{args.train_epochs}")
        train_m = _run_epoch(train_loader, gameformer, causal, args.device, args.num_neighbors,
                             args.lambda_kld, args.lambda_ci, args.lambda_mask,
                             optimizer=optimizer, desc="Training", perturb_prob=args.perturb_prob,
                             perturb_dvel=args.perturb_dvel,
                             perturb_collision_thresh=args.perturb_collision_thresh)
        val_m = _run_epoch(valid_loader, gameformer, causal, args.device, args.num_neighbors,
                           args.lambda_kld, args.lambda_ci, args.lambda_mask,
                           optimizer=None, desc="Validation")

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
            f"cfdacc={train_m['cfdacc']:.3f} peak={train_m['mcas_peak']:.3f} cfdpk={train_m['mcfd_peak']:.3f} | "
            f"val: minADE={val_m['minADE']:.3f} casacc={val_m['casacc']:.3f} "
            f"cfdacc={val_m['cfdacc']:.3f} peak={val_m['mcas_peak']:.3f} cfdpk={val_m['mcfd_peak']:.3f}"
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
    parser.add_argument("--device", type=str, help="run on which device (default: cuda)", default="cuda")
    parser.add_argument("--pretrained_path", type=str, help="Path to frozen GameFormer model", required=True)
    parser.add_argument("--graph_layers", type=int, help="number of ego-causal disentangler layers", default=3)
    parser.add_argument("--modes", type=int, help="number of trajectory head modes K", default=6)
    # Agirliklar Causal-Planner lightning_trainer.py:263-265 ile ayni:
    #   loss = <traj> + 1.0*decision_loss + 0.5*decision_causal_inference_loss + 0.5*soft_mask_loss
    parser.add_argument("--lambda_kld", type=float, default=1.0,
                        help="CP decision_loss: CE(psi_cas, 5-sinif manevra etiketi) [CP=1.0]")
    parser.add_argument("--lambda_ci", type=float, default=0.5,
                        help="CP decision_causal_inference_loss: KL(uniform||p_cfd)+0.1*(-H) [CP=0.5]")
    parser.add_argument("--lambda_mask", type=float, default=0.5,
                        help="CP soft_mask_loss: (comp+excl+norm) ajan + harita TOPLAMI [CP=0.5]")
    parser.add_argument("--perturb_prob", type=float, default=0.0,
                        help="StatePerturbation olasiligi (0=kapali). Kapali-dongu drift toparlama "
                             "augmentation'i; SADECE egitimde uygulanir. POC icin 0.5 oner.")
    parser.add_argument("--perturb_dvel", type=float, default=1.0,
                        help="Ego anlik forward hiz perturbation genligi (m/s). 0=KAPALI (ablasyon).")
    parser.add_argument("--perturb_collision_thresh", type=float, default=2.5,
                        help="Collision-safety mesafe esigi (m). 0=safety KAPALI (ablasyon).")
    args = parser.parse_args()

    model_training(args)
