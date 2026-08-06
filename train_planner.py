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


def causal_loss_and_metrics(out, ego_future, lambda_kld, lambda_ci, lambda_mask):
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
    m_star = best_mode[:, 0]                                       # [B] kazanan mod (ana head)
    K = out['psi_cas'].shape[-1]

    # --- decision_loss (CP): causal dal dogru modu bilsin ---
    l_kld = F.cross_entropy(out['psi_cas'], m_star)

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
               lambda_kld, lambda_ci, lambda_mask,
               optimizer=None, desc="Training"):
    train = optimizer is not None
    causal.train() if train else causal.eval()
    gameformer.eval()
    agg = defaultdict(list)

    with tqdm(data_loader, desc=desc, unit="batch") as data_epoch:
        for batch in data_epoch:
            inputs, ego_future, _, _ = read_batch(batch, device)

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

    causal = CausalPlanner(layers=args.graph_layers, modes=args.modes, dropout=args.dropout,
                           nbr_enrich=args.nbr_enrich).to(args.device)

    optimizer = optim.AdamW(causal.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
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
                             optimizer=optimizer, desc="Training")
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
            f"cfdacc={train_m['cfdacc']:.3f} peak={train_m['mcas_peak']:.3f} cfdpk={train_m['mcfd_peak']:.3f} "
            f"hgap={train_m['mcas_ent'] - train_m['mcas_headent']:.3f} "
            f"hgap_mp={train_m['mcas_map_ent'] - train_m['mcas_map_headent']:.3f} "
            f"gcos={train_m['gate_cos_last']:.3f} | "
            f"val: minADE={val_m['minADE']:.3f} casacc={val_m['casacc']:.3f} "
            f"cfdacc={val_m['cfdacc']:.3f} peak={val_m['mcas_peak']:.3f} cfdpk={val_m['mcfd_peak']:.3f} "
            f"hgap={val_m['mcas_ent'] - val_m['mcas_headent']:.3f} "
            f"hgap_mp={val_m['mcas_map_ent'] - val_m['mcas_map_headent']:.3f} "
            f"gcos={val_m['gate_cos_last']:.3f} "
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
    parser.add_argument("--lambda_mask", type=float, default=0.5,
                        help="CP soft_mask_loss: (comp+excl+norm) ajan + harita TOPLAMI [CP=0.5]")
    args = parser.parse_args()

    model_training(args)
