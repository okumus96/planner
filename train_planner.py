import argparse
import csv
import logging
import os

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from GameFormer.ar_wrapper import ModeSelector
from GameFormer.predictor import GameFormer
from GameFormer.train_utils import DrivingData, get_expert_mode_index, initLogging, set_seed


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
    # encoder's mask: agent_mask is the first (1+N_dataset) entries; True = padded
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

def topk_correct(scores, target, k):
    """Returns boolean tensor [B], True if target is in top-k of scores."""
    topk_idx = scores.topk(k, dim=1).indices       # [B, k]
    return (topk_idx == target.unsqueeze(1)).any(dim=1)


def build_gaussian_soft_target(gt_mode_idx, num_lat=5, num_lon=12,
                               sigma_lat=0.7, sigma_lon=1.0):
    """
    Convert hard mode index [B] to 2D-Gaussian soft target [B, num_lat*num_lon].

    The lat/lon grid is treated as two orthogonal axes; we put independent 1D
    Gaussians on each, then take the outer product. Result is normalized per-sample.
    """
    device = gt_mode_idx.device
    B = gt_mode_idx.shape[0]
    gt_lat = (gt_mode_idx // num_lon).float()                              # [B]
    gt_lon = (gt_mode_idx % num_lon).float()                               # [B]

    lat_grid = torch.arange(num_lat, device=device).float()                # [num_lat]
    lat_prob = torch.exp(-((lat_grid[None, :] - gt_lat[:, None]) ** 2)
                         / (2 * sigma_lat ** 2))                           # [B, num_lat]
    lat_prob = lat_prob / lat_prob.sum(-1, keepdim=True)

    lon_grid = torch.arange(num_lon, device=device).float()                # [num_lon]
    lon_prob = torch.exp(-((lon_grid[None, :] - gt_lon[:, None]) ** 2)
                         / (2 * sigma_lon ** 2))                           # [B, num_lon]
    lon_prob = lon_prob / lon_prob.sum(-1, keepdim=True)

    soft = lat_prob.unsqueeze(2) * lon_prob.unsqueeze(1)                   # [B, num_lat, num_lon]
    return soft.reshape(B, num_lat * num_lon)                              # [B, 60]


def soft_ce_loss(scores, soft_target):
    """Cross-entropy with a soft target distribution. scores: [B, C] logits."""
    log_p = nn.functional.log_softmax(scores, dim=-1)
    return -(soft_target * log_p).sum(-1).mean()


def train_epoch(data_loader, gameformer, mode_selector, optimizer, device,
                sigma_lat, sigma_lon):
    losses, acc1, acc3, acc5 = [], [], [], []
    mode_selector.train()

    with tqdm(data_loader, desc="Training", unit="batch") as data_epoch:
        for batch in data_epoch:
            inputs, ego_future, neighbors_future, c_lat_candidates = read_batch(batch, device)
            gt_mode_idx, _, _ = get_expert_mode_index(ego_future, c_lat_candidates)
            soft_target = build_gaussian_soft_target(
                gt_mode_idx, num_lat=5, num_lon=12,
                sigma_lat=sigma_lat, sigma_lon=sigma_lon,
            )

            optimizer.zero_grad()
            with torch.no_grad():
                encoder_outputs = gameformer.encoder(inputs)
                top1_fut, nbr_states, nbr_valid = extract_neighbor_top1_futures(
                    gameformer, encoder_outputs, num_neighbors=10
                )

            mode_scores, _ = mode_selector(
                encoder_outputs['encoding'],
                c_lat_candidates,
                scene_mask=encoder_outputs['mask'],
                neighbor_top1_futures=top1_fut,
                neighbor_current_states=nbr_states,
                neighbor_valid=nbr_valid,
            )
            loss = soft_ce_loss(mode_scores, soft_target)

            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            acc1.append(topk_correct(mode_scores, gt_mode_idx, 1).float().mean().item())
            acc3.append(topk_correct(mode_scores, gt_mode_idx, 3).float().mean().item())
            acc5.append(topk_correct(mode_scores, gt_mode_idx, 5).float().mean().item())
            data_epoch.set_postfix(
                loss=f"{np.mean(losses):.4f}",
                top1=f"{np.mean(acc1):.4f}",
                top3=f"{np.mean(acc3):.4f}",
                top5=f"{np.mean(acc5):.4f}",
            )

    return float(np.mean(losses)), float(np.mean(acc1)), float(np.mean(acc3)), float(np.mean(acc5))


def valid_epoch(data_loader, gameformer, mode_selector, device,
                sigma_lat, sigma_lon):
    losses, acc1, acc3, acc5 = [], [], [], []
    mode_selector.eval()
    gameformer.eval()

    with tqdm(data_loader, desc="Validation", unit="batch") as data_epoch:
        for batch in data_epoch:
            inputs, ego_future, neighbors_future, c_lat_candidates = read_batch(batch, device)
            gt_mode_idx, _, _ = get_expert_mode_index(ego_future, c_lat_candidates)
            soft_target = build_gaussian_soft_target(
                gt_mode_idx, num_lat=5, num_lon=12,
                sigma_lat=sigma_lat, sigma_lon=sigma_lon,
            )

            with torch.no_grad():
                encoder_outputs = gameformer.encoder(inputs)
                top1_fut, nbr_states, nbr_valid = extract_neighbor_top1_futures(
                    gameformer, encoder_outputs, num_neighbors=10
                )
                mode_scores, _ = mode_selector(
                    encoder_outputs['encoding'],
                    c_lat_candidates,
                    scene_mask=encoder_outputs['mask'],
                    neighbor_top1_futures=top1_fut,
                    neighbor_current_states=nbr_states,
                    neighbor_valid=nbr_valid,
                )
                loss = soft_ce_loss(mode_scores, soft_target)

            losses.append(loss.item())
            acc1.append(topk_correct(mode_scores, gt_mode_idx, 1).float().mean().item())
            acc3.append(topk_correct(mode_scores, gt_mode_idx, 3).float().mean().item())
            acc5.append(topk_correct(mode_scores, gt_mode_idx, 5).float().mean().item())
            data_epoch.set_postfix(
                loss=f"{np.mean(losses):.4f}",
                top1=f"{np.mean(acc1):.4f}",
                top3=f"{np.mean(acc3):.4f}",
                top5=f"{np.mean(acc5):.4f}",
            )

    return float(np.mean(losses)), float(np.mean(acc1)), float(np.mean(acc3)), float(np.mean(acc5))


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

    mode_selector = ModeSelector().to(args.device)

    optimizer = optim.AdamW(mode_selector.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 12, 14, 16, 18], gamma=0.5)

    train_set = DrivingData(args.train_set + "/*.npz", args.num_neighbors)
    valid_set = DrivingData(args.valid_set + "/*.npz", args.num_neighbors)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=os.cpu_count())
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=os.cpu_count())
    logging.info("Dataset Prepared: {} train data, {} validation data\n".format(len(train_set), len(valid_set)))

    for epoch in range(args.train_epochs):
        logging.info(f"Epoch {epoch + 1}/{args.train_epochs}")
        train_loss, train_top1, train_top3, train_top5 = train_epoch(
            train_loader, gameformer, mode_selector, optimizer, args.device,
            args.sigma_lat, args.sigma_lon,
        )
        val_loss, val_top1, val_top3, val_top5 = valid_epoch(
            valid_loader, gameformer, mode_selector, args.device,
            args.sigma_lat, args.sigma_lon,
        )

        log = {
            "epoch": epoch + 1,
            "train-loss": train_loss,
            "train-top1": train_top1,
            "train-top3": train_top3,
            "train-top5": train_top5,
            "lr": optimizer.param_groups[0]["lr"],
            "val-loss": val_loss,
            "val-top1": val_top1,
            "val-top3": val_top3,
            "val-top5": val_top5,
        }

        log_file = f"./training_log/{args.name}/train_log.csv"
        if epoch == 0:
            with open(log_file, "w", newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(log.keys())
                writer.writerow(log.values())
        else:
            with open(log_file, "a", newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(log.values())

        scheduler.step()

        torch.save(
            mode_selector.state_dict(),
            f"training_log/{args.name}/mode_selector_epoch_{epoch + 1}_valACC_{val_top1:.4f}.pth",
        )
        logging.info(f"Mode selector saved in training_log/{args.name}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training mode selector only")
    parser.add_argument("--name", type=str, help='log name (default: "Exp1")', default="Exp1")
    parser.add_argument("--train_set", type=str, help="path to train data")
    parser.add_argument("--valid_set", type=str, help="path to validation data")
    parser.add_argument("--seed", type=int, help="fix random seed", default=3407)
    parser.add_argument("--encoder_layers", type=int, help="number of encoding layers", default=3)
    parser.add_argument("--decoder_levels", type=int, help="levels of reasoning", default=2)
    parser.add_argument("--num_neighbors", type=int, help="number of neighbor agents to predict", default=10)
    parser.add_argument("--train_epochs", type=int, help="epochs of training", default=20)
    parser.add_argument("--batch_size", type=int, help="batch size (default: 32)", default=32)
    parser.add_argument("--learning_rate", type=float, help="learning rate (default: 1e-4)", default=1e-4)
    parser.add_argument("--device", type=str, help="run on which device (default: cuda)", default="cuda")
    parser.add_argument("--pretrained_path", type=str, help="Path to frozen GameFormer model", required=True)
    parser.add_argument("--sigma_lat", type=float, help="Gaussian soft-target sigma over lat axis", default=0.7)
    parser.add_argument("--sigma_lon", type=float, help="Gaussian soft-target sigma over lon axis", default=1.0)
    args = parser.parse_args()

    model_training(args)