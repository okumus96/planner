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

def train_epoch(data_loader, gameformer, mode_selector, optimizer, device):
    losses = []
    accuracies = []
    mode_selector.train()

    with tqdm(data_loader, desc="Training", unit="batch") as data_epoch:
        for batch in data_epoch:
            inputs, ego_future, neighbors_future, c_lat_candidates = read_batch(batch, device)
            gt_mode_idx, _, _ = get_expert_mode_index(ego_future, c_lat_candidates)

            optimizer.zero_grad()
            with torch.no_grad():
                encoder_outputs = gameformer.encoder(inputs)
                _, env_encoding = gameformer.decoder(encoder_outputs)

            mode_scores, _ = mode_selector(
                env_encoding, c_lat_candidates,
                scene_encoding=encoder_outputs['encoding'],
                scene_mask=encoder_outputs['mask'],
            )
            loss = nn.functional.cross_entropy(mode_scores, gt_mode_idx)

            loss.backward()
            #nn.utils.clip_grad_norm_(mode_selector.parameters(), 5.0)
            optimizer.step()

            predictions = mode_scores.argmax(dim=1)
            accuracy = (predictions == gt_mode_idx).float().mean().item()

            losses.append(loss.item())
            accuracies.append(accuracy)
            data_epoch.set_postfix(loss="{:.4f}".format(np.mean(losses)), acc="{:.4f}".format(np.mean(accuracies)))

    return float(np.mean(losses)), float(np.mean(accuracies))


def valid_epoch(data_loader, gameformer, mode_selector, device):
    losses = []
    accuracies = []
    mode_selector.eval()
    gameformer.eval()

    with tqdm(data_loader, desc="Validation", unit="batch") as data_epoch:
        for batch in data_epoch:
            inputs, ego_future, neighbors_future, c_lat_candidates = read_batch(batch, device)
            gt_mode_idx, _, _ = get_expert_mode_index(ego_future, c_lat_candidates)

            with torch.no_grad():
                encoder_outputs = gameformer.encoder(inputs)
                _, env_encoding = gameformer.decoder(encoder_outputs)
                mode_scores, _ = mode_selector(
                    env_encoding, c_lat_candidates,
                    scene_encoding=encoder_outputs['encoding'],
                    scene_mask=encoder_outputs['mask'],
                )
                loss = nn.functional.cross_entropy(mode_scores, gt_mode_idx)

            predictions = mode_scores.argmax(dim=1)
            accuracy = (predictions == gt_mode_idx).float().mean().item()

            losses.append(loss.item())
            accuracies.append(accuracy)
            data_epoch.set_postfix(loss="{:.4f}".format(np.mean(losses)), acc="{:.4f}".format(np.mean(accuracies)))

    return float(np.mean(losses)), float(np.mean(accuracies))


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
        train_loss, train_acc = train_epoch(train_loader, gameformer, mode_selector, optimizer, args.device)
        val_loss, val_acc = valid_epoch(valid_loader, gameformer, mode_selector, args.device)

        log = {
            "epoch": epoch + 1,
            "train-loss": train_loss,
            "train-acc": train_acc,
            "lr": optimizer.param_groups[0]["lr"],
            "val-loss": val_loss,
            "val-acc": val_acc,
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
            f"training_log/{args.name}/mode_selector_epoch_{epoch + 1}_valACC_{val_acc:.4f}.pth",
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
    args = parser.parse_args()

    model_training(args)