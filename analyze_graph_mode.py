"""
Scene graph predicate analysis for SceneGraphModeSelector.

Çalıştırma:
    conda activate nuplan_final
    python analyze_graph_mode.py \
        --gameformer_path training_log/normal/model_epoch_19_valADE_1.6487.pth \
        --selector_path  training_log/graph_mode/mode_selector_epoch_11_valACC_0.6323.pth \
        --valid_set      /home/lt-hta-ai4/ssd1/nuplan/processed_data/validation \
        --num_batches    50 \
        --plot
"""

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from GameFormer.ar_wrapper import PREDICATE_NAMES, NUM_PREDICATES, SceneGraphModeSelector, compute_tau_hat
from GameFormer.predictor import GameFormer
from GameFormer.train_utils import DrivingData


def read_batch(batch, device):
    inputs = {
        "ego_agent_past":      batch[0].to(device).float(),
        "neighbor_agents_past": batch[1].to(device).float(),
        "map_lanes":           batch[2].to(device).float(),
        "map_crosswalks":      batch[3].to(device).float(),
        "route_lanes":         batch[4].to(device).float(),
    }
    c_lat_candidates = batch[7].to(device).float()
    return inputs, c_lat_candidates


def run_analysis(args):
    device = torch.device(args.device)

    # --- Model yükle ---
    print("GameFormer yükleniyor...")
    gameformer = GameFormer(encoder_layers=3, decoder_levels=2, neighbors=10)
    gameformer.load_state_dict(torch.load(args.gameformer_path, map_location=device))
    gameformer = gameformer.to(device).eval()
    for p in gameformer.parameters():
        p.requires_grad = False

    print("SceneGraphModeSelector yükleniyor...")
    selector = SceneGraphModeSelector(num_neighbors=10).to(device).eval()
    selector.load_state_dict(torch.load(args.selector_path, map_location=device))

    # --- Veri ---
    valid_set = DrivingData(args.valid_set + "/*.npz", 10)
    loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # --- İstatistik biriktiriciler ---
    total_agents     = 0          # var olan (padding olmayan) komşu sayısı
    no_match_agents  = 0          # hiç predicate uymayan komşu
    pred_counts      = np.zeros(NUM_PREDICATES, dtype=np.int64)  # predicate toplamları

    # Relevance dağılımı: predicate başına ortalama relevance
    pred_relevance_sum   = np.zeros(NUM_PREDICATES, dtype=np.float64)
    pred_relevance_count = np.zeros(NUM_PREDICATES, dtype=np.int64)

    # no_match için ayrı relevance takibi
    no_match_rel_sum   = 0.0
    no_match_rel_count = 0

    processed = 0
    print(f"\nAnaliz başlıyor — max {args.num_batches} batch...\n")

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= args.num_batches:
                break

            inputs, c_lat = read_batch(batch, device)
            B = inputs['ego_agent_past'].shape[0]

            encoder_outputs = gameformer.encoder(inputs)
            decoder_outputs, _, neighbor_content = gameformer.decoder(encoder_outputs)

            current_states = encoder_outputs['actors'][:, :, -1]   # [B, N+1, 5]
            tau_hat = compute_tau_hat(current_states, decoder_outputs)  # [B, N, K]

            _, _, relevance, _ = selector(
                encoder_outputs['encoding'],
                encoder_outputs['mask'],
                decoder_outputs,
                neighbor_content,
                c_lat,
                current_states,
            )
            # relevance: [B, N]

            # Var olan komşuları belirle (padding değil)
            neigh_states = current_states[:, 1:11]                          # [B, N, 5]
            dist = torch.norm(neigh_states[:, :, :2] - current_states[:, 0:1, :2], dim=-1)  # [B,N]
            exists = ((torch.hypot(neigh_states[:, :, 3], neigh_states[:, :, 4]) > 0.05)
                      | (dist < 60.0))                                       # [B,N]

            tau_np  = tau_hat.cpu().numpy()          # [B, N, K]
            rel_np  = relevance.cpu().numpy()        # [B, N]
            ex_np   = exists.cpu().numpy()           # [B, N]

            for b in range(B):
                for n in range(tau_np.shape[1]):
                    if not ex_np[b, n]:
                        continue
                    total_agents += 1
                    active = tau_np[b, n]  # [K]
                    if active.sum() == 0:
                        no_match_agents += 1
                        no_match_rel_sum   += rel_np[b, n]
                        no_match_rel_count += 1
                    else:
                        for k in range(NUM_PREDICATES):
                            if active[k] > 0.5:
                                pred_counts[k]        += 1
                                pred_relevance_sum[k]   += rel_np[b, n]
                                pred_relevance_count[k] += 1

            processed += B
            print(f"\r  batch {i+1}/{args.num_batches} — {processed} sahne işlendi", end="", flush=True)

    print(f"\n\nToplam sahne: {processed}")
    print(f"Toplam komşu (existing): {total_agents}")
    print(f"Hiç predicate almayan  : {no_match_agents}  ({100*no_match_agents/max(total_agents,1):.1f}%)\n")

    # --- Predicate tablosu ---
    print(f"{'Predicate':<22} {'Aktivasyon':>12}  {'%':>6}  {'Ort. Relevance':>15}")
    print("-" * 62)
    for k in range(NUM_PREDICATES):
        pct  = 100 * pred_counts[k] / max(total_agents, 1)
        avg_rel = (pred_relevance_sum[k] / pred_relevance_count[k]
                   if pred_relevance_count[k] > 0 else float('nan'))
        print(f"{PREDICATE_NAMES[k]:<22} {pred_counts[k]:>12,}  {pct:>5.1f}%  {avg_rel:>14.4f}")

    if no_match_count := no_match_rel_count:
        avg_no = no_match_rel_sum / no_match_count
        print(f"{'[no_match]':<22} {no_match_agents:>12,}  "
              f"{100*no_match_agents/max(total_agents,1):>5.1f}%  {avg_no:>14.4f}")
    print("-" * 62)

    # --- Opsiyonel plot ---
    if args.plot:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Sol: predicate frekansları
            labels    = list(PREDICATE_NAMES) + ['no_match']
            counts_all = list(pred_counts) + [no_match_agents]
            colors     = ['#4e79a7'] * NUM_PREDICATES + ['#f28e2b']
            axes[0].bar(labels, counts_all, color=colors)
            axes[0].set_title('Predicate Aktivasyon Sayıları')
            axes[0].set_xlabel('Predicate')
            axes[0].set_ylabel('Ajan × Sahne Sayısı')
            axes[0].tick_params(axis='x', rotation=40)

            # Sağ: ortalama relevance per predicate
            avg_rels = [pred_relevance_sum[k] / pred_relevance_count[k]
                        if pred_relevance_count[k] > 0 else 0
                        for k in range(NUM_PREDICATES)]
            if no_match_rel_count > 0:
                avg_rels.append(no_match_rel_sum / no_match_rel_count)
            else:
                avg_rels.append(0.0)

            axes[1].bar(labels, avg_rels, color=colors)
            axes[1].set_title('Predicate Başına Ortalama Relevance')
            axes[1].set_xlabel('Predicate')
            axes[1].set_ylabel('Ortalama σ(relevance_head)')
            axes[1].tick_params(axis='x', rotation=40)
            axes[1].set_ylim(0, 1)

            plt.tight_layout()
            out_path = 'analyze_graph_mode_result.png'
            plt.savefig(out_path, dpi=150)
            print(f"\nGrafik kaydedildi: {out_path}")
        except ImportError:
            print("\nmatplotlib bulunamadı, grafik atlandı.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gameformer_path", type=str,
                        default="training_log/normal/model_epoch_19_valADE_1.6487.pth")
    parser.add_argument("--selector_path",   type=str,
                        default="training_log/graph_mode/mode_selector_epoch_11_valACC_0.6323.pth")
    parser.add_argument("--valid_set",       type=str,
                        default="/home/lt-hta-ai4/ssd1/nuplan/processed_data/validation")
    parser.add_argument("--num_batches",     type=int, default=50)
    parser.add_argument("--batch_size",      type=int, default=32)
    parser.add_argument("--device",          type=str, default="cuda")
    parser.add_argument("--plot",            action="store_true")
    args = parser.parse_args()

    run_analysis(args)
