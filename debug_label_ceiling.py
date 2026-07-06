"""M2 — Label-ambiguity ceiling audit (read-only analysis, no training).

Sorular:
  1) Mevcut mode etiketleri ne kadar belirsiz? (kac mod "esdeger dogru"?)
  2) Bu belirsizlik altinda ulasilabilir top-1 tavani nedir?
  3) Chord (mevcut) vs arc-length (duzeltilmis) lon etiketi ne kadar farkli?
  4) (Opsiyonel) Mevcut checkpoint, esdeger-kume dogrulugunda nerede?

Esdegerlik tanimlari (rapor edilen sabitler):
  - LAT: min-mesafesi en iyi rotadan <= eps_lat (vars. 2 m) icinde olan gecerli rotalar.
  - LON: merkezi, arc-length ortalama hiza <= 1 bin genisligi (15/11 ~ 1.36 m/s)
         yakin olan binler.
  - Tavan: mukemmel bir secici esdeger kumeden rastgele birini secse
         top-1 = E[1 / |kume|].

Kullanim (ceiling-only, model gerekmez):
  python debug_label_ceiling.py --valid_set /path/to/validation

Checkpoint degerlendirmesi ile:
  python debug_label_ceiling.py --valid_set ... \
      --pretrained_path training_log/normal/model_epoch_19_valADE_1.6487.pth \
      --mode_selector_ckpt training_log/gat_v5/mode_selector_epoch_XX_valACC_0.XXXX.pth
"""
import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from GameFormer.train_utils import DrivingData

NUM_LAT = 5
NUM_LON = 12
MAX_SPEED = 15.0
HORIZON_S = 8.0
BIN_WIDTH = MAX_SPEED / (NUM_LON - 1)  # ~1.36 m/s


def lat_equivalence(ego_future, c_lat_candidates, eps_lat):
    """get_expert_mode_index'in lat mantigini birebir kopyalar, ek olarak
    best + eps_lat icindeki rotalari 'esdeger' isaretler.
    Doner: best_lat [B], equiv_lat [B,5] bool, lat_valid [B,5] bool."""
    gt_endpoint = ego_future[:, -1, :2]                                   # [B,2]
    diffs = c_lat_candidates[..., :2] - gt_endpoint[:, None, None]        # [B,5,T,2]
    dists_all = torch.norm(diffs, dim=-1)                                 # [B,5,T]
    point_valid = (torch.abs(c_lat_candidates).sum(dim=-1) > 1e-4)        # [B,5,T]
    dists_all = dists_all.masked_fill(~point_valid, float('inf'))
    min_dist = dists_all.min(dim=-1).values                               # [B,5]
    lat_valid = (torch.abs(c_lat_candidates).sum(dim=(2, 3)) > 1e-4)      # [B,5]
    min_dist = min_dist.masked_fill(~lat_valid, float('inf'))
    best = min_dist.min(dim=-1, keepdim=True).values                      # [B,1]
    equiv = (min_dist <= best + eps_lat) & lat_valid                      # [B,5]
    return min_dist.argmin(dim=-1), equiv, lat_valid


def lon_speeds(ego_future):
    """Chord (mevcut etiket) ve arc-length (duzeltilmis) ortalama hizlar [B]."""
    chord = torch.norm(ego_future[:, -1, :2] - ego_future[:, 0, :2], dim=-1) / HORIZON_S
    pos = torch.cat([torch.zeros_like(ego_future[:, :1, :2]), ego_future[:, :, :2]], dim=1)
    arc = torch.norm(pos[:, 1:] - pos[:, :-1], dim=-1).sum(-1) / HORIZON_S
    return chord, arc


def speed_to_bin(speed, device):
    bins = torch.linspace(0, MAX_SPEED, NUM_LON, device=device)           # [12]
    return torch.abs(bins[None] - speed[:, None]).argmin(dim=-1), bins


def main():
    parser = argparse.ArgumentParser(description="Label-ambiguity ceiling audit (M2)")
    parser.add_argument("--valid_set", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_neighbors", type=int, default=10)
    parser.add_argument("--eps_lat", type=float, default=2.0, help="lat esdegerlik esigi [m]")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    # Opsiyonel checkpoint degerlendirmesi
    parser.add_argument("--pretrained_path", type=str, default=None, help="frozen GameFormer ckpt")
    parser.add_argument("--mode_selector_ckpt", type=str, default=None)
    parser.add_argument("--encoder_layers", type=int, default=3)
    parser.add_argument("--decoder_levels", type=int, default=2)
    parser.add_argument("--graph_layers", type=int, default=2)
    args = parser.parse_args()
    device = args.device

    dataset = DrivingData(args.valid_set + "/*.npz", args.num_neighbors)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=min(8, os.cpu_count()))
    print(f"{len(dataset)} samples from {args.valid_set}")

    # --- Opsiyonel model kurulumu (valid_epoch ile ayni forward) ---
    model_stack = None
    if args.mode_selector_ckpt is not None:
        assert args.pretrained_path is not None, "--mode_selector_ckpt icin --pretrained_path gerekli"
        from GameFormer.predictor import GameFormer
        from GameFormer.ar_wrapper import ModeSelector
        from GameFormer.relevance_graph import SceneRelevanceGraph
        from train_planner import read_batch, extract_neighbor_top1_futures, freeze_gameformer

        gameformer = GameFormer(encoder_layers=args.encoder_layers,
                                decoder_levels=args.decoder_levels,
                                neighbors=args.num_neighbors)
        gameformer.load_state_dict(torch.load(args.pretrained_path, map_location=device))
        gameformer = gameformer.to(device)
        freeze_gameformer(gameformer)

        mode_selector = ModeSelector().to(device)
        mode_selector.load_state_dict(torch.load(args.mode_selector_ckpt, map_location=device))
        mode_selector.eval()

        graph_path = args.mode_selector_ckpt.replace("mode_selector", "relevance_graph")
        relevance_graph = SceneRelevanceGraph(layers=args.graph_layers).to(device)
        relevance_graph.load_state_dict(torch.load(graph_path, map_location=device))
        relevance_graph.eval()
        model_stack = (gameformer, mode_selector, relevance_graph)
        print(f"Loaded checkpoint: {args.mode_selector_ckpt}")

    # --- Akümülatörler ---
    lat_set_hist = np.zeros(NUM_LAT + 1, dtype=np.int64)   # |lat equiv set| dagilimi
    lon_set_hist = np.zeros(NUM_LON + 1, dtype=np.int64)   # |lon equiv set| dagilimi
    inv_joint_sum, n_total = 0.0, 0
    lon_disagree, lon_shift_sum = 0, 0.0
    chord_speeds_all, arc_speeds_all = [], []
    # model metrikleri
    m_top1 = m_equiv = m_lat_equiv = m_lon_pm1 = 0
    m_top1_chord = m_lon_pm1_chord = 0

    for batch in tqdm(loader, desc="Auditing"):
        ego_future = batch[5].to(device).float()
        c_lat = batch[7].to(device).float()
        B = ego_future.shape[0]

        best_lat, equiv_lat, lat_valid = lat_equivalence(ego_future, c_lat, args.eps_lat)
        chord, arc = lon_speeds(ego_future)
        chord_lon, bins = speed_to_bin(chord, device)
        arc_lon, _ = speed_to_bin(arc, device)
        # LON esdeger kumesi: arc hiza <= 1 bin genisligi yakin binler
        equiv_lon = (torch.abs(bins[None] - arc[:, None]) <= BIN_WIDTH)    # [B,12]

        n_lat = equiv_lat.sum(-1)                                          # [B]
        n_lon = equiv_lon.sum(-1)                                          # [B]
        joint = (n_lat * n_lon).clamp_min(1)
        inv_joint_sum += (1.0 / joint.float()).sum().item()
        n_total += B
        lat_set_hist += np.bincount(n_lat.cpu().numpy(), minlength=NUM_LAT + 1)[:NUM_LAT + 1]
        lon_set_hist += np.bincount(n_lon.cpu().numpy(), minlength=NUM_LON + 1)[:NUM_LON + 1]

        lon_disagree += (chord_lon != arc_lon).sum().item()
        lon_shift_sum += (arc - chord).sum().item()
        chord_speeds_all.append(chord.cpu().numpy())
        arc_speeds_all.append(arc.cpu().numpy())

        if model_stack is not None:
            gameformer, mode_selector, relevance_graph = model_stack
            inputs, _, _, _ = read_batch(batch, device)
            with torch.no_grad():
                enc = gameformer.encoder(inputs)
                top1_fut, nbr_states, nbr_valid = extract_neighbor_top1_futures(
                    gameformer, enc, num_neighbors=args.num_neighbors)
                graph_out = relevance_graph(enc, inputs, num_agents=args.num_neighbors + 1,
                                            neighbor_futures=top1_fut, neighbor_states=nbr_states)
                out = mode_selector(
                    enc['encoding'], c_lat, scene_mask=enc['mask'],
                    neighbor_top1_futures=top1_fut, neighbor_current_states=nbr_states,
                    neighbor_valid=nbr_valid,
                    graph_context=graph_out['context'], graph_valid=graph_out['valid'],
                    importance=None)
                mode_scores = out[0] if isinstance(out, tuple) else out
            pred = mode_scores.argmax(dim=1)
            pred_lat, pred_lon = pred // NUM_LON, pred % NUM_LON
            gt_mode = best_lat * NUM_LON + arc_lon  # arc etiketli GT (yeni baseline)
            gt_mode_chord = best_lat * NUM_LON + chord_lon  # eski (chord) etiket
            m_top1 += (pred == gt_mode).sum().item()
            m_top1_chord += (pred == gt_mode_chord).sum().item()
            m_lon_pm1_chord += ((pred_lon - chord_lon).abs() <= 1).sum().item()
            in_lat = equiv_lat.gather(1, pred_lat[:, None]).squeeze(1)
            in_lon = equiv_lon.gather(1, pred_lon[:, None]).squeeze(1)
            m_equiv += (in_lat & in_lon).sum().item()
            m_lat_equiv += in_lat.sum().item()
            m_lon_pm1 += ((pred_lon - arc_lon).abs() <= 1).sum().item()

    chord_speeds = np.concatenate(chord_speeds_all)
    arc_speeds = np.concatenate(arc_speeds_all)

    print("\n========== M2: LABEL-AMBIGUITY CEILING AUDIT ==========")
    print(f"samples: {n_total} | eps_lat = {args.eps_lat} m | lon bin width = {BIN_WIDTH:.3f} m/s")
    print("\n--- LAT equivalence-set size distribution (|set| -> %) ---")
    for k in range(1, NUM_LAT + 1):
        if lat_set_hist[k]:
            print(f"  {k}: {100.0 * lat_set_hist[k] / n_total:6.2f}%")
    print("\n--- LON equivalence-set size distribution (|set| -> %) ---")
    for k in range(1, NUM_LON + 1):
        if lon_set_hist[k]:
            print(f"  {k}: {100.0 * lon_set_hist[k] / n_total:6.2f}%")
    print(f"\nTOP-1 CEILING (E[1/|joint set|]): {100.0 * inv_joint_sum / n_total:.2f}%")
    print("  (mukemmel secici bile esdeger kume icinden rastgele etiketi tutturur)")

    print("\n--- CHORD vs ARC lon label ---")
    print(f"  disagreement rate: {100.0 * lon_disagree / n_total:.2f}% of samples change lon bin")
    print(f"  mean speed shift (arc - chord): {lon_shift_sum / n_total:+.3f} m/s")
    print(f"  chord speed: mean {chord_speeds.mean():.2f}, p50 {np.median(chord_speeds):.2f} m/s")
    print(f"  arc   speed: mean {arc_speeds.mean():.2f}, p50 {np.median(arc_speeds):.2f} m/s")
    hist, edges = np.histogram(arc_speeds, bins=np.linspace(0, MAX_SPEED, NUM_LON + 1))
    print("  arc-speed bin occupancy (class balance):")
    for i, h in enumerate(hist):
        print(f"    [{edges[i]:5.2f},{edges[i+1]:5.2f}): {100.0 * h / n_total:6.2f}%")

    if model_stack is not None:
        print("\n--- CHECKPOINT (arc-labeled GT) ---")
        print(f"  exact top-1:            {100.0 * m_top1 / n_total:.2f}%")
        print(f"  equivalent-set acc:     {100.0 * m_equiv / n_total:.2f}%  <- headline")
        print(f"  lat-equivalent acc:     {100.0 * m_lat_equiv / n_total:.2f}%")
        print(f"  lon +-1 acc:            {100.0 * m_lon_pm1 / n_total:.2f}%")
        print("--- CHECKPOINT (chord-labeled GT, eski etiket; train_log ile kiyas) ---")
        print(f"  exact top-1:            {100.0 * m_top1_chord / n_total:.2f}%")
        print(f"  lon +-1 acc:            {100.0 * m_lon_pm1_chord / n_total:.2f}%")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
