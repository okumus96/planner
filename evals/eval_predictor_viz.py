"""
Standalone evaluation script for GameFormer neighbor predictions.

Two input modes:

  1) nuPlan scenario config (--config), e.g. config/test14-random_reduced.yaml
     Builds scenarios directly from the nuPlan database and extracts features
     on-the-fly via the same DataProcessor used in data_process.py.

  2) Preprocessed .npz directory (--valid_set)
     Loads samples through GameFormer.train_utils.DrivingData (legacy path).

For each scenario/sample the script:
  - Runs the frozen GameFormer to obtain decoder_outputs[level_K_interactions]
  - Draws GT vs top-3 predicted neighbor trajectories on a map background
  - Reports minADE@6, minFDE@6, top1_ADE metrics aggregated across samples

Usage (nuPlan config mode):
    python3 eval_predictor_viz.py \\
        --pretrained_path training_log/Exp1/gameformer_best.pth \\
        --config config/test14-random_reduced.yaml \\
        --data_path /path/to/nuplan/dataset \\
        --map_path /path/to/nuplan-maps-v1.0 \\
        --num_samples 20 --out_dir ./eval_viz

Usage (.npz mode):
    python3 eval_predictor_viz.py \\
        --pretrained_path training_log/Exp1/gameformer_best.pth \\
        --valid_set /path/to/valid_npz \\
        --num_samples 20 --out_dir ./eval_viz
"""

import argparse
import os
import warnings
from collections import defaultdict
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import torch

from GameFormer.predictor import GameFormer


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_min_metrics(pred_xy, gt_xy, valid_mask):
    """pred_xy [N,M,T,2], gt_xy [N,T,2], valid_mask [N,T] -> minADE, minFDE per agent."""
    disp = np.linalg.norm(pred_xy - gt_xy[:, None], axis=-1)  # [N, M, T]
    valid = valid_mask[:, None, :]
    ade_per_mode = (disp * valid).sum(axis=-1) / np.maximum(valid.sum(axis=-1), 1)
    fde_per_mode = disp[..., -1]
    return ade_per_mode.min(axis=-1), fde_per_mode.min(axis=-1)


def top1_ade_per_agent(pred_xy, gt_xy, scores, valid_mask):
    top1 = np.argmax(scores, axis=-1)
    n_range = np.arange(len(top1))
    selected = pred_xy[n_range, top1]
    disp = np.linalg.norm(selected - gt_xy, axis=-1)
    return (disp * valid_mask).sum(axis=-1) / np.maximum(valid_mask.sum(axis=-1), 1)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def draw_sample(map_lanes, map_crosswalks, pred_xy, scores, gt_xy, save_path, title_text):
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    for lane in map_lanes:
        if np.abs(lane).sum() < 1e-4:
            continue
        ax.plot(lane[:, 0], lane[:, 1], '-', color='lightgray', linewidth=0.8, zorder=0)

    if map_crosswalks is not None:
        for cw in map_crosswalks:
            if np.abs(cw).sum() < 1e-4:
                continue
            ax.plot(cw[:, 0], cw[:, 1], '--', color='orange', linewidth=0.6, zorder=0)

    # Ego: tek bir nokta — ego frame'de origin'de
    ax.scatter(0, 0, color='blue', s=140, marker='o', zorder=5,
               edgecolors='black', linewidths=1.2, label='Ego (current)')

    N, M, T, _ = pred_xy.shape
    top3_idx = np.argsort(scores, axis=-1)[:, -3:][:, ::-1]
    # User-picked rank colors: red, purple, cyan
    colors_pred = ['crimson', 'mediumpurple', 'darkcyan']
    drew_legend = False

    for n in range(N):
        gt_n = gt_xy[n]                                    # [T, 2]
        valid_t = np.abs(gt_n).sum(axis=-1) > 1e-3         # [T] — true where GT exists
        if int(valid_t.sum()) < 2:
            continue

        # Plot only valid portion (NaN-mask prevents lines jumping to origin
        # when an agent disappears and gets zero-padded).
        gt_plot = gt_n.astype(np.float32).copy()
        gt_plot[~valid_t] = np.nan
        ax.plot(gt_plot[:, 0], gt_plot[:, 1], '-', color='green', linewidth=1.8,
                alpha=0.9, zorder=2,
                label='Neighbor GT' if not drew_legend else None)

        # Current/start position marker = first valid timestep
        first_valid = int(np.argmax(valid_t))
        ax.scatter(gt_n[first_valid, 0], gt_n[first_valid, 1],
                   color='green', s=40, marker='s', zorder=3)

        # Predictions: only plot up to last valid GT timestep for fair comparison
        last_valid = int(np.where(valid_t)[0][-1]) + 1
        for rank, m_idx in enumerate(top3_idx[n]):
            traj = pred_xy[n, m_idx, :last_valid]
            ax.plot(traj[:, 0], traj[:, 1], '--', color=colors_pred[rank], linewidth=1.3,
                    alpha=0.85 if rank == 0 else 0.65, zorder=1,
                    label=f'Neighbor pred rank-{rank+1}' if not drew_legend else None)
        drew_legend = True

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_title(title_text, fontsize=11)
    ax.set_xlabel('x (m, ego frame)')
    ax.set_ylabel('y (m, ego frame)')

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# Sample iterators (two backends)
# ---------------------------------------------------------------------------

def iter_samples_from_npz(args):
    """Yield (features_np_dict, sample_id) tuples from preprocessed .npz files."""
    from torch.utils.data import DataLoader
    from GameFormer.train_utils import DrivingData

    valid_set = DrivingData(args.valid_set + "/*.npz", args.num_neighbors)
    loader = DataLoader(valid_set, batch_size=1, shuffle=args.shuffle, num_workers=0)

    for i, batch in enumerate(loader):
        if i >= args.num_samples:
            break
        (ego_past, neighbors_past, map_lanes, map_crosswalks, route_lanes,
         ego_future, neighbors_future, _c_lat, _) = batch

        sample = {
            'ego_agent_past': ego_past[0].numpy(),
            'neighbor_agents_past': neighbors_past[0].numpy(),
            'map_lanes': map_lanes[0].numpy(),
            'map_crosswalks': map_crosswalks[0].numpy(),
            'route_lanes': route_lanes[0].numpy(),
            'ego_agent_future': ego_future[0].numpy(),
            'neighbor_agents_future': neighbors_future[0].numpy(),
        }
        yield sample, f"npz_{i:03d}", None


# Feature extraction config (mirrors data_process.py DataProcessor defaults)
_PAST_HORIZON = 2.0
_NUM_PAST = 20
_FUTURE_HORIZON = 8.0
_NUM_FUTURE = 80
_NUM_AGENTS = 20
_MAP_FEATS = ['LANE', 'ROUTE_LANES', 'CROSSWALK']
_MAX_ELEMS = {'LANE': 40, 'ROUTE_LANES': 10, 'CROSSWALK': 5}
_MAX_PTS = {'LANE': 50, 'ROUTE_LANES': 50, 'CROSSWALK': 30}
_RADIUS = 60
_INTERP = 'linear'


def _extract_sample_at_iteration(scenario, iteration, num_neighbors):
    """Build the model-input dict + GT from a scenario at any iteration.

    Mirrors data_process.DataProcessor's per-method logic but parametrized
    on `iteration` so the same scenario can be evaluated at multiple timesteps.
    """
    from nuplan.common.actor_state.state_representation import Point2D
    from GameFormer.data_utils import (
        sampled_past_ego_states_to_tensor,
        sampled_past_timestamps_to_tensor,
        sampled_tracked_objects_to_tensor_list,
        agent_past_process,
        agent_future_process,
        get_neighbor_vector_set_map,
        map_process,
        convert_absolute_to_relative_poses,
    )

    map_api = scenario.map_api
    current_ego_state = scenario.get_ego_state_at_iteration(iteration)

    # Past ego
    past_ego_states = list(scenario.get_ego_past_trajectory(
        iteration=iteration, num_samples=_NUM_PAST, time_horizon=_PAST_HORIZON,
    ))
    past_ego_tensor = sampled_past_ego_states_to_tensor(past_ego_states + [current_ego_state])

    past_ts = list(scenario.get_past_timestamps(
        iteration=iteration, num_samples=_NUM_PAST, time_horizon=_PAST_HORIZON,
    )) + [scenario.get_time_point(iteration)]
    past_ts_tensor = sampled_past_timestamps_to_tensor(past_ts)

    # Past tracked objects
    current_tracked = scenario.get_tracked_objects_at_iteration(iteration).tracked_objects
    past_tracked = [t.tracked_objects for t in scenario.get_past_tracked_objects(
        iteration=iteration, time_horizon=_PAST_HORIZON, num_samples=_NUM_PAST,
    )]
    past_obj_tensors, past_obj_types = sampled_tracked_objects_to_tensor_list(
        past_tracked + [current_tracked]
    )

    ego_past, neighbor_past, neighbor_indices = agent_past_process(
        past_ego_tensor, past_ts_tensor, past_obj_tensors, past_obj_types, _NUM_AGENTS,
    )

    # Map
    ego_coords = Point2D(current_ego_state.rear_axle.x, current_ego_state.rear_axle.y)
    route_ids = scenario.get_route_roadblock_ids()
    tl_data = scenario.get_traffic_light_status_at_iteration(iteration)
    coords, tl_data = get_neighbor_vector_set_map(
        map_api, _MAP_FEATS, ego_coords, _RADIUS, route_ids, tl_data,
    )
    vector_map = map_process(
        current_ego_state.rear_axle, coords, tl_data,
        _MAP_FEATS, _MAX_ELEMS, _MAX_PTS, _INTERP,
    )

    # Future
    future_ego_states = list(scenario.get_ego_future_trajectory(
        iteration=iteration, num_samples=_NUM_FUTURE, time_horizon=_FUTURE_HORIZON,
    ))
    ego_future = convert_absolute_to_relative_poses(
        current_ego_state.rear_axle, [s.rear_axle for s in future_ego_states],
    )

    future_tracked = [t.tracked_objects for t in scenario.get_future_tracked_objects(
        iteration=iteration, time_horizon=_FUTURE_HORIZON, num_samples=_NUM_FUTURE,
    )]
    future_obj_tensors, _ = sampled_tracked_objects_to_tensor_list(
        [current_tracked] + future_tracked
    )
    neighbor_future = agent_future_process(
        current_ego_state, future_obj_tensors, _NUM_AGENTS, neighbor_indices,
    )

    def _np(x):
        return x.detach().cpu().numpy() if hasattr(x, 'detach') else np.asarray(x)

    return {
        'ego_agent_past': _np(ego_past),
        'neighbor_agents_past': _np(neighbor_past)[:num_neighbors],
        'map_lanes': _np(vector_map['lanes']),
        'map_crosswalks': _np(vector_map['crosswalks']),
        'route_lanes': _np(vector_map['route_lanes']),
        'ego_agent_future': _np(ego_future),
        'neighbor_agents_future': _np(neighbor_future)[:num_neighbors],
    }


def iter_samples_from_config(args):
    """Yield (features_np_dict, sample_id, scenario_type) tuples from nuPlan scenarios.

    Two sub-modes:
      - Default (--iter_within_scenario=False): one sample per scenario at iteration=0
      - Per-iteration (--iter_within_scenario=True): sweep iterations within each scenario
    """
    from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
    from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioMapping

    from common_utils import get_scenario_map
    from data_process import scenario_filter_helper

    map_version = "nuplan-maps-v1.0"
    scenario_mapping = ScenarioMapping(scenario_map=get_scenario_map(), subsample_ratio_override=0.5)
    builder = NuPlanScenarioBuilder(args.data_path, args.map_path, None, None, map_version,
                                    scenario_mapping=scenario_mapping)
    worker = SingleMachineParallelExecutor(use_process_pool=True)

    filter_params = scenario_filter_helper(args.config)
    scenario_filter = ScenarioFilter(*filter_params)
    scenarios = builder.get_scenarios(scenario_filter, worker)

    if len(scenarios) == 0:
        raise RuntimeError(f"No scenarios matched the filter in {args.config}")

    # Cap scenario count
    if args.max_scenarios is not None:
        scenarios = scenarios[:args.max_scenarios]

    for s_idx, scenario in enumerate(scenarios):
        token = getattr(scenario, 'token', f"i{s_idx}")
        map_name = getattr(scenario, '_map_name', 'map')
        scenario_type = (getattr(scenario, 'scenario_type', None)
                         or getattr(scenario, '_scenario_type', None)
                         or 'unknown')
        sid_base = f"{map_name}_{token[:8]}"

        if not args.iter_within_scenario:
            try:
                sample = _extract_sample_at_iteration(scenario, 0, args.num_neighbors)
            except Exception as e:
                print(f"  [{sid_base}] extract failed: {e}")
                continue
            yield sample, sid_base, scenario_type
            continue

        # Per-iteration sweep: only evaluate iterations that have a full 8s future
        n_iters = scenario.get_number_of_iterations()
        max_safe = max(0, n_iters - _NUM_FUTURE)
        if args.max_iters_per_scenario is not None:
            max_safe = min(max_safe, args.max_iters_per_scenario)

        for it in range(0, max_safe, args.iter_step):
            try:
                sample = _extract_sample_at_iteration(scenario, it, args.num_neighbors)
            except Exception as e:
                print(f"  [{sid_base} it={it}] extract failed: {e}")
                continue
            yield sample, f"{sid_base}_it{it:03d}", scenario_type


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    if args.config is None and args.valid_set is None:
        raise SystemExit("Either --config (nuPlan YAML) or --valid_set (npz dir) must be provided.")
    if args.config is not None and (args.data_path is None or args.map_path is None):
        raise SystemExit("--config requires --data_path and --map_path (nuPlan database + maps).")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print(f"Warning: --device={args.device} requested but CUDA is not available. Falling back to CPU.")
        device = "cpu"
    else:
        device = args.device

    gameformer = GameFormer(
        encoder_layers=args.encoder_layers,
        decoder_levels=args.decoder_levels,
        neighbors=args.num_neighbors,
    )
    state = torch.load(args.pretrained_path, map_location=device)
    gameformer.load_state_dict(state)
    gameformer = gameformer.to(device).eval()

    os.makedirs(args.out_dir, exist_ok=True)

    iterator = (iter_samples_from_config(args) if args.config is not None
                else iter_samples_from_npz(args))

    all_min_ade, all_min_fde, all_top1_ade = [], [], []
    all_min_ade_3s, all_min_ade_5s = [], []   # horizon-stratified (3s, 5s); 8s = all_min_ade
    all_min_fde_3s, all_min_fde_5s = [], []
    per_type = defaultdict(lambda: {'min_ade': [], 'min_fde': [], 'top1_ade': []})
    sample_count = 0

    for sample, sample_id, scenario_type in iterator:
        if sample_count >= args.num_samples:
            break
        # Build model inputs (add batch dim, move to device)
        inputs = {
            'ego_agent_past': torch.from_numpy(sample['ego_agent_past']).float().unsqueeze(0).to(device),
            'neighbor_agents_past': torch.from_numpy(sample['neighbor_agents_past']).float().unsqueeze(0).to(device),
            'map_lanes': torch.from_numpy(sample['map_lanes']).float().unsqueeze(0).to(device),
            'map_crosswalks': torch.from_numpy(sample['map_crosswalks']).float().unsqueeze(0).to(device),
            'route_lanes': torch.from_numpy(sample['route_lanes']).float().unsqueeze(0).to(device),
        }

        with torch.no_grad():
            decoder_outputs, _, _ = gameformer(inputs)

        levels = max(int(k.split('_')[1]) for k in decoder_outputs.keys()
                     if 'interactions' in k)
        trajectories = decoder_outputs[f'level_{levels}_interactions']  # [B, N+1, M, T, 4]
        scores = decoder_outputs[f'level_{levels}_scores']               # [B, N+1, M]

        pred_xy = trajectories[0, 1:, :, :, :2].cpu().numpy()  # [N, M, T, 2]
        score_np = scores[0, 1:].cpu().numpy()                  # [N, M]
        gt_xy = sample['neighbor_agents_future'][..., :2]       # [N, T, 2]

        # Align N if model predicts more neighbors than GT contains (or vice versa)
        n_min = min(pred_xy.shape[0], gt_xy.shape[0])
        pred_xy = pred_xy[:n_min]
        score_np = score_np[:n_min]
        gt_xy = gt_xy[:n_min]

        # Align T if mismatch
        t_min = min(pred_xy.shape[2], gt_xy.shape[1])
        pred_xy = pred_xy[:, :, :t_min]
        gt_xy = gt_xy[:, :t_min]

        valid_mask = (np.abs(gt_xy).sum(axis=-1) > 1e-3).astype(np.float32)  # [N, T]
        valid_agent = valid_mask[:, -1] > 0

        min_ade, min_fde = compute_min_metrics(pred_xy, gt_xy, valid_mask)
        top1_ade = top1_ade_per_agent(pred_xy, gt_xy, score_np, valid_mask)

        # Horizon-stratified: 3s (30 steps) and 5s (50 steps) — assumes 10Hz prediction
        T_full = pred_xy.shape[2]
        h3 = min(30, T_full)
        h5 = min(50, T_full)
        min_ade_3s, min_fde_3s = compute_min_metrics(pred_xy[:, :, :h3], gt_xy[:, :h3], valid_mask[:, :h3])
        min_ade_5s, min_fde_5s = compute_min_metrics(pred_xy[:, :, :h5], gt_xy[:, :h5], valid_mask[:, :h5])

        n_valid = int(valid_agent.sum())
        if n_valid > 0:
            all_min_ade.extend(min_ade[valid_agent].tolist())
            all_min_fde.extend(min_fde[valid_agent].tolist())
            all_top1_ade.extend(top1_ade[valid_agent].tolist())
            all_min_ade_3s.extend(min_ade_3s[valid_agent].tolist())
            all_min_ade_5s.extend(min_ade_5s[valid_agent].tolist())
            all_min_fde_3s.extend(min_fde_3s[valid_agent].tolist())
            all_min_fde_5s.extend(min_fde_5s[valid_agent].tolist())
            if scenario_type is not None:
                per_type[scenario_type]['min_ade'].extend(min_ade[valid_agent].tolist())
                per_type[scenario_type]['min_fde'].extend(min_fde[valid_agent].tolist())
                per_type[scenario_type]['top1_ade'].extend(top1_ade[valid_agent].tolist())
            s_minade = float(np.mean(min_ade[valid_agent]))
            s_minfde = float(np.mean(min_fde[valid_agent]))
            s_top1 = float(np.mean(top1_ade[valid_agent]))
        else:
            s_minade = s_minfde = s_top1 = float('nan')

        title = (f"{sample_id} ({n_valid} tracked nbrs) — "
                 f"minADE@6={s_minade:.2f}m  "
                 f"minFDE@6={s_minfde:.2f}m  "
                 f"top1_ADE={s_top1:.2f}m")

        if sample_count < args.max_viz:
            out_path = os.path.join(args.out_dir, f"sample_{sample_count:03d}_{sample_id}.png")
            draw_sample(sample['map_lanes'], sample['map_crosswalks'],
                        pred_xy, score_np, gt_xy, out_path, title)
        print(f"[{sample_count + 1}] {title}")
        sample_count += 1

    if len(all_min_ade) == 0:
        print("No valid neighbors found in any sample.")
        return

    arr_ade = np.array(all_min_ade)
    arr_fde = np.array(all_min_fde)
    arr_top1 = np.array(all_top1_ade)
    arr_ade_3s = np.array(all_min_ade_3s)
    arr_ade_5s = np.array(all_min_ade_5s)
    arr_fde_3s = np.array(all_min_fde_3s)
    arr_fde_5s = np.array(all_min_fde_5s)

    def stats(x):
        return {
            'mean': float(np.mean(x)),
            'median': float(np.median(x)),
            'p25': float(np.percentile(x, 25)),
            'p75': float(np.percentile(x, 75)),
            'max': float(np.max(x)),
        }

    s_ade = stats(arr_ade)
    s_fde = stats(arr_fde)
    s_top1 = stats(arr_top1)
    selector_gap = s_ade['mean'] and s_top1['mean'] / max(s_ade['mean'], 1e-6)

    # Standard miss-rate metrics
    mr_2m = float(np.mean(arr_fde > 2.0))
    mr_4m = float(np.mean(arr_fde > 4.0))
    score_head_failures = float(np.mean(arr_top1 > 2 * arr_ade))  # top1 > 2x oracle = score head wrong

    lines = []
    lines.append(f"Samples processed: {sample_count}")
    lines.append(f"Total valid neighbor predictions: {len(all_min_ade)}\n")

    lines.append("=== Overall (full 8s horizon) ===")
    lines.append(f"{'metric':<14}{'mean':>8}{'median':>9}{'p25':>8}{'p75':>8}{'max':>9}")
    for name, s in [('minADE@6', s_ade), ('minFDE@6', s_fde), ('top1_ADE', s_top1)]:
        lines.append(f"{name:<14}{s['mean']:>8.3f}{s['median']:>9.3f}{s['p25']:>8.3f}{s['p75']:>8.3f}{s['max']:>9.2f}")
    lines.append("")

    lines.append("=== Horizon-stratified mean (3s / 5s / 8s) ===")
    lines.append(f"  minADE  :   {float(np.mean(arr_ade_3s)):.3f}  /  {float(np.mean(arr_ade_5s)):.3f}  /  {s_ade['mean']:.3f}  m")
    lines.append(f"  minFDE  :   {float(np.mean(arr_fde_3s)):.3f}  /  {float(np.mean(arr_fde_5s)):.3f}  /  {s_fde['mean']:.3f}  m")
    lines.append("")

    lines.append("=== Diagnostic ratios ===")
    lines.append(f"  top1_ADE / minADE@6      = {selector_gap:.2f}   (1.0 = perfect score head; >2 = bad)")
    lines.append(f"  miss-rate (FDE > 2 m)    = {mr_2m * 100:.1f}%")
    lines.append(f"  miss-rate (FDE > 4 m)    = {mr_4m * 100:.1f}%")
    lines.append(f"  score-head wrong (>2x)   = {score_head_failures * 100:.1f}%   (agents where top1 > 2× oracle)")
    lines.append("")

    if per_type:
        lines.append("=== Per scenario type (mean across agents) ===")
        lines.append(f"  {'scenario_type':<38}{'N':>5}{'minADE':>9}{'minFDE':>9}{'top1':>8}{'gap':>7}")
        for stype in sorted(per_type.keys()):
            d = per_type[stype]
            if not d['min_ade']:
                continue
            ma = float(np.mean(d['min_ade']))
            mf = float(np.mean(d['min_fde']))
            t1 = float(np.mean(d['top1_ade']))
            gap = t1 / max(ma, 1e-6)
            lines.append(f"  {stype:<38}{len(d['min_ade']):>5}{ma:>9.3f}{mf:>9.3f}{t1:>8.3f}{gap:>7.2f}")
        lines.append("")

    summary = "\n".join(lines) + "\n"
    with open(os.path.join(args.out_dir, "metrics_summary.txt"), "w") as f:
        f.write(summary)

    print("\n=== AGGREGATE ===")
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize and evaluate GameFormer neighbor predictions")
    parser.add_argument("--pretrained_path", type=str, required=True,
                        help="Path to frozen GameFormer checkpoint")

    # Source of samples (mutually exclusive in practice)
    parser.add_argument("--config", type=str, default=None,
                        help="nuPlan ScenarioFilter YAML (e.g. config/test14-random_reduced.yaml)")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to nuPlan database root (required with --config)")
    parser.add_argument("--map_path", type=str, default=None,
                        help="Path to nuplan-maps-v1.0 dir (required with --config)")
    parser.add_argument("--valid_set", type=str, default=None,
                        help="Path to preprocessed .npz dir (legacy path, alt to --config)")

    parser.add_argument("--num_samples", type=int, default=20,
                        help="Global cap on samples to evaluate (across all scenarios/iterations).")
    parser.add_argument("--num_neighbors", type=int, default=10)
    parser.add_argument("--encoder_layers", type=int, default=3)
    parser.add_argument("--decoder_levels", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out_dir", type=str, default="./eval_viz")
    parser.add_argument("--shuffle", action="store_true",
                        help="Shuffle .npz sample order (only used in --valid_set mode)")

    # Per-iteration sweep (config mode only)
    parser.add_argument("--iter_within_scenario", action="store_true",
                        help="Sweep all valid iterations within each scenario, not just iteration=0.")
    parser.add_argument("--max_scenarios", type=int, default=None,
                        help="Cap number of scenarios (use with --iter_within_scenario; default = no cap).")
    parser.add_argument("--iter_step", type=int, default=5,
                        help="Iteration stride when sweeping. 1=every 0.1s, 5=every 0.5s (default), 10=every 1s.")
    parser.add_argument("--max_iters_per_scenario", type=int, default=None,
                        help="Cap iterations per scenario (default = all with full 8s future).")

    parser.add_argument("--max_viz", type=int, default=20,
                        help="Max number of PNG plots to save (metrics still aggregated for all samples).")
    args = parser.parse_args()
    main(args)
