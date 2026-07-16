"""PURE causal planner (refiner/post-processing YOK).

Frozen GameFormer encoder + CausalPlanner trajectory head -> plani DOGRUDAN dondurur.
Amac: causal head'in CIPLAK closed-loop kalitesini gormek (LatticePlanner refiner drift/collision
sorunlarini ortmez). transform_predictions_to_states + observation_adapter mevcut pipeline'dan
tekrar kullanilir; tek fark plan uretiminin GameFormer decoder yerine CausalPlanner head'inden gelmesi.
"""
import numpy as np
import torch

from .planner_utils import *                         # T, DT, transform_predictions_to_states, ...
from .observation import observation_adapter
from GameFormer.predictor import GameFormer
from GameFormer.causal_graph import CausalPlanner
from train_planner import extract_neighbor_top1_futures

from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.planner.abstract_planner import (
    AbstractPlanner, PlannerInitialization, PlannerInput,
)
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory


class CausalPurePlanner(AbstractPlanner):
    def __init__(self, backbone_path, causal_path, num_neighbors=10, graph_layers=3, modes=6,
                 device=None):
        self._future_horizon = T
        self._step_interval = DT
        self._backbone_path = backbone_path
        self._causal_path = causal_path
        self._num_neighbors = num_neighbors
        self._graph_layers = graph_layers
        self._modes = modes
        use_cuda = (device in (None, 'cuda')) and torch.cuda.is_available()
        self._device = torch.device('cuda' if use_cuda else 'cpu')

    def name(self) -> str:
        return "CausalPurePlanner"

    def observation_type(self):
        return DetectionsTracks

    def initialize(self, initialization: PlannerInitialization):
        self._map_api = initialization.map_api
        self._route_roadblock_ids = initialization.route_roadblock_ids
        # frozen GameFormer backbone (encoder + decoder for neighbor futures)
        self._gameformer = GameFormer(encoder_layers=3, decoder_levels=2, neighbors=self._num_neighbors)
        self._gameformer.load_state_dict(torch.load(self._backbone_path, map_location=self._device))
        self._gameformer.to(self._device).eval()
        # causal head (trained module)
        self._causal = CausalPlanner(layers=self._graph_layers, modes=self._modes)
        self._causal.load_state_dict(torch.load(self._causal_path, map_location=self._device))
        self._causal.to(self._device).eval()

    @torch.no_grad()
    def _plan(self, history, traffic_light_data):
        features = observation_adapter(history, traffic_light_data, self._map_api,
                                       self._route_roadblock_ids, self._device)
        enc = self._gameformer.encoder(features)
        top1, nbr_states, _ = extract_neighbor_top1_futures(self._gameformer, enc, self._num_neighbors)
        out = self._causal(enc, features, num_agents=self._num_neighbors + 1,
                           neighbor_futures=top1, neighbor_states=nbr_states)
        traj = out['traj'][0, 0]                        # [M, 80, 4] (mu_x, mu_y, log_sig_x, log_sig_y)
        best = int(out['score'][0, 0].argmax().item())
        xy = traj[best, :, :2].detach().cpu().numpy()   # [80, 2] ego-frame
        # heading, ardisik xy farkindan (head heading uretmiyor); ilk adim = ikinci adimla ayni
        diffs = np.diff(xy, axis=0)                      # [79, 2]
        heading = np.arctan2(diffs[:, 1], diffs[:, 0])
        heading = np.concatenate([heading[:1], heading]) # [80]
        ego_plan = np.concatenate([xy, heading[:, None]], axis=1).astype(np.float32)  # [80, 3]
        states = transform_predictions_to_states(ego_plan, history.ego_states,
                                                 self._future_horizon, self._step_interval)
        return InterpolatedTrajectory(states)

    def compute_planner_trajectory(self, current_input: PlannerInput):
        history = current_input.history
        traffic_light_data = list(current_input.traffic_light_data)
        return self._plan(history, traffic_light_data)
