"""Causal + Refiner deployment.

CausalPlanner'in trajectory ciktisini `neural_plan` olarak alir ve MEVCUT refiner'a
(TrajectoryPlanner + LatticePlanner ref_path) verir -> closed-loop-saglam plan, ama LEARNED intent
causal graph'tan gelir. (Refiner robustlugu saglar; augmentation'a gerek yok.)

Ablasyon (RemoveNonCausal deployment'ta): `keep_k` verilirse M_cas'a gore en causal k komsu tutulur,
gerisi (veya `mask_random=True` ile ayni sayida RASTGELE komsu) girdiden SIFIRLANIR -> planin/CLS'nin
ne kadar degistigine bakariz. keep_k=None -> maskeleme yok (tum ajanlar).

plannerv2.Planner'dan miras alir; sadece model yukleme + _plan override edilir (ref_path2 yerine ref_path,
GameFormer neural_plan yerine causal neural_plan).
"""
import numpy as np
import torch

from .plannerv2 import Planner as PlannerV2
from .planner_utils import *                       # T, DT, transform_predictions_to_states, TrajectoryPlanner...
from .observation import observation_adapter
from GameFormer.predictor import GameFormer
from GameFormer.causal_graph import CausalPlanner
from train_planner import extract_neighbor_top1_futures

from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory


class CausalRefinerPlanner(PlannerV2):
    def __init__(self, backbone_path, causal_path, num_neighbors=10, graph_layers=3, modes=6,
                 use_causal=True, remove='none', remove_k=1, plan_source='cas', nbr_enrich=0, device=None):
        super().__init__(model_path=causal_path, device=device, debug=False,
                         debug_dir=None, debug_max_plots=0, oracle_mode=False)
        self._backbone_path = backbone_path
        self._causal_path = causal_path
        self._num_neighbors = num_neighbors
        self._graph_layers = graph_layers
        self._modes = modes
        self._nbr_enrich = nbr_enrich
        self._use_causal = use_causal    # neural_plan: CausalPlanner (True) vs GameFormer (False)
        # RemoveNonCausal-via-CLS: her frame'de M_cas'a gore TEK ajani cikar (girdiden sifirla).
        # 'high'=en causal ajani cikar (CLS dusmeli), 'low'=en az causal (CLS degismemeli), 'random'=kontrol.
        self._remove = remove
        self._remove_k = remove_k        # kac ajan cikarilacak (high/low en causal/en az causal k tane)
        # 'cas' (varsayilan, ANA plan) vs 'cfd': plani f_cfd'den uretir (hicbir ajan SILINMEZ, remove'dan
        # BAGIMSIZ). "Confounding graph gercekten davranis-belirleyici mi?" sorusunu closed-loop'ta test eder.
        assert plan_source in ('cas', 'cfd')
        self._plan_source = plan_source

    def name(self) -> str:
        return (f"CausalRefinerPlanner[{'causal' if self._use_causal else 'baseline-gf'};"
                f"remove={self._remove}x{self._remove_k};plan={self._plan_source}]")

    def _remove_agents(self, features):
        """M_cas'a gore en causal/az causal k komsuyu girdiden sifirla. RemoveNonCausal, closed-loop."""
        out = self._run_causal(features)
        M = out['M_cas'][0]                          # [N]
        valid = out['nbr_valid'][0]                  # [N] bool
        n_valid = int(valid.sum().item())
        if n_valid < 1:
            return features
        k = min(self._remove_k, n_valid)
        if self._remove == 'high':
            js = M.masked_fill(~valid, -1.0).argsort(descending=True)[:k]
        elif self._remove == 'low':
            js = M.masked_fill(~valid, 2.0).argsort()[:k]
        elif self._remove == 'cfd_high':
            # M_CFD'ye gore en 'confounding' k ajani cikar. Probe f_cfd'nin de f_cas kadar karar
            # bilgisi tasidigini gosterdi -> bu ajanlar gercekten onemsiz mi, test et.
            # CLS dususe -> onlar da causal (=> 'confounding' etiketi yanlis).
            js = out['M_cfd'][0].masked_fill(~valid, -1.0).argsort(descending=True)[:k]
        else:  # random
            idx = torch.nonzero(valid).flatten()
            js = idx[torch.randperm(len(idx), device=idx.device)[:k]]
        feats = dict(features)
        npast = features['neighbor_agents_past'].clone()
        npast[0, js] = 0.0                           # secilen komsulari 'yok' yap
        feats['neighbor_agents_past'] = npast
        return feats

    def _initialize_model(self):
        # GameFormer backbone (frozen) — predictions + encoder icin
        self.backbone = GameFormer(encoder_layers=3, decoder_levels=2, neighbors=self._num_neighbors)
        self.backbone.load_state_dict(torch.load(self._backbone_path, map_location=self._device))
        self.backbone.to(self._device).eval()
        # CausalPlanner (egitilmis) — neural_plan + M_cas
        self.causal = CausalPlanner(layers=self._graph_layers, modes=self._modes, nbr_enrich=self._nbr_enrich)
        # strict=False: model sonradan modul kazandi (gate_bias, nbr_head, cfd_recon); eski checkpoint'ler
        # bu anahtarlari icermez. Ucu de PLAN YOLUNUN DISINDA: gate_bias yalniz gate='sigmoid' dalinda
        # okunur (planner softmax kurar), nbr_head/cfd_recon yalniz egitim loss'larini besler. Yine de
        # ne eksikse yazdir -- listede head.* / attn_* / out_fc_* gorunurse sonuc GECERSIZDIR.
        _miss, _unexp = self.causal.load_state_dict(
            torch.load(self._causal_path, map_location=self._device), strict=False)
        if _miss or _unexp:
            print(f'[load] missing={list(_miss)}  unexpected={list(_unexp)}')
        self.causal.to(self._device).eval()
        self.relevance_graph = None

    @torch.no_grad()
    def _run_causal(self, features):
        enc = self.backbone.encoder(features)
        top1, nbr_states, _ = extract_neighbor_top1_futures(self.backbone, enc, self._num_neighbors)
        out = self.causal(enc, features, num_agents=self._num_neighbors + 1,
                          neighbor_futures=top1, neighbor_states=nbr_states,
                          also_cfd_plan=(self._plan_source == 'cfd'))
        return out

    @torch.no_grad()
    def _causal_neural_plan(self, features):
        out = self._run_causal(features)
        traj_key, score_key = ('traj_cfd', 'score_cfd') if self._plan_source == 'cfd' else ('traj', 'score')
        traj = out[traj_key][0, 0]                    # [M,80,4]
        best = int(out[score_key][0, 0].argmax().item())
        xy = traj[best, :, :2].detach().cpu().numpy()   # [80,2] ego-frame
        diffs = np.diff(xy, axis=0)
        heading = np.arctan2(diffs[:, 1], diffs[:, 0])
        heading = np.concatenate([heading[:1], heading])
        plan = np.concatenate([xy, heading[:, None]], axis=1).astype(np.float32)  # [80,3]
        return torch.from_numpy(plan).unsqueeze(0).to(self._device)               # [1,80,3]

    def _plan(self, ego_state, history, traffic_light_data, observation, iteration=None):
        features = observation_adapter(history, traffic_light_data, self._map_api,
                                       self._route_roadblock_ids, self._device)
        if self._remove != 'none':                   # RemoveNonCausal-via-CLS: k ajani cikar
            features = self._remove_agents(features)
        ref_path = self._get_reference_path(ego_state, traffic_light_data, observation)

        with torch.no_grad():
            _, gf_plan, predictions, scores, ego_cur, nbr_cur = self._get_prediction(features)
            # neural_plan: causal (gate'li) VEYA GameFormer (tum ajanlar, baseline)
            plan = self._causal_neural_plan(features) if self._use_causal else gf_plan

        if ref_path is None:
            # ego rotada degil -> refiner yok, ciplak plan
            plan_np = plan[0].detach().cpu().numpy()
            states = transform_predictions_to_states(plan_np, history.ego_states, self._future_horizon, DT)
            return InterpolatedTrajectory(states)

        final_plan = self._trajectory_planner.plan(ego_state, ego_cur, nbr_cur,
                                                   predictions, plan, scores, ref_path, observation)
        states = transform_predictions_to_states(final_plan, history.ego_states, self._future_horizon, DT)
        return InterpolatedTrajectory(states)

    def compute_planner_trajectory(self, current_input):
        history = current_input.history
        tl = list(current_input.traffic_light_data)
        ego_state, observation = history.current_state
        return self._plan(ego_state, history, tl, observation, current_input.iteration.index)
