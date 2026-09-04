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
                 use_causal=True, remove='none', remove_k=1, plan_source='cas', nbr_enrich=0, ego_residual=1, joint_softmax=0,
                 gate_channels=0, typed_kv=0, channel_evidence=0, gate_trust='all',
                 dod_meta=0, lon_merge=0, uniform_mask=0, dec_moe=0, lat_moe=0, cc_select=0,
                 l1=0, l1_bottleneck=0, l1_drop_input=0, psi_prior_alpha=0.0,
                 device=None):
        super().__init__(model_path=causal_path, device=device, debug=False,
                         debug_dir=None, debug_max_plots=0, oracle_mode=False)
        self._backbone_path = backbone_path
        self._causal_path = causal_path
        self._num_neighbors = num_neighbors
        self._graph_layers = graph_layers
        self._modes = modes
        self._nbr_enrich = nbr_enrich
        self._ego_residual = ego_residual
        self._joint_softmax = joint_softmax
        # Predicate kanallari (channels branch). Checkpoint hangi bayraklarla EGITILDIYSE ayni
        # bayraklar verilmeli; ozellikle typed_kv=1 ckpt'lerde untyped attention dali (Wk_ag/attn_cas)
        # egitimde HIC gradyan almadi -> kanallar deployment'ta uretilemezse model o egitilmemis
        # dala duser ve sonuc GECERSIZDIR (asagida _run_causal icindeki uyari bunu yakalar).
        self._gate_channels = gate_channels
        self._typed_kv = typed_kv
        self._channel_evidence = channel_evidence
        self._gate_trust = gate_trust
        # dod_meta (H): ckpt hangi degerle egitildiyse o (psi_lon/psi_lat + factored embedding'ler)
        self._dod_meta = dod_meta
        self._lon_merge = lon_merge
        self._dec_moe = dec_moe          # dec_moe ckpt: 5x5 sozluk + aile-dallari; routing b*'dan
        self._lat_moe = lat_moe          # lat_moe ckpt: 4x5 sozluk + lat-sinifi GMM expert'leri
        self._l1, self._l1_bottleneck = l1, l1_bottleneck
        self._l1_drop_input = l1_drop_input
        # cc_select (egitimsiz, 2026-08-26): KARAR-TUTARLI mod secimi — 6 mod icinden ilan
        # edilen b*'a uyan en yuksek skorlu secilir (oncelik: lon&lat -> lat -> lon -> argmax).
        # Skorcu karar-kor oldugu icin argmax baglam-tercihli modu secebiliyordu; bu kural
        # eval'deki karar-tutarli compliance'i deployment davranisina tasir. Retrain yok.
        self._cc_select = cc_select
        # psi prior duzeltmesi (egitimsiz, 2026-09-02): psi logit'lerinden alpha*log(CE agirligi)
        # cikarilir. Agirlikli CE (train_planner.py:220) b*'i nadir sinifa ('slow') kaydiriyor;
        # b* decision_emb ile head'e girdigi icin ALTI MODU BIRDEN yavaslatiyor. Validation
        # olcumu (n=1118): alpha 0 -> 0.5 ile b*_lon dogrulugu %60.6 -> %65.7, planin 8 s yay
        # sapmasi (ego hareketli) -2.83 m -> -0.84 m, FDE 6.926 -> 6.663 m. alpha=0 = eski davranis.
        self._psi_prior_alpha = float(psi_prior_alpha)
        self._uniform_mask = uniform_mask
        self._ch_logged = False          # ilk frame'de kanal durumunu bir kez yazdir
        self._ch_missing_warned = False  # kanal istendi ama ref path yok uyarisi (bir kez)
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
        # prior yalniz acikken ada yazilir -> eski kosularin planner_name'i degismez, parquet'ler
        # karsilastirilabilir kalir.
        _pr = f";prior={self._psi_prior_alpha:g}" if self._psi_prior_alpha else ""
        return (f"CausalRefinerPlanner[{'causal' if self._use_causal else 'baseline-gf'};"
                f"remove={self._remove}x{self._remove_k};plan={self._plan_source}{_pr}]")

    def _remove_agents(self, features, ch_ref_path=None):
        """M_cas'a gore en causal/az causal k komsuyu girdiden sifirla. RemoveNonCausal, closed-loop."""
        out = self._run_causal(features, ch_ref_path)
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
        self.causal = CausalPlanner(layers=self._graph_layers, modes=self._modes, nbr_enrich=self._nbr_enrich,
                                    ego_residual=self._ego_residual,
                                    gate_channels=self._gate_channels, typed_kv=self._typed_kv,
                                    channel_evidence=self._channel_evidence, gate_trust=self._gate_trust,
                                    dod_meta=self._dod_meta, dec_moe=self._dec_moe,
                                    lat_moe=self._lat_moe,
                                    l1=self._l1, l1_bottleneck=self._l1_bottleneck,
                                    l1_drop_input=self._l1_drop_input,
                                    num_l1_ag=6, num_l1_mp=2,
                                    num_lon=(4 if self._lat_moe else 5 if self._dec_moe
                                             else 6 if self._lon_merge else 9),
                                    num_lat=(5 if (self._dec_moe or self._lat_moe) else 7),
                                    uniform_mask=self._uniform_mask, joint_softmax=self._joint_softmax)
        # strict=False: model sonradan modul kazandi (gate_bias, nbr_head, cfd_recon); eski checkpoint'ler
        # bu anahtarlari icermez. Ucu de PLAN YOLUNUN DISINDA: gate_bias yalniz gate='sigmoid' dalinda
        # okunur (planner softmax kurar), nbr_head/cfd_recon yalniz egitim loss'larini besler. Yine de
        # ne eksikse yazdir -- listede head.* / attn_* / out_fc_* gorunurse sonuc GECERSIZDIR.
        _miss, _unexp = self.causal.load_state_dict(
            torch.load(self._causal_path, map_location=self._device), strict=False)
        if _miss or _unexp:
            print(f'[load] missing={list(_miss)}  unexpected={list(_unexp)}')
        self.causal.psi_prior_alpha = self._psi_prior_alpha
        self.causal.to(self._device).eval()
        if self._psi_prior_alpha:
            print(f"[psi] prior duzeltmesi AKTIF: alpha={self._psi_prior_alpha:g} "
                  f"(psi logit -= alpha*log(CE agirligi); yalniz cikarimda)")
        self.relevance_graph = None

    def _channels_ref_path(self, ego_state, traffic_light_data):
        """Kanal hesabi icin aday rotalar [1,5,P,6] — extract_channels/cache ile AYNI semantik.

        Ego-koridoru adayini compute_channels icindeki select_ego_corridor secer (2026-08-18:
        sabit aday-0 varsayimi kaldirildi; secim sira-bagimsiz, lattice sirasi fark etmez).
        Rota yoksa (tum adaylar bos) DUZ KORIDOR sentezlenir (2026-08-20): eskiden None
        donuyordu -> kanallar kapaniyordu -> model egitimde HIC gormedigi girdiyle calisiyordu
        (v3: egitilmemis untyped dal; v4: tum bloklar sifir) -> hard sette 0.0'lanan senaryolar.
        Sentez = "mevcut serit duz devam ediyor" hipotezi (ego-frame +x, -2..120 m); kanallar
        boylece HER frame hesaplanir, girdi dagilim-ici kalir.
        """
        if not (self._gate_channels or self._typed_kv or self._channel_evidence):
            return None
        c_lat, _ = self.get_multimodal_reference_paths2(
            ego_state, traffic_light_data, points_per_route=MAX_LEN * 10)
        if np.abs(c_lat).sum() < 1e-6:                     # TUM adaylar bos -> sentezle
            self._synth_frames = getattr(self, '_synth_frames', 0) + 1
            if self._synth_frames == 1 or self._synth_frames % 100 == 0:
                print(f"[channels] ref_path yok -> duz koridor sentezlendi "
                      f"(toplam {self._synth_frames} frame)")
            P = MAX_LEN * 10
            synth = np.zeros((5, P, 6), dtype=np.float32)
            synth[0, :, 0] = np.linspace(-2.0, float(MAX_LEN), P)   # x: -2 m -> 120 m
            synth[0, :, 4] = 15.0                                    # v_max varsayilan
            return torch.tensor(synth, dtype=torch.float32, device=self._device).unsqueeze(0)
        return torch.tensor(c_lat, dtype=torch.float32, device=self._device).unsqueeze(0)

    @torch.no_grad()
    def _run_causal(self, features, ch_ref_path=None):
        enc = self.backbone.encoder(features)
        top1, nbr_states, _ = extract_neighbor_top1_futures(self.backbone, enc, self._num_neighbors)
        out = self.causal(enc, features, num_agents=self._num_neighbors + 1,
                          neighbor_futures=top1, neighbor_states=nbr_states,
                          also_cfd_plan=(self._plan_source == 'cfd'), ref_path=ch_ref_path)
        # Kanal durumu teshisi: ilk frame'de bir kez dogrula (typed ckpt + kanal yok = egitilmemis dal!)
        if not self._ch_logged:
            ca = out.get('ch_active')
            if ca is None:
                print(f"[channels] deploy: KAPALI (gate={self._gate_channels} typed={self._typed_kv} "
                      f"evid={self._channel_evidence})")
            else:
                print(f"[channels] deploy: AKTIF — ilk frame'de {int(ca[0].sum())} ajan-kanal, "
                      f"{int(out['mch_active'][0].sum())} harita-kanal girisi yandi "
                      f"(typed={'evet' if out.get('M_cas_typed') is not None else 'hayir'})")
            self._ch_logged = True
        if ((self._gate_channels or self._typed_kv) and out.get('ch_active') is None
                and not self._ch_missing_warned):
            print("[channels] UYARI: kanal bayraklari acik ama bu frame'de ref path yok -> "
                  "kanallar devre disi; typed ckpt'te bu frame'ler EGITILMEMIS untyped dala duser "
                  "(rota-disi frame'lerde beklenir, yayginsa sonuc supheli).")
            self._ch_missing_warned = True
        return out

    @torch.no_grad()
    def _cc_pick(self, traj, score, out, ch_ref, fallback):
        """Karar-tutarli mod secimi: ilan edilen b*'a uyan modlar icinden en yuksek skorlu.
        Oncelik: (lon & lat) -> lat -> lon -> argmax (fallback). Relabel, egitim/eval ile
        AYNI fonksiyon (decision_labels), ckpt sozlugune katlanir."""
        from GameFormer.decision_labels import (decision_labels, LON4_MAP, LAT5V_MAP,
                                                LON5_MAP, LAT5_MAP, LAT5L_MAP)
        M = traj.shape[0]
        xy = traj[:, :, :2].detach().cpu()
        d = xy[:, 1:] - xy[:, :-1]
        hd = torch.atan2(d[..., 1], d[..., 0])
        hd = torch.cat([hd[:, :1], hd], dim=1)
        plans = torch.cat([xy, hd.unsqueeze(-1)], dim=-1)                 # [M,80,3]
        rl, rt = decision_labels(plans, ch_ref.cpu().expand(M, -1, -1, -1))
        if self._lat_moe:
            _lm = LAT5L_MAP if int(self._lat_moe) >= 2 else LAT5V_MAP
            rl, rt = torch.tensor(LON4_MAP)[rl], torch.tensor(_lm)[rt]
        elif self._dec_moe:
            rl, rt = torch.tensor(LON5_MAP)[rl], torch.tensor(LAT5_MAP)[rt]
        bl = int(out['psi_lon_cas'][0].argmax())
        bt = int(out['psi_lat_cas'][0].argmax())
        sc = score.detach().cpu()
        for ok in ((rl == bl) & (rt == bt), (rt == bt), (rl == bl)):
            if bool(ok.any()):
                s2 = sc.clone()
                s2[~ok] = -1e9
                return int(s2.argmax())
        return fallback

    def _causal_neural_plan(self, features, ch_ref_path=None):
        out = self._run_causal(features, ch_ref_path)
        traj_key, score_key = ('traj_cfd', 'score_cfd') if self._plan_source == 'cfd' else ('traj', 'score')
        traj = out[traj_key][0, 0]                    # [M,80,4]
        best = int(out[score_key][0, 0].argmax().item())
        if (self._cc_select and self._dod_meta and ch_ref_path is not None
                and out.get('psi_lon_cas') is not None):
            best = self._cc_pick(traj, out[score_key][0, 0], out, ch_ref_path, best)
        xy = traj[best, :, :2].detach().cpu().numpy()   # [80,2] ego-frame
        diffs = np.diff(xy, axis=0)
        heading = np.arctan2(diffs[:, 1], diffs[:, 0])
        heading = np.concatenate([heading[:1], heading])
        plan = np.concatenate([xy, heading[:, None]], axis=1).astype(np.float32)  # [80,3]
        return torch.from_numpy(plan).unsqueeze(0).to(self._device)               # [1,80,3]

    def _plan(self, ego_state, history, traffic_light_data, observation, iteration=None):
        features = observation_adapter(history, traffic_light_data, self._map_api,
                                       self._route_roadblock_ids, self._device)
        # Kanal ref path'i _remove_agents'tan ONCE hesaplanmali (o da _run_causal cagiriyor).
        ch_ref = self._channels_ref_path(ego_state, traffic_light_data)
        if self._remove != 'none':                   # RemoveNonCausal-via-CLS: k ajani cikar
            features = self._remove_agents(features, ch_ref)
        ref_path = self._get_reference_path(ego_state, traffic_light_data, observation)

        with torch.no_grad():
            _, gf_plan, predictions, scores, ego_cur, nbr_cur = self._get_prediction(features)
            # neural_plan: causal (gate'li) VEYA GameFormer (tum ajanlar, baseline)
            plan = self._causal_neural_plan(features, ch_ref) if self._use_causal else gf_plan

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
