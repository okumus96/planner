"""
EgoCentric Causal Agent Graph (Causal-Planner tarzi, agent-agent, ego-merkezli).

Amac: ego'nun karar verme surecine ETKI EDEN (causal) ajanlari, spurious/etkisiz
(confounding) ajanlardan ayirmak. Cikti = M_cas [B, N] (her komsu ajanin ego icin
"causal" olma agirligi). Bu, kirik SceneRelevanceGraph + ModeSelector ikilisinin yerini alir.

Pipeline (frozen GameFormer backbone uzerinde, sadece bu modul egitilir):
  agent_tokens (fusion ONCESI, temiz per-ajan)  ->  EgoCausalDisentangler
      -> f_cas / f_cfd (causal / confounding ego baglami) + M_cas / M_cfd (per-ajan gate)
  f_cas + ego + AGENT-FREE harita/route baglami  ->  CausalEgoHead (K-modlu trajectory)
      -> L_TRAJ (imitation, GMM WTA)  == yogun egitim sinyali (workhorse)
  psi_cas(f_cas) -> ego modunu tahmin et (L_KLD, informative)   \  ayrisma: causal/confound
  psi_cfd(f_cfd) -> uniform'a it (L_ENT, max entropy)           /  entropi makasi + soft-mask loss

"Causal" tanimi = decision-relevance (per-ajan LABEL YOK). Post-hoc dogrulama:
RemoveNonCausal (dusuk-M_cas ajanlari at -> plan degismesin) + importance viz.

Tasarim notlari:
  - Disentangle = Causal-Planner tarzi softmax KOMSULAR uzerinde (M_cas = softmax(Qav.K_cas)).
    Yani M_cas komsular arasi bir DAGILIM (konveks kombinasyon) -> dogal tepe (rekabet), seyreltme
    yok, cfd_bias/peakiness/normalizasyon HILESI YOK.
    Deliverable M_cas[j] = ajanin goreli causal onemi (komsular uzerinde dagilim, toplam = 1).
  - Harita/route baglami bu modulun KENDI PolylineEncoder'lari ile uretilir (agent-free garanti);
    fused encoder ciktisi ('encoding') KULLANILMAZ (tum ajanlari her token'a karistirir -> gating
    anlamsizlasir; ayni gerekce agent_tokens icin de kodda mevcut).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .predictor_modules import FutureEncoder, GMMPredictor, CrossTransformer
from .relevance_graph import (
    PolylineEncoder, build_edge_features, _polyline_pose_and_valid, EDGE_FEATURE_DIM,
    NODE_TYPE_EGO, NODE_TYPE_VEHICLE, NODE_TYPE_PEDESTRIAN, NODE_TYPE_BICYCLE,
)

NUM_AGENT_TYPES = 4  # ego, vehicle, pedestrian, bicycle


def _agent_types(neighbor_agents_past, num_neighbors):
    """[B, N_all, T, 11] ham komsu -> [B, 1+N] dugum tipi (ego=0, veh=1, ped=2, bic=3).
    Son 3 ozellik one-hot tip (vehicle/ped/bicycle); son timestep'ten alinir."""
    B = neighbor_agents_past.shape[0]
    device = neighbor_agents_past.device
    onehot = neighbor_agents_past[:, :num_neighbors, -1, 8:11]        # [B, N, 3]
    nbr_type = onehot.argmax(-1) + NODE_TYPE_VEHICLE                  # 0->1,1->2,2->3
    # tamamen sifir (padded) ajanlar da vehicle'a dusebilir; onemsiz (valid maske ile elenecek)
    ego_type = torch.full((B, 1), NODE_TYPE_EGO, dtype=torch.long, device=device)
    return torch.cat([ego_type, nbr_type.long()], dim=1)             # [B, 1+N]


class EgoCausalLayer(nn.Module):
    """Ego-oriented DUAL-RELATION causal katmani (Causal-Planner CGD'sine sadik, hdgt_encoder.py).
    Ego query IKI iliskiye attend eder:
        'other' (g2a muadili degil): komsu AJANLAR
        'g2a'                       : HARITA polygonlari (lane/crosswalk/route)
    Her iliski icin AYRI softmax (causal/confound) KEY'ler uzerinde (Eq 5):
        M_cas = softmax(q.k_cas) ,  M_cfd = softmax(q.k_cfd)
    Birlestirme (Eq 6): f_cas = LN( Linear[self ; ajan_cas ; harita_cas] + h_ego ) ; f_cfd benzer.
    Harita SADECE bu gate uzerinden trajectory'ye girer (f_cas) -> harita-gate'i anlamli.
    """

    def __init__(self, dim=256, heads=8, edge_dim=EDGE_FEATURE_DIM, dropout=0.1):
        super().__init__()
        assert dim % heads == 0
        self.dim, self.heads, self.dh = dim, heads, dim // heads
        # 'other' (ajan) iliskisi
        self.Wq_ag = nn.Linear(dim, dim); self.Wk_ag = nn.Linear(dim, dim); self.Wv_ag = nn.Linear(dim, dim)
        self.We_k_ag = nn.Linear(edge_dim, dim); self.We_v_ag = nn.Linear(edge_dim, dim)
        self.attn_cas = nn.Parameter(torch.empty(heads, self.dh))     # ajan causal attn vektoru
        self.attn_cfd = nn.Parameter(torch.empty(heads, self.dh))
        # 'g2a' (harita) iliskisi
        self.Wq_mp = nn.Linear(dim, dim); self.Wk_mp = nn.Linear(dim, dim); self.Wv_mp = nn.Linear(dim, dim)
        self.We_k_mp = nn.Linear(edge_dim, dim); self.We_v_mp = nn.Linear(edge_dim, dim)
        self.attn_cas_mp = nn.Parameter(torch.empty(heads, self.dh))  # harita causal attn vektoru
        self.attn_cfd_mp = nn.Parameter(torch.empty(heads, self.dh))
        for p in (self.attn_cas, self.attn_cfd, self.attn_cas_mp, self.attn_cfd_mp):
            nn.init.xavier_uniform_(p)
        # birlestirme (Eq 6): [self ; ajan ; harita] -> f_cas / f_cfd
        self.self_fc = nn.Sequential(nn.Linear(dim, dim), nn.ReLU())
        # AYRI cikis projeksiyonu + LayerNorm (CP hdgt_encoder.py ile ayni: out_fc_causal/out_fc_confound
        # ve out_ffn_causal/out_ffn_confound ayridir; residual SADECE causal dalda).
        self.out_fc_cas = nn.Linear(3 * dim, dim)
        self.out_fc_cfd = nn.Linear(3 * dim, dim)
        self.norm_cas = nn.LayerNorm(dim)
        self.norm_cfd = nn.LayerNorm(dim)
        # ego node guncelleme (katmanlar arasi)
        self.update = nn.GRUCell(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.leaky = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def _attend(self, q1, k, ek, msg, valid, attn_cas, attn_cfd):
        """q1 [B,1,H,dh]; k/ek/msg [B,Nk,H,dh]; valid [B,Nk] bool. AYRI softmax causal/confound.
        Doner: cas [B,D], cfd [B,D], M_cas [B,Nk], M_cfd [B,Nk] (head-ort)."""
        B = k.shape[0]
        s = self.leaky(q1 + k + ek)                          # [B,Nk,H,dh]
        a_cas = (s * attn_cas).sum(-1)                        # [B,Nk,H]
        a_cfd = (s * attn_cfd).sum(-1)
        invalid = ~valid[:, :, None]                          # [B,Nk,1]
        neg_inf = torch.finfo(a_cas.dtype).min
        M_cas_h = torch.softmax(a_cas.masked_fill(invalid, neg_inf), dim=1).masked_fill(invalid, 0.0)
        M_cfd_h = torch.softmax(a_cfd.masked_fill(invalid, neg_inf), dim=1).masked_fill(invalid, 0.0)
        cas = (M_cas_h.unsqueeze(-1) * msg).sum(dim=1).reshape(B, self.dim)   # [B,D]
        cfd = (M_cfd_h.unsqueeze(-1) * msg).sum(dim=1).reshape(B, self.dim)
        return cas, cfd, M_cas_h.mean(-1), M_cfd_h.mean(-1)

    def forward(self, h_ego, h_nbr, edge_ego, nbr_valid, h_map, edge_map, map_valid):
        """h_ego [B,D]; h_nbr [B,N,D], edge_ego [B,N,De], nbr_valid [B,N];
        h_map [B,S,D], edge_map [B,S,De] (polygon->ego), map_valid [B,S].
        Doner: h_ego_new, f_cas, f_cfd, M_cas(ajan), M_cfd(ajan), M_cas_mp(harita), M_cfd_mp(harita)."""
        B, N = h_nbr.shape[0], h_nbr.shape[1]
        S = h_map.shape[1]
        H, dh = self.heads, self.dh

        # --- ajan iliskisi (other) ---
        q_ag = self.Wq_ag(h_ego).view(B, 1, H, dh)
        k_ag = self.Wk_ag(h_nbr).view(B, N, H, dh)
        ek_ag = self.We_k_ag(edge_ego).view(B, N, H, dh)
        msg_ag = self.Wv_ag(h_nbr).view(B, N, H, dh) + self.We_v_ag(edge_ego).view(B, N, H, dh)
        ag_cas, ag_cfd, M_cas_ag, M_cfd_ag = self._attend(
            q_ag, k_ag, ek_ag, msg_ag, nbr_valid, self.attn_cas, self.attn_cfd)

        # --- harita iliskisi (g2a) ---
        q_mp = self.Wq_mp(h_ego).view(B, 1, H, dh)
        k_mp = self.Wk_mp(h_map).view(B, S, H, dh)
        ek_mp = self.We_k_mp(edge_map).view(B, S, H, dh)
        msg_mp = self.Wv_mp(h_map).view(B, S, H, dh) + self.We_v_mp(edge_map).view(B, S, H, dh)
        mp_cas, mp_cfd, M_cas_mp, M_cfd_mp = self._attend(
            q_mp, k_mp, ek_mp, msg_mp, map_valid, self.attn_cas_mp, self.attn_cfd_mp)

        # --- birlestirme (Eq 6): [self ; ajan ; harita] ---
        self_fea = self.self_fc(h_ego)                                        # [B,D]
        f_cas = self.norm_cas(self.out_fc_cas(torch.cat([self_fea, ag_cas, mp_cas], dim=-1)) + h_ego)
        f_cfd = self.norm_cfd(self.out_fc_cfd(torch.cat([self_fea, ag_cfd, mp_cfd], dim=-1)))
        f_cas = self.dropout(f_cas)
        f_cfd = self.dropout(f_cfd)
        h_ego_new = self.norm(self.update(f_cas + f_cfd, h_ego))
        return h_ego_new, f_cas, f_cfd, M_cas_ag, M_cfd_ag, M_cas_mp, M_cfd_mp


class EgoCausalDisentangler(nn.Module):
    """(A) Ego + komsu ajan dugumlerinden causal/confounding ayrisimini ureten modul.

    Dugum ozellikleri = fusion ONCESI agent_tokens (detached). Opsiyonel: komsu tahmini
    gelecekleri node'lara kaynastir (gelecek-bilincli). L katman; her katman ego node'unu
    gunceller, komsular sabit kalir. Son katmanin f_cas/f_cfd + M_cas/M_cfd ciktisi kullanilir.
    """

    def __init__(self, dim=256, heads=8, layers=3, dropout=0.1):
        super().__init__()
        self.future_encoder = FutureEncoder()
        self.future_fuse = nn.Sequential(nn.Linear(2 * dim, dim), nn.ReLU())
        self.type_embedding = nn.Embedding(NUM_AGENT_TYPES, dim)
        self.input_norm = nn.LayerNorm(dim)
        # HARITA polygon kodlayicilari (agent-free; head'den TASINDI -> harita artik gate uzerinden girer).
        # Frozen GameFormer encoder'ina DOKUNULMAZ; bunlar bizim egitilebilir modullerimiz.
        self.lane_encoder = PolylineEncoder(3, dim)
        self.crosswalk_encoder = PolylineEncoder(3, dim)
        self.route_encoder = PolylineEncoder(3, dim)
        self.map_norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([
            EgoCausalLayer(dim, heads, EDGE_FEATURE_DIM, dropout) for _ in range(layers)
        ])

    def forward(self, agent_feat, agent_valid, agent_pose, agent_types, inputs,
                neighbor_futures=None, neighbor_states=None):
        """agent_feat [B,Na,D], agent_valid [B,Na] bool, agent_pose [B,Na,5], agent_types [B,Na],
        inputs (harita icin). neighbor_futures [B,N,T,2], neighbor_states [B,N,>=5] (opsiyonel)."""
        B, Na, D = agent_feat.shape
        N = Na - 1

        if neighbor_futures is not None and neighbor_states is not None:
            fut_in = neighbor_futures[:, :N].unsqueeze(2)                              # [B,N,1,T,2]
            fut_emb = self.future_encoder(fut_in, neighbor_states[:, :N]).squeeze(2)   # [B,N,D]
            nbr_fused = self.future_fuse(torch.cat([agent_feat[:, 1:1 + N], fut_emb], dim=-1))
            agent_feat = torch.cat([agent_feat[:, :1], nbr_fused], dim=1)              # [B,Na,D]

        h = self.input_norm(agent_feat + self.type_embedding(agent_types))            # [B,Na,D]
        edge = build_edge_features(agent_pose)                                         # [B,Na,Na,De]
        edge_ego = edge[:, 0, 1:]                                                      # komsu j -> ego: [B,N,De]

        h_ego = h[:, 0]                                                                # [B,D]
        h_nbr = h[:, 1:]                                                               # [B,N,D]
        nbr_valid = agent_valid[:, 1:]                                                 # [B,N]

        # --- HARITA polygonlari: kendi encoder'larimizla token (agent-free) + polygon->ego kenari ---
        lanes = inputs['map_lanes'][..., :3].float()
        cwalks = inputs['map_crosswalks'][..., :3].float()
        routes = inputs['route_lanes'][..., :3].float()
        h_map = torch.cat([self.lane_encoder(lanes), self.crosswalk_encoder(cwalks),
                           self.route_encoder(routes)], dim=1)                          # [B,S,D]
        h_map = self.map_norm(h_map)
        lane_pose, lane_v = _polyline_pose_and_valid(lanes)
        cw_pose, cw_v = _polyline_pose_and_valid(cwalks)
        rt_pose, rt_v = _polyline_pose_and_valid(routes)
        map_pose = torch.cat([lane_pose, cw_pose, rt_pose], dim=1)                      # [B,S,5]
        map_valid = torch.cat([lane_v, cw_v, rt_v], dim=1)                              # [B,S]
        # [ego_pose ; map_pose] birlesik -> edge[:,0,1:] = polygon(kaynak) -> ego(hedef)
        map_edge_full = build_edge_features(torch.cat([agent_pose[:, 0:1], map_pose], dim=1))  # [B,1+S,1+S,De]
        edge_map = map_edge_full[:, 0, 1:]                                              # [B,S,De]

        f_cas = f_cfd = M_cas = M_cfd = M_cas_mp = M_cfd_mp = None
        for layer in self.layers:
            h_ego, f_cas, f_cfd, M_cas, M_cfd, M_cas_mp, M_cfd_mp = layer(
                h_ego, h_nbr, edge_ego, nbr_valid, h_map, edge_map, map_valid)

        return {
            'f_cas': f_cas, 'f_cfd': f_cfd, 'M_cas': M_cas, 'M_cfd': M_cfd,
            'M_cas_map': M_cas_mp, 'M_cfd_map': M_cfd_mp, 'map_valid': map_valid,
            # ego_feat: katmanlar boyunca f_cas+f_cfd ile guncellenen ego (mask hesabi icin, ICSEL).
            # ego_clean: guncellenmemis ego node'u (SADECE ego'nun kendi gecmisi + tip; komsu/confounding
            # bilgisi YOK). Trajectory head bunu kullanir -> confounding ajan trajectory'ye SIZMAZ.
            'ego_feat': h_ego, 'ego_clean': h[:, 0], 'nbr_valid': nbr_valid,
        }


class CausalEgoHead(nn.Module):
    """(B) K-modlu ego trajectory head. Baglam = [f_cas, ego_clean].

    Causal-Planner DOD gibi: decoder SADECE causal-graph ozelligini (f_cas) kullanir. Harita ARTIK
    f_cas icinde GATE'li olarak gelir (disentangler'daki g2a iliskisi) -> ayri agent-free harita
    baglami YOK (yoksa harita-gate anlamsiz olurdu). GMMPredictor + imitation (WTA GMM).
    """

    def __init__(self, dim=256, modes=6, dropout=0.1):
        super().__init__()
        self.dim, self.modes = dim, modes
        self.mode_query = nn.Embedding(modes, dim)
        self.cross = CrossTransformer(dim=dim, dropout=dropout)
        self.predictor = GMMPredictor(modalities=modes)

    def forward(self, f_cas, ego_token):
        # ego_token = TEMIZ ego (ego_clean): confounding sizintisi olmasin diye. Head'e sahne bilgisi
        # SADECE f_cas (causal-gated ajan+harita) uzerinden girer; f_cfd trajectory'ye hic dokunmaz.
        B = f_cas.shape[0]
        ctx = torch.cat([f_cas[:, None], ego_token[:, None]], dim=1)             # [B,2,D]
        ctx_pad = torch.zeros(B, 2, dtype=torch.bool, device=f_cas.device)       # ikisi de gecerli
        q = self.mode_query.weight[None].expand(B, -1, -1)                       # [B,M,D]
        content = self.cross(q, ctx, ctx, mask=ctx_pad)                          # [B,M,D]
        traj, score = self.predictor(content.unsqueeze(1))                       # [B,1,M,80,4], [B,1,M]
        return traj, score


class CausalPlanner(nn.Module):
    """Ust modul: disentangler (A) + ego head (B) + adversarial psi head'leri (C)."""

    def __init__(self, dim=256, heads=8, layers=3, modes=6, dropout=0.1):
        super().__init__()
        self.disentangler = EgoCausalDisentangler(dim, heads, layers, dropout)
        self.head = CausalEgoHead(dim, modes, dropout)
        # PAYLASILAN karar head'i (Causal-Planner gibi: decisioon_decoder causal VE confound icin AYNI).
        # Ayri head'ler kullanirsak confound head'i "girdiyi yok say -> uniform" hilesiyle entropy'yi
        # trivial cozuyor (gradyan 0 -> f_cfd sekillenmez). Paylasinca head causal'i dogru tahmin etmek
        # ZORUNDA -> hile yapamaz -> uniform baskisi f_cfd FEATURE'larina akar -> entropy CANLI kalir.
        self.psi = nn.Sequential(nn.Linear(dim, 128), nn.ReLU(), nn.Linear(128, modes))

    def forward(self, encoder_outputs, inputs, num_agents,
                neighbor_futures=None, neighbor_states=None, also_cfd_plan=False):
        Na = num_agents
        agent_feat = encoder_outputs['agent_tokens'][:, :Na].detach()   # [B,Na,D] temiz, fusion oncesi
        agent_valid = ~encoder_outputs['mask'][:, :Na]                  # [B,Na] True=gecerli
        agent_pose = encoder_outputs['actors'][:, :Na, -1].detach()     # [B,Na,5]
        agent_types = _agent_types(inputs['neighbor_agents_past'], Na - 1)

        dis = self.disentangler(agent_feat, agent_valid, agent_pose, agent_types, inputs,
                                neighbor_futures=neighbor_futures, neighbor_states=neighbor_states)
        f_cas, f_cfd = dis['f_cas'], dis['f_cfd']

        # ANA plan: head'e TEMIZ ego (ego_clean) + SADECE f_cas girer. f_cas artik gate'li ajan+harita
        # tasir -> trajectory yalnizca causal ajanlar+harita+ego'ya bagli. Inference bunu kullanir.
        traj, score = self.head(f_cas, dis['ego_clean'])                # [B,1,M,80,4], [B,1,M]

        # Tanı/ablasyon amaçlı: f_cfd'den de bir plan üret (aynı EĞİTİLMİŞ head ile). Varsayılan KAPALI
        # -> eğitimde ekstra hesap/gradyan yok, mevcut davranış birebir korunur. AÇILDIĞINDA: "confounding
        # graph gerçekten davranış-belirleyici bilgi taşıyor mu?" sorusunu closed-loop'ta test eder.
        traj_cfd = score_cfd = None
        if also_cfd_plan:
            traj_cfd, score_cfd = self.head(f_cfd, dis['ego_clean'])

        out = {
            'traj': traj, 'score': score,
            'traj_cfd': traj_cfd, 'score_cfd': score_cfd,
            'M_cas': dis['M_cas'], 'M_cfd': dis['M_cfd'],                # ajan causal/confound [B,N]
            'M_cas_map': dis['M_cas_map'], 'M_cfd_map': dis['M_cfd_map'],  # harita causal/confound [B,S]
            'map_valid': dis['map_valid'],
            'f_cas': f_cas, 'f_cfd': f_cfd,
            'nbr_valid': dis['nbr_valid'],
            'psi_cas': self.psi(f_cas),            # PAYLASILAN head -> m* (L_KLD, informative)
            'psi_cfd': self.psi(f_cfd),            # AYNI head -> uniform (L_ENT); hile yapamaz -> entropy canli
        }
        return out
