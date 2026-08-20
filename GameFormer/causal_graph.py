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
from .channels import (compute_channels, compute_map_channels, select_ego_corridor,
                       NUM_CHANNELS, NUM_EVIDENCE, NUM_MAP_CHANNELS, NUM_MAP_EVIDENCE,
                       CH_COLLISION_COURSE, CH_SHARES_INTERSECTION, CH_MERGES)

NUM_AGENT_TYPES = 4  # ego, vehicle, pedestrian, bicycle
CONFLICT_FEATURE_DIM = 4  # [d_route, d_ego_aligned, d_ego_spatial, approaching]
# Zayif-kanit kanallari (GT-vs-GF IoU: %37/%50/%67, CHANNELS_AUDIT.md) -- gate_trust='reliable'
# modunda GATING karari bunlara dayandirilmaz (typed girdileri yine kurulur).
UNRELIABLE_CHANNELS = (CH_COLLISION_COURSE, CH_SHARES_INTERSECTION, CH_MERGES)


def _conflict_features(neighbor_futures, neighbor_states, route_xy, route_valid, ego_speed,
                       ref_xy=None, ref_valid=None, dt=0.1, aligned_mode='straight'):
    """Per-komsu future-conflict feature'lari [B,N,4]. "Causal" icin geometrik proxy: ajanin
    TAHMIN EDILEN future'i ego'nun yoluyla cakisiyor mu. Hepsi ego-frame. SIZINTI YOK -- ego GT
    future kullanilmaz, yalniz sabit-hiz kinematik ekstrapolasyon + route.
      neighbor_futures [B,N,T,2] (frozen decoder top-1); neighbor_states [B,N,>=2];
      route_xy [B,P,2]; route_valid [B,P] bool; ego_speed [B] (m/s);
      ref_xy [B,R,2]/ref_valid [B,R]: lattice planner REFERANS YOLU (c_lat_candidates, graph-search
      ciktisi, data_process.py:156'da uretilip npz'ye yazilmis). Tek bir yol -- route_lanes gibi 10
      ayri serit degil. GF ciktisi ya da GT ego future KULLANILMAZ.
    Doner: [B,N,4] = log1p(d_route), log1p(d_ego_aligned), log1p(d_ego_spatial), approaching/10."""
    B, N, T, _ = neighbor_futures.shape
    dev = neighbor_futures.device
    BIG, CAP = 1e6, 100.0
    fut_valid = neighbor_futures.abs().sum(-1) > 1e-6                       # [B,N,T] 0-pad'i maskele

    # 1) d_route: future noktalarinin route'a min mesafesi -> komsu ego koridoruna giriyor mu
    if route_xy.shape[1] > 0:
        d = torch.cdist(neighbor_futures.reshape(B, N * T, 2), route_xy)    # [B,N*T,P]
        d = d.masked_fill(~route_valid[:, None, :], BIG)
        d_route = d.min(-1).values.reshape(B, N, T)
    else:
        d_route = torch.full((B, N, T), BIG, device=dev)
    d_route = d_route.masked_fill(~fut_valid, BIG).min(-1).values           # [B,N]

    # 2) d_ego_aligned: ZAMAN-HIZALI cakisma. Ego'nun t anindaki konumu, REFERANS YOL uzerinde
    # yay-uzunlugu ego_speed*dt*t kadar ilerlemis nokta.
    #
    # ONCESI DUZ CIZGIYDI: ego_pos(t) = [ego_speed*dt*t, 0]. Ego sola donerken bu "ego duz gidecek"
    # diyordu -> donus boyunca gercekten onemli ajanlar (karsidan gelen, kesisen) hayali duz cizgiden
    # uzak kalip cezalaniyordu. Olculdu: L_conflict'li 3 kosunun UCUNDE de ayni iki donus senaryosu
    # sifirlaniyor (starting_right_turn 0.862->0.000, starting_left_turn 0.798->0.000); CLS kaybinin
    # (0.8579->0.8126) TAMAMI bu iki senaryo. Duz sahnelerde davranis degismez (ref path duz ise
    # yay-yurumesi duz cizgiyle ayni noktayi verir).
    t_idx = torch.arange(1, T + 1, device=dev, dtype=neighbor_futures.dtype)
    s_t = ego_speed[:, None] * dt * t_idx[None, :]                          # [B,T] kat edilecek yay
    if aligned_mode == 'arc' and ref_xy is not None and ref_xy.shape[1] > 1:
        seg = torch.norm(ref_xy[:, 1:] - ref_xy[:, :-1], dim=-1)            # [B,R-1]
        seg = seg * (ref_valid[:, 1:] & ref_valid[:, :-1]).to(seg.dtype)
        cum = torch.cat([torch.zeros(B, 1, device=dev, dtype=seg.dtype), seg.cumsum(-1)], dim=1)
        pick = (cum[:, None, :] - s_t[:, :, None]).abs()                    # [B,T,R]
        pick = pick.masked_fill(~ref_valid[:, None, :], BIG)
        idx = pick.argmin(-1)                                               # [B,T]
        ego_pos = torch.gather(ref_xy, 1, idx[..., None].expand(-1, -1, 2))  # [B,T,2]
    else:
        ego_pos = torch.stack([s_t, torch.zeros_like(s_t)], dim=-1)         # DUZ cizgi (varsayilan)
    d_align = torch.norm(neighbor_futures - ego_pos[:, None], dim=-1)       # [B,N,T]
    d_ego_aligned = d_align.masked_fill(~fut_valid, BIG).min(-1).values     # [B,N]

    # 3) d_ego_reach: komsunun future'i, ego'nun bu ufukta GERCEKTEN katedecegi koridora ne kadar
    # yaklasiyor. Koridor = route'un ego_speed*T*dt icinde kalan parcasi (graph-search referans yolu,
    # npz'de route_lanes olarak hazir; GF ciktisi ya da GT KULLANILMAZ).
    #
    # ONCEKI HALI YANLISTI: d_ego_spatial = min_t ||nbr_future(t)|| yani ego'nun t=0 KONUMUNA mesafe.
    # Arkadaki arac ileri giderken ego'nun SIMDIKI noktasindan gecer -> mesafe ~0 -> ceza yok -> secilir;
    # ondeki arac ileri giderken origin'den UZAKLASIR -> cezalanir. Olculdu: secilen ajanin arkada olma
    # orani 0.288 (conflict'siz) -> 0.62 (conflict'li), hem dogal dagilimin (0.46) hem en-yakinin (0.52)
    # ustunde. Tanim geregi de gecmise bakiyordu: ego oradan zaten gecti, orada cakisma OLAMAZ.
    #
    # d_route'dan farki HIZ: d_route tum route'u alir (ufkun cok otesine uzanabilir), bu ise ego'nun
    # bu 8 saniyede ulasabilecegi parcayla sinirlar.
    corr_xy = ref_xy if ref_xy is not None else route_xy
    corr_v = ref_valid if ref_valid is not None else route_valid
    if corr_xy is not None and corr_xy.shape[1] > 0:
        # TABAN 10 m. 30'a cikarmayi denedim (confE): koridor uzayinca daha cok ajan "yakin" sayiliyor,
        # ceza ayrimi koreliyor. confE ayni anda yay-yurumesini de getirdigi icin CLS dususu (0.8126 ->
        # 0.7935) hangisinden geldigi belirsizdi; taban 10'a geri alindi ki yay-yurumesi tek basina olculsun.
        reach = (ego_speed * dt * T).clamp(min=10.0)                        # [B]
        reach_v = corr_v & (torch.norm(corr_xy, dim=-1) <= reach[:, None])
        d_r = torch.cdist(neighbor_futures.reshape(B, N * T, 2), corr_xy)   # [B,N*T,R]
        d_r = d_r.masked_fill(~reach_v[:, None, :], BIG)
        d_ego_spatial = d_r.min(-1).values.reshape(B, N, T)
        d_ego_spatial = d_ego_spatial.masked_fill(~fut_valid, BIG).min(-1).values   # [B,N]
    else:
        d_ego_spatial = torch.full((B, N), BIG, device=dev)

    # 4) approaching: anlik mesafe - zaman-hizali en yakin yaklasma (pozitif = yaklasiyor).
    # d_ego_aligned uzerinden: ikisi de EGO'ya mesafe, ayni birim. (d_ego_spatial artik koridora
    # mesafe olduğu icin ondan cikarmak elma-armut olurdu.)
    d0 = torch.norm(neighbor_states[..., :2], dim=-1)                       # [B,N]
    approaching = (d0 - d_ego_aligned).clamp(-CAP, CAP)                     # [B,N]

    feats = torch.stack([
        torch.log1p(d_route.clamp(max=CAP)),
        torch.log1p(d_ego_aligned.clamp(max=CAP)),
        torch.log1p(d_ego_spatial.clamp(max=CAP)),
        approaching / 10.0,
    ], dim=-1)                                                             # [B,N,4]
    return feats.detach()   # sabit girdi/hedef; grad yalniz M_cas (loss) veya We_k_ag (feature) uzerinden


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


class _EdgeMLP(nn.Module):
    """Edge-feature encoder: prenorm MLP (CP'nin edge_MLP'sine sadik: Linear->GELU->Linear, hidden=4x,
    LayerNorm GIRISTE). Onceki tek-Linear (We_k_ag/We_v_ag) yerine -- kenar bilgisini (goreli poz)
    zenginlestirir; FAZ 1'in "sig temsil" teshisinin bir parcasi."""

    def __init__(self, in_dim, out_dim, hidden_mult=4, dropout=0.1):
        super().__init__()
        hidden = out_dim * hidden_mult
        self.norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm(x)
        h = self.fc2(self.act(self.fc1(h)))
        return self.dropout(h)


class _FFN(nn.Module):
    """SwiGLU-gated feed-forward (CP'nin PositionwiseFeedForward'ina sadik): kendi prenorm + residual'i
    var, w2(SiLU(w1(x)) * w3(x)), hidden=4x. Attention SADECE token'lar ARASI dogrusal karisim yapar;
    FFN her token'i KENDI icinde dogrusal-olmayan bicimde isler -- bu, EgoCausalLayer'da hic yoktu."""

    def __init__(self, dim, hidden_mult=4, dropout=0.1):
        super().__init__()
        hidden = dim * hidden_mult
        self.norm = nn.LayerNorm(dim)
        self.w1 = nn.Linear(dim, hidden)
        self.w2 = nn.Linear(hidden, dim)
        self.w3 = nn.Linear(dim, hidden)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        h = self.norm(x)
        h = self.w2(self.act(self.w1(h)) * self.w3(h))
        return self.dropout(h) + residual


class _NbrMapEnrichLayer(nn.Module):
    """CP scene-embedding'in (a2g) KIMLIK-KORUYAN hali: HER komsu (ego DEGIL) haritaya (polygon)
    attend edip KENDI yol/serit baglamini kazanir. AJAN->HARITA SADECE -> ajan->ajan karisim YOK,
    komsu kimligi korunur (M_cas hala "ajan j" uzerinden anlamli). Boylece komsular "serit-farkinda"
    olur -> causal-vs-confounding ayrimi (CP: ayni-serit-onundeki=causal, karsi-serit=confounding)
    mumkun olur. Harita STATIK (map<-agent guncellemesi YOK). Ego'ya DOKUNULMAZ -> gate marjinallesmez.
    Kenar-farkindalikli (GATv2-tarzi), komsu->harita goreli geometri skora girer."""

    def __init__(self, dim=256, heads=8, edge_dim=EDGE_FEATURE_DIM, dropout=0.1):
        super().__init__()
        assert dim % heads == 0
        self.dim, self.heads, self.dh = dim, heads, dim // heads
        self.Wq = nn.Linear(dim, dim)      # komsu query (her komsu ayri)
        self.Wk = nn.Linear(dim, dim)      # harita key
        self.Wv = nn.Linear(dim, dim)      # harita value
        self.We_k = _EdgeMLP(edge_dim, dim, dropout=dropout)
        self.We_v = _EdgeMLP(edge_dim, dim, dropout=dropout)
        self.attn = nn.Parameter(torch.empty(heads, self.dh)); nn.init.xavier_uniform_(self.attn)
        self.norm = nn.LayerNorm(dim)
        self.ffn = _FFN(dim, dropout=dropout)
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, h_nbr, h_map, edge_nbr_map, map_valid):
        """h_nbr [B,N,D] (her komsu bir query, HEPSI guncellenir); h_map [B,S,D] (STATIK key/value);
        edge_nbr_map [B,N,S,De] (harita(kaynak)->komsu(hedef) goreli geometri); map_valid [B,S]."""
        B, N = h_nbr.shape[0], h_nbr.shape[1]
        S = h_map.shape[1]
        H, dh = self.heads, self.dh
        q = self.Wq(h_nbr).view(B, N, 1, H, dh)
        k = self.Wk(h_map).view(B, 1, S, H, dh)
        ek = self.We_k(edge_nbr_map).view(B, N, S, H, dh)
        v = self.Wv(h_map).view(B, 1, S, H, dh) + self.We_v(edge_nbr_map).view(B, N, S, H, dh)
        s = self.leaky(q + k + ek)                                      # [B,N,S,H,dh]
        a = (s * self.attn).sum(-1)                                     # [B,N,S,H]
        invalid = ~map_valid[:, None, :, None]                          # [B,1,S,1]
        M = torch.softmax(a.masked_fill(invalid, torch.finfo(a.dtype).min), dim=2).masked_fill(invalid, 0.0)
        ctx = (M.unsqueeze(-1) * v).sum(dim=2).reshape(B, N, H * dh)     # [B,N,D] her komsuya harita baglami
        return self.ffn(self.norm(h_nbr + ctx))                         # [B,N,D] serit-farkinda komsular


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

    def __init__(self, dim=256, heads=8, edge_dim=EDGE_FEATURE_DIM, dropout=0.1, gate='softmax',
                 ag_edge_dim=None, conflict_bias=False, typed_kv=False, map_edge_dim=None):
        super().__init__()
        assert dim % heads == 0
        self.dim, self.heads, self.dh = dim, heads, dim // heads
        self.gate = gate                      # 'softmax' (dagilim) | 'sigmoid' (bagimsiz uyelik)
        # TYPED K/V (channels branch): softmax destegi ajan degil YANAN (ajan, kanal) girdisi;
        # her girdi kendi kanalinin node-K/V setiyle islenir (edge MLP'leri PAYLASILIR).
        # Kanal basina ayri set (paylasim yok, v1): training olceginde nadir kanallar da doyuyor.
        self.typed_kv = bool(typed_kv)
        if self.typed_kv:
            assert gate == 'softmax', "typed_kv v1 yalniz softmax kapiyla"
            self.Wk_ch = nn.ModuleList([nn.Linear(dim, dim) for _ in range(NUM_CHANNELS)])
            self.Wv_ch = nn.ModuleList([nn.Linear(dim, dim) for _ in range(NUM_CHANNELS)])
            self.attn_cas_ch = nn.Parameter(torch.empty(NUM_CHANNELS, heads, self.dh))
            self.attn_cfd_ch = nn.Parameter(torch.empty(NUM_CHANNELS, heads, self.dh))
            self.Wk_mch = nn.ModuleList([nn.Linear(dim, dim) for _ in range(NUM_MAP_CHANNELS)])
            self.Wv_mch = nn.ModuleList([nn.Linear(dim, dim) for _ in range(NUM_MAP_CHANNELS)])
            self.attn_cas_mch = nn.Parameter(torch.empty(NUM_MAP_CHANNELS, heads, self.dh))
            self.attn_cfd_mch = nn.Parameter(torch.empty(NUM_MAP_CHANNELS, heads, self.dh))
            for p in (self.attn_cas_ch, self.attn_cfd_ch, self.attn_cas_mch, self.attn_cfd_mch):
                nn.init.xavier_uniform_(p)
        # EXPLICIT CONFLICT BIAS: causal logit'ten softplus(w)*conflict-mesafe cikarilir. softplus(w)>=0
        # oldugu icin YON yapisal garanti (uzak ajan = yuksek mesafe = yuksek ceza), gucunu model ogrenir.
        self.conflict_bias = conflict_bias
        if conflict_bias:
            self.conflict_w = nn.Parameter(torch.full((3,), -1.0))   # softplus(-1)=0.31, nazik baslangic
        # ag_edge_dim: AJAN edge'i geometri(edge_dim) + opsiyonel conflict feature; harita edge'i HEP edge_dim
        ag_edge_dim = edge_dim if ag_edge_dim is None else ag_edge_dim
        # 'other' (ajan) iliskisi. Query HEP ego (tek), ama KEY/VALUE komsu-tipine gore AYRI (CP
        # wks["other_*"]/wvs["other"] = 4 tiplik ModuleList; tip = okunulan komsunun tipi). Bizde target
        # hep ego oldugu icin query/output/self_fc tek kalir, SADECE komsu-tarafi K/V tip-basina.
        self.Wq_ag = nn.Linear(dim, dim)
        self.Wk_ag = nn.ModuleList([nn.Linear(dim, dim) for _ in range(NUM_AGENT_TYPES)])
        self.Wv_ag = nn.ModuleList([nn.Linear(dim, dim) for _ in range(NUM_AGENT_TYPES)])
        self.We_k_ag = _EdgeMLP(ag_edge_dim, dim, dropout=dropout); self.We_v_ag = _EdgeMLP(ag_edge_dim, dim, dropout=dropout)
        self.attn_cas = nn.Parameter(torch.empty(heads, self.dh))     # ajan causal attn vektoru
        self.attn_cfd = nn.Parameter(torch.empty(heads, self.dh))
        # 'g2a' (harita) iliskisi
        self.Wq_mp = nn.Linear(dim, dim); self.Wk_mp = nn.Linear(dim, dim); self.Wv_mp = nn.Linear(dim, dim)
        # map_edge_dim: harita kenari geometri(edge_dim) + opsiyonel kanal kaniti (channel_evidence)
        map_edge_dim = edge_dim if map_edge_dim is None else map_edge_dim
        self.We_k_mp = _EdgeMLP(map_edge_dim, dim, dropout=dropout); self.We_v_mp = _EdgeMLP(map_edge_dim, dim, dropout=dropout)
        self.attn_cas_mp = nn.Parameter(torch.empty(heads, self.dh))  # harita causal attn vektoru
        self.attn_cfd_mp = nn.Parameter(torch.empty(heads, self.dh))
        for p in (self.attn_cas, self.attn_cfd, self.attn_cas_mp, self.attn_cfd_mp):
            nn.init.xavier_uniform_(p)
        # STEP 3 -- SIGMOID KAPI modu (gate='sigmoid'). Softmax komsular uzerinde bir DAGILIM zorlar:
        # toplam hep 1, yani bos yolda bile "biri nedensel" demek zorunda, ve ajanlar birbiriyle
        # yarisir. Oysa sorumuz ajan-BASINA evet/hayir. Sigmoid'de her kapi bagimsiz [0,1] -> hepsine
        # hayir diyebilir, ve CP'nin comp/excl kayiplari ILK KEZ tatmin edilebilir hale gelir
        # (comp: g_cas+g_cfd=1, excl: g_cas*g_cfd=0 -> birlikte {0,1} ikili atama).
        # Bias -2 ile baslar (sigmoid(-2)~0.12): aksi halde sigmoid(0)=0.5 ile sahnenin YARISI
        # bastan "nedensel" olur ve ilk epoch'lar bunu geri cekmekle gecer.
        self.gate_bias = nn.Parameter(torch.full((4,), -2.0))   # [cas_ag, cfd_ag, cas_mp, cfd_mp]
        # birlestirme (Eq 6): [self ; ajan ; harita] -> f_cas / f_cfd
        self.self_fc = nn.Sequential(nn.Linear(dim, dim), nn.ReLU())
        # AYRI cikis projeksiyonu + LayerNorm (CP hdgt_encoder.py ile ayni: out_fc_causal/out_fc_confound
        # ve out_ffn_causal/out_ffn_confound ayridir; residual SADECE causal dalda).
        self.out_fc_cas = nn.Linear(3 * dim, dim)
        self.out_fc_cfd = nn.Linear(3 * dim, dim)
        self.norm_cas = nn.LayerNorm(dim)
        self.norm_cfd = nn.LayerNorm(dim)
        # h_ego RESIDUAL anahtari (CP Eq 6 = True). Kapatinca ego bilgisi f_cas'a YALNIZ self_fea
        # uzerinden girer; agent teriminin goreli payi artar -> filtrenin (RemoveNonCausal) daha
        # keskin olmasi beklenir. Parametre DEGIL, saf davranis anahtari: egitilmis bir checkpoint'e
        # cikarim aninda False atayip modelin bu yola ne kadar dayandigi olculebilir.
        self.ego_residual = True
        # FAZ 1: FFN eklendi (CP'de var, bizde yoktu -- attention SADECE token'lar arasi dogrusal
        # karisim yapar, FFN her token'i KENDI icinde dogrusal-olmayan isler). Kendi prenorm+residual'i var.
        self.ffn_cas = _FFN(dim, dropout=dropout)
        self.ffn_cfd = _FFN(dim, dropout=dropout)
        self.leaky = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def _attend(self, q1, k, ek, msg, valid, attn_cas, attn_cfd, bias, conflict_bias=None):
        """q1 [B,1,H,dh]; k/ek/msg [B,Nk,H,dh]; valid [B,Nk] bool. AYRI softmax causal/confound.
        Doner: cas [B,D], cfd [B,D], M_cas [B,Nk], M_cfd [B,Nk] (head-ort),
        ent_cas_mean/ent_cas_headmean/ent_cfd_mean/ent_cfd_headmean [B] (elestiri #3, bkz asagi)."""
        B = k.shape[0]
        s = self.leaky(q1 + k + ek)                          # [B,Nk,H,dh]
        a_cas = (s * attn_cas).sum(-1)                        # [B,Nk,H]
        a_cfd = (s * attn_cfd).sum(-1)
        if conflict_bias is not None:
            # SADECE causal logit: uzak/cakismayan ajan -> logit dusuk -> bastirilir. cfd serbest kalir.
            a_cas = a_cas - conflict_bias[:, :, None]
        invalid = ~valid[:, :, None]                          # [B,Nk,1]
        neg_inf = torch.finfo(a_cas.dtype).min
        if self.gate == 'sigmoid':
            # STEP 3: bagimsiz kapilar. Toplama Sum(g)/clamp(min=1) ile normalize -- clamp KRITIK:
            # toplam kapi 1'in altina duserse bolme YAPILMAZ, cikti sifira dogru kuculur. "Kimse
            # nedensel degil" boylece ifade edilebilir hale gelir (softmax'ta imkansizdi).
            M_cas_h = torch.sigmoid(a_cas + bias[0]).masked_fill(invalid, 0.0)
            M_cfd_h = torch.sigmoid(a_cfd + bias[1]).masked_fill(invalid, 0.0)
            den_cas = M_cas_h.sum(dim=1).clamp(min=1.0)[:, None, :, None]      # [B,1,H,1]
            den_cfd = M_cfd_h.sum(dim=1).clamp(min=1.0)[:, None, :, None]
            cas = ((M_cas_h.unsqueeze(-1) * msg) / den_cas).sum(dim=1).reshape(B, self.dim)
            cfd = ((M_cfd_h.unsqueeze(-1) * msg) / den_cfd).sum(dim=1).reshape(B, self.dim)
        else:
            M_cas_h = torch.softmax(a_cas.masked_fill(invalid, neg_inf), dim=1).masked_fill(invalid, 0.0)
            M_cfd_h = torch.softmax(a_cfd.masked_fill(invalid, neg_inf), dim=1).masked_fill(invalid, 0.0)
            # GATING guvenligi: sahnede hic gecerli girdi kalmadiysa softmax(-inf'ler) NaN uretir;
            # o satirlarda M = 0, katki = 0 (bos sahne = bos causal baglam).
            has_any = valid.any(dim=1)                                            # [B]
            if not bool(has_any.all()):
                z = has_any[:, None, None].float()
                M_cas_h = torch.nan_to_num(M_cas_h) * z
                M_cfd_h = torch.nan_to_num(M_cfd_h) * z
            if getattr(self, 'uniform_mask', False):
                # RULES-ONLY baseline: kural hangi girdinin softmax'a girecegini secer,
                # AGIRLIK ogrenilmez -- gate'ten gecen girdiler uzerinde uniform. Confound
                # dali dokunulmadan birakilir (plan yolunda degil).
                u = valid.float()[:, :, None].expand_as(M_cas_h)
                M_cas_h = u / u.sum(dim=1, keepdim=True).clamp(min=1.0)
            cas = (M_cas_h.unsqueeze(-1) * msg).sum(dim=1).reshape(B, self.dim)   # [B,D]
            cfd = (M_cfd_h.unsqueeze(-1) * msg).sum(dim=1).reshape(B, self.dim)

        # ELESTIRI #3: TOPLAMA (yukarida) her head'in KENDI M_cas_h agirligini kullanir, ama
        # rapor edilen M_cas = M_cas_h.mean(-1) (head-ortalamasi). Head'ler farkli komsuya tepe
        # yapiyorsa ortalama duzlesir (entropy(mean) YUKSEK) ama her head kendi icinde keskin
        # kalabilir (mean(entropy_head) DUSUK) -- ikisini ayri logla, entgap=(mean-headmean)
        # buyukse M_cas gorsellestirmesi/mcas_peak head-anlasmazligini gizliyor demektir.
        eps = 1e-12
        M_cas_mean = M_cas_h.mean(-1)                                              # [B,Nk]
        M_cfd_mean = M_cfd_h.mean(-1)
        if self.gate == 'sigmoid':
            # Kapilar bir dagilim DEGIL (toplam != 1). Entropiyi normalize edilmis p = g/Sum(g)
            # uzerinden hesapla ki sayilar softmax kosulariyla ayni olcekte kalsin; UYELIK bilgisi
            # ayrica gcas_mean/gcas_frac05 metrikleriyle raporlanir.
            pc = M_cas_h / M_cas_h.sum(dim=1, keepdim=True).clamp(min=eps)
            pf = M_cfd_h / M_cfd_h.sum(dim=1, keepdim=True).clamp(min=eps)
        else:
            pc, pf = M_cas_h, M_cfd_h
        ent_cas_mean = -(M_cas_mean.clamp(min=eps).log() * M_cas_mean).sum(-1)      # [B], NATS (ham)
        ent_cas_headmean = (-(pc.clamp(min=eps).log() * pc).sum(1)).mean(-1)   # [B]
        ent_cfd_mean = -(M_cfd_mean.clamp(min=eps).log() * M_cfd_mean).sum(-1)
        ent_cfd_headmean = (-(pf.clamp(min=eps).log() * pf).sum(1)).mean(-1)

        # NORMALIZASYON (elestiri): maksimum entropi log(n_valid), n_valid sahneden sahneye
        # (2-40) degisir. Ham nats'i batch uzerinden ortalamak kalabalik sahneleri (buyuk log(n))
        # seyrek sahnelere (kucuk log(n)) baskin kilar -- gap/ent HER SEYİ log(n_valid)'e bolerek
        # [0,1] araligina (uniform=1, tam tepe=0) normalize ediyoruz, batch-agnostik kiyaslanabilir.
        n_valid = valid.sum(-1).clamp(min=1).float()                                # [B]
        log_n = n_valid.clamp(min=2).log()          # n_valid=1 -> entropi zaten 0, payda guvenli
        ent_cas_mean = ent_cas_mean / log_n
        ent_cas_headmean = ent_cas_headmean / log_n
        ent_cfd_mean = ent_cfd_mean / log_n
        ent_cfd_headmean = ent_cfd_headmean / log_n

        return (cas, cfd, M_cas_mean, M_cfd_mean,
                ent_cas_mean, ent_cas_headmean, ent_cfd_mean, ent_cfd_headmean)

    def _attend_typed(self, q1, h_src, ek, edge_v, entry_valid, Wk_list, Wv_list,
                      attn_cas_ch, attn_cfd_ch, conflict_bias=None):
        """TYPED attention: softmax YANAN (kaynak, kanal) girdileri uzerinde.
        q1 [B,1,H,dh]; h_src [B,S,D]; ek/edge_v [B,S,H,dh] (edge MLP'leri kanallar arasi PAYLASILIR);
        entry_valid [B,S,R] bool. Doner: cas/cfd [B,D], M_cas/M_cfd ajan-duzeyi [B,S] (kanal toplami,
        eski eval/log uyumu), M_cas_typed/M_cfd_typed [B,S,R]."""
        B, S, D = h_src.shape
        H, dh = self.heads, self.dh
        R = entry_valid.shape[-1]
        # kanal-basina node K/V: [B,S,R,H,dh]
        k_ch = torch.stack([m(h_src) for m in Wk_list], dim=2).view(B, S, R, H, dh)
        v_ch = torch.stack([m(h_src) for m in Wv_list], dim=2).view(B, S, R, H, dh)
        msg = v_ch + edge_v[:, :, None]                                   # edge value paylasilir
        s = self.leaky(q1[:, :, None] + k_ch + ek[:, :, None])            # [B,S,R,H,dh]
        a_cas = (s * attn_cas_ch[None, None]).sum(-1)                     # [B,S,R,H]
        a_cfd = (s * attn_cfd_ch[None, None]).sum(-1)
        if conflict_bias is not None:                                     # per-ajan ceza -> girdilere yay
            a_cas = a_cas - conflict_bias[:, :, None, None]
        neg_inf = torch.finfo(a_cas.dtype).min
        inv = ~entry_valid[..., None]                                     # [B,S,R,1]
        a_cas = a_cas.masked_fill(inv, neg_inf).view(B, S * R, H)
        a_cfd = a_cfd.masked_fill(inv, neg_inf).view(B, S * R, H)
        M_cas_h = torch.softmax(a_cas, dim=1)
        M_cfd_h = torch.softmax(a_cfd, dim=1)
        has_any = entry_valid.view(B, -1).any(dim=1)                      # bos sahne guvenligi
        z = has_any[:, None, None].float()
        M_cas_h = torch.nan_to_num(M_cas_h) * z
        M_cfd_h = torch.nan_to_num(M_cfd_h) * z
        M_cas_h = M_cas_h.view(B, S, R, H).masked_fill(~entry_valid[..., None], 0.0)
        M_cfd_h = M_cfd_h.view(B, S, R, H).masked_fill(~entry_valid[..., None], 0.0)
        if getattr(self, 'uniform_mask', False):
            # RULES-ONLY: yanan (kaynak, kanal) girdileri uzerinde uniform agirlik
            u = entry_valid[..., None].float().expand_as(M_cas_h)
            M_cas_h = u / u.sum(dim=(1, 2), keepdim=True).clamp(min=1.0)
        cas = (M_cas_h[..., None] * msg).sum(dim=(1, 2)).reshape(B, self.dim)
        cfd = (M_cfd_h[..., None] * msg).sum(dim=(1, 2)).reshape(B, self.dim)
        M_cas_typed = M_cas_h.mean(-1)                                    # [B,S,R] (head-ort)
        M_cfd_typed = M_cfd_h.mean(-1)
        return cas, cfd, M_cas_typed.sum(-1), M_cfd_typed.sum(-1), M_cas_typed, M_cfd_typed

    @staticmethod
    def _per_type(mods, x, types):
        """x [B,N,D], types [B,N] long, mods = T tiplik ModuleList -> her token'a KENDI tipinin
        Linear'ini uygula (CP wks/wvs["other"][t] mantigi). Tumu hesaplanip tip'e gore toplanir."""
        stacked = torch.stack([m(x) for m in mods], dim=2)                  # [B,N,T,D]
        idx = types.clamp(min=0, max=len(mods) - 1)[:, :, None, None].expand(-1, -1, 1, x.shape[-1])
        return stacked.gather(2, idx).squeeze(2)                            # [B,N,D]

    def forward(self, h_ego, h_nbr, nbr_types, edge_ego, nbr_valid, h_map, edge_map, map_valid,
                conflict=None, ch_active=None, mch_active=None):
        """h_ego [B,D]; h_nbr [B,N,D] (K=V ayni kaynak), nbr_types [B,N] long (komsu tipi -> tip-basina
        K/V), edge_ego [B,N,De], nbr_valid [B,N]; h_map [B,S,D], edge_map [B,S,De] (polygon->ego), map_valid [B,S].
        Doner: h_ego_new, f_cas, f_cfd, M_cas(ajan), M_cfd(ajan), M_cas_mp(harita), M_cfd_mp(harita)."""
        B, N = h_nbr.shape[0], h_nbr.shape[1]
        S = h_map.shape[1]
        H, dh = self.heads, self.dh

        # --- ajan iliskisi (other): KEY/VALUE komsu-tipine gore AYRI (CP tip-basina wks/wvs) ---
        q_ag = self.Wq_ag(h_ego).view(B, 1, H, dh)
        ek_ag = self.We_k_ag(edge_ego).view(B, N, H, dh)
        ev_ag = self.We_v_ag(edge_ego).view(B, N, H, dh)
        cb = None
        if self.conflict_bias and conflict is not None:
            cb = (F.softplus(self.conflict_w) * conflict[..., :3]).sum(-1)          # [B,N] >= 0
        M_cas_typed = M_cfd_typed = None
        if self.typed_kv and ch_active is not None:
            # TYPED: girdiler = yanan (ajan, kanal); node K/V kanal setinden, edge paylasilir.
            entry_valid = nbr_valid[:, :, None] & ch_active
            (ag_cas, ag_cfd, M_cas_ag, M_cfd_ag, M_cas_typed, M_cfd_typed) = self._attend_typed(
                q_ag, h_nbr, ek_ag, ev_ag, entry_valid, self.Wk_ch, self.Wv_ch,
                self.attn_cas_ch, self.attn_cfd_ch, conflict_bias=cb)
            ent_cas_mean = ent_cas_headmean = ent_cfd_mean = ent_cfd_headmean = \
                torch.zeros(B, device=h_ego.device)
        else:
            k_ag = self._per_type(self.Wk_ag, h_nbr, nbr_types).view(B, N, H, dh)
            msg_ag = self._per_type(self.Wv_ag, h_nbr, nbr_types).view(B, N, H, dh) + ev_ag
            (ag_cas, ag_cfd, M_cas_ag, M_cfd_ag,
             ent_cas_mean, ent_cas_headmean, ent_cfd_mean, ent_cfd_headmean) = self._attend(
                q_ag, k_ag, ek_ag, msg_ag, nbr_valid, self.attn_cas, self.attn_cfd, self.gate_bias[0:2],
                conflict_bias=cb)

        # --- harita iliskisi (g2a) ---
        q_mp = self.Wq_mp(h_ego).view(B, 1, H, dh)
        ek_mp = self.We_k_mp(edge_map).view(B, S, H, dh)
        ev_mp = self.We_v_mp(edge_map).view(B, S, H, dh)
        M_cas_mp_typed = M_cfd_mp_typed = None
        if self.typed_kv and mch_active is not None:
            entry_valid_mp = map_valid[:, :, None] & mch_active
            (mp_cas, mp_cfd, M_cas_mp, M_cfd_mp, M_cas_mp_typed, M_cfd_mp_typed) = self._attend_typed(
                q_mp, h_map, ek_mp, ev_mp, entry_valid_mp, self.Wk_mch, self.Wv_mch,
                self.attn_cas_mch, self.attn_cfd_mch)
            ent_cas_mp_mean = ent_cas_mp_headmean = ent_cfd_mp_mean = ent_cfd_mp_headmean = \
                torch.zeros(B, device=h_ego.device)
        else:
            k_mp = self.Wk_mp(h_map).view(B, S, H, dh)
            msg_mp = self.Wv_mp(h_map).view(B, S, H, dh) + ev_mp
            (mp_cas, mp_cfd, M_cas_mp, M_cfd_mp,
             ent_cas_mp_mean, ent_cas_mp_headmean, ent_cfd_mp_mean, ent_cfd_mp_headmean) = self._attend(
                q_mp, k_mp, ek_mp, msg_mp, map_valid, self.attn_cas_mp, self.attn_cfd_mp, self.gate_bias[2:4])

        # --- GATE'SIZ TAM-SAHNE OZETI (f_all): causal/confound ayrimi YOK, ogrenilen secim YOK.
        # Gecerli komsularin/polygonlarin DUZ ORTALAMASI. Step 2'de [f_cas; f_cfd] bunu geri
        # kurmak zorunda -> f_cfd'nin sabit vektore cokmesi (fcfd_var ~ 1e-4) yasaklanir.
        # Sadece hedef olarak kullanilir (loss'ta detach) -> ana toplamayi carpitmaz.
        # typed modda msg_ag/msg_mp yaratilmadi -> f_all icin UNTYPED value transformlariyla uret
        # (f_all gate'siz TESHIS/recon hedefi; typed olmasi gerekmiyor).
        if self.typed_kv and ch_active is not None:
            msg_ag = self._per_type(self.Wv_ag, h_nbr, nbr_types).view(B, N, H, dh) + ev_ag
        if self.typed_kv and mch_active is not None:
            msg_mp = self.Wv_mp(h_map).view(B, S, H, dh) + ev_mp
        w_ag = nbr_valid.float()
        w_ag = w_ag / w_ag.sum(-1, keepdim=True).clamp(min=1.0)               # [B,N]
        all_ag = (w_ag[:, :, None, None] * msg_ag).sum(dim=1).reshape(B, self.dim)
        w_mp = map_valid.float()
        w_mp = w_mp / w_mp.sum(-1, keepdim=True).clamp(min=1.0)               # [B,S]
        all_mp = (w_mp[:, :, None, None] * msg_mp).sum(dim=1).reshape(B, self.dim)

        # --- birlestirme (Eq 6): [self ; ajan ; harita] ---
        self_fea = self.self_fc(h_ego)                                        # [B,D]
        f_all = torch.cat([self_fea, all_ag, all_mp], dim=-1)                 # [B,3D] recon hedefi
        cas_pre = self.out_fc_cas(torch.cat([self_fea, ag_cas, mp_cas], dim=-1))
        f_cas = self.norm_cas(cas_pre + h_ego if self.ego_residual else cas_pre)
        f_cfd = self.norm_cfd(self.out_fc_cfd(torch.cat([self_fea, ag_cfd, mp_cfd], dim=-1)))
        f_cas = self.ffn_cas(f_cas)      # FAZ 1: kendi prenorm+residual'iyla ek dogrusal-olmayan isleme
        f_cfd = self.ffn_cfd(f_cfd)
        f_cas = self.dropout(f_cas)
        f_cfd = self.dropout(f_cfd)
        # cos(f_cas, h_ego) -- ~1 => gate marjinal (h_ego bypass'i baskin), dusuk => gate canli.
        with torch.no_grad():
            gate_cos = F.cosine_similarity(f_cas, h_ego, dim=-1)       # [B]
        # ego'yu katmanlar arasi f_cas ile tasi (GRU YOK; f_cfd DAHIL DEGIL -> confound sizmaz).
        h_ego_new = f_cas
        return (h_ego_new, f_cas, f_cfd, M_cas_ag, M_cfd_ag, M_cas_mp, M_cfd_mp,
                ent_cas_mean, ent_cas_headmean, ent_cfd_mean, ent_cfd_headmean,
                ent_cas_mp_mean, ent_cas_mp_headmean, ent_cfd_mp_mean, ent_cfd_mp_headmean,
                gate_cos, f_all, M_cas_typed, M_cfd_typed, M_cas_mp_typed, M_cfd_mp_typed)


class EgoCausalDisentangler(nn.Module):
    """(A) Ego + komsu ajan dugumlerinden causal/confounding ayrisimini ureten modul.

    Dugum ozellikleri = fusion ONCESI agent_tokens (detached). Opsiyonel: komsu tahmini
    gelecekleri node'lara kaynastir (gelecek-bilincli). L katman; her katman ego node'unu
    gunceller, komsular sabit kalir. Son katmanin f_cas/f_cfd + M_cas/M_cfd ciktisi kullanilir.
    """

    def __init__(self, dim=256, heads=8, layers=3, dropout=0.1, nbr_enrich=0, gate='softmax',
                 conflict_feats=0, conflict_bias=0, gate_channels=0, typed_kv=0,
                 channel_evidence=0, gate_trust='all'):
        super().__init__()
        # --- predicate kanallari (channels branch) ---
        self.gate_channels = bool(gate_channels)   # yapisal gating: kanali yanmayan girdi yarismaz
        self.typed_kv = bool(typed_kv)             # (ajan,kanal) girdileri + kanal-basina K/V
        self.channel_evidence = bool(channel_evidence)  # 9/8-dim kanit edge'e concat
        assert gate_trust in ('all', 'reliable')
        self.gate_trust = gate_trust               # 'reliable': zayif-IoU kanallar GATE karari disi
        self.conflict_feats = conflict_feats     # 1 -> ajan edge'ine conflict feature'lari EKLE
        self.conflict_bias = conflict_bias       # 1 -> causal logit'e explicit conflict cezasi
        # CausalPlanner bunu ayrica set eder; feature/bias kapaliyken bile L_conflict icin hesaplatir.
        self.compute_conflict = bool(conflict_feats or conflict_bias)
        self.aligned_mode = 'straight'   # CausalPlanner set eder: 'straight' (A/B/D) | 'arc' (E/F)
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
        # KOMSU->HARITA enrichment (CP scene-embedding a2g, kimlik-koruyan): causal split'ten ONCE her
        # komsuya serit/yol baglami kazandirir. nbr_enrich=0 => mevcut davranis (enrichment YOK).
        self.nbr_enrich = nn.ModuleList([
            _NbrMapEnrichLayer(dim, heads, EDGE_FEATURE_DIM, dropout) for _ in range(nbr_enrich)
        ])
        self.layers = nn.ModuleList([
            EgoCausalLayer(dim, heads, EDGE_FEATURE_DIM, dropout, gate=gate,
                           ag_edge_dim=(EDGE_FEATURE_DIM
                                        + (CONFLICT_FEATURE_DIM if conflict_feats else 0)
                                        + (NUM_EVIDENCE if channel_evidence else 0)),
                           map_edge_dim=(EDGE_FEATURE_DIM
                                         + (NUM_MAP_EVIDENCE if channel_evidence else 0)),
                           conflict_bias=bool(conflict_bias),
                           typed_kv=bool(typed_kv)) for _ in range(layers)
        ])

    def forward(self, agent_feat, agent_valid, agent_pose, agent_types, inputs,
                neighbor_futures=None, neighbor_states=None, ref_path=None):
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

        ego_clean = h[:, 0]           # head'e verilen temiz ego (SADECE ego'nun kendi gecmisi + tip)
        nbr_valid = agent_valid[:, 1:]                                                 # [B,N]
        nbr_types = agent_types[:, 1:]                                                 # [B,N] tip-basina K/V

        # --- PREDICATE KANALLARI (channels branch) ---
        # Kaynak onceligi: (1) cache (inputs, extract_channels damgasi -- training yolu);
        # (2) on-the-fly (deployment: GF future + ref_path eldeyse compute_channels);
        # (3) yoksa devre disi (flag'ler etkisiz, eski davranis).
        need_ch = self.gate_channels or self.typed_kv or self.channel_evidence
        ch_active = ch_evid = mch_active = mch_evid = None
        if need_ch:
            if "channel_active" in inputs:
                ch_active = inputs["channel_active"][:, :N].bool()
                ch_evid = inputs["channel_evidence"][:, :N].float()
                mch_active = inputs["map_channel_active"].bool()
                mch_evid = inputs["map_channel_evidence"].float()
            elif neighbor_futures is not None and ref_path is not None:
                ch_active, ch_evid = compute_channels(
                    inputs["neighbor_agents_past"][:, :N], inputs["ego_agent_past"],
                    neighbor_futures[:, :N], ref_path)
                mch_active, mch_evid = compute_map_channels(
                    inputs["map_lanes"], inputs["map_crosswalks"],
                    inputs["route_lanes"], ref_path)

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

        h_ego = h[:, 0]                                                                # [B,D]
        h_nbr = h[:, 1:]                                                               # [B,N,D] temiz per-ajan (K=V)

        # --- KOMSU->HARITA ENRICHMENT (causal split ONCESI): her komsu serit/yol baglami kazanir ---
        # Ego'ya DOKUNULMAZ (gate marjinallesmesin). Ajan->ajan YOK (kimlik korunur). Harita STATIK.
        if len(self.nbr_enrich) > 0:
            nbr_map_edge = build_edge_features(torch.cat([agent_pose[:, 1:], map_pose], dim=1))  # [B,N+S,N+S,De]
            edge_nbr_map = nbr_map_edge[:, :N, N:]                                      # [B,N,S,De] harita->komsu
            for enrich in self.nbr_enrich:
                h_nbr = enrich(h_nbr, h_map, edge_nbr_map, map_valid)

        # --- FUTURE-CONFLICT feature'lari ---
        # UC AYRI kullanim, uc ayri bayrak (gat-conflict'te ilk ikisi tek bayraktaydi):
        #   compute_conflict -> hesapla ve out'a koy (L_conflict loss'u icin YETER, girdiyi degistirmez)
        #   conflict_feats   -> AYRICA ajan edge'ine ekle (model girdi olarak gorur)
        #   conflict_bias    -> AYRICA causal logit'ten explicit ceza cikar
        # Ayrilmasinin sebebi: "bilgiyi girdiye koymak" (olculdu: 1.24->1.26x, etkisiz) ile
        # "maskeyi denetlemek" (1.65x) farkli seyler; ikisi ayri ayri olculebilmeli.
        conf = None
        if self.compute_conflict and neighbor_futures is not None and neighbor_states is not None:
            route_pts = inputs['route_lanes'][..., :2].float().reshape(B, -1, 2)       # [B,P,2]
            route_v = route_pts.abs().sum(-1) > 1e-6                                    # [B,P]
            ego_speed = torch.norm(agent_pose[:, 0, 3:5], dim=-1)                       # [B]
            ref_xy = ref_v = None
            if ref_path is not None:
                # SADECE ego-seridi adayi (2026-08-18: sabit "aday 0" varsayimi kaldirildi --
                # select_ego_corridor, ego konumuna projeksiyon ile seridinden baslayan adayi
                # secer; sira-bagimsiz, lateral-sortlu loader c_lat'inda da dogru calisir).
                # Diger adaylar alternatif rotalar; hepsini birlestirmek route_lanes'teki
                # hatanin aynisi olurdu (sahnelerin %81'i 2-5 adayli).
                _sel = select_ego_corridor(ref_path)                                    # [B]
                rp = ref_path[torch.arange(ref_path.shape[0], device=ref_path.device),
                              _sel][..., :2].float()                                    # [B,R,2]
                ref_v = rp.abs().sum(-1) > 1e-6
                ref_xy = rp
            conf = _conflict_features(neighbor_futures[:, :N], neighbor_states[:, :N],
                                      route_pts, route_v, ego_speed,
                                      ref_xy=ref_xy, ref_valid=ref_v,
                                      aligned_mode=self.aligned_mode)                    # [B,N,4]
            if self.conflict_feats:
                edge_ego = torch.cat([edge_ego, conf], dim=-1)                          # [B,N,De+4]

        # --- KANAL uygulamalari (channels branch) ---
        # (1) evidence -> edge concat (sira: geometri + [conflict] + [evidence], ag_edge_dim ile ayni)
        if self.channel_evidence and ch_evid is not None:
            edge_ego = torch.cat([edge_ego, ch_evid], dim=-1)
            edge_map = torch.cat([edge_map, mch_evid], dim=-1)
        # (2) yapisal gating: hicbir kanali yanmayan girdi softmax'a giremez.
        #     gate_trust='reliable': zayif-IoU kanallar (collision/intersect/merges) GATE kararina
        #     sayilmaz -- typed girdileri (ajan gecerliyse) yine kurulur.
        gated_valid, gated_map_valid = nbr_valid, map_valid
        if self.gate_channels and ch_active is not None:
            gate_src = ch_active
            if self.gate_trust == 'reliable':
                gate_src = ch_active.clone()
                gate_src[..., list(UNRELIABLE_CHANNELS)] = False
            gated_valid = nbr_valid & gate_src.any(-1)
            gated_map_valid = map_valid & mch_active.any(-1)

        # --- STAGE B: ego-merkezli causal-split (ajan + harita, causal/confound) ---
        f_cas = f_cfd = M_cas = M_cfd = M_cas_mp = M_cfd_mp = f_all = None
        ent_cas_mean = ent_cas_headmean = ent_cfd_mean = ent_cfd_headmean = None
        ent_cas_mp_mean = ent_cas_mp_headmean = ent_cfd_mp_mean = ent_cfd_mp_headmean = None
        M_cas_ty = M_cfd_ty = M_cas_mp_ty = M_cfd_mp_ty = None
        gate_cos_layers = []          # elestiri: katman-basina cos(f_cas, h_ego), bypass/GRU-fix etkilesimi
        for layer in self.layers:
            (h_ego, f_cas, f_cfd, M_cas, M_cfd, M_cas_mp, M_cfd_mp,
             ent_cas_mean, ent_cas_headmean, ent_cfd_mean, ent_cfd_headmean,
             ent_cas_mp_mean, ent_cas_mp_headmean, ent_cfd_mp_mean, ent_cfd_mp_headmean,
             gate_cos, f_all, M_cas_ty, M_cfd_ty, M_cas_mp_ty, M_cfd_mp_ty) = layer(
                h_ego, h_nbr, nbr_types, edge_ego, gated_valid, h_map, edge_map, gated_map_valid,
                conflict=conf,
                ch_active=(ch_active if self.typed_kv else None),
                mch_active=(mch_active if self.typed_kv else None))
            gate_cos_layers.append(gate_cos)
        gate_cos_stack = torch.stack(gate_cos_layers, dim=1)   # [B, L] -- L=len(self.layers)

        return {
            'f_cas': f_cas, 'f_cfd': f_cfd, 'M_cas': M_cas, 'M_cfd': M_cfd,
            'M_cas_map': M_cas_mp, 'M_cfd_map': M_cfd_mp, 'map_valid': map_valid,
            'f_all': f_all,                               # [B,3D] gate'siz tam-sahne ozeti (recon hedefi)
            'conflict': conf,                             # [B,N,4] future-conflict (L_conflict hedefi)
            # elestiri #3: M_cas/M_cfd'nin (head-ortalamasi) entropisi vs head-basina entropinin
            # ortalamasi (normalize: /log(n_valid), batch-agnostik) -- ikisi arasindaki fark,
            # head'lerin farkli komsulara/polygonlara tepe yapip yapmadigini gosterir. Harita icin de
            # ayni tesihs tutuluyor: peak/uniform=7.33x iddiasi head-anlasmazligi artefakti mi diye.
            'M_cas_ent': ent_cas_mean, 'M_cas_headent': ent_cas_headmean,
            'M_cfd_ent': ent_cfd_mean, 'M_cfd_headent': ent_cfd_headmean,
            'M_cas_map_ent': ent_cas_mp_mean, 'M_cas_map_headent': ent_cas_mp_headmean,
            'M_cfd_map_ent': ent_cfd_mp_mean, 'M_cfd_map_headent': ent_cfd_mp_headmean,
            # ego_clean: hicbir zaman guncellenmeyen ham ego (ego'nun kendi gecmisi + tip); head bunu kullanir.
            'ego_feat': h_ego, 'ego_clean': ego_clean, 'nbr_valid': nbr_valid,
            # kanal ciktilari (channels branch): typed maskeler [B,N,R]/[B,S,R2], gate'li gecerlilik,
            # ve kullanilan aktivasyonlar (analiz/faithfulness icin)
            'M_cas_typed': M_cas_ty, 'M_cfd_typed': M_cfd_ty,
            'M_cas_map_typed': M_cas_mp_ty, 'M_cfd_map_typed': M_cfd_mp_ty,
            'gated_valid': gated_valid, 'gated_map_valid': gated_map_valid,
            'ch_active': ch_active, 'mch_active': mch_active,
            # katman-basina cos(f_cas, h_ego) [B,L] -- ~1'e yakinsa gate marjinal (h_ego bypass'i baskin).
            'gate_cos': gate_cos_stack,
        }


class CausalEgoHead(nn.Module):
    """(B) K-modlu ego trajectory head. Baglam = [f_cas, ego_clean].

    Causal-Planner DOD gibi: decoder SADECE causal-graph ozelligini (f_cas) kullanir. Harita ARTIK
    f_cas icinde GATE'li olarak gelir (disentangler'daki g2a iliskisi) -> ayri agent-free harita
    baglami YOK (yoksa harita-gate anlamsiz olurdu). GMMPredictor + imitation (WTA GMM).
    """

    def __init__(self, dim=256, modes=6, dropout=0.1, num_maneuvers=5,
                 dod_meta=False, num_lon=9, num_lat=7):
        super().__init__()
        self.dim, self.modes = dim, modes
        self.dod_meta = bool(dod_meta)
        self.mode_query = nn.Embedding(modes, dim)
        # DOD (CP III-C): tahmin edilen karar (b* = argmax psi) decoder query'sine besleniyor.
        # decision_emb[b*] her mode query ile birlesip MLP'den gecer -> q_enh = MLP([q_mode; theta_b*]).
        # dod_meta (H): 5-sinif geometrik manevra YERINE factored (lon x lat) meta-aksiyon
        # (nuReasoning taksonomisi, decision_labels.py). Iki embedding TOPLANIR ve AYNI porttan
        # girer -> karar slotu TEK kalir (paralel yol yok, b*-swap falsification testi temiz).
        if self.dod_meta:
            self.decision_emb_lon = nn.Embedding(num_lon, dim)
            self.decision_emb_lat = nn.Embedding(num_lat, dim)
        else:
            self.decision_emb = nn.Embedding(num_maneuvers, dim)
        self.q_enh = nn.Sequential(nn.Linear(2 * dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.cross = CrossTransformer(dim=dim, dropout=dropout)
        self.predictor = GMMPredictor(modalities=modes)

    def forward(self, f_cas, ego_token, b_star=None):
        # ego_token = TEMIZ ego (ego_clean): confounding sizintisi olmasin diye. Head'e sahne bilgisi
        # SADECE f_cas (causal-gated ajan+harita) uzerinden girer; f_cfd trajectory'ye hic dokunmaz.
        B = f_cas.shape[0]
        ctx = torch.cat([f_cas[:, None], ego_token[:, None]], dim=1)             # [B,2,D]
        ctx_pad = torch.zeros(B, 2, dtype=torch.bool, device=f_cas.device)       # ikisi de gecerli
        q = self.mode_query.weight[None].expand(B, -1, -1)                       # [B,M,D]
        if b_star is not None:
            # DOD: her mode query'ye tahmin edilen karari (embedding) besle.
            if self.dod_meta:
                b_lon, b_lat = b_star                                            # ([B], [B])
                dec_vec = self.decision_emb_lon(b_lon) + self.decision_emb_lat(b_lat)
            else:
                dec_vec = self.decision_emb(b_star)
            dec = dec_vec[:, None].expand(-1, self.modes, -1)                    # [B,M,D]
            q = self.q_enh(torch.cat([q, dec], dim=-1))                          # [B,M,D] karar-hizali query
        content = self.cross(q, ctx, ctx, mask=ctx_pad)                          # [B,M,D]
        traj, score = self.predictor(content.unsqueeze(1))                       # [B,1,M,80,4], [B,1,M]
        return traj, score


class CausalPlanner(nn.Module):
    """Ust modul: disentangler (A) + ego head (B) + adversarial psi head'leri (C)."""

    def __init__(self, dim=256, heads=8, layers=3, modes=6, dropout=0.1, nbr_enrich=0, num_maneuvers=5,
                 recon_drop=0.5, num_neighbors=10, future_steps=80, gate='softmax',
                 conflict_feats=0, conflict_bias=0, compute_conflict=0, aligned_mode='straight',
                 ego_residual=1, gate_channels=0, typed_kv=0, channel_evidence=0, gate_trust='all',
                 dod_meta=0, num_lon=9, num_lat=7, uniform_mask=0):
        super().__init__()
        self.dod_meta = bool(dod_meta)
        self.disentangler = EgoCausalDisentangler(dim, heads, layers, dropout, nbr_enrich=nbr_enrich,
                                                  gate=gate, conflict_feats=conflict_feats,
                                                  conflict_bias=conflict_bias,
                                                  gate_channels=gate_channels, typed_kv=typed_kv,
                                                  channel_evidence=channel_evidence,
                                                  gate_trust=gate_trust)
        for _l in self.disentangler.layers:
            _l.ego_residual = bool(ego_residual)
            _l.uniform_mask = bool(uniform_mask)      # rules-only baseline (inference-time)
        # compute_conflict: feature'lar edge'e girmese/bias olmasa bile hesaplansin (loss icin)
        self.disentangler.compute_conflict = bool(compute_conflict or conflict_feats or conflict_bias)
        self.disentangler.aligned_mode = aligned_mode
        self.head = CausalEgoHead(dim, modes, dropout, num_maneuvers=num_maneuvers,
                                  dod_meta=dod_meta, num_lon=num_lon, num_lat=num_lat)
        # PAYLASILAN karar head'i (Causal-Planner gibi: decisioon_decoder causal VE confound icin AYNI).
        # Ayri head'ler kullanirsak confound head'i "girdiyi yok say -> uniform" hilesiyle entropy'yi
        # trivial cozuyor (gradyan 0 -> f_cfd sekillenmez). Paylasinca head causal'i dogru tahmin etmek
        # ZORUNDA -> hile yapamaz -> uniform baskisi f_cfd FEATURE'larina akar -> entropy CANLI kalir.
        # Cikti = 5-sinif MANEVRA (CP get_decision.py) VEYA dod_meta=1'de factored (lon x lat)
        # meta-aksiyon ciftleri (decision_labels.py) -- her iki durumda SABIT GT-turevi hedef.
        if self.dod_meta:
            self.psi_lon = nn.Sequential(nn.Linear(dim, 128), nn.ReLU(), nn.Linear(128, num_lon))
            self.psi_lat = nn.Sequential(nn.Linear(dim, 128), nn.ReLU(), nn.Linear(128, num_lat))
        else:
            self.psi = nn.Sequential(nn.Linear(dim, 128), nn.ReLU(), nn.Linear(128, num_maneuvers))

        # STEP 2 -- f_cfd COLLAPSE FIX. Olculen kusur: fcfd_var/fcas_var ~ 1e-4 (3 kosuda), yani
        # f_cfd her sahnede AYNI vektor. Sebep: loss ondan tek sey istiyor ("manevrayi ele verme"),
        # sabit vektor bunu bedavaya sagliyor -- yani confound dalinin hicbir MUSTERISI yok.
        # Cozum: [f_cas ; f_cfd] birlikte gate'siz tam-sahne ozetini (f_all) geri kurabilsin.
        # PATHWAY DROPOUT sart: f_cas her zaman verilirse recon "f_cas'i kopyala"yi ogrenir ve
        # f_cfd'ye hic bakmaz (DOD'da ise yarayan ayni numara). p ile f_cas SIFIRLANIR -> o orneklerde
        # hedefi TEK BASINA f_cfd tasimak zorunda.
        self.recon_drop = recon_drop
        self.cfd_recon = nn.Sequential(nn.Linear(2 * dim, 2 * dim), nn.ReLU(), nn.Linear(2 * dim, 3 * dim))

        # STEP 2b (option B) -- CP'nin agent_predictor'unun CONFOUND dala uygulanmis hali.
        # CP (planning_model.py:519) komsu geleceklerini CAUSAL dugum ozelliklerinden tahmin eder
        # (others_reg_loss); confound dalinin ise TEK tuketicisi decision_decoder'dir -> orada da
        # cokme serbest. Biz ayni yardimci gorevi f_cfd'ye veriyoruz: f_cfd TEK BASINA tum komsularin
        # gelecegini tasimali. Recon'dan farki: hedef ajan-BASINA, yani uniform maske optimal DEGIL --
        # her ajanin gelecegi farkli miktarda bilgi ister, maske tahsisi ogrenmek ZORUNDA.
        # Agirliksiz biraktik bilerek: M_cfd ile agirliklandirsak model "kolay ajani sec"e kacardi.
        self.num_neighbors, self.future_steps = num_neighbors, future_steps
        self.nbr_head = nn.Sequential(nn.Linear(dim, 2 * dim), nn.ReLU(),
                                      nn.Linear(2 * dim, num_neighbors * future_steps * 2))

    def forward(self, encoder_outputs, inputs, num_agents,
                neighbor_futures=None, neighbor_states=None, also_cfd_plan=False, ref_path=None):
        Na = num_agents
        agent_feat = encoder_outputs['agent_tokens'][:, :Na].detach()   # [B,Na,D] temiz, fusion oncesi
        agent_valid = ~encoder_outputs['mask'][:, :Na]                  # [B,Na] True=gecerli
        agent_pose = encoder_outputs['actors'][:, :Na, -1].detach()     # [B,Na,5]
        agent_types = _agent_types(inputs['neighbor_agents_past'], Na - 1)

        dis = self.disentangler(agent_feat, agent_valid, agent_pose, agent_types, inputs,
                                neighbor_futures=neighbor_futures, neighbor_states=neighbor_states,
                                ref_path=ref_path)
        f_cas, f_cfd = dis['f_cas'], dis['f_cfd']

        # STEP 2: [f_cas ; f_cfd] -> f_all rekonstruksiyonu. f_cas per-ornek olasilikla SIFIRLANIR
        # (yalniz egitimde) -> o orneklerde hedefi tek basina f_cfd tasimali. Inference'ta kapali;
        # cfd_recon plan yoluna hic dokunmaz, sadece f_cfd'ye gradyan saglar.
        keep = 1.0
        if self.training and self.recon_drop > 0.0:
            keep = (torch.rand(f_cas.shape[0], 1, device=f_cas.device) >= self.recon_drop).float()
        recon_pred = self.cfd_recon(torch.cat([keep * f_cas, f_cfd], dim=-1))   # [B,3D]

        # STEP 2b: komsu gelecekleri YALNIZ f_cfd'den (f_cas'a bakmadan) -> f_cfd sahne icerigi tasimali.
        nbr_pred = self.nbr_head(f_cfd).view(-1, self.num_neighbors, self.future_steps, 2)

        # DOD: psi'yi head'den ONCE hesapla; tahmin edilen karar (b*) decoder query'sine besle.
        psi_cas = psi_cfd = None
        psi_lon_cas = psi_lat_cas = psi_lon_cfd = psi_lat_cfd = None
        if self.dod_meta:
            psi_lon_cas, psi_lat_cas = self.psi_lon(f_cas), self.psi_lat(f_cas)
            psi_lon_cfd, psi_lat_cfd = self.psi_lon(f_cfd), self.psi_lat(f_cfd)
            b_star = (psi_lon_cas.argmax(-1), psi_lat_cas.argmax(-1))   # (argmax non-diff -> psi'ye sizmaz)
            b_cfd = (psi_lon_cfd.argmax(-1), psi_lat_cfd.argmax(-1))
        else:
            psi_cas = self.psi(f_cas)
            psi_cfd = self.psi(f_cfd)
            b_star = psi_cas.argmax(-1)                                  # [B] (argmax non-diff -> psi'ye sizmaz)
            b_cfd = psi_cfd.argmax(-1)

        # ANA plan: head'e TEMIZ ego (ego_clean) + SADECE f_cas + karar-embedding girer. f_cas artik
        # gate'li ajan+harita tasir -> trajectory yalnizca causal ajanlar+harita+ego+karara bagli.
        traj, score = self.head(f_cas, dis['ego_clean'], b_star)        # [B,1,M,80,4], [B,1,M]

        # Tanı/ablasyon amaçlı: f_cfd'den de bir plan üret (aynı EĞİTİLMİŞ head ile). Varsayılan KAPALI.
        traj_cfd = score_cfd = None
        if also_cfd_plan:
            traj_cfd, score_cfd = self.head(f_cfd, dis['ego_clean'], b_cfd)

        out = {
            'traj': traj, 'score': score,
            'traj_cfd': traj_cfd, 'score_cfd': score_cfd,
            'M_cas': dis['M_cas'], 'M_cfd': dis['M_cfd'],                # ajan causal/confound [B,N]
            'M_cas_map': dis['M_cas_map'], 'M_cfd_map': dis['M_cfd_map'],  # harita causal/confound [B,S]
            'map_valid': dis['map_valid'],
            # elestiri #3: M_cas/M_cfd head-ortalamasi vs head-basina entropi (bkz EgoCausalDisentangler)
            'M_cas_ent': dis['M_cas_ent'], 'M_cas_headent': dis['M_cas_headent'],
            'M_cfd_ent': dis['M_cfd_ent'], 'M_cfd_headent': dis['M_cfd_headent'],
            'M_cas_map_ent': dis['M_cas_map_ent'], 'M_cas_map_headent': dis['M_cas_map_headent'],
            'M_cfd_map_ent': dis['M_cfd_map_ent'], 'M_cfd_map_headent': dis['M_cfd_map_headent'],
            'f_cas': f_cas, 'f_cfd': f_cfd,
            'ego_clean': dis['ego_clean'],     # b*-swap/CF: head'i disaridan farkli b* ile yeniden cagirmak icin
            'nbr_valid': dis['nbr_valid'],
            'gate_cos': dis['gate_cos'],                   # katman-basina cos(f_cas, h_ego)
            'psi_cas': psi_cas,                    # PAYLASILAN head -> manevra (L_KLD, informative); dod_meta'da None
            'psi_cfd': psi_cfd,                    # AYNI head -> uniform (L_ENT); hile yapamaz -> entropy canli
            # dod_meta (H): factored karar logitleri (dod_meta=0'da None)
            'psi_lon_cas': psi_lon_cas, 'psi_lat_cas': psi_lat_cas,
            'psi_lon_cfd': psi_lon_cfd, 'psi_lat_cfd': psi_lat_cfd,
            'recon_pred': recon_pred,              # STEP 2: [f_cas(dropout'lu); f_cfd] -> f_all tahmini
            'f_all': dis['f_all'],                 # STEP 2: hedef (loss'ta DETACH edilir)
            'nbr_pred': nbr_pred,                  # STEP 2b: f_cfd -> komsu gelecekleri [B,N,80,2]
            'conflict': dis['conflict'],           # [B,N,4] future-conflict (L_conflict icin)
            # kanal ciktilari (channels branch)
            'M_cas_typed': dis['M_cas_typed'], 'M_cfd_typed': dis['M_cfd_typed'],
            'M_cas_map_typed': dis['M_cas_map_typed'], 'M_cfd_map_typed': dis['M_cfd_map_typed'],
            'gated_valid': dis['gated_valid'], 'gated_map_valid': dis['gated_map_valid'],
            'ch_active': dis['ch_active'], 'mch_active': dis['mch_active'],
        }
        return out
