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

import math

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

    def __init__(self, dim=256, heads=8, edge_dim=EDGE_FEATURE_DIM, dropout=0.1, sepkey=False):
        super().__init__()
        assert dim % heads == 0
        self.dim, self.heads, self.dh = dim, heads, dim // heads
        self.sepkey = sepkey   # E-a: komsu QUERY + FFN per-komsu-tipi (CP AgentHetGNN wqs[t]/out_ffn[t])
        if sepkey:
            self.Wq = nn.ModuleList([nn.Linear(dim, dim) for _ in range(NUM_AGENT_TYPES)])  # per-tip query
            self.ffn = nn.ModuleList([_FFN(dim, dropout=dropout) for _ in range(NUM_AGENT_TYPES)])
        else:
            self.Wq = nn.Linear(dim, dim)      # komsu query (paylasimli)
            self.ffn = _FFN(dim, dropout=dropout)
        self.Wk = nn.Linear(dim, dim)      # harita key (paylasimli -- harita tipsiz)
        self.Wv = nn.Linear(dim, dim)      # harita value
        self.We_k = _EdgeMLP(edge_dim, dim, dropout=dropout)
        self.We_v = _EdgeMLP(edge_dim, dim, dropout=dropout)
        self.attn = nn.Parameter(torch.empty(heads, self.dh)); nn.init.xavier_uniform_(self.attn)
        self.norm = nn.LayerNorm(dim)
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, h_nbr, h_map, edge_nbr_map, map_valid, nbr_types=None):
        """h_nbr [B,N,D] (her komsu bir query, HEPSI guncellenir); h_map [B,S,D] (STATIK key/value);
        edge_nbr_map [B,N,S,De] (harita(kaynak)->komsu(hedef) goreli geometri); map_valid [B,S];
        nbr_types [B,N] (sepkey acikken per-tip query/ffn icin)."""
        B, N = h_nbr.shape[0], h_nbr.shape[1]
        S = h_map.shape[1]
        H, dh = self.heads, self.dh
        q_in = EgoCausalLayer._per_type(self.Wq, h_nbr, nbr_types) if self.sepkey else self.Wq(h_nbr)
        q = q_in.view(B, N, 1, H, dh)
        k = self.Wk(h_map).view(B, 1, S, H, dh)
        ek = self.We_k(edge_nbr_map).view(B, N, S, H, dh)
        v = self.Wv(h_map).view(B, 1, S, H, dh) + self.We_v(edge_nbr_map).view(B, N, S, H, dh)
        s = self.leaky(q + k + ek)                                      # [B,N,S,H,dh]
        a = (s * self.attn).sum(-1)                                     # [B,N,S,H]
        invalid = ~map_valid[:, None, :, None]                          # [B,1,S,1]
        M = torch.softmax(a.masked_fill(invalid, torch.finfo(a.dtype).min), dim=2).masked_fill(invalid, 0.0)
        ctx = (M.unsqueeze(-1) * v).sum(dim=2).reshape(B, N, H * dh)     # [B,N,D] her komsuya harita baglami
        out = self.norm(h_nbr + ctx)
        return EgoCausalLayer._per_type(self.ffn, out, nbr_types) if self.sepkey else self.ffn(out)


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

    def __init__(self, dim=256, heads=8, edge_dim=EDGE_FEATURE_DIM, dropout=0.1, sep_temp=False):
        super().__init__()
        assert dim % heads == 0
        self.dim, self.heads, self.dh = dim, heads, dim // heads
        self.sep_temp = sep_temp   # #3: sayi-adaptif temperature (CP log_32(n+1)); varsayilan KAPALI
        # 'other' (ajan) iliskisi. Query HEP ego (tek); VALUE komsu-tipine gore AYRI (cas/cfd ORTAK value).
        # #1: cas ve cfd icin AYRI KEY (CP wks["other_causal/confound"][t]) -> ayrimi KEY yapar, TEK attn.
        self.Wq_ag = nn.Linear(dim, dim)
        self.Wk_cas_ag = nn.ModuleList([nn.Linear(dim, dim) for _ in range(NUM_AGENT_TYPES)])   # causal key (per-tip)
        self.Wk_cfd_ag = nn.ModuleList([nn.Linear(dim, dim) for _ in range(NUM_AGENT_TYPES)])   # confound key (per-tip)
        self.Wv_ag = nn.ModuleList([nn.Linear(dim, dim) for _ in range(NUM_AGENT_TYPES)])       # value ORTAK
        self.We_k_ag = _EdgeMLP(edge_dim, dim, dropout=dropout); self.We_v_ag = _EdgeMLP(edge_dim, dim, dropout=dropout)
        self.attn_ag = nn.Parameter(torch.empty(heads, self.dh))      # TEK scoring vektoru (ayrimi key yapar)
        # 'g2a' (harita) iliskisi. Ayri cas/cfd key (paylasimli, harita tipsiz), tek value, tek attn.
        self.Wq_mp = nn.Linear(dim, dim); self.Wv_mp = nn.Linear(dim, dim)
        self.Wk_cas_mp = nn.Linear(dim, dim); self.Wk_cfd_mp = nn.Linear(dim, dim)
        self.We_k_mp = _EdgeMLP(edge_dim, dim, dropout=dropout); self.We_v_mp = _EdgeMLP(edge_dim, dim, dropout=dropout)
        self.attn_mp = nn.Parameter(torch.empty(heads, self.dh))
        for p in (self.attn_ag, self.attn_mp):
            nn.init.xavier_uniform_(p)
        # birlestirme (Eq 6): [self ; ajan ; harita] -> f_cas / f_cfd
        self.self_fc = nn.Sequential(nn.Linear(dim, dim), nn.ReLU())
        # AYRI cikis projeksiyonu + LayerNorm (CP hdgt_encoder.py ile ayni: out_fc_causal/out_fc_confound
        # ve out_ffn_causal/out_ffn_confound ayridir; residual SADECE causal dalda).
        self.out_fc_cas = nn.Linear(3 * dim, dim)
        self.out_fc_cfd = nn.Linear(3 * dim, dim)
        self.norm_cas = nn.LayerNorm(dim)
        self.norm_cfd = nn.LayerNorm(dim)
        # FAZ 1: FFN eklendi (CP'de var, bizde yoktu -- attention SADECE token'lar arasi dogrusal
        # karisim yapar, FFN her token'i KENDI icinde dogrusal-olmayan isler). Kendi prenorm+residual'i var.
        self.ffn_cas = _FFN(dim, dropout=dropout)
        self.ffn_cfd = _FFN(dim, dropout=dropout)
        self.leaky = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def _attend(self, q1, k_cas, k_cfd, ek, msg, valid, attn_cas, attn_cfd):
        """q1 [B,1,H,dh]; k_cas/k_cfd/ek/msg [B,Nk,H,dh]; valid [B,Nk] bool. AYRI softmax causal/confound.
        #1: cas/cfd AYRI KEY (CP wks["other_causal/confound"]) -> ayrimi KEY yapar; attn_cas==attn_cfd (tek
        scoring). Doner: cas/cfd [B,D], M_cas/M_cfd [B,Nk] (head-ort), ent'ler [B]."""
        B = k_cas.shape[0]
        s_cas = self.leaky(q1 + k_cas + ek)                  # [B,Nk,H,dh]
        s_cfd = self.leaky(q1 + k_cfd + ek)
        a_cas = (s_cas * attn_cas).sum(-1)                   # [B,Nk,H]
        a_cfd = (s_cfd * attn_cfd).sum(-1)
        # #3 (varsayilan KAPALI, --sep_temp): sayi-adaptif temperature (CP: attn *= log_32(n+1)); n = gecerli
        # anahtar sayisi. Kalabalik (n>31) keskinlestir, seyrek (n<31) yumusat. Bizde ~10 ajanda YUMUSATIYOR
        # (fayda vermedi, o yuzden default kapali).
        if self.sep_temp:
            n_valid = valid.sum(-1).clamp(min=1).float()[:, None, None]    # [B,1,1]
            temp = torch.log1p(n_valid) / math.log(32.0)
            a_cas = a_cas * temp; a_cfd = a_cfd * temp
        invalid = ~valid[:, :, None]                          # [B,Nk,1]
        neg_inf = torch.finfo(a_cas.dtype).min
        M_cas_h = torch.softmax(a_cas.masked_fill(invalid, neg_inf), dim=1).masked_fill(invalid, 0.0)
        M_cfd_h = torch.softmax(a_cfd.masked_fill(invalid, neg_inf), dim=1).masked_fill(invalid, 0.0)
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
        ent_cas_mean = -(M_cas_mean.clamp(min=eps).log() * M_cas_mean).sum(-1)      # [B], NATS (ham)
        ent_cas_headmean = (-(M_cas_h.clamp(min=eps).log() * M_cas_h).sum(1)).mean(-1)   # [B]
        ent_cfd_mean = -(M_cfd_mean.clamp(min=eps).log() * M_cfd_mean).sum(-1)
        ent_cfd_headmean = (-(M_cfd_h.clamp(min=eps).log() * M_cfd_h).sum(1)).mean(-1)

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

    @staticmethod
    def _per_type(mods, x, types):
        """x [B,N,D], types [B,N] long, mods = T tiplik ModuleList -> her token'a KENDI tipinin
        Linear'ini uygula (CP wks/wvs["other"][t] mantigi). Tumu hesaplanip tip'e gore toplanir."""
        stacked = torch.stack([m(x) for m in mods], dim=2)                  # [B,N,T,D]
        idx = types.clamp(min=0, max=len(mods) - 1)[:, :, None, None].expand(-1, -1, 1, x.shape[-1])
        return stacked.gather(2, idx).squeeze(2)                            # [B,N,D]

    def forward(self, h_ego, h_nbr, nbr_types, edge_ego, nbr_valid, h_map, edge_map, map_valid):
        """h_ego [B,D]; h_nbr [B,N,D] (K=V ayni kaynak), nbr_types [B,N] long (komsu tipi -> tip-basina
        K/V), edge_ego [B,N,De], nbr_valid [B,N]; h_map [B,S,D], edge_map [B,S,De] (polygon->ego), map_valid [B,S].
        Doner: h_ego_new, f_cas, f_cfd, M_cas(ajan), M_cfd(ajan), M_cas_mp(harita), M_cfd_mp(harita)."""
        B, N = h_nbr.shape[0], h_nbr.shape[1]
        S = h_map.shape[1]
        H, dh = self.heads, self.dh

        # --- ajan iliskisi (other): AYRI cas/cfd KEY (per-tip), ORTAK value, TEK attn (#1, CP-tarzi) ---
        q_ag = self.Wq_ag(h_ego).view(B, 1, H, dh)
        kcas_ag = self._per_type(self.Wk_cas_ag, h_nbr, nbr_types).view(B, N, H, dh)
        kcfd_ag = self._per_type(self.Wk_cfd_ag, h_nbr, nbr_types).view(B, N, H, dh)
        ek_ag = self.We_k_ag(edge_ego).view(B, N, H, dh)
        msg_ag = self._per_type(self.Wv_ag, h_nbr, nbr_types).view(B, N, H, dh) + self.We_v_ag(edge_ego).view(B, N, H, dh)
        (ag_cas, ag_cfd, M_cas_ag, M_cfd_ag,
         ent_cas_mean, ent_cas_headmean, ent_cfd_mean, ent_cfd_headmean) = self._attend(
            q_ag, kcas_ag, kcfd_ag, ek_ag, msg_ag, nbr_valid, self.attn_ag, self.attn_ag)

        # --- harita iliskisi (g2a): AYRI cas/cfd KEY (paylasimli), ORTAK value, TEK attn ---
        q_mp = self.Wq_mp(h_ego).view(B, 1, H, dh)
        kcas_mp = self.Wk_cas_mp(h_map).view(B, S, H, dh)
        kcfd_mp = self.Wk_cfd_mp(h_map).view(B, S, H, dh)
        ek_mp = self.We_k_mp(edge_map).view(B, S, H, dh)
        msg_mp = self.Wv_mp(h_map).view(B, S, H, dh) + self.We_v_mp(edge_map).view(B, S, H, dh)
        (mp_cas, mp_cfd, M_cas_mp, M_cfd_mp,
         ent_cas_mp_mean, ent_cas_mp_headmean, ent_cfd_mp_mean, ent_cfd_mp_headmean) = self._attend(
            q_mp, kcas_mp, kcfd_mp, ek_mp, msg_mp, map_valid, self.attn_mp, self.attn_mp)

        # --- birlestirme (Eq 6): [self ; ajan ; harita] ---
        self_fea = self.self_fc(h_ego)                                        # [B,D]
        f_cas = self.norm_cas(self.out_fc_cas(torch.cat([self_fea, ag_cas, mp_cas], dim=-1)) + h_ego)
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
                gate_cos)


class EgoCausalDisentangler(nn.Module):
    """(A) Ego + komsu ajan dugumlerinden causal/confounding ayrisimini ureten modul.

    Dugum ozellikleri = fusion ONCESI agent_tokens (detached). Opsiyonel: komsu tahmini
    gelecekleri node'lara kaynastir (gelecek-bilincli). L katman; her katman ego node'unu
    gunceller, komsular sabit kalir. Son katmanin f_cas/f_cfd + M_cas/M_cfd ciktisi kullanilir.
    """

    def __init__(self, dim=256, heads=8, layers=3, dropout=0.1, nbr_enrich=0, sep_temp=0, nbr_sepkey=0):
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
        # KOMSU->HARITA enrichment (CP scene-embedding a2g, kimlik-koruyan): causal split'ten ONCE her
        # komsuya serit/yol baglami kazandirir. nbr_enrich=0 => mevcut davranis (enrichment YOK).
        self.nbr_enrich = nn.ModuleList([
            _NbrMapEnrichLayer(dim, heads, EDGE_FEATURE_DIM, dropout, sepkey=bool(nbr_sepkey)) for _ in range(nbr_enrich)
        ])
        self.layers = nn.ModuleList([
            EgoCausalLayer(dim, heads, EDGE_FEATURE_DIM, dropout, sep_temp=bool(sep_temp)) for _ in range(layers)
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

        ego_clean = h[:, 0]           # head'e verilen temiz ego (SADECE ego'nun kendi gecmisi + tip)
        nbr_valid = agent_valid[:, 1:]                                                 # [B,N]
        nbr_types = agent_types[:, 1:]                                                 # [B,N] tip-basina K/V

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
                h_nbr = enrich(h_nbr, h_map, edge_nbr_map, map_valid, nbr_types=nbr_types)

        # --- STAGE B: ego-merkezli causal-split (ajan + harita, causal/confound) ---
        f_cas = f_cfd = M_cas = M_cfd = M_cas_mp = M_cfd_mp = None
        ent_cas_mean = ent_cas_headmean = ent_cfd_mean = ent_cfd_headmean = None
        ent_cas_mp_mean = ent_cas_mp_headmean = ent_cfd_mp_mean = ent_cfd_mp_headmean = None
        gate_cos_layers = []          # elestiri: katman-basina cos(f_cas, h_ego), bypass/GRU-fix etkilesimi
        for layer in self.layers:
            (h_ego, f_cas, f_cfd, M_cas, M_cfd, M_cas_mp, M_cfd_mp,
             ent_cas_mean, ent_cas_headmean, ent_cfd_mean, ent_cfd_headmean,
             ent_cas_mp_mean, ent_cas_mp_headmean, ent_cfd_mp_mean, ent_cfd_mp_headmean,
             gate_cos) = layer(
                h_ego, h_nbr, nbr_types, edge_ego, nbr_valid, h_map, edge_map, map_valid)
            gate_cos_layers.append(gate_cos)
        gate_cos_stack = torch.stack(gate_cos_layers, dim=1)   # [B, L] -- L=len(self.layers)

        return {
            'f_cas': f_cas, 'f_cfd': f_cfd, 'M_cas': M_cas, 'M_cfd': M_cfd,
            'M_cas_map': M_cas_mp, 'M_cfd_map': M_cfd_mp, 'map_valid': map_valid,
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
            # katman-basina cos(f_cas, h_ego) [B,L] -- ~1'e yakinsa gate marjinal (h_ego bypass'i baskin).
            'gate_cos': gate_cos_stack,
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

    def __init__(self, dim=256, heads=8, layers=3, modes=6, dropout=0.1, nbr_enrich=0, sep_temp=0, nbr_sepkey=0):
        super().__init__()
        self.disentangler = EgoCausalDisentangler(dim, heads, layers, dropout, nbr_enrich=nbr_enrich,
                                                  sep_temp=sep_temp, nbr_sepkey=nbr_sepkey)
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
            # elestiri #3: M_cas/M_cfd head-ortalamasi vs head-basina entropi (bkz EgoCausalDisentangler)
            'M_cas_ent': dis['M_cas_ent'], 'M_cas_headent': dis['M_cas_headent'],
            'M_cfd_ent': dis['M_cfd_ent'], 'M_cfd_headent': dis['M_cfd_headent'],
            'M_cas_map_ent': dis['M_cas_map_ent'], 'M_cas_map_headent': dis['M_cas_map_headent'],
            'M_cfd_map_ent': dis['M_cfd_map_ent'], 'M_cfd_map_headent': dis['M_cfd_map_headent'],
            'f_cas': f_cas, 'f_cfd': f_cfd,
            'nbr_valid': dis['nbr_valid'],
            'gate_cos': dis['gate_cos'],                   # katman-basina cos(f_cas, h_ego)
            'psi_cas': self.psi(f_cas),            # PAYLASILAN head -> m* (L_KLD, informative)
            'psi_cfd': self.psi(f_cfd),            # AYNI head -> uniform (L_ENT); hile yapamaz -> entropy canli
        }
        return out
