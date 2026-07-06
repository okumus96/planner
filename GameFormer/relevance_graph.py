"""
SceneRelevanceGraph: Bir surus sahnesinde hangi ajanlarin ve harita elemanlarinin
ego'nun karar verme sureci icin onemli oldugunu ogrenen heterojen, kenar-oznitelikli
(edge-featured) GATv2 modulu.

Tasarim (plana gore):
  - Dugumler: ego + komsu ajanlar + per-element lane/crosswalk/route (havuzlanmamis).
  - Heterojenlik: dugum tipi gommesi (ego/vehicle/pedestrian/bicycle/lane/crosswalk/route)
    her dugume eklenir; kenarlarda relatif-poz oznitelikleri tasinir (HEAT cizgisi).
  - GATv2 (dinamik attention): skor a^T LeakyReLU(Wq h_i + Wk h_j + We e_ij) seklinde,
    nonlinearite 'a' ile carpimdan ONCE (GAT'in statik attention'indan farki budur).
  - Graf yapisi: en az bir ucu ajan olan kenarlar (ajan<->ajan + ajan<->harita bipartite);
    harita<->harita kenarlari maskelenir.
  - Cikti: rafine dugum gommeleri H [B,V,D] + ego-merkezli onem dagilimi alpha [B,V]
    ("Importance is in your attention" fikri: ego query olarak tum dugumlere attend eder).

Bu modul frozen GameFormer backbone'una DOKUNMAZ; ajan gommelerini encoder ciktisindan
(detached) alir, harita per-element gommelerini kendi kucuk encoder'i ile uretir.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .predictor_modules import FutureEncoder


# Dugum tipi indeksleri (heterojen gomme icin)
NODE_TYPE_EGO = 0
NODE_TYPE_VEHICLE = 1
NODE_TYPE_PEDESTRIAN = 2
NODE_TYPE_BICYCLE = 3
NODE_TYPE_LANE = 4
NODE_TYPE_CROSSWALK = 5
NODE_TYPE_ROUTE = 6
NUM_NODE_TYPES = 7

EDGE_FEATURE_DIM = 7  # [dx, dy, log1p(dist), cos(dtheta), sin(dtheta), dvx, dvy]


class PolylineEncoder(nn.Module):
    """Ham polyline [B,E,P,F] -> eleman basina tek gomme [B,E,D] (PointNet + max over points).

    Mevcut VectorMapEncoder / ar_wrapper.lat_encoder deseniyle ayni mantik, ama her elemani
    TEK token'a indirger (per-element granularite -> 'hangi harita elemani onemli' diyebilmek
    icin gerekli). Frozen backbone'un segment-bazli havuzlamasinin aksine burada her lane/
    crosswalk/route tek dugumdur.
    """

    def __init__(self, in_dim, dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, dim),
        )

    def forward(self, x):
        # x: [B, E, P, F]
        h = self.mlp(x)            # [B, E, P, D]
        h = torch.max(h, dim=2)[0]  # [B, E, D]
        return h


def _polyline_pose_and_valid(poly):
    """Ham polyline [B,E,P,>=3] (x,y,heading) -> dugum pozu [B,E,5] (x,y,heading,0,0)
    ve gecerlilik [B,E] (en az bir sifir-olmayan nokta varsa gecerli).

    Pozisyon: gecerli noktalarin centroid'i. Heading: dairesel ortalama (cos/sin).
    Harita elemanlarinin hizi yoktur -> vx=vy=0.
    """
    B, E, P, _ = poly.shape
    pt_valid = (poly[..., :2].abs().sum(-1) > 1e-6)          # [B, E, P]
    elem_valid = pt_valid.any(dim=-1)                         # [B, E]

    cnt = pt_valid.sum(-1).clamp(min=1).unsqueeze(-1).float()  # [B, E, 1]
    m = pt_valid.unsqueeze(-1).float()                         # [B, E, P, 1]
    centroid = (poly[..., :2] * m).sum(2) / cnt                # [B, E, 2]

    heading = poly[..., 2]
    cos_m = (torch.cos(heading) * pt_valid.float()).sum(-1) / cnt.squeeze(-1)
    sin_m = (torch.sin(heading) * pt_valid.float()).sum(-1) / cnt.squeeze(-1)
    head = torch.atan2(sin_m, cos_m)                           # [B, E]

    zeros = torch.zeros(B, E, 2, device=poly.device, dtype=poly.dtype)
    pose = torch.cat([centroid, head.unsqueeze(-1), zeros], dim=-1)  # [B, E, 5]
    return pose, elem_valid


def build_edge_features(node_pose):
    """node_pose [B,V,5] (x,y,heading,vx,vy) -> kenar oznitelikleri [B,V,V,EDGE_FEATURE_DIM].

    e[:, i, j] = j (kaynak) ile i (hedef) arasindaki relatif geometri (ego-frame).
    """
    p = node_pose
    xi = p[:, :, None, :]   # hedef i  [B,V,1,5]
    xj = p[:, None, :, :]   # kaynak j [B,1,V,5]
    dxy = xj[..., :2] - xi[..., :2]                       # [B,V,V,2]
    dist = torch.norm(dxy, dim=-1)                        # [B,V,V]
    dtheta = xj[..., 2] - xi[..., 2]                      # [B,V,V]
    dvxy = xj[..., 3:5] - xi[..., 3:5]                    # [B,V,V,2]
    edge = torch.stack([
        dxy[..., 0], dxy[..., 1], torch.log1p(dist),
        torch.cos(dtheta), torch.sin(dtheta),
        dvxy[..., 0], dvxy[..., 1],
    ], dim=-1)                                            # [B,V,V,7]
    return edge


class HeteroEdgeGATv2Layer(nn.Module):
    """Cok-kafali, kenar-oznitelikli GATv2 katmani.

    Skor (GATv2 / dinamik attention):
        e_ij = a^T LeakyReLU(Wq h_i + Wk h_j + We_k edge_ij)
    Agregasyon (HEAT tarzi kenar-iyilestirme):
        out_i = sum_j softmax_j(e_ij) * (Wv h_j + We_v edge_ij)
    """

    def __init__(self, dim=256, heads=8, edge_dim=EDGE_FEATURE_DIM, dropout=0.1):
        super().__init__()
        assert dim % heads == 0
        self.dim = dim
        self.heads = heads
        self.dh = dim // heads

        self.Wq = nn.Linear(dim, dim)
        self.Wk = nn.Linear(dim, dim)
        self.Wv = nn.Linear(dim, dim)
        self.We_k = nn.Linear(edge_dim, dim)
        self.We_v = nn.Linear(edge_dim, dim)
        self.attn_vec = nn.Parameter(torch.empty(heads, self.dh))
        nn.init.xavier_uniform_(self.attn_vec)

        self.out_proj = nn.Linear(dim, dim)
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 4, dim), nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, h, edge, edge_allow):
        """
        h:          [B, V, D]    dugum gommeleri
        edge:       [B, V, V, De] kenar oznitelikleri (i=hedef, j=kaynak)
        edge_allow: [B, V, V]    bool, j->i kenarinin gecerli/izinli olup olmadigi
        """
        B, V, _ = h.shape
        H, dh = self.heads, self.dh

        q = self.Wq(h).view(B, V, H, dh)            # [B,V,H,dh]
        k = self.Wk(h).view(B, V, H, dh)
        v = self.Wv(h).view(B, V, H, dh)
        ek = self.We_k(edge).view(B, V, V, H, dh)   # [B,V,V,H,dh]
        ev = self.We_v(edge).view(B, V, V, H, dh)

        # GATv2: nonlinearite, 'a' ile carpimdan ONCE
        s = self.leaky_relu(q[:, :, None] + k[:, None, :] + ek)   # [B,V,V,H,dh]
        scores = (s * self.attn_vec).sum(-1)                       # [B,V,V,H]

        # Izinsiz / gecersiz kenarlari maskele
        neg_inf = torch.finfo(scores.dtype).min
        mask = edge_allow.unsqueeze(-1)                            # [B,V,V,1]
        scores = scores.masked_fill(~mask, neg_inf)

        attn_full = torch.softmax(scores, dim=2)                   # [B,V,V,H], j (dim=2) uzerinde
        attn = self.dropout(attn_full)

        msg = v[:, None, :] + ev                                   # [B,V,V,H,dh]
        out = (attn.unsqueeze(-1) * msg).sum(dim=2)                # [B,V,H,dh]
        out = out.reshape(B, V, self.dim)
        out = self.out_proj(out)

        h = self.norm_1(h + self.dropout(out))
        h = self.norm_2(h + self.ffn(h))
        # attn_weights[b, i, j] = i (hedef) dugumunun j (kaynak) dugumune verdigi attention
        # (kafalar uzerinde ortalama) -> kim kiminle ne kadar guclu bagli (gorsel icin)
        return h, attn_full.mean(dim=-1)                           # h, [B,V,V]


class SceneRelevanceGraph(nn.Module):
    """Sahne onem grafigi. Ajan + per-element harita dugumlerinden olusan heterojen graf
    uzerinde GATv2 calistirir; rafine gommeler ve ego-merkezli onem dagilimi uretir.
    """

    def __init__(self, dim=256, heads=8, layers=2, dropout=0.1):
        super().__init__()
        self.dim = dim

        # Per-element harita kodlayicilar (lane/route: 3 ozn, crosswalk: 3 ozn)
        self.lane_encoder = PolylineEncoder(3, dim)
        self.crosswalk_encoder = PolylineEncoder(3, dim)
        self.route_encoder = PolylineEncoder(3, dim)

        # Ajan gommeleri encoder'dan 256-dim gelir; tip gommesi ile heterojenlik
        self.type_embedding = nn.Embedding(NUM_NODE_TYPES, dim)
        self.input_norm = nn.LayerNorm(dim)

        # Faz 2 (prediction-aware): komsu tahmini gelecek yorungesini node ozelligine kaynastir.
        # FutureEncoder: trajs[B,N,M,T,2] + current_states[B,N,5] -> [B,N,M,D] (M=1 ile cagrilir).
        self.future_encoder = FutureEncoder()
        self.future_fuse = nn.Sequential(nn.Linear(2 * dim, dim), nn.ReLU())

        self.layers = nn.ModuleList([
            HeteroEdgeGATv2Layer(dim, heads, EDGE_FEATURE_DIM, dropout) for _ in range(layers)
        ])

        # NOT: eski imp_q/imp_k readout'u kaldirildi -> importance artik SON katmanin ego
        # attention satiri (forward'da hesaplaniyor; bias dustugu icin ayri readout egitilemezdi).

    def _agent_types(self, neighbor_agents_past, num_neighbors):
        """neighbor_agents_past [B, N, T, F] son-zaman tip one-hot kanallarindan ajan tipi.
        Indeks [8,9,10] = (vehicle, pedestrian, bicycle). Ego ayrica eklenir.
        """
        B = neighbor_agents_past.shape[0]
        device = neighbor_agents_past.device
        nbr = neighbor_agents_past[:, :num_neighbors, -1]        # [B, N, F]
        if nbr.shape[-1] >= 11:
            onehot = nbr[..., 8:11]                              # [B, N, 3]
            cls = onehot.argmax(-1)                              # 0=veh,1=ped,2=bic
        else:
            cls = torch.zeros(B, num_neighbors, dtype=torch.long, device=device)
        nbr_type = cls + NODE_TYPE_VEHICLE                       # 1..3
        ego_type = torch.full((B, 1), NODE_TYPE_EGO, dtype=torch.long, device=device)
        return torch.cat([ego_type, nbr_type], dim=1)            # [B, 1+N]

    def forward(self, encoder_outputs, inputs, num_agents, return_attention=False,
                neighbor_futures=None, neighbor_states=None):
        """
        encoder_outputs: GameFormer encoder ciktisi (dict) -> 'encoding','mask','actors'
        inputs:          ham ozellikler (dict) -> 'map_lanes','map_crosswalks','route_lanes',
                         'neighbor_agents_past'
        num_agents:      1 + num_neighbors (ego + komsular)

        Doner: dict {
          'context':    H [B, V, D]      rafine dugum gommeleri (mode_selector context'i),
          'valid':      [B, V] bool      gecerli dugum maskesi,
          'importance': alpha [B, V]     ego-merkezli onem dagilimi (toplam 1),
          'node_types': [B, V] long,
          'slices':     dict             dugum tipi -> (start, end) indeks dilimleri,
        }
        """
        device = (encoder_outputs['agent_tokens'] if 'agent_tokens' in encoder_outputs
                  else encoder_outputs['encoding']).device
        Na = num_agents

        # --- Ajan dugumleri (frozen encoder'dan, detached) ---
        # ONEMLI: FUSION ONCESI temiz per-ajan gommeleri kullan. Encoder'in 6-kat full-attention
        # fusion'i ('encoding') tum ajan+harita token'larini birbirine karistirir; o ciktiyi node
        # yapmak, grafin ogrenecegi interaction'i zaten node ozelliginin icine gomer -> daireseldir
        # ve attention yorumlanamaz olur. 'agent_tokens' = ego_encoder(ego) + agent_encoder(komsu_i),
        # fusion'dan ONCE -> her node yalnizca KENDI gecmisini tasir, interaction'i graf kurar.
        if 'agent_tokens' in encoder_outputs:
            agent_feat = encoder_outputs['agent_tokens'][:, :Na].detach()  # [B, Na, D] (temiz, fusion oncesi)
        else:
            agent_feat = encoder_outputs['encoding'][:, :Na].detach()      # geri-uyum (fused, eski davranis)
        agent_valid = ~encoder_outputs['mask'][:, :Na]                     # [B, Na] True=gecerli
        agent_pose = encoder_outputs['actors'][:, :Na, -1].detach()        # [B, Na, 5]
        agent_types = self._agent_types(inputs['neighbor_agents_past'], Na - 1)  # [B, Na]
        B = agent_feat.shape[0]

        # --- Faz 2: komsu tahmini gelecekleri node ozelligine kaynastir (varsa) ---
        # Ego (idx 0) anlik kalir; niyeti route node'lari + anlik hizdan gelir ("ego'nun committed
        # future'i yok"). Komsular (idx 1..N) "nereye gidiyor" bilgisini node'a tasir -> importance
        # ve interaction artik gelecek-bilincli olur (sadece anlik geometri degil).
        if neighbor_futures is not None and neighbor_states is not None:
            N = Na - 1
            fut_in = neighbor_futures[:, :N].unsqueeze(2)                              # [B, N, 1, T, 2]
            fut_emb = self.future_encoder(fut_in, neighbor_states[:, :N]).squeeze(2)   # [B, N, D]
            nbr_fused = self.future_fuse(torch.cat([agent_feat[:, 1:1 + N], fut_emb], dim=-1))  # [B, N, D]
            agent_feat = torch.cat([agent_feat[:, :1], nbr_fused], dim=1)              # [B, Na, D]

        # --- Per-element harita dugumleri (kendi trainable encoder'imiz) ---
        lanes = inputs['map_lanes'][..., :3].float()
        cwalks = inputs['map_crosswalks'][..., :3].float()
        routes = inputs['route_lanes'][..., :3].float()

        lane_feat = self.lane_encoder(lanes)                               # [B, L, D]
        cw_feat = self.crosswalk_encoder(cwalks)                           # [B, C, D]
        rt_feat = self.route_encoder(routes)                               # [B, R, D]

        lane_pose, lane_valid = _polyline_pose_and_valid(lanes)
        cw_pose, cw_valid = _polyline_pose_and_valid(cwalks)
        rt_pose, rt_valid = _polyline_pose_and_valid(routes)

        L, C, R = lane_feat.shape[1], cw_feat.shape[1], rt_feat.shape[1]

        def _types(n, t):
            return torch.full((B, n), t, dtype=torch.long, device=device)

        # --- Tum dugumleri birlestir ---
        feat = torch.cat([agent_feat, lane_feat, cw_feat, rt_feat], dim=1)     # [B, V, D]
        pose = torch.cat([agent_pose, lane_pose, cw_pose, rt_pose], dim=1)     # [B, V, 5]
        valid = torch.cat([agent_valid, lane_valid, cw_valid, rt_valid], dim=1)  # [B, V]
        ntypes = torch.cat([
            agent_types,
            _types(L, NODE_TYPE_LANE), _types(C, NODE_TYPE_CROSSWALK), _types(R, NODE_TYPE_ROUTE),
        ], dim=1)                                                              # [B, V]
        V = feat.shape[1]

        slices = {
            'agent': (0, Na), 'lane': (Na, Na + L),
            'crosswalk': (Na + L, Na + L + C), 'route': (Na + L + C, V),
        }

        # --- Heterojenlik: tip gommesi + giris normu ---
        h = self.input_norm(feat + self.type_embedding(ntypes))

        # --- Kenar oznitelikleri ve izin maskesi ---
        edge = build_edge_features(pose)                                       # [B,V,V,De]
        is_agent = (ntypes < NODE_TYPE_LANE)                                   # [B,V]
        # Izin: kaynak j gecerli VE (i ajan VEYA j ajan) -> harita<->harita kenari yok
        touch_agent = is_agent[:, :, None] | is_agent[:, None, :]              # [B,V,V]
        edge_allow = valid[:, None, :] & touch_agent                          # [B,V,V]

        # --- GATv2 katmanlari ---
        attns = []
        for layer in self.layers:
            h, a = layer(h, edge, edge_allow)
            attns.append(a)

        # --- Importance = SON katmanin ego attention satiri (head-ortalamali) ---
        # imp_q/imp_k readout'u KALDIRILDI: bias dustugu icin egitilemezdi (random kalirdi).
        # Bunun yerine ego'nun (idx 0) SON-katman attention dagilimini kullaniyoruz: task
        # uzerinden egitiliyor, ego-merkezli, kaynaklar uzerinde softmax oldugu icin TOPLAM=1.
        # Head'ler ORTALANIYOR (layer ici attn_full.mean) -> toplam=1 korunur (max-pool bozardi).
        last_attn = attns[-1]                                                  # [B,V,V] son katman, head-avg
        importance = last_attn[:, 0, :]                                        # ego satiri -> [B,V], toplam=1

        out = {
            'context': h,
            'valid': valid,
            'importance': importance,
            'node_types': ntypes,
            'node_pose': pose,
            'slices': slices,
        }
        if return_attention:
            # SON katman yonlu attention [B,V,V] (head-avg); importance = bunun ego satiri.
            out['attention'] = last_attn.detach()
        return out


_TYPE_NAMES = {
    NODE_TYPE_EGO: 'ego', NODE_TYPE_VEHICLE: 'vehicle', NODE_TYPE_PEDESTRIAN: 'pedestrian',
    NODE_TYPE_BICYCLE: 'bicycle', NODE_TYPE_LANE: 'lane', NODE_TYPE_CROSSWALK: 'crosswalk',
    NODE_TYPE_ROUTE: 'route',
}


def plot_scene_importance(graph_out, raw_inputs=None, batch_idx=0, save_path=None,
                          ax=None, title=None, topk=None):
    """SceneRelevanceGraph onem dagilimini ego-frame'de gorsellestir.

    graph_out:  SceneRelevanceGraph.forward ciktisi (dict).
    raw_inputs: opsiyonel; verilirse harita polyline'lari da (acik gri) cizilir.
    topk:       verilirse sadece en onemli k dugum vurgulanir (digerleri soluk).

    Her dugum (ajan/harita elemani) onem skoruna gore renklendirilir; ego ayri isaretlenir.
    Koordinatlar ego-frame'dedir (eval_predictor_viz.draw_sample ile ayni eksen).
    """
    import numpy as np
    import matplotlib.pyplot as plt

    pose = graph_out['node_pose'][batch_idx].detach().cpu().numpy()     # [V, 5]
    imp = graph_out['importance'][batch_idx].detach().cpu().numpy()     # [V]
    valid = graph_out['valid'][batch_idx].detach().cpu().numpy()        # [V]
    ntypes = graph_out['node_types'][batch_idx].detach().cpu().numpy()  # [V]

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    # Harita polyline'larini acik gri ciz (varsa)
    if raw_inputs is not None:
        def _np(x):
            return x[batch_idx].detach().cpu().numpy() if torch.is_tensor(x) else x[batch_idx]
        for key, style in (('map_lanes', '-'), ('route_lanes', '-'), ('map_crosswalks', '--')):
            if key in raw_inputs:
                polys = _np(raw_inputs[key])
                for poly in polys:
                    m = np.abs(poly[:, :2]).sum(-1) > 1e-6
                    if m.any():
                        ax.plot(poly[m, 0], poly[m, 1], style, color='lightgray', linewidth=0.8, zorder=0)

    # topk maskesi
    highlight = np.ones_like(valid, dtype=bool)
    if topk is not None:
        order = np.argsort(-np.where(valid, imp, -np.inf))
        highlight = np.zeros_like(valid, dtype=bool)
        highlight[order[:topk]] = True

    vmax = max(imp[valid].max(), 1e-6) if valid.any() else 1.0
    sc = None
    for v in range(len(imp)):
        if not valid[v]:
            continue
        x, y = pose[v, 0], pose[v, 1]
        is_ego = (ntypes[v] == NODE_TYPE_EGO)
        is_agent = (ntypes[v] < NODE_TYPE_LANE)
        alpha = 1.0 if highlight[v] else 0.15
        if is_ego:
            ax.scatter(x, y, c='blue', s=200, marker='*', edgecolors='k',
                       zorder=6, label='ego')
            continue
        marker = 'o' if is_agent else 's'   # ajan=daire, harita=kare
        size = 60 + 600 * (imp[v] / vmax)
        sc = ax.scatter(x, y, c=[imp[v]], cmap='hot', vmin=0, vmax=vmax,
                        s=size, marker=marker, edgecolors='k', linewidths=0.6,
                        alpha=alpha, zorder=5)
        if highlight[v] and imp[v] > 0.3 * vmax:
            ax.annotate(_TYPE_NAMES.get(int(ntypes[v]), '?'), (x, y),
                        fontsize=7, zorder=7)

    if sc is not None:
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label='importance')
    ax.set_aspect('equal')
    ax.set_xlabel('x (m, ego frame)')
    ax.set_ylabel('y (m, ego frame)')
    ax.set_title(title or 'Scene relevance (agents = circles, map = squares)')
    ax.legend(loc='upper right', fontsize=8)

    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close()
    return ax


def select_relevant_nodes(graph_out, top_k=5, batch_idx=0):
    """En onemli top_k dugum (ego haric) + ego'yu sec. Hem graf cizimi hem BEV id
    annotasyonu ayni dugum kumesini kullansin diye ortak yardimci.

    Doner: shown (np.int [<=top_k+1]), ego dahil node id'leri.
    """
    import numpy as np
    imp = graph_out['importance'][batch_idx].detach().cpu().numpy()
    valid = graph_out['valid'][batch_idx].detach().cpu().numpy()
    ntypes = graph_out['node_types'][batch_idx].detach().cpu().numpy()
    ego_sel = np.where(valid & (ntypes == NODE_TYPE_EGO))[0]
    nonego = np.where(valid & (ntypes != NODE_TYPE_EGO))[0]
    top_idx = nonego[np.argsort(-imp[nonego])][:top_k] if len(nonego) else np.array([], dtype=int)
    return np.concatenate([ego_sel, top_idx]).astype(int)


def annotate_map_node_ids(ax, graph_out, shown, batch_idx=0, fontsize=6):
    """SOL BEV ekseninin uzerine, secilen dugumlerin id'lerini gercek (x,y) konumlarina
    kucuk kirmizi etiketlerle yaz. Boylece sagdaki soyut graftaki id ile soldaki harita
    elemani eslesir.
    """
    import numpy as np
    pose = graph_out['node_pose'][batch_idx].detach().cpu().numpy()
    for v in np.asarray(shown).astype(int):
        ax.annotate(str(int(v)), (pose[v, 0], pose[v, 1]), fontsize=fontsize, color='red',
                    zorder=20, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='red', alpha=0.75, lw=0.4))


def plot_scene_graph(graph_out, raw_inputs=None, batch_idx=0, ax=None, save_path=None,
                     title=None, top_k=5, min_edge=0.01, annotate=True):
    """EGO-YILDIZ importance grafigi (harita YOK): ego merkezde, en onemli top_k eleman cevrede.
    Kenarlar SADECE ego->node; kalinlik/etiket = importance (ego self-loop CIKARILIP gosterilen
    elemanlar uzerinde yeniden normalize edildi -> cizilen kenarlar TOPLAM 1). Ham attention
    skoru YOK; her kenar 0-1 importance (ego'nun o elemana verdigi onem).

    Doner: shown (gosterilen node id'leri) -> caller BEV'e ayni id'leri basabilir.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    imp = graph_out['importance'][batch_idx].detach().cpu().numpy()      # [V] ego attention satiri (toplam 1)
    ntypes = graph_out['node_types'][batch_idx].detach().cpu().numpy()   # [V]

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    shown = select_relevant_nodes(graph_out, top_k=top_k, batch_idx=batch_idx)

    # --- Soyut cember duzeni: ego merkezde, digerleri cevrede ---
    ego_nodes = [int(v) for v in shown if ntypes[v] == NODE_TYPE_EGO]
    others = [int(v) for v in shown if ntypes[v] != NODE_TYPE_EGO]
    layout = {}
    if ego_nodes:
        layout[ego_nodes[0]] = (0.0, 0.0)
    for k, node in enumerate(others):
        ang = np.pi / 2 - 2 * np.pi * k / max(len(others), 1)   # tepeden saat yonu
        layout[node] = (float(np.cos(ang)), float(np.sin(ang)))

    # --- ego self-loop CIKAR + gosterilen elemanlar uzerinde yeniden normalize (kenarlar TOPLAM 1) ---
    others_imp = np.array([imp[v] for v in others], dtype=float)
    others_imp = others_imp / max(others_imp.sum(), 1e-9)
    imp_of = {node: float(w) for node, w in zip(others, others_imp)}

    # --- Kenarlar: SADECE ego->node, kalinlik/etiket = importance (toplam 1) ---
    ex, ey = layout[ego_nodes[0]] if ego_nodes else (0.0, 0.0)
    wmax = max(others_imp.max(), 1e-6) if len(others_imp) else 1.0
    for node in others:
        w = imp_of[node]
        if w < min_edge:
            continue
        x, y = layout[node]
        rel = w / wmax
        ax.plot([ex, x], [ey, y], '-', color='crimson',
                linewidth=1.0 + 6.0 * rel, alpha=0.35 + 0.6 * rel, zorder=2)
        mx, my = (ex + x) / 2, (ey + y) / 2
        ax.annotate(f"{w:.2f}", (mx, my), fontsize=8, color='darkred', zorder=8,
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='crimson', alpha=0.9, lw=0.5))

    # --- Dugumler: marker=tip, renk/boyut=onem (KOYU=yuksek), ICINE node id ---
    type_marker = {
        NODE_TYPE_VEHICLE: 'o', NODE_TYPE_PEDESTRIAN: 'P', NODE_TYPE_BICYCLE: 'X',
        NODE_TYPE_LANE: 's', NODE_TYPE_CROSSWALK: 'D', NODE_TYPE_ROUTE: '^',
    }
    vmax = max(others_imp.max(), 1e-6) if len(others_imp) else 1.0
    sc = None
    for v in shown:
        x, y = layout[int(v)]
        if ntypes[v] == NODE_TYPE_EGO:
            ax.scatter(x, y, c='deepskyblue', s=900, marker='*', edgecolors='k', linewidths=1.2, zorder=6)
        else:
            mk = type_marker.get(int(ntypes[v]), 'o')
            sc = ax.scatter(x, y, c=[imp_of[int(v)]], cmap='YlOrRd', vmin=0, vmax=vmax,
                            s=850, marker=mk, edgecolors='k', linewidths=1.0, zorder=5)
        if annotate:
            ax.annotate(str(int(v)), (x, y), fontsize=10, fontweight='bold', color='black',
                        zorder=9, ha='center', va='center',
                        bbox=dict(boxstyle='circle,pad=0.12', fc='white', ec='none', alpha=0.6))

    if sc is not None:
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04,
                     label='onem (importance) — acik sari=dusuk, koyu kirmizi=YUKSEK')

    legend_handles = [
        Line2D([0], [0], marker='*', color='w', markerfacecolor='deepskyblue', markeredgecolor='k', markersize=16, label='ego'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markeredgecolor='k', markersize=11, label='arac'),
        Line2D([0], [0], marker='P', color='w', markerfacecolor='orange', markeredgecolor='k', markersize=11, label='yaya'),
        Line2D([0], [0], marker='X', color='w', markerfacecolor='orange', markeredgecolor='k', markersize=11, label='bisiklet'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='orange', markeredgecolor='k', markersize=11, label='lane (serit)'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='orange', markeredgecolor='k', markersize=10, label='crosswalk'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='orange', markeredgecolor='k', markersize=11, label='route'),
        Line2D([0], [0], color='crimson', lw=4, label='kenar = ego onemi (importance, toplam 1)'),
    ]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=7, framealpha=0.9)

    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title or 'Ego-yildiz onem grafigi (kenar = importance, toplam 1; node ici = id)')

    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close()
    return shown


def build_relevance_record(features, graph_out, batch_idx=0, extra=None):
    """features (tensor dict) + graph_out -> kaydedilebilir/cizilebilir kompakt numpy 'record'.
    --debug sirasinda her planlama adimi icin bunu biriktirip pickle'lariz; sonra ayri bir
    script (render_relevance_video.py) bu record'lardan senaryo videosu uretir.
    """
    import numpy as np

    def npy(x):
        return x[batch_idx].detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x[batch_idx])

    rec = {
        'importance': graph_out['importance'][batch_idx].detach().cpu().numpy().astype('float32'),
        'valid': graph_out['valid'][batch_idx].detach().cpu().numpy(),
        'node_types': graph_out['node_types'][batch_idx].detach().cpu().numpy().astype('int16'),
        'node_pose': graph_out['node_pose'][batch_idx].detach().cpu().numpy().astype('float32'),
        'slices': {k: tuple(int(i) for i in v) for k, v in graph_out['slices'].items()},
        'lanes_xy': npy(features['map_lanes'])[..., :2].astype('float32'),
        'route_xy': npy(features['route_lanes'])[..., :2].astype('float32'),
        'cw_xy': npy(features['map_crosswalks'])[..., :2].astype('float32'),
    }
    # Sag panel grafigi (kim kime bagli) icin attention matrisini de kaydet (varsa)
    if 'attention' in graph_out:
        rec['attention'] = graph_out['attention'][batch_idx].detach().cpu().numpy().astype('float16')
    if extra:
        rec.update(extra)
    return rec


def draw_relevance(ax, rec, threshold=0.65, draw_base=True, annotate=True):
    """ESIK tabanli onem cizimi (tek panel, BEV uzerinde). Sadece NORMALIZE onemi
    (imp / kare-ici-max) >= threshold olan dugumleri VURGULAR:
      - gecen ajanlar: kirmizi halka + normalize skor,
      - gecen harita elemanlari (lane/route/crosswalk): o polyline kirmizi/kalin.
    draw_base=True ise alttaki tum haritayi soluk gri ciz (video icin); plannerv2 canli
    plot'ta zaten zengin BEV var, draw_base=False ile sadece vurgu eklenir.

    rec: build_relevance_record ciktisi (numpy).
    """
    import numpy as np

    imp = rec['importance']; valid = rec['valid']; nt = rec['node_types']
    pose = rec['node_pose']; sl = rec['slices']
    vmax = max(imp[valid].max(), 1e-6) if valid.any() else 1.0
    norm = imp / vmax
    passed = valid & (norm >= threshold)

    if draw_base:
        for arr in (rec['lanes_xy'], rec['route_xy'], rec['cw_xy']):
            for poly in arr:
                m = np.abs(poly).sum(-1) > 1e-6
                if m.any():
                    ax.plot(poly[m, 0], poly[m, 1], '-', color='lightgray', linewidth=0.7, zorder=0)

    def _poly_for(node):
        for key, arr_key in (('lane', 'lanes_xy'), ('crosswalk', 'cw_xy'), ('route', 'route_xy')):
            if key in sl and sl[key][0] <= node < sl[key][1]:
                return rec[arr_key][node - sl[key][0]]
        return None

    type_marker = {NODE_TYPE_VEHICLE: 'o', NODE_TYPE_PEDESTRIAN: 'P', NODE_TYPE_BICYCLE: 'X'}

    # Esigi gecen elemanlari vurgula
    for v in np.where(passed)[0]:
        if nt[v] == NODE_TYPE_EGO:
            continue
        a0, a1 = sl['agent']
        if a0 <= v < a1:   # ajan
            mk = type_marker.get(int(nt[v]), 'o')
            ax.scatter(pose[v, 0], pose[v, 1], s=320, marker=mk, facecolors='none',
                       edgecolors='red', linewidths=2.5, zorder=22)
            if annotate:
                ax.annotate(f"{norm[v]:.2f}", (pose[v, 0], pose[v, 1]), fontsize=8, color='red',
                            zorder=23, ha='center', va='bottom')
        else:              # harita elemani
            poly = _poly_for(int(v))
            if poly is not None:
                m = np.abs(poly).sum(-1) > 1e-6
                if m.any():
                    ax.plot(poly[m, 0], poly[m, 1], '-', color='red', linewidth=3.0, alpha=0.9, zorder=10)
                    if annotate:
                        mid = poly[m][len(poly[m]) // 2]
                        ax.annotate(f"{norm[v]:.2f}", (mid[0], mid[1]), fontsize=7, color='darkred', zorder=11)

    # ego referans
    a0 = sl['agent'][0]
    ax.scatter(pose[a0, 0], pose[a0, 1], c='deepskyblue', s=220, marker='*', edgecolors='k', zorder=25)
    return int(passed.sum())


def plot_bev_relevance(features, graph_out, ax=None, batch_idx=0, save_path=None,
                       title=None, top_k_each=3, cmap='YlOrRd'):
    """Importance'i DOGRUDAN BEV haritasi uzerine bindirir: her lane/route/crosswalk
    polyline'i ve her ajan, GAT onemine gore renklendirilir (KOYU = onemli). Boylece
    "arac bu sahnede su noktalara/ajanlara bakarak karar verdi" gorunumu, node<->harita
    eslesmesi kendiliginden saglanmis sekilde elde edilir. Koordinatlar ego-frame (gercek
    harita) — sol BEV paneliyle ayni eksen.

    features:   ham girdi dict (map_lanes/route_lanes/map_crosswalks).
    graph_out:  SceneRelevanceGraph ciktisi (importance/valid/node_types/node_pose/slices).
    top_k_each: ajan ve harita icin ayri ayri kac elemana onem degeri yazilsin.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    imp = graph_out['importance'][batch_idx].detach().cpu().numpy()
    valid = graph_out['valid'][batch_idx].detach().cpu().numpy()
    ntypes = graph_out['node_types'][batch_idx].detach().cpu().numpy()
    pose = graph_out['node_pose'][batch_idx].detach().cpu().numpy()
    sl = graph_out['slices']

    def _np(x):
        return x[batch_idx].detach().cpu().numpy() if torch.is_tensor(x) else x[batch_idx]

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(1, 1, figsize=(11, 11))

    vmax = max(imp[valid].max(), 1e-6) if valid.any() else 1.0
    cmo = plt.get_cmap(cmap)

    # --- Harita polyline'lari: importance'a gore renk + kalinlik (koyu/kalin = onemli) ---
    def _draw_polys(key, sl_key):
        if key not in features or sl_key not in sl:
            return
        polys = _np(features[key])[..., :2]
        start = sl[sl_key][0]
        for e in range(polys.shape[0]):
            node = start + e
            if node >= len(imp) or not valid[node]:
                continue
            m = np.abs(polys[e]).sum(-1) > 1e-6
            if not m.any():
                continue
            w = float(imp[node] / vmax)
            ax.plot(polys[e][m, 0], polys[e][m, 1], '-', color=cmo(w),
                    linewidth=1.0 + 4.5 * w, alpha=0.25 + 0.7 * w, zorder=2 + int(8 * w))

    _draw_polys('map_lanes', 'lane')
    _draw_polys('route_lanes', 'route')
    _draw_polys('map_crosswalks', 'crosswalk')

    # --- Ajanlar: marker=tip, renk/boyut=importance, ego=yildiz ---
    type_marker = {NODE_TYPE_VEHICLE: 'o', NODE_TYPE_PEDESTRIAN: 'P', NODE_TYPE_BICYCLE: 'X'}
    a0, a1 = sl['agent']
    for v in range(a0, a1):
        if not valid[v]:
            continue
        x, y = pose[v, 0], pose[v, 1]
        if ntypes[v] == NODE_TYPE_EGO:
            ax.scatter(x, y, c='deepskyblue', s=280, marker='*', edgecolors='k', linewidths=1.0, zorder=20)
        else:
            mk = type_marker.get(int(ntypes[v]), 'o')
            ax.scatter(x, y, c=[imp[v]], cmap=cmap, vmin=0, vmax=vmax,
                       s=120 + 450 * (imp[v] / vmax), marker=mk,
                       edgecolors='k', linewidths=0.9, zorder=21)

    # --- Her kategoriden top_k_each elemana onem degerini yaz (ajan VE harita gorunsun) ---
    def _topk(rng, k):
        idxs = [v for v in range(rng[0], rng[1]) if valid[v]]
        return sorted(idxs, key=lambda v: -imp[v])[:k]

    map_range = (sl['lane'][0], sl['route'][1])   # lane+crosswalk+route bitisik
    for v in _topk(sl['agent'], top_k_each) + _topk(map_range, top_k_each):
        if ntypes[v] == NODE_TYPE_EGO:
            continue
        ax.annotate(f"{imp[v]:.2f}", (pose[v, 0], pose[v, 1]), fontsize=8, color='navy', zorder=22,
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='navy', alpha=0.8, lw=0.5))

    sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, vmax))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label='onem (importance) — KOYU = yuksek')

    ax.set_aspect('equal')
    ax.set_xlabel('x (m, ego frame)')
    ax.set_ylabel('y (m, ego frame)')
    ax.set_title(title or 'Onem haritasi: arac neye bakarak karar verdi (koyu=onemli)')

    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close()
    return ax
