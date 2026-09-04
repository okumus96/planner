"""RELATION-IDENTIFIABILITY PROBE (okuma-amacli teshis; ana kodu DEGISTIRMEZ).

Soru: typed attention'a giren mesaj, hangi ILISKIDEN geldigini kodluyor mu?
  msg[s,r] = Wv_ch[r](h_s) + edge_v(edge_s)
Ilk terim iliskiye ozel, ikinci terim o komsunun TUM iliski girdilerinde AYNI (paylasilan).

Uc olcum:
  P1  probe(v_ch[s,r]) -> r      : iliski donusumleri birbirinden ayirt edilebilir mi?
                                   (dusuk => Wv_ch[r] matrisleri birbirine cokmus, 2. asama)
  P2  probe(msg[s,r])  -> r      : paylasilan edge terimi eklendikten sonra kimlik ayakta mi?
                                   (P1 yuksek, P2 dusuk => paylasilan edge sinyali bogyor, 1. asama)
  P3  probe(msg), etiketler KARISTIRILMIS : sans tabani kontrolu (~1/R cikmali)

Ayrica: ||edge_v|| / ||msg|| orani (mesajin ne kadari iliskiden BAGIMSIZ), ve
Wv_ch[r]/Wk_ch[r] matrisleri arasi ikili kosinus benzerligi.

Ornek (v2):
  PYTHONPATH=.:evals python evals/eval_relation_probe.py \
    --pretrained_path training_log/normal/model_epoch_19_valADE_1.6487.pth \
    --causal_path training_log/v3_latmoe/causal_epoch_12_minADE_0.7998.pth \
    --valid_set /home/lt-hta-ai4/ssd1/nuplan/processed_data/validation \
    --graph_layers 1 --nbr_enrich 2 --ego_residual 0 --gate_channels 1 --typed_kv 1 \
    --dod_meta 1 --lat_moe 1 --num_batches 20 --device cuda:1
"""
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from GameFormer.predictor import GameFormer
from GameFormer.causal_graph import CausalPlanner, EgoCausalLayer
from GameFormer.train_utils import DrivingData
from GameFormer.channels import CHANNEL_NAMES, NUM_CHANNELS
from GameFormer.decision_labels import NUM_LON4, NUM_LAT5V, NUM_LON5, NUM_LAT5
from train_planner import read_batch, extract_neighbor_top1_futures, freeze_gameformer

_CAP = {}          # monkey-patch'in son ajan-dali cagrisini biriktirdigi yer


def _install_capture():
    """EgoCausalLayer._attend_typed'i SARMALA: ajan dalinin (h_src, edge_v, entry_valid, Wv_list)
    girdilerini yakala. Orijinal fonksiyon aynen cagrilir -> model davranisi DEGISMEZ."""
    orig = EgoCausalLayer._attend_typed

    def wrapped(self, q1, h_src, ek, edge_v, entry_valid, Wk_list, Wv_list,
                attn_cas_ch, attn_cfd_ch, conflict_bias=None):
        if entry_valid.shape[-1] == NUM_CHANNELS:        # ajan dali (harita dalini atla)
            _CAP['h_src'] = h_src.detach()
            _CAP['edge_v'] = edge_v.detach()
            _CAP['entry_valid'] = entry_valid.detach()
            _CAP['Wv'] = Wv_list
        return orig(self, q1, h_src, ek, edge_v, entry_valid, Wk_list, Wv_list,
                    attn_cas_ch, attn_cfd_ch, conflict_bias=conflict_bias)

    EgoCausalLayer._attend_typed = wrapped
    return orig


def fit_probe(X, y, R, epochs=300, seed=0, name=""):
    """Cok sinifli LINEER probe (lojistik regresyon). Egitim istatistikleriyle standardize,
    %80/%20 bolme. Doner: (test acc, sans tabani)."""
    dev = X.device
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(X.shape[0], generator=g).to(dev)
    ntr = int(0.8 * perm.numel())
    tr, te = perm[:ntr], perm[ntr:]
    mu, sd = X[tr].mean(0), X[tr].std(0).clamp(min=1e-6)
    Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
    ytr, yte = y[tr], y[te]
    lin = nn.Linear(X.shape[1], R).to(dev)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-2, weight_decay=1e-4)
    for _ in range(epochs):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(lin(Xtr), ytr)
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = (lin(Xte).argmax(-1) == yte).float().mean().item()
        # sans tabani = en sik sinifin test payi (dengesiz dagilimda 1/R'den buyuk olabilir)
        maj = torch.bincount(yte, minlength=R).max().item() / yte.numel()
    print(f"  {name:34s} acc={acc:6.3f}   majority-baseline={maj:.3f}   n={X.shape[0]}")
    return acc, maj


def fit_binary(X, yb, epochs=300, seed=0, name=""):
    """Tek iliski icin IKILI lineer probe. Doner: acc - taban (taban = cogunluk sinifi orani)."""
    dev = X.device
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(X.shape[0], generator=g).to(dev)
    ntr = int(0.8 * perm.numel())
    tr, te = perm[:ntr], perm[ntr:]
    mu, sd = X[tr].mean(0), X[tr].std(0).clamp(min=1e-6)
    Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
    ytr, yte = yb[tr].long(), yb[te].long()
    lin = nn.Linear(X.shape[1], 2).to(dev)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-2, weight_decay=1e-4)
    for _ in range(epochs):
        opt.zero_grad()
        nn.functional.cross_entropy(lin(Xtr), ytr).backward()
        opt.step()
    with torch.no_grad():
        acc = (lin(Xte).argmax(-1) == yte).float().mean().item()
    base = max(yte.float().mean().item(), 1 - yte.float().mean().item())
    print(f"{name} acc={acc:6.3f}  taban={base:.3f}  fark={acc-base:+.3f}")
    return acc - base


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_path", required=True)
    p.add_argument("--causal_path", required=True)
    p.add_argument("--valid_set", required=True)
    p.add_argument("--num_neighbors", type=int, default=10)
    p.add_argument("--encoder_layers", type=int, default=3)
    p.add_argument("--decoder_levels", type=int, default=2)
    p.add_argument("--graph_layers", type=int, default=1)
    p.add_argument("--nbr_enrich", type=int, default=0)
    p.add_argument("--ego_residual", type=int, default=1)
    p.add_argument("--gate", type=str, default="softmax")
    p.add_argument("--gate_channels", type=int, default=0)
    p.add_argument("--typed_kv", type=int, default=0)
    p.add_argument("--channel_evidence", type=int, default=0)
    p.add_argument("--gate_trust", type=str, default="all")
    p.add_argument("--dod_meta", type=int, default=0)
    p.add_argument("--dec_moe", type=int, default=0)
    p.add_argument("--lat_moe", type=int, default=0)
    p.add_argument("--lon_merge", type=int, default=0)
    p.add_argument("--modes", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_batches", type=int, default=20)
    p.add_argument("--random_init", type=int, default=0,
                   help="1 = ckpt yukleme; rastgele-agirlik kontrol tabani")
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()
    assert args.typed_kv, "probe typed_kv ckpt ister (iliski ekseni yoksa soru anlamsiz)"
    dev = torch.device(args.device)

    gameformer = GameFormer(encoder_layers=args.encoder_layers, decoder_levels=args.decoder_levels,
                            modalities=args.modes, neighbors=args.num_neighbors).to(dev)
    gameformer.load_state_dict(torch.load(args.pretrained_path, map_location=dev))
    freeze_gameformer(gameformer)
    gameformer.eval()

    causal = CausalPlanner(layers=args.graph_layers, modes=args.modes, nbr_enrich=args.nbr_enrich,
                           gate=args.gate, ego_residual=args.ego_residual,
                           gate_channels=args.gate_channels, typed_kv=args.typed_kv,
                           channel_evidence=args.channel_evidence, gate_trust=args.gate_trust,
                           dod_meta=args.dod_meta, dec_moe=args.dec_moe, lat_moe=args.lat_moe,
                           num_lon=(NUM_LON4 if args.lat_moe else NUM_LON5 if args.dec_moe
                                    else 6 if args.lon_merge else 9),
                           num_lat=(NUM_LAT5V if args.lat_moe
                                    else NUM_LAT5 if args.dec_moe else 7)).to(dev)
    if args.random_init:
        # KONTROL: egitilmis agirliklari YUKLEME. Rastgele baslatilmis Wv_ch[r] matrisleri de
        # neredeyse dik olur -> probe zaten yuksek cikabilir. Egitimin kimlik ayrisimina
        # EK katkisi = (egitilmis acc) - (rastgele acc).
        print("[load] RANDOM INIT (kontrol kosumu -- ckpt yuklenmedi)")
    else:
        miss, unexp = causal.load_state_dict(torch.load(args.causal_path, map_location=dev), strict=False)
        print(f"[load] missing={list(miss) or 'NONE'}  unexpected={list(unexp) or 'NONE'}")
    causal.eval()

    _install_capture()

    ds = DrivingData(args.valid_set + "/*.npz", args.num_neighbors)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    Xv, Xm, Y, Yf, Xmf = [], [], [], [], []
    Fc, Fy, Fm = [], [], []      # v_ch / msg / r-etiketi (tum-r) ; yanan-girdi seti
    n_edge, n_node, n_msg, nb = 0.0, 0.0, 0.0, 0
    fired = np.zeros(NUM_CHANNELS, dtype=np.int64)

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if bi >= args.num_batches:
                break
            inputs, ego_future, _, ref_path = read_batch(batch, dev)
            enc = gameformer.encoder(inputs)
            top1, nbr_states, _ = extract_neighbor_top1_futures(gameformer, enc, args.num_neighbors)
            out_d = causal(enc, inputs, num_agents=args.num_neighbors + 1,
                           neighbor_futures=top1, neighbor_states=nbr_states, ref_path=ref_path)
            mcas_typed = out_d.get('M_cas_typed')
            if 'h_src' not in _CAP:
                raise RuntimeError("yakalama basarisiz: typed ajan dali hic cagrilmadi")

            h_src, edge_v, ev = _CAP['h_src'], _CAP['edge_v'], _CAP['entry_valid']
            B, S, D = h_src.shape
            R = ev.shape[-1]
            v_ch = torch.stack([m(h_src) for m in _CAP['Wv']], dim=2)       # [B,S,R,D]
            e_b = edge_v.reshape(B, S, D)[:, :, None]                       # [B,S,1,D] paylasilan
            msg = v_ch + e_b                                                # [B,S,R,D]

            any_rel = ev.any(-1)                                            # [B,S] gecerli komsu
            idx = any_rel.nonzero(as_tuple=False)
            if idx.numel() == 0:
                continue
            b_i, s_i = idx[:, 0], idx[:, 1]
            # --- AJAN-ICI set: ayni h_s, ayni edge; SADECE r degisiyor (saf donusum sorusu) ---
            Xv.append(v_ch[b_i, s_i].reshape(-1, D).float().cpu())          # [n*R, D]
            Xm.append(msg[b_i, s_i].reshape(-1, D).float().cpu())
            Y.append(torch.arange(R).repeat(idx.shape[0]))
            # --- YANAN-GIRDI seti: modelin gercekte gordugu (ajan-iliski korelasyonu iceride) ---
            f_idx = ev.nonzero(as_tuple=False)
            Xmf.append(msg[f_idx[:, 0], f_idx[:, 1], f_idx[:, 2]].float().cpu())
            Yf.append(f_idx[:, 2].cpu())
            # --- 3. ASAMA: toplamdan SONRA kimlik ayakta mi? f_cas -> "hangi iliskiler yandi"
            Fc.append(out_d['f_cas'].detach().float().cpu())                # [B,D]
            Fy.append(ev.any(1).float().cpu())                              # [B,R] sahnede yandi mi
            Fm.append(mcas_typed.sum(1).float().cpu() if mcas_typed is not None
                      else torch.zeros(B, R))                               # [B,R] kutle profili
            fired += np.bincount(f_idx[:, 2].cpu().numpy(), minlength=NUM_CHANNELS)
            # --- norm istatistikleri ---
            n_edge += e_b.expand(-1, -1, R, -1)[b_i, s_i].norm(dim=-1).mean().item()
            n_node += v_ch[b_i, s_i].norm(dim=-1).mean().item()
            n_msg += msg[b_i, s_i].norm(dim=-1).mean().item()
            nb += 1

    Xv = torch.cat(Xv).to(dev); Xm = torch.cat(Xm).to(dev); Y = torch.cat(Y).to(dev)
    Xmf = torch.cat(Xmf).to(dev); Yf = torch.cat(Yf).to(dev)

    print(f"\n=== NORMLAR (ajan-ici, {nb} batch ortalamasi) ===")
    print(f"  ||edge_v|| (PAYLASILAN, r'den bagimsiz) = {n_edge/nb:.3f}")
    print(f"  ||v_ch[r]|| (iliskiye OZEL)             = {n_node/nb:.3f}")
    print(f"  ||msg||                                 = {n_msg/nb:.3f}")
    print(f"  paylasilan pay ||edge||/||msg||         = {n_edge/n_msg:.3f}")

    print(f"\n=== PROBE (lineer, %80/%20 bolme; R={NUM_CHANNELS}, sans=1/R={1/NUM_CHANNELS:.3f}) ===")
    fit_probe(Xv, Y, NUM_CHANNELS, name="P1  v_ch[r] -> r  (saf donusum)")
    fit_probe(Xm, Y, NUM_CHANNELS, name="P2  msg[r]  -> r  (edge dahil)")
    Ysh = Y[torch.randperm(Y.numel(), device=dev)]
    fit_probe(Xm, Ysh, NUM_CHANNELS, name="P3  msg -> KARISTIRILMIS r (kontrol)")
    fit_probe(Xmf, Yf, NUM_CHANNELS, name="P4  yanan girdiler (korelasyon dahil)")

    Fc = torch.cat(Fc).to(dev); Fy = torch.cat(Fy).to(dev); Fm = torch.cat(Fm).to(dev)
    print(f"\n=== 3. ASAMA: TOPLAMDAN SONRA (f_cas -> iliski yandi mi), n={Fc.shape[0]} sahne ===")
    print("  (taban = o iliskinin sahnelerdeki yanma orani; fark = probe'un tabana EK kazanci)")
    gains = []
    for r in range(NUM_CHANNELS):
        a = fit_binary(Fc, Fy[:, r], name=f"  f_cas -> {CHANNEL_NAMES[r][:26]:26s}")
        gains.append(a)
    print(f"  ORTALAMA ek kazanc (acc - taban): {np.mean([g for g in gains]):+.3f}")
    print("\n  UST SINIR karsilastirmasi (kutle profilinden okumak -- toplanmamis sinyal):")
    gp = [fit_binary(Fm, Fy[:, r], name=f"  profil -> {CHANNEL_NAMES[r][:26]:26s}")
          for r in range(NUM_CHANNELS)]
    print(f"  ORTALAMA ek kazanc (profil): {np.mean(gp):+.3f}")

    print("\n=== MATRIS BENZERLIGI (Wv_ch[r] ikili kosinus, duzlestirilmis) ===")
    W = torch.stack([m.weight.detach().reshape(-1) for m in _CAP['Wv']])
    Wn = W / W.norm(dim=1, keepdim=True)
    C = (Wn @ Wn.T).cpu().numpy()
    off = C[~np.eye(len(C), dtype=bool)]
    print(f"  kosegen disi kosinus: ort={off.mean():.3f}  min={off.min():.3f}  max={off.max():.3f}")

    print("\n=== YANMA SAYILARI ===")
    for i, nm in enumerate(CHANNEL_NAMES):
        print(f"  {nm:38s} {fired[i]:8d}  ({100*fired[i]/max(1,fired.sum()):5.1f}%)")


if __name__ == "__main__":
    main()
