"""Bir checkpoint icin validation metriklerini YENIDEN hesapla (yeniden EGITIM YOK).

Amac: eski kosularda loglanmamis metrikleri (ornegin fcfd_var/fcas_var collapse
monitoru) geriye donuk elde etmek, boylece eski kosular baseline olarak kullanilabilsin.
Tek validation gecisi ~1-2 dk.
"""

import argparse

import torch
from torch.utils.data import DataLoader

from GameFormer.predictor import GameFormer
from GameFormer.causal_graph import CausalPlanner
from GameFormer.train_utils import DrivingData
from train_planner import _run_epoch, freeze_gameformer


def main():
    p = argparse.ArgumentParser(description="Recompute val metrics for an existing checkpoint")
    p.add_argument("--pretrained_path", required=True, help="frozen GameFormer checkpoint")
    p.add_argument("--causal_path", required=True, help="trained CausalPlanner checkpoint")
    p.add_argument("--valid_set", required=True)
    p.add_argument("--num_neighbors", type=int, default=10)
    p.add_argument("--encoder_layers", type=int, default=3)
    p.add_argument("--decoder_levels", type=int, default=2)
    p.add_argument("--graph_layers", type=int, default=1)
    p.add_argument("--nbr_enrich", type=int, default=0)
    p.add_argument("--ego_residual", type=int, default=1,
                   help="checkpoint hangi degerle EGITILDIYSE o verilmeli (0 = h_ego residual yok)")
    p.add_argument("--gate", type=str, default="softmax", choices=["softmax", "sigmoid"],
                   help="checkpoint hangi kapi moduyla EGITILDIYSE o verilmeli (aksi halde maske yanlis hesaplanir)")
    p.add_argument("--modes", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lambda_kld", type=float, default=1.0)
    p.add_argument("--lambda_ci", type=float, default=0.5)
    p.add_argument("--lambda_mask", type=float, default=0.5)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    dev = torch.device(args.device)

    gameformer = GameFormer(encoder_layers=args.encoder_layers, decoder_levels=args.decoder_levels,
                            neighbors=args.num_neighbors).to(dev)
    gameformer.load_state_dict(torch.load(args.pretrained_path, map_location=dev))
    freeze_gameformer(gameformer)

    causal = CausalPlanner(layers=args.graph_layers, modes=args.modes, nbr_enrich=args.nbr_enrich, gate=args.gate, ego_residual=args.ego_residual).to(dev)
    # strict=False: model yeni modul kazandikca (gate_bias, nbr_head, cfd_recon) eski checkpoint'ler
    # o anahtarlari icermez. Kullanilmayan modullerin eksik olmasi zararsiz -- ama ne eksikse yazdir.
    missing, unexpected = causal.load_state_dict(torch.load(args.causal_path, map_location=dev), strict=False)
    if missing or unexpected:
        print(f"[load] missing={list(missing)}  unexpected={list(unexpected)}")
    causal.eval()

    valid_set = DrivingData(args.valid_set + "/*.npz", args.num_neighbors)
    loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    m = _run_epoch(loader, gameformer, causal, dev, args.num_neighbors,
                   args.lambda_kld, args.lambda_ci, args.lambda_mask,
                   optimizer=None, desc="Recompute")

    ratio = m['fcfd_var'] / max(m['fcas_var'], 1e-9)
    print(f"\n=== {args.causal_path.split('/')[-2]} / {args.causal_path.split('/')[-1]} ===")
    for k in ('minADE', 'minFDE', 'casacc', 'cfdacc', 'mcas_peak', 'mcas_map_peak',
              'mcfd_peak', 'unif', 'comp', 'excl', 'norm', 'fcas_var', 'fcfd_var'):
        if k in m:
            print(f"  {k:16s} {m[k]:.4f}")
    print(f"  {'cfdvar_ratio':16s} {ratio:.5f}   (->0 = f_cfd cokmus)")


if __name__ == "__main__":
    main()
