"""Backbone-baseline lat audit (read-only).

Hipotez (kullanici): "backbone zaten ego gelecegini tahmin ediyor; ModeSelector SECIMI bozuyor."
Test: ModeSelector'i HIC kullanmadan, sadece frozen GameFormer'in TAHMIN ETTIGI ego endpoint'ine
en yakin rotayi lat olarak sec; bunu GT-endpoint lat'i ile karsilastir.

Cikan sayilar:
  - backbone_lat_acc: "backbone tahminiyle rota secseydik" lat dogrulugu. ModeSelector lat'indan
    (val-lat ~0.75) YUKSEK ise -> sorun belirsizlik degil, ModeSelector. DUSUK ise -> backbone
    tahmini de rotayi belirleyemiyor (daha derin sinir).
  - #gecerli-rota kovasina gore kirilim: kac secenek varken ne kadar isabet (belirsizlik proxy'si).
  - upper bound: backbone tahmini yerine GT endpoint kullanilirsa (=etiketin kendisi) her zaman %100;
    onun yerine 'oracle within eps' raporlanir.
  - ego FDE: backbone ego endpoint tahmini ne kadar iyi (m).

Kullanim:
  python debug_backbone_baseline.py \
      --valid_set /home/lt-hta-ai4/ssd1/nuplan/processed_data/validation \
      --pretrained_path training_log/normal/model_epoch_19_valADE_1.6487.pth
"""
import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from GameFormer.train_utils import DrivingData
from GameFormer.predictor import GameFormer
from train_planner import read_batch

NUM_LAT = 5


def closest_route(endpoint, c_lat):
    """endpoint [B,2], c_lat [B,5,T,6] -> best_lat [B], lat_valid [B,5].
    get_expert_mode_index'in lat mantiginin birebir aynisi (min mesafe, padded maske)."""
    diffs = c_lat[..., :2] - endpoint[:, None, None]                  # [B,5,T,2]
    d = torch.norm(diffs, dim=-1)                                     # [B,5,T]
    point_valid = (c_lat.abs().sum(-1) > 1e-4)                        # [B,5,T]
    d = d.masked_fill(~point_valid, float('inf'))
    min_dist = d.min(-1).values                                      # [B,5]
    lat_valid = (c_lat.abs().sum((2, 3)) > 1e-4)                     # [B,5]
    min_dist = min_dist.masked_fill(~lat_valid, float('inf'))
    return min_dist.argmin(-1), min_dist, lat_valid


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--valid_set", type=str, required=True)
    p.add_argument("--pretrained_path", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_neighbors", type=int, default=10)
    p.add_argument("--encoder_layers", type=int, default=3)
    p.add_argument("--decoder_levels", type=int, default=2)
    p.add_argument("--eps_lat", type=float, default=2.0, help="oracle 'esdeger' esigi [m]")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    device = args.device

    ds = DrivingData(args.valid_set + "/*.npz", args.num_neighbors)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=min(8, os.cpu_count()))
    print(f"{len(ds)} samples")

    gf = GameFormer(encoder_layers=args.encoder_layers,
                    decoder_levels=args.decoder_levels, neighbors=args.num_neighbors)
    gf.load_state_dict(torch.load(args.pretrained_path, map_location=device))
    gf = gf.to(device).eval()
    for prm in gf.parameters():
        prm.requires_grad_(False)

    n = 0
    lat_match = 0                                  # backbone lat == GT lat
    oracle_match = 0                               # backbone lat, GT esdeger-kumesinde mi (eps_lat)
    fde_sum = 0.0
    bucket_tot = np.zeros(NUM_LAT + 1)             # #gecerli rota -> ornek
    bucket_match = np.zeros(NUM_LAT + 1)           # #gecerli rota -> backbone isabet

    for batch in tqdm(loader, desc="Backbone baseline"):
        inputs, ego_future, _, c_lat = read_batch(batch, device)
        B = ego_future.shape[0]
        with torch.no_grad():
            enc = gf.encoder(inputs)
            dec, _ = gf.decoder(enc)
            lk = max(int(k.split('_')[1]) for k in dec if 'interactions' in k)
            ego_traj = dec[f'level_{lk}_interactions'][:, 0]          # [B,M,T,4]
            ego_scores = dec[f'level_{lk}_scores'][:, 0]             # [B,M]
            best_mode = ego_scores.argmax(-1)                        # [B]
            pred_ep = ego_traj[torch.arange(B), best_mode, -1, :2]   # tahmini endpoint [B,2]

        gt_ep = ego_future[:, -1, :2]
        lat_pred, _, lat_valid = closest_route(pred_ep, c_lat)
        lat_gt, gt_min_dist, _ = closest_route(gt_ep, c_lat)

        match = (lat_pred == lat_gt)
        lat_match += match.sum().item()
        # oracle: backbone'un sectigi rota, GT'ye gore best+eps icinde mi (esdeger dogru sayilir mi)
        best_gt = gt_min_dist.min(-1, keepdim=True).values
        equiv = (gt_min_dist <= best_gt + args.eps_lat) & lat_valid
        oracle_match += equiv.gather(1, lat_pred[:, None]).squeeze(1).sum().item()
        fde_sum += torch.norm(pred_ep - gt_ep, dim=-1).sum().item()

        nvalid = lat_valid.sum(-1)                                   # [B]
        for k in range(1, NUM_LAT + 1):
            m = (nvalid == k)
            bucket_tot[k] += m.sum().item()
            bucket_match[k] += (match & m).sum().item()
        n += B

    print("\n========== BACKBONE-BASELINE LAT AUDIT ==========")
    print(f"samples: {n}")
    print(f"backbone_lat_acc (exact):     {100.0 * lat_match / n:.2f}%   <- ModeSelector lat (~75) ile kiyasla")
    print(f"backbone_lat_acc (eps={args.eps_lat}m oracle): {100.0 * oracle_match / n:.2f}%")
    print(f"ego endpoint FDE (mean):      {fde_sum / n:.2f} m")
    print("\n--- #gecerli rota kovasina gore backbone lat isabeti (belirsizlik proxy'si) ---")
    for k in range(1, NUM_LAT + 1):
        if bucket_tot[k]:
            print(f"  {k} rota: {100.0 * bucket_tot[k] / n:5.1f}% ornek | isabet {100.0 * bucket_match[k] / max(bucket_tot[k],1):5.1f}%")
    print("=================================================\n")


if __name__ == "__main__":
    main()
