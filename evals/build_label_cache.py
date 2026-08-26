"""Egitim seti karar-etiket cache'i (dup_boost icin): DrivingData'nin __getitem__'inde
hesaplanan HAM 9x7 (lon, lat) etiketlerini tek npz'ye dokur. Bir kez kosulur; train_planner
--dup_boost --label_cache ile okur. (~177k ornek, num_workers'a gore 10-20 dk.)

Kosum (repo kokunden):
  python evals/build_label_cache.py --data /home/lt-hta-ai4/ssd1/nuplan/processed_data/train \
    --out train_label_cache.npz
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader

from GameFormer.train_utils import DrivingData


def main(a):
    ds = DrivingData(a.data + "/*.npz", a.num_neighbors)
    ld = DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=a.workers)
    lon, lat = [], []
    n = 0
    for batch in ld:
        lon.append(batch[13].to(torch.int8))          # DrivingData slot 13: decision_lon (ham 9)
        lat.append(batch[14].to(torch.int8))          # slot 14: decision_lat (ham 7)
        n += batch[13].shape[0]
        if n % 20000 < a.batch_size:
            print(f"  {n}/{len(ds)}")
    lon = torch.cat(lon).numpy()
    lat = torch.cat(lat).numpy()
    assert len(lon) == len(ds)
    np.savez(a.out, lon=lon, lat=lat)
    bl = np.bincount(lon.astype(np.int64), minlength=9)
    bt = np.bincount(lat.astype(np.int64), minlength=7)
    print(f"{len(lon)} etiket -> {a.out}\nlon: {bl.tolist()}\nlat: {bt.tolist()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", default="train_label_cache.npz")
    p.add_argument("--num_neighbors", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--workers", type=int, default=24)
    main(p.parse_args())
