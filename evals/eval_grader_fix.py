"""HAKEM DUZELTMESI GT DOGRULAMASI: eski (turn-once) vs yeni (koridor-bazli LC) lat hakemi,
GT ego future'lari uzerinde 7x7 gecis matrisi.

Kabul kriteri: gercek donusler yerinde kalmali (turn satirlarinin >=~%99'u diagonalde);
hareket yalniz lc/inlane/no_lateral <-> lane_change sinir vakalarinda olmali.

Kosum:
  python eval_grader_fix.py --valid_set /home/lt-hta-ai4/ssd1/nuplan/processed_data/validation \
    [--limit 0]
"""
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from GameFormer.train_utils import DrivingData
from GameFormer.decision_labels import decision_labels, LAT_CLASSES


@torch.no_grad()
def main(a):
    ds = DrivingData(a.data + "/*.npz", 10)
    ld = DataLoader(ds, batch_size=64, shuffle=False, num_workers=8)
    C = np.zeros((7, 7), dtype=int)                      # [eski, yeni]
    n = 0
    for batch in ld:
        ego_future = batch[6] if a.future_slot >= 0 else None
        # DrivingData cikti sirasi train_planner.read_batch ile ayni: ego_future = batch[6]?
        # Guvenli yol: read_batch kullan.
        from train_planner import read_batch
        inputs, ego_future, _, ref_path = read_batch(batch, 'cpu')
        # GT future zaten (x,y,heading) mi? read_batch ego_future [B,80,3] dondurur.
        _, old = decision_labels(ego_future, ref_path, turn_fix=False)
        _, new = decision_labels(ego_future, ref_path, turn_fix=True)
        for o, y in zip(old.tolist(), new.tolist()):
            C[o, y] += 1
        n += ego_future.shape[0]
        if a.limit and n >= a.limit:
            break

    print(f"\n=== hakem gecis matrisi (GT future, n={n}) — satir: ESKI, sutun: YENI ===")
    hdr = "".join(f"{c[:9]:>10s}" for c in LAT_CLASSES)
    print(f"{'':16s}{hdr}")
    for i, c in enumerate(LAT_CLASSES):
        row = "".join(f"{C[i, j]:10d}" for j in range(7))
        print(f"{c:16s}{row}")
    moved = C.sum() - np.trace(C)
    print(f"\ntoplam degisen: {moved} ({100*moved/max(C.sum(),1):.2f}%)")
    for i in (0, 1):                                     # turn kararliligi
        tot = C[i].sum()
        if tot:
            print(f"{LAT_CLASSES[i]} kararlilik: {100*C[i,i]/tot:.1f}%  (n={tot})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--future_slot", type=int, default=-1)
    main(p.parse_args())
