"""to_left/to_right TESHISI: zorlanmis lat planlarinin koridor-goreli kayma histogrami.

Soru: to_* compliance'inin ~%31'de kalmasi (a) uretec hic kaymiyor mu, (b) 0.6 m esiginin
hemen altinda mi yigiliyor? Kayma, relabeler ile BIREBIR ayni hesap: _project d_lat'inin
son-1/8 medyani - ilk-1/8 medyani (_lat_one'daki delta). Zorlanan yon dogrultusunda isaretlenir
(to_right icin -delta). 6 modun EN IYI yon-kaymasi alinir (any-mode tavaniyla ayni mantik).

Kosum:
  python eval_lat_drift.py --pretrained_path training_log/normal/model_epoch_19_valADE_1.6487.pth \
    --causal_path training_log/v3_latmoe/causal_epoch_12_minADE_0.7998.pth \
    --valid_set /home/lt-hta-ai4/ssd1/nuplan/processed_data/validation --device cuda:1
"""
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from GameFormer.predictor import GameFormer
from GameFormer.causal_graph import CausalPlanner
from GameFormer.channels import _corridor_arrays, _project
from GameFormer.train_utils import DrivingData
from train_planner import read_batch, extract_neighbor_top1_futures, freeze_gameformer

BUCKETS = [(-1e9, 0.0, 'yanlis yon (<0)'), (0.0, 0.3, '0-0.3 m'), (0.3, 0.6, '0.3-0.6 m'),
           (0.6, 2.0, '0.6-2.0 m (inlane bandi)'), (2.0, 1e9, '>=2.0 m (LC bandi)')]


def drift(pts, ref):
    """pts [N,80,2], ref [N,R,P,C] -> koridor-goreli delta d_lat [N] (relabeler formulu)."""
    cxy, cyaw, ccum, cvalid = _corridor_arrays(ref)
    _, d_lat, _, _ = _project(pts.float(), cxy, cyaw, ccum, cvalid)      # [N,80]
    k = max(d_lat.shape[1] // 8, 1)
    return (d_lat[:, -k:].median(dim=1).values - d_lat[:, :k].median(dim=1).values)


@torch.no_grad()
def main(a):
    dev = a.device
    gf = GameFormer(encoder_layers=3, decoder_levels=2, neighbors=a.num_neighbors)
    gf.load_state_dict(torch.load(a.pretrained_path, map_location=dev))
    gf = gf.to(dev); freeze_gameformer(gf)
    m = CausalPlanner(layers=1, modes=6, nbr_enrich=2, ego_residual=0, gate_channels=1,
                      typed_kv=1, dod_meta=1, lat_moe=1, num_lon=4, num_lat=5).to(dev)
    miss, unexp = m.load_state_dict(torch.load(a.causal_path, map_location=dev), strict=False)
    assert not miss and not unexp
    m.eval()

    ds = DrivingData(a.valid_set + "/*.npz", a.num_neighbors)
    ld = DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=4)
    res = {2: [], 3: []}                                   # to_left / to_right yon-kaymalari
    n = 0
    for batch in ld:
        inp, ef, nf, rp = read_batch(batch, dev)
        enc = gf.encoder(inp)
        t1, ns, _ = extract_neighbor_top1_futures(gf, enc, a.num_neighbors)
        out = m(enc, inp, num_agents=a.num_neighbors + 1, neighbor_futures=t1, neighbor_states=ns)
        B = ef.shape[0]
        M = out['traj'].shape[2]
        b_lon = out['psi_lon_cas'].argmax(-1)
        b_lat = out['psi_lat_cas'].argmax(-1)
        rep = rp.repeat_interleave(M, dim=0)
        for cls, sgn in ((2, +1.0), (3, -1.0)):            # to_left: +d_lat; to_right: -d_lat
            ft = torch.full_like(b_lat, cls)
            trajF, _ = m.head(out['f_cas'], out['ego_clean'], (b_lon, ft))
            pts = trajF[:, 0][..., :2].reshape(B * M, -1, 2)
            d = (sgn * drift(pts, rep)).view(B, M).max(dim=1).values      # en iyi yon-kaymasi
            mask = (b_lat != cls)
            res[cls] += d[mask].tolist()
        n += B
        if a.limit and n >= a.limit:
            break

    for cls, name in ((2, 'to_left'), (3, 'to_right')):
        d = np.array(res[cls])
        print(f"\n=== force {name}  (n={len(d)} zorlama; 6 modun EN IYI yon-kaymasi) ===")
        for lo, hi, lab in BUCKETS:
            c = int(((d >= lo) & (d < hi)).sum())
            print(f"  {lab:24s} {c:5d}  ({100*c/len(d):5.1f}%)")
        print(f"  medyan={np.median(d):+.2f} m   ort={d.mean():+.2f} m   "
              f">=0.6m (tavan)={100*(d>=0.6).mean():.1f}%")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_path", required=True)
    p.add_argument("--causal_path", required=True)
    p.add_argument("--valid_set", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_neighbors", type=int, default=10)
    p.add_argument("--device", type=str, default="cuda:1")
    main(p.parse_args())
