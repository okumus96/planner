"""force to_left/to_right GORSELLESTIRME: expert ne uretiyor, hakem ne diyor?

Sahne secimi: ilan edilen lat = none ve ego hareketli (gercek zorlama). Her panelde:
taban plan (siyah), force to_left (mavi) ve force to_right (turuncu) — karar-tutarli secimin
sectigi plan KALIN, ayni zorlamanin diger 5 modu SOLUK. Legend'da hakemin verdigi HAM 7-sinif
etiket + koridor-goreli kayma (m). Amac: %31 to_* compliance'inin anatomisini gozle gormek
(kayma var mi, hakem turn'e mi atiyor, LC olceginde mi).

Kosum:
  python viz_lat_forced.py --pretrained_path training_log/normal/model_epoch_19_valADE_1.6487.pth \
    --causal_path training_log/v3_latmoe/causal_epoch_12_minADE_0.7998.pth \
    --valid_set /home/lt-hta-ai4/ssd1/nuplan/processed_data/validation --device cuda:1
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from GameFormer.predictor import GameFormer
from GameFormer.causal_graph import CausalPlanner
from GameFormer.train_utils import DrivingData
from GameFormer.decision_labels import (decision_labels, LAT_CLASSES, LAT5V_MAP,
                                        LON4_CLASSES, LAT5V_CLASSES)
from train_planner import read_batch, extract_neighbor_top1_futures, freeze_gameformer
from eval_bswap import plan_with_heading, start_speed
from eval_lat_drift import drift

LAT5_T = torch.tensor(LAT5V_MAP)
FORCES = [(2, '#1565c0', 'force to_left', +1.0), (3, '#ef6c00', 'force to_right', -1.0)]


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
    ld = DataLoader(ds, batch_size=1, shuffle=False)
    os.makedirs(a.out_dir, exist_ok=True)

    done = 0
    for si, batch in enumerate(ld):
        if si % a.stride:
            continue
        inp, ef, nf, rp = read_batch(batch, dev)
        enc = gf.encoder(inp)
        t1, ns, _ = extract_neighbor_top1_futures(gf, enc, a.num_neighbors)
        out = m(enc, inp, num_agents=a.num_neighbors + 1, neighbor_futures=t1, neighbor_states=ns)
        b_lon = int(out['psi_lon_cas'][0].argmax())
        b_lat = int(out['psi_lat_cas'][0].argmax())
        plan0 = out['traj'][0, 0][int(out['score'][0, 0].argmax())][..., :2].cpu()
        if b_lat != 4 or float(start_speed(plan0[None])[0]) < a.min_v0:   # ilan=none + hareketli
            continue

        fig, ax = plt.subplots(figsize=(7, 8))
        for r in range(rp.shape[1]):
            path = rp[0, r, :, :2].cpu()
            if path.abs().sum() > 0:
                ax.plot(path[:, 1], path[:, 0], color='0.88', lw=5, zorder=0)
        nbr = inp['neighbor_agents_past'][0, :a.num_neighbors, -1, :2].cpu()
        for j in range(nbr.shape[0]):
            if nbr[j].abs().sum() == 0:
                continue
            ax.plot(t1[0, j, :, 1].cpu(), t1[0, j, :, 0].cpu(), color='0.65', lw=0.8, zorder=1)
            ax.plot(nbr[j, 1], nbr[j, 0], 's', color='0.4', ms=7, zorder=2)
        ax.plot(0, 0, '^', color='k', ms=11, zorder=6)
        ax.plot(plan0[:, 1], plan0[:, 0], color='k', lw=2.8, zorder=5,
                label=f"taban: ({LON4_CLASSES[b_lon]}, none)")

        M = out['traj'].shape[2]
        rep = rp.repeat_interleave(M, dim=0)
        for cls, col, name, sgn in FORCES:
            fl = torch.full((1,), b_lon, dtype=torch.long, device=dev)
            ft = torch.full((1,), cls, dtype=torch.long, device=dev)
            trajF, scoreF = m.head(out['f_cas'], out['ego_clean'], (fl, ft))
            modes = trajF[0, 0][..., :2].cpu()                            # [M,80,2]
            for k in range(M):                                            # soluk: tum modlar
                ax.plot(modes[k, :, 1], modes[k, :, 0], color=col, lw=0.9, alpha=0.28, zorder=3)
            _, rt7 = decision_labels(plan_with_heading(modes), rp.cpu().expand(M, -1, -1, -1))
            ok = (LAT5_T[rt7] == cls)
            sc = scoreF[0, 0].cpu().clone()
            if ok.any():
                sc[~ok] = -1e9
            k = int(sc.argmax())                                          # karar-tutarli secim
            dr = float((sgn * drift(modes.to(dev), rep)).view(M)[k])
            ax.plot(modes[k, :, 1], modes[k, :, 0], color=col, lw=2.2, zorder=4,
                    ls='-' if bool(ok[k]) else '--',
                    label=f"{name} → hakem: {LAT_CLASSES[int(rt7[k])]}, kayma {dr:+.1f} m")

        # zoom: plan cevresi (uzun koridor yanal farki gorunmez kiliyor)
        xmax = float(plan0[:, 0].max()) + 15.0
        ymax = max(12.0, float(plan0[:, 1].abs().max()) + 10.0)
        ax.set_xlim(ymax, -ymax)                     # invert: sol solda
        ax.set_ylim(-12.0, xmax)
        ax.set_aspect('equal')
        ax.set_title(f"sahne {si} — force to_left/to_right (v3_latmoe e12)")
        ax.legend(loc='lower left', fontsize=8)
        ax.set_xlabel('y [m] (sol +)'); ax.set_ylabel('x [m] (ileri)')
        fig.tight_layout()
        fig.savefig(f"{a.out_dir}/latforce_scene{si:04d}.png", dpi=130)
        plt.close(fig)
        done += 1
        print(f"[viz] sahne {si}")
        if done >= a.num_scenes:
            break
    print(f"{done} panel -> {a.out_dir}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_path", required=True)
    p.add_argument("--causal_path", required=True)
    p.add_argument("--valid_set", required=True)
    p.add_argument("--out_dir", type=str, default="viz_out/latforce")
    p.add_argument("--num_scenes", type=int, default=6)
    p.add_argument("--stride", type=int, default=31)
    p.add_argument("--min_v0", type=float, default=3.0)
    p.add_argument("--num_neighbors", type=int, default=10)
    p.add_argument("--device", type=str, default="cuda:0")
    main(p.parse_args())
