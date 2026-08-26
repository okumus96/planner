"""b*-swap GORSELLESTIRME (v3_latmoe): sahne BEV + taban plan + zorlanmis-karar planlari.

Her panel: ref-path adaylari (gri), komsu konumlari + GF top-1 future'lari (ince),
taban plan (siyah kalin, ilan edilen kararla) ve 4 zorlama (stop/accel/turn_l/turn_r)
karar-tutarli secimle. Kesikli cizgi = uyan mod yok (argmax'a dusuldu).

Kosum:
  python viz_bswap.py --pretrained_path training_log/normal/model_epoch_19_valADE_1.6487.pth \
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
from GameFormer.decision_labels import (decision_labels, LON4_MAP, LAT5V_MAP,
                                        LON4_CLASSES, LAT5V_CLASSES)
from train_planner import read_batch, extract_neighbor_top1_futures, freeze_gameformer
from eval_bswap import plan_with_heading, start_speed

LON4_T = torch.tensor(LON4_MAP)
LAT5_T = torch.tensor(LAT5V_MAP)
# zorlamalar: (eksen, sinif-indeksi, renk, ad)
FORCES = [('lon', 0, '#c62828', 'force stop'),
          ('lon', 2, '#2e7d32', 'force accel'),
          ('lat', 0, '#1565c0', 'force turn_left'),
          ('lat', 1, '#ef6c00', 'force turn_right')]


def cc_pick(trajF, scoreF, ref_b, axis, cls):
    """Karar-tutarli secim: zorlanan sinifa uyan modlar icinden en yuksek skorlu.
    Doner: (plan [80,2], uydu_mu)."""
    modes = trajF[0, 0][..., :2]                                   # [M,80,2]
    M = modes.shape[0]
    rl, rt = decision_labels(plan_with_heading(modes).cpu(), ref_b.cpu().expand(M, -1, -1, -1))
    fold = LON4_T[rl] if axis == 'lon' else LAT5_T[rt]
    ok = (fold == cls)
    sc = scoreF[0, 0].cpu().clone()
    if ok.any():
        sc[~ok] = -1e9
        return modes[int(sc.argmax())].cpu(), True
    return modes[int(scoreF[0, 0].argmax())].cpu(), False


@torch.no_grad()
def main(a):
    dev = a.device
    gf = GameFormer(encoder_layers=3, decoder_levels=2, neighbors=a.num_neighbors)
    gf.load_state_dict(torch.load(a.pretrained_path, map_location=dev))
    gf = gf.to(dev); freeze_gameformer(gf)
    m = CausalPlanner(layers=1, modes=6, nbr_enrich=2, ego_residual=0, gate_channels=1,
                      typed_kv=1, dod_meta=1, lat_moe=1, num_lon=4, num_lat=5).to(dev)
    miss, unexp = m.load_state_dict(torch.load(a.causal_path, map_location=dev), strict=False)
    assert not miss and not unexp, f"ckpt uyusmazligi: {miss} {unexp}"
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
        base = out['traj'][0, 0][..., :2]
        plan0 = base[int(out['score'][0, 0].argmax())].cpu()
        if float(start_speed(plan0[None])[0]) < a.min_v0:          # duruk sahneler sikici
            continue

        fig, ax = plt.subplots(figsize=(7, 8))
        for r in range(rp.shape[1]):                               # ref-path adaylari
            path = rp[0, r, :, :2].cpu()
            if path.abs().sum() > 0:
                ax.plot(path[:, 1], path[:, 0], color='0.85', lw=5, zorder=0)
        nbr = inp['neighbor_agents_past'][0, :a.num_neighbors, -1, :2].cpu()
        for j in range(nbr.shape[0]):                              # komsular + GF future
            if nbr[j].abs().sum() == 0:
                continue
            ax.plot(t1[0, j, :, 1].cpu(), t1[0, j, :, 0].cpu(), color='0.6', lw=0.8, zorder=1)
            ax.plot(nbr[j, 1], nbr[j, 0], 's', color='0.35', ms=7, zorder=2)
        ax.plot(0, 0, '^', color='k', ms=11, zorder=5)             # ego
        ax.plot(plan0[:, 1], plan0[:, 0], color='k', lw=2.8, zorder=4,
                label=f"taban: ({LON4_CLASSES[b_lon]}, {LAT5V_CLASSES[b_lat]})")

        f_cas, ego_c = out['f_cas'], out['ego_clean']
        for axis, cls, col, name in FORCES:
            fl = torch.full((1,), cls if axis == 'lon' else b_lon, dtype=torch.long, device=dev)
            ft = torch.full((1,), cls if axis == 'lat' else b_lat, dtype=torch.long, device=dev)
            if axis == 'lon':
                fl[:] = cls
            trajF, scoreF = m.head(f_cas, ego_c, (fl, ft))
            plan, ok = cc_pick(trajF, scoreF, rp[0:1], axis, cls)
            ax.plot(plan[:, 1], plan[:, 0], color=col, lw=1.8, ls='-' if ok else '--',
                    zorder=3, label=name + ('' if ok else ' (uyan mod yok)'))

        ax.set_aspect('equal'); ax.invert_xaxis()                  # +y = sol; solda gorunsun
        ax.set_title(f"sahne {si} — b*-swap, karar-tutarli secim (v3_latmoe e12)")
        ax.legend(loc='upper right', fontsize=8)
        ax.set_xlabel('y [m] (sol +)'); ax.set_ylabel('x [m] (ileri)')
        fig.tight_layout()
        fig.savefig(f"{a.out_dir}/bswap_scene{si:04d}.png", dpi=130)
        plt.close(fig)
        done += 1
        print(f"[viz] sahne {si}: ilan=({LON4_CLASSES[b_lon]},{LAT5V_CLASSES[b_lat]})")
        if done >= a.num_scenes:
            break
    print(f"{done} panel -> {a.out_dir}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_path", required=True)
    p.add_argument("--causal_path", required=True)
    p.add_argument("--valid_set", required=True)
    p.add_argument("--out_dir", type=str, default="viz_out/bswap")
    p.add_argument("--num_scenes", type=int, default=8)
    p.add_argument("--stride", type=int, default=97)
    p.add_argument("--min_v0", type=float, default=2.0)
    p.add_argument("--num_neighbors", type=int, default=10)
    p.add_argument("--device", type=str, default="cuda:1")
    main(p.parse_args())
