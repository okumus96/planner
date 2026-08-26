"""Karar etiketi (lon x lat) offline dogrulayici — channels workflow'unun aynisi.

  --selftest          : sentetik bilinen-cevap sahneleri (esik/oncelik/fallback dogrulamasi)
  --data DIR          : npz seti uzerinde dagilim: lon/lat sayimlari, joint tablo,
                        eski 5-sinif maneuver_labels ile capraz tablo, nadir-sinif ornekleri
  --viz N --out PNG   : ornek sahneler (ref aday-0 + GT ego gelecegi + etiket basligi)

Egitime baglamadan ONCE kosulur: dagilim quickly/gently birlestirme kararini,
bos siniflari (reverse beklenen ~0) ve esik sanity'sini verir.
"""
import argparse
import glob
import json
import os
from collections import Counter, defaultdict

import numpy as np
import torch

from GameFormer.decision_labels import (decision_labels, LON_CLASSES, LAT_CLASSES,
                                        NUM_LON, NUM_LAT)
from train_planner import maneuver_labels, _MANEUVER

MAN_NAMES = {v: k for k, v in _MANEUVER.items()}


# ---------------- selftest ----------------

def _traj_from_speed(v_profile, y=None, yaw=None):
    """v_profile [80] (m/s) -> [80,3] ego-frame (x ileri). y/yaw verilmezse 0."""
    x = np.concatenate([[0.0], np.cumsum(v_profile[:-1] * 0.1)])
    y = np.zeros(80) if y is None else y
    yaw = np.zeros(80) if yaw is None else yaw
    return np.stack([x, y, yaw], axis=1).astype(np.float32)


def _straight_ref():
    """Aday-0 = x ekseni boyunca duz koridor [1,5,1200,6]."""
    rp = np.zeros((1, 5, 1200, 6), dtype=np.float32)
    rp[0, 0, :, 0] = np.linspace(-10.0, 110.0, 1200)
    return torch.from_numpy(rp)


def selftest():
    t = 0.1 * np.arange(80)
    cases = []  # (isim, traj [80,3], ref|None, beklenen_lon|None, beklenen_lat|None)

    cases.append(('maintain+none', _traj_from_speed(np.full(80, 10.0)), None, 'maintain', 'no_lateral'))
    v = np.maximum(8.0 - 4.0 * t, 0.0)
    cases.append(('stop_quickly', _traj_from_speed(v), None, 'stop_quickly', None))
    v = np.maximum(3.5 - 1.0 * t, 0.0)
    cases.append(('stop_gently', _traj_from_speed(v), None, 'stop_gently', None))
    v = np.where(t < 2.0, 12.0 - 3.0 * t, 6.0)
    cases.append(('slow_quickly', _traj_from_speed(v), None, 'slow_quickly', None))
    v = np.maximum(10.0 - 0.75 * t, 7.0)
    cases.append(('slow_gently', _traj_from_speed(v), None, 'slow_gently', None))
    v = np.minimum(2.0 + 2.0 * t, 10.0)
    cases.append(('accel_quickly', _traj_from_speed(v), None, 'accel_quickly', None))
    v = np.minimum(8.0 + 0.6 * t, 10.4)
    cases.append(('accel_gently', _traj_from_speed(v), None, 'accel_gently', None))
    tr = np.zeros((80, 3), dtype=np.float32)
    tr[:, 0] = 0.02 * np.arange(80) / 80.0            # milimetrik jitter
    cases.append(('remain_stopped', tr, None, 'remain_stopped', 'no_lateral'))
    tr = _traj_from_speed(np.full(80, 2.0)); tr[:, 0] *= -1.0
    cases.append(('reverse', tr, None, 'reverse', None))

    # sola donus: R=20 m ceyrek yay, sabit hiz ~3.9 m/s
    ang = np.linspace(0, np.pi / 2, 80)
    tr = np.stack([20.0 * np.sin(ang), 20.0 * (1 - np.cos(ang)), ang], axis=1).astype(np.float32)
    cases.append(('turn_left', tr, None, None, 'turn_left'))
    trr = tr.copy(); trr[:, 1] *= -1; trr[:, 2] *= -1
    cases.append(('turn_right', trr, None, None, 'turn_right'))

    # serit degisimi (smoothstep y 0->3.5, heading basta/sonda ~0), koridorlu VE fallback
    x = 8.0 * t
    u = np.clip((t - 2.0) / 4.0, 0.0, 1.0)
    ylc = 3.5 * u * u * (3 - 2 * u)
    yawlc = np.arctan2(np.gradient(ylc), np.gradient(x))
    lc = np.stack([x, ylc, yawlc], axis=1).astype(np.float32)
    cases.append(('lane_change_left (koridor)', lc, _straight_ref(), None, 'lane_change_left'))
    cases.append(('lane_change_left (fallback)', lc, None, None, 'lane_change_left'))
    lcr = lc.copy(); lcr[:, 1] *= -1; lcr[:, 2] *= -1
    cases.append(('lane_change_right (koridor)', lcr, _straight_ref(), None, 'lane_change_right'))

    yin = 1.0 * u * u * (3 - 2 * u)
    inl = np.stack([x, -yin, np.arctan2(np.gradient(-yin), np.gradient(x))], axis=1).astype(np.float32)
    cases.append(('inlane_right', inl, _straight_ref(), None, 'inlane_right'))

    # kavisli yol, serit degisimi YOK (fallback guard testi): curvature 0.02 < TURN_C_LO,
    # heading surekli donuyor -> fallback'te no_lateral'e dusmeli (yanlis LC YASAK)
    R = 50.0
    ang = 6.0 * t / R                                  # v=6 m/s, k=0.02
    tr = np.stack([R * np.sin(ang), R * (1 - np.cos(ang)), ang], axis=1).astype(np.float32)
    cases.append(('kavisli yol (LC degil)', tr, None, None, 'no_lateral'))

    n_pass = 0
    for name, traj, ref, want_lon, want_lat in cases:
        ef = torch.from_numpy(traj).unsqueeze(0)
        lon, lat = decision_labels(ef, ref)
        gl, gt_ = LON_CLASSES[int(lon[0])], LAT_CLASSES[int(lat[0])]
        ok = (want_lon is None or gl == want_lon) and (want_lat is None or gt_ == want_lat)
        n_pass += ok
        mark = 'PASS' if ok else 'FAIL'
        print(f'  [{mark}] {name:34s} -> lon={gl:15s} lat={gt_:18s}'
              f'  (beklenen: {want_lon or "-"}/{want_lat or "-"})')
    print(f'\nselftest: {n_pass}/{len(cases)} PASS')
    return n_pass == len(cases)


# ---------------- dataset stats ----------------

def run_stats(args):
    files = sorted(glob.glob(os.path.join(args.data, '*.npz')))
    if args.limit:
        files = files[:args.limit]
    lon_c, lat_c = Counter(), Counter()
    joint_c = Counter()
    cross = defaultdict(Counter)          # eski 5-sinif -> yeni lat sinifi
    examples = defaultdict(list)
    B = 64
    for i in range(0, len(files), B):
        chunk = files[i:i + B]
        efs, rps = [], []
        for f in chunk:
            d = np.load(f)
            efs.append(d['ego_agent_future'])
            rps.append(d['c_lat_candidates'])
        ef = torch.from_numpy(np.stack(efs)).float()
        rp = torch.from_numpy(np.stack(rps)).float()
        lon, lat = decision_labels(ef, rp)
        man = maneuver_labels(ef)
        for b, f in enumerate(chunk):
            ln, la = LON_CLASSES[int(lon[b])], LAT_CLASSES[int(lat[b])]
            lon_c[ln] += 1
            lat_c[la] += 1
            joint_c[(ln, la)] += 1
            cross[MAN_NAMES[int(man[b])]][la] += 1
            if len(examples[ln]) < 3:
                examples[ln].append(os.path.basename(f))
            if len(examples[la]) < 3:
                examples[la].append(os.path.basename(f))
    n = sum(lon_c.values())
    print(f'\n=== Karar etiketi dagilimi ({n} sahne, {args.data}) ===\n')
    print('LONGITUDINAL (ilk 4 s):')
    for c in LON_CLASSES:
        print(f'  {c:15s} {lon_c[c]:6d}  ({100.0 * lon_c[c] / max(n, 1):5.1f}%)')
    print('\nLATERAL (tam 8 s):')
    for c in LAT_CLASSES:
        print(f'  {c:18s} {lat_c[c]:6d}  ({100.0 * lat_c[c] / max(n, 1):5.1f}%)')
    print('\nJOINT (en sik 12):')
    for (ln, la), k in joint_c.most_common(12):
        print(f'  {ln:15s} x {la:18s} {k:6d}  ({100.0 * k / max(n, 1):5.1f}%)')
    print('\nESKI 5-SINIF -> YENI LATERAL (capraz tablo; turning_left satiri turn_left\'e '
          'dusmeli — sanity):')
    for m in _MANEUVER:
        row = cross[m]
        tot = sum(row.values())
        if tot == 0:
            continue
        top = ', '.join(f'{c}:{k}' for c, k in row.most_common(3))
        print(f'  {m:14s} (n={tot:5d}): {top}')
    print('\nNadir sinif ornekleri:')
    for c in LON_CLASSES + LAT_CLASSES:
        cnt = lon_c[c] + lat_c[c]
        if 0 < cnt <= max(5, n // 200) and examples[c]:
            print(f'  {c}: {examples[c]}')
    if args.json:
        with open(args.json, 'w') as fh:
            json.dump({'n': n, 'lon': dict(lon_c), 'lat': dict(lat_c),
                       'joint': {f'{a}|{b}': k for (a, b), k in joint_c.items()},
                       'cross_maneuver_to_lat': {m: dict(c) for m, c in cross.items()}}, fh, indent=2)
        print(f'\nJSON: {args.json}')
    return lon_c, lat_c


# ---------------- viz ----------------

def run_viz(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    files = sorted(glob.glob(os.path.join(args.data, '*.npz')))
    rng = np.random.RandomState(7)
    # cesitlilik icin: rastgele degil, farkli etiketlerden ornek toplamaya calis
    picked, seen_labels = [], set()
    for f in rng.permutation(files):
        d = np.load(f)
        ef = torch.from_numpy(d['ego_agent_future']).float().unsqueeze(0)
        rp = torch.from_numpy(d['c_lat_candidates']).float().unsqueeze(0)
        lon, lat = decision_labels(ef, rp)
        key = (int(lon[0]), int(lat[0]))
        if key in seen_labels and len(picked) < args.viz - 2:
            continue
        seen_labels.add(key)
        picked.append((f, d, int(lon[0]), int(lat[0])))
        if len(picked) >= args.viz:
            break
    cols = min(3, len(picked))
    rows = int(np.ceil(len(picked) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.0, rows * 4.0), squeeze=False)
    for i, (f, d, lo, la) in enumerate(picked):
        ax = axes[i // cols][i % cols]
        rp0 = d['c_lat_candidates'][0]
        m = np.abs(rp0[:, :2]).sum(-1) > 1e-6
        if m.sum() > 1:
            ax.plot(rp0[m, 0], rp0[m, 1], '-', color='#bbbbbb', lw=4, alpha=0.6,
                    zorder=1, solid_capstyle='round')
        ef = d['ego_agent_future']
        v = np.linalg.norm(np.diff(ef[:, :2], axis=0), axis=1) / 0.1
        sc = ax.scatter(ef[1:, 0], ef[1:, 1], c=v, cmap='viridis', s=6, zorder=3)
        ax.plot(0, 0, 's', color='#212121', ms=8, zorder=4)
        ax.axvline(ef[min(40, len(ef) - 1), 0], color='#e57373', lw=0.8, ls=':', zorder=2)
        ax.set_title(f'{LON_CLASSES[lo]}  +  {LAT_CLASSES[la]}', fontsize=9)
        ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02).set_label('v (m/s)', fontsize=7)
    for i in range(len(picked), rows * cols):
        axes[i // cols][i % cols].axis('off')
    fig.suptitle('Karar etiketleri — GT ego gelecegi (renk=hiz), gri=aday-0 koridoru, '
                 'kirmizi nokta cizgi=4s siniri', fontsize=10)
    fig.savefig(args.out, dpi=140, bbox_inches='tight')
    print(f'Saved: {args.out}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--selftest', action='store_true')
    p.add_argument('--data', type=str, default=None)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--viz', type=int, default=0)
    p.add_argument('--out', type=str, default='viz_out/decision_labels.png')
    p.add_argument('--json', type=str, default=None)
    args = p.parse_args()
    if args.selftest:
        ok = selftest()
        raise SystemExit(0 if ok else 1)
    if args.data and args.viz:
        run_viz(args)
    elif args.data:
        run_stats(args)
    else:
        p.print_help()
