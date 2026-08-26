"""BINDING ANALIZ — "neden degismedi?": non-responder sahnelerin ortak paydasi.

Girdi: eval_binding_set.py'nin seti (JSON) + eval_binding_test.py'nin sonuclari (JSON).
Secilen mudahale icin (varsayilan: enc = TAM silme) model-frenliyor dilimini ikiye boler:
  responder     : argmax degisti (flip)
  non-responder : TAM SILMEYE ragmen argmax ayni
ve iki grubu elimizdeki her aciklayici degiskende karsilastirir:
  ego hizi (simdiki), ego'nun KENDI yavaslamasi (son 2 s hiz farki — atalet hipotezi),
  taban P(SLOW) (kararin keskinligi), taban sinif, ds/ttc/closing (hedef yakinligi),
  hedefin kutle payi, 2. sebep sayisi, isik. Ayrica non-responder'lardan BEV ornekleri cizer
  (eval_binding_set._viz yeniden kullanilir).

Kosum (GPU gerekmez):
  python eval_binding_analyze.py --set_json binding_set_v6.json \
    --test_json binding_test_results.json \
    --valid_set /home/lt-hta-ai4/ssd1/nuplan/processed_data/validation
"""
import argparse
import json
import numpy as np

from GameFormer.train_utils import DrivingData
from eval_binding_set import _viz, _spread


def ego_kin(ds_, scene):
    """(simdiki hiz, son 2 s hiz degisimi) — ego'nun kendi yavaslama trendi."""
    ego = np.asarray(ds_[scene][0])                     # [21, 7]
    v = np.hypot(ego[:, 3], ego[:, 4])
    return float(v[-1]), float(v[-1] - v[0])


def stats(rows, name, vals):
    v = np.asarray(vals, dtype=float)
    return f"  {name:34s} {np.mean(v):>8.2f}  (medyan {np.median(v):>6.2f})"


def main(a):
    sel = {s['scene']: s for s in json.load(open(a.set_json))}
    rows = json.load(open(a.test_json))
    ds_ = DrivingData(a.valid_set + "/*.npz", a.num_neighbors)

    br = [r for r in rows if r['model_braking'] and a.which in r]
    resp = [r for r in br if r[a.which]['flip']]
    nonr = [r for r in br if not r[a.which]['flip']]
    print(f"\n=== NON-RESPONDER ANALIZI — mudahale: {a.which} ===")
    print(f"model-frenliyor dilimi: {len(br)} sahne -> responder {len(resp)} | "
          f"non-responder {len(nonr)}")

    kin = {r['scene']: ego_kin(ds_, r['scene']) for r in br}
    for tag, grp in (("RESPONDER (karar degisti)", resp),
                     ("NON-RESPONDER (tam silmeye ragmen ayni)", nonr)):
        if not grp:
            continue
        print(f"\n  --- {tag}  n={len(grp)} ---")
        print(stats(grp, "ego hizi [m/s]", [kin[r['scene']][0] for r in grp]))
        print(stats(grp, "ego hiz degisimi son 2s [m/s]", [kin[r['scene']][1] for r in grp]))
        print(stats(grp, "taban P(SLOW)", [r['base_slow'] for r in grp]))
        print(stats(grp, "hedefin kutle payi", [r['mass_share'] for r in grp]))
        print(stats(grp, "hedef ds [m]", [sel[r['scene']]['ds'] for r in grp]))
        print(stats(grp, "hedef ttc [s]", [sel[r['scene']]['ttc'] for r in grp]))
        print(stats(grp, "hedef closing [m/s]", [sel[r['scene']]['closing'] for r in grp]))
        print(stats(grp, "2. sebep ajan sayisi", [r['other_caution_agents'] for r in grp]))
        print(stats(grp, "isik var (0/1)", [int(r['traffic_light']) for r in grp]))
        print(stats(grp, f"dP_SLOW ({a.which})", [r[a.which]['dslow'] for r in grp]))
        print(stats(grp, f"dv_end ({a.which}) [m/s]", [r[a.which]['dv'] for r in grp]))
        from collections import Counter
        print("   taban karar:", dict(Counter(r['base_lon'] for r in grp).most_common()))

    if nonr and a.viz:
        # non-responder BEV'leri: set kaydindaki gorsel alanlar + test bilgisi basliga
        panels = []
        for r in _spread(nonr, 9):
            s = dict(sel[r['scene']])
            s['lon_model'] = (f"{r['base_lon']} (SILINDI->AYNI, "
                              f"dP={r[a.which]['dslow']:+.2f})")
            s['mass_share'] = r['mass_share']
            panels.append(s)
        ns = argparse.Namespace(valid_set=a.valid_set, num_neighbors=a.num_neighbors,
                                viz=a.viz)
        _viz(panels, ns)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--set_json", type=str, default="binding_set_v6.json")
    p.add_argument("--test_json", type=str, default="binding_test_results.json")
    p.add_argument("--valid_set", required=True)
    p.add_argument("--which", type=str, default="enc", choices=["edge", "graph", "enc"])
    p.add_argument("--num_neighbors", type=int, default=10)
    p.add_argument("--viz", type=str, default="viz_out/nonresponders.png")
    p.add_argument("--device", type=str, default="cpu")
    main(p.parse_args())
