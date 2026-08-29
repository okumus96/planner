"""LABEL IDENTITY analiz: eval_label_identity.py JSON'larindan kontrast tablolari.

Her (r_base, r_swap) cifti POST-HOC bir kontrast:
  S1 = tgt_only_<r_base>   S2 = tgt_only_<r_swap>   S3 = tgt_cut
Metrikler: p_slow (=P4[stop]+P4[slow]), |S1-S2|, |S1-S3|, KL(S1||S2)/KL(S1||S3) (4x5 ortak
dagilim = lon x lat bagimsiz -> KL toplanir), argmax degisim oranlari.
Filtre (on-kayitli): |p_slow(S3) - p_slow(S1)| >= 0.05.
"""
import argparse
import json
import os

import numpy as np

RES = "/home/lt-hta-ai4/GameFormer-Planner/results_label_identity"
BENIGN = 'adjacent_right'
BASES = ['follows', 'same_lane_ahead', 'onObservedCollisionCourseWith']
SWAPS = ['adjacent_right', 'sharesIntersectionWith']
LON4 = ['stop', 'slow', 'accel', 'maintain']


def load(tag):
    return json.load(open(os.path.join(RES, f"label_identity_{tag}.json")))


def p_slow(st):
    return st['P4'][0] + st['P4'][1]


def p_go(st):
    return st['P4'][2] + st['P4'][3]


def kl(a, b):
    a, b = np.asarray(a) + 1e-9, np.asarray(b) + 1e-9
    a, b = a / a.sum(), b / b.sum()
    return float((a * np.log(a / b)).sum())


def kl45(s1, s2):
    """4x5 ortak dagilim KL = KL_lon + KL_lat (psi_lon/psi_lat AYRI head -> bagimsiz carpim)."""
    return kl(s1['P4'], s2['P4']) + kl(s1['P5'], s2['P5'])


def boot(x, n=4000, seed=0):
    if len(x) < 2:
        return (float('nan'), float('nan'))
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    m = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def contrast(rows, who, rb, rw, filt=True):
    """who='tgt'|'ctl'. Doner: metrik sozlugu veya None."""
    k1, k2, k3 = f'{who}_only_{rb}', f'{who}_only_{rw}', f'{who}_cut'
    d12, d13, kl12, kl13, f12, f13, s1v, s2v, s3v, g12 = ([] for _ in range(10))
    for r in rows:
        S = r['states']
        a, b, c = S[k1], S[k2], S[k3]
        if filt and abs(p_slow(c) - p_slow(a)) < 0.05:
            continue
        s1v.append(p_slow(a)); s2v.append(p_slow(b)); s3v.append(p_slow(c))
        d12.append(p_slow(b) - p_slow(a)); d13.append(p_slow(c) - p_slow(a))
        kl12.append(kl45(a, b)); kl13.append(kl45(a, c))
        f12.append(int(np.argmax(a['P4']) != np.argmax(b['P4'])))
        f13.append(int(np.argmax(a['P4']) != np.argmax(c['P4'])))
        g12.append(p_go(b) - p_go(a))
    if not d12:
        return None
    lo, hi = boot(np.abs(d12))
    return dict(n=len(d12), s1=np.mean(s1v), s2=np.mean(s2v), s3=np.mean(s3v),
                d12=np.mean(d12), a12=np.mean(np.abs(d12)), a12_lo=lo, a12_hi=hi,
                d13=np.mean(d13), a13=np.mean(np.abs(d13)),
                kl12=np.mean(kl12), kl13=np.mean(kl13),
                f12=100 * np.mean(f12), f13=100 * np.mean(f13), g12=np.mean(g12))


def table(rows, who, tag, filt, only_natural=False):
    lab = "FILTRELI" if filt else "FILTRESIZ"
    nat = " | r_base hedefte GERCEKTEN yaniyor" if only_natural else ""
    print(f"\n### {tag} — {'HEDEF' if who == 'tgt' else 'KONTROL'} ({lab}{nat})")
    print(f"{'r_base':>30s} {'r_swap':>24s} {'n':>5s} {'S1':>6s} {'S2':>6s} {'S3':>6s} "
          f"{'|S1-S2|':>8s} {'95%CI':>15s} {'|S1-S3|':>8s} {'KL12':>7s} {'KL13':>7s} "
          f"{'flip12':>7s} {'flip13':>7s}")
    for rb in BASES:
        for rw in SWAPS:
            rs = rows
            if only_natural:
                rs = [r for r in rows if rb in r[f'{who}_rels']]
            g = contrast(rs, who, rb, rw, filt)
            if g is None:
                print(f"{rb:>30s} {rw:>24s}     0  —")
                continue
            print(f"{rb:>30s} {rw:>24s} {g['n']:>5d} {g['s1']:>6.3f} {g['s2']:>6.3f} "
                  f"{g['s3']:>6.3f} {g['a12']:>8.4f} [{g['a12_lo']:.4f},{g['a12_hi']:.4f}] "
                  f"{g['a13']:>8.4f} {g['kl12']:>7.3f} {g['kl13']:>7.3f} "
                  f"{g['f12']:>6.1f}% {g['f13']:>6.1f}%")


def gradient(rows, tag, rb='same_lane_ahead', filt=True):
    """Anlamsal-mesafe gradyani: ayni r_base'ten TUM r_swap'lere etki buyuklugu."""
    rels = [x for x in ALL_RELS if x != rb]
    print(f"\n### {tag} — ANLAMSAL MESAFE GRADYANI (r_base={rb}, {'filtreli' if filt else 'filtresiz'})")
    print(f"{'r_swap':>34s} {'n':>5s} {'|dP_slow|':>10s} {'ort dP':>8s} {'KL':>7s} {'flip%':>7s}")
    out = []
    for rw in rels:
        g = contrast(rows, 'tgt', rb, rw, filt)
        if g:
            out.append((g['a12'], rw, g))
    for a, rw, g in sorted(out, reverse=True):
        print(f"{rw:>34s} {g['n']:>5d} {g['a12']:>10.4f} {g['d12']:>+8.4f} "
              f"{g['kl12']:>7.3f} {g['f12']:>6.1f}%")


ALL_RELS = None

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs='+', default=['v2', 'v3', 'v4', 'v6'])
    a = ap.parse_args()
    data = {t: load(t) for t in a.tags}
    ALL_RELS = data[a.tags[0]]['rel_test']

    # ---------- 1. SAHNE MUHASEBESI ----------
    d0 = data[a.tags[0]]
    R = d0['rows']
    print("=" * 100)
    print("1. SAHNE MUHASEBESI")
    print("=" * 100)
    T = [r for r in R if r['has_target']]
    print(f"  tum validation sahnesi                                 {len(R):5d}")
    print(f"  ahead-tipi iliski yakan komsu YOK -> ATLANDI           {len(R) - len(T):5d}")
    print(f"  hedef bulundu (TEST POPULASYONU)                       {len(T):5d}")
    print(f"  kontrol ajani bulundu                                  {sum(r['has_ctrl'] for r in R):5d}")
    print(f"  hedefli sahnelerde kontrol de var                      {sum(r['has_ctrl'] for r in T):5d}")
    print("\n  hedef ajanin secimi modeller arasinda ayni mi:")
    for t in a.tags[1:]:
        same = sum(1 for x, y in zip(T, [r for r in data[t]['rows'] if r['has_target']])
                   if x['target'] == y['target'])
        print(f"    {a.tags[0]} vs {t}: {same}/{len(T)} ayni hedef")
    print("\n  on-kayitli filtre |p_slow(S3)-p_slow(S1)| >= 0.05, r_base basina:")
    for t in a.tags:
        rr = [r for r in data[t]['rows'] if r['has_target']]
        s = "  ".join(f"{rb[:14]}={sum(1 for r in rr if abs(p_slow(r['states']['tgt_cut']) - p_slow(r['states'][f'tgt_only_{rb}'])) >= 0.05):3d}"
                      for rb in BASES)
        print(f"    {t:3s}: {s}   (havuz {len(rr)})")
    print("\n  dilimler (model-bagimsiz):")
    print(f"    uzman frenlemis                                      {sum(r['expert_braked'] for r in T):5d}")
    print(f"    tek-sebep vitrin (1 ahead & fren & canli & isiksiz)  "
          f"{sum(1 for r in T if r['expert_braked'] and r['n_ahead'] == 1 and not r['traffic_light'] and (r['tgt_ttc'] <= 5.0 or (r['tgt_closing'] > 0 and abs(r['tgt_ds']) < 30))):5d}")
    print(f"    + uzman ACIL                                         "
          f"{sum(1 for r in T if r['expert_urgent'] and r['n_ahead'] == 1 and not r['traffic_light'] and (r['tgt_ttc'] <= 5.0 or (r['tgt_closing'] > 0 and abs(r['tgt_ds']) < 30))):5d}")

    # ---------- 2. MODEL BASINA TABLOLAR ----------
    print("\n" + "=" * 100)
    print("2. MODEL BASINA (r_base, r_swap) TABLOLARI")
    print("=" * 100)
    for t in a.tags:
        T = [r for r in data[t]['rows'] if r['has_target']]
        table(T, 'tgt', t, filt=False)
        table(T, 'tgt', t, filt=True)
        table(T, 'tgt', t, filt=True, only_natural=True)

    # ---------- 3. KONTROL (GURULTU TABANI) ----------
    print("\n" + "=" * 100)
    print("3. KONTROL AJANI — GURULTU TABANI")
    print("=" * 100)
    for t in a.tags:
        C = [r for r in data[t]['rows'] if r['has_ctrl']]
        Cs = [r for r in C if len(r['ctl_rels']) >= 1]     # GERCEK takas (enjeksiyon degil)
        Ci = [r for r in C if len(r['ctl_rels']) == 0]     # bos slota ENJEKSIYON
        print(f"\n  [{t}] kontrol ajani base'de >=1 iliski yakiyor: {len(Cs)}/{len(C)} "
              f"(0 yakan = enjeksiyon: {len(Ci)})")
        table(Cs, 'ctl', f"{t} kontrol-TAKAS", filt=False)
        table(Ci, 'ctl', f"{t} kontrol-ENJEKSIYON", filt=False)

    # ---------- 4. IC TABAN: anlamsal olarak ESDEGER etiket cifti ----------
    print("\n" + "=" * 100)
    print("4. IC TABAN — anlamsal ESDEGER cift (same_lane_ahead <-> follows) vs UZAK cift")
    print("   Ayni ajan, ayni yapisal islem; sadece ciftin anlamsal mesafesi farkli.")
    print("=" * 100)
    print(f"{'model':>6s} {'cift':>44s} {'n':>5s} {'|dP_slow|':>10s} {'95% CI':>18s} {'KL':>7s} {'flip%':>7s}")
    for t in a.tags:
        T = [r for r in data[t]['rows'] if r['has_target']]
        for rb, rw, lab in [('same_lane_ahead', 'follows', 'ESDEGER (ikisi de "onumdeki lider")'),
                            ('same_lane_ahead', 'adjacent_right', 'UZAK (caution -> benign)'),
                            ('same_lane_ahead', 'same_lane_behind', 'CELISKILI (onde -> arkada)')]:
            g = contrast(T, 'tgt', rb, rw, filt=True)
            if g:
                print(f"{t:>6s} {lab:>44s} {g['n']:>5d} {g['a12']:>10.4f} "
                      f"[{g['a12_lo']:.4f},{g['a12_hi']:.4f}] {g['kl12']:>7.3f} {g['f12']:>6.1f}%")

    # ---------- 5. GRADYAN ----------
    print("\n" + "=" * 100)
    print("5. ANLAMSAL MESAFE GRADYANI")
    print("=" * 100)
    for t in a.tags:
        gradient([r for r in data[t]['rows'] if r['has_target']], t)

    # ---------- 6. DORT-MODEL KARSILASTIRMA ----------
    print("\n" + "=" * 100)
    print("6. DORT MODEL — en bilgilendirici cift: (follows -> adjacent_right)")
    print("=" * 100)
    print(f"{'model':>6s} {'psi girdisi':>22s} {'n':>5s} {'S1':>6s} {'S2':>6s} {'S3':>6s} "
          f"{'|S1-S2|':>8s} {'95% CI':>18s} {'|S1-S3|':>8s} {'KL12':>7s} {'KL13':>7s} "
          f"{'flip12':>7s} {'flip13':>7s} {'taban':>7s}")
    ARCH = {'v2': 'havuzlanmis f_cas', 'v3': 'havuzlanmis f_cas',
            'v4': 'rel bloklari (672)', 'v6': 'rel bloklari + rel_ev'}
    for t in a.tags:
        T = [r for r in data[t]['rows'] if r['has_target']]
        C = [r for r in data[t]['rows'] if r['has_ctrl'] and len(r['ctl_rels']) >= 1]
        g = contrast(T, 'tgt', 'follows', BENIGN, filt=True)
        c = contrast(C, 'ctl', 'follows', BENIGN, filt=False)
        print(f"{t:>6s} {ARCH.get(t, '?'):>22s} {g['n']:>5d} {g['s1']:>6.3f} {g['s2']:>6.3f} "
              f"{g['s3']:>6.3f} {g['a12']:>8.4f} [{g['a12_lo']:.4f},{g['a12_hi']:.4f}] "
              f"{g['a13']:>8.4f} {g['kl12']:>7.3f} {g['kl13']:>7.3f} {g['f12']:>6.1f}% "
              f"{g['f13']:>6.1f}% {c['a12']:>7.4f}")

    # ---------- 7. DIKKAT KALDIRACI ----------
    print("\n" + "=" * 100)
    print("7. ETKI vs DIKKAT KALDIRACI (base'de sahnedeki toplam yanan girdi sayisi)")
    print("=" * 100)
    print(f"{'model':>6s} {'n_ent':>10s} {'n':>5s} {'|S1-S2|':>9s} {'|S1-S3|':>9s} "
          f"{'hedef kutle payi':>18s} {'tepe kutle':>11s} {'norm entropi':>13s}")
    for t in a.tags:
        T = [r for r in data[t]['rows'] if r['has_target']]
        for lo, hi in [(1, 2), (3, 5), (6, 10), (11, 999)]:
            g_ = [r for r in T if lo <= r['states']['base']['n_ent'] <= hi]
            if len(g_) < 3:
                continue
            g = contrast(g_, 'tgt', 'follows', BENIGN, filt=False)
            print(f"{t:>6s} {f'{lo}-{hi}':>10s} {g['n']:>5d} {g['a12']:>9.4f} {g['a13']:>9.4f} "
                  f"{np.mean([r['base_mass_share'] for r in g_]):>18.3f} "
                  f"{np.mean([r['states']['base']['peak'] for r in g_]):>11.3f} "
                  f"{np.mean([r['states']['base']['ent'] for r in g_]):>13.3f}")

    # ---------- 8b. ESLESMIS KONTROL: ayni filtre kontrol ajanina da uygulanir ----------
    print("\n" + "=" * 100)
    print("8b. ESLESMIS KONTROL — ayni on-kayitli filtre (|S1-S3|>=0.05) KONTROL ajanina da")
    print("    Hedefin filtreli sayilari yalnizca boyle karsilastirilabilir (kaldirac eslesir).")
    print("=" * 100)
    print(f"{'model':>6s} {'ajan':>22s} {'n':>5s} {'|S1-S2|':>9s} {'95% CI':>18s} "
          f"{'|S1-S3|':>9s} {'oran S2/S3':>11s} {'flip12':>7s}")
    for t in a.tags:
        D = data[t]['rows']
        for lab, rs, who in (('HEDEF', [r for r in D if r['has_target']], 'tgt'),
                             ('kontrol-TAKAS', [r for r in D if r['has_ctrl']
                                                and len(r['ctl_rels']) >= 1], 'ctl'),
                             ('kontrol-ENJEKSIYON', [r for r in D if r['has_ctrl']
                                                     and len(r['ctl_rels']) == 0], 'ctl')):
            g = contrast(rs, who, 'follows', BENIGN, filt=True)
            if g is None:
                print(f"{t:>6s} {lab:>22s}     0  (filtreyi gecen yok)")
                continue
            print(f"{t:>6s} {lab:>22s} {g['n']:>5d} {g['a12']:>9.4f} "
                  f"[{g['a12_lo']:.4f},{g['a12_hi']:.4f}] {g['a13']:>9.4f} "
                  f"{g['a12'] / max(g['a13'], 1e-9):>11.2f} {g['f12']:>6.1f}%")

    # ---------- 9. SEMANTIK AILE TESTI ----------
    print("\n" + "=" * 100)
    print("9. SEMANTIK AILE TESTI — etiketin ANLAMI mi, yoksa sadece 'degisti' mi?")
    print("   Ayni ajan, ayni yapisal islem (hep TEK girdi). Sahne basina:")
    print("     C = ort p_slow( only_{collision_course, sharesIntersection} )   [CAUTION]")
    print("     B = ort p_slow( only_{adjacent_right, adjacent_left, same_lane_behind} ) [BENIGN]")
    print("   delta = C - B. Etiket ailesi okunuyorsa delta > 0 ve kontrolden BUYUK olmali.")
    print("=" * 100)
    CAU = ['onObservedCollisionCourseWith', 'sharesIntersectionWith']
    BEN = ['adjacent_right', 'adjacent_left', 'same_lane_behind']

    def semantic(rows, who, filt_rb=None):
        d = []
        for r in rows:
            S = r['states']
            if filt_rb is not None:
                if abs(p_slow(S[f'{who}_cut']) - p_slow(S[f'{who}_only_{filt_rb}'])) < 0.05:
                    continue
            c = np.mean([p_slow(S[f'{who}_only_{x}']) for x in CAU])
            b = np.mean([p_slow(S[f'{who}_only_{x}']) for x in BEN])
            d.append(c - b)
        return d

    print(f"{'model':>6s} {'ajan/dilim':>28s} {'n':>5s} {'delta (C-B)':>12s} {'95% CI':>20s} "
          f"{'>0 olan %':>10s}")
    for t in a.tags:
        D = data[t]['rows']
        for lab, rs, who, frb in (
                ('HEDEF (tum 412)', [r for r in D if r['has_target']], 'tgt', None),
                ('HEDEF (filtreli)', [r for r in D if r['has_target']], 'tgt', 'follows'),
                ('HEDEF (vitrin 73)', [r for r in D if r['has_target'] and r['expert_braked']
                                       and r['n_ahead'] == 1 and not r['traffic_light']
                                       and (r['tgt_ttc'] <= 5.0 or (r['tgt_closing'] > 0
                                            and abs(r['tgt_ds']) < 30))], 'tgt', None),
                ('kontrol-TAKAS', [r for r in D if r['has_ctrl'] and len(r['ctl_rels']) >= 1],
                 'ctl', None),
                ('kontrol-ENJEKSIYON', [r for r in D if r['has_ctrl'] and len(r['ctl_rels']) == 0],
                 'ctl', None)):
            d = semantic(rs, who, frb)
            if not d:
                continue
            lo, hi = boot(d)
            print(f"{t:>6s} {lab:>28s} {len(d):>5d} {np.mean(d):>+12.4f} "
                  f"[{lo:+.4f},{hi:+.4f}] {100 * np.mean(np.array(d) > 0):>9.1f}%")

    # ---------- 10. ESLESMIS SEMANTIK TEST (ayni sahnede hedef vs kontrol) ----------
    print("\n" + "=" * 100)
    print("10. ESLESMIS SEMANTIK TEST — ayni sahnede hedef ve kontrol ajani")
    print("    delta_hedef - delta_kontrol (sahne-ici eslesme; sahne zorlugu sadelesir).")
    print("    >0 ve CI 0'i icermiyorsa: aile-semantigi HEDEFE OZGU, evrensel bir kayma degil.")
    print("=" * 100)
    print(f"{'model':>6s} {'n':>5s} {'d_hedef':>9s} {'d_kontrol':>10s} {'FARK':>9s} "
          f"{'95% CI (fark)':>21s} {'>0 %':>7s}")
    for t in a.tags:
        both = [r for r in data[t]['rows'] if r['has_target'] and r['has_ctrl']]
        dt = np.array(semantic(both, 'tgt'))
        dc = np.array(semantic(both, 'ctl'))
        df = dt - dc
        lo, hi = boot(df)
        print(f"{t:>6s} {len(df):>5d} {dt.mean():>+9.4f} {dc.mean():>+10.4f} {df.mean():>+9.4f} "
              f"[{lo:+.4f},{hi:+.4f}] {100 * np.mean(df > 0):>6.1f}%")

    # ---------- 8. VITRIN DILIMI ----------
    print("\n" + "=" * 100)
    print("8. VITRIN DILIMI (tek-sebep & uzman frenlemis & kinematik canli & isiksiz)")
    print("=" * 100)
    def show(r):
        return (r['expert_braked'] and r['n_ahead'] == 1 and not r['traffic_light']
                and (r['tgt_ttc'] <= 5.0 or (r['tgt_closing'] > 0 and abs(r['tgt_ds']) < 30)))
    print(f"{'model':>6s} {'r_base':>30s} {'r_swap':>22s} {'n':>4s} {'S1':>6s} {'S2':>6s} "
          f"{'S3':>6s} {'|S1-S2|':>8s} {'|S1-S3|':>8s} {'flip12':>7s}")
    for t in a.tags:
        T = [r for r in data[t]['rows'] if r['has_target'] and show(r)]
        for rb in ['follows', 'same_lane_ahead']:
            g = contrast(T, 'tgt', rb, BENIGN, filt=False)
            if g:
                print(f"{t:>6s} {rb:>30s} {BENIGN:>22s} {g['n']:>4d} {g['s1']:>6.3f} "
                      f"{g['s2']:>6.3f} {g['s3']:>6.3f} {g['a12']:>8.4f} {g['a13']:>8.4f} "
                      f"{g['f12']:>6.1f}%")
