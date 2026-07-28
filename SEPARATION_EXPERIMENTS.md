# Separation Experiments Log

Amaç: causal agent graph'ta **uzak-alakasız ajan sızıntısını** çözmek (harita gate'i zaten iyi).
Taban: `gat-critfixes` @ 7c156da (nbrenrich2, m*-psi, man_lbl/DOD YOK).

Değerlendirme: `viz_causal.py` (uzak-ajan vakaları GÖRSEL) + `eval_remove_noncausal.py`
(Delta_high/Delta_low, corr, interactive). Kabul: uzak-alakasız ilgi belirgin azalır +
RemoveNonCausal en az korunur/artar + minADE ~korunur (~0.72).

## Deneyler

| # | config | minADE | mcas_peak | RemoveNonCausal (ratio / corr / interactive) | viz gözlemi |
|---|--------|--------|-----------|----------------------------------------------|-------------|
| B0 | nbrenrich2 (baseline) | 0.718 | 0.278 | 12.9x / 0.575 / 44.0x | karşı-yol pedestrian + uzak-şerit ajanına ilgi |
| E1 | + conflict_feats=1 | 0.713 | 0.261 | 13.9x / 0.529 / 21.9x | ❌ NEGATİF — panel-1 uzak yaya 0.25→0.44; lider vakalar iyi |
| E2 | + conflict_bias=1 (E1 üstüne) | TBD | TBD | TBD | TBD |

### E1 teşhisi (diag_conflict.py, 15 batch) — Mod B: attention conflict'i KULLANIYOR ama ÇOK ZAYIF
- corr(M_cas, conflict) negatif ama cılız: d_route −0.19, d_ego_aligned −0.16.
- Uzak ajan M_cas ort **0.097** vs yakın **0.113** — bastırma yok denecek kadar az. Top-1'lerin **%43'ü uzak**.
- Top-1 pedestrian'lar sistematik uzak (d yüksek) ama yine seçiliyor (M_cas 0.256) = kullanıcı vakası.
- **Sonuç:** feature-as-input zayıf (L_TRAJ güçlü bastırmayı ödüllendirmiyor) → **E2 = explicit bias.**

## Notlar
- **E1 = future-conflict edge features** (d_route, d_ego_aligned, d_ego_spatial, approaching).
  Ajan edge'ine eklenir (7→11); harita edge'i dokunulmaz. Toggle `--conflict_feats 1`.
- Fallback (E1 yetersizse): (a) conflict'i explicit attention-bias yap, (b) M_cas'a entropi/top-k seyreklik.
- Goal 2 (CLS-R>0.75, refiner'a dokunmadan) ertelendi — filtre çözülünce.
