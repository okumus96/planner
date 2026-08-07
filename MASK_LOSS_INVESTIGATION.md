# Causal/Confound Mask Loss — Investigation Log

Branch `gat-maskfix` (built on `gat-manlbl-dod` = `v1.1` + manlbl + DOD).
All experiments use `--nbr_enrich 2 --graph_layers 1` on the frozen GameFormer backbone
`training_log/normal/model_epoch_19_valADE_1.6487.pth`.

---

## TL;DR

We set out to make the causal mask `M_cas` sharper without breaking the confound branch.
Three things were **measured**, not argued:

1. **Causal-Planner's `soft_mask_loss` cannot work under either parameterisation.**
   Under softmax `complementarity` is *provably unsatisfiable*; under sigmoid it is
   *vacuously satisfiable*. `normalization` is identically zero under softmax and
   *contradicts* `complementarity` under sigmoid.
2. **The confound feature `f_cfd` collapses to a constant** (`var(f_cfd)/var(f_cas) ≈ 1.6e-4`
   across three independent runs). CP has the same structure and therefore, we predict,
   the same defect.
3. **Removing the mask loss entirely improves every metric.** The best causal graph we
   have came from *deleting* CP's term and replacing it with two terms that are
   dimensionally appropriate for a distribution.

The remaining, unsolved problem: the mask is now **decisive but not always correct** —
it drifts to distant, non-interacting agents. Nothing in the objective encodes
*interaction*, so no reweighting of the current terms can fix it. See
[Next steps](#next-steps).

---

## Starting point

Runs that predate this investigation, for reference. CLS-R is
`test14-random_reduced` (43 scenarios), closed-loop reactive, `--deploy refiner`.

| run | minADE | CLS-R | RNC hi/lo | matched | corr | `fcfd_var/fcas_var` |
|---|---|---|---|---|---|---|
| `nbrenrich2_full` | 0.7140 | 0.8111 | 12.3× | 4.9× | 0.58 | — |
| `manlbl_nbrenrich2_full` | 0.7131 | 0.8336 | 15.2× | 5.2× | 0.55 | — |
| `dod_manlbl_nbrenrich2_full` | 0.6803 | **0.8579** | 12.1× | 4.0× | 0.53 | 0.00012 |
| `dodrop50_manlbl_nbrenrich2` | 0.7046 | 0.8322 | 15.0× | 4.6× | 0.585 | — |

`dodrop50` was never simulated before this work; we ran its CLS here (0.8322) and it
**refuted** the hypothesis that DOD's CLS gain and its filter cost were separable —
pathway dropout gives back both together.

---

## Finding 1 — `comp` is unsatisfiable under softmax

`comp = MSE(M_cas + M_cfd, 1)`, applied elementwise over valid neighbours.

Both masks are softmax distributions, so each row sums to 1 and the **total mass is 2**.
`comp` demands an elementwise sum of 1 over *N* entries, i.e. total mass *N*.
Unsatisfiable for any `N ≠ 2`.

Worse, it is minimised at **uniform**: subject to `Σs = 2`, `Σ(s−1)²` is minimised at
`s_j = 2/N`, giving `(2/N − 1)²`. With ~10 neighbours plus the map relation this is ≈1.55.

**Measured** — `train_log.csv`, first vs last epoch, every run:

```
comp  1.5449 → 1.5456      excl  0.0021 → 0.0021      norm  0.0 → 0.0
```

Identical to four decimals across 20 epochs. `comp` is not merely dead — it is a
**uniformity prior fighting selectivity**, weighted 0.5.

### Controlled ablation

`maskabl05` (λ_mask = 0.5) vs `maskabl0` (λ_mask = 0), same seed, same config:

| | λ_mask = 0.5 | λ_mask = 0 |
|---|---|---|
| best minADE | 0.7140 (e13) | **0.7044** (e15) |
| `mcas_peak` | 0.2776 | **0.3337** (+21%) |
| `mcas_map_peak` | 0.2063 | **0.2431** (+18%) |
| comp | 1.5557 | 1.5621 |
| excl | 0.0007 | 0.0008 |

The control reproduces `nbrenrich2_full` exactly (0.7140 @ e13), so the differences are
real. Optimising the term changes its own value by **0.4%**: *a loss whose value is
independent of whether you optimise it is dead.*

RemoveNonCausal on `maskabl0` e15 confirms the gain is real selectivity, not cosmetics:
**hi/lo 15.7×** (vs 12.3×), Δhigh 0.5204 (vs 0.4607), and agents scoring `M_cas > 0.5`
went from 11 to 41.

---

## Finding 2 — `f_cfd` collapses

`fcfd_var / fcas_var`, measured with the collapse monitor added in this work:

| run | ratio |
|---|---|
| `maskabl0` | 0.00017 |
| `maskabl05` | 0.00016 |
| `dod_manlbl_nbrenrich2_full` | 0.00012 |

`f_cas` sits at ~2.1 (the LayerNorm scale); `f_cfd` at ~0.0004. **The confound feature is
the same vector for every scene.**

Cause: the objective asks `f_cfd` for exactly one thing — *be uninformative about the
manoeuvre* — and a constant vector satisfies `KL(uniform ‖ psi(f_cfd))` for free. The
branch has **no consumer**.

### CP has the same structure

Every use of `self_node_fea_confound` / `agent_node_fea_confound` in
`~/Causal-Planner/src/models/causal/planning_model.py`:

- lines 395–408 — construction / reshape
- lines 413–448 — LSTEM memory (state propagation, not a task)
- **line 460 — `decisioon_decoder(self_node_fea_confound)`** ← the only real consumer
- lines 452–457 — **commented out** (`agent_node_fea_causal + 0.2*agent_node_fea_confound`)

The neighbour-prediction auxiliary (`others_reg_loss`,
`lightning_trainer.py:193`) reads `agent_node_fea = agent_node_fea_causal`
(`planning_model.py:468`, used at `:519`) — the **causal** branch. CP regularises the
causal side and leaves the confound side with a single consumer.

---

## Experiments

### Step 2 — give `f_cfd` a job (reconstruction)

`[f_cas ; f_cfd] → f_all`, where `f_all` is the **ungated** mean-pooled scene summary
(no causal/confound split, no learned selection). `f_cas` is zeroed with `p = recon_drop`
so the target cannot be carried by `f_cas` alone; the target is detached so the main
aggregation is not distorted.

Flags: `--lambda_recon` (default 0), `--recon_drop` (default 0.5).

| vs baseline `dod_manlbl` | baseline | `step2_recon_nbrenrich2` |
|---|---|---|
| `fcfd_var/fcas_var` | 0.00012 | **0.216** (1800×) |
| minADE | 0.6803 | 0.6833 |
| RNC hi/lo | 12.1× | **12.7×** |
| RNC **matched** | 4.0× | **5.2×** |
| RNC corr | 0.53 | **0.530** |
| Δlow | 0.0414 | **0.0374** |
| `mcfd_peak` | 0.1699 | 0.1213 (uniform = 0.1205) |

**Gained:** the collapse is fixed, and `matched` 5.2× is the best value this lineage has
produced. **Cost:** `M_cfd` collapsed to *exactly uniform* — because `f_all` is a uniform
mean, the cheapest way to reconstruct it is a uniform mask. Visible in
`viz_out/manlbl/step2_recon_cfd_top1.png`: `maxA = 0.10–0.11` in all nine frames.

### Step 2b — per-agent task instead (neighbour futures)

`f_cfd → all neighbours' GT futures`, unweighted. This is CP's `agent_predictor` trick
applied to the branch CP left unregularised. Per-agent targets mean a uniform mask is
**not** optimal.

Flag: `--lambda_nbr` (default 0), run at 0.1.

| | recon(2) | `step2b_nbr_nbrenrich2` |
|---|---|---|
| minADE | 0.6833 | **0.6731** |
| `mcfd_peak / uniform` | 1.007× | **1.81×** |
| `fcfd_var/fcas_var` | 0.216 | **1.74** |
| RNC hi/lo | **12.7×** | 8.9× |
| RNC matched | **5.2×** | 3.4× |

**Gained:** best minADE, and the only lever that made `M_cfd` selective.
**Cost:** the causal mask regressed on every RemoveNonCausal metric.

### Step 3 — sigmoid gates

Replace `softmax` over neighbours with independent `sigmoid` gates, aggregating as
`Σ(g·msg) / clamp(Σg, min=1)`. Motivation: a distribution *forces* the model to nominate
someone even on an empty road; independent gates can say "nobody". And CP's
`comp`/`excl` become expressible.

Flags: `--gate {softmax,sigmoid}` (default softmax), `--lambda_budget`, `--budget_rho`,
`--budget_lo`.

| | recon(2) | `step3` (ceiling only) | `step3b` (band 0.10–0.25) |
|---|---|---|---|
| **comp** | 1.5569 (dead) | **0.0003 (live)** | 0.0004 |
| minADE | 0.6833 | 0.6814 | 0.6831 |
| `gcas_mean` | — | 0.053 | 0.101 |
| `gcas_frac05` | — | 0.000 | 0.000 |
| `mcas_peak` | 0.2527 | 0.1189 | 0.1715 |
| `mcfd_peak` | 0.1213 | 0.9662 | 0.9281 |
| `mcas_ent` | 0.908 | — | 0.929 |
| RNC hi/lo | **12.7×** | 7.0× | 5.4× |
| RNC matched | **5.2×** | 3.5× | 3.2× |
| RNC **corr** | **0.530** | 0.372 | 0.460 |

`comp` went from frozen at 1.5556 to 0.0003 — the falsification test passed. But:

- **Step 3 fell into a degenerate corner we had not guarded:** `g_cas ≈ 0.05`,
  `g_cfd ≈ 0.97`. *"Nothing is causal, everything is confound"* satisfies `comp`
  (sum ≈ 1) and `excl` (product ≈ 0) perfectly. The one-sided budget hinge only
  penalised **high** `g_cas`, so it pushed *toward* this corner.
  See `viz_out/manlbl/step3_sigmoid_cfd_top1.png` — every frame saturated at 0.97–1.00.
- **Root cause:** on the causal side `maxMap = 0.20–0.39` while `maxA = 0.03–0.21`.
  `f_cas` feeds itself from the **map**; it does not need agents, so nothing holds the
  agent mask up. Softmax's mass conservation had been doing that job for free.
- **Step 3b** (two-sided band) restored the level (`gcas_mean` 0.05 → 0.10) and
  recovered part of the calibration (corr 0.372 → 0.460), but not the ranking
  (hi/lo fell further). Predicted in advance as the middle of three outcomes.
- **Step 3c** (sigmoid + nbr + band, stopped at epoch 10): `mcfd_peak` 0.929 → 0.899,
  `mcas_ent` 0.929 → **0.986**. The hypothesis fails **structurally**: under `comp`,
  `g_cfd ≡ 1 − g_cas`, so `M_cfd` has no independent degrees of freedom and *no*
  auxiliary task on the confound side can differentiate it.

**Verdict:** sigmoid abandoned after three runs. `comp` is not unsatisfiable here, it is
**underdetermined** — it constrains only the per-agent *sum*, never *which* agent goes
where, so infinitely many solutions satisfy it and the cheapest is saturation.
Tuning λ cannot select among them.

**Diagnostic:** `cos(attn_cas, attn_cfd) = −0.18` (agents) / `−0.46` (map), with
`‖attn_cfd‖ ≈ 1.6–1.9× ‖attn_cas‖`. The model did not learn the exact mirror
`sigmoid(x)+sigmoid(−x)=1`; it saturated the confound logits instead.

### Step A — distribution-appropriate terms under softmax

Keep softmax, drop `comp`/`norm`, and add two terms that are dimensionally correct for
a distribution.

Flags: `--lambda_bc`, `--lambda_peak`, `--peak_tau` (all default 0 / 0.5).

**Bhattacharyya overlap** `BC = Σ_j √(M_cas[j]·M_cfd[j])`, range [0,1], **independent of N**.
Why `excl` misses it — computed on 4 neighbours:

| configuration | `excl` | `BC` |
|---|---|---|
| both uniform | 0.00391 | **1.000** |
| both near-uniform | 0.00420 | **0.999** |
| both peaked, **same** agent | **0.13051** | **1.000** |
| both 0.9 on same agent | **0.16403** | **1.000** |
| peaked, **different** agents | 0.00091 | 0.512 |
| cas peaked, cfd spread | 0.00048 | 0.564 |

`excl` is **not** blind in general — it catches double-peaks well. It is blind
**in the flat regime**, where its scale collapses as `1/N⁴` (1e-4 at N=10, 3.9e-7 at
N=40) while `BC` stays exactly 1.0 at any N. Our models sit in exactly that regime
(`mcas_ent = 0.908`), which is why `excl` never moved. BC should be seen as a
**complement** to `excl`, not a replacement.

**Entropy hinge** `relu(H(M_cas)/log n − τ)`, applied to the *head-averaged* entropy —
penalising the average forces heads to both sharpen **and agree** (measured gap:
`mcas_ent` 0.908 vs `mcas_headent` 0.513). A hinge, not `−H`, so "two or three agents
matter" is free and only "all ten equally" is charged.

| | recon(2) | **stepA λ_bc=0.2** | stepA λ_bc=0.05 |
|---|---|---|---|
| minADE | **0.6833** | 0.7065 | 0.7014 |
| `mcas_ent` | 0.908 | **0.194** | **0.188** |
| `mcas_peak` | 0.253 (2.1× unif) | **0.807 (6.7×)** | 0.813 (6.8×) |
| `mcfd_peak / uniform` | 1.005× | **3.72×** | 1.78× |
| `fcfd_var/fcas_var` | 0.216 | 0.305 | 0.230 |
| Δhigh | 0.4751 | 0.4658 | 0.4318 |
| Δlow | 0.0374 | **0.0001** | 0.0012 |
| hi/lo | 12.7× | 3889× | 353× |
| matched | 5.2× | 21.1× | 13.8× |
| corr | **0.530** | 0.513 | 0.486 |
| `M_cas > 0.5` | n = 9 | **n = 497** | n = 502 |

λ_bc changes the causal mask **not at all** (0.194 vs 0.188) — the result is robust to
that knob; it only controls `M_cfd`. The peaking comes from the hinge, the separation
from BC.

**Gained:** the mask is finally *decisive*. 91% of maximum entropy → 19%; confident
assignments 9 → 497; the confound mask is no longer uniform (1.005× → 3.72×).
`viz_out/manlbl/stepA_bc02_cas_top1.png` shows one clear pick per scene at 0.60–0.99
with everything else visibly excluded.

**Cost:** minADE 0.6833 → 0.7065 (+3.4%), `cfdacc` 0.123 → 0.152.

---

## Methodological caveat — RemoveNonCausal saturates

`f_cas = Σ M_cas[j]·msg_j` is a convex combination, so `M_cas[j] ≈ 0` means agent *j*
is **literally not in the sum**. `Δlow → 0` is therefore *structurally guaranteed* for
any sufficiently peaked mask, and a model assigning 1.0 to a **randomly chosen** agent
would score `Δlow ≈ 0` and `hi/lo → ∞` just as well.

`Δlow = 0.0001 m` is a genuine and desirable property of the model. But `hi/lo = 3889×`
vs `353×` is noise in a denominator near zero — the ratio stops discriminating between
models. The same mechanism partly inflates `matched` and, because the mask is bimodal
(1270 agents near 0, 497 above 0.5), `corr` as well.

The two non-inflated readings for Step A are **`Δhigh` (0.4658 vs 0.4751) and
`corr` (0.513 vs 0.530)** — neither improved. So Step A bought **decisiveness and
interpretability, not ranking quality**.

Ratios should be reported with this caveat; `Δhigh` in absolute metres and the
distance-matched control are the defensible evidence.

---

## The unsolved problem

The mask is decisive but drifts to distant, non-interacting agents (visible in ~4 of 9
frames in `stepA_bc02_cas_top1.png`: far-corner vehicles selected at 0.88–0.99).

Nothing in the objective encodes **interaction**:

| term | what it rewards |
|---|---|
| `L_traj` | any agent whose state correlates with the ego future |
| `CE(psi, manlbl)` | any agent that predicts the ego manoeuvre class |
| BC / hinge | shape only — peaked, disjoint; never *which* |

A car stopped at the same red light 200 m away is an excellent predictor of "ego is
stationary". It costs `L_traj` nothing and helps `CE`. **No reweighting of the current
terms can fix this — the criterion is not in the loss at all.**

---

## Next steps

### A — Measure the drift (do first, no retraining)
For the top-1 selected agent per scene, compute: distance to ego at *t*=0, **minimum
distance between that agent's GT future and the ego's GT future** over the horizon, and
time-to-closest-approach. Report distributions for `recon(2)`, `stepA`, `nbr(2b)`.
Runs on existing checkpoints. Turns an impression over nine frames into a number over
472 scenes, and tests the observation that `nbr(2b)` looked better.

### B — Let the model express interaction
`gat-conflict` commit `299d22a` ("future-conflict features + `L_conflict`") sits directly
on `v1.1` and cherry-picks cleanly onto `gat-maskfix`. The attention edges currently carry
relative position and velocity but **nothing about whether the two futures cross**. If the
feature is absent from the input, the model cannot prefer the conflicting agent even if
the loss asked it to. Cheapest structural fix, code already written.

### C — Counterfactual supervision
Define causality operationally: for neighbour *j*, delete it and re-run the **frozen**
GameFormer; the shift in its predicted ego plan is a label for how causal *j* is. Train
`M_cas` against it. No human annotation. Caveats: *N* forward passes per sample
(subsample), and it makes RemoveNonCausal **circular** — a held-out counterfactual
variant would be needed for evaluation.

### D — Predicate-grounded semantic edges
Fully supervised edge types from the nuPlan predicate graph (see `PREDICATE_GROUNDED_PLAN.md`).
Breaks the "no per-agent labels" constraint but is the honest ceiling on correctness.

### Also outstanding
- CLS for `stepA` — is the +3.4% minADE a real closed-loop cost, or does a sharp mask help?
- State perturbation (`cp_manlbl_pert`: pure CLS 0.58 → 0.70) never tried on this lineage;
  orthogonal to the mask, so it cannot recreate this trade.

---

## Code changes

All additions are **opt-in**; defaults reproduce prior behaviour.

| flag | default | purpose |
|---|---|---|
| `--lambda_recon` | 0 | Step 2: `[f_cas;f_cfd] → f_all` reconstruction |
| `--recon_drop` | 0.5 | pathway dropout on `f_cas` in that head |
| `--lambda_nbr` | 0 | Step 2b: `f_cfd →` neighbour GT futures |
| `--gate` | `softmax` | Step 3: `softmax` \| `sigmoid` neighbour gating |
| `--lambda_budget` | 0 | Step 3: budget hinge weight |
| `--budget_rho` | 0.2 | budget band upper bound |
| `--budget_lo` | 0.0 | budget band lower bound (0 = one-sided) |
| `--lambda_bc` | 0 | Step A: Bhattacharyya overlap penalty |
| `--lambda_peak` | 0 | Step A: entropy hinge weight |
| `--peak_tau` | 0.5 | entropy hinge threshold (1 = uniform, 0 = one-hot) |

New metrics in `train_log.csv`: `fcfd_var`, `fcas_var`, `recon`, `nbr`, `budget`, `bc`,
`peak_hinge`, `gcas_mean`, `gcfd_mean`, `gcas_frac05`.

`norm` was removed from `l_mask` — identically zero under softmax, and inconsistent with
`comp` under sigmoid. Still logged as a metric.

`train.log` now records the full config (`vars(args)`). Older runs recorded only
batch size / lr / device, so their λ values had to be inferred.

`eval_metrics_from_ckpt.py` (new) recomputes validation metrics for an existing
checkpoint in ~10 s, so a newly added metric can be backfilled without a 2.1 h retrain.
`viz_causal.py` and `eval_remove_noncausal.py` gained `--gate`, and both load with
`strict=False` (printing what was missing) so pre-Step-2 checkpoints still evaluate.

---

## Artifacts

- **Diagram** — `manlbl` / DOD / trajectory-head wiring, v1.1 vs current:
  https://claude.ai/code/artifact/9869524b-9dc2-420e-aaa9-d3788ef69800
- **Visualisations** — `viz_out/manlbl/` (gitignored, on disk):
  `step2_recon_{cas,cfd}_top1.png`, `step2b_nbr_{cas,cfd}_top1.png`,
  `step3_sigmoid_{cas,cfd}_top1.png`, `step3b_band_{cas,cfd}_top1.png`,
  `stepA_bc02_{cas,cfd}_top1.png`, `dod_manlbl_nbrenrich2_{cas,cfd}_top1.png`
- **Training logs** — `training_log/{maskabl0,maskabl05,step2_recon,step2b_nbr,
  step3_sigmoid,step3b_band,step3c_band_nbr,stepA_bc_peak,stepA_bc005}_nbrenrich2/`
- **CP source references** — `~/Causal-Planner/src/models/`:
  `causal/planning_model.py:460,468,488,519` · `causal/lightning_trainer.py:193,298-330` ·
  `pluto/modules/hdgt_encoder.py:198,745,748` · `causal/layers/time_decoder.py:170`

## Reproduction

```bash
# Step A (best mask), softmax + recon + BC + entropy hinge
python train_planner.py --name stepA_bc_peak_nbrenrich2 \
  --train_set <train> --valid_set <valid> \
  --pretrained_path training_log/normal/model_epoch_19_valADE_1.6487.pth \
  --train_epochs 20 --batch_size 32 --weight_decay 0.01 \
  --nbr_enrich 2 --lambda_recon 1.0 \
  --lambda_mask 0 --lambda_bc 0.2 --lambda_peak 0.5 --peak_tau 0.5

# acceptance test
python eval_remove_noncausal.py \
  --pretrained_path training_log/normal/model_epoch_19_valADE_1.6487.pth \
  --causal_path training_log/stepA_bc_peak_nbrenrich2/causal_epoch_18_minADE_0.7065.pth \
  --valid_set <valid> --graph_layers 1 --nbr_enrich 2

# backfill a metric on an old checkpoint (~10 s, no retraining)
python eval_metrics_from_ckpt.py --pretrained_path <gf> --causal_path <ckpt> \
  --valid_set <valid> --graph_layers 1 --nbr_enrich 2
```
