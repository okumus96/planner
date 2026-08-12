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

**And then a fourth, which supersedes the other three in importance:**

4. **The mask does not select interacting agents.** Measured against ground-truth
   trajectories only — no model output involved — the highest-scored agent is
   **1.03–1.26× closer to the ego than a randomly chosen neighbour**, while the
   one-line heuristic "pick the nearest agent" scores **2.12×**. The sharpest model
   (Step A) sits at **1.03×**, i.e. **indistinguishable from random**.

Everything we improved — entropy, peak, overlap, collapse — is a property of the
mask's **shape**. None of it moved the mask's **content**. See
[Finding 3](#finding-3--the-mask-does-not-select-interacting-agents) and
[Next steps](#next-steps).

**And a fifth, from the closed-loop grid once it was completed:**

5. **CLS is blind to mask correctness — if anything it prefers a wrong mask.** `confI`, whose
   mask is at *exact chance* (1.01×, 93% of picks behind the ego), scores **0.8364**; `confH`,
   identical except for `L_conflict` and 1.9× better at finding interacting agents, scores
   **0.8059**. Across all twelve simulated runs there is no positive relationship between the
   two. Closed-loop score therefore **cannot validate the causal claim** and must be reported
   only as a no-regression guard. See
   [Finding 5](#finding-5--closed-loop-score-is-blind-to-mask-correctness).

**Current working model: `nbr(2b)` (`step2b_nbr_nbrenrich2`, epoch 15).** CP mask loss kept,
`f_cfd` alive (`cfdvar` 1.742 vs the baseline's 0.00012), mask non-degenerate (`mcfd_peak`
1.81× uniform), best minADE of the lineage (**0.6731**), CLS **0.8336** — about one scenario
below the untouched baseline.

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

## Finding 3 — the mask does not select interacting agents

*This is the most important measurement in the log. It reframes everything above.*

### Why a new test was needed

Every acceptance test we had asks the model about **itself**. RemoveNonCausal deletes an
agent and checks whether *the model's own plan* moves. For a strictly-gated head this is
circular: `f_cas = Σ M_cas[j]·msg_j`, so `M_cas[j] ≈ 0` means agent *j* is not in the sum
at all, and removing it cannot change anything. The test passes for **any** peaked mask,
including one that peaks on a randomly chosen agent.

`eval_interaction.py` never touches the model's output. It uses **ground-truth
trajectories only**, so it is independent evidence about the *selection*, not about the
gating mechanism.

### What is measured

For the agent the mask scores highest in each scene:

| column | definition | reads as |
|---|---|---|
| `d0` | distance from ego to that agent at *t* = 0, in metres (ego-frame, so just `‖pos‖`) | how far away it is **right now** |
| **`dmin`** | take the ego's GT future (80 steps, 8 s) and the agent's GT future; at each timestep measure the gap; keep the minimum | **"at their closest moment, how close do these two actually get?"** small = they genuinely interact |
| `path` | same, but ignoring time: minimum distance between *any* point on the ego's path and *any* point on the agent's path | "do their routes cross at all, even at different times?" A car that clears the junction 10 s ahead of us has small `path`, large `dmin` |
| `tmin` | the timestep at which `dmin` occurs, in seconds | when the encounter happens |
| `dmin < 5 m` | fraction of scenes whose `dmin` is under 5 m | share of picks that are genuine close encounters |
| `vs random` | `mean dmin(random) / mean dmin(top1)` | **1.00 = the mask's pick is as far away as a random neighbour** |
| `mean M_cas` | the score the model gave its own top pick | how confident it was while doing this |

### The three reference rows

| row | what it is | role |
|---|---|---|
| **top1** | agent with the highest `M_cas` | the thing under test |
| **random** | a uniformly random valid neighbour | **floor** — chance level |
| **near** | the physically closest agent at *t* = 0 | **bar** — a one-line heuristic with no learning |

Both references are necessary. Without `random` we cannot tell whether 1.24× is good;
without `near` we cannot tell whether a learned mask beats a trivial rule.

**Reading rule**

```
top1 ≈ random   ->  the mask knows nothing about interaction
top1 ≈ near     ->  the mask is just re-encoding distance
near  < top1    ->  the mask is worse than distance
top1  < near    ->  the mask carries information beyond distance   <- the goal
```

### Results (1104 validation scenes, identical scene set for all rows)

| | `d0` (m) | **`dmin`** (m) | `path` (m) | `tmin` (s) | `dmin<5m` | **vs random** | mean `M_cas` |
|---|---|---|---|---|---|---|---|
| **random** (floor) | 18.13 | 15.37 | 12.96 | 2.17 | 0.101 | 1.00× | — |
| **near** (trivial rule) | 8.28 | **7.25** | 5.87 | 1.73 | **0.455** | **2.12×** | — |
| `dod_manlbl` (baseline) | 16.21 | 12.40 | 8.34 | 2.74 | 0.155 | 1.24× | 0.255 |
| `recon(2)` | 16.29 | 12.65 | 8.83 | 2.66 | 0.173 | 1.21× | 0.251 |
| `nbr(2b)` | 15.41 | **12.24** | 8.44 | 2.55 | 0.146 | **1.26×** | 0.231 |
| **`stepA`** | 17.60 | 14.89 | 10.94 | 2.62 | 0.123 | **1.03×** | **0.809** |

### What it says

**1. No model is meaningfully above chance.** All sit at 1.21–1.26×, while "pick the
nearest agent" scores 2.12×. The learned mask captures **less than half** the interaction
that a one-line heuristic does.

**2. Step A is at chance (1.03×).** Sharpening the mask made the interaction alignment
*worse* (1.24× → 1.03×); `dmin<5m` fell from 0.155 to 0.123, essentially onto random's
0.101. The model became **confident about a criterion that is not interaction** — it
assigns `M_cas = 0.809` to an agent no more relevant than one drawn at random. This is
the concrete cost of optimising shape without content: decisiveness without correctness
is worse than indecision, because the model now confidently excludes everything else.

**3. `nbr(2b)` is nominally best but the effect is not real.** 1.26× vs the baseline's
1.24×. The impression from nine visualisation frames does not survive 1104 scenes.

**4. The mask is not distance either.** `top1 d0 = 16.2 m` vs `near 8.3 m`, and the two
pick the same agent only **16%** of the time. So it is neither distance nor interaction —
it is whatever happens to help `L_traj` and `CE`.

**5. The opportunity is large.** A neighbour comes within 5 m of the ego in **45.5%** of
scenes. The models find it in 12–17%.

### Consequences for the earlier results

Every acceptance test above has to be re-read:

- **RemoveNonCausal passing** is circular, as argued. We now know the selected agent is
  usually not interacting, and the test still passes with `hi/lo` up to 3889×.
- **The ROAR distance-matched control passing** was never evidence of causality. It shows
  the mask is not re-encoding distance — which is trivially true of a mask that is near
  random on interaction. **"Not distance" ≠ "causal".**
- **`mcas_ent` 0.908 → 0.194, `mcas_peak` 6.7× uniform, `bc` 1.0 → 0.03** are all *shape*
  statistics. This measurement shows none of them implied anything about content.

### Honest caveat

`dmin` is a **proxy**, not ground truth. An agent far away but on a collision course can
be causal; a parked car 3 m away need not be. So we cannot declare the 1.2× models
"wrong" from this alone.

Step A's **1.03×** is not subject to that defence: a genuinely causal selection cannot
correlate with physical interaction at chance level.

### The target this gives us

`top1 vs random` must move from 1.03–1.26× toward — and ideally past — `near`'s **2.12×**.
Beating the distance heuristic *on the interaction metric* is the real test of the claim
that `M_cas` carries information beyond geometry.

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

The mask is decisive but does not select interacting agents — visible in ~4 of 9 frames
of `stepA_bc02_cas_top1.png` (far-corner vehicles at 0.88–0.99), and **quantified** in
[Finding 3](#finding-3--the-mask-does-not-select-interacting-agents): 1.03× vs random for
Step A, against 2.12× for a nearest-agent heuristic.

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

## Finding 4 — conflict supervision: what the runs are, and what they showed

### Run naming

Every run below sits on `gat-maskfix` with `--nbr_enrich 2 --graph_layers 1 --lambda_recon 1.0`
and inherits `manlbl` + DOD from `gat-manlbl-dod`. They differ only in the flags listed.

| name | log dir | conflict corridor | `L_conflict` | BC + hinge |
|---|---|---|---|---|
| **dod_manlbl** | `dod_manlbl_nbrenrich2_full` | — | — | — |
| **recon(2)** | `step2_recon_nbrenrich2` | — | — | — |
| **nbr(2b)** | `step2b_nbr_nbrenrich2` | — | — | — (uses `--lambda_nbr 0.1`) |
| **stepA** | `stepA_bc_peak_nbrenrich2` | — | — | `λ_bc 0.2`, `λ_peak 0.5` |
| **conf_only** | `conf_only_nbrenrich2` | ego position at *t*=0 (**buggy**) | 0.1 | — |
| **confA** | `confA_full_nbrenrich2` | ego position at *t*=0 (**buggy**) | 0.1 | `λ_bc 0.2` |
| **confB** | `confB_refpath_nbrenrich2` | lattice ref path, reach floor 10 m | 0.1 | — |
| **confB+BC** | `confa_refpath_full_nbrenrich2` | lattice ref path, reach floor 10 m | 0.1 | `λ_bc 0.1` |
| **confD** | `confD_gfcorridor_nbrenrich2` | **frozen GF ego plan** | 0.1 | `λ_bc 0.1` |
| **confE** | `confE_turnfix_nbrenrich2` | ref path + **arc-walk** `d_ego_aligned`, floor 30 m | 0.1 | `λ_bc 0.1` |
| **confF** | `confF_arcwalk_ref_nbrenrich2` | ref path + arc-walk, floor back to **10 m** | 0.1 | `λ_bc 0.1` |

The three corridor definitions, in order of discovery:

1. **`conf_only` / `confA`** — `d_ego_spatial = min_t ‖nbr_future(t)‖`, i.e. distance to where the ego
   *is now*. A car **behind** the ego drives through that point, so it scored a near-zero penalty and
   was selected; a car ahead moved away from it and was penalised. Rear-selection rate 0.62.
2. **`confB` / `confB+BC`** — distance to the **lattice planner reference path**
   (`c_lat_candidates[:, 0]`, produced by the same `get_candidate_paths` the simulation planner uses,
   precomputed in the npz), truncated to `ego_speed × 8 s`. Rear-selection 0.62 → **0.41**.
   Candidate 0 is the ego's own lane in 71.5% of scenes; even when it isn't, it lies 0.67 m from the
   ego's GT future versus 0.60 m for the oracle-best candidate — the choice is not the bottleneck.
3. **`confD`** — the frozen GameFormer's own predicted ego plan (`inter[:, 0]`, free from the decoder
   call we already make, not GT). Tried because the ref path is a geometric line while the GF plan
   carries turns and speed. **Worse** on every causal metric; also couples the feature to the frozen
   backbone, which conflicts with presenting GF as non-reactive prediction.

### Two things the runs corrected in our own understanding

**The rear bias comes from BC + hinge, not from `L_conflict`.** We initially attributed it to the
buggy corridor. Measuring `stepA` (BC + hinge, *no* conflict) gives a rear-selection rate of
**0.871** — the worst of any run, against 0.288 for plain `recon(2)`. `L_conflict` *repairs* it
(0.871 → 0.62 with the buggy corridor, → 0.41 with the ref path).

**Our "misses the lead car" finding was largely a broken metric.** Defining the lead vehicle by
ego-frame lateral offset (`|y| < 2.5 m`) breaks on curves, where a same-lane car has large `|y|`.
It gave a lead-pick rate of 0.258. With a curve-safe definition (within 2 m of the reference path
and ahead) the rate is **0.548**. Three fix hypotheses — `d_route` source, removing `d_ego_aligned`,
removing the reach truncation — were all tested against the broken metric, which is why none of
them moved it.

---

## Results

### Open-loop and mask shape

`unif` = 0.1205 throughout; `mcas_ent` is normalised (1 = uniform, 0 = one-hot).

| run | minADE | `mcas_ent` | `mcas_peak` | `mcfd_peak` | `cfdvar` | casacc |
|---|---|---|---|---|---|---|
| dod_manlbl | 0.6803 | 0.898 | 0.257 | 0.170 | **0.00012** | 0.902 |
| recon(2) | 0.6833 | 0.908 | 0.253 | 0.121 | 0.216 | 0.894 |
| nbr(2b) | **0.6731** | 0.923 | 0.233 | 0.218 | **1.742** | 0.902 |
| stepA | 0.7065 | 0.194 | 0.807 | 0.448 | 0.305 | 0.888 |
| conf_only | 0.6918 | 0.327 | 0.708 | 0.114 | 0.213 | 0.892 |
| confA | 0.7006 | 0.184 | 0.827 | 0.392 | 0.296 | 0.891 |
| confB | 0.6822 | 0.292 | 0.735 | 0.114 | 0.181 | 0.889 |
| confB+BC | **0.6727** | 0.167 | 0.839 | 0.280 | 0.234 | 0.888 |
| confD | 0.6768 | 0.173 | 0.827 | 0.293 | 0.243 | 0.897 |
| confE | 0.6977 | **0.150** | **0.861** | 0.278 | 0.227 | 0.883 |
| **confF** | **0.6732** | 0.148 | 0.859 | 0.283 | 0.217 | **0.893** |
| confH | 0.6934 | 0.124 | 0.827 | **0.914** | **0.00009** | 0.899 |
| **confI** | 0.6997 | 0.169 | 0.794 | **0.923** | **0.00005** | 0.884 |

`confH` and `confI` are the two runs with `λ_recon = 0`. Both drive `mcfd_peak` past **0.91**
(one-hot) and `cfdvar` to ~5e-5 (dead). With BC on and **no consumer for `f_cfd`**, BC is the only
gradient reaching `M_cfd`; under softmax's full support its residual scales with the number of
active entries, so descent walks monotonically to the single-entry corner with nothing to oppose
it. Compare `recon(2)` (`mcfd_peak` 0.121 = uniform) and `nbr(2b)` (0.218 = 1.81× uniform):
**the consumer, not the weight on BC, decides the concentration.**

### Interaction check (model-independent, GT trajectories only)

References: `random` = chance (`dmin` 15.37 m, `dmin<5m` 0.101, rear 0.462);
`near` = nearest-agent heuristic (`dmin` 7.25 m, `path` 5.87 m, `dmin<5m` 0.455, rear 0.518, **2.12×**).

| run | **vs random** | `dmin` | `path` | `dmin<5m` | **rear** | top1==near |
|---|---|---|---|---|---|---|
| dod_manlbl | 1.24× | 12.40 | 8.34 | 0.155 | 0.307 | 0.163 |
| recon(2) | 1.21× | 12.65 | 8.83 | 0.173 | **0.288** | 0.188 |
| nbr(2b) | 1.26× | 12.24 | 8.44 | 0.146 | 0.373 | 0.185 |
| stepA | **1.03×** | 14.89 | 10.94 | 0.123 | **0.871** | 0.139 |
| conf_only | 1.88× | 8.17 | **4.12** | 0.395 | 0.621 | 0.486 |
| confA | 1.84× | 8.37 | 4.26 | 0.380 | 0.624 | 0.477 |
| **confB** | **1.90×** | 8.11 | 4.82 | 0.389 | 0.412 | 0.436 |
| confB+BC | 1.85× | 8.29 | 4.82 | 0.382 | 0.412 | 0.434 |
| confD | 1.78× | 8.62 | 4.36 | 0.371 | 0.462 | 0.425 |
| confE | 1.89× | 8.14 | 4.89 | **0.399** | **0.409** | 0.442 |
| **confF** | **1.90×** | **8.10** | 4.85 | **0.406** | 0.418 | 0.438 |
| confH | 1.90× | 8.09 | 4.83 | 0.404 | 0.412 | 0.437 |
| confG (`λ_conf` 0.01) | 1.15× | 13.37 | 9.68 | 0.150 | 0.860 | 0.178 |
| **confI** | **1.01×** | 15.26 | 12.31 | 0.110 | **0.928** | 0.131 |

**`λ_conflict` does not scale down.** `confG` is `confF` with `λ_conf` 0.1 → 0.01 and nothing else.
Interaction falls 1.90× → **1.15×** and rear bias jumps 0.418 → **0.860**, i.e. almost all the way
back to `confI`'s BC-only failure mode (1.01×, 0.928). At one tenth the weight the conflict term
loses to BC outright. The term is effectively **binary at these magnitudes**: 0.1 or nothing.

Two axes move independently, and no run wins both:

- **`dmin` / `dmin<5m` — closeness.** Only `L_conflict` moves it (12.4 m → 8.1 m, 0.15 → 0.40).
  The mask family stays at 12.2–12.7 m.
- **rear fraction — direction.** Only the **mask family** gets it right: `recon(2)` **0.288** and
  `nbr(2b)` 0.373 beat every conflict run (0.41–0.62) *and* beat both references — `random` is
  0.462 and even `near` is 0.518, so half of all nearest agents are behind the ego. Low rear
  bias is therefore **not** obtainable by distance alone.

`confI` is the failure case of both: 0.928 rear and 15.26 m `dmin` — worse than random on
direction and at chance on closeness, with a mask peaked at 0.79.

`path` is the one axis where the learned mask **beats the trivial heuristic** (4.12–4.89 m against
`near`'s 5.87 m). On `dmin` it never does (8.1 m against 7.25 m). Consistent with `L_conflict`
supervising route conflict rather than time-of-arrival.

### RemoveNonCausal

Ratios are structurally inflated for peaked masks (see the caveat above); `corr` and `Δhigh` are the
readings that transfer.

| run | Δhigh | Δlow | hi/lo | matched | **corr** |
|---|---|---|---|---|---|
| dod_manlbl | 0.5018 | 0.0414 | 12.1× | 4.0× | 0.53 |
| recon(2) | 0.4751 | 0.0374 | 12.7× | 5.2× | 0.530 |
| nbr(2b) | 0.4318 | 0.0477 | 8.9× | 3.4× | 0.54 |
| stepA | 0.4658 | 0.0001 | 3889× | 21.1× | 0.513 |
| conf_only | 0.4871 | 0.0023 | 215× | 5.1× | 0.529 |
| confA | 0.4983 | 0.0003 | 1640× | 7.5× | 0.526 |
| confB | 0.5080 | 0.0012 | 418× | 6.7× | 0.546 |
| confB+BC | 0.5548 | 0.0000 | 25839× | 13.6× | 0.548 |
| **confE** | 0.5070 | 0.0000 | 17603× | 15.1× | **0.566** |
| confF | **0.5538** | 0.0001 | 7313× | 13.3× | 0.511 |
| confI | 0.4422 | 0.0030 | 146.7× | 8.0× | 0.471 |

`confI` is the clearest demonstration of the saturation caveat: it **passes** RemoveNonCausal at
146.7× while sitting at *exact chance* on interaction (1.01×). Passing this test proves the mask
is peaked, not that it is right. The mask-family runs — `dod_manlbl` 12.1×, `nbr(2b)` 8.9× — look
far worse on the ratio precisely because their masks are soft (`mcas_peak` ~2× uniform), so
`Δlow` does not collapse to zero and the ratio stays informative.

### Complete run configuration

Every run below is `manlbl` + DOD on the frozen backbone, with
`--nbr_enrich 2 --graph_layers 1 --gate softmax --modes 6 --lambda_kld 1.0 --lambda_ci 0.5`,
seed 3407, 20 epochs. They differ **only** in the columns shown. `λ_mask` is CP's
`comp + excl`; the `--lambda_mask` default is 0.5, so a run has the CP mask loss unless it
was explicitly passed `--lambda_mask 0`.

| run | log dir | `λ_mask` | `λ_recon` | `λ_nbr` | `λ_bc` | `λ_peak` | `λ_conf` | corridor | `aligned_mode` |
|---|---|---|---|---|---|---|---|---|---|
| **dod_manlbl** | `dod_manlbl_nbrenrich2_full` | **0.5** | 0 | 0 | 0 | 0 | 0 | — | — |
| **dodrop50** | `dodrop50_manlbl_nbrenrich2` | **0.5** | 0 | 0 | 0 | 0 | 0 | — | — *(+ DOD pathway dropout 0.5)* |
| **recon(2)** | `step2_recon_nbrenrich2` | **0.5** | **1.0** | 0 | 0 | 0 | 0 | — | — |
| **nbr(2b)** | `step2b_nbr_nbrenrich2` | **0.5** | 0 | **0.1** | 0 | 0 | 0 | — | — |
| **stepA** | `stepA_bc_peak_nbrenrich2` | 0 | 1.0 | 0 | **0.2** | 0.5 | 0 | — | — |
| stepA-bc005 | `stepA_bc005_nbrenrich2` | 0 | 1.0 | 0 | 0.05 | 0.5 | 0 | — | — |
| **conf_only** | `conf_only_nbrenrich2` | 0 | 1.0 | 0 | 0 | 0 | 0.1 | ego pos @ *t*=0 **(buggy)** | — |
| **confA** | `confA_full_nbrenrich2` | 0 | 1.0 | 0 | 0.2 | 0.5 | 0.1 | ego pos @ *t*=0 **(buggy)** | — |
| **confB** | `confB_refpath_nbrenrich2` | 0 | 1.0 | 0 | 0 | 0 | 0.1 | ref path, floor 10 m | straight |
| **confB+BC** | `confa_refpath_full_nbrenrich2` | 0 | 1.0 | 0 | 0.1 | 0.5 | 0.1 | ref path, floor 10 m | straight |
| **confD** | `confD_gfcorridor_nbrenrich2` | 0 | 1.0 | 0 | 0.1 | 0.5 | 0.1 | **frozen GF ego plan** | straight |
| **confE** | `confE_turnfix_nbrenrich2` | 0 | 1.0 | 0 | 0.1 | 0.5 | 0.1 | ref path, **floor 30 m** | **arc** |
| **confF** | `confF_arcwalk_ref_nbrenrich2` | 0 | 1.0 | 0 | 0.1 | 0.5 | 0.1 | ref path, floor 10 m | **arc** |
| confF-gf | `confF_arcwalk_gf_nbrenrich2` | 0 | 1.0 | 0 | 0.1 | 0.5 | 0.1 | frozen GF ego plan | arc |
| confG | `confG_lc001_nbrenrich2` | 0 | 1.0 | 0 | 0.1 | 0.5 | **0.01** | ref path | arc |
| **confH** | `confH_norecon_nbrenrich2` | 0 | **0** | 0 | 0.1 | 0.5 | 0.1 | ref path | straight |
| **confI** | `confI_bcpeak_only_nbrenrich2` | 0 | **0** | 0 | 0.1 | 0.5 | **0** | — | — |

The three families, stated plainly:

- **mask family** (`λ_mask 0.5`, no BC/peak/conflict): dod_manlbl, dodrop50, recon(2), nbr(2b)
- **BC/peak family** (`λ_mask 0`, no conflict): stepA, confI
- **conflict family** (`λ_mask 0`, `λ_conf > 0`): conf_only … confH

### Closed-loop (test14-random_reduced, 43 scenarios, reactive, `--deploy refiner`)

One scenario is worth `1/43 = 0.023` CLS. Differences smaller than that are **one scenario
or less** and must not be interpreted.

| run | family | CLS-R | coll | drivable | progress | TTC | comfort | route prog. |
|---|---|---|---|---|---|---|---|---|
| **dod_manlbl** | mask | **0.8579** | 1.000 | **0.947** | 0.974 | **1.000** | 0.974 | 0.787 |
| **confI** | BC/peak | **0.8364** | 1.000 | 0.921 | 0.974 | 0.974 | 0.947 | 0.810 |
| **nbr(2b)** | mask | **0.8336** | 1.000 | 0.921 | 0.974 | 0.947 | 0.947 | **0.822** |
| dodrop50 | mask | 0.8322 | 1.000 | 0.921 | 0.974 | **1.000** | 0.974 | 0.790 |
| **recon(2)** | mask | **0.8256** | 1.000 | 0.921 | 0.974 | 0.947 | 0.947 | 0.797 |
| confD | conflict | 0.8176 | 1.000 | 0.921 | 0.947 | 0.974 | 0.947 | 0.817 |
| confB | conflict | 0.8126 | 1.000 | 0.921 | 0.947 | 0.974 | 0.974 | 0.792 |
| confB+BC | conflict | 0.8126 | 1.000 | 0.921 | 0.947 | 0.974 | 0.974 | 0.792 |
| confH | conflict | 0.8059 | 1.000 | 0.921 | 0.947 | 0.947 | **0.895** | 0.816 |
| **confF** | conflict | **0.7940** | 1.000 | 0.921 | 0.921 | **1.000** | 0.947 | 0.803 |
| confE | conflict | 0.7935 | 1.000 | 0.921 | 0.921 | 0.974 | 0.921 | 0.811 |
| stepA | BC/peak | *never run* | | | | | | |

Collision score is **1.000 in every single run**. Driving-direction compliance is 1.000
everywhere and speed-limit compliance is 0.99+ everywhere; the entire CLS spread lives in
drivable-area, TTC, comfort and progress, i.e. in a handful of individual scenarios.

#### Bookkeeping correction — the "stepA = 0.8256" run was not stepA

The simulation on **2026-08-10 19:00:50** was logged in this file as stepA's CLS. It is not.
Its 16 metric parquets are **byte-identical** (`md5 33a55c78…`) to today's `recon(2)` run, and
the checkpoint that run needed `strict=False` for was missing `gate_bias` **and `nbr_head.*`** —
which is `step2_recon`'s signature. `stepA_bc_peak`'s checkpoint has **no missing keys at all**
(verified). So that run was `step2_recon`, measured twice with identical results.

Two consequences: **stepA still has no CLS score**, and the duplicate is a free determinism
check — the simulator reproduces bit-exactly across days for the same checkpoint.

#### `L_conflict` costs 0.031 CLS — measured, not inferred

`confH` and `confI` differ by **exactly one flag** (`--lambda_conflict 0.1` vs `0`). Everything
else — `λ_bc 0.1`, `λ_peak 0.5`, `λ_recon 0`, `λ_mask 0`, corridor, seed — is identical.

| | `λ_conf` | CLS-R | interaction | rear |
|---|---|---|---|---|
| confH | 0.1 | 0.8059 | **1.90×** | 0.41 |
| confI | 0 | **0.8364** | **1.01×** | 0.93 |

**−0.0305 CLS, +0.89× interaction.** This supersedes the earlier "~0.045" figure, which came
from averaging across families that also differed in `λ_recon` and corridor.

There is still **no clean recon on/off pair** anywhere in the table — every candidate pair also
changes `aligned_mode` or `λ_bc`. Recon's CLS effect is unmeasured, not zero.

`confE` bundled two changes (arc-walk **and** reach floor 10 m → 30 m) — a one-variable
violation. `confF` reverts only the floor: minADE 0.6977 → **0.6732** and interaction back
to **1.90×**, so the floor was the harmful half. Raising it lengthens the corridor, which
makes more agents count as "near it" and blunts the penalty's discrimination. **In CLS,
however, the fix did not transfer: confF 0.7940 vs confE 0.7935 — identical.** Open-loop
minADE and interaction both improved while closed-loop score did not move at all.

**Every `L_conflict` variant is below the no-conflict baseline, and the gap is concentrated in a
handful of turning scenarios.** Per-scenario, `confB+BC` zeroes two turns that the baseline scored
(`starting_right_turn` 0.862 → 0.000, `starting_left_turn` 0.798 → 0.000); those two account for
`(0.862 + 0.798)/43 = 0.039` of the 0.045 total drop, while 8 other scenarios *gain* 0.02–0.05.

`confE` was an attempt to fix exactly this — walk the ego along the reference path by arc length
instead of a straight line, and raise the reach floor to 30 m. It **backfired**: the two target
zeros did not return, a third turn was zeroed (0.796 → 0.000), two more regressed, and CLS fell to
0.7935. Straight/fast scenarios improved instead (`high_magnitude_speed` 0.797 → 0.987). Likely
cause: arc-walking at a *constant* current speed strands the imagined ego mid-turn, since a car
starting a turn is slow and will accelerate; the straight line at least pointed forward.

Note also that turning is a **chronic** weakness, not one `L_conflict` introduced: three turning
scenarios already score 0.000 in the baseline. 16 of the 43 scenarios are turns.

---

## Finding 5 — closed-loop score is blind to mask correctness

Both `step2_recon` and `step2b_nbr` carry CP's mask loss (`λ_mask 0.5`) and a **consumer** for
`f_cfd`. They were the only two checkpoints in the mask family with a live confound branch and
they had never been simulated. Running them completes the grid, and the grid says something
uncomfortable.

| run | mask loss | `f_cfd` alive | `mcas_peak` | interaction | rear | CLS-R |
|---|---|---|---|---|---|---|
| dod_manlbl | ✅ | ❌ 0.00012 | 0.257 (2.1× unif) | 1.24× | 0.307 | **0.8579** |
| **nbr(2b)** | ✅ | ✅ **1.742** | 0.233 (1.9× unif) | 1.26× | 0.373 | **0.8336** |
| **recon(2)** | ✅ | ✅ 0.216 | 0.253 (2.1× unif) | 1.21× | **0.288** | 0.8256 |
| **confI** | ❌ | ❌ 0.00005 | **0.796 (6.6×)** | **1.01×** | **0.928** | **0.8364** |
| confH | ❌ | ❌ 0.00009 | 0.827 (6.9×) | **1.90×** | 0.41 | 0.8059 |

`confI` has the **worst mask this project has produced** — 1.01× is exact chance, and 93% of its
picks are agents *behind* the ego, against a random baseline of 46%. It also scores the
**highest CLS of any non-baseline run** (0.8364), 0.003 above `nbr(2b)` — a twelfth of a
scenario, i.e. a tie.

Meanwhile `confH`, whose mask is 1.9× better at finding genuinely interacting agents, scores
**0.031 lower**.

**Across the whole table there is no positive relationship between mask correctness and CLS;
the sign is if anything negative.** This is Finding 3 confirmed in closed loop: the plan is
carried by the *aggregate content* of `f_cas`, not by which agent it points at. A convex
combination over a scene summary does not care much which neighbour dominates it.

Three consequences:

1. **CLS cannot validate the causal claim.** Optimising the mask against CLS actively selects
   for masks that are wrong. Causality must be validated by RemoveNonCausal + `eval_interaction`
   + qualitative viz, and CLS reported only as a *no-regression* guard.
2. **The paper's claim must be "interpretable selection at no meaningful driving cost"**, not
   "the causal graph improves driving." `dod_manlbl` 0.8579 → `nbr(2b)` 0.8336 is 0.024 ≈ **1.05
   scenarios**; the deficit is entirely 1 drivable-area and 2 TTC scenarios, while `nbr(2b)`
   posts the **highest route progress of any run measured** (0.822 vs the baseline's 0.787).
3. **R1's question is answered: yes.** A live confound branch (`cfdvar` 1.74 vs 0.0001) and a
   non-degenerate mask (`mcas_peak` 1.9× uniform, `mcfd_peak` 1.81× uniform — neither flat nor
   one-hot) cost nothing measurable in closed loop relative to every alternative except the
   untouched baseline. `nbr(2b)` is the first checkpoint that is simultaneously respectable on
   minADE (**0.6731**, best of the lineage), CLS (0.8336), collapse (1.742) and concentration.

### Why `nbr` produces an interior mask when nothing else does

Sorted by what each term pays as mask mass concentrates:

| term | marginal cost of concentrating | measured `mcfd_peak` |
|---|---|---|
| `L_bc` | **decreases** — under softmax's full support the residual scales with the number of active entries, so fewer entries is strictly cheaper | 0.915 (one-hot) |
| `L_peak` | **decreases** — one-sided ceiling `relu(H − τ)`, no floor | (applies to `M_cas` only) |
| `L_traj`, `CE(psi)` | indifferent | — |
| `L_recon` | **increases** — but the target `f_all` is a uniform mean, so the optimum is *uniform* | 0.121 = uniform exactly |
| **`L_nbr`** | **increases** — reconstructing *N distinct* agent futures from one pooled vector needs mass on those N agents; putting it all on one loses the other N−1 | **0.218 = 1.81× uniform** |

`L_nbr` is the only term in the objective with a genuine **interior optimum**. This is the
structural answer to "why does the mask collapse onto a single agent and stop there": nothing
else in the loss pays a price for stopping there. An entropy floor would force the same outcome
by fiat; `L_nbr` reaches it from a task.

---

## Which run is best, and for what

| criterion | winner | value | note |
|---|---|---|---|
| **Closed-loop driving** | **dod_manlbl** | CLS **0.8579** | no conflict supervision at all; `f_cfd` dead |
| **Causal selection** | **confB** | **1.90×** vs random, rear 0.412 | `confE`/`confF` within noise at 1.89–1.90× |
| **Ranking quality** | **confE** | corr **0.566** | best of every run measured |
| **Open-loop accuracy** | confB+BC / nbr(2b) | minADE **0.6727 / 0.6731** | |
| **Confound branch alive** | **nbr(2b)** | `cfdvar` **1.742** | *and* CLS 0.8336, *and* best minADE |
| **Non-degenerate mask** | **nbr(2b)** | `mcfd_peak` **1.81× uniform** | only run with an interior optimum |
| **Route progress** | **nbr(2b)** | **0.822** | above the 0.8579 baseline's 0.787 |
| **Best overall compromise** | **nbr(2b)** | CLS 0.8336 · minADE 0.6731 · `cfdvar` 1.742 · 1.26× | mask family, no BC, no conflict |
| **Best for the causal claim alone** | **confB** | 1.90× · corr 0.546 · minADE 0.6822 · CLS 0.8126 | GF-independent, softest conflict mask (`ent` 0.292) |

**There is no run that is simultaneously best on causality and on CLS**, and Finding 5 shows why:
CLS does not reward causality at all. The conflict family buys a large jump in interaction
alignment (1.24× → 1.90×) at a cost of **0.031 CLS** (`confH` vs `confI`, the only controlled
pair), localised in turning scenarios.

Recommendation, updated after R1: **`nbr(2b)` is the working model.** It keeps CP's mask loss, is
the only checkpoint with both a live confound branch and a non-degenerate mask, has the best
open-loop accuracy of the lineage, and gives up ~1 scenario of CLS against a baseline whose
confound branch is a constant. `confB` remains the run to cite if the interaction number is the
headline — but it costs 0.021 more CLS than `nbr(2b)` for a mask that is still below the trivial
nearest-agent heuristic.

---

## Go-to configuration

**`training_log/step2b_nbr_nbrenrich2/causal_epoch_15_minADE_0.6731.pth`**

```bash
--lambda_mask 0.5 --lambda_nbr 0.1 \
--lambda_recon 0 --lambda_bc 0 --lambda_peak 0 --lambda_conflict 0 \
--nbr_enrich 2 --graph_layers 1 --gate softmax        # + manlbl + DOD
```

| | value | context |
|---|---|---|
| CLS-R | 0.8336 | 1 scenario below the all-time best, which has a dead confound branch |
| minADE | **0.6731** | best of the entire lineage |
| `cfdvar` | **1.742** | only genuinely live confound branch (baseline 0.00012) |
| `mcas_peak` / `mcfd_peak` | 1.9× / **1.81×** uniform | only interior mask — neither flat nor one-hot |
| rear | **0.373** | best direction of any run; beats `random` 0.462 **and** `near` 0.518 |
| RNC | 8.9× · matched 3.4× · corr 0.54 | low ratio *because* the mask is soft — the test stays informative here |

It is also the **simplest objective in the table**: CP's mask loss plus one auxiliary head, four
fewer hyperparameters than anything in the conf family, and nothing in it that has to be defended
as a workaround.

**Stated weakness:** interaction **1.26×**, below the trivial nearest-agent heuristic at 2.12×. It
therefore cannot carry a "selects interacting agents" claim. That claim belongs to `confB` as the
reference ablation — 1.90× at a measured cost of 0.031 CLS.

**Two models is the finding, not a gap.** Finding 5 says no single configuration can be best on
both axes, because closed-loop score does not reward mask correctness. Reporting `nbr(2b)` as the
driving model and `confB` as the interaction ablation states that honestly; a tuned compromise
between them would hide it.

### Deliberately not run

Each is one flag and ~3 h train + evaluate, and none of them changes a conclusion:

| candidate | why skipped |
|---|---|
| **confJ** — mask + nbr + conflict | its only unique promise is folding the two models into one; `confB` already has interaction **and** a live `f_cfd` (`cfdvar` 0.181), so this is polish, not capability. Finding 5 predicts the 0.031 tax survives |
| **confK** — `--conflict_terms tight` | best case moves `dmin` 8.1 → ~7.5 on an axis CLS does not reward |
| **confL** — fixed speed-independent reach | same, plus it only matters if confJ/confK happen |
| **arc retry** with a proper longitudinal model | 3 h to maybe recover 0.019 CLS — under one scenario |

If exactly one run were affordable it should be **confJ**, unchanged from the spec in
[Next steps E](#e--mask--conflict-on-the-current-architecture--never-run): `--conflict_terms all`,
`--aligned_mode straight`, `--ego_corridor refpath`, `--lambda_conflict 0.1`. Note that
`λ_conflict` does **not** scale down — `confG` at 0.01 collapses to 1.15× and 0.860 rear.

---

## Next steps

### A — Measure the drift ✅ DONE
Implemented as `eval_interaction.py`; results in
[Finding 3](#finding-3--the-mask-does-not-select-interacting-agents). Outcome: all models
sit near chance on interaction, Step A exactly at chance, and the `nbr(2b)` impression
from the visualisations did not survive 1104 scenes. This is now the primary metric —
every future change should report `top1 vs random` alongside the shape statistics.

### B — Let the model express interaction  ← NOW THE PRIORITY
`gat-conflict` commit `299d22a` ("future-conflict features + `L_conflict`") sits directly
on `v1.1` and cherry-picks cleanly onto `gat-maskfix`. The attention edges currently carry
relative position and velocity but **nothing about whether the two futures cross**. If the
feature is absent from the input, the model cannot prefer the conflicting agent even if
the loss asked it to. Cheapest structural fix, code already written.

Finding 3 upgrades this from a guess to the obvious next move: the mask is at chance on
interaction partly because **interaction is not in the input at all**. Success criterion:
`top1 vs random` above 1.5×, ideally past `near`'s 2.12×.

### C — Counterfactual supervision
Define causality operationally: for neighbour *j*, delete it and re-run the **frozen**
GameFormer; the shift in its predicted ego plan is a label for how causal *j* is. Train
`M_cas` against it. No human annotation. Caveats: *N* forward passes per sample
(subsample), and it makes RemoveNonCausal **circular** — a held-out counterfactual
variant would be needed for evaluation.

### D — Predicate-grounded semantic edges
Fully supervised edge types from the nuPlan predicate graph (see `PREDICATE_GROUNDED_PLAN.md`).
Breaks the "no per-agent labels" constraint but is the honest ceiling on correctness.

### E — mask + conflict on the current architecture  ← NEVER RUN
The one cell of the grid that has never existed. `git merge-base --is-ancestor 3aae522 299d22a`
returns **NO**: `gat-conflict` branched off *before* manlbl + DOD, so the old
`lconflict_nbrenrich2` (07-28, `casacc` 0.329 vs today's 0.89) was a different model. Every
conflict run on `gat-maskfix` was launched with `--lambda_mask 0`.

Given Finding 5 the interesting version is **mask + nbr + conflict** — keep the only two terms
with an interior optimum and add the only term that moves interaction:

```
--lambda_mask 0.5 --lambda_nbr 0.1 --lambda_conflict 0.1 \
--lambda_bc 0 --lambda_peak 0 --lambda_recon 0 \
--ego_corridor refpath --aligned_mode arc
```

Expected: interaction 1.26× → ~1.9×, `cfdvar` stays alive, CLS ~0.80 (the 0.031 conflict tax).

### F — reformulate `L_conflict` instead of reweighting it
`E_{M_cas}[penalty]` pulls the **entire** distribution, on the same tensor the trajectory head
consumes — a direct objective conflict, which is why the damage concentrates in turns. λ tuning
already failed (`confG` at 0.01). A top-1 **margin/ranking** loss would constrain only the argmax
and leave the remaining mass free for the plan. The functional form is the untried variable.

### Also outstanding
- **CLS for `stepA`** — still never run; the 2026-08-10 simulation attributed to it was
  `step2_recon` (see the bookkeeping correction above).
- **No clean `λ_recon` on/off pair exists.** Every candidate also changes `aligned_mode` or
  `λ_bc`. Recon's closed-loop effect is unmeasured.
- **CLS for `confG`, `confF-gf`, `conf_only`, `confA`** — trained, never simulated.
- State perturbation (`cp_manlbl_pert`: pure CLS 0.58 → 0.70) never tried on this lineage;
  orthogonal to the mask, so it cannot recreate this trade.

---

## Loss reference

Total objective, `train_planner.py:245`:

$$
\mathcal{L} \;=\; \mathcal{L}_{\text{traj}}
\;+\; \lambda_{\text{kld}}\mathcal{L}_{\text{kld}}
\;+\; \lambda_{\text{ci}}\mathcal{L}_{\text{ci}}
\;+\; \lambda_{\text{mask}}\mathcal{L}_{\text{mask}}
\;+\; \lambda_{\text{recon}}\mathcal{L}_{\text{recon}}
\;+\; \lambda_{\text{nbr}}\mathcal{L}_{\text{nbr}}
\;+\; \lambda_{\text{budget}}\mathcal{L}_{\text{budget}}
\;+\; \lambda_{\text{bc}}\mathcal{L}_{\text{bc}}
\;+\; \lambda_{\text{peak}}\mathcal{L}_{\text{peak}}
\;+\; \lambda_{\text{conf}}\mathcal{L}_{\text{conf}}
$$

Notation: $B$ batch, $N=10$ neighbours, $S$ map polygons, $T=80$ future steps at $\Delta t = 0.1$ s,
$D=256$. $\mathcal{V}$ / $\mathcal{S}$ are the valid neighbour / polygon index sets.
$M^{cas}, M^{cfd} \in \Delta^{N-1}$ are **softmax over neighbours**, so each sums to 1.

| term | default $\lambda$ | status |
|---|---|---|
| $\mathcal{L}_{\text{traj}}$ | 1.0 (fixed) | dominant, GMM winner-take-all imitation |
| $\mathcal{L}_{\text{kld}}$ | 1.0 | live |
| $\mathcal{L}_{\text{ci}}$ | 0.5 | live, but on its own it lets $f^{cfd}$ collapse |
| $\mathcal{L}_{\text{mask}}$ | 0.5 | **dead** — see Finding 1; `λ=0` improved every metric |
| $\mathcal{L}_{\text{recon}}$ | 0 | fixes the collapse, but drives $M^{cfd}$ to *exactly* uniform; CLS effect **unmeasured** (no clean on/off pair) |
| $\mathcal{L}_{\text{nbr}}$ | 0 | **the go-to term** — only one with an interior optimum ($M^{cfd}$ 1.81× uniform), keeps $f^{cfd}$ alive (`cfdvar` 1.742), best minADE, CLS 0.8336. Lowers the RNC ratio (12.1× → 8.9×) because the mask stays soft, not because it is worse |
| $\mathcal{L}_{\text{budget}}$ | 0 | sigmoid-gate only; that line was abandoned |
| $\mathcal{L}_{\text{bc}}$ | 0 | separates the masks, sharpens; **source of the rear bias** |
| $\mathcal{L}_{\text{peak}}$ | 0 | never fires in practice |
| $\mathcal{L}_{\text{conf}}$ | 0 | interaction $1.24\times \to 1.90\times$, costs 0.031 CLS (confH/confI) |

---

### $\mathcal{L}_{\text{bc}}$ — Bhattacharyya overlap

$$
\mathrm{BC}\big(M^{cas}, M^{cfd}\big) \;=\; \sum_{j \in \mathcal{V}} \sqrt{M^{cas}_j \, M^{cfd}_j}
\qquad
\mathcal{L}_{\text{bc}} \;=\; \mathbb{E}_B\big[\mathrm{BC}_{\text{agent}}\big] \;+\; \mathbb{E}_B\big[\mathrm{BC}_{\text{map}}\big]
$$

Both masks are distributions over the same support, so BC measures their overlap directly:

$$\mathrm{BC} = 1 \iff M^{cas} \equiv M^{cfd}, \qquad \mathrm{BC} = 0 \iff \text{disjoint support}$$

**Why the square root.** Multiplying two small probabilities gives a small number *regardless of
overlap*. That is why `excl` $= \mathrm{MSE}(M^{cas} \odot M^{cfd}, 0)$ is blind: at uniform it reads
$1/N^4$ and calls the problem solved. The square root restores the scale,
$\sqrt{0.25 \cdot 0.25} = 0.25$ — the same order as the probabilities themselves.

**Independent of $N$.** At uniform, $\mathrm{BC} = N \cdot \tfrac{1}{N} = 1$ exactly, for any $N$;
`excl` collapses as $1/N^4$ (1e-4 at $N=10$, 3.9e-7 at $N=40$). Measured: uniform gives
BC 1.000 / excl 0.0039; peaked-on-different-agents gives BC 0.512 / excl 0.0009 — BC moves 0.49,
`excl` moves 0.003.

**Gradient.**

$$\frac{\partial \mathrm{BC}}{\partial M^{cas}_j} \;=\; \frac{1}{2}\sqrt{\frac{M^{cfd}_j}{M^{cas}_j}}$$

Strong pressure to vacate agent $j$ when the other branch claims it, zero when it does not.

---

### $\mathcal{L}_{\text{peak}}$ — entropy hinge

$$
H_{\text{norm}}\big(M^{cas}\big) \;=\; \frac{-\sum_{j} M^{cas}_j \log M^{cas}_j}{\log n_{\text{valid}}}
\qquad
\mathcal{L}_{\text{peak}} \;=\; \mathbb{E}_B\Big[\mathrm{relu}\big(H^{\text{agent}}_{\text{norm}} - \tau\big)\Big] + \mathbb{E}_B\Big[\mathrm{relu}\big(H^{\text{map}}_{\text{norm}} - \tau\big)\Big]
$$

with $\tau = 0.5$. Normalised so $H_{\text{norm}} = 1$ is uniform and $0$ is one-hot. The
$\log n_{\text{valid}}$ divisor is necessary because scenes carry between 2 and 40 neighbours; raw
entropy would let crowded scenes dominate the batch.

**A hinge, not $-H$.** Minimising entropy outright drives every scene to one-hot, but some scenes
genuinely have three relevant agents. The hinge charges nothing below $\tau$, so "two or three
agents matter" is free and "all ten equally" is not.

**Applied to the head-averaged mask.** Measured gap: $H_{\text{norm}} = 0.908$ on the head average
versus $0.513$ per head — the heads are individually peaked but disagree, and averaging flattens the
result. Penalising the average forces them to sharpen **and agree**; penalising per head would leave
them pointing at different neighbours and change nothing in the delivered mask.

**It never fires.** With $H_{\text{norm}} = 0.167 < \tau$, the term reads $\approx 0.002$. The
sharpening we observe comes from $\mathcal{L}_{\text{bc}}$, not from here.

---

### $\mathcal{L}_{\text{conf}}$ — expected conflict distance

$$
\mathcal{L}_{\text{conf}} \;=\; \mathbb{E}_B\Big[\textstyle\sum_{j} M^{cas}_j \, p_j\Big] \;=\; \mathbb{E}_B\Big[\mathbb{E}_{M^{cas}}[\,p\,]\Big],
\qquad
p_j \;=\; \tfrac{1}{3}\big(f^{\text{route}}_j + f^{\text{align}}_j + f^{\text{reach}}_j\big)
$$

Since $M^{cas}$ is a distribution, the sum is exactly the expected penalty under the mask. Writing
$\mathbf{n}_j(t)$ for neighbour $j$'s predicted future (frozen decoder top-1), $\mathbf{r}_p$ for
route-lane points and $\mathbf{r}^{\text{ref}}_p$ for the lattice reference path:

$$f^{\text{route}}_j = \log\Big(1 + \min_{t}\min_{p} \big\| \mathbf{n}_j(t) - \mathbf{r}_p \big\|\Big)$$

$$f^{\text{align}}_j = \log\Big(1 + \min_{t} \big\| \mathbf{n}_j(t) - \mathbf{e}(t) \big\|\Big)$$

$$f^{\text{reach}}_j = \log\Big(1 + \min_{t}\min_{p \in \mathcal{R}} \big\| \mathbf{n}_j(t) - \mathbf{r}^{\text{ref}}_p \big\|\Big),
\qquad \mathcal{R} = \Big\{p : \big\|\mathbf{r}^{\text{ref}}_p\big\| \le \max\big(v\,T\Delta t,\; 10\big)\Big\}$$

$\mathbf{e}(t)$ is the ego's assumed position at time $t$. Two modes, `--aligned_mode`:
`straight` (default) uses $[v\,\Delta t\, t,\, 0]$ and **ignores the reference path entirely**;
`arc` walks arc length $v \, \Delta t \, t$ along it (`confE`/`confF`; refuted, −0.019 CLS).

**The penalty is detached.** Gradient flows only through $M^{cas}$:

$$\frac{\partial \mathcal{L}_{\text{conf}}}{\partial M^{cas}_j} = p_j$$

The features are a fixed target — the model cannot move them, only shift mask weight away from
high-penalty agents. The $\min_t$ is deliberate: a conflict is an instant, one close pass suffices.

#### Anatomy of the three terms — sources, geometry, and what is wrong with them

| # | term | corridor source | geometry | length bound |
|---|---|---|---|---|
| 1 | $f^{\text{route}}$ | `inputs['route_lanes']` — **all lanes flattened** | spatial, time-blind | **none** |
| 2 | $f^{\text{align}}$ | `straight`: none · `arc`: `ref_path[0]` | **time-aligned**, one ego point per $t$ | horizon |
| 3 | $f^{\text{reach}}$ | `ref_path[0]` | spatial, time-blind | $\max(vT\Delta t, 10)$ |

Three structural observations, none of which required a new run:

**(a) Terms 1 and 3 are redundant, and 1 is the worse copy.** Both compute "min distance from the
neighbour's future to some version of the ego's path." Term 1 uses *all* route lanes instead of the
single reference path — the multi-lane corridor already rejected as too loose — and has no length
bound, so an agent 200 m down the route scores as conflicting. Averaged equally, the penalty is
**⅔ spatial and ⅓ temporal**, with the spatial half double-counted.

This is visible in the results. The mask **beats** the nearest-agent heuristic on `path`
(time-independent proximity, 4.12–4.89 m vs 5.87 m) and **never** beats it on `dmin` (same-time
closest approach, 8.1 m vs 7.25 m). The loss weights path over timing 2:1 and the outcome comes out
2:1 the same way.

**(b) Neither surviving term is sufficient alone**, which is why `--conflict_terms reach` (term 3
only) is not the answer either:

- *Lead vehicle, 30 m ahead, matched speed* — term 3 flags it (sits on the corridor); term 2 **misses
  it entirely**, because a lead car keeps its distance at every $t$. Term 2 alone structurally
  cannot see the single most important agent on the road.
- *Oncoming car crossing 1 s before the ego arrives* — term 2 correctly ignores it; term 3 flags it
  even though it will be long gone.

Term 3 = recall (who touches my path; pure geometry, no speed assumption).
Term 2 = precision (who is there at the same moment; depends entirely on the speed assumption).

**(c) One root cause spans both remaining weaknesses: constant-speed extrapolation over 8 s.**
It is why `arc` failed — walking at the current speed strands the imagined ego mid-turn, since a car
starting a turn is slow and about to accelerate — and it is also how $\mathcal{R}$ is defined. A car
stopped at a red light has $v = 0$, so its reach falls to the 10 m floor even though it will cover
60 m; the floor exists only to stop that collapsing to zero. The same flawed assumption appears in
two places.

Note the resulting inconsistency in the default configuration: with `aligned_mode=straight`, term 2
assumes the ego drives straight while term 3 knows it follows `ref_path[0]`. That is not principled —
it is an empirical fallback after one bad implementation of the alternative. **`straight` is a patch,
not the right answer**; the correct fix is the reference path plus a non-constant-speed longitudinal
model, which has never been tried.

**Untested consequences** (each one flag, one run — deliberately skipped, see below):
`--conflict_terms tight` = drop term 1, rebalancing to 1:1 temporal:spatial; and a fixed
speed-independent reach (~50 m) replacing $\max(vT\Delta t, 10)$.

---

### $\mathcal{L}_{\text{recon}}$ — repairing the $f^{cfd}$ collapse

$$
\mathcal{L}_{\text{recon}} \;=\; \Big\| \, g\big(\big[\, k \cdot f^{cas} \,;\, f^{cfd} \,\big]\big) \;-\; \mathrm{sg}\big[\mathbf{f}^{\text{all}}\big] \, \Big\|_2^2,
\qquad k \sim \mathrm{Bernoulli}\big(1 - p_{\text{drop}}\big)
$$

where $\mathrm{sg}[\cdot]$ is stop-gradient and $\mathbf{f}^{\text{all}} \in \mathbb{R}^{3D}$ is the
**ungated** scene summary — no causal/confound split, no learned selection:

$$
\mathbf{f}^{\text{all}} = \Big[\; \underbrace{\text{self\_fc}(h_{\text{ego}})}_{D} \;;\;
\underbrace{\tfrac{1}{|\mathcal{V}|}\textstyle\sum_{j \in \mathcal{V}} \mathbf{m}_j}_{D} \;;\;
\underbrace{\tfrac{1}{|\mathcal{S}|}\textstyle\sum_{s \in \mathcal{S}} \mathbf{m}_s}_{D} \;\Big]
$$

**Why it is needed.** The objective asks $f^{cfd}$ for one thing only — be uninformative about the
manoeuvre — and a constant vector satisfies that for free. Measured
$\operatorname{Var}(f^{cfd})/\operatorname{Var}(f^{cas}) \approx 1.6\times10^{-4}$ across three
independent runs.

**Why the pathway dropout $k$.** With $k \equiv 1$ the head learns to copy $f^{cas}$ and ignores
$f^{cfd}$ entirely. On the samples where $k = 0$, the target must be carried by $f^{cfd}$ alone.

**Why the stop-gradient.** Without it the main aggregation reshapes $\mathbf{f}^{\text{all}}$ to be
easy to predict — the loss would be minimised by flattening the target rather than by informing
$f^{cfd}$.

Result: $\operatorname{Var}$ ratio $0.00012 \to 0.216$.

---

### $\mathcal{L}_{\text{nbr}}$ — a per-agent task for $f^{cfd}$

$$
\hat{\mathbf{y}} = \mathrm{MLP}\big(f^{cfd}\big) \in \mathbb{R}^{N \times T \times 2},
\qquad
\mathcal{L}_{\text{nbr}} = \frac{\displaystyle\sum_{b,j,t} v_{bjt}\;\tfrac{1}{2}\!\!\sum_{c \in \{x,y\}}\!\! \mathrm{smooth}_{L_1}\!\big(\hat{y}_{bjtc} - y^{\text{gt}}_{bjtc}\big)}{\displaystyle\sum_{b,j,t} v_{bjt}}
$$

$v_{bjt}$ masks out zero-padded ground truth and invalid neighbours. A single 256-dimensional
$f^{cfd}$ must reproduce all $N$ neighbours' 8-second futures.

**How it differs from $\mathcal{L}_{\text{recon}}$, and this is the whole point.** The
reconstruction target is a *uniform mean*, and the cheapest way to reproduce a uniform mean is a
**uniform mask** — measured, $M^{cfd}$ peak $0.1213$ against a uniform baseline of $0.1205$, i.e.
exactly flat. The neighbour target is **per-agent**: a uniform mask cannot carry ten separate
futures, so the model is forced to learn an allocation. Measured, $M^{cfd}$ peak / uniform goes
$1.007\times \to 1.81\times$.

**Deliberately unweighted.** Writing $\sum_j M^{cfd}_j \cdot \mathrm{err}_j$ would let the model
lower the loss by shifting mask weight onto easy agents. Unweighted, the gradient reaches $M^{cfd}$
through $f^{cfd}$ and moves the allocation to wherever it reduces error most.

**Cost.** It degrades the causal mask — distance-matched ratio $5.2\times \to 3.4\times$ — because
enriching $f^{cfd}$ pulls the shared upstream representation toward encoding every agent's future.


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
# interaction check — model-independent, GT trajectories only, ~2 min
python eval_interaction.py \
  --pretrained_path training_log/normal/model_epoch_19_valADE_1.6487.pth \
  --causal_path <ckpt> --valid_set <valid> --graph_layers 1 --nbr_enrich 2

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
