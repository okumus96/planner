# Paper Results — closed-loop benchmarks

Single source of truth for numbers that go into the paper. Every row names the run folder it
came from. Reactive closed-loop score (CLS-R) unless stated otherwise.

Run artifacts live in `testing_log/paper_results/<name>/` (renamed from their original
timestamps; the `.nuboard` files were re-pointed, so nuBoard still opens them directly).

---

## 1. test14-hard — 272 scenarios (headline benchmark)

| # | system | CLS-R | run folder |
|---|---|---|---|
| 1 | **Ours** — predicate-gated + typed + meta-action decision (`dodmeta_v3_egoline`, e13) | **0.7312** | `test14hard_ours_dodmeta_v3_egoline` |
| 2 | **Rules-only** — same checkpoint, mask forced uniform over gated entries (`--uniform_mask 1`) | **0.6659** | `test14hard_rulesonly_uniform_mask` |
| 3 | Vanilla GameFormer-Planner (same refiner pipeline) | ~0.686 | earlier run, `2026-06-03 22:21:05` (to be re-run for the paper) |

### Per-metric breakdown, ours vs rules-only

| metric | ours | rules-only | Δ |
|---|---|---|---|
| no at-fault collisions | 0.9301 | 0.9301 | **0.000** |
| drivable-area compliance | 0.9596 | 0.9632 | −0.004 |
| making progress | 0.8971 | 0.8199 | **+0.077** |
| time-to-collision within bound | 0.8382 | 0.8750 | −0.037 |
| comfort | 0.9154 | 0.9301 | −0.015 |
| route progress | 0.7132 | 0.5964 | **+0.117** |

**Reading.** The entire +0.065 gap is progress; safety is identical (collisions equal to three
decimals) and the uniform variant is even marginally better on TTC and comfort. Mechanism:
with flat weights every rule-selected agent is treated as equally constraining, so the planner
becomes over-cautious and stalls; the learned allocation is what identifies *which* of the
relevant agents actually constrains the ego and lets it proceed past the rest.

**This is the answer to "if the rules do the filtering, why learn at all?"** — rules pick the
candidate set, learning converts that set into forward motion. Note also that rules-only
(0.666) falls *below* the vanilla baseline (~0.686): rules alone are not merely insufficient,
they are harmful without a learned allocation.

**Difficulty scaling.** Ours vs vanilla is **+0.05 on hard** against +0.029 on the reduced
38-scenario set — the advantage grows as scenarios get harder, which is the direction the
method predicts.

### Caveats to state in the paper

1. **Row 2 is an inference-time intervention, not a trained baseline.** The checkpoint was
   trained expecting learned weights and we flattened them at test time, so part of the drop is
   train/test mismatch. As it stands this is a strong *faithfulness* result (the learned weights
   are load-bearing). To claim "a rules-only system drives worse", train one model with the mask
   fixed uniform — one run, still outstanding.
2. **Our pipeline includes the lattice refiner** (as in GameFormer-Planner). Published
   test14-hard numbers — Causal-Planner 0.692, Diffusion Planner 0.692, BeTopNet 0.688, human
   expert 0.688, PlanTF 0.603 — include methods explicitly evaluated *without* post-processing.
   Report ours-vs-vanilla under the identical pipeline as the controlled comparison and treat
   the published table as context, not as a like-for-like claim.
3. **Unresolved:** an earlier `CausalRefinerPlanner` run on the same 272 scenarios
   (`2026-07-17 09:48:47`) scored **0.7448**, above ours. The checkpoint behind it has not been
   identified. Resolve before claiming ours is the lineage best on hard.

---

## 2. test14-random_reduced — 38 scenarios (development set)

Used during development. Too small for a primary table: one scenario is worth 0.026 CLS, so
differences below that are noise.

| system | CLS-R | note |
|---|---|---|
| dod_manlbl (5-class decision, no channels) | 0.8579 | best of the pre-channel lineage |
| **dod_meta v1** — gate+typed+H | 0.8487 | λ_nbr accidentally 0, `f_cfd` collapsed |
| **dod_meta v2** — gate+typed+H, λ_nbr 0.1 | 0.8456 | clean λ_nbr pair with v1 |
| **typed** — gate+typed, 5-class decision | 0.8453 | |
| **gate** — gate only, 5-class decision | 0.8373 | |
| GF baseline (frozen GF plan → refiner) | 0.8199 | only row with an at-fault collision |
| noresid (no channels) | 0.8160 | lineage baseline |

**Ablation ladder (same generation, old corridor):** noresid 0.8160 → +gating **0.8373**
(+0.021, the only real driving gain) → +typing 0.8453 (+0.008, inside noise) → +meta-action
decision 0.8487 (+0.003, inside noise). One mechanism buys the driving performance; the other
two buy capability at no measurable cost.

---

## 3. Attribution measurements (validation, 1118 scenes)

| measurement | value | meaning |
|---|---|---|
| interaction selection vs random | 1.24× → **1.45×** (gating) | structural gating alone, no new loss term |
| calibration: weight vs plan displacement after deleting an agent | r = 0.552 → **0.610** (gate), 0.617 (v3) | higher weight ⇒ larger effect, reliably |
| deleting a causal vs non-causal agent | ~1000× difference in plan displacement | the mask is load-bearing |
| influence concentration | one agent carries **~92%** of all agent influence | the filter concentrates rather than spreading |
| decision-conditioned attribution | braking → collision-course **3.3×**, follows **2.1×** over base rate; right turns → VRU **1.66×**, left turns 0.55× | attribution is semantically coherent |
| decision agreement (announced vs emitted plan) | lon 75.8% / lat 83.3% | the decision describes the plan it produces |

**Not claimed:** that a *specific relation type* drives a decision. Three interventional
attempts (edge-level dose–response, braking-conditioned edge removal, decision-conditioned lift)
gave correlational coherence only. Typing is reported as a readout. See the dossier section
"What A proves and what it does not".

---

## Outstanding runs

1. Vanilla GameFormer-Planner on test14-hard, re-run under the current pipeline (row 3 above).
2. Trained rules-only baseline (mask fixed uniform during training).
3. Full test14-random for comparability with published tables.
4. Identify the checkpoint behind the 0.7448 run of 2026-07-17.
