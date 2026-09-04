# Decision Dossier: Edge-Enriched Causal Planning on nuPlan (v2)

> **Referans dokumanlar (bu repo):** detay gerektiginde bunlara bakilabilir —
> - `PAPER_RESULTS.md` — kapali-dongu benchmark sayilarinin tek kaynagi (test14-hard/reduced
>   tablolari, rules-only, ablasyon merdiveni; her satir kosum klasorunu isaret eder).
> - `MASK_LOSS_INVESTIGATION.md` — arastirma gunlugu (1600+ satir): mask-loss bulgulari
>   (Finding 1-6), kanal soyu sonuclari, "Paper positioning" bolumu, tum kosum konfigurasyonlari.
> - `CHANNELS_AUDIT.md` — 11 ajan + 8 harita kanalinin KG karsiliklari, sapmalar, esikler,
>   selftest dogrulamasi (channels.py'nin denetim kaydi).
> - `CAUSAL_PLANNER_COMPARISON.md` — bizim CausalPlanner ile resmi Causal-Planner kodunun
>   birebir karsilastirmasi (neyi sadik reprodukte ettik, neyi degistirdik).
> - Binding-test sonuclari ve v3/v4/v6 envanteri: bu dosyanin SONUNDAKI
>   "Binding test" bolumu.

**Status:** Direction converging on refined A+B. User is still deliberating; no implementation has begun. This file is the running dossier; it becomes the implementation plan once the user confirms.

## Hard constraints (from user)

1. **The predicate KG (`nuPlan_Predicates_KG`) is the friend's separate paper.** User's paper cites and briefly describes it as an external tool. Never claim, release, or re-publish the predicates.
2. **Venue (CONFIRMED by user 2026-08-11): ICRA or IROS MAIN CONFERENCE full paper** — not a workshop, not a journal. (Earlier "journal-grade" interpretation was my misreading — the stale memory file `gameformer-causal-paper-direction.md` still says journal; FIX IT once out of plan mode.) Completeness still required: full test14-random + test14-hard (the 43-scenario reduced set is too small as a primary result; 1 scenario = 0.023 CLS), reactive + nonreactive, ablations, property-matrix related work. Timeline reality: ICRA submission deadlines historically fall mid-September (~5 weeks from now — very tight for retraining + full benchmarks); IROS deadlines fall ~March (realistic primary target). Verify current dates.
3. **No GameFormer-derived training labels.** User does not trust frozen GF outputs. Supervision must be model-free. (Supporting evidence from user's own logs: confD, which used the frozen GF ego plan as conflict corridor, did not beat the ref-path variants confB/confF.)
4. **Success criterion:** beat the works the user builds on — vanilla GameFormer-Planner and the Causal-Planner (IROS 2025) reproduction — on CLS etc. SOTA not required. Caveat from user's own Finding 5 (`MASK_LOSS_INVESTIGATION.md`): CLS is blind to mask correctness (chance-mask confI scored 0.8364 > better-mask confH 0.8059; no CLS↔mask-quality correlation across 12 runs), so the causal claim must be validated by mask-correctness + faithfulness metrics, with CLS as the do-no-harm+gain claim. Missing datum: vanilla GF-Planner CLS on the same benchmarks — must be measured.
5. **Scope decisions:** C (meta-path mediation) = explicitly future work. D (resource paper) = off the table standalone (friend's KG); at most, release user's own label-generation code/labels as a by-product. E (explanation-only planner) = not standalone; folded into the A+B paper as the interpretability/falsification section.

## The refined A+B design

### A — sharpened formulation (answer to the user's "one-hot" objection)

Input-side predicate features ≈ one-hot ≈ inert — confirmed by user's own ablation (conflict_feats in input: 1.24×→1.26×). Therefore A is NOT "add predicate features to edges." A is: **lift the causal mask from agents to typed relations** — decompose each ego→agent edge into named predicate channels and let the gate distribute causal mass over (agent, relation-type) channels. Three roles of typed relations, in increasing importance:
- as input information: near-zero for boolean geometric predicates; nonzero only for map-topological numerics (signed path distance through connectors, corridor adjacency) absent from the 7-dim geometric edge;
- as parameters: per-relation-type message transforms (mechanical generalization of `_per_type`, `causal_graph.py:355-361`);
- **as the support of the causal distribution and the vocabulary of the loss/readout — the actual contribution.** Without types, B's supervision cannot be stated and the falsification protocol has no units.

### B — restructured to respect the no-GF constraint

- **Supervision channel (training labels): 100% model-free.** KG interaction events (confirmed-status, with confidence/evidence fields) + ground-truth-trajectory interaction geometry (the model-independent machinery of `eval_interaction.py`). GF appears nowhere in the supervision path.
- **Interventional channel (evaluation only): closed-loop agent removal through the user's OWN planner** — the existing `--remove` machinery (`causal_refiner_planner.py:52-77`) + branch-swap. This is what the user proposed; it is correct as *falsification*, but must NOT be used as training labels (self-removal labels are circular — user's own documented RemoveNonCausal caveat; the model can make any mask "correct" by making itself sensitive to whatever it attends).
- **If a model-based label channel is ever wanted:** neutral non-learned reference (PDM-Closed/IDM replay or GT-feasibility analysis), never GF, never self.
- **Loss:** expected-penalty / CE template on `M_cas` (`train_planner.py:239-254` precedent — the only lever that moved mask content, 1.24×→1.90×).

### Roles for the existing auxiliary heads (user's recon/nbr question)

- Neither head is causal evidence (logs: recon(2) fixes f_cfd collapse but drives M_cfd uniform, mask ~chance 1.21×; nbr(2b) buys minADE 0.6731 / CLS 0.8336 but interaction selection 1.26× "not real" and matched RNC degrades 5.2×→3.4×).
- Principled repurposing: **feature-space counterfactuals** — zero agent j's token, forward pass, measure Δplan/Δf_cas. Cheap, amortized. Uses: (i) screening which agents merit true closed-loop removal; (ii) a **faithfulness self-consistency loss** (plan invariant to removing low-M_cas agents; sensitive to removing top-M_cas agent), degenerate solution blocked by imitation loss + symbolic anchors.
- Resulting three-constraint story: recon = losslessness of the split; faithfulness = decision-relevance; symbolic supervision = semantic correctness. Each covers a failure mode already observed (collapse / decorative mask / wrong content).
- nbr head: diagnostic or small-λ only; report the predictability-vs-causality tradeoff.

## New options under evaluation (user request, 2026-08-11): runtime monitoring / reasoning / decision making

Preliminary analysis (pending a prior-art verification pass on belief-monitoring and decision-conditioned planning):

- **F — Predicate runtime monitor.** Online cheap predicates + M_cas compared at runtime; inconsistency ("planner ignores an agent that is mergesInFrontOf ego") triggers fallback via the existing lattice/refiner. Trajectory-level monitoring is crowded (RSS, STL monitors, RuleFuser, rule hierarchies, runtime-assurance/simplex); monitoring the planner's *internal beliefs* appears fresh. No retraining needed (wrapper). Risks: gains may come from the fallback heuristic (needs ablation); CLS-blindness applies — evaluate on interactive scenario slices. Best role: the deployment/system section of the main paper, or a companion ITSC/IV paper — likely not the ICRA centerpiece alone.
- **G — Symbolic reasoning layer.** Ontology/rule inference over predicates in the loop. Weakest standalone: ontology-reasoning prior art is old (IV 2017 Prolog/OWL line), learned logic scoring on nuPlan exists (FLoRA), latency/brittleness risks, and its outputs must enter the planner via A/B-style injection or H-style decisions anyway. The genuinely novel reasoning (multi-hop mediation) is exactly option C = future work. Verdict: absorb, do not center.
- **H — Predicate-grounded decision making.** Three sub-variants: H2 (predicate costs in refiner/scorer — overlaps FLoRA, engineering-grade), H3 (classic hierarchical behavior planner — old), and **H1 (the gem): predicate-grounded decision conditioning.** Replace/augment the current 5-class geometric maneuver label in the DOD pathway (`b* = argmax psi_cas` embedded into mode queries; label port at `train_planner.py:68-125`) with *relational interaction decisions* (yield-to-j / proceed-before-j / follow-k) supervised model-free from realized outcomes via KG order/interaction predicates + GT. Answers the user's Gap 1 ("no decision semantics") literally; DOD conditioning is already the user's best-CLS mechanism (dod_manlbl 0.8579), so semantic upgrading plausibly moves CLS — matching the user's success criterion. Must position vs. M2I-style pass/yield relation labeling (prediction-only) and game-theoretic planners.

**Updated leading recommendation (pending verification + user decision):** core paper = **B + H1 on A's vocabulary** — grounded *beliefs* (typed-relation causal mask, model-free symbolic supervision) + grounded *decisions* (predicate-grounded decision conditioning), with the interventional falsification protocol as evaluation and E-style explanations as the qualitative payoff. F optional as a system section or companion paper; G absorbed; C future work.

## No-regret prerequisites (unchanged)

1. Planner-side join fix: persist `neighbor_indices` + track tokens into `.npz` (`data_process.py:494-496`; returned by `agent_past_process`, `data_utils.py:198`); thread through `train_utils.py:65-85` and `read_batch` (`train_planner.py:48-59`, incl. the 9-vs-8 unpack fix).
2. KG-side join fix: run with `pairwise` + `agent` categories (or `--include-category-dependencies`) so pair endpoints and agent types resolve.
3. Pilot KG extraction over a modest scenario set; label-quality audit: agreement between KG events, conflict features, and (evaluation-only) removal effects.
4. Measure vanilla GF-Planner CLS on the paper benchmarks (baseline row currently missing).

## Next step

On user confirmation of (a) venue interpretation and (b) the refined A+B direction: produce the concrete implementation plan (label pipeline, model changes, training protocol, evaluation matrix incl. full test14 benchmarks, ablation table, paper outline) — Phase 2 Plan agent, then ExitPlanMode.

---

## STEP 1 — the channel-activation module, built and validated offline (before touching the model at all)

The function `compute_channels(features, neighbor_futures, ref_path) → active[N,R], evidence[N,E]` is the foundation of everything: A needs it, A-lite gating needs it, H's counterpart selection needs it, F's monitors need it, and the rules-direct baseline needs it. Every rung of the fallback ladder stands on this one piece — so it's built first, and built standalone.

Concretely, in the repo:

1. **Fix the relation set** — small and closed. Write the threshold tests for each. Not starting from zero: `_conflict_features` (causal_graph.py:43-131) already computes route-distance, time-aligned gap, and reach-corridor tests from exactly the right inputs — the channel tests are richer variants of the same machinery. The friend's PREDICATES.md gives the formal definitions to transcribe (and cite).
2. **Run it offline over cached scenarios** — everything needed is already in the `.npz` cache (neighbor pasts + types, lanes, route, `c_lat_candidates` as ref path) plus futures from BOTH sources: GT futures (in the cache) and frozen-GF top-1 (`extract_neighbor_top1_futures`). A standalone script in the style of `eval_interaction.py` — model-independent, no training.
3. **Extract the three numbers that decide the design**, from the same run:
   - **Predicates-per-pair distribution** — the one-channel hypothesis, tested. If ~90%+ of pairs have exactly one fired relation, the within-pair allocation machinery simplifies and it becomes a supporting stat for the paper.
   - **Coverage — the A-killer risk, measured:** in what fraction of frames does a clearly-interacting agent (nearest, or high closing speed) have ZERO fired predicates? This number decides how the fallback channel must be designed before any training happens.
   - **GF-vs-GT activation agreement** per relation type — computes for free since both future sources run; grounds the GF=GT assumption empirically and becomes a motivation/ablation figure.
4. **Eyeball it** — BEV plots of a handful of scenarios with agents colored by fired relation (viz_causal machinery adapts directly). Thresholds always need a visual sanity pass; this is where "oncoming fires on parked cars" type bugs get caught.

### Step-1 status (2026-08-14)
- **Relation set superseded the draft above** ({lead, cut-in, oncoming, adjacent, fallback} was the early sketch): final set = **KG-anchored 11 channels** — corridor-ahead/behind (hasSpatialMapRelation=same_map + sign of hasSignedPathDistanceTo), adjacent-left/right, connected-succ/pred (reserved slots; need map_api, silent in npz-only v1), collision-course (onObservedCollisionCourseWith on predicted futures), path-crossing (sharesIntersectionWith geometric proxy), proximity-fallback (near/veryNear), merging-in-progress, overtaking-in-progress. Thresholds from KG constants (same-flow 0.45 rad, near 5 m, CPA 8 s/2 m, overtake 1.25/0.3/1.0 m). Interaction & intent predicate families: NOT used anywhere (user decision). Gate = existing KNN-10 + channel-firing (isRelevantToEgo dropped as redundant).
- **Built:** `GameFormer/channels.py` (compute_channels, batched torch, future-source-agnostic) + `eval_channels.py` (selftest / stats / BEV viz). Evidence is per-pair `[N,8]` (Δs, d_lat, d_fs, closing, TTC, t_entry, Δθ_flow, v_lat) — attached to every fired channel of the pair.
- **Selftest: PASS** (4 synthetic known-answer scenes: lead→corridor-ahead only; far-parked→nothing; cut-in→adjacent-right+merging; crosser→path-crossing).
- **Pending:** stats run on processed validation npz (1104 scenes) — `eval_channels.py --data /home/lt-hta-ai4/ssd1/nuplan/processed_data/validation --pretrained_path training_log/normal/model_epoch_19_valADE_1.6487.pth --viz 9 --json channels_stats.json`. Decision rules: ~0%-firing channels get dropped/merged; multi-fire histogram tests the one-channel hypothesis; coverage ≈0 validates the fallback; low GT-vs-GF agreement marks future-dependent channels as weak-evidence. data_process on test14 deliberately postponed (user decision — relation-set testing only, validation npz suffices).
- **Parallel:** user is running the residual experiment (Run 2, `--ego_residual 0` flag already added to the codebase) — its CLS/gate_cos outcome + the faithfulness run (Run 3) will fix A's final training recipe.

## PAPER NARRATIVE (2026-08-19) — gap, solution, sell

### The gap (capability framing — NOT "others were wrong")

Learned planners encode all agents and map elements uniformly; the interactions that actually
constrain a maneuver — the vehicle to follow, the car merging ahead, the pedestrian crossing the
intended path — end up inside anonymous attention weights. Causal planners (Causal-Planner,
IROS'25) improve on this by splitting the interaction graph into causal/confounding parts, but
the split is still performed over **untyped, unconstrained** edges. Four capabilities are missing:

- **G1 — no relational semantics.** The mask says *which* element matters, never *why*.
  "Agent 3: 0.42" cannot be decomposed, compared across scenes, or checked.
- **G2 — symbolic traffic knowledge cannot be injected.** Ontologies define exactly these
  relations, but the standard route (extra input features) leaves behaviour unchanged —
  measured here: 1.24x -> 1.26x, inert.
- **G3 — the graph cannot be queried or intervened on.** With anonymous edges you can only
  delete an *agent*; "what if this agent were not a lead vehicle?" is not even expressible.
- **G4 — the decision is invisible.** Planners emit geometry, not decisions; maneuver-level
  intent stays implicit (CP's DOD conditions on a 5-class geometric label).

### What we built

**A — symbolically instantiated, relation-typed causal graph.** The edges of the ego-centric
causal graph are instantiated by symbolic traffic predicates (11 agent + 8 map relations,
transcribed with published thresholds, computed online from predicted futures + map). An element
with no fired predicate cannot enter the causal softmax; the causal distribution is over
(element, relation) pairs with relation-specific K/V transforms. **Rules define the edge set and
the edge types; learning only allocates weight over them.**

**H — meta-action decision conditioning.** The decision slot that conditions the trajectory
decoder is upgraded from the 5-class geometric label to a factored 9 longitudinal x 7 lateral
meta-action, derived label-free from the demonstration. Single slot, no parallel pathway.

### The claim: a causal graph you can READ, CONDITION and INTERVENE ON

| capability | why it needs this design | evidence |
|---|---|---|
| **READ** — which relation carried the weight | needs a relation-indexed support; a per-agent mask cannot decompose it, and a parallel predicate module does not know the model's allocation | decision-conditioned lift (val, v3): braking loads on collision-course **3.3x** and follows **2.1x** over base rate; acceleration pushes both **below** base (0.66x); right turns load on VRU **1.66x** where left turns do not (0.55x) |
| **CONDITION** — the filter improves | rules restrict the support the softmax may use | gating is the only component that moves CLS: 0.8160 -> **0.8373** (+0.021); selection 1.24x -> 1.45x; RNC calibration 0.552 -> 0.610 |
| **INTERVENE** — edge-level do() | you can delete a *relation* while keeping the agent only if edges are typed | **TO RUN** (TODO 11): do(relation = off) -> decision flip rate, speed-profile shift, dplan; control = irrelevant relations |

**Sell in one sentence:**
> *"We turn the causal graph of a learned planner from a metaphor into an object you can read,
> condition on, and intervene on — at the level of individual relations."*

CLS is **not** the claim; it is the do-no-harm row (both parents beaten, at-fault collisions
eliminated). Typing and H are **not** sold as CLS gains — measured neutral, stated as such.

### Reviewer objection cards

- *"The gains are within noise."* -> The contribution is a queryable causal graph, not a score;
  CLS is the do-no-harm row. And no baseline can perform the edge-level intervention we report.
- *"Why not compute the predicates in parallel, outside the network?"* -> A parallel module
  reports what the world contains; it cannot report what the planner used, and you cannot
  intervene on it. Measured: input-level injection is inert (1.26x), structural gating is not
  (1.45x).
- *"Why type the edges if typing does not improve driving?"* -> Typing is what makes
  relation-resolved reading and relation-level intervention possible; it is measured to cost
  nothing (0.8373 vs 0.8453 = 0.3 of a scenario). 85% of gated-in agents carry exactly one
  relation, which explains the absent performance effect.
- *"Is the attribution real or decorative?"* -> Two independent lines: correlational
  (decision-conditioned lift) and interventional (edge-level do(), plus agent removal with a
  distance-matched control, r = 0.62, monotone dose-response).
- *"Why not a VLM/VLA?"* -> Their rationale is generated text with no enforced coupling to the
  action (nuVLA even disables it at inference and planning is unchanged); ours is the computation
  itself, runs at planner rate, needs zero annotation, and is deterministic.

### What A proves and what it does not — measured 2026-08-19 (edge-level CF)

Three interventional experiments were run on `dodmeta_v3_egoline` (validation, 1118 scenes).
Verdict: **A's provable contribution is the GATE (predicate-defined support), not the TYPING.**

**Proven — safe to write in the paper:**
- Which agents may enter the causal graph is decided by symbolic predicates, and this improves
  both the filter and driving: selection 1.24x -> 1.45x, RNC calibration r 0.552 -> 0.610,
  CLS 0.8160 -> 0.8373. Removing an agent the graph marks causal moves the plan ~1000x more
  than removing one it marks non-causal (RNC, with a distance-matched control at 6.8x).
- Influence is concentrated, not spread: for agents carrying >=2 relations, cutting the single
  top-mass edge reproduces ~70% of the effect of removing the whole agent (18.2% vs 26.4%
  decision flips; 0.705 vs 0.997 m). NOTE: this comparison targets the SAME agent, so
  selection-by-mass does not bias it — but it says nothing about relation *semantics*.
- Map edges shape geometry, agent edges drive decisions (map removal: 7% decision flips but
  0.41 m plan shift; agent removal: 26% flips).

**NOT proven — must be left out:**
- "The planner's braking decision is driven by the `follows` relation specifically."
  Conditioned on braking scenes (n=145), cutting the top constraint edge
  (follows / same_lane_ahead / collision-course) made the planner leave the braking class
  **4.8%** of the time, while the control (behind / adjacent / overtakes) did so **6.0%** —
  i.e. the control was *higher*. The only differential is speed: +0.112 m/s vs +0.017 m/s
  (~7x) and dplan 0.82 vs 0.64 m. Direction right, magnitude small, decision metric null.
- The "high-mass edge vs random edge" contrast (2-3x dose-response) is **confounded by
  selection**: the top edge carries more mass by construction. It shows weight predicts
  effect, not that relation identity matters.

**Why it likely came out this way (diagnosis, not excuse):**
1. **No objective ever asked for relational routing.** Losses are imitation + CP mask losses +
   kld/ci. Typed K/V give the capacity to differentiate relations but nothing creates pressure
   to use them — the same "structure vs allocation" lesson measured twice before.
2. **The relation vocabulary is redundant.** `follows` is geometrically inside
   `same_lane_ahead`, `near` overlaps almost everything; cutting one leaves the same
   information available through another. Relations are not disjoint carriers.
3. **Downstream collapse.** After the softmax everything sums into a single f_cas; no
   architectural component requires relation identity to survive past the attention.

Consequence for the paper: sell the gate; present typing as a readable-attribution affordance
with its honest numbers; do not make relation-level causal claims about decisions.

**Two corollaries worth remembering (2026-08-19):**
- *If the relations were mutually exclusive, typing would collapse into merely labelling agents*
  — the (agent, relation) softmax would degenerate into a per-agent softmax with a name attached.
  Typing only has room to matter where agents carry several relations at once, which is 15.5%
  of gated-in agents. Any vocabulary redesign should weigh this: more disjoint relations make
  the naming cleaner but make typing structurally less meaningful.
- *This is the third time the same lesson has appeared in this study:* conflict features as
  input -> inert (1.24x -> 1.26x); relation-typed K/V without loss pressure -> inert (1.45x vs
  1.46x, CLS 0.8373 vs 0.8453); decision slot as a deterministic function of f_cas -> ignored
  by the head (b*-swap). **Capacity never organizes itself; only losses and structural
  constraints have ever moved anything in this codebase.**

**User decisions (2026-08-19):** the predicate vocabulary will be revised (TODO 3-4), and
L_faith / the "B" supervision line is considered feasible within the 4-week window — i.e. the
missing pressure term (cause 1 above) is back on the table rather than deferred.



### Counterfactual reasoning — where it now sits

Edge-level intervention **is** counterfactual reasoning, structural rather than linguistic:
do(collision_course(ego, j) = false) is a literal intervention on a named scene variable, and we
answer it by running the planner under that hypothesis instead of asking a language model to
imagine it. This **revives the CF contribution that was shelved when b*-swap failed**: b*-swap
intervened downstream, on a variable the head had learned to ignore (it is a deterministic
function of f_cas), whereas relation ablation intervenes upstream, on a genuine input of the
causal attention — so the redundancy that killed b*-swap does not apply.

## Deployment corridor-failure bug — diagnosed and fixed (2026-08-21)

**Chain:** typed attention needs channels; channels need a corridor; in DEPLOYMENT the lattice
can fail to produce one (ego off-route, mid-intersection). Training NEVER sees this (channels
always from the npz cache), so:
- v3 and earlier CLS runs: on corridor-failure frames the model silently fell into the
  **untrained untyped attention branch** (weights exist, zero gradients ever) -> those few
  frames ran with random attention. Training itself was 100% typed and clean; only a few
  deployment frames per run were dirty measurement.
- v4 (bottleneck) turned the same event into a hard crash (256 vs 672 dims) — which is how it
  was discovered.
- **Fix (commit e0ce978 on channels / a1d962e on relation-bottleneck):** when the lattice
  yields no route, synthesize a straight "current lane continues" corridor in ego frame ->
  channels compute normally every frame, input stays in-distribution. Inference-only change:
  NO retraining needed, checkpoints untouched. Known residual: on curves the straight-corridor
  heuristic mislabels some relations (wrong-ish but in-distribution). A `_synth_frames` counter
  now reports how often the path fires per run — check it after the next hard run.
- Gating clarification (user question): branch choice is per FRAME, not per agent — gated-out
  agents get zero weight INSIDE the typed softmax; they never route through the untyped branch.

## TODOS
1. Vizi duzeltmek lazim. anlasilir ve attractive 
2. Video uretmek lazim.

3. predicate havuzunu duzeltmek finalize etmek lazim. 
4. Adj vs icin ref path koridor length i cokz uzuns ona bakmak lazim

5. Benchmarking yapmak lazim CLS/ remove high low etc. 

8. Edge Evidence i denemedik. 
9. Harita ile agentlaari ayni softmax havuzu icerisinden puanlama?
10. bottleneck
7. Faithfullness loss isi.






## FUTURE WORK — karar itaati + CF reasoning (kapsam karari 2026-08-18)

6. Counterfactual mi eklesek -> FUTURE WORK (asagida, b*-swap sonucu geregi)

10. yani b* forcelasin trajectory ciktisini. -> OLCULDU: forcelamiyor (asagida)
11. **[FAZ 1 — ONCELIK] Kenar-duzeyi counterfactual (relation ablation).** do(r = off): ilgili
    kanalin TUM girislerini softmax oncesi kapat (ajan sahnede kalir, o ilisik yolu kesilir),
    yeniden kostur; olc: karar degisim orani, uc-hiz farki, dplan. Kontrol: alakasiz iliskiler
    (ornegin overtakes) kapatildiginda degisim ~0 olmali. Cikti: iliski x karar mudahale matrisi
    + kosegen iddiasi ("lift'i yuksek olan iliski, kaldirilinca karari degistiren iliskidir").
    Uygulama: disentangler `inputs["channel_active"]`'i oncelikli okuyor -> eval script'inde
    kanali hesaplayip sutunu sifirlayarak enjekte et; MODEL DEGISIKLIGI GEREKMIYOR.
12. **[FAZ 2] Full test14-random + test14-hard** (final model + vanilla). Yayinlanmis tablolarla
    ayni olcek olmadan makale okunmaz; reduced-38 sayilari kiyaslanabilir degil.
13. **[FAZ 3] rules-only baseline** (yanan kanallarda uniform agirlik, ogrenme yok) — kisitlanmis
    destek ICINDE ogrenmenin gerekli oldugunu gosterir. + veri verimliligi (%25/%50).

**Olcum (eval_bswap.py, dodmeta_v2, 1118 val sahne):** karar slotu sadik RAPORTOR ama KOL degil.
Agreement (ilan edilen karar planin yeniden-etiketiyle uyum): lon %75.8 / lat %81.7 / joint %63.2.
Compliance (b* zorlaninca plan takip ediyor mu): aile %0-33, Δplan ~0.2 m, doz-yon ~sans.
Sebep: egitimde b* = argmax psi(f_cas) — f_cas'in fonksiyonu, head'e marjinal sifir bilgi,
head okumayi ogrenmemis (Finding 6 yan-kanal yasasinin karar-slotu tezahuru).

**Paper'da:** iki sayi da raporlanir — "kendi slotumuza mudahale ettik, readout-ama-kol-degil
bulduk" teshisi (probu reviewer'a kendi metodolojimiz ogretiyor; olculmus limitation = titizlik,
kesfedilmis limitation = red). Final checkpoint'te agreement + swap bir kez daha olculur.

**FW-1. CF reasoning (nuReasoning-tarzi):** alternatif (lon x lat) aksiyonlari head'e zorla,
rollout'lari tahmin edilen komsu future'lari + drivable alana karsi deterministik skorla ->
Safe/Suboptimal/Unsafe. VLM'siz, annotation'siz, faithful-by-construction CF degerlendirmesi.
ONKOSUL: compliance fix'i (head zorlanan karari takip etmeli; su an rollout'lar tepkisiz).

**FW-2. Karar itaati (compliance) cozum adaylari** (umut/maliyet sirasiyla):
- **Teacher forcing:** p olasilikla b* = GT etiketi (argmax psi degil) — GT f_cas'in fonksiyonu
  degil -> slot bilgilendirici olur -> head okumayi ogrenmek zorunda (command/goal-conditioned
  BC standardi). Tek satir. [Istege bagli piyango: v3 retrain'de 2. GPU bosta kalirsa sessiz
  `_tf` varyanti kostur, swap hakem olsun — kullanma zorunlulugu yok.]
- **Counterfactual consistency loss:** egitimde rastgele b' zorla, rollout'un b' semantigine
  uymamasini turevlenebilir vekillerle cezalandir (lon: uc-hiz, lat: heading isareti). GT
  gerektirmez; el yapimi vekil + tuning riski.
- **Branched heads (CIL):** karar ailesi basina ayri head — itaat mimari garanti; 9x7 icin
  aile-duzeyine katlamak gerekir.
- **CFG-tarzi guidance:** egitimde slot dropout + inference'ta kosullamayi olcekle guclendir;
  deterministik decoder icin deneysel.
- **Deployment-side sert zorlama:** karar lattice adaylarini / refiner maliyetini filtreler,
  veya runtime monitor karar-plan celiskisini online bayraklar — retrain'siz, yurutulen plani
  zorlar (iddianin yeri asagi kayar).

**FW-3. L_faith (secim/tahsis kaldiraci)** ve **FW-4. edge evidence ablasyonu** — kapsam
dondurma karariyla birlikte future work (gerekcesi MASK_LOSS_INVESTIGATION.md'de).



---

# Binding test — decision-level intervention on a GT-grounded scene set

Date: 2026-08-24. Scripts: `eval_binding_set.py` (set selection), `eval_binding_test.py`
(interventions), `eval_binding_analyze.py` (responder/non-responder comparison).
Artifacts: `binding_set_v6.json` (90 scenes), `binding_test_results.json` (per-scene records),
`viz_out/binding_set.png`, `viz_out/nonresponders.png`.

---

## 1. Models: v3 / v4 / v5 / v6 — what each is, where each lives

**Decision (2026-08-24): the current stable main model is v3.** v4/v5/v6 are experimental
variants built on top of it; none has been promoted.

| model | run folder | flags on top of common base | branch / code | val minADE | CLS-R test14-hard |
|---|---|---|---|---|---|
| **v3** (stable) | `dodmeta_v3_egoline` (e13) | — | predates both branch tips; loads on either branch with `--rel_bottleneck 0 --rel_evidence 0` | 0.7372 | **0.7312** |
| v4 | `dodmeta_v4_relbottleneck` (e18) | `--rel_bottleneck 1` | `relation-bottleneck` (commit `0e60251`) | 0.7320 | 0.7151 |
| v5 | `dodmeta_v5_egodrop` | `--rel_bottleneck 1 --ego_drop >0` | `relation-bottleneck` | — | not measured; set aside after per-edge effects weakened |
| v6 | `dodmeta_v6_relev` (e18) | `--rel_bottleneck 1 --rel_evidence 1` | `relation-bottleneck`, `rel_evidence` code **uncommitted working tree** as of this date | 0.7139 | not measured; test14-random_reduced CLS-R **0.8460** (38 scenes, run `2026-08-24 23:35:06`) |

Common base (identical in all four, verified by config diff): gate_channels 1, typed_kv 1,
dod_meta 1, ego_residual 0, nbr_enrich 2, graph_layers 1, seed 3407, lambdas
(mask 0.5 / kld 1.0 / ci 0.5 / nbr 0.1 / recon 0), aligned_mode straight, ego_corridor refpath,
20 epochs. v3-vs-v4 differ in exactly one flag (`rel_bottleneck`), v4-vs-v6 in exactly one
(`rel_evidence`).

What v4 and v6 were attempts at (stated purpose at build time):

- **v4 — relation bottleneck**: the decision head `psi` reads a block-structured vector
  (ego block 64 dims + one private 32-dim projection per relation, 11 agent + 8 map) instead of
  the pooled `f_cas`. A relation that did not fire contributes an exactly-zero block; no relation
  can write into another's dimensions. Trajectory head untouched. Purpose: relation identity
  survives to the decision head (the "downstream collapse" diagnosis).
- **v6 — v4 + per-relation evidence**: in the typed attention, each (input, relation) entry gets
  its own edge vector = shared geometry + only the evidence dimensions in that relation's
  definition (`REL_EVIDENCE` / `MAP_REL_EVIDENCE` in `channels.py`; e.g. follows → {Δs, closing},
  near → {d_fs}). Purpose: relations stop being informational copies of each other, so cutting
  one deletes information (the "redundant vocabulary" diagnosis).

Note: `joint_softmax` (commit `022b7aa`) exists only on the `channels` branch and is **not** part
of v4/v6. Loading any checkpoint requires passing exactly the flags it was trained with;
`rel_bottleneck`/`rel_evidence` mismatches fail with shape errors.

---

## 2. Test set — why it exists and how it was chosen

Population-level relation-ablation tables (`eval_relation_cf.py`, all 1118 validation scenes)
average over many scenes where the correct counterfactual answer is "no change" (e.g. a
comfortable lead at cruise speed). This set restricts measurement to scenes where removing the
target is expected to change the longitudinal decision.

Three iterations of the set definition, with the reason each was replaced:

- **set-v1** — four filters ANDed: model braking; exactly one agent firing any of
  {collision_course, follows, same_lane_ahead, merges, VRU}; no traffic-light record; constraint
  physically live (TTC < 5 s or closing > 0 ∧ Δs < 30 m). 76 scenes. Visual inspection: the
  12 VRU-only members were stationary sidewalk pedestrians with TTC = 8.0 (the "no predicted
  close approach" sentinel); membership rested on the closing-speed branch inflated by ego's own
  motion.
- **set-v2** — single filter: exactly one agent firing `collision_course`. Visual inspection
  showed prediction artifacts: a parked adjacent car firing at ds ≈ 0, parallel-lane traffic
  whose predicted path dips under the CPA threshold, an agent behind ego (ds = −7.3 m).
  `collision_course` is computed on GF-predicted futures and is listed in `UNRELIABLE_CHANNELS`.
- **set-v3 (final)** — two filters, both model-free:
  1. the expert braked: cached `decision_lon` label (derived from the logged human future,
     the H training label) ∈ {stop_quickly, stop_gently, slow_quickly, slow_gently};
  2. exactly one agent fires `follows` OR `same_lane_ahead` (computed from present kinematics,
     no predictions; ahead-of-ego by construction via the sign of path distance).
  Everything else is recorded per scene as a diagnostic, not filtered: traffic-light record,
  other agents carrying collision/merges/VRU, GT-future-confirmed collision on the target,
  the tested model's own decision, the target's causal-mass share.

Result: 90 scenes. Set membership uses only the expert label and cached channels, so the same
scenes apply to v3/v4/v6; per-model quantities (model_braking, mass_share) are recomputed by the
test script from the model under test.

**Reactive sub-slice** (added after the non-responder analysis, §5): expert braked *urgently* —
`decision_lon` ∈ {stop_quickly, slow_quickly}. Model-free. n = 29.

---

## 3. Interventions and metrics (`eval_binding_test.py`)

Five conditions per scene, single difference between them:

| condition | modification |
|---|---|
| base | none |
| edge | target's `follows` + `same_lane_ahead` channel entries set to false; agent keeps its other channels |
| graph | all of the target's channels set to false → agent leaves the causal graph via the gate; still visible to the frozen GF encoder |
| enc | target removed by encoder mask; neighbor predictions recomputed; channels also cleared |
| ctrl | farthest agent with no follows/lane_ahead channel removed at encoder level |

Metrics (after − before): **flip** = argmax over the 9 longitudinal classes changed (includes
within-SLOW moves such as stop_quickly → slow_quickly); **unbrake** = argmax was in the SLOW
family and left it; **dP_SLOW / dP_GO** = change in summed softmax probability of the family
(SLOW = stop_quickly, stop_gently, slow_quickly, slow_gently; GO = accel_quickly, accel_gently,
maintain); **dv_end** = plan speed change at 3.4–4.0 s; **dplan** = mean L2 between plans.

---

### Plain-language glossary (all conditions and metrics)

One scene, plain words: ego is braking; directly ahead there is a slow car; the model has
labeled it "ahead of me in my lane / I am following it". One car can carry several labels at
once (the same lead car is typically also labeled "on a crash path with me" and "at the same
intersection as me"). The scene is re-run several times; each run changes exactly one thing:

| name | what is changed | raw data (position/speed/history/map)? |
|---|---|---|
| base | nothing | unchanged |
| edge | the car's "ahead of me / following" labels are turned off; its other labels stay | unchanged |
| swapR | "ahead of me / following" labels are replaced with "car in the lane to my right"; its other labels (crash path, same intersection, ...) stay | unchanged |
| swapA | ALL of the car's worrying labels are replaced with the single label "car in the lane to my right" | unchanged |
| graph | ALL of the car's labels are turned off; with no label the gate excludes it — the car leaves the causal graph but is still visible to the frozen backbone | unchanged |
| enc | the car is deleted from the input entirely (encoder mask); the other cars' predictions are recomputed | the car is gone |
| ctrl | a far-away car with no worrying label is deleted instead of the lead | that far car is gone |

Metrics, plain words:

- **flip** — the model's top decision choice (among the 9 longitudinal classes) changed at
  all, including e.g. hard-stop -> gentle-slow.
- **unbrake** — the top choice was a braking class before and a non-braking class
  (maintain/accelerate) after.
- **dP_SLOW / dP_GO** — how much probability the model moved out of the braking family / into
  the go family (after minus before; -0.05 = 5 percentage points less braking probability).
- **dv_end** — how much faster the planned trajectory is at ~4 s (m/s, after minus before).
- **dplan** — average distance between the old and new planned paths (m).
- **"ctrl is null"** shorthand used in discussion = the ctrl row shows 0.0% flip and ~0.000 dP,
  i.e. deleting an irrelevant car changes nothing.

## 4. Results — v6 (`dodmeta_v6_relev` e18)

v4 and v3 on the same set: commands prepared, not yet run.

Model-braking slice (n = 82):

| condition | flip | unbrake | dP_SLOW | dP_GO | dv_end | dplan |
|---|---|---|---|---|---|---|
| edge | 18.3% | 4.9% | −0.043 | +0.043 | +0.30 | 0.89 |
| graph | 42.7% | 8.5% | −0.089 | +0.089 | +0.52 | 1.64 |
| enc | 46.3% | 9.8% | −0.091 | +0.091 | +0.53 | 1.66 |
| ctrl | 1.2% | 0.0% | +0.000 | −0.000 | +0.01 | 0.03 |

Reactive slice — expert braked urgently, model-free (n = 29):

| condition | flip | unbrake | dP_SLOW | dv_end |
|---|---|---|---|---|
| edge | 20.7% | 3.4% | −0.052 | +0.45 |
| graph | 62.1% | 6.9% | −0.076 | +0.76 |
| enc | 69.0% | 10.3% | −0.078 | +0.77 |
| ctrl | 0.0% | 0.0% | −0.000 | +0.00 |

Complement (expert braked gently, n = 53): enc flip 34.0%, unbrake 9.4%, dP_SLOW −0.098.

Model-conditioned variants of the reactive criterion (slicing by the tested model's own
baseline decision, measured before intervention): model-urgent (n = 27): edge 22.2% / graph
70.4% / enc 74.1% flip; expert-urgent ∩ model-urgent (n = 19): edge 21.1% / graph 73.7% /
enc 78.9% flip, ctrl 0.0%.

Mass-share split (model braking, median 0.57): mass ≥ median (n = 42) graph flip 52.4% /
enc 54.8%; mass < median (n = 40) graph 32.5% / enc 37.5%. Cleanest slice (braking ∧ no
second-cause agent ∧ no light, n = 58): edge 15.5% / graph 44.8% / enc 46.6% / ctrl 1.7% flip.

### Flip destinations (enc, reactive slice, 20 flips)

| from → to | n | family |
|---|---|---|
| stop_quickly → slow_quickly | 9 | within SLOW |
| slow_gently → slow_quickly | 3 | within SLOW |
| stop_quickly → slow_gently | 2 | within SLOW |
| slow_quickly → maintain | 2 | leaves SLOW |
| stop_quickly → accel_quickly | 1 | leaves SLOW |
| stop_gently → slow_quickly | 1 | within SLOW |
| stop_gently → slow_gently | 1 | within SLOW |
| stop_quickly → stop_gently | 1 | within SLOW |

17 of 20 flips remain inside the SLOW family; 3 leave it.

### Single-edge vs multi-edge targets (model braking)

Targets firing **only** follows/lane_ahead (n = 3): edge and graph are the identical input
modification by construction (those are all the target's channels), so their outputs are
bit-identical; enc differs at the 5th–6th decimal. Per-scene records:

| scene | target rels | base → after (all three conditions) | dP_SLOW | dv_end |
|---|---|---|---|---|
| #218 | lane_ahead | slow_quickly → slow_quickly (no flip) | +0.011 | −0.06 |
| #409 | lane_ahead | stop_gently → maintain | −0.440 | +3.00 |
| #443 | lane_ahead+follows | slow_gently → slow_quickly | +0.410 | +0.62 |

Targets also firing collision/sharesIntersection/etc. (n = 79): edge 16.5% / graph 41.8% /
enc 45.6% flip.

### Responder vs non-responder comparison (enc, model-braking slice; `eval_binding_analyze.py`)

38 responders (argmax changed) vs 44 non-responders (argmax unchanged under full deletion):

| variable (mean / median) | responders | non-responders |
|---|---|---|
| ego speed [m/s] | 4.43 / 4.07 | 3.37 / 2.99 |
| ego speed change, last 2 s [m/s] | −1.75 / −1.87 | −1.48 / −1.36 |
| base P(SLOW) | 0.96 / 1.00 | 0.98 / 1.00 |
| target mass share | 0.64 / 0.62 | 0.56 / 0.54 |
| target Δs [m] | 17.3 / 15.7 | 16.7 / 14.8 |
| target TTC [s] | 4.15 / 3.60 | 4.74 / 4.30 |
| target closing [m/s] | 3.08 / 3.21 | 2.15 / 2.41 |
| second-cause agents | 0.42 / 0 | 0.34 / 0 |
| traffic light present | 5% | 9% |
| dP_SLOW (enc) | −0.16 | −0.03 |
| dv_end (enc) [m/s] | +0.90 | +0.20 |
| base decision | stop_quickly 17, stop_gently 14, slow_gently 4, slow_quickly 3 | stop_gently 29, slow_gently 8, stop_quickly 4, slow_quickly 3 |

BEV inspection of non-responders (`viz_out/nonresponders.png`): stationary target (0.0 m/s)
11–15 m ahead among other stopped vehicles at an intersection; ego at 1.6–3.5 m/s. This
comparison is what motivated the reactive sub-slice in §2.

---

## 5. Results — v4 (`dodmeta_v4_relbottleneck` e18), same set (2026-08-24)

Same `binding_set_v6.json`, same script; `--rel_bottleneck 1 --rel_evidence 0`.
Per-scene records: `binding_test_v4.json`. Model-braking count: 84/90 (v6: 82/90 — the two
models brake on slightly different scenes; slices are per-model as designed).

Model-braking slice (n = 84):

| condition | flip | unbrake | dP_SLOW | dP_GO | dv_end | dplan |
|---|---|---|---|---|---|---|
| edge | 15.5% | 2.4% | -0.040 | +0.040 | +0.16 | 0.85 |
| graph | 45.2% | 10.7% | -0.115 | +0.115 | +0.46 | 1.50 |
| enc | 45.2% | 10.7% | -0.116 | +0.116 | +0.46 | 1.51 |
| ctrl | 0.0% | 0.0% | +0.000 | -0.000 | -0.00 | 0.02 |

Reactive slice — expert braked urgently (n = 29): edge 27.6% / graph 65.5% / enc 65.5% /
ctrl 0.0% flip; graph dP_SLOW -0.086, dv_end +0.57.
Situational complement (n = 55): graph/enc flip 34.5%.
Mass split (median 0.50): mass >= median graph flip 48.8%; below 41.5%.
Cleanest slice (n = 59): graph/enc 52.5%, ctrl 0.0%.

v4-vs-v6 on the shared slices (flip):

| slice / condition | v4 | v6 |
|---|---|---|
| reactive graph | 65.5% | 62.1% |
| reactive enc | 65.5% | 69.0% |
| reactive edge | 27.6% | 20.7% |
| main enc | 45.2% | 46.3% |
| ctrl (all slices) | 0.0% | 0.0-1.2% |

At these sample sizes (n = 29 reactive) the v4-v6 differences are 1-2 scenes and not consistent
in direction across slices. In v4, graph and enc produce identical flip counts in every slice;
dP values differ in the third decimal.

## 6. Results — v3 (`dodmeta_v3_egoline` e13, stable model), same set (2026-08-24)

Same `binding_set_v6.json`, same script; `--rel_bottleneck 0 --rel_evidence 0`. Run on the
`relation-bottleneck` working tree — valid because both flags off build the v3 architecture
exactly (verified: zero missing/unexpected keys on load; flag-gated code paths). Additionally
bit-exact-checked against the `channels` branch code (git worktree, same checkpoint, same batch,
2026-08-24): attention masks M_cas / M_cas_typed identical to the bit (max diff 0.0); psi
logits / trajectories differ only by float summation order (max diff 1.5e-05, from the
sum-then-sum rewrite in the bottleneck commit); decision argmax identical in 16/16 scenes. Per-scene
records: `binding_test_v3.json`. Model-braking count: 85/90.

Model-braking slice (n = 85):

| condition | flip | unbrake | dP_SLOW | dP_GO | dv_end | dplan |
|---|---|---|---|---|---|---|
| edge | 20.0% | 4.7% | -0.037 | +0.037 | +0.26 | 0.83 |
| graph | 42.4% | 9.4% | -0.090 | +0.090 | +0.43 | 1.19 |
| enc | 43.5% | 9.4% | -0.091 | +0.091 | +0.44 | 1.20 |
| ctrl | 0.0% | 0.0% | -0.001 | +0.001 | +0.03 | 0.10 |

Reactive slice (n = 30): edge 30.0% / graph 60.0% / enc 60.0% / ctrl 0.0% flip;
graph dP_SLOW -0.071, dv_end +0.51. Situational complement (n = 55): graph 32.7% / enc 34.5%.
Mass split (median 0.53): >= median graph 47.7% / enc 50.0%; below 36.6%.
Cleanest slice (n = 59): graph 44.1% / enc 45.8%, ctrl 0.0%.

### Three-way comparison (flip)

| slice / condition | v3 (stable) | v4 (+bottleneck) | v6 (+rel_evidence) |
|---|---|---|---|
| reactive edge | 30.0% | 27.6% | 20.7% |
| reactive graph | 60.0% | 65.5% | 62.1% |
| reactive enc | 60.0% | 65.5% | 69.0% |
| main graph | 42.4% | 45.2% | 42.7% |
| main enc | 43.5% | 45.2% | 46.3% |
| situational enc | 34.5% | 34.5% | 34.0% |
| ctrl, all slices | 0.0% | 0.0% | 0.0-1.2% |

Cross-model differences are 1-3 scenes per cell with no consistent direction. graph ~= enc holds
in all three models (largest gap: v6 reactive, 62.1 vs 69.0). The reactive/situational split and
the null control replicate in all three. These properties therefore do not depend on
`rel_bottleneck` or `rel_evidence`; they are present in the stable v3 model.

## 7. Type-swap intervention — do(type = wrong type) (2026-08-24)

Question: does the *identity* of the relation label steer the decision, or are typed entries
generic slots whose content does the work? The removal tests (sections 4-6) cannot answer this:
cutting an entry removes both the label and the entry.

Two added conditions in `eval_binding_test.py` (agent stays in scene and in graph; node
features, evidence, mass pathway untouched; only the boolean labels change):

- **swapR**: target's `follows`+`same_lane_ahead` labels replaced by `adjacent_right`; its other
  caution labels (collision/sharesIntersection/...) remain.
- **swapA**: ALL of the target's caution labels (follows, lane_ahead, collision,
  sharesIntersection, merges, VRU) replaced by `adjacent_right` — the lead relabeled as a
  benign side car.

Reference points: base, graph (agent removed from the causal graph), ctrl.

Reactive slice (expert braked urgently; flip%):

| condition | v3 (n=30) | v4 (n=29) | v6 (n=29) |
|---|---|---|---|
| edge | 30.0% | 27.6% | 20.7% |
| swapR | 33.3% | 27.6% | 24.1% |
| **swapA** | **63.3%** | **58.6%** | **48.3%** |
| graph | 60.0% | 65.5% | 62.1% |
| enc | 60.0% | 65.5% | 69.0% |
| ctrl | 0.0% | 0.0% | 0.0% |

Model-braking slice (flip%): swapA v3 38.8 / v4 33.3 / v6 31.7 vs graph 42.4 / 45.2 / 42.7;
swapR 16.5 / 17.9 / 18.3. dP_SLOW (reactive): swapA v3 -0.051 / v4 -0.067 / v6 -0.015 vs graph
-0.071 / -0.086 / -0.076. unbrake (reactive): swapA 6.7 / 6.9 / 0.0 vs graph 10.0 / 10.3 / 10.3.
Per-scene records: `binding_test_v3.json` / `_v4.json` / `binding_test_results.json`-era v6 file
regenerated as `binding_test_v6.json`.

Observed in all three models:
- swapA flip is within 5-14 points of graph flip and 48-63 points above base (base = 0 change
  by definition). In swapA the agent's node features still enter the attention through the
  `adjacent_right` entry.
- swapR flip is within 4 points of edge flip. In swapR the target keeps its
  collision/sharesIntersection labels (present on the target in 79/82 scenes).
- swapA unbrake < graph unbrake in all three models.
- ctrl is 0.0% flip in all three models.

Note on scope relative to the earlier population-level edge-ablation measurement
(pre-bottleneck, "typing carries no information"): that measurement cut one label while sibling
labels remained; this measurement replaces the labels. The two operate on different quantities.

## 8. Matched-control + injection battery — swapC / inject / ginj (2026-08-25)

Three conditions added to `eval_binding_test.py` (same set/script; per-scene records
`binding_test_v3b.json` / `_v4b.json` / `_v6b.json` — these re-run all earlier conditions too;
old jsons untouched):

- **swapC**: matched control for swapA — target's caution labels collapsed to a SINGLE
  surviving entry, but the survivor is `collision_course` (caution) instead of
  `adjacent_right` (benign). Identical structural perturbation; only the surviving label's
  family differs.
- **inject**: reverse swapA — the ctrl agent (farthest, no ahead-relation) is GIVEN
  follows+lane_ahead+collision labels; scene otherwise untouched.
- **ginj**: graph + inject — real cause leaves the causal graph, fake cause enters it.

Model-braking slice (flip% / unbrake% / dP_SLOW):

| condition | v3 | v4 | v6 |
|---|---|---|---|
| swapA | 38.8 / 3.5 / −0.044 | 33.3 / 6.0 / −0.056 | 31.7 / 1.2 / −0.023 |
| swapC | 34.1 / 5.9 / −0.056 | 33.3 / 4.8 / −0.063 | 46.3 / 4.9 / −0.083 |
| graph | 42.4 / 9.4 / −0.090 | 45.2 / 10.7 / −0.115 | 42.7 / 8.5 / −0.089 |
| inject | 7.1 / 1.2 / +0.000 | 11.9 / 0.0 / +0.004 | 3.7 / 1.2 / −0.001 |
| ginj | 30.6 / 5.9 / −0.056 | 54.8 / 14.3 / −0.137 | 56.1 / 8.5 / −0.112 |
| ctrl | 0.0 | 0.0 | 1.2 |

**Verdict — the label-semantics reading of §7's swapA does NOT survive its matched control:**

1. **swapC ≈ swapA** in flips in all three models, and on the semantic metrics (unbrake,
   dP_SLOW, dv_end) swapC unbrakes MORE than swapA in all three — the wrong direction for
   family-level label semantics. §7's swapA effect measured the structural perturbation
   (entries removed / mass redistributed), not the family of the surviving label.
2. **inject ≈ 0** everywhere: a caution label wrapped around benign kinematics (far,
   non-closing agent) does not create braking. No label superstition.
3. **ginj INVERTS in v4/v6** (55–56% flips > graph's 43–45%; dP_SLOW more negative than
   graph): with the real cause gone, caution slots filled with benign far-agent content push
   the decision further toward GO than leaving them empty. In v3 (pooled f_cas) ginj 30.6% <
   graph 42.4% — the bottleneck models transmit slot content to the decision more directly.

One sentence consistent with all nine conditions: **labels are ROUTING, content is MEANING.**
The decision tracks the kinematic content carried into (and removed from) the typed slots; the
label determines where content flows, not what it asserts. Under this reading inject≈0 and the
ginj inversion are semantically CORRECT planner behavior (a far, receding "lead" licenses go),
and swapA/swapC/graph all measure content removal.

Paper consequences: (i) INTERVENE claim stays support-level — the edge < graph ≈ enc dose
ladder with ctrl≈0 is unchanged and robust across all three models; (ii) the swap/inject
battery becomes the analysis section ("what the types mean to the network: routing, not
standalone semantics") — do NOT claim family-level label semantics from swapA; (iii) metric
note: 9-class `flip` inflates under any perturbation (within-SLOW reshuffles) — lead with
unbrake / dP_SLOW / dv_end for semantic claims, keep flip as a sensitivity row.

Open probe (cheap): ginj variant that relabels a NEARBY benign agent instead of the farthest —
separates "content read correctly through a caution slot" from "OOD dilution" by putting
close-but-parallel content into the slot.

## 9. dec_moe — karar-kapili expert decoder (tasarim onayi + implementasyon 2026-08-25)

Amac: b*-swap teshisinin ("karar readout ama kol degil"; compliance %0-33) mimari cozumu —
CIL 2018 gozlemi: komut input verilirse ag yok sayar, branch SECMELI. MoE literatur taramasi
(STR2/DriveMoE/EMoE: hepsi performans icin, hicbiri karar-plan hizalamasi icin) ayri kayitta.

Onaylanan tasarim (kullanici, 2026-08-25): **cift eksen + her iki eksende azaltilmis sozluk.**
- Sozluk 5x5 (cache 9x7 KALIR, egitim aninda remap): LON5 {remain_stopped, stop, slow, accel,
  maintain} (q/g katlanir, reverse->remain_stopped; 20k'da 0 ornek); LAT5 {turn_l, turn_r,
  lc_l, lc_r, none} (inlane->none). Yan kazanc: psi kolaylasir -> routing dogrulugu artar.
- **Faktorlu gating** (joint 3x3=9 expert DEGIL — HOLDxRIGHT %0.5 aclik): lat AILESI
  {left %31.8 / right %16.2 / none %51.9} q_enh dalini secer (yon/mod hedefi); lon AILESI
  {brake %9.7 / hold %22.5 / cruise %67.8} predictor dalini secer (hiz profili yasasi).
  3+3 modul, 9 kombinasyon; her modul kendi ekseninin TUM verisini gorur. mode_query + cross
  paylasilir. Aileler binding SLOW/HOLD/GO ile hizali.
- Routing: egitimde GT (oracle/teacher forcing — b* f_cas'in deterministik fonksiyonu oldugu
  icin slot bilgisizdi; GT degil -> head okumak zorunda), inference'ta argmax psi ailesi.

Implementasyon: **channels branch'inde** (rel_evidence once relation-bottleneck'e commit'lendi,
fb63d68). `--dec_moe 1` flag'i; dosyalar: decision_labels.py (LON5/LAT5 map+CE agirliklari+
aileler), causal_graph.py (head dallari + planner teacher-forcing), train_planner.py (remap +
ctor + arg), run_nuplan_test.py + causal_refiner_planner.py (plumbing; yalniz refiner).
lon_merge ile birlesmez (assert). Smoke PASS (24-ornek gercek batch, cuda:1): sekiller dogru,
gradyan yalniz aktif dallara akiyor, eval-mode b*-routing calisiyor; 13.75M param.

Egitim HENUZ BASLATILMADI (kullanici karari bekleniyor). Kosum adi (kullanici, 2026-08-25):
**`v3_moe`** = v3 config + `--dec_moe 1`, 20 epoch, cuda:1. Dogrulama sirasi:
val minADE + psi AILE-dogrulugu -> b*-swap (asil hedef: compliance/dose-direction) -> binding
(eval scriptleri 5x5'e uyarlanacak: SLOW_IDX vb. sinif-adi sabitleri) -> CLS hard.

## 10. b*-swap uclu ablasyon — v3 / v3_tf / v3_moe (2026-08-25)

Uc kosum, ayni `eval_bswap.py` (dec_moe destegi eklendi: relabel 9x7 -> 5x5 katlama),
ayni 1118 val sahnesi. v3_tf = `--dod_tf 1` (YALNIZ teacher forcing: egitimde embedding'e
GT (lon,lat); 9x7 aynen, dal yok, mimari v3 ile ozdes). Checkpoints: v3 e13 0.7372,
v3_tf e16 1.0847, v3_moe e16 1.0833.

| olcum | v3 | v3_tf (TF-only) | v3_moe (TF+dal+5x5) |
|---|---|---|---|
| agreement lon/lat/joint | 75.8/83.3/64.1 | 83.7/87.4/73.5 | 98.1/90.9/89.3 (5-sinif) |
| remain_stopped zorla dv_end | +0.01 | -1.28 | **-4.91** |
| stop zorla dv_end | ~0.00 | -1.46/-1.52 | -1.41 |
| slow zorla dv_end | ~-0.02 | -0.79/-0.88 | -1.16 |
| accel zorla dv_end | +0.02 | +1.28/+1.39 | +1.63 |
| doz-yon stop/slow | 45.8% (sans) | **80.6%** | 65.7% |
| doz-yon accel | 67.5% | 92.9% | **100.0%** |
| doz-yon turn L/R | 54/33 | 46/43 | 63/35 |
| val minADE | **0.7372** | 1.0847 | 1.0833 |

**Verdict:**
1. **LON kolunu TF tek basina uretiyor.** Isaretli, buyuk dv tepkileri ve 81-93% doz-yon
   dallarsiz geliyor — kolun kaynagi GT etiketin bilgilendiriciligi (slot artik f_cas'in
   fonksiyonu degil), mimari salter degil.
2. **Dallarin marjinal katkisi:** ekstrem tepki (remain_stopped -4.9 m/s, plan 19 m — HOLD
   uzmani gercekten duran plan uretebiliyor), accel monotonlugu %100, agreement 98 (kismen
   5-sinif etkisi). Maliyeti yok ama kolun ana kaynagi degil.
3. **Bedel ikisinde de AYNI (~1.08 val minADE, v3'un 0.737'sine karsi):** exposure bias'in
   kaynagi GT-kosullamanin kendisi; dallar ek bedel getirmiyor. Iki varyant da su haliyle
   surus dogrulugunda v3'un gerisinde — kol kazanildi, dogruluk odendi.
4. **LAT her uc modelde de kirik** (doz-yon ~sans; v3_tf'te turn_left zorla -> hd -0.52,
   YANLIS yon). Muhtemel kismi aciklama: lat zorlamasi cogu sahnede baglamla fiziksel
   celiskide (yol geometrisi yonu dikte eder) — mimari kusurdan ayirt etmek icin
   "yapilabilir lat" alt-kumesinde olcum gerekir.

**Sonraki dogal adim:** scheduled sampling (egitimde p olasilikla GT yerine psi tahminiyle
kosullama, p epoch'la artar) — en basit tasiyici v3_tf uzerinde: mimari v3 ile ozdes kalir,
hedef kolu koruyup minADE makasini kapatmak. Karar kullanicida.

## 11. Fizibilite tavani + any-mode teshisi (2026-08-25)

`eval_bswap.py`'a eklendi: (a) fizibilite dilimi — zorlama yalniz kinematik uyulabilir
sahnelerde sayilir (remain_stopped: v0<0.5; stop: v0>=1; slow: v0>=2 m/s; v0 = taban planin
ilk 0.5 s hizi); (b) any-mode — 6 moddan HERHANGI biri zorlanan aileye uyuyor mu (best-mode
>> any-mode farki = MOD SECICI suclu). LON, aile uyumu (best|feas / any&feas):

| zorlanan | v3 | v3_tf | v3_moe |
|---|---|---|---|
| remain_stopped | 17.9 / 61.5 | 60.6 / 98.5 | **100.0 / 100.0** (n_feas 71) |
| stop | 7-14 / 11-21 | 13-24 / 24-35 | 8.5 / 23.4 |
| slow | 12-14 / 59-70 | 23-26 / **91-93** | 41.5 / **95.5** |
| accel | 14-15 / 62 | 43-46 / 72-76 | 36.9 / 71.9 |
| maintain | 11.9 / 81.1 | 48.3 / **96.3** | 43.3 / 89.7 |

LAT any-mode (best / any): v3_tf turns 5.3/12.0 ve 2.0/4.3; lc_r 19.7/26.5; none 29.1/38.5.
v3_moe benzer. Lat MOD SETINDE bile itaatkar plan yok.

**Verdict:**
1. **Fizibilite tavani gercek:** slow zorlamasi ham %12-26'dan fizibil dilimde any-mode
   %91-96'ya cikiyor; moe'de fizibil remain_stopped best-mode %100 (71/71).
2. **LON itaatsizliginin ana suclusu MOD SECICI:** v3_tf'te any&feas - best|feas farki
   slow'da 67 puan (93 vs 26), maintain'de 48 puan (96 vs 48). Decoder itaatkar modu
   URETIYOR, skor basi karar-kor oldugu icin baglam-tercihli modu seciyor. Egitimsiz fix:
   zorlama/CF sirasinda karara-uygun modlar arasindan en yuksek skorlusu secilir ->
   compliance ~any&feas seviyesine cikar (remain ~100, slow ~91-96, maintain ~90-96,
   accel ~72-76). Deployment karsiligi mesru: refiner zaten aday setinden secim yapiyor.
3. **stop GERCEK decoder acigi:** any&feas bile %23-35 — mod cesitliligi "hareket halinden
   tam durma" alternatifi icermiyor (egitimde stop_* %2.3). Etiket birlestirme tek basina
   cozmez; skor-basini karara sartlama / stop agirliklandirma (training) veya deployment'ta
   lattice adaylari gerekir.
4. **LAT yapisal olarak kirik** (any-mode'da bile) — mod seticinde donus alternatifi yok;
   lat compliance iddiasi birakilmali, lat yalniz readout (agreement) olarak raporlanmali.

Acik kararlar: (i) karar-tutarli mod secimi eklenip tablo yeniden kosulacak mi;
(ii) etiket sadelestirme (kullanici onerisi: lon 4 = stop/slow/accel/maintain, reverse yok;
lat 5 = turn_l/turn_r/to_left/to_right/none) — v2 egitiminde SS ile birlikte;
(iii) scheduled sampling. Karar kullanicida.

## 12. v3_latmoe (v2) — tasarim + implementasyon (2026-08-25)

4x5 gorunumlu son bswap koslarindan cikan tasarim (kullanici onayi): **lat-anahtarli cikis
expert'leri.** Gerekce OLCULMUS lateral mode collapse: any-mode lon %55-96 (mod seti hiz
alternatifi iceriyor) vs any-mode lat %4-20 (donus alternatifi YOK) -> uretec expert'i
coken eksene konur (SOTA'nin MoE gerekcesiyle ayni: mode averaging/collapse).

Recete (`--lat_moe 1`): (i) 4x5 GERCEK sozluk — LON4 stop{remain,stop_q,stop_g}/slow/accel/
maintain (reverse->stop, 0 ornek), LAT5V turn_l/turn_r/to_left{lc_l,inlane_l}/to_right/none;
sinif=aile; cache 9x7 kalir, remap egitimde (CE agirliklari 20k sayimlardan). (ii) CA sonrasi
LAT SINIFI basina 5 GMM expert (katlama yok, kullanici tercihi — CF'te bilgi kaybi olmasin);
q_enh + cross paylasilan; lon dallanmaz (embedding+TF; v3'te bile lon any-mode yuksekti).
(iii) TF: egitimde routing+embedding GT'den. (iv) `--ss_max p`: scheduled sampling, GT yerine
tahminle kosullama olasiligi 0->p lineer rampa (exposure-bias makasina karsi; olculen makas:
train 0.48-0.55 / val 1.08). (v) degerlendirmede karar-tutarli mod secimi (eval_bswap par.4).
Expert veri paylari (20k): turn_l %30.7 / turn_r %14.8 / to_left %3.8 / to_right %4.5 /
none %46.3; lon4: stop %24.8 / slow %7.4 / accel %36.3 / maintain %31.6.

Implementasyon (channels working tree): decision_labels.py (LON4/LAT5V), causal_graph.py
(head lat_moe dallari + planner TF/SS), train_planner.py (--lat_moe/--ss_max + remap + epoch
rampasi), run_nuplan_test/causal_refiner_planner/eval_bswap plumbing. Smoke PASS (grad yalniz
aktif lat dallarina, SS calisiyor, 13.98M param). CF off-manifold notu: dayatilan manevra
yolda yoksa plan yol agindan tasar — FW-1 CF skorlayicisinin "unsafe/infeasible" etiketi tam
olarak bunu olcer (kusur degil, CF ciktisi).

## 13. v3_latmoe sonuclari — LAT KOLU CALISIYOR (2026-08-25)

`v3_latmoe` e12 (val minADE **0.7998**; SS 0.5 makasi kapatti: saf-TF 1.08 -> 0.80, v3 0.737).
4x5 bswap (1118 sahne; argmax -> karar-tutarli):

- **LAT compliance (onceki tum modeller %2-20 idi):** turn_left **92.3 -> 98.7**, turn_right
  **68.2 -> 96.7**, to_left 8.6 -> 31.0, to_right 16.2 -> 31.8, none 37.4 -> 48.6.
  Lat-anahtarli GMM expert'leri lateral collapse'i kirdi (kullanici hipotezi dogrulandi).
- **LON compliance TF seviyesinde korundu** (SS'e ragmen): stop 17.0->25.5, slow 5.7->41.9,
  accel 40.8->71.3, maintain 40.3->87.7 (v3_tf: 30.3/45.3/70.1/96.3; moe stop 45.8 —
  lon expert'i olmadigi icin stop latmoe'de daha dusuk).
- Agreement: 95.9/86.9/83.5 -> 99.6/92.6/91.6.

Zayif kalanlar: stop (lon expert yok — secenek: skor-basi karar-sartlama veya moe'nun HOLD
dersini lon'a tasima), to_left/to_right (%31 — en kucuk veri, yine de 3-6x onceki).

**CLS reduced (2026-08-25 21:09 kosumu, e12): 0.8409** — collision 1.000, drivable 0.921,
TTC 0.974, comfort 0.947, progress 0.813. GF 0.8199'un +0.021 ustunde, v3 soyu bandinin
(0.833-0.858) icinde -> DO-NO-HARM GECTI (argmax/karar-kor skorcuyla bile). Viz:
`viz_bswap.py` -> viz_out/bswap/ (taban + zorlanmis-karar planlari, karar-tutarli secim).
Sirada: test14-hard 272 (paper satiri), binding bataryasi latmoe'de, skor-sartlama karari.

## 14. NOT — v4 adayi (kullanici: "not alalim simdilik", 2026-08-26)

Drift teshisi + latforce panelleri sonrasi mutabik kalinan ama HENUZ BASLANMAYAN paket:
1. **Hakem duzeltmesi:** turn/LC siniri koridor-bazli — |d_lat delta| buyuk VE uc heading
   koridora PARALEL -> lane_change; heading koridordan kalici ayrilmis -> turn. Once eval
   relabeler'inda + GT etiket-uyusma dogrulamasi; temizse egitim etiketine.
2. **Lat sozlugu:** {turn_l, turn_r, lane_change_l, lane_change_r, none} — to_* kalkar
   (muglak orta sinif), inlane -> none (karar degil suruklenme), LC ilanda birinci sinif.
   5 expert; LC dallari icin oversampling ~x8 sart (lc_l 235 / lc_r 291 @20k — baska veri yok,
   oversampling = mevcut verinin yeniden tartimi).
3. **Ayni sampler'a stop/slow sahneleri de** (lon zayifligi icin; asagidaki teshis).
Mevcut kanit: forced to_* ciktilarinin dagilimi expert diyetiyle tutarli — to_* egitim verisi
~%70 inlane oldugu icin uretim agirlikla inlane olcekli (>=2 m LC-olcek yalniz ~%21).

## 8-old. Pending on this set

- v3 and v4 runs completed 2026-08-24 (sections 5-6); three-way table in section 6.
- Type-swap runs (all three models) completed 2026-08-24 (section 7).
- Plan-horizon speed profile (dv at 0–2 s vs 7.4–8 s after removal) — proposed check on the
  flip-destination pattern; not yet implemented.
- v6 closed-loop CLS (test14-hard) not measured.
