# Talk Notes — The Gap and How We Filled It (walked over existing figures)

No-slides talk guide. Each section: **[SHOW: file]** + what to say + the numbers to quote.
Deliberate order: gap → mechanism (A → H) → positioning vs the two rival families →
evidence → honest limits.

---

## 0. Opening (60 seconds, no image)

> "Learned planners drive well, but they cannot answer two questions:
> **'Because of WHOM did you make this decision?'** and
> **'How do I know your answer is true?'**
> Our work makes both questions answerable — not by generating text about the plan,
> but by changing the planner's INTERNAL STRUCTURE."

---

## 1. The gap (no image, three points)

1. **No decision semantics** — the planner emits only a trajectory; "what did it decide"
   has no readable answer, just a latent.
2. **No verifiable reasoning** — attention/masks get labeled "causal" but are bound by no
   rule; they may look anywhere.
3. **Unfalsifiable attribution — and the field's standard metric cannot catch it.** Our own
   measurement (Finding 5): **a chance-level-mask model scores CLS 0.8364 while a
   better-mask model scores 0.8059** — closed-loop score is blind to mask correctness.
   So "our score is good, therefore its reasoning is sound" is an invalid inference.

> Transition: "Causal-Planner (IROS'25) was a step in this direction — in our faithful
> reproduction its learned masks measure at chance level (1.03–1.26× vs random). A learned
> mask alone is not enough. Our answer: bind the mask's SKELETON to symbolic rules."

---

## 2. Approach A — predicate-gated, relation-typed causal attention

**[SHOW: channels_bev.png — channel visualization]**
> "From a symbolic predicate knowledge graph (a colleague's separate work, which we cite)
> we transcribed 11 agent + 8 map RELATION CHANNELS — follows, collision-course, merges,
> adjacent, inLane, route-corridor... Thresholds come from the KG's published constants.
> This layer is fully deterministic and auditable: a channel either fires or it doesn't,
> and the reason is readable."

**[SHOW: viz_out/typed_noresid_bev.png — typed mask]**
> "The key move: we do NOT feed these channels to the model as input features — we tried
> that, it does nothing (1.24× → 1.26×, inert). Instead they define the SUPPORT of the
> causal attention: an agent with no fired predicate structurally CANNOT enter the softmax,
> and every entry that can lives as a named (agent, relation) pair. The rules say WHO MAY
> matter and WHY; learning distributes HOW MUCH."
- Number: structure alone, with zero new losses, lifts selection **1.24× → 1.45×**.
- Number: calibration record — corr(M_cas, Δplan) across the lineage **0.55 → 0.63**.

---

## 3. Approach H — meta-action decision conditioning

**[SHOW: viz_out/decision_labels.png — taxonomy examples]**
> "The second gap was the decision. We upgraded the slot that conditions the trajectory
> head from a 5-class geometric label to a factored 9-longitudinal × 7-lateral META-ACTION
> taxonomy adapted from nuReasoning (2026). Labels are 100% model-free, derived from the
> GT trajectory — longitudinal from the first 4 s of the speed profile, lateral from the
> corridor-relative motion over the full horizon. The old 5 classes are a strict
> coarsening of the new lateral set (the cross-tab is exact) — so the decision slot stays
> SINGLE; there is no parallel path."

**[SHOW: viz_out/dodmeta_v2_bev.png — scenes with decision cards]**
> "Now the planner's decision is readable in every scene: 'accelerate gently + turn left
> (0.99)'. And the confidences are calibrated — the old 5-class head saturated at 1.00
> everywhere."
- Number: decision accuracy lat **0.85** / lon **0.69** (weighted CE — even the 0.8%-rare
  lane-change-left class gets 7/9 right).

---

## 4. Everything at once — the paper figure

**[SHOW: viz_out/paper_strip.png]**
> "The whole chain in one image: light-band map, RELATION COLORS (green inLane, blue
> successor, purple adjacent, pink route), heatmap causal agents, the relation readout
> under each panel ('A ego→agent: collide .48 · intersect .26'), the decision top-left,
> the blue plan.
> **Every piece of the explanation is itself a variable that COMPUTES the plan.**"

> The pitch sentence (memorize): **"A VLM's explanation is text generated ABOUT the plan —
> fluent, possibly wrong, possibly contradicting the plan. Our explanation IS the
> mechanism that produces the plan: it CANNOT LIE; at worst it is wrong together with
> the planner."**

---

## 5. Positioning — why not a VLM/VLA, why not rule-based? (know this cold)

### vs VLM/VLA planners (DriveVLM, AutoVLA, nuVLA/nuReasoning line, Alpamayo…)

Their real strengths (concede first — it disarms):
- Open-world semantics and commonsense; long-tail scene understanding; language interface;
  scale with data.

Their weaknesses on OUR axis — and our corresponding advantage:
1. **Faithfulness:** their rationale is a parallel text generation with no enforced
   coupling to the action — it can hallucinate and contradict the plan. Concrete ammo:
   nuReasoning's own annotation pipeline reports only **84.7% (decision) / 76.1%
   (counterfactual) agreement with human experts** before correction — that is the
   hallucination rate of the annotator itself. Ours: the explanation variables (fired
   channels, typed weights, b*) are the actual inputs of the trajectory computation —
   faithful by construction, and we MEASURE the coupling (agreement 76/82%, plus an
   intervention suite).
2. **Testability:** you cannot intervene on a VLM's chain-of-thought and check the action
   responds; there is no experiment. Ours is built for intervention: RemoveNonCausal with
   a distance-matched control, dose-response curves, b*-swap.
3. **Closed-loop:** the VLA reasoning line is overwhelmingly open-loop (nuReasoning states
   this as its own limitation). We evaluate closed-loop CLS in nuPlan.
4. **Cost:** they need billion-parameter models at inference and VLM+human annotation
   pipelines for supervision. Our module is small, runs at planner rate on a frozen
   backbone, and every label is free (GT-derived, rule-computed) — zero annotation.

Honest concession if pushed: we do NOT get their open-world coverage — our predicate
vocabulary is closed. The approaches are complementary; for a SAFETY claim, verifiability
is the non-negotiable part, and that is the part they lack.

### The two deep questions behind this (know these cold)

**Q-A: "Isn't a VLA just VLM + a trajectory head — same as yours? Its reasoning also
biases the head."**
> Partially true — and the difference is precisely what 'faithful by construction' means.
> Our claim is **IDENTITY, not correctness**: the three elements of our explanation are
> not a second artifact ABOUT the computation, they are intermediate values READ OFF the
> computation itself — (1) a non-fired agent structurally CANNOT enter the softmax (hard
> mask, not learned); (2) the reported typed weight IS the softmax coefficient that builds
> f_cas; (3) the reported decision IS the tensor fed to the head. No generation step in
> between — no place for hallucination to enter. In a VLA the rationale is a separate
> decode: no architectural identity ties any sentence in the text to any coefficient in
> the action pathway. Concrete: **nuVLA disables text generation at inference and planning
> works unchanged** — their own result shows the rationale is not a required part of the
> computation.
> Where the architectures DO rhyme is the coupling-strength question ("does the decision
> actually steer the head?") — and the difference there is that ours is ANSWERABLE: b* is
> a discrete, intervenable variable, we ran the swap, measured weak coupling, named the
> mechanism, reported it. A VLM has no equivalent experiment (the head reads hidden
> states, not the decoded text).
> Paper wording, verbatim: **"faithful-by-construction SUPPORT and READOUT; MEASURED
> coupling"** — we never spread the by-construction label over everything; drawing that
> line ourselves is exactly what the VLM line cannot do.

**Q-B: "Why put the predicates INSIDE the network? Running them in parallel would give
two outputs — isn't that the same?"**
> No — a parallel predicate report explains the WORLD, not the PLANNER. It would keep
> printing "collision course with j" even while the planner ignores j entirely: plausible
> but unfaithful — the exact failure mode we criticize in VLMs, and worse, not even
> generated from the model's internals. (That parallel artifact already exists: it is the
> KG extractor itself — our colleague's paper. Our contribution is exactly the step from
> world-description to model-explanation.)
> And the ladder is fully MEASURED, which is the talk's backbone:
> - **Parallel** (outside) → describes the world, no bond to the planner; the learned mask
>   stays at chance (1.03–1.26×, our CP reproduction measurement);
> - **Input features** (inside, not structural) → inert: 1.24× → 1.26×;
> - **Structure** (our design) → works: selection 1.24× → 1.45×, calibration 0.55 → 0.63,
>   CLS +0.029 over vanilla, and "agent j's weight flowed through the follows channel" is
>   now a real intermediate value of the model.
> One sentence: *"Printing predicates is not enough, feeding them is not enough — they
> must become the skeleton of attention; we measured the first two rungs, the third one
> works."*

### vs rule-based planners (IDM/MOBIL, PDM-Closed, RSS-style rule checking, rule hierarchies)

Their real strengths (concede first):
- Interpretable and verifiable BY DESIGN; predictable; extremely strong closed-loop
  baselines (PDM-Closed).

Their weaknesses — and our advantage:
1. **Rules DECIDE → brittleness.** Hand-tuned costs and priorities cap performance in
   interactive scenes; the long tail of rule combinations is unmanageable. In our design
   **rules only define the STRUCTURE** (who may matter, and under which named relation);
   learning does the allocation and the control. Scaffold, not driver.
2. **They explain the wrong object.** A rule-based stack explains its heuristic — fine —
   but the moment you add learning back (every modern hybrid does), the learned part is a
   black box again. Rule-constrained OUTPUT (RSS, hierarchies) can veto a trajectory but
   cannot tell you why the model attended to an agent. We put the rules INSIDE the
   attention, where the attribution lives.
3. **Performance:** we keep learned-planner behavior and beat our learned baselines
   closed-loop (vanilla GF 0.8199 with a collision → ours 0.8487, collision-free) —
   a purely rule-based configuration of the same stack is not on that Pareto point.

Honest concession if pushed: we provide no HARD safety guarantee the way RSS does — a
rule-based safety envelope stays complementary (our lattice refiner already plays a mild
version of that role).

---

## 6. Results (number block — speakable without showing a table)

- **Closed-loop (test14-random_reduced):** vanilla GameFormer-Planner **0.8199** (the only
  row with an at-fault collision) → typed **0.8453** → +H **0.8487** (TTC 1.000, best
  causal route progress). Both parent baselines beaten; collisions eliminated.
- **First model in the lineage where CLS and mask quality move in the SAME direction** —
  a one-sentence result against the Finding-5 backdrop.
- **Interventional validation:** RemoveNonCausal PASS; distance-matched (ROAR-style)
  control 6.8–8.4× — the "it's just proximity" objection is dead; dose-response monotone
  (0.004 → 0.94).
- **Decision diagnostic (an intervention we ran on ourselves):** agreement 75.8%/81.7% —
  the announced decision correctly DESCRIBES the plan; but forced decisions are not
  obeyed (compliance 0–33%). The decision is a faithful REPORTER, not yet a LEVER —
  we measured it, we report it, the repair is future work.

---

## 7. Honest limits + future work (say them before being asked)

1. Selection is below the nearest-agent heuristic on the proximity axis — but beats it on
   direction and path-conflict axes, and has what no heuristic can have: calibrated
   intervention response. The faithfulness loss that would supervise allocation is
   identified and deferred → FW.
2. Decision compliance (above) → FW: teacher forcing / consistency loss / CIL-style
   branches / deployment-side enforcement — all listed in the dossier.
3. Counterfactual risk assessment (nuReasoning-style Safe/Unsafe) is gated on that
   compliance fix → FW.
4. Full test14 runs are in progress; the 38-scenario set is the preliminary benchmark.

---

## 8. Likely questions — ready answers

- **"Is the KG yours?"** → No — a colleague's separate contribution; we use and cite its
  definitions and constants. Our contribution is turning predicates into attention
  STRUCTURE.
- **"What if the rules are wrong?"** → The channel layer is auditable and audited:
  GT-vs-predicted activation agreement is reported per channel; weak channels
  (collision/intersect/merges) are flagged, and the gate can be told not to trust them
  (gate_trust).
- **"CLS is already good — why should I care about the mask?"** → Finding 5: CLS is blind
  to mask correctness. If you want trust, you must validate by intervention; that is our
  protocol.
- **"Isn't the decision decorative?"** → We asked ourselves exactly that: agreement is
  high (the description is right), compliance is low (it is not a lever) — measured,
  reported, repair path laid out. A measured limitation, not a hidden one.
- **"Why not just use a VLM?"** → Section 5, point 1-2: their explanation is unverifiable
  text with a measured double-digit annotator hallucination rate; ours is intervenable
  structure. Complementary — but a safety claim needs the verifiable half.
- **"Isn't a VLA the same architecture as yours?"** → Section 5, Q-A (identity vs
  measured coupling; nuVLA's own disabled-at-inference result).
- **"Why inside the network? A parallel predicate output would be the same."** →
  Section 5, Q-B (the measured ladder: parallel → input → structure).

---

## 9. Reference cheat-sheet — the vocabularies we chose (quote from here)

**Agent relation channels (11)** — ego→agent, thresholds from the KG's published constants
(same-flow 0.45 rad; near 5 m; CPA horizon 8 s, critical clearance 1.0 m; VRU radius 12 m;
directional caps ahead 80 m / behind 25 m / adjacent 35 m; lane width 3.5 m):
1. same_lane_ahead   2. same_lane_behind   3. adjacent_left   4. adjacent_right
5. onObservedCollisionCourseWith (predicted futures, CPA)   6. sharesIntersectionWith
(geometric proxy)   7. near (fallback)   8. follows (KG envelope: headway ≤ 5 s, gap ≤ 80 m
/ queue variant)   9. merges (corridor change + target-stream conditions)   10. overtakes
(KG sequence: behind → adjacent pass → ahead)   11. vulnerable_road_user_near_ego_path

**Map relation channels (8)** — ego→map element (lanes / crosswalks / route tokens):
1. inLane (ego's own lane)   2. adjacent_left   3. adjacent_right   4. successor (the road
ego will take)   5. inIntersection (reserved — needs map_api, silent in v1)
6. ego_route_corridor   7. traffic_control (TL-linked, unknown-state excluded)   8. near (20 m)

**Meta-action decision taxonomy (9 lon × 7 lat)** — adapted from nuReasoning; labels
model-free from GT (lon from the first 4 s of the speed profile, lat from the full 8 s,
corridor-relative lane-change detection = KG changesLane semantics):
- **Longitudinal (9):** remain_stopped · stop_quickly · stop_gently · slow_quickly ·
  slow_gently · accel_quickly · accel_gently · maintain · reverse (empty in nuPlan)
- **Lateral (7):** turn_left · turn_right · lane_change_left · lane_change_right ·
  inlane_left · inlane_right · no_lateral
- Deliberate deviation from nuReasoning: their "remain_stopped ⇒ no_lateral" rule is NOT
  applied — our windows are split (4 s / 8 s), so *remain_stopped × turn_left* = "stopped
  now, about to turn" is exactly the unprotected-turn signature we want the head to see.

---

*Figures: `viz_out/paper_strip.png` (main), `viz_out/typed_noresid_bev.png`,
`viz_out/dodmeta_v2_bev.png`, `viz_out/decision_labels.png`, `channels_bev.png`.
Numbers source: `MASK_LOSS_INVESTIGATION.md` (tables + the Paper positioning section).*
