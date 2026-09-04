# Channel Audit — predicate-based relation channels (KG counterparts, deviations, validation)

Date: 2026-08-15 · Code: `GameFormer/channels.py` · Test: `eval_channels.py --selftest` (15/15 PASS)
Source KG: `~/nuPlan_Predicates_KG` (the predicate extractor — cited as its author's own paper)

Two directed relation families, matching the two attention relations inside the causal module
(`causal_graph.py`: the `other` relation = **ego ← agent**, the `g2a` relation = **ego ← map element**):

- **ego → agent channels** (`compute_channels`, R = 11) — CLI/stat name: `--mode r1`
- **ego → map channels** (`compute_map_channels`, R = 8) — CLI/stat name: `--mode r2`

Rule: **a channel's identity = the name of the KG predicate/reason it instantiates**; deliberate
constructions with no KG counterpart are flagged. This file is both the audit record of v1 (the
npz-only geometric port) and the specification of v2 (data_process + map_api, calling the KG
functions unchanged).

Direction thresholds (both from the KG): **agents** use `SAME_FLOW_RAD = 0.45` (same-flow,
`hasTravelDirectionDifferenceTo`); **map elements** use `MAP_DIR_RAD = 0.60` (the KG's own
map/motion-agreement constant for `hasEffectiveTravelHeading`) — adopted after a measured
borderline failure (an adjacent lane at dtheta 0.55 against a curving corridor left a visible gap
in the band; 0.45 was too tight for map geometry, π-opposite lanes remain safely excluded).

Distance caps (ego→agent) come from the KG's **directional** threshold family
(relevance_logic.py:46-49, "Directional semantics should be stricter than pair relevance"):
ahead 80 m / behind 25 m / adjacent 35 m. The looser 100/35/45 values are the KG's
relevance-*screening* layer and were measured to admit nonsense (an adjacent fire at −38 m).

---

## 1. Ego → agent channels (R = 11)

| # | Channel | KG predicate + **function** | v1 computation | Deviations / notes |
|---|---|---|---|---|
| 0 | `same_lane_ahead` | relevance reason — **`evaluate_pair_relevance` (relevance_logic.py:173, reason :217-220)** = `same_map` (+) via `_pair_map_relation` + `_signed_path_distance` (map.py:983) | corridor-band projection (\|d_lat\| ≤ w/2) ∧ same-flow 0.45 ∧ 0 < ds ≤ 80 m ∧ VehicleLike | lane-ID equality replaced by **ref-path corridor projection** (npz has no lane IDs); behind-agents fall back to the ego-frame axis |
| 1 | `same_lane_behind` | reason at relevance_logic.py:221-223 | same, −25 m ≤ ds < 0 | as above |
| 2 | `adjacent_left` | `_pair_map_relation` → `adjacent_left` **(map.py:952-959)**, reason :225-227 | lateral band (w/2, 1.5w] ∧ same-flow ∧ center ≤ 35 m ∧ **ds ≥ −25 m** ∧ VehicleLike | **corridor-band semantics ≠ lane-ID adjacency**: "beside my intended PATH", so an agent beside a successor lane also counts (strict KG topology would call it unrelated). Deliberate, planning-motivated, bounded by the 35 m cap; v2 (map_api) uses true primary-lane adjacency. Longitudinal behind-cap −25 m (KG `directional_same_lane_behind_m`) added by USER REQUEST: the straight-line virtual backward projection admitted agents at −34 m under the center cap alone |
| 3 | `adjacent_right` | symmetric | symmetric | as above |
| 4 | `onObservedCollisionCourseWith` | risk family **(base.py:1316-1317)**; CPA machinery **`constant_velocity_cpa` (relevance_logic.py:119)** | time-aligned min clearance between the agent's future and the ego corridor sweep (arc-walk at current speed), 8 s horizon, clearance 1.0 m, width-based radii; ∧ closing > 0 | KG evaluates **observed** futures + footprint polygons; we evaluate **GF-predicted** futures + circles. Clearance = `critical_cpa_clearance_m` **1.0** (the 2.0 relevance-screening margin made normal side passes fire — measured: a rear adjacent-lane car in f4d2). **Weak-evidence channel: GT-vs-GF IoU 37%** — deployment decisions inherit GF prediction errors (measured example: GF predicted a rear car sweeping through the corridor; GT showed it slowing in its own lane) |
| 5 | `sharesIntersectionWith` | agent-pair predicate — **`_shared_intersection` (map.py:926-940)**, reason `shared_intersection_conflict` (:248-250) | **v1 geometric proxy:** predicted path crosses the corridor at > 30° within 60 m (30°/60 m thresholds are OURS, `*`) | v2 = true intersection-ID association via map_api. **Weak-evidence: IoU 50%** |
| 6 | `near` | `np:near` free-space ∈ (2, 5] m (PREDICATES.md; `categories/spatial.py`) | d_fs ≤ 5 m ∧ no other channel (fallback) | `veryNear` (≤ 2 m) folded in — the graded value lives in the d_fs evidence; footprint distance approximated by circles |
| 7 | `follows` | **the KG follows envelope, frame-level** (PREDICATES.md follows): moving (g/v ≤ 5 s, g ≤ 80 m) ∨ queue (v_ego ≤ 2, v_O ≤ 4, g ≤ 12 m); nearest-leader uniqueness margin 0.50 m; MotorVehicleLike | corridor-ahead ∧ envelope ∧ unique nearest leader ∧ 1 s persistence from the 2 s past window; bumper gap g = ds − L/2 − 2.31 | deviation: KG requires 1 s **condition** persistence; v1 uses 1 s **"was ahead"** persistence as proxy. IoU 100% (no future used) |
| 8 | `merges` | `np:merges` — the KG **catalog parent concept** of mergesInFrontOf/Behind (catalog-only entry) | anticipated: adjacent band ∧ lateral velocity toward corridor ≥ 0.2 (`*`) ∧ predicted corridor entry within 8 s | single channel by USER DECISION; the front/behind direction is `EV_DS_ENTRY`'s sign (+ = mergesInFrontOf-like, − = mergesBehind-like) — splitting later is a one-liner. Rare (60 fires) → weight-sharing candidate. IoU 67% |
| 9 | `overtakes` | `np:overtakes` constants (PREDICATES.md): side-by-side ≥ 1.25 m, rel-speed ≥ 0.3, clearance ≥ 1.0 | anticipated prefix of the sequence: was not ahead 2 s ago ∧ relative advance ≥ 1 m ∧ alongside (\|ds\| ≤ 10) ∧ closing longitudinally ∧ predicted ≥ 1 m ahead; MotorVehicle only | the KG certifies the **completed** maneuver (up to 30 s, stable return); ours is the in-progress form. Order-transition condition added after a measured over-fire (adjacent faster traffic counted as overtaking, 4.1% → 2.9%). IoU 92% |
| 10 | `vulnerable_road_user_near_ego_path` | reason — relevance_logic.py:256-267; **narrow bound = the KG VRU-critical distance 12 m (relevance_logic.py:267)**, not the 35 m screening radius | pedestrian/bicycle ∧ center ≤ 12 m ∧ closing > 0; needs no corridor (also active in corridor-less scenes) | added by USER DECISION after the coverage breakdown measured **42% of misses were pedestrians**: peds have no location-based channel (lane channels are VehicleLike-gated), leaving a 5–12 m dead zone between the `near` bubble and the future-dependent channels. This channel is the pedestrian's lane-channel equivalent: static, prediction-noise-free |

**Type guards (transcribed from KG definitions):** lane-topology channels (0-3) + merges: vehicle + bicycle (`VehicleLike`); follows + overtakes: motor vehicles only; pedestrians enter via 4/5/6/10. Caught originally as a missed transcription (a pedestrian fired `overtakes`) by the user.

**Evidence vector (E = 9, attached to every fired channel of the pair):** ds, d_lat, d_fs, closing speed (`hasClosingSpeedTo` sign convention), time-aligned TTC, corridor-entry time, dtheta vs corridor, lateral velocity toward corridor, entry-ds (merge direction).

### Validation (1,118 scenes / 10,659 pairs, GF-based reporting, FINAL run 2026-08-15)

| Channel | Fires | % | GT-vs-GF positive IoU |
|---|---|---|---|
| same_lane_ahead | 486 | 4.6% | 100% |
| same_lane_behind | 496 | 4.7% | 100% |
| adjacent_left / right | 1126 / 931 | 10.6 / 8.7% | 100% |
| onObservedCollisionCourseWith | 245 | 2.3% | **37%** |
| sharesIntersectionWith | 538 | 5.0% | **50%** |
| near | 961 | 9.0% | 98% |
| follows | 359 | 3.4% | 100% |
| merges | 60 | 0.6% | 67% |
| overtakes | 306 | 2.9% | 92% |
| vulnerable_road_user_near_ego_path | 753 | 7.1% | 100% (no future) |

Consistency exact: 6,261 total fires = 1×4,297 + 2×474 + 3×208 + 4×98. Multi-fire: 0 → 52.4%, 1 → 40.3%, 2 → 4.4%, 3 → 2.0%, 4 → 0.9% (among fired: 84.6% single).

**VRU channel effect (measured):** coverage 30.5% → **25.8%** (−233 misses); `near` 1,233 → 961 (−272): close pedestrians/bicycles now enter the mask under their own name instead of the generic fallback (interpretability gain), and 431 previously zero-channel pairs became visible. The residual 25.8% is dominated by intended exclusions: oncoming traffic (the heuristic counts them because closing = sum of speeds), out-of-band same-direction lanes, and pedestrians beyond the deliberately narrow 12 m bound.

**Weak-evidence channels (37/50/67% IoU):** their deployment decisions inherit GF prediction noise. Consequences: (i) training reads the GF-based channels (train/deploy consistency — the model trains on the noise it will face); (ii) the IoU table is the paper's "which symbolic relations survive prediction noise" figure; (iii) their role in hard gating must be revisited at `_attend` integration time.

---

## 2. Ego → map channels (R = 8)

| Channel | KG predicate + **function** | v1 computation | Deviations |
|---|---|---|---|
| `inLane` | `np:inLane(E,L)` + `np:hasPrimaryLane` selection — **`append_map_memberships` (map.py:571)** + **`_select_primary_candidate` (map.py:392)** via `map_api.get_proximal_map_objects` | nearest centerline point to origin ≤ w/2 ∧ heading same-direction (0.60) ∧ not a crosswalk token | centerline band instead of polygon (w = 3.5 assumed); **no primary-lane disambiguation** — overlapping lanes can all fire (measured 2.38/scene vs. expected 1–2); v2's `_select_primary_candidate` closes it |
| `adjacent_left` / `adjacent_right` | `_pair_map_relation` **(map.py:942-968)**, ID collection map.py:591-694 | median lateral of element points ∈ (w/2, 1.5w] ∧ **same-direction** ∧ **reaches ego** (furthest point x > 0) | geometric band instead of ID topology. FIXED after user viz findings: (a) `abs(cosΔθ)` let opposite-direction lanes fire — abs removed; (b) no longitudinal restriction let fully-behind lanes fire — reaches-ego added |
| `successor` | `_pair_map_relation` (map.py:960-963), connector IDs :139-145 | > 30% of points inside the corridor band, ahead ∧ not inLane | successor CHAIN ("on the path I will traverse"), not the direct-successor ID relation — deliberate, flagged |
| `inIntersection` | `np:inIntersection` / `np:hasPrimaryMapIntersection` (intersection IDs map.py:188, 522, 796). NOTE: earlier name `sharesIntersectionWith` was a domain error — that is an *agent-pair* predicate (`_shared_intersection`, map.py:926-940); for a map element the correct concept is the inIntersection composition (element ∈ ego's primary intersection) | **RESERVED — never fires in v1** | intersection polygons absent from the npz; fills in v2. Measured motivation: the crossing lane over ego's path at a junction is topologically `unrelated` yet decision-critical (f4d2 inspect, element idx 22) |
| `ego_route_corridor` | reason — relevance_logic.py:233-246 + `route_relation` (:158); route family `categories/route.py` | route token class ∧ **reaches ego** (x > 0) | **lookbehind = 0 by USER DECISION** ("start from the route we are inside") — the KG's rank-lookbehind = 1 is an agent rule. Note: nuPlan's `route_lanes` = ALL lanes of each route roadblock (dataset semantics) |
| `traffic_control` | reason `shared_traffic_control` (relevance_logic.py:252-254) + **`append_native_traffic_light_status` (traffic_control.py:187)** | lane carries a real signal (one-hot dims green/yellow/red; **unknown dim EXCLUDED** — nuPlan `LaneSegmentTrafficLightData`) ∧ path-related | counting the unknown dim was the first version's bug (every on-path lane fired) — fixed; no 45 m cap; lane-based not connector-based |
| `near` (fallback) | `np:near` (5 m free-space) as name only | min center-to-polyline distance ≤ **20 m** ∧ nothing else fired | threshold is OURS (no KG entity-to-map-element distance predicate exists). **USER DECISION: name and 20 m kept.** Fire share 25.5% — the with/without-`near` gating ablation stays planned |

### Validation (1,118 scenes / 61,490 elements, final run)

inLane 2,664 (4.3%, 2.38/scene) · adjacent_left 3,558 / right 2,793 (L:R ≈ 1.27:1, plausible under right-hand traffic) · successor 4,807 (4.3/scene) · ego_route_corridor 7,709 (6.9/scene; −20% after lookbehind-0) · traffic_control 303 (0.27/scene) · near 15,670 (25.5%) · **zero-channel 49.2%** — the structural-gating payoff: half the map tokens leave the softmax competition; without `near`, exclusion ≈ 75% (the two ends of the fallback ablation). Among fired elements 80.5% single-channel; most doubles are structural route-token twins (`inLane+route`, `successor+route`). Consistency check exact: 37,504 fires = 1×25,122 + 2×5,918 + 3×182.

---

## 3. Metrics guide (what each number is for)

- **Fire rates** — broken-rule detector (caught: TL unknown-dim, opposite-direction adjacents, −38 m adjacent, overtakes over-fire), calibration evidence (every threshold change shows up as a delta), learnability forecast (rare channels → weight-sharing decisions: merges 60, traffic_control 303).
- **Multi-fire histogram** — the zero row = how much structural gating excludes (payoff + risk in one number); the among-fired distribution = whether channel definitions carve the world cleanly.
- **Coverage + miss breakdown** — the guard against the design's worst failure mode (structurally blinding the mask to a relevant agent). "Important-looking" is a *heuristic* (d_fs ≤ 5 ∨ closing > 0.5 ∧ CV-TTC ≤ 8 s), deliberately broad — hence the type/direction breakdown that separates intended exclusions (oncoming) from real gaps (pedestrians → channel 10).
- **GT-vs-GF raw agreement + positive IoU** — every channel computed twice (ground-truth futures vs. GF-predicted futures). Raw agreement is inflated by both-off pairs; positive IoU is the honest reliability of future-dependent channels at deployment. It tests the GF=GT assumption per channel, justified the train-on-GF-channels decision, and marks the weak-evidence channels.

## 4. Visualization decisions (`eval_channels.py::_draw`, KG notebook cell 9/10 palette)

Agent colors mapped onto the KG relation palette (follows dark blue, merges orange, overtakes red, collision stop-line red, near gray, VRU teal); shapes by agent type (rect/circle/triangle); legend shows `short = full-KG-name` (tag/legend consistency fix). Map: neutral gray lanes; per-channel colors inLane green / successor blue / adjacent purple / route pink dashed / traffic_control red dashed; crosswalks drawn only in map modes. `--mode {all,r1,r2}` gates stats + JSON + viz + legend together. **The drawn future line = the future that MADE the decision** (GF when a checkpoint is given) — previously GT lines were drawn under GF decisions, which made correct labels look absurd (measured: f4d2's rear car, GT ending at (2.8, 3.8) vs GF predicting (31, −4.3) through the corridor).

## 5. Open items

1. ~~VRU channel fire rate + coverage delta~~ — DONE (753 fires / 7.1%, coverage −4.7 pts; see validation table).
2. Weak-evidence trio's role in hard gating: decide at `_attend` integration.
3. v2 (data_process + map_api): compute channels with the KG functions unchanged, cache into npz (training), per-cycle calls at inference; closes the deviations marked above (lane-ID adjacency, primary-lane disambiguation, true `inIntersection`, connector-based traffic control).
4. Cross-check run: extractor with `spatial,map,pairwise` on 10–20 scenarios, frame×pair/element agreement — the numeric fidelity report of the v1 geometry.
5. Ablations queued: map `near` on/off; merges/traffic_control weight-sharing.

---

# v2 — KG predicate inventory and the L0/L1/L2 layering (2026-08-30)

The v1 audit above maps our 11 agent + 8 map channels onto the KG. This section does the
reverse: it surveys what the KG *actually emits* and sorts it into the three layers the
relational decision trace needs.

Source of truth is **not** `PREDICATES.md` — that file documents 5 sections (spatial, motion,
temporal, map, interaction) and omits several categories that the code emits. The full
inventory is `nuplan_predicates_modular/categories/*.py`, 18 modules. Notably `PREDICATES.md`
contains no yield, no traffic-light and no collision/TTC predicates, while the code emits all
three families.

## The layering

**L0 — structure / gating.** Current-frame, static relations. Their job is to decide which
agents and map elements enter the causal graph. Measured (redundancy probe, 10 659 pairs):
8 of our 11 agent channels are recoverable from the raw 7-d edge at >= 0.90 balanced accuracy,
so *structure* is the honest job for them; asking them to carry semantics is asking for what
they provably do not have.

**L1 — interaction decision.** Requires future or temporal evidence; expresses what the ego is
*doing about* another agent or map element. This is the concept-bottleneck layer.

**L2 — maneuver.** The ego's own action. Unchanged from v2/`lat_moe`.

## L0 (structure)

| KG category | predicates | our channel |
|---|---|---|
| `spatial` | `near`, `veryNear` | `CH_NEAR` |
| position (§1) | `inFrontOf`, `behind`, `leftOf`, `rightOf`, `front/rearLeft/RightOf`, `overlapping`, `touching` | `CH_SAME_LANE_AHEAD/BEHIND`, `CH_ADJACENT_LEFT/RIGHT` |
| `map` | `inLane`, `inLaneConnector`, `inIntersection`, `inCrosswalk`, `hasPrimaryLane`, `hasSpatialMapRelation`, `inSameLaneAs`, `sharesIntersectionWith`, `hasParentRoadblock` | the 8 map channels |
| `route` | `onExpertRoute` | `MCH_ROUTE` |
| `visibility` | `geometricallyVisibleToEgo`, `withinEgoFieldOfView`, `occludedByAgent` | **absent — strong gating candidate** |
| `traffic_control` | `hasRelevantTrafficLight`, `hasTrafficLightStatus`, `controlsLaneConnector` | `MCH_TRAFFIC` (coarse) |

## L1 (interaction decision)

| KG category | predicates | status here |
|---|---|---|
| **`intent`** | **`yieldingTo`**, **`waitingFor`**, `creatingGapFor`, `competingForGapWith` | **absent — the core of the layer** |
| **`traffic_control`** | **`waitsAtRedLight`**, `runsRedLight`, `crossesStopLine` | **absent — the map-side twin of `yieldingTo`** |
| `future_observation` | `onObservedCollisionCourseWith`, `cautionCollisionRisk`, `criticalCollisionRisk` | `CH_COLLISION_COURSE` — currently an *input*; belongs here as an *output* |
| `interaction` | `mergesInFrontOf`, `mergesBehind`, `overtakes` | `CH_MERGES`, `CH_OVERTAKES` — same move |
| `interaction` | `follows` | named as interaction but geometric in content (0.96 recoverable); **stays in L0** |

## L2 (maneuver) — unchanged

| KG category | predicates | ours |
|---|---|---|
| `maneuver` | `turnsLeft/Right`, `goesStraight`, `keepsLane`, `changesLaneLeft/Right`, `makesUTurn`, `enters/exitsLaneConnector` | LAT5V / LAT5L (`lat_moe`) — near one-to-one |
| — | (no longitudinal counterpart in the KG) | LON4 `stop/slow/accel/maintain` (ours) |

## Evidence (quantities, not predicates)

`geometry` (centre/free-space/lateral/longitudinal distances, footprint gaps) ·
`motion` (`hasClosingSpeedTo`, relative speeds, travel-direction difference) ·
`risk` (`hasPredictedTimeToClosestApproach`, `hasPredictedClosestApproachClearance`,
`hasTimeHeadwayCandidate`) · `temporal` (durations, per-frame deltas).

## `np:yieldingTo` — definition and measured frequency

`categories/intent.py:66-117`, thresholds from `predicate_logic.py`:

```
1) buffered future paths of subject and object intersect -> common conflict region
2) subject decelerating (<= -0.30 m/s^2) OR stopped (<= 0.30 m/s)
3) object reaches the conflict point first (object_arrival + 1.5 s < subject_arrival)
4) sustained >= 1.0 s
```

Measured on our cache (1118 validation scenes, ego as subject, GT futures, conflict region as
the closest-approach point of the two paths instead of the shapely buffer intersection —
same test, vectorised): **27 scenes, 2.4%, always exactly one target.**

Too sparse to supervise a head on its own (0.27% of agent-pairs). Consequence for the design:
L1 must be **multi-class over the whole interaction family** (none / follows / yielding /
merging / overtaking / collision-risk), not a yield-only binary.

## Two constraints this survey imposes

1. **Whatever is promoted to L1 must leave the input channels.** If `collision_course` is both
   an input channel and the L1 target, the head copies the input bit, `L_attr` goes to zero and
   `M_cas` gets no useful gradient. `yieldingTo` is safe (never was an input); `collision_course`,
   `merges`, `overtakes` are not.
2. **The map side has its own L1.** `waitsAtRedLight` / `crossesStopLine` are the map twin of
   `yieldingTo`. This is the first construct that would join the agent and map branches, which
   today live in separate softmaxes.

---

# v3 — The final L0/L1 vocabulary and its KG provenance

Written 2026-09-02. Supersedes the channel lists in v1 and v2. L2 is out of scope here: it has
no KG counterpart at all (see the last section).

## Why a layering at all

The KG is a general-purpose scene description — ~140 predicates over 18 categories, bound to no
downstream task. A planner decides. The passage between them is a **task-specific projection**,
and three constraints determine it.

**Constraint 1 — computability at inference. Decides the LAYER.**
KG predicates split by whether their truth condition references `t > now`. Those that do not are
*state*; those that do are *anticipation*. At deployment there is no future. So state predicates
can be **evaluated** and become inputs (L0); anticipation predicates must be **estimated** and
become learned outputs (L1). This is not a design preference — the deployment setting forces it,
which is what makes the layer boundary principled rather than arbitrary.

**Constraint 2 — decision-relevance. Decides MEMBERSHIP.**
A predicate earns a slot only if it changes what ego should do. This is the answer to "why not
`frontRightOf`": the KG's spatial partition is **body-frame and geometry-complete** — it must
distinguish every relative placement. Ours is **corridor-frame and action-complete** — it need
only distinguish placements that imply different maneuvers. `rightOf` versus `frontRightOf` does
not change whether ego brakes; lane membership and longitudinal order do. The same constraint
sends measurements (`hasFreeSpaceDistanceTo`) to the evidence vector rather than the vocabulary:
they are continuous inputs to a decision, not decisions.

**Constraint 3 — learnability. Decides SURVIVAL.**
A class firing in under 0.1% of instances can be neither learned nor evaluated. `near` was
dropped for the opposite reason — frequent but uninformative (measured: of 1061 agents entering
the graph on `near` alone, 1058 carry GT label `none`).

## What the layering buys

1. **A typed trace.** The decision is expressible as a sentence over named relations rather than
   attention weights.
2. **Falsifiability.** L1 has ground truth from the KG rules, so concept accuracy is measurable.
   An attention weight has no ground truth — this was the gap noted in v2.
3. **Intervenability.** A concept can be set by hand and the decision change observed. "Setting"
   an attention weight is not a meaningful operation.
4. **Separation of observed from believed.** L0 is evidence, L1 is inference. Without that split
   one cannot distinguish "the model believed wrongly" from "the evidence was wrong".

## Reading the tables

**Correspondence:** `DIRECT` one KG predicate or reason value · `NARROWING` several KG predicates
expressed as one of ours · `COMBINATION` a composition of several KG predicates · `NONE` no KG
counterpart.

**Need:** `extraction` new data must be pulled from nuPlan — `data_process.py` rerun, 177k scenes
· `processing` data is present, the rule is rewritten — `extract_channels.py` rerun, ~1 h ·
`ready` currently computed.

Note on naming: L0 agent channel names come from the **value vocabulary** of the single predicate
`np:hasRelevanceReason` (relevance_logic.py:213-282), not from separate `np:` predicates. There is
no `np:sameLaneAhead`.

## L0 — agent (10 channels, input)

| ours | KG | correspondence | need |
|---|---|---|---|
| `same_lane_ahead` | `hasRelevanceReason="same_lane_ahead"` + `np:inSameLaneAs` + `np:hasSignedPathDistanceTo>0` | COMBINATION | ready |
| `same_lane_behind` | `hasRelevanceReason="same_lane_behind"` + `np:inSameLaneAs` + `np:hasSignedPathDistanceTo<0` | COMBINATION | ready |
| `leftAdjacent` | `np:hasSpatialMapRelation = leftAdjacent` | DIRECT | **extraction** (real topology; today a geometric proxy) |
| `rightAdjacent` | `np:hasSpatialMapRelation = rightAdjacent` | DIRECT | **extraction** |
| `sharesIntersectionWith` | `np:sharesIntersectionWith` | DIRECT | **extraction** (intersection polygons) |
| `VRU_near_ego_path` | `hasRelevanceReason="vulnerable_road_user_near_ego_path"` + `np:hasAgentType` | COMBINATION | ready |
| `inCrosswalk` | `np:inCrosswalk` (+ `np:intersectsCrosswalk`) | DIRECT | processing |
| `staticObstacleOnPath` | `hasRelevanceReason="static_obstacle_on_ego_path"` | DIRECT | processing |
| `onRouteCorridor` | `hasRelevanceReason="ego_route_corridor"` / `"object_on_route"` | DIRECT | processing |
| `sharedTrafficControl` | `hasRelevanceReason="shared_traffic_control"` | DIRECT | **extraction** (traffic-light state) |

Set-level deviations: the KG's eight-way footprint family (`inFrontOf`, `behind`, `leftOf`,
`rightOf`, `frontLeftOf`, `frontRightOf`, `rearLeftOf`, `rearRightOf`) is **narrowed** to four
lane-relative channels. The KG's contact ladder (`overlapping` / `touching` / `veryNear` / `near`,
mutually exclusive by free-space distance) is not represented as channels at all — the degree
lives in the evidence vector as a continuous `d_fs`.

Dropped from v2: `near` (see Constraint 3), `follows` / `merges` / `overtakes` (promoted to L1),
`onObservedCollisionCourseWith` (promoted to L1 — it reads predicted futures, so by Constraint 1
it is anticipation, not state).

## L0 — map (8 channels, input)

| ours | KG | correspondence | need |
|---|---|---|---|
| `inLane` | `np:inLane` ∨ `np:inLaneConnector` (+ `np:hasPrimaryLane`, `np:hasPrimaryLaneConnector`) | NARROWING | ready (type split needs **extraction**) |
| `leftAdjacent` | `np:hasSpatialMapRelation = leftAdjacent` | DIRECT | **extraction** |
| `rightAdjacent` | `np:hasSpatialMapRelation = rightAdjacent` | DIRECT | **extraction** |
| `successor` | `np:hasSpatialMapRelation = successor` | DIRECT | **extraction** |
| `inIntersection` | `np:inIntersection` (+ `np:intersectsIntersection`, `np:hasPrimaryMapIntersection`) | NARROWING | **extraction** (channel is dead today: 0 firings in 1118 scenes) |
| `inCrosswalk` | `np:inCrosswalk` (+ `np:intersectsCrosswalk`) | NARROWING | processing |
| `onExpertRoute` | `np:onExpertRoute` ∘ `np:hasParentRoadblock` | COMBINATION | processing |
| `trafficControl` | `np:hasTrafficLightStatus` + `np:hasRelevantTrafficLight` + `np:controlsLaneConnector` + `np:hasDistanceToStopLine` | COMBINATION | **extraction** |

Dropped: map-side `near` — the KG's `np:near` is an agent-agent predicate (`2 < d_fs <= 5 m`) with
no map-element meaning; ours was a 20 m catch-all carrying 28.3% of valid elements on its own.
`predecessor` was not taken: elements entirely behind ego are already excluded by `reaches_ego`.

## L1 — agent (8 classes, learned) · subject = agent, object = ego (implicit)

| ours | KG | correspondence | need |
|---|---|---|---|
| `none` | — | NONE (our default class) | ready |
| `leads` | `np:follows(ego, agent)`, direction reversed | DIRECT | ready |
| `hasPriority` | `np:yieldingTo` **without** the ego-deceleration condition; uses `intent.py` internal `object_clears_first` | variant of DIRECT | processing |
| `blocks` | `np:waitingFor` **without** the ego-stopped condition; uses `intent.py` internal `object_blocks_soon` | variant of DIRECT | processing |
| `cutsInAhead` | `np:mergesInFrontOf(agent, ego)` | DIRECT | ready |
| `cutsInBehind` | `np:mergesBehind(agent, ego)` | DIRECT | ready |
| `overtakes` | `np:overtakes(agent, ego)` | DIRECT | ready |
| `collisionRisk` | `np:onObservedCollisionCourseWith` + `np:criticalCollisionRisk` + `np:cautionCollisionRisk` | NARROWING | processing |

On `hasPriority` / `blocks`: the KG predicates `np:yieldingTo` and `np:waitingFor` **do exist**
(intent.py:113, :131). What has no KG counterpart is our *variant*, which drops the ego-side
condition. We drop it because `waitingFor` requires ego to be stopped, which is `lon=stop` — the
L2 label restated. Keeping the KG definition preserves provenance but makes L1 partly circular
with L2. **This trade-off is not yet settled by measurement**: the overlap between `waitingFor`
and `lon=stop` has not been quantified.

On `collisionRisk`: the three-level split was measured and rejected. Predicting the maneuver from
ego kinematics alone gives balanced accuracy 0.5466; adding a single collision flag gives 0.5640
(+0.0175); adding the caution/critical split gives 0.5618 (+0.0153) — worse. Using the levels
alone, both encodings score identically (0.3307).

## L1 — map (3 classes, learned)

| ours | KG | correspondence | need |
|---|---|---|---|
| `none` | — | NONE | ready |
| `keepsLane` | `np:keepsLane` | DIRECT | ready |
| `stopsAtTrafficControl` | `np:waitsAtRedLight` over the control element (light or stop line) | DIRECT | **extraction** |

`crossesStopLine` and `runsRedLight` were not taken: crossing is an event, stopping is a
constraint, and only the constraint shapes the plan. `entersLaneConnector` / `exitsLaneConnector`
were not taken.

**On stop signs.** An earlier draft treated the stop-sign case as an extension we would have to
invent. Settled against the data instead, the answer is simpler. `SemanticMapLayer.STOP_SIGN`
exists in the devkit enum but the nuPlan map API does not serve it — `get_available_map_objects()`
returns `LANE, LANE_CONNECTOR, ROADBLOCK, ROADBLOCK_CONNECTOR, STOP_LINE, CROSSWALK, INTERSECTION,
WALKWAYS, CARPARK_AREA`, and querying STOP_SIGN raises. Stop signs are therefore missing from the
**data**, not from the KG. `STOP_LINE` is served and well populated (measured: 6.0 polygons per
scene, no empty scene), so we treat a stop line as the control element regardless of what governs
it. The class is uniform and fully KG-derived: the element is identified by
`np:hasDistanceToStopLine` and `np:hasTrafficLightStatus`, stopping at it by `np:waitsAtRedLight`.

## Three cross-cutting deviations

1. **Single-label collapse.** The KG may assert several predicates for one pair simultaneously.
   We reduce to one label by priority. Measured: 7 multi-label cases in validation.
2. **Single-frame evaluation.** The KG re-evaluates every frame and tracks continuity across
   frames with `update_condition_streak`. We evaluate one window from `t=0`.
3. **Predicted rather than observed futures.** The KG's semantic predicates use observed (GT)
   futures. Our runtime channels use GameFormer top-1 predictions. Training labels come from GT,
   inputs from predictions, and at inference GT is absent entirely.

## L2 has no KG provenance

`lon` (`stop` / `slow` / `accel` / `maintain`) and `lat` (`turn_left` / `turn_right` / `to_left` /
`to_right` / `none`) have **no KG counterpart**. `maneuver.py` defines `np:turnsLeft`,
`np:turnsRight`, `np:goesStraight`, `np:makesUTurn` but does not emit them — the only maneuver
predicates actually asserted are `np:keepsLane`, `np:entersLaneConnector`, `np:exitsLaneConnector`.
Our L2 comes from `decision_labels.py`, derived from the geometry of the expert trajectory. Any
claim that the decision vocabulary is KG-grounded holds for L0 and L1 and **must not be extended
to L2**.

## Status of this table

Of 29 entries, 19 are DIRECT, 7 are NARROWING or COMBINATION, and 3 have no KG counterpart
(`none` twice, plus the `hasPriority` / `blocks` variants). `stopsAtTrafficControl` moved to DIRECT
once the stop-sign question was settled against the data rather than against the KG.

**None of it has been checked against the KG's own output.** Seven attempts to run the KG on the
same scenarios produced no `interaction` or `intent` assertions; the causes found so far were a
missing `--derive-semantic-predicates` flag, `agent-agent` pair directions being required for the
semantic layer, and memory exhaustion when both are enabled. Until a KG run succeeds and a
per-predicate agreement rate is measured, every correspondence above is **asserted, not verified**.

---

## v3 — geçici olarak kapatılan L0 ajan kanalları (2026-09-03)

Aşağıdaki iki kanal `GameFormer/channels_v3.py` içinde **kapatıldı**. Kod silinmedi,
yorumda duruyor; her birinin gerekçesi ve geri getirme koşulu dosyanın başındaki
bloklarda yazılı. L0 ajan sözlüğü böylece **10 → 8** kanala indi.

Ölçümlerin hepsi yeniden çıkarılmış validation split'i üzerinde (1118 sahne, 10659
ego–ajan çifti); ölçüm aracı `evals/eval_channels.py --v3 --only agent`.

### 1. `onRouteCorridor` — KG reason `ego_route_corridor` (ağırlık 40)

**Kapatma nedeni: subsumption.** Dört şerit kanalını birden kapsıyordu:

| koşul | P(onRouteCorridor \| koşul) |
|---|---|
| `same_lane_ahead` | 93.6% |
| `staticObstacleOnPath` | 94.1% |
| `leftAdjacent` | 91.3% |
| `same_lane_behind` | 90.0% |

%39.6 yanma oranıyla, bir CBM'de atfı belirsizleştiriyordu: kararın hangi kavramdan
geldiği ayırt edilemiyor. **Koridoru daraltmak bunu çözmüyor** — hangi koridor tanımı
alınırsa alınsın ego şeridini içerir, dolayısıyla `same_lane_ahead` her zaman altkümesi
olur; KG'nin dar varyantı `generator_forward_corridor` (`|lat| ≤ 7.0 m`, ağırlık 105)
ise ±2 şerit genişliğinde olduğu için `leftAdjacent`/`rightAdjacent`'ı da kapsar.

**Kaybedilen:** bu kanalın tek başına getirdiği ajanlar çöp değil. 500 sahnede, ajanın
8 s GT ufkunda ego'ya en yakın merkez–merkez mesafesi:

| grup | n | medyan | <20 m |
|---|---|---|---|
| TEK `onRouteCorridor` | 536 | 10.3 m | 95.1% |
| başka kanal yanan | 2236 | 9.2 m | 88.5% |
| hiçbir kanal yanmayan | 1898 | 15.2 m | 63.6% |

Yani hiçbir yapısal ilişkiyle açıklanamayan ama gerçekten yaklaşan ajanlar. Kapalı
olduğu sürece bu ~%11 ajan grafiğe hiç girmiyor.

**Geri getirme seçenekleri:** (a) *residual* kural — ajan için başka hiçbir kanal
yanmıyorsa yansın; overlap inşaat gereği 0 olur, kapsama aynen korunur, yanma
%39.6 → ~%11. (b) L1'e taşı: "rotada ama yapısal ilişkiyle açıklanamayan" bir
anticipatory kavram olarak, L0'ın yapısal seçiciliğini bozmadan.

### 2. `same_lane_behind` — KG `directional_same_lane_behind_m` = 25 m

**Kapatma nedeni: nedensellik yönü ters.** Ego'nun planı takipçisine bağlı değil;
nuPlan CLS at-fault çarpışmayı cezalandırıyor, arkadan çarpılmayı değil. 500 sahnede,
ajanın ego'ya göre koridor yayı üzerindeki boylamsal konumu:

| grup | n | 8 s sonunda ego'nun önünde | ufukta bir an önüne geçen | medyan en yakın |
|---|---|---|---|---|
| `same_lane_behind` | 228 | 2.6% | 2.6% | 7.6 m |
| `same_lane_ahead` | 219 | 98.6% | 100.0% | 11.3 m |

Takipçilerin %97.4'ü ufuk boyunca arkada kalıyor → öngörülecek etkileşim doğmuyor, ve
L1'in hiçbir sınıfına (yield / wait / merge / overtake) dönüşemiyor; hepsi diğer ajanın
önde veya kesişen olmasını gerektiriyor. Şerit değiştirirken hedef şeritteki takipçi
önemlidir, ama o `leftAdjacent`/`rightAdjacent`'ın işi — bizde `lon_ok = ds ≥ −25 m`
ile arkadakiler zaten dahil; `same_lane_behind` spesifik olarak *kendi şeridimizdeki*
takipçi.

**Geri getirme seçeneği:** negatif kontrol olarak. Müdahale edildiğinde planı
**değiştirmemesi gereken** bir kavram, intervention-correctness denetimi için çapa
olurdu (o denetimi şu an −9.0 puanla kaybediyoruz). Bir slota mal olur.

### 3. `staticObstacleOnPath` — KG reason `static_obstacle_on_ego_path`

**Kapatma nedeni: bağımsız katkısı yok.** `P(same_lane_ahead | staticObstacleOnPath) = 95.4%`
— tanım gereği zaten `same_lane_ahead`'in "duran" alt kümesi
(`inlane & ds>0 & ds≤60 & speed≤0.5`).

Yakan 174 ajanın tipi: **araç 165 (%94.8)**, yaya 8 (%4.6), bisiklet 1 (%0.6). `same_lane_ahead`'in
yakalamadığı 8 ajanın **hepsi yaya** ve hepsi 12 m'den uzak. Yakın olanları `VRU_near_ego_path`
zaten alıyor: ego şeridinde `|d_lat| ≤ 1.75 m`, `ds ≤ 12 m` olan 15 VRU'nun 12'si yanıyor;
kaçan 3'ü yay-mesafesi / kuş-uçuşu metrik farkından, sistematik boşluk değil. Karar: 12 m'nin
ötesindeki duran yaya ego planını etkilemiyor.

**Asıl sebep — veride debris yok.** `neighbor_agents_past[..., 8:11]` sadece 3 slot:
araç / yaya / bisiklet. Ölçüldü: **3825 geçerli ajanın hepsi** bu üçten birine düşüyor,
sınıfsız ajan **0**. Yani koni, bariyer, czone_sign, generic_object eğitim tensörüne hiç
girmiyor — bu kanal fiilen "duran araç" demek. nuPlan'da bu nesneler var ve kendi kodumuz
deployment'ta zaten sorguluyor ([`state_lattice_path_planner.py:88-90`](Planner/state_lattice_path_planner.py#L88-L90)),
sadece eğitim tensörüne ulaşmıyorlar.

**Geri getirme koşulu:** o dört nesne tipini eğitim tensörüne eklemek — yani bir
`data_process` değişikliği daha + tam yeniden extraction. Yapılırsa kanal gerçekten
bağımsız bir kavram olur ("yolda duran, araç olmayan engel"). Yapılmazsa en fazla
`same_lane_ahead`'in refinement'ı olarak tutulabilir.

### Kapatmanın L0 ajan tablosuna etkisi

Üç kanal kapandıktan sonra L0 ajan sözlüğü **10 → 7**:

| kanal | yanma | oran | TEK | tek/yanma |
|---|---|---|---|---|
| `same_lane_ahead` | 486 | 4.6% | 288 | 59.3% |
| `leftAdjacent` | 1124 | 10.5% | 808 | 71.9% |
| `rightAdjacent` | 929 | 8.7% | 693 | 74.6% |
| `sharesIntersectionWith` | 1435 | 13.5% | 419 | 29.2% |
| `VRU_near_ego_path` | 339 | 3.2% | 201 | 59.3% |
| `inCrosswalk` | 358 | 3.4% | 80 | 22.3% |
| `sharedTrafficControl` | 1705 | 16.0% | 725 | 42.5% |

Multi-fire histogramı: 0 kanal %56.6 · 1 kanal %30.2 · 2 kanal %10.5 · 3 kanal %2.5 ·
4 kanal %0.3 · **5+ kanal %0.0**.

Kapatmanın etkisi tek-yanma oranlarında görülüyor (önce → sonra):
`leftAdjacent` %5.6 → %71.9, `rightAdjacent` %27.0 → %74.6, `same_lane_ahead` %1.6 → %59.3.
Kanallar artık birbirinin yerine geçmiyor; 5+ kanal yanan çift kalmadı.

**Bedeli açıkça kaydedilsin:** grafiğe hiç girmeyen ajan oranı %41.8 → %56.6. Bu üç
kanal kapalı kaldığı sürece kapsama düşük; yukarıdaki geri getirme koşullarından
biri sağlanana kadar bu bilinen bir açık.

### Henüz çözülmemiş — L0 harita tarafı

- `inCrosswalk` ve `onExpertRoute` **totolojik**: `act[M_IN_CROSSWALK] = is_cw & elem_valid`,
  `act[M_ON_EXPERT_ROUTE] = is_rt & reach & elem_valid`. `is_cw`/`is_rt` yalnızca elemanın
  `S = L + C + R` dizilimindeki dilimini söylüyor; model bunu token pozisyonundan zaten
  biliyor. 8 harita kanalının 2'si sıfır bilgi taşıyor.
- `inIntersection` (%22.6, TEK %40.6) elemanın **ego'ya en yakın** noktasına bakıyor;
  60 m ileride kavşaktan geçen uzun şerit yanmıyor, en yakın noktası kavşağa değen şerit
  tüm boyunca yanıyor.
- `trafficControl` (%24.1, TEK %48.7) = "şeridin herhangi bir TL kaydı var" veya "en yakın
  nokta dur poligonuna ≤ 3 m". "TL kaydı var" ≠ "ego için kontrol ediyor"; sinyalize
  kavşakta çapraz trafiğin şeritleri de kayıt taşıyor.

### Doğrulanmayan şüphe (kayda geçsin)

`_point_in_polys`'te halka kapanma hatasından şüphelenildi: kavşak poligonlarının %26'sı,
dur poligonlarının %100'ü `V < P = 20` dolgulu ve kapanış kenarı (`V−1 → 0`) `roll` ile
geçersiz noktaya düşüp atılıyordu. Ölçüldü: **11.4 M kararda 0 değişiklik**, çünkü nuPlan
poligonları zaten açıkça kapalı saklıyor (son geçerli nokta == ilk nokta, 1294/1294).
Sertleştirme kodda bırakıldı ama hiçbir sonucu değiştirmedi.

---

## v3 — `sharedTrafficControl` kapatıldı, `sharesIntersectionWith` KG'ye göre yeniden yazıldı

KG mini split'te koşuldu (20 senaryo × 20 kare, tek işçi, `MemoryMax=14G`) ve iki predicate'in
KG'deki **gerçek tasarımı** okundu. Ölçüm birimi: ego–ajan çift-karesi (n = 27,393).

### KG'de bu ikisi ne yapıyor

| | eksen | mekanizma | KG'de yanma |
|---|---|---|---|
| `shared_intersection_conflict` | **geometri** | kavşak poligonu ID kesişimi | 3747 (13.7%) |
| `shared_traffic_control` | **trafik kuralı** | lane-connector kimlik kesişimi | 119 (**0.4%**) |

Kasıtlı olarak farklı iki eksen. KG'deki örtüşmeleri: **IoU %0.7** (25 çift-kare).
`P(intersection | tctl) = 21.0%`, `P(tctl | intersection) = 0.7%`.

Bizim uygulamamızdaki %37 / %44 örtüşme **KG'den miras değil, bizim ürettiğimiz** bir şeydi.

### `sharedTrafficControl` — KAPATILDI

KG tasarımı ([`relevance.py:12,124-131`](../nuPlan_Predicates_KG/nuplan_predicates_modular/categories/relevance.py#L124-L131)):

```python
light_connectors  = {ışık taşıyan lane_connector id'leri, O KARE}
entity_connectors = {primary_map_object_id} ∪ outgoing_object_ids
relevant_light    = bool(light_connectors & subject_connectors & object_connectors)
# kapı: center ≤ 45 m, skor +55, öncelik 60
```

**Üçlü kimlik kesişimi**: aynı connector hem sinyalli olacak, hem ego'nun mevcut-veya-sonraki,
hem ajanın. Bu yüzden aynı kırmızıda yan yana duran iki araç yanmıyor (paralel connector'lar,
farklı ID) — sadece ego ile aynı connector'daki, tam önündeki araç yanıyor.

npz'de ne şerit ID'si var, ne connector bağlantısı, ne hangi connector'un sinyalli olduğu.
Bizim uygulamamızda `"shared"` kısmı hiç yoktu; ölçüldü: bizde **%16.0**, KG'de **%0.4** —
40 kat şişik; yanmalarımızın sadece %12.9'u ego ile kontrol paylaşıyordu.

Yeniden extraction'a değmemesinin üç sebebi:
1. KG'de %0.4 — 27 bin çift-karede 119 tane.
2. Üçlü kesişim gereği fiilen `same_lane_ahead`'in alt kümesi.
3. KG'nin kendi öncelik tablosu onu `same_lane_ahead`'in **altına** koyuyor
   (`shared_traffic_control: 60`, `same_lane_ahead: 75`).

`traversing_traffic_light_intersection` tipindeki bir senaryoda (`00015fc2840d5313`),
1615 çift-karede `shared_traffic_control` **sıfır kez** yandı — yani nadir olması
"ışıklar az" demek değil, üçlü kesişimin pratikte tutmaması demek.

### `sharesIntersectionWith` — KG tasarımına göre YENİDEN YAZILDI

KG ([`map.py:505-535, 926-939`](../nuPlan_Predicates_KG/nuplan_predicates_modular/categories/map.py#L926-L939)),
her varlık için kavşak üyeliğini **iki kümede** tutuyor:

```python
if item["center_covered"]:                       # MERKEZ poligonun içinde
    record["intersection_ids"].add(object_id)          -> np:inIntersection
if item["intersection_area_m2"] > area_epsilon:  # AYAK İZİ poligonla örtüşüyor
    record["intersection_footprint_ids"].add(object_id) -> np:intersectsIntersection
```
Çiftte bu iki kümenin **birleşimi** kesiştiriliyor, üstüne `center ≤ 55 m`.

Bizim önceki halimizdeki iki sapma düzeltildi:

| | önce | sonra (KG'ye sadık) |
|---|---|---|
| ego tarafı | koridorun **0–60 m ilerisi** | ego'nun **şu anki ayak izi** |
| ajan tarafı | sadece merkez noktası | merkez + 4 köşe (iki kümenin birleşimi) |
| mesafe kapısı | yok | `center ≤ 55 m` |

Ego henüz kavşağa varmadan kanal artık yanmıyor.

### L0 ajan sözlüğü: 10 → 5

| kanal | yanma | oran | TEK | tek/yanma |
|---|---|---|---|---|
| `same_lane_ahead` | 486 | 4.6% | 429 | 88.3% |
| `leftAdjacent` | 1124 | 10.5% | 911 | 81.0% |
| `rightAdjacent` | 929 | 8.7% | 781 | 84.1% |
| `sharesIntersectionWith` | 1348 | 12.6% | 891 | 66.1% |
| `VRU_near_ego_path` | 339 | 3.2% | 293 | 86.4% |

Multi-fire: 0 kanal %64.7 · 1 kanal %31.0 · 2 kanal %4.3 · **3 kanal 1 çift** · 4+ yok.
Sözlük artık pratikte bir **partition**. Tek-yanma oranlarının ilk hallerinden farkı:
`same_lane_ahead` %1.6 → %88.3, `leftAdjacent` %5.6 → %81.0, `rightAdjacent` %27.0 → %84.1.

`sharesIntersectionWith` oranı KG ile tutuyor: **biz %12.6, KG %13.7**.

### Açık kalan tek konu

Yeni `sharesIntersectionWith`'in yön dağılımı: aynı yön %69.9, kesen %18.3, karşı %11.7
(önceki hali: %67.9 / %20.3 / %11.8). Neredeyse değişmedi — **çünkü KG'nin kendi tanımında da
yön testi yok**, adında "conflict" geçmesine rağmen saf ID kesişimi. Yani bu bizim uygulama
hatamız değil, KG tasarımının bir özelliği. Yön filtresi eklenecekse "KG ilişkisi + karar-ilgisi
kısıtı" olarak deklare edilmeli. Henüz eklenmedi.

**Kapsama bedeli:** grafiğe hiç girmeyen ajan %41.8 → %64.7.

---

## v3 — `sharesIntersectionWith`'e kesişim filtresi (2026-09-03)

Önce ajan-bazında doğrulama yapıldı: KG, viz'deki iki trainval sahnesinde
(`b9ab639fd02f5199`, `40042cb53ea15317`) izole DB'lerle koşuldu ve eşleşen **8/8 ajanda**
bizim üyelik-tabanlı kanalımızla KG'nin `shared_intersection_conflict`'i **birebir aynı**
çıktı — ego'nun hemen yanındaki "anlamsız" ajanlar dahil. Yani sorun uygulamada değil,
**KG tanımının kendisinde**: adı "conflict", testi salt kutu üyeliği
(`bool(subject_ids & object_ids)`, [map.py:939]) ve kavşak kutuları büyük
(köşegen medyan 28.6 m, p90 69.8 m).

**Eklenen filtre (bizim karar-ilgisi kısıtımız — KG'de YOK, makalede katkı olarak
deklare edilecek):** KG üyeliğinin üstüne, ajanın **şeridi ego koridorunu kesmeli**.

- Şerit-kesişme testi: şerit polyline'ının koridora göre yanal işareti ardışık iki
  noktada değişiyorsa (merkez çizgiyi geçiyor) **ve** o noktadaki şerit yönü koridor
  tegetinden `SAME_FLOW_RAD`'dan fazla sapıyorsa. Açı şartı ego'nun kendi şeridinin
  `lat≈0` titremesini eler.
- Ajan→şerit ataması: ≤3 m nokta mesafesi içinde, ajan yönüyle ≤60° uyumlu en yakın şerit.
- Şerit atanamayana geri düşüş: ajanın anlık yönü koridor tegetini 30–150° kesiyorsa.
- **Yapısal kalır**: gelecek yok, hız yok; heading yalnızca şerit seçimi/açı için.
  Eskiden L0'da olup L1'e taşınan `collision_course`'tan farkı: o *araçların yörüngeleri*
  hakkındaydı (kinematik), bu *yolların topolojisi* hakkında — kesişen şeritte duran
  araç da yanar, paralel şeritte hızlı giden asla yanmaz.

Etki (1118 sahne):

| | önce | sonra |
|---|---|---|
| yanma | 1348 (12.6%) | **203 (1.9%)** |
| aynı yön | 69.9% | **17.5%** |
| KESEN | 18.3% | **75.7%** |
| karşı yön | 11.7% | 6.8% |

Kalan %17.5 aynı-yön çoğunlukla sola dönecek ajanlar: şeridi ego yolunu kesiyor ama anlık
heading henüz paralel — şerit-tabanlı testin heading-tabanlıdan farkı tam bu vaka.

### Nihai L0 ajan sözlüğü (5 kanal)

| kanal | yanma | oran | TEK | tek/yanma |
|---|---|---|---|---|
| `same_lane_ahead` | 486 | 4.6% | 468 | 96.3% |
| `leftAdjacent` | 1124 | 10.5% | 1117 | 99.4% |
| `rightAdjacent` | 929 | 8.7% | 922 | 99.2% |
| `sharesIntersectionWith`* | 203 | 1.9% | 142 | 70.0% |
| `VRU_near_ego_path` | 339 | 3.2% | 302 | 89.1% |

\* KG ilişkisi + bizim kesişim kısıtımız.

Multi-fire: 0 kanal %71.7 · 1 kanal %27.7 · 2 kanal %0.6 · 3+ **yok**. Sözlük fiilen
karşılıklı-dışlayıcı. Görseller: `viz_out/v3_inspect/v3_agent_5ch_xfilter.png` (genel),
`v3_intersect_xfilter.png` (sadece intersect yanan sahneler, `--focus` ile).

`eval_channels.py`'ye `--focus <kanal>` eklendi: yalnızca o ajan kanalının yandığı
sahneleri çizer.

**Bilinen sorun (çözüm ertelendi):** kesişim filtresi kesişme noktasını `s ∈ [−5, 80]`
kabul ediyor; ego çatışma noktasını geçtikten sonra (kavşaktan çıkış fazı) arkadaki ajan
yanabiliyor — ajan ego'nun geleceğini değil geçmişini kesiyor. Ölçülen örnek:
`e990429a8ff45085`, ajan 4 (−19.5, −0.1), poligon merkezi ego'nun 8 m arkasında.
Aday düzeltme: kesişme şartı `s > 0` (çatışma ileride olmalı). Henüz uygulanmadı.

---

## v3 — kalan 4 kanalın kalibrasyonu (2026-09-03, 800 sahne)

| kanal | yöneten büyüklük | bizim eşik | KG eşiği | karar |
|---|---|---|---|---|
| `same_lane_ahead` | `ds` | ≤ 80 m | `directional_same_lane_ahead_m` = 80 | **aynı, kaldı** |
| `left/rightAdjacent` | `d_center` | 40 m | `directional_adjacent_m` = 35 | **35'e hizalandı** |
| `VRU_near_ego_path` | `d_center` | 12 m | `vulnerable_road_user_distance_m` = 35 | **35'e hizalandı** |
| `sharesIntersectionWith` | `center` | 55 m | 55 | aynı (bilinen sorun yukarıda) |

Ölçüm gerekçeleri:

- **ahead**: yanmaların ds dağılımı p50 14.6 / p90 29.4 / p99 55.5 m — 80 m tavan pratikte
  bağlayıcı değil (%4.2'si >40 m ve onlar da uzak kalıyor, 8 s min mesafe p50 40 m). Dokunulmadı.
- **adjacent 40→35**: 35–40 m bandında sadece 27 yanma (%1.8) vardı ve 8 s min mesafeleri
  p50 24.9 m (uzak kalıyorlar). Hizalama neredeyse bedava, sapma işareti kalktı.
  Yanma: adjL 1124→1109, adjR 929→910.
- **VRU 12→35**: 12–35 m bandında, aynı yanal/ds şartlarını sağlayan **133 VRU** vardı ve
  çöp değiller: 8 s min mesafe p50 12.1 m, **%40'ı 10 m'ye, %82.9'u 20 m'ye giriyor**.
  Önceki kapsama analizindeki "kaçanların %53'ü yaya" bulgusunun büyük kısmı buydu.
  Yanma: 339→449. `*bizim` sapma işareti kalktı — kanal artık tamamen KG değerinde.
- **adjacent'ın arkada payı** (düzeltilmiş `ds<0`): **%47.0** (lon_ok = ds ≥ −25 tasarımı gereği).
  Bilinçli tercih: hedef şeritteki takipçi, şerit değişiminin ana kısıtı — `same_lane_behind`'i
  kapatırken dayandığımız argüman tam olarak buydu. Değiştirilmedi.

### Kalibrasyon sonrası tablo (1118 sahne)

| kanal | yanma | oran | TEK | tek/yanma |
|---|---|---|---|---|
| `same_lane_ahead` | 486 | 4.6% | 468 | 96.3% |
| `leftAdjacent` | 1109 | 10.4% | 1102 | 99.4% |
| `rightAdjacent` | 910 | 8.5% | 902 | 99.1% |
| `sharesIntersectionWith` | 203 | 1.9% | 134 | 66.0% |
| `VRU_near_ego_path` | 449 | 4.2% | 403 | 89.8% |

Multi-fire: 0 kanal %71.1 · 1 kanal %28.2 · 2 kanal %0.7 · 3+ yok.
Beş kanalın dördü artık birebir KG eşiklerinde; tek sapma `sharesIntersectionWith`'in
kesişim kısıtı (deklare edilmiş katkı) ve onun bilinen çıkış-fazı sorunu.

---

## v3 — boylamsal menzil: hıza bağlı kalibrasyon (kullanıcı kararı, 2026-09-03)

Kural: öne doğru menzil ego hızına bağlı, arkaya doğru sabit ve dar.

```
ds_max = clamp(v_ego · LON_TAU_S, LON_MIN_M, AHEAD_MAX_M)     # one dogru
ds_min = −LON_BEHIND_M                                         # arkaya (adjacent)
LON_TAU_S = 3.0 s · LON_MIN_M = 25 m · LON_BEHIND_M = 10 m · AHEAD_MAX_M = 80 (KG, dış sınır)
```

`same_lane_ahead` ve `left/rightAdjacent`'ın boylamsal kapsamı bu kurala bağlandı
(eski: ahead ds ≤ 80 statik; adjacent ds ≥ −25).

**Taban neden 25:** ilk deneme taban=10 idi ve öne menzili fiilen 10 m yaptı — validation'da
ego yavaş (hız p25 = 0.0, p50 = 2.2 m/s), τ ancak v > taban/τ üstünde söz sahibi. Taban=10,
liderlerin %74'ünü kesti; kesilenler ego p50 1.8 m/s iken 15 m ilerideki araçlardı — yani
kuyrukta fiilen takip edilen lider (L1 `follow` etiketinin öznesi). Süpürme (600 sahne,
271 aday lider): taban 10 → %74.2 kayıp · 15 → %39.1 · 20 → %23.2 · **25 → %13.7** · 30 → %8.1.
25 m, durgun kuyrukta 3-4 araç boyu lideri korur; uzak kuyruk kesilir; v > 8.3 m/s'de
taban devre dışı kalır ve menzil v·3 s olur.

### Boylamsal kalibrasyon sonrası tablo

| kanal | yanma | oran | TEK | önceki (statik) |
|---|---|---|---|---|
| `same_lane_ahead` | 415 | 3.9% | 95.7% | 486 |
| `leftAdjacent` | 907 | 8.5% | 99.2% | 1109 |
| `rightAdjacent` | 688 | 6.5% | 99.3% | 910 |
| `sharesIntersectionWith` | 203 | 1.9% | 67.5% | 203 |
| `VRU_near_ego_path` | 449 | 4.2% | 89.8% | 449 |

Multi-fire: 0 kanal %75.7 · 1 kanal %23.6 · 2 kanal %0.7 · 3+ yok.
Görsel: `viz_out/v3_inspect/v3_agent_5ch_floor25.png` (aynı seed-1 sahneleri; yakın liderler
korunuyor, sürünürken uzak kuyruk ve ±10 m dışındaki adjacent'lar kesiliyor, hızlı ego'da
27 m lider duruyor).

Not: bu kural KG'nin statik `directional_*` eşiklerinden bilinçli bir sapmadır (KG tavanları
dış sınır olarak korunur); makalede "zaman-aralığı tabanlı karar-ilgisi kalibrasyonu" olarak
deklare edilecek.

---

## v3 — NİHAİ L0 AJAN SÖZLÜĞÜ (2026-09-03, kalibrasyon sonrası)

`same_lane_behind` kullanıcı kararıyla **dar menzille yeniden açıldı**: eski −25 m tanımı
nedensellik-ters diye kapatılmıştı; yeni tanım `ds ≥ −10 m` (`LON_BEHIND_M`) — uzak kuyruk
değil, sadece tampondaki takipçi. τ da 3 → **5 s** yapıldı (validation'da fark +5 yanma —
taban 25 m, v > 5 m/s'e kadar bağlayıcı; asıl etkisi deployment hızlarında, 15 m/s'de
menzil 45 → 75 m ≈ KG'nin 80'i).

### Aday sözlüğün tamamı — kullanım durumu

| kanal | kullanım | koşul / not |
|---|:---:|---|
| `same_lane_ahead` | **1** | `inlane & same_flow & 0 < ds ≤ clamp(v·5s, 25, 80)` |
| `same_lane_behind` | **1** | `inlane & same_flow & −10 ≤ ds < 0` (KG 25'ten daraltıldı) |
| `leftAdjacent` | **1** | yanal bant & same_flow & `d_center ≤ 35` (KG) & `−10 ≤ ds ≤ clamp(v·5s,25,80)` |
| `rightAdjacent` | **1** | aynı, sağ |
| `sharesIntersectionWith` | **1** | KG üyelik (ayak izi, ≤55 m) **+ kesişim kısıtı** (bizim; bilinen çıkış-fazı sorunu açık) |
| `VRU_near_ego_path` | **1** | vru & `d_center ≤ 35` (KG) & yanal ≤ 1.5·şerit & ds ≥ −5 |
| `inCrosswalk` | **0** | kapalı — düzeltilmiş haliyle bile katkı ~14 ajan/800 sahne |
| `staticObstacleOnPath` | **0** | kapalı — veride debris yok, fiilen `ahead`'in alt kümesi (%95.4) |
| `onRouteCorridor` | **0** | kapalı — subsumption (4 şerit kanalını kapsıyordu); residual/L1 seçenekleri notlu |
| `sharedTrafficControl` | **0** | kapalı — connector ID'leri npz'de yok; KG'de zaten %0.4 |

### Nihai tablo (1118 sahne, 10 659 çift)

| kanal | yanma | oran | TEK | tek/yanma |
|---|---|---|---|---|
| `same_lane_ahead` | 420 | 3.9% | 402 | 95.7% |
| `same_lane_behind` | 279 | 2.6% | 273 | 97.8% |
| `leftAdjacent` | 913 | 8.6% | 906 | 99.2% |
| `rightAdjacent` | 692 | 6.5% | 687 | 99.3% |
| `sharesIntersectionWith` | 203 | 1.9% | 131 | 64.5% |
| `VRU_near_ego_path` | 449 | 4.2% | 403 | 89.8% |

Multi-fire: 0 kanal %73.0 · 1 kanal %26.3 · 2 kanal %0.7 · 3+ **yok**.
