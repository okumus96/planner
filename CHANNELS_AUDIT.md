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
