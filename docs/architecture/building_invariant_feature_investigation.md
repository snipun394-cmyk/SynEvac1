# Building-Invariant Feature Representation Investigation

Status: **INVESTIGATION ONLY.** No code changed. No feature added, removed, or promoted to any
canonical schema. No model retrained. No GNN implemented. No architecture redesigned. Recommendation
and Guidance untouched. This document is the sole deliverable.

---

## Scope decision (stated upfront, load-bearing for everything below)

The milestone brief frames this as a direct continuation of "the Predictive Model Benchmark Campaign,"
which found no identity leakage and asked why cross-topology generalization is poor. That framing
conflates two genuinely different pipelines, and getting this right changes which evidence is admissible:

- **Pipeline B** (`ai_registry`, `CANONICAL_LIVE_SCHEMA`, 25 whole-building fields) is what the
  Benchmark Campaign actually trained, evaluated, and registered — against **one fixed building**.
  Pipeline B has **never been tested across multiple buildings at all**. Its cross-building
  generalization is unmeasured, not measured-and-poor.
- **Pipeline A** (`predictive_dataset`, `schema_v4`, 15 per-candidate fields) is where "cross-topology
  generalization is poor" is an actual, quantified, twice-replicated finding — the [[cross_topology_
  generalization_investigation]] (commit `e8d728a`) and its predecessor [[localized_predictive_model_
  v3_1_robustness_milestone]] both measured this directly, on 4-24 structural variants across up to 6
  topology families.

**This investigation is therefore conducted primarily against Pipeline A**, since that is the pipeline
with an actual generalization failure to root-cause — consistent with this session's own established
practice of correcting an imprecise premise rather than silently inheriting it (see the immediately
preceding chat turn, and the Dataset Pipeline Audit milestone's "two independent pipelines" finding).
Pipeline B's schema is still fully audited in Phase 1 below, for completeness and because any structural
descriptor found valuable for Pipeline A is a candidate for Pipeline B too — but Phase 3's concept-shift
analysis and Phase 5's gap analysis are Pipeline-A-evidenced, and say so explicitly wherever they draw on
Pipeline-A-only data.

---

## Phase 1 — Feature Audit

### Pipeline A: `predictive_dataset.schema_v4.CANDIDATE_FEATURE_SCHEMA_V4` (15 fields, per-candidate/tick)

Grain: one row per (scenario, tick, candidate) — a Door/Exit/Stair edge in the building's navigation
graph. Sources: `predictive_dataset/schema.py` (9 frozen V1 fields) + `schema_v4.py`'s 6 promoted fields.
Building-invariance and topology-dependence verdicts for the 6 numeric/count fields marked "audited"
below are drawn directly from the Cross-Topology Investigation's own SMD (covariate shift) and
conditional-rate-spread (concept shift) measurements — not estimated here. The remaining fields were not
individually measured by that investigation and are classified qualitatively, disclosed as such.

| Feature | Physical meaning | Mathematical meaning | Source | Building-invariant (definition)? | Topology-dependent (value↔target relationship)? | Expected cross-building usefulness |
|---|---|---|---|---|---|---|
| `total_active_occupant_count` | Whole-building occupants still evacuating | Count | `MultiAgentSimulationResult.occupants` (not yet arrived) | Yes | **Concept shift, audited**: 35× conditional-rate spread at matched occupancy across families | Low alone; needs building-size normalization to compare across buildings of different capacity |
| `candidate_type` | Door / Exit / Stair | Categorical | `Edge.edge_type` | Yes | Distributional shift by design (e.g. `multi_exit_wide` has zero Stairs) | High as a conditioning variable, not a standalone predictor |
| `candidate_capacity` | Engineering capacity estimate | Count (persons/interval) | `crowd_intelligence.capacity.*_capacity()` | Yes | Not individually audited; qualitatively low concept-shift risk (a physical/engineering quantity, not behavior-dependent) | Moderate-high; a genuine physical invariant |
| `candidate_walking_distance` | Physical edge length | Meters | `Edge.walking_distance` | Yes | **Covariate shift WITHOUT concept shift, audited** (SMD 1.20, conditional-rate spread only 0.063) | High once building-relative-normalized (raw scale differs by building size, but the value↔outcome relationship transfers) |
| `candidate_traversable` | Can currently be walked | Boolean | `Edge.traversable` | Yes | Not individually audited; qualitatively low concept-shift risk | Moderate; a gating condition, not itself predictive of degree |
| `candidate_adjacent_zone_occupancy` | Occupants in approach-side zone now | Count | `OccupancySnapshot` | Yes | **Covariate shift, audited** (SMD 1.06); concept-shift not isolated separately from `total_active_occupant_count` | Low-moderate alone; same building-size confound as total occupancy |
| `candidate_queue_length` | Occupants currently queued here | Count | discrete-event queue bookkeeping | Yes | **Neither** covariate nor concept shift, audited (SMD 0.059, rate spread 0.075) — **the single most cross-building-trustworthy feature measured** | **High** — this is the one feature the Investigation found the model can trust unconditionally across topology |
| `candidate_approaching_count` | Occupants whose route includes this candidate | Count | Route-derived, plan-prior | Not individually audited | Not individually audited; likely correlated with queue_length's stability given shared demand semantics | Likely moderate-high, untested directly |
| `candidate_congestion_level` | LOW…CRITICAL demand-vs-capacity classification | Categorical, derived from queue+approaching vs. capacity | `crowd_intelligence.congestion` | Yes | Derived from a stable feature (queue) and a moderate one (capacity) — likely more stable than its raw inputs' worst component | Moderate-high |
| `candidate_recent_flow_rate` | Completed crossings, trailing 60s | Count/rate | movement completion events | Yes | **Concept shift, audited**: 33× conditional-rate spread at matched flow-rate across families (worst in `twin_stair_highrise`) | Low-moderate alone; the single feature most responsible for `twin_stair_highrise`'s uniquely poor ROC-AUC |
| `candidate_congestion_trend` | RISING/STABLE/FALLING/UNKNOWN | Categorical, derived from a 30s demand-proxy delta | derived | Yes | Not individually audited; false-positive/negative concentration analysis (Phase 4 of the Investigation) shows RISING/UNKNOWN trend is where errors concentrate **consistently across all 4 families** — suggests this feature's relationship IS fairly stable | Moderate-high |
| `candidate_alternative_route_count` | Sibling candidates sharing a zone | Count, purely structural | zone-adjacency | Yes | **Covariate shift, audited** (SMD 0.715); concept-shift not isolated | Moderate; local, shallow structural signal (see Phase 2 — genuinely one-hop only) |
| `candidate_betweenness_centrality` | Fraction of all-pairs shortest paths through this edge | `networkx.edge_betweenness_centrality`, Brandes' algorithm, distance-weighted, [0,1] | `graph_context_v4.py` | Yes | Not individually audited in isolation; the 3-feature graph-context bundle gave **+3.3% avg, non-negative in every family** | **Validated positive**, modest — the strongest evidence-backed structural feature this investigation has |
| `candidate_is_bridge` | Cut-edge (single point of failure) | Boolean, `networkx.bridges` | `graph_context_v4.py` | Yes | Not individually audited in isolation (bundled result above) | Validated positive (bundled), modest |
| `candidate_upstream_catchment_count` | Zones whose shortest path to OUTSIDE uses this edge | Count, weighted-Dijkstra, **single shortest path only per zone** (disclosed simplification) | `graph_context_v4.py` | Yes | Not individually audited in isolation (bundled result above) | Validated positive (bundled), modest — **but see Phase 2: its own single-shortest-path-only limitation is a specific, evidenced gap** |

**Experimental, not-yet-canonical fields** (`predictive_dataset/experimental_features_v4.py`,
`rel_queue_to_capacity`, `rel_flow_to_capacity`, `rel_walking_distance_to_building_mean`,
`rel_alt_route_share`, `rel_adjacent_occupancy_to_building_occupancy`): ratios of the above against
static per-building normalization constants. **Twice tested, twice found unreliable**: V3.1's Dataset V2
run found "no clear, robust benefit... mostly flat-to-slightly-negative"; the Cross-Topology
Investigation's Dataset V3 run found the same pattern (+0.6% average, helping the two hard families but
actively hurting the two easier ones by 3-6%). Not part of the audit table above because they are not
canonical, but directly relevant to Phase 5.

### Pipeline B: `ai_features.CANONICAL_LIVE_SCHEMA` (25 fields, whole-building aggregate)

Grain: one row per scenario (pre-simulation configuration → whole-scenario outcome). Full field-by-field
semantics already documented exhaustively in `ai_features/feature_schema.py` itself (reproduced in
condensed form; see that file for the complete `AIFeatureField` metadata, including live-vs-simulation
source paths already audited field-by-field in [[ai_live_feature_parity]]).

| Feature group | Fields | Building-invariant (definition)? | Topology-dependent? |
|---|---|---|---|
| Occupancy (3) | `total_occupant_count`, `occupancy_observed`, `mean_occupant_track_confidence` | Yes | **Untested** — Pipeline B has one building |
| Camera health (3) | `camera_total_count`/`active`/`offline` | Yes | Untested; near-constant in this campaign (benchmark found ≈0 importance) |
| Sensor health (3) | `sensor_total_count`/`active`/`offline` | Yes | Untested; same near-constant pattern |
| Smoke/heat detector condition (6) | coverage/alarm/fault × 2 kinds | Yes | Untested |
| `building_alarm_status`, FACP (6) | aggregated status + panel fields | Yes | Untested |
| Building control (2) | pending/confirmed counts | Yes | Untested |
| **Structural/graph** | **None** | n/a | n/a |

**The headline finding of this audit table**: Pipeline B's entire 25-field schema contains **zero
structural/topological descriptors of any kind** — no walking distance, no capacity, no graph position,
nothing analogous to Pipeline A's `candidate_*` fields. This is by design (Pipeline B predicts a
whole-scenario outcome from whole-building aggregate state, never a per-location one), not an oversight —
but it means **none of Phase 2-5's structural-feature analysis transfers to Pipeline B without first
deciding whether Pipeline B should grow a per-location grain at all**, a design question well beyond this
investigation's scope (see Phase 6).

---

## Phase 2 — Missing Structural Information

Each of the milestone brief's 13 example descriptors, assessed against what `graph_context_v4.py`
already computes, plus 4 additional candidates identified by this investigation's own analysis of the
codebase and the Cross-Topology Investigation's own disclosed simplifications.

| # | Descriptor | Already covered? | Physical meaning | Verdict |
|---|---|---|---|---|
| 1 | Downstream capacity | **No** | The minimum capacity along the ENTIRE path from this candidate to outside (a true bottleneck/min-cut value) — `candidate_capacity` is this ONE edge's own capacity only; nothing aggregates capacity along a path | **Genuinely absent** |
| 2 | Alternate path availability | **Partial** | `candidate_alternative_route_count` counts sibling candidates sharing a zone — a one-hop, local notion. A true "how many genuinely independent (edge-disjoint) simple paths exist from this zone to OUTSIDE" (Menger's-theorem territory) is not computed | **Partially covered — existing feature is shallow** |
| 3 | Network redundancy | **Partial** | `is_bridge` is the boolean extreme (zero redundancy). A graded measure — e.g. local edge-connectivity, or "how many additional edge failures would disconnect this zone" — does not exist | **Partially covered — only the binary extreme case exists** |
| 4 | Local branching factor | **Mostly covered** | Degree of the candidate's own zone(s) — `candidate_alternative_route_count` is essentially "approach-side zone degree − 1" already | **Largely covered**, though the FAR-side zone's branching is not separately captured |
| 5 | Shortest-path diversity | **No** | `upstream_catchment_count` explicitly counts only ONE shortest path per zone when ties exist (`graph_context_v4.py`'s own disclosed simplification: "a zone with multiple equally-short paths is only counted once, along ONE of them"). Two zones — one with a unique shortest path, one with two equally-short paths — are currently indistinguishable | **Genuinely absent, and directly evidenced by an existing, disclosed code-level simplification** |
| 6 | Exit accessibility | **Partial** | No explicit "distance to nearest exit" or "number of exits reachable within K hops from the far side of this candidate" feature exists; only this candidate's OWN edge length is known | **Partially covered** |
| 7 | Downstream bottleneck probability | **No** | Whether ANOTHER candidate further along this path is currently congested — a genuinely RELATIONAL, multi-hop, dynamically-varying feature. Nothing in the current per-candidate-independent flat representation propagates state between candidates | **Genuinely absent — and the closest analog to what message-passing/GNN architectures compute automatically** |
| 8 | Graph diameter | **No** | Whole-building scalar: max shortest-path length between any two nodes. Not computed at all, at any grain | **Genuinely absent, cheap to add as a global-context field** |
| 9 | Local clustering | **No** | Per-zone/edge cycle density. A coarse whole-variant `has_cycle` boolean exists in `topology_signature.py` for dataset-diversity bookkeeping, but nothing per-candidate | **Genuinely absent at the trainable-feature grain** |
| 10 | Articulation-neighbourhood metrics | **Partial** | `is_bridge` is the EDGE-level cut analog; the NODE-level analog (articulation points / cut-vertices — a ZONE whose removal disconnects the graph) is not computed, nor is "proximity to an articulation point" | **Partially covered — only the edge-level version exists** |
| 11 | Path overlap | **No** | Whether this candidate's "alternative routes" (per #2/#4) actually reconverge downstream onto the SAME single stair/exit, making them not truly independent | **Genuinely absent — connects directly to #1 and #2, and is arguably the single most physically important gap** (a building can look highly redundant at the door level while funneling everyone through one shared stair) |
| 12 | Edge criticality | **Partial** | `betweenness_centrality` + `is_bridge` give a STATIC ranking. A true removal-IMPACT criticality (e.g., how much would evacuation time increase if this edge failed) requires re-solving shortest paths with the edge removed — not computed | **Partially covered — ranking exists, impact-magnitude does not** |
| 13 | Evacuation subgraph characteristics | **No** | Properties of the subgraph CURRENTLY reachable from a zone given the CURRENT (dynamic) `candidate_traversable` state of every edge — e.g., total capacity of currently-reachable exits. All existing graph-context features (betweenness/bridge/catchment) are computed from STATIC geometry only, never re-evaluated against current blocked/open state | **Genuinely absent — and directly motivated by the Benchmark Campaign's own finding that `exit_block_tier=blocked` scenarios are the hardest slice for evacuation-time prediction (MAE 277.7s vs 91.0s)** |

**Additional candidates identified by this investigation:**

| # | Descriptor | Physical meaning | Why it's a candidate |
|---|---|---|---|
| 14 | Capacity-weighted betweenness | Same Brandes' computation as #existing, but edge weight = `1/capacity` (or a combined distance×capacity cost) instead of distance alone | A structurally-central edge that is ALSO narrow is far more critical than one that is central but wide — the current betweenness feature is blind to capacity entirely |
| 15 | Hop-count depth from OUTSIDE | Discrete edge-count (not walking distance) from this candidate to the nearest exit | A cheap, complementary discrete signal to the continuous distance/catchment features — may be more building-size-invariant than raw distance |
| 16 | Vertical floor-distance | Number of floors between this candidate and ground/exit level | The Benchmark Campaign found `floor_of_ignition=floor_2_upper` scenarios have meaningfully worse regression MAE (142.3s vs 111.3s) and lower classification ROC-AUC — no current feature explicitly encodes "how many floors of vertical travel remain," only the resulting walking distance (which conflates horizontal and vertical) |
| 17 | Reachable-exit capacity under current state | Sum of `candidate_capacity` over every EXIT currently reachable (given current `candidate_traversable` states) from this candidate's zone | The genuinely DYNAMIC counterpart to #13/#1 — directly answers "if I get past this candidate, how much total downstream throughput can actually absorb me right now" |

**Summary**: of the 13 examples given, **1 is largely already covered** (#4), **3 exist only in a static/binary/shallow form with a graded or dynamic version missing** (#3, #10, #12), **4 exist in a shallow local form with a true multi-hop/relational version missing** (#2, #6, #11, and #7 is the fully-relational extreme of this same family), and **5 do not exist in any form** (#1, #5, #8, #9, #13). The pattern across nearly every genuine gap is the same: **the current representation computes properties of a single edge in isolation (even the graph-context features, despite being genuinely graph-theoretic, are still one static number PER EDGE) — it has no feature that aggregates or propagates information ACROSS multiple candidates along a path.** This is the single organizing theme this investigation's Phase 3/5 build on.

---

## Phase 3 — Concept Shift Investigation

**Precisely why identical feature values correspond to different outcomes across families** (grounding
every claim in the Cross-Topology Investigation's own measured evidence, §3 and §4 of that document):

1. **`total_active_occupant_count`, matched at HIGH (16-30)**: 0.6% positive rate in `multi_exit_wide`
   vs. 19.6% in `single_exit_lowrise` — a 35× difference. **Missing structural context**: the SAME raw
   occupant count means completely different pressure depending on the building's total THROUGHPUT
   capacity (how many exits/doors of what width exist to absorb that occupancy) — which is exactly
   **Phase 2's gap #1 (downstream capacity)**. `multi_exit_wide` has, per its own name, more exit
   capacity relative to its occupancy than `single_exit_lowrise` does. **Resolvable in principle by
   engineered features**: a `total_active_occupant_count / building_total_exit_capacity` ratio is a
   direct, computable answer to exactly this ambiguity — notably, this SPECIFIC ratio was never tried;
   the 5 ratios that WERE tried (Phase 6 experiment) normalized by walking distance and candidate count,
   not by aggregate capacity. This is a genuine, specific, testable gap this investigation surfaces that
   prior milestones did not.

2. **`candidate_recent_flow_rate`, matched at >10**: 62.7% positive in `twin_stair_highrise` vs.
   1.9%-10.1% elsewhere. **Missing structural context**: the same flow reading through a Stair in a
   3-4-floor twin-stair building means "this is one of only two vertical arteries for the entire
   building," whereas the same reading through a Door in a 1-floor hub building means "one of several
   equivalent horizontal routes." This is precisely **Phase 2's gap #16 (vertical floor-distance)**
   combined with gap #1 (downstream capacity) — a flow rate is only alarming relative to how much OTHER
   capacity exists to relieve it, and vertical evacuation has structurally less of that than horizontal.
   §13 of the Investigation adds a second, independent contributing factor: `twin_stair_highrise`'s
   congestion episodes have a shorter median lead time (6.13s vs 8.3-8.9s) — a genuine dynamical
   difference in HOW FAST vertical congestion develops, which no static structural feature (existing or
   proposed) can capture; this specific piece of the ambiguity is a property of stair evacuation physics,
   not a missing feature.

3. **Why graph-context features (betweenness/bridge/catchment) only partially resolve this (+3.3%
   average, not transformative)**: they are exactly the RIGHT kind of feature (structural, not
   scale-dependent) but they are STATIC per-edge numbers, computed once from geometry alone. They tell
   you "this edge is structurally important in general" but not "this edge is important RIGHT NOW given
   which OTHER edges are currently open, congested, or already saturated" (Phase 2's gap #13, #17) — nor
   do they capture whether the "alternative routes" they imply actually diverge or reconverge downstream
   (gap #11). `upstream_catchment_count`'s own single-shortest-path-only simplification (gap #5) is a
   second, concrete, code-evidenced reason the feature systematically undercounts structural dependency
   in any building with genuine path ties (e.g., a symmetric two-stair building).

**Classification of every ambiguity found:**

- **Resolvable with engineered features** (still classical, no learned representation needed):
  downstream/reachable capacity ratios (gaps #1, #17), shortest-path-count instead of shortest-path-only
  (gap #5), vertical floor-distance as its own field distinct from raw walking distance (gap #16),
  capacity-weighted betweenness (gap #14), graph diameter and per-zone clustering as global/local context
  (gaps #8, #9). All of these are pure functions of static Building/NavigationGraph geometry (or,
  for #17, of geometry + the SAME already-live-parity-audited `candidate_traversable` state every
  existing feature already reads) — no new information source is required, only new arithmetic over
  information the codebase already has.
- **Only partially resolvable with engineered features, requiring real relational/multi-hop
  computation**: path overlap (gap #11) and downstream bottleneck probability (gap #7). These CAN be
  computed classically (e.g., explicit path-enumeration or a second-pass "does this candidate's
  downstream neighbor currently have queue_length > threshold" join), but doing so well starts to look
  like hand-rolling a shallow, fixed-depth version of exactly what graph message-passing does
  automatically and at arbitrary depth. A 1-hop or 2-hop engineered version is worth trying (Phase 5/6);
  an arbitrary-depth, learned version of this is the actual case for graph-based learning.
  `twin_stair_highrise`'s shorter median congestion lead time (found in Phase 3 point 2 above) is the
  clearest example of an ambiguity this investigation found that is **not** a missing-feature problem at
  all — it is a genuine dynamical/physical difference between vertical and horizontal evacuation that no
  static structural feature, however clever, resolves.

---

## Phase 4 — Literature Mapping

No implementation, no recommendation yet — six families of topology-aware representation, from the
literature this domain draws on (graph theory, network science, transportation/traffic forecasting, and
graph machine learning), each mapped against SynEvac's own architecture and constraints.

### 1. Classical graph-structural features (what SynEvac already has)
**Idea**: hand-computed graph-theoretic quantities (centrality, cut-edges, shortest-path counts) treated
as ordinary tabular features fed to a classical ML model (RF/GB/XGBoost/Logistic Regression — exactly
what the Benchmark Campaign trained). **Advantages**: cheap, deterministic, fully interpretable, trivial
live/sim parity (already proven — every graph-context field in `schema_v4.py` has an identical live
computation path), zero new dependencies (networkx already in use), composes with the existing
`prediction_evaluation`/`ai_registry` infrastructure with no change. **Disadvantages**: each new
descriptor must be hand-designed and individually validated (as the Cross-Topology Investigation did for
exactly 3 of them); cannot automatically discover which multi-hop relationships matter; caps out at
whatever a human can think to compute. **Complexity**: low (this codebase's own working example). **SynEvac
compatibility**: complete — this IS SynEvac's current approach.

### 2. Network robustness / vulnerability analysis (edge & vertex criticality via removal-impact)
**Idea**: rather than a static centrality ranking, measure the ACTUAL impact of removing an edge/vertex
(e.g., re-solve shortest paths or max-flow with it removed, compare total evacuation-time or catchment
change). Standard in infrastructure/network-resilience literature (bridge/power-grid vulnerability
analysis). **Advantages**: directly answers Phase 2's gap #12 (edge criticality) and is a natural
extension of already-proven code (`graph_context_v4.py`'s own shortest-path/bridge machinery, called
once per candidate-removed graph instead of once per building). Still classical, still tabular, still a
per-edge scalar. **Disadvantages**: O(E) times more expensive than the current O(1)-per-building
computation (still cheap at this codebase's single-digit-to-low-tens candidate count, per
`graph_context_v4.py`'s own documented complexity analysis); still static (computed from geometry, not
current dynamic state, unless deliberately re-run against current `candidate_traversable`). **Complexity**:
low-moderate — a direct extension of existing, tested code. **SynEvac compatibility**: high; the most
natural "next 3 features to try" candidate.

### 3. Max-flow / min-cut and queueing-network capacity models
**Idea**: treat the building as a capacitated flow network (Ford-Fulkerson/Edmonds-Karp max-flow, or a
capacitated queueing network) and derive features like "maximum sustainable throughput from this zone to
OUTSIDE" or "this edge's min-cut membership" — directly answers Phase 2's gap #1 (downstream capacity)
properly, as a mathematically well-founded quantity rather than an ad hoc ratio. **Advantages**: rigorous,
well-understood, computable with `networkx.maximum_flow`/`minimum_cut` (already a dependency).
**Disadvantages**: requires deciding what "capacity" means as a flow-network edge weight (this codebase's
`candidate_capacity` is a persons/interval RATE, not a network-flow capacity in the classic static sense
— translating rate-capacity into flow-capacity needs a modeling decision, e.g. a time-horizon
assumption). **Complexity**: moderate — new modeling decisions, not just new arithmetic, but still
classical and still tabular-output-compatible. **SynEvac compatibility**: high in principle, moderate in
practice (the capacity-semantics translation is the real work, not the algorithm).

### 4. Graph kernels / spectral graph features
**Idea**: whole-graph or whole-subgraph descriptors derived from spectral graph theory (Laplacian
eigenvalues/eigenvectors — algebraic connectivity, spectral gap) or graph kernels (Weisfeiler-Lehman
subtree kernel, shortest-path kernel) that summarize a WHOLE building's shape as a fixed-length vector,
usable as global-context features (analogous to Pipeline A's existing single `total_active_occupant_
count` global field) or for whole-graph similarity/clustering (e.g., "this unseen building is spectrally
similar to family X"). **Advantages**: mathematically well-founded, captures GLOBAL shape properties
(like graph diameter, gap #8) that no per-edge feature can; graph diameter/algebraic connectivity are
cheap (`networkx.diameter`/`algebraic_connectivity`). **Disadvantages**: full graph-kernel similarity
computation is more machinery than this codebase's current scale plausibly needs (SynEvac's buildings are
single-digit-to-low-tens of candidates, not large graphs where kernel methods earn their keep);
eigenvector-based features are less directly interpretable than a betweenness/bridge/catchment number.
**Complexity**: low for simple spectral scalars (diameter, algebraic connectivity — a few lines with
networkx), moderate-high for full kernel methods. **SynEvac compatibility**: high for the simple scalars
(direct, natural additions to close gap #8/#9), low-value for full kernel methods at this building scale.

### 5. Node/edge structural embeddings (node2vec, DeepWalk, struc2vec)
**Idea**: unsupervised random-walk-based methods that learn a fixed-length vector per node/edge encoding
its structural role, without hand-designing which structural property matters — feed the learned
embedding into the same classical ML models as an additional feature block. **Advantages**: can capture
structural regularities a human didn't think to engineer; `struc2vec` specifically targets
STRUCTURAL-ROLE similarity (not just proximity), which is the closest fit to "is this candidate
topologically similar to a bottleneck-prone candidate in a DIFFERENT building" — directly relevant to
cross-topology generalization. **Disadvantages**: a new training step per building (or per building
family) — embeddings for an unseen building's candidates aren't defined until the embedding model has
seen that building's graph, an alignment/cold-start problem for genuinely NEW buildings unless embeddings
are trained jointly across all training buildings (an actual research question, not a known-solved one
for this codebase's scale); introduces a new dependency (`node2vec`/`gensim` or similar — not currently
installed) and a new artifact type (a trained embedding model) with its own versioning/registry story on
top of the existing `ai_registry.ModelRegistry`. **Complexity**: moderate-high — new dependency, new
training pipeline, new artifact lifecycle. **SynEvac compatibility**: moderate — technically graftable
onto the existing tabular-feature pipeline (embeddings are still just numbers per row), but the cold-start
problem for unseen buildings is exactly the property this investigation cares about, and is unresolved by
this family without further design work.

### 6. Graph Neural Networks (GCN, GraphSAGE, GAT — message-passing architectures)
**Idea**: replace the flat per-candidate feature vector with a graph-structured input (nodes = zones,
edges = candidates) and let a learned message-passing model directly compute Phase 2's gap #7 (downstream
bottleneck probability) and gap #11 (path overlap) as LEARNED, arbitrary-depth relational functions,
rather than hand-engineered fixed-depth approximations. **Advantages**: directly targets exactly the
organizing gap Phase 2/3 found (aggregation across multiple candidates along a path) without requiring a
human to correctly guess the right hand-engineered proxy; GraphSAGE/GAT specifically support inductive
generalization to unseen graph structures (unlike a plain GCN, which is transductive by default) — the
single most relevant property for cross-building generalization specifically. **Disadvantages**: a
genuinely new model class requiring a new training/serialization/inference stack incompatible with
`ai_training.models.base.BaseModel`'s current `List[Dict[str, Any]]`-row contract (would need graph-batch
input, not flat rows); new dependency (PyTorch Geometric or DGL — `torch` IS already installed per this
session's own dependency check, but neither graph-learning library is); loses the current architecture's
complete interpretability (native/permutation feature importance, calibration analysis, everything Phase
4-6 of the Benchmark Campaign relied on) without additional, non-trivial GNN-explainability work; requires
materially more training data per unique graph shape than the classical models this codebase has ever
needed. **Complexity**: high — new dependency, new model class, new data pipeline, new evaluation story.
**SynEvac compatibility**: low TODAY (would require redesigning `ai_training.models.base.BaseModel`'s own
row-based contract, `ai_registry.metadata.ModelMetadata`'s `ordered_feature_names` assumption, and
`prediction_evaluation`'s per-row pairing logic — none of which this milestone is authorized to touch),
but philosophically the best-targeted family for the SPECIFIC gap this investigation found, if a future
milestone is explicitly chartered to redesign that much.

### 7. Spatio-temporal graph learning for traffic/crowd forecasting (STGCN, DCRNN) — the closest external analogy
**Idea**: traffic-forecasting literature (predicting congestion on a road network) combines graph
structure (family 6) with temporal sequence modeling (predicting how congestion PROPAGATES over time
across connected edges) — directly analogous to SynEvac's own `candidate_congestion_trend`/`candidate_
recent_flow_rate` dynamics, but modeled jointly across candidates instead of independently per row.
**Advantages**: the single closest published analog to "predict evacuation congestion onset a few
seconds ahead," with a mature literature specifically about generalizing across different road-network
topologies (the traffic-forecasting equivalent of SynEvac's own cross-building problem). **Disadvantages**:
inherits every disadvantage of family 6 (GNN complexity) PLUS a temporal-sequence modeling layer on top —
the most complex family surveyed. **Complexity**: highest of all families surveyed. **SynEvac
compatibility**: lowest near-term (furthest from the current architecture), but the most directly relevant
prior art if a future milestone does pursue graph-based learning — should be the literature entry point
for that future milestone, not something to prototype now.

---

## Phase 5 — Gap Analysis

**Has SynEvac's current representation reached the practical limit of classical engineered features?**

**No — not uniformly.** The evidence supports a split verdict, not a single yes/no:

- **Ratio/normalization-style engineered features have been tried twice and both times found unreliable**
  (V3.1 on Dataset V2: "no clear, robust benefit"; Cross-Topology Investigation on Dataset V3: +0.6%
  average, actively harmful on 2 of 4 families). This SPECIFIC sub-family of classical feature
  engineering — rescaling existing raw values by static per-building constants — looks genuinely close to
  its ceiling. Phase 3's own analysis explains why: rescaling cannot fix CONCEPT shift (a relationship
  that differs by family), only covariate shift (a distribution that differs by family) — and the two
  features that matter most for outcome quality (`total_active_occupant_count`, `candidate_recent_flow_
  rate`) carry real concept shift, not just scale differences.

- **Graph-STRUCTURAL features have only been tried once, as one 3-feature bundle, and were the single
  best-evidenced positive result found so far** (+3.3% average, non-negative in every family — the
  cleanest signal in the entire Cross-Topology Investigation). Only 3 of Phase 2's 17 identified candidate
  descriptors have actually been tested. This sub-family has clear, specific, well-justified room to grow
  before any conclusion that classical engineering is exhausted: **gaps #1 (downstream capacity), #5
  (shortest-path diversity), #12 (edge criticality via removal-impact), #14 (capacity-weighted
  betweenness), and #16 (vertical floor-distance) are all classical, cheap, live-parity-safe extensions of
  code that already exists and already works** (`graph_context_v4.py`'s own shortest-path/bridge
  machinery, straightforwardly extended).

- **Two specific ambiguities found in Phase 3 (path overlap / downstream bottleneck probability, and
  `twin_stair_highrise`'s dynamical lead-time asymmetry) are where a genuine ceiling starts to appear** —
  not because no classical feature COULD approximate them (a fixed 1-2-hop engineered version is worth
  trying), but because their FULL, arbitrary-depth, dynamically-conditioned form is structurally what
  graph message-passing computes and hand-engineering does not scale to.

**If no** (the answer here): **exactly what should be added, and why** — the 5 gaps named above,
in priority order by expected impact/effort ratio:

| Priority | Feature | Expected impact | Effort |
|---|---|---|---|
| 1 | Shortest-path diversity (fix gap #5 — count ALL tied-shortest paths per zone, not just one) | Moderate — directly fixes a disclosed undercounting bug in an already-validated (+3.3%) feature family, likely improves `upstream_catchment_count`'s own signal quality further | Low — same Dijkstra call already made, sum over all shortest paths instead of one |
| 2 | Capacity-weighted betweenness (gap #14) | Moderate — directly targets §1's `total_active_occupant_count` concept-shift finding by making the existing centrality feature capacity-aware | Low — same Brandes' call, different edge weight |
| 3 | Downstream/reachable capacity ratio (gap #1/#17), specifically `occupant_count / building_total_exit_capacity` | High — a novel, specific, well-motivated ratio never previously tried (unlike the 5 already-tried-and-failed ratios), directly targeting the single largest concept-shift finding (35× spread) | Low-moderate — needs one new building-level aggregate (sum of exit capacities), otherwise reuses existing infrastructure |
| 4 | Vertical floor-distance (gap #16) | Moderate — directly targets the Benchmark Campaign's own found weak spot (upper-floor scenarios) and `twin_stair_highrise`'s specific failure | Low — `Staircase.vertical_height()` / floor-graph distance already exists from the Stair Simulation Reliability milestone |
| 5 | Edge criticality via removal-impact (gap #12) | Uncertain but plausible, given betweenness's own success — the natural "graded" successor to the binary `is_bridge` | Moderate — O(E) re-solves instead of O(1), still cheap at this codebase's scale, more code than 1-4 |

Items 6-7 in Phase 2's list (path overlap, downstream bottleneck probability) are the two the evidence
says to defer, not because they're unimportant (Phase 3 identifies them as the most directly analogous to
`twin_stair_highrise`'s failure) but because a well-scoped classical attempt (a fixed-depth engineered
proxy) should be tried and evaluated BEFORE concluding graph learning is needed for them specifically —
consistent with this investigation's own finding that ML has never yet failed to beat the deterministic
baseline in any holdout tested (2.3×-3.6× lift in the worst case), so there is no evidence of a hard wall,
only of diminishing and uneven returns from the SPECIFIC feature family (ratios) tried so far.

---

## Phase 6 — Recommendation Report

### Current feature strengths
- Zero identity leakage (Benchmark Campaign's own finding, confirmed structurally by this investigation's
  full field audit — Phase 1's tables contain no zone/stair/exit/door ID anywhere).
- `candidate_queue_length` is empirically the single most cross-topology-trustworthy feature measured
  (near-zero covariate AND concept shift) — a genuine, validated invariant worth protecting/prioritizing
  in any future feature-selection decision.
- The 3 existing graph-context features are the only representation change tested so far that is
  non-negative in every holdout family (+3.3% average) — full live/sim parity already proven
  (`graph_context_v4.py`'s own architecture, reused unmodified by both simulation and — if ever wired —
  live extraction).
- Every feature in both schemas (Pipeline A and B) has an audited, honest live-availability
  classification (`FeatureAvailability` enum) — no feature exists in the schema without a real answer to
  "could a genuine deployment produce this."

### Current feature weaknesses
- Two features (`total_active_occupant_count`, `candidate_recent_flow_rate`) carry genuine, quantified
  CONCEPT shift (35× and 33× conditional-rate spread) that no amount of rescaling has fixed in two
  independent attempts.
- Every graph-context feature computed so far is a STATIC, per-single-edge scalar — none aggregates or
  propagates information across multiple candidates along a path (Phase 2's organizing finding).
- `upstream_catchment_count` specifically undercounts structural dependency in any building with tied
  shortest paths (a disclosed simplification in the feature's own source code, not previously flagged as
  a generalization risk until this investigation).
- Pipeline B's schema has zero structural descriptors of any kind — untested for cross-building
  generalization in either direction (neither proven poor nor proven adequate).

### Missing structural descriptors (ranked, from Phase 5)
Shortest-path diversity fix → capacity-weighted betweenness → downstream/reachable-capacity ratio →
vertical floor-distance → removal-impact edge criticality. Full table with impact/effort estimates in
Phase 5 above.

### Compatibility with existing infrastructure
- **Prediction Evaluation Framework** (`prediction_evaluation/`): unaffected either way — it operates on
  already-extracted feature rows and model outputs, agnostic to what the feature vector contains.
- **Dataset Generator** (`predictive_dataset/`, `ai_registry/`): all 5 proposed features are pure
  functions of static Building/NavigationGraph geometry (or, for the capacity ratio, one new
  building-level aggregate) — exactly the same computational pattern `graph_context_v4.py` already
  established and proved safe, requiring no change to `campaign_runner_v4.py`'s own execution loop beyond
  adding new columns, the same additive pattern `schema_v4.py` itself used to extend `schema.py`.
- **Shadow Mode** (`live_system.live_ai_gateway`, `ai_registry.registry`): unaffected by anything in this
  investigation — no schema was changed, no model retrained, no registration touched. If a future
  milestone DID promote any of these 5 features, the existing `ai_registry.registry.validate_model_
  compatibility()`'s ordered-feature-name check (proven in the Benchmark Campaign's own negative-control
  test) would correctly require a coordinated schema-version bump, exactly as designed.
- **Backward compatibility**: every proposed feature is additive (new columns), following the exact
  `schema.py` → `schema_v4.py` precedent — Dataset V1/V2/V3 and any model trained against them remain
  fully valid and reproducible.

### Recommendation

**Continue with engineered classical features as the immediate next step — do not introduce richer graph
features as a wholesale representation change, and do not consider graph-based learning for the
IMMEDIATE next milestone.** Reasoning, directly from the evidence assembled above:

1. Only 3 of at least 17 identified candidate structural descriptors have ever been tried, and that one
   attempt was the single best-evidenced positive result in this codebase's entire generalization research
   history (+3.3%, non-negative everywhere). Concluding classical features are exhausted after testing
   ~18% of the identified candidate space would not be evidence-supported.
2. The 5 specific features prioritized in Phase 5 are cheap (low-moderate effort, reusing proven
   infrastructure), fully backward-compatible, and each targets a SPECIFIC, quantified finding from this
   investigation (not a generic "more features" hope) — a well-justified, low-risk next experiment.
3. Graph-based learning (Phase 4, families 5-7) is the philosophically correct tool for exactly ONE
   identified gap (path overlap / downstream bottleneck probability, the fully-relational, arbitrary-depth
   case) — but that gap has not yet been attempted with a cheaper, fixed-depth classical proxy, and every
   graph-learning family surveyed requires architectural changes (`BaseModel`'s row contract, `ai_registry.
   metadata`'s feature-name assumptions, `prediction_evaluation`'s pairing logic) this milestone is
   explicitly not chartered to make.
4. ML has never once failed to beat the deterministic baseline in any holdout tested across this
   codebase's entire generalization research history (2.3×-3.6× lift in the worst measured case) — there
   is no evidence classical features have hit a hard wall, only evidence that ONE sub-family (ratios) has
   plateaued while another (graph-structural) is under-explored.

**Proposed roadmap for the next research phase** (candidate milestone, not a commitment — mirrors the
Cross-Topology Investigation's own "candidates only" discipline):
1. Implement and evaluate the 5 prioritized features from Phase 5 (shortest-path-diversity fix,
   capacity-weighted betweenness, downstream/reachable-capacity ratio, vertical floor-distance,
   removal-impact criticality) against the SAME Dataset V3 family-holdout protocol the Cross-Topology
   Investigation already established — a controlled, apples-to-apples extension of existing, validated
   methodology.
2. If the downstream-capacity-ratio feature (priority 3, the single highest-expected-impact item, directly
   targeting the 35×-spread finding) shows a result comparable to or better than the existing graph-context
   bundle, that is strong evidence classical engineering has more room to run and graph learning should
   remain deferred.
3. If, after that round, `path overlap`/`downstream bottleneck probability` remain unresolved even by a
   deliberately-tried fixed-depth engineered proxy (e.g., a 1-hop "does my immediate downstream neighbor
   currently have queue_length > threshold" join feature), THAT specific, narrow, evidenced failure — not
   a general "classical features aren't working" impression — would be the concrete, well-justified trigger
   to charter a future milestone investigating graph-based learning (Phase 4's families 5-7), starting from
   the traffic-forecasting literature (family 7) as the closest prior art, and beginning with an explicit
   architectural-compatibility design pass (since every family surveyed requires changes to `BaseModel`,
   `ai_registry.metadata`, and `prediction_evaluation` this investigation was not chartered to make).
4. Separately, and lower priority: decide whether Pipeline B's schema should ever grow a per-location
   grain (making Phase 2-5's analysis apply to it directly), or whether Pipeline B's whole-building-only
   design is a deliberate, permanent scope boundary — currently undecided, flagged here rather than
   assumed either way.
