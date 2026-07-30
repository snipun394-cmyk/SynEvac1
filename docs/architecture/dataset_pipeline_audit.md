# Synthetic Dataset Generation Pipeline Audit

Status: audit complete, zero redesign. This document reports findings only — the dataset generator,
schema, and target definition are unchanged by this milestone. Where a genuine gap was found, it is
disclosed as a limitation/recommendation, not silently patched.

---

## Phase 1 — Pipeline architecture

SynEvac1 has **two independent synthetic-dataset pipelines** that share only the low-level simulation
substrate (`scenario_generator`, `scenario_runner`, `simulation_runtime`, `ai_decision`) and never
cross-import each other:

```
                         ┌───────────────────────────────────────────┐
                         │   Shared low-level simulation substrate    │
                         │ scenario_generator / scenario_runner /      │
                         │ simulation_runtime / ai_decision            │
                         └───────────────┬─────────────────┬──────────┘
                                          │                 │
                    ┌─────────────────────▼───┐   ┌─────────▼──────────────────┐
                    │   PIPELINE A             │   │   PIPELINE B                │
                    │ predictive_dataset/      │   │ ai_registry / designer.campaign │
                    │ campaign_runner_v4.py    │   │ .campaign_worker.CampaignWorker │
                    │                          │   │                              │
                    │ topologies_v4 (24 vars,  │   │ training_scenario.py (1 fixed│
                    │  6 families)             │   │  2-floor building)           │
                    │ → candidate.py           │   │ → dataset_builder / ground_  │
                    │ → simulation_extractor_v4│   │   truth / decision_policy    │
                    │   (15 fields, per-edge,  │   │   exports (Campaign Studio)  │
                    │   per-tick)              │   │ → training_dataset (loader/  │
                    │   uses crowd_intelligence│   │   exporter)                  │
                    │ → target_generator_v2    │   │ → ai_training.load_campaign_ │
                    │   (frozen since Phase 10)│   │   dataset()                  │
                    │ → streamed CSV writer    │   │ → ai_registry.campaign.build_│
                    │   candidate_dataset_v4   │   │   live_compatible_dataset()  │
                    │   .csv + scenario_       │   │   → ai_features.extract_     │
                    │   metadata.json          │   │     canonical_training_row() │
                    │                          │   │     (25-field CANONICAL_LIVE_│
                    │ schema_v4.py (v4.0,      │   │     SCHEMA)                  │
                    │  15 fields)              │   │ → ai_training.models.*.      │
                    │                          │   │   build_table() → sklearn    │
                    │ consumed by: predictive_ │   │ → ai_registry.metadata.      │
                    │ model/ (research-only    │   │   save_live_model()          │
                    │ pandas+XGBoost scripts,  │   │ → ai_registry.registry.      │
                    │ no registry, no live path)│   │   ModelRegistry (schema-     │
                    │                          │   │   version-checked at load)   │
                    │                          │   │   → live_system.live_ai_     │
                    │                          │   │     gateway (Shadow-Mode)    │
                    └──────────────────────────┘   └────────────────────────────┘
                       No shared code above the                No shared code
                       simulation substrate line                (except reused
                                                                  AIFeatureField/
                                                                  FeatureAvailability
                                                                  dataclass types)
```

**Pipeline A** (`predictive_dataset/`) is the large-scale, SCENARIO × TIMESTEP × CANDIDATE research
dataset used throughout the Predictive Dataset Campaign V1-V4 milestones — one row per Door/Exit/Stair
edge per simulation tick. Row grain and schema are versioned by module (`schema.py` v1.0 frozen,
`schema_v4.py` v4.0 current), never mutated in place. Consumed only by ad hoc, research-only
`predictive_model/` scripts (`feature_prep_v2_1.py`, `tree_models.py`) — never registered, never wired
to live inference.

**Pipeline B** (`ai_registry/` + `designer.campaign.campaign_worker`) is the SCENARIO-grain pipeline that
actually produces registry-tracked, potentially live-deployable models — the one with a real Model
Registry, Shadow-Mode inference gateway, and a runtime schema-compatibility check
(`ai_registry.registry.validate_model_compatibility`). Uses one fixed 2-floor building
(`ai_registry.training_scenario.make_training_building()`), not Pipeline A's 24-variant family set.

**Ownership / file formats**: Pipeline A writes plain CSV (streamed row-by-row via `csv.DictWriter`,
never buffered in memory) plus a JSON `scenario_metadata.json` (fire/door/exit/stair/occupant summary,
one entry per accepted scenario) and a JSON campaign report. Pipeline B's `CampaignWorker` produces a
richer Campaign-Studio-native artifact set (`scenario_features.csv` + per-scenario zone_results/timeline/
ground_truth/decision_policy files), read back only through `training_dataset.load_campaign()` /
`ai_training.load_campaign_dataset()` — never through raw `pandas.read_csv`.

**Verbatim CLI invocation** (Pipeline A, the one relevant to "thousands of scenarios"):
```
python scripts/run_predictive_dataset_campaign_v4_pilot.py       # 600 scenarios
python scripts/run_predictive_dataset_campaign_v4_fullscale.py   # 3,000 scenarios
```
Both are real, existing, unmodified entry points — no CLI flags; scenario counts and master seed are
fixed in `predictive_dataset/campaign_config_v4.py`. Pipeline B has no standalone CLI script;
`generate_training_campaign()` instantiates a `PyQt6.QtWidgets.QApplication` internally and is only ever
called programmatically (see existing test fixtures).

**Datasets are never committed to git.** `/data/predictive_dataset_campaign_v4/` (and every prior
version's directory) is `.gitignore`d — every campaign is described in the repo as "regenerable from
source, never committed," with its own doc citing the exact numbers produced. This audit confirms that
discipline is followed consistently across all campaign versions on disk.

---

## Phase 2 — Stress test

A dedicated stress-test harness (`scripts` not committed to the repo — a scratch script driving the
real, unmodified `predictive_dataset.campaign_runner_v4.run_campaign_v4()`, the exact function the real
pilot/fullscale scripts call) generated datasets at 10, 100, 500, 1000, 2500, 5000, and 10000 TOTAL
scenarios, spread proportionally across all 24 current structural variants (mirroring how the real
pilot/fullscale scripts already scale `PILOT_SCENARIOS_PER_VARIANT`/`FULLSCALE_SCENARIOS_PER_VARIANT`),
each run in a fresh process on the current codebase (post all Stair-work commits through `8a51ca6`).

| Requested | Accepted | Failed | Rows | Wall (s) | Rows/sec | Scenarios/sec | Disk (MB) | Process RSS Δ (MB) |
|---|---|---|---|---|---|---|---|---|
| 10 | 24 | 0 | 35,841 | 3.5 | 10,194 | 6.82 | 6.3 | 5.1 |
| 100 | 96 | 0 | 126,702 | 13.6 | 9,296 | 7.04 | 22.3 | 6.5 |
| 500 | 504 | 0 | 642,657 | 69.7 | 9,228 | 7.23 | 113.0 | 10.1 |
| 1000 | 1,008 | 0 | 1,341,385 | 145.6 | 9,212 | 6.92 | 236.0 | 9.4 |
| 2500 | 2,496 | 0 | 3,281,496 | 355.5 | 9,232 | 7.02 | 577.7 | 12.5 |
| 5000 | 4,992 | 0 | 6,586,414 | 756.3 | 8,711 | 6.60 | 1,160.1 | 14.1 |
| 10000 | 10,008 | 0 | 13,085,902 | 1,533.3 | 8,536 | 6.53 | 2,304.3 | 17.9 |

(Accepted scenario counts exceed the "requested" total slightly because the per-variant scenario count
is `round(total / 24)`, then multiplied back out across 24 variants — the same rounding behavior the real
pilot/fullscale scripts already exhibit, e.g. the real fullscale run requests 125×24=3,000 exactly only
because 3,000/24 divides evenly.)

**Zero crashes, zero deadlocks, zero corrupted files, zero duplicate scenario IDs, zero missing output,
zero partial writes** across all seven runs — `failed_scenarios` was 0 in every run, every CSV's own
row count matched an independent re-read of the file, every `scenario_metadata.json` scenario count
exactly matched `accepted_scenarios`, and `duplicate_row_keys` (on `(scenario_id, observation_time,
candidate_id)`) was 0 in every run.

**Memory**: process RSS delta (our own generating process's memory footprint, before → after) stayed
flat and small (5-14MB) across a 500x increase in scenario count (10 → 5000) — strong evidence against a
memory leak, since a real leak would show growth roughly proportional to scenario count, not a
near-constant few-megabyte footprint. This is a direct, structural consequence of `run_campaign_v4`'s
own streaming-CSV-writer design (never buffers more than one row in memory) plus its own built-in
`psutil`-based abort guard (`MIN_AVAILABLE_MEMORY_BYTES = 300_000_000` — never triggered in any run here).
System-wide "available memory" fluctuated up and down between runs (other processes on this machine,
~7.9GB total RAM, competing for the same pool) — that figure is noisy and was NOT used as the leak
signal; only this process's own RSS was.

**CPU**: `cpu_utilization_fraction` was ~0.99 in every run — generation is single-threaded and entirely
CPU-bound (one scenario simulated at a time, no parallelism across scenarios or variants). On this
6-physical/12-logical-core machine, 11 logical cores sit idle throughout a campaign — a real, disclosed
opportunity (not a defect, and explicitly not pursued here per "do NOT redesign the generator unless a
genuine defect is discovered") for a future milestone that wants materially faster large-campaign
turnaround via multiprocessing across variants.

---

## Phase 3 — Dataset integrity

A full pandas-based audit of the 2500-scenario run (3,281,496 rows, the largest dataset already
regenerated at the time of this check) found:

- **Zero fully-duplicate rows, zero duplicate `(scenario_id, observation_time, candidate_id)` keys.**
- **Zero negative values** in any of observation_time/occupant-count/capacity/distance/queue/
  approaching/flow/catchment columns.
- **Two columns have real, non-corruption nulls**, both by documented design:
  - `candidate_adjacent_zone_occupancy` (63% null) — `None` exactly when `edge.from_node ==
    Node.OUTSIDE_NODE_ID` (`predictive_dataset/simulation_extractor.py:101-106`), i.e. there is no
    "from zone" to report an occupancy for. Not missing data — a structurally undefined quantity for
    that candidate type.
  - `target_v2` (30% null/`None`) — `None` means "this candidate is CURRENTLY congested at the
    observation instant itself" (excluded from the True/False onset-prediction label by Target V2's own
    documented semantics, not a data-quality defect). `lead_time_seconds_v2` is null in exactly the rows
    where `target_v2 != True` (0 violations found) — the label triple's own internal consistency holds.
- **All 24 CSV columns present in every row, exact header match** against `CSV_COLUMNS`
  (`predictive_dataset/campaign_runner_v4.py`) in every stress run.
- **197 of 2,496 scenarios (7.9%) have no `evacuation_duration`** in `scenario_metadata.json` — `None`
  exactly when `movement_result.arrival_times` is empty (`simulator/coordinator.py:496`), i.e. a
  scenario where no occupant ever reached an exit before the simulation ended (expected for the
  deliberately-included `total_lockout`/single-exit-blocked topology families the campaign's own
  `COVERAGE_TARGETS_V4["total_lockout_scenarios_with_rows"]` exists specifically to guarantee at least 3
  of). Not a bug — a documented scenario family working as designed.
- **Hazard information lives at the scenario level, not per-row.** The per-candidate-per-tick schema
  (`schema_v4.py`) has no hazard/fire column; `ignition_zone_id`, `fire_growth_time_seconds`, and
  `fire_profile` are recorded once per scenario in `scenario_metadata.json` (populated for all 2,496
  scenarios in this check — every scenario has a fire). Occupants, routes (implicitly, via candidate
  reachability/`candidate_traversable`), timestamps, targets, and feature vectors are all present in
  every row; "hazard" and "labels" (the whole-scenario summary fields) are correctly scenario-scoped
  rather than row-scoped — this is a schema design choice, not missing data.

---

## Phase 4 — Stair-work dataset impact

**Finding: the recent Stair work (Stair Flow Intelligence `bf86bb4`, Stair Predictive-Feature Live
Parity `faf53e4`, Stair Simulation Reliability Audit `02b958b`) changed nothing in Pipeline A's
generated dataset — confirmed both by source-diff inspection and by an empirical before/after
regeneration comparison.**

**Source-diff evidence:**
- `bf86bb4` (Stair Flow Intelligence): **zero files under `predictive_dataset/` touched** — its own
  commit message explicitly notes "`predictive_dataset`/`predictive_model` untouched."
- `02b958b` (Stair Simulation Reliability): **zero files under `predictive_dataset/` touched.** It fixed
  `navigation/graph_builder.py` and `simulator/coordinator.py` (the shared substrate both pipelines sit
  on) — but its own commit message states the fix's own audit already found **"all 24 currently-active
  predictive topologies (zero degenerate stairs found)"** before this milestone even began.
  `stair_flow` (the new package from `bf86bb4`) is imported only by `predictive_dataset/
  live_extractor_v2_1.py` / `live_extractor_v4.py` — the LIVE feature-extraction path — **never** by
  `simulation_extractor*.py`, the module Pipeline A's campaign runner actually calls.
- `faf53e4` (Live Parity): touched `predictive_dataset/live_extractor_v2_1.py`,
  `live_extractor_v4.py` (again, LIVE-only, never invoked by `campaign_runner_v3.py`/`v4.py`) and
  `predictive_dataset/schema_v4.py` — but the `schema_v4.py` diff is a **docstring/metadata-string
  change only** (the `source=`/`missing_value_note=` text of one `AIFeatureField`, documenting the LIVE
  side's new capability), explicitly preserving `"SIM: never None -- exact ground truth is always
  computable"` unchanged. No field name, type, ordering, or simulation-side value computation changed.

**Empirical confirmation** (this audit, not reused from a prior milestone): the on-disk, committed-era
`data/predictive_dataset_campaign_v4/candidate_dataset_v4.csv` was regenerated on 2026-07-29 00:30 —
genuinely **before** `02b958b` (2026-07-30 12:35), `bf86bb4` (2026-07-30 11:36), and `faf53e4`
(2026-07-30 12:01) — confirmed by file mtime vs. commit timestamps. Filtering both that pre-fix dataset
and this audit's freshly-regenerated (current-HEAD, post-fix) 2500-scenario Stair rows to compare:

| Metric | OLD (pre-fix, 226,022 Stair rows) | NEW (post-fix, 185,566 Stair rows) |
|---|---|---|
| Stair rows with `candidate_walking_distance == 0.0` | 0 | 0 |
| Zero-distance flagged Stair candidate_ids (`quality_checks.zero_walking_distance_candidates`) | none | none |
| `candidate_traversable` | 100% True | 100% True |
| Stair `target_v2` positive rate (True / (True+False)) | 1.266% | 1.279% |
| Stair share of all rows | 5.73% | 5.65% |

The two are statistically indistinguishable (the small differences are consistent with sampling noise
from different scenario counts, not a systematic shift). **No new fields, no removed fields, no changed
target definition, no changed distributions, no changed labels attributable to the Stair work** — the
`02b958b` fix is a genuine correctness guard against a defect class the current 24-variant topology set
never actually triggers, not a change that alters today's dataset. A future topology variant that DOES
place a Stair with an unresolved `from_floor_id` would now correctly get `walking_distance=None` +
a recorded validation issue instead of a silent, degenerate `0.0` — the guard's value is prospective, not
retroactive to the existing variant set.

**No ML target changed.** Target V2 (`target_generator_v2.py`, frozen since Phase 10 of the V3 campaign
milestone) is untouched, unimported by, and unaffected by every Stair-work commit examined.

---

## Phase 5 — Feature schema audit

Pipeline A's schema (`predictive_dataset/schema_v4.py`, `SCHEMA_VERSION_V4 = "4.0"`): 15 named
`CANDIDATE_FEATURE_NAMES_V4` fields, deterministic order (part of the contract — every extractor emits
keys in this exact order), each carrying an explicit dtype (int/float/bool/str), `nullable` flag, and
`missing_value_note` where nullable. Categorical fields (`candidate_type`, `candidate_congestion_level`,
`candidate_congestion_trend`) are emitted as their raw string category in the CSV — one-hot encoding is
performed downstream, by the consuming `predictive_model.feature_prep_v2_1.build_experimental_feature_
matrix()`, not baked into the dataset itself. "UNKNOWN"/missing handling: numeric nullable fields get a
companion `*_missing` indicator column at feature-prep time; categorical fields get a `*=__missing__`
indicator category — both handled uniformly, not per-field special-cased.

Pipeline B's schema (`ai_features/feature_schema.py`, `CANONICAL_LIVE_SCHEMA`, `SCHEMA_VERSION = "1.0"`):
25 whole-building aggregate fields, classified into a 6-value `FeatureAvailability` enum
(`LIVE_OBSERVABLE`, `LIVE_ESTIMABLE`, `SIMULATION_ONLY`, `FUTURE_INFORMATION`, `OUTCOME_LEAKAGE`,
`UNKNOWN`) — deliberately excludes hazard/fire/ignition/behaviour-profile fields as `SIMULATION_ONLY`
(no real-sensor path exists today) and any evacuation-outcome field as `OUTCOME_LEAKAGE`.

**These are two distinct schemas in two distinct version spaces** — they share only the reusable
`AIFeatureField`/`FeatureAvailability` container dataclass/enum types (`schema_v4.py` imports these
types from `ai_features.feature_schema`, never any field list). Zero field-name overlap between the two.
**A real naming-collision risk exists but causes no actual bug today**:
`predictive_dataset.schema.SCHEMA_VERSION` (the old, frozen V1 candidate schema) and
`ai_features.feature_schema.SCHEMA_VERSION` are coincidentally both the literal string `"1.0"`, in two
completely unrelated version spaces — nothing in the current codebase cross-compares them, but a
future integration that did a naive string-equality schema check across packages could be misled.
Worth a namespaced rename (e.g. `CANDIDATE_SCHEMA_VERSION` vs. `CANONICAL_LIVE_SCHEMA_VERSION`) the next
time either module is touched for an unrelated reason — not urgent enough to justify a standalone change
today.

**Do existing trained models still expect this schema?** For Pipeline B: yes, and it is actively,
mechanically enforced — `ai_registry.registry.ModelRegistry.validate_model_compatibility()`
(`ai_registry/registry.py:170-231`) re-derives `check_model_compatibility()` against the CURRENTLY
running `CANONICAL_LIVE_SCHEMA` on every call (never trusts a model's self-declared compatibility alone),
rejects any `RESEARCH_ONLY`-declared model outright, and additionally re-verifies a model's persisted
`ordered_feature_names` still exactly matches `CANONICAL_LIVE_FEATURE_NAMES`'s current order even when
both claim the same `feature_schema_version` — catching a silent reorder-without-version-bump a plain
column-set check would miss. `get_latest_compatible_model()` is documented as "the one method a future
live inference path should ever call" and never falls back to an incompatible model. Pipeline A has
**no equivalent runtime compatibility checker** — nothing in `predictive_dataset/` cross-validates a
trained research model's expected schema against `schema_v4.py` at load time; this is consistent with
Pipeline A never producing a registry-tracked, live-deployable model in the first place (see Phase 6).

---

## Phase 6 — Training pipeline compatibility

**Pipeline A** (research-only): this audit generated a fresh 500-scenario V4 dataset (post all Stair
fixes, current HEAD) and fed it — with **zero code modification** — through the existing, unmodified
`predictive_model.feature_prep_v2_1.build_experimental_feature_matrix()` → `predictive_model.tree_models.
build_tree_models()` (XGBoost) → `predictive_model.metrics.compute_metrics()` chain, the exact functions
`scripts/model_v4_generalization_evaluation.py` already uses for real research. Result: 441,617 trainable
rows extracted from 642,657 total, a 27-column feature matrix built, an XGBoost model fit and evaluated
in 7.9s (ROC-AUC 0.976, PR-AUC 0.639 on a 70/30 split — numbers in the range prior V4 milestones already
reported, not a new finding, just confirming nothing regressed). **The newest datasets train without
modification.** One caveat: every Pipeline A research script (this one included) hand-rolls its own
`target_v2 → target` column rename before calling into `predictive_model` — there is no shared, versioned
dataset-loading function for Pipeline A (unlike Pipeline B's `ai_training.load_campaign_dataset()`), so a
future schema version that renamed a column would silently break every consuming script independently,
with no single place to fix it. Documented as a limitation (Phase 8), not fixed here (a shared loader
would be a genuine redesign, out of this audit's scope without a demonstrated defect beyond this
disclosed fragility).

**Pipeline B**: already proven end-to-end, repeatedly, by this session's own existing test suite
(`tests/test_live_ai_runtime_integration.py`, `tests/test_prediction_evaluation_e2e.py`) — both generate
a real training campaign, call `ai_training.load_campaign_dataset()` → `ai_registry.campaign.
build_live_compatible_dataset()` → `ai_registry.training.train_bottleneck_occurrence_model()`, and both
pass under current HEAD (confirmed again in this milestone's own Phase 12 full-suite run). No dataset-
loading or training code was touched by any Stair-work commit, and none needed to be.

---

## Phase 7 — Inference pipeline compatibility

Model Registry (`ai_registry.registry.ModelRegistry`), Shadow AI (`live_system.live_ai_gateway.
RegistryLiveAIInferenceGateway`), Feature Builder (`ai_features.building_state_extractor`/
`simulation_extractor`), and Inference Gateway are all exercised together by the same existing E2E tests
cited in Phase 6 — `RegistryLiveAIInferenceGateway(service, include_evacuation_time=False).predict(state,
t)` is called repeatedly and produces valid `LiveAIPredictionSnapshot` objects consumed successfully by
this session's own `prediction_evaluation` framework. Models trained on the latest (Stair-work-unaffected,
per Phase 4) data load correctly, receive the expected 25-field canonical feature vector in the expected
order, and produce predictions without schema mismatch — mechanically guarded by
`validate_model_compatibility()` (Phase 5) on every registry lookup, not merely assumed. Pipeline A has no
inference-side counterpart to audit — it has never produced a registry-tracked or live-deployable model.

---

## Phase 8 — Backward compatibility & versioning

**Older datasets remain usable.** Pipeline A already practices disciplined, working dataset versioning:
each campaign version (`schema.py` v1.0 through `schema_v4.py` v4.0) is a separate, never-mutated module,
and each generated campaign lives in its own version-suffixed directory
(`data/predictive_dataset_campaign_v1` through `_v4`) — old datasets and the code that reads them both
remain intact and re-runnable. **No new dataset-versioning mechanism is needed** — the existing discipline
already satisfies this milestone's own bar, and this audit found no case where it was violated.

**Older trained models remain loadable** for Pipeline B: `ModelMetadata` (`ai_registry/metadata.py`)
already carries `model_version`, `training_dataset_identifier`, and `feature_schema_version` per model,
and `validate_model_compatibility()` checks the CURRENT schema against each model's own recorded values
at lookup time rather than assuming compatibility — an old model whose schema has since diverged is
correctly rejected by `get_latest_compatible_model()`, not silently mismatched. Pipeline A has no
model-loading concept to audit here (see Phase 5/6).

**Should model versions be incremented because of the Stair work?** No — Phase 4 found zero schema or
target changes attributable to it, so no trained model's `feature_schema_version` or `ordered_
feature_names` contract has been invalidated. Nothing needs a forced version bump on that basis.

**Recommendation (new, genuine gap found by this audit, not previously documented):** no campaign
report or dataset directory anywhere in Pipeline A records which commit of the simulation engine
produced it. This audit had to infer that fact indirectly, by comparing the dataset file's OS-level
modification timestamp against `git log` commit timestamps (Phase 4) — a fragile, manual, easily-wrong
technique. Recommend Pipeline A's campaign report JSON (`campaign_config.to_dict()`) gain one additional
field — a git commit hash captured at generation time — the next time any campaign-config-touching change
is made. Not implemented here (out of this audit's "verify, do not redesign" scope), but concretely
justified by a real difficulty this audit itself ran into.

---

## Phase 9 — Large-scale readiness

**Yes to 1000, 5000, and 10000 — all three were actually generated, not extrapolated.** Every scale
tested from 10 through 10000 scenarios (10,008 scenarios accepted at the top end, 13,085,902 rows,
1,533.3s wall, zero failures) completed with zero failures, zero corruption, and near-linear, predictable
wall-time scaling (8,500-10,200 rows/sec sustained, 6.5-7.2 scenarios/sec across the entire 1000x range
from 10 to 10000) — no cliff, no instability, no manual intervention required at any scale tested. Process
RSS delta grew from 5.1MB (10 scenarios) to only 17.9MB (10000 scenarios) — sub-linear, confirming no
memory leak even at the largest scale run. The built-in `psutil` memory-abort guard
(`MIN_AVAILABLE_MEMORY_BYTES = 300MB`) never triggered even on this audit's own tight ~7.9GB-RAM machine,
across all seven runs including the largest.

---

## Phase 10 — Performance

- **Sustained throughput**: ~9,000-10,200 rows/sec across 10-2500 scenarios, dipping to ~8,500-8,700
  rows/sec at 5000 and 10000 (a mild, expected slowdown as larger/more-candidate-count topology variants
  accumulate more total simulation work per variant pass — not a stability concern; no error, no memory
  pressure signal accompanies it; throughput plateaus rather than continuing to degrade between 5000 and
  10000, 8,711 → 8,536 rows/sec).
- **Scenarios/sec**: 6.5-7.2 across every scale tested, including the full 10000-scenario run (6.53).
- **Disk**: ~175-176 bytes/row, stable across every scale from 10 to 10000 scenarios (schema size is
  fixed; no row-size drift) — the 10000-scenario dataset totals 2.30GB.
- **Wall-clock for a full 10000-scenario campaign**: 1,533.3s (~25.6 minutes) on this audit's own
  hardware (6 physical/12 logical cores, ~7.9GB total RAM) — a concrete, measured number for anyone
  planning a large validation campaign's schedule, not an extrapolation.
- **Feature generation cost**: not separately isolable from total wall time in the current runner (no
  internal profiling breakpoints between "simulate" and "extract features" phases) — a possible future
  instrumentation addition, not pursued here.
- **CPU**: ~99% utilization of one logical core throughout every run; the other 11 logical cores on this
  6-core/12-thread machine are idle for the campaign's entire duration (see Phase 2).

---

## Limitations

- This audit's "before" comparison for Phase 4 relies on file-mtime-vs-commit-timestamp inference
  (Phase 8's own recommended fix would remove this fragility for future audits).
- Pipeline A has no shared/versioned dataset loader (Phase 6) — every consuming script hand-rolls its own
  column handling.
- No git-commit provenance is recorded in any campaign report (Phase 8).
- `SCHEMA_VERSION = "1.0"` is reused, coincidentally, across two unrelated schema spaces (Phase 5).
- Feature-generation cost is not separately profiled from simulation cost (Phase 10).
- None of these are correctness defects in the generated data itself — every dataset-content check in
  Phases 2-4 passed cleanly at every scale tested.

## Recommendations

1. Add a git-commit-hash field to Pipeline A's campaign report JSON at generation time (Phase 8).
2. Consider a shared, versioned Pipeline-A dataset loader analogous to `ai_training.load_campaign_
   dataset()`, the next time a schema version bump is made (reduces the "N independent scripts each
   hand-roll column renaming" fragility found in Phase 6) — not urgent today.
3. Namespace the two coincidentally-identical `SCHEMA_VERSION = "1.0"` constants (Phase 5) the next time
   either module is touched for an unrelated reason.
4. If a materially faster large-campaign turnaround is ever needed, parallelizing across the 24
   structural variants (embarrassingly parallel, no shared state between variants) would use the 11
   currently-idle logical cores on hardware like this audit's own machine — not pursued here since no
   throughput problem was found at the scales tested.

## Recommended interpretation

The dataset generation pipeline is correct, stable, and unaffected by the recent Stair-work milestones
at every scale this audit tested. The genuine gaps found (commit provenance, shared loader, schema-
version namespace collision) are all forward-looking hygiene recommendations, not defects in any
dataset already generated or any dataset a fresh run today would produce.
