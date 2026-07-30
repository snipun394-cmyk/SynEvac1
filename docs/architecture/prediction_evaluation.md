# Prediction vs Reality Evaluation Framework

Status: implemented. A new, standalone `prediction_evaluation/` package -- purely evaluative, never
wired into Simulation, Live Runtime, Recommendation, Guidance, or any operational UI (mechanically
proven, see "Safety isolation" below). Builds on `docs/architecture/shadow_mode_prediction.md`; neither
the predictive model nor Recommendation/Guidance were touched by this milestone.

---

## Motivation

Shadow Mode already produces, timestamps, and stores a prediction every cycle a `live_ai_gateway` is
configured. Nothing previously compared those predictions against what actually happened. This milestone
is that comparison -- a scientific evaluation layer, not a model improvement, not a new decision path.

## Evaluation pipeline

```
PredictionRecord (Phase 1)          GroundTruthSample (Phase 3)
  prediction_id, timestamp,           timestamp, source,
  model_id/version,                   total_occupant_count,
  feature_schema_version,             congestion_detected,
  prediction_horizon_seconds,         congested_asset_ids,
  payload (opaque, duck-typed),       queue_lengths, flow_rates,
  source, scenario_id/building_id,    hazard_overall_severity,
  context_tags                        evacuation_complete/_time_seconds
        │                                     ▲
        │  PredictionRegistry.record()        │  extract_ground_truth_sample()
        │  (append-only, immutable)           │  (pure function, duck-typed against
        ▼                                     │   CrowdIntelligenceSnapshot/hazard_
  GroundTruthTimeline.resolve(                 │   summary/evacuation_progress_snapshot
      prediction.timestamp                     │   -- IDENTICAL whether the caller is
      + prediction.prediction_horizon_seconds, │   Simulation or Live Runtime)
      tolerance_seconds)                       │
        │                                GroundTruthTimeline
        ▼                                (sorted, bisect-searchable)
  MatchedEvaluation (Phase 2)
        │
        ├──> pairs.py (extracts (y_true, y_pred[, y_proba]) per model output)
        │       ├──> classification_metrics.py (Phase 4: precision/recall/F1/
        │       │      confusion matrix/ROC/calibration/confidence bias)
        │       └──> regression_metrics.py (Phase 4: MAE/RMSE/MAPE/bias/CI/
        │              worst-best/std)
        ├──> horizon_analysis.py (Phase 5: grouped by 5/10/20/30/60s)
        ├──> condition_analysis.py (Phase 6: grouped by caller-supplied tags)
        ├──> statistics.py (Phase 8: per-scenario/per-building/overall)
        ├──> comparison.py (Phase 9: N models, same scenarios)
        └──> visualization.py (Phase 7: static plots, evaluation-only)
```

`evaluator.evaluate(predictions, timeline, tolerance_seconds, horizon_buckets_seconds)` is the one
convenience entry point tying every stage into a single `EvaluationReport` -- never required; every
module above is independently usable (e.g. `comparison.compare_models()` for a model-A-vs-model-B study
that never needs `horizon_analysis`).

## Ground truth definition, precisely -- a disclosed scope difference from training

`ai_registry.training.train_bottleneck_occurrence_model()`'s own documented `prediction_target` is
`bool(GroundTruth.doors_that_became_bottlenecks)` -- a **whole-scenario retrospective** label (`ground_
truth.bottleneck.compute_engineering_findings()`'s own `_has_any_queueing()`: did ANY door ever queue,
anywhere, across an ENTIRE completed simulation run). This milestone's own `GroundTruthSample.
congestion_detected` is instead a **windowed, point-in-time** indicator: was `crowd_intelligence.models.
BuildingCrowdSummary.congested_doors`/`congested_exits`/`congested_stairs` non-empty AT (or near) time
`T + horizon` specifically (the same `congestion_level >= HIGH` threshold `CrowdIntelligenceEngine`
already applies identically in both Simulation and Live Runtime -- reused, not reinvented).

**These are genuinely different questions** -- "did congestion ever happen anywhere in this whole run"
vs. "is there congestion right now, at this specific future instant." This is the honest, correct thing
to evaluate a "prediction AT T+horizon" against (Phase 2's own explicit framing), even though it is not
literally the model's own training target. The difference is disclosed here, not hidden -- a model
trained on the whole-scenario label may show different accuracy under this windowed evaluation than its
own training-time metrics reported, and that is expected, not a bug in either the model or this
framework. A future milestone wanting evaluation to exactly mirror the training target would need a
`compute_engineering_findings()`-style extractor operating over an entire recorded run, not a per-
timestamp snapshot -- not built here (would require re-deriving a different, whole-run-scoped ground-
truth concept, out of this milestone's "implement only the evaluation framework" scope).

`evacuation_time_experimental`'s ground truth (`evacuation_time_seconds`) has no such scope mismatch:
for Simulation, `MultiAgentSimulationResult.total_evacuation_time` (already computed, reused directly).
For Live Runtime, `evacuation_progress.models.EvacuationProgressSnapshot.known_active_occupants == 0`
(only when `known_total_observed_occupants > 0`, honestly) is the best available live analog -- genuinely
weaker than Simulation's own clean total (a live session's own start-of-evacuation instant is less crisply
defined than a simulation's `depart_time=0.0`), disclosed as a known limitation.

## Metrics computed (Phase 4)

**Classification** (`bottleneck_occurrence`): precision, recall, F1, a 4-tuple confusion matrix (fixed
`[False, True]` label order, never inferred from whichever classes happen to appear in a sample), ROC
curve + AUC (only when both classes are present in the sample -- otherwise honestly `None`), a 10-bucket
calibration curve + mean calibration error, and a signed confidence bias (mean predicted probability
minus empirical positive rate).

**Regression** (`evacuation_time`): MAE, RMSE, MAPE (`None` whenever any true value is exactly 0 --
division by zero is undefined, never silently clamped), signed bias, mean/median/std of signed error,
worst-case/best-case absolute error, and a 95%-CI on the mean signed error (normal approximation,
disclosed, not a fabricated exact interval).

Both reuse `sklearn.metrics` directly (`precision_score`/`recall_score`/`f1_score`/`confusion_matrix`/
`roc_curve`/`auc`/`mean_absolute_error`/`mean_squared_error`/`mean_absolute_percentage_error`) -- already
an established dependency (`ai_training`, `ai_registry`, `research_framework/figures.py`) -- never
reimplemented by hand.

**Occupancy/queue/flow/hazard**: extracted honestly as `GroundTruthSample` fields (`total_occupant_count`,
`queue_lengths`, `flow_rates`, `hazard_overall_severity`), but the CURRENT model outputs
(`bottleneck_occurrence`, `evacuation_time`) have no directly corresponding PREDICTED value for these --
there is no "predicted occupancy" or "predicted queue length" field anywhere on `LiveAIPredictionSnapshot`
today. `visualization.plot_ground_truth_metric_over_time()` honestly plots the ground-truth signal alone
for exploratory use, never a fabricated prediction series. A future model type predicting these
quantities directly would plug into `pairs.py` with its own extractor function, reusing every metric/
horizon/condition/comparison module unchanged.

## Time horizon analysis (Phase 5)

`horizon_analysis.analyze_by_horizon()` groups by `prediction.prediction_horizon_seconds` into the five
named buckets (5/10/20/30/60s, exact match, not `match_time_delta_seconds`, a different concept -- match
slack vs. requested horizon), computing the identical classification/regression metrics per bucket.
Proved with a controlled test (`tests/test_prediction_evaluation.py::HorizonAnalysisTests::test_accuracy_
can_differ_across_horizons`) that accuracy genuinely can, and is expected to, differ by horizon.

## Building condition analysis (Phase 6)

`PredictionRecord.context_tags` is free-form key/value metadata a caller attaches when recording a
prediction -- this package defines no closed vocabulary or validation; `condition_analysis.
SUGGESTED_TAG_KEYS` documents Phase 6's own named categories (`occupancy_level`, `floor_mode`,
`exit_status`, `congestion_level`, `fire_origin`, `building_layout`) purely as a naming convention, never
enforced. `analyze_by_condition(evaluations, tag_key)` groups by whichever key a caller actually used.

## Visualization (Phase 7) -- evaluation only

`prediction_evaluation/visualization.py`: predicted-vs-actual scatter (regression), error-over-time,
accuracy-by-horizon (dual classification/regression panel), confusion matrix, calibration curve, and
ground-truth occupancy/congestion-over-time. Same Agg-backend, `CHART_DPI`/`_save`/`_ensure_parent`
convention as `research_framework/figures.py`/`campaign_analytics/visualizations.py` (restated, not
imported -- this package must stay import-independent of both, per its own architecture guard). **Not
imported by `command_center/` or any operational UI anywhere** -- mechanically proven (see below).

## Statistical analysis (Phase 8)

`regression_metrics.RegressionMetrics` already carries every field Phase 8 names (mean/median/std/
worst/best/95% CI) -- computed once, in Phase 4's own module, never a second time. `statistics.py`'s own
job is purely the ADDITIONAL grouping Phase 8 asks for that `condition_analysis.py` does not cover:
`per_scenario_statistics()`/`per_building_statistics()` group by `PredictionRecord.scenario_id`/
`building_id` (dedicated fields, not free-form tags), plus `overall_statistics()` for the ungrouped
totals.

## Model comparison (Phase 9)

`comparison.compare_models({label: evaluations, ...})` -- any number of independently-evaluated
`MatchedEvaluation` sequences, keyed by whatever label a caller chooses (typically `model_id`, but a
caller comparing two checkpoints of the same registered model can key by `model_version` instead).
`ModelComparisonReport.better_classifier()`/`.better_regressor()` rank by F1/MAE respectively, returning
`None` (never an arbitrary pick) when fewer than two models produced a comparable metric. **The caller is
responsible for ensuring every model's own evaluation set covers the SAME underlying scenarios** -- this
module never verifies scenario identity itself (it has no way to know what "the same scenario" means
across two independently-run evaluation passes without being told).

## Safety isolation (Phase 10) -- mechanically proven

`tests/test_prediction_evaluation_architecture_guards.py` proves, by source-text scan, BOTH directions:

1. `prediction_evaluation/` never imports `evacuation_recommendation`, `evacuation_guidance`, `simulator`,
   `simulation_runtime`, `live_system`, `live_runtime`, `live_runtime_launcher`, `voice_evacuation`,
   `speaker_manager`, `dynamic_signage`, `sign_manager`, `building_control`, `facp`, `command_center`,
   `ai_registry`, `ai_inference`, or `ai_training` -- and never calls a FACP/Voice/Building-Control action
   verb or a direct model-inference method (`.predict_bottleneck_occurrence(`/`.predict_evacuation_time(`).
2. **None of those operational packages import `prediction_evaluation` either** -- the reverse direction,
   equally enforced. There is no code path, in either direction, connecting evaluation to any operational
   decision or execution surface.

Since evaluation runs entirely out-of-band (a caller records predictions and ground truth AFTER a
Simulation run or a Live Runtime cycle has already happened, using data it already has), this import-level
proof is a stronger, more durable guarantee than a runtime timing measurement -- there is no code path
through which evaluation COULD affect Simulation timing, Live Runtime, Recommendation, or Guidance,
regardless of how much evaluation work is performed or how slow it runs.

## Performance

`tests/test_prediction_evaluation_e2e.py`: a full `evaluate()` pass (matching + classification metrics +
regression metrics + horizon breakdown + per-scenario/per-building statistics) over 20 real predictions
from a real trained model completed in ~103 ms -- entirely offline, entirely after the fact, with zero
measurable relationship to any Simulation/Live Runtime cycle time (see "Safety isolation" above for why
that relationship is structurally impossible, not merely fast).

## Simulation / Live Runtime parity

`extract_ground_truth_sample()` is a single, source-agnostic function -- proved directly
(`test_simulation_and_live_sources_use_identical_extraction_logic`) that feeding the identical
`CrowdIntelligenceSnapshot`-shaped input through it with `source="simulation"` vs. `source="live"`
produces byte-identical `GroundTruthSample` fields except for the `source` label itself. There is no
separate simulation-only or live-only ground-truth computation anywhere in this package.

## Known limitations

- The disclosed `congestion_detected` scope difference from `bottleneck_occurrence`'s own whole-scenario
  training target (see "Ground truth definition, precisely" above) -- the single most important caveat
  for interpreting classification metrics from this framework.
- No predicted occupancy/queue-length/flow-rate output exists on the current model(s) to evaluate against
  the corresponding, already-extracted ground-truth fields -- those fields exist for a future model type,
  and for exploratory ground-truth-only visualization today.
- Live evacuation-completion ground truth (`evacuation_complete`) is weaker than Simulation's own clean
  `total_evacuation_time` -- a live session has no equally crisp "evacuation began at t=0" reference point.
- `GroundTruthTimeline.resolve()` performs a nearest-timestamp match, never interpolation -- a caller
  needing sub-tolerance precision must record ground-truth samples at a correspondingly fine cadence.
- Calibration/ROC statistics require both outcome classes present in a given group (horizon bucket,
  condition group, scenario, ...) -- small or highly imbalanced groups honestly report `None` rather than
  a numerically unstable or misleading value.

## Recommended interpretation

Treat classification metrics from this framework as measuring "does the model's probability correctly
anticipate windowed congestion at the stated horizon," not "does the model correctly predict the same
whole-scenario outcome it was trained on" -- these are related but distinct questions, and only the first
is what live/simulation timestamp-matched evaluation can honestly answer. Horizon and condition
breakdowns are the most actionable outputs for deciding whether a model is ready for any future,
still-unstarted decision-integration milestone -- a model whose accuracy collapses sharply past 20s, or
that performs far worse under multi-floor/blocked-exit conditions, is not "broken," but is disclosing
exactly where its own reliability boundary lies.
