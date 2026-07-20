# Live-Compatible AI Model Training & Model Registry

Status: **two production-candidate-track models trained, versioned, and registry-served; nothing wired into `LiveOrchestrator` or `advisory_system`.** This document is the record of the `ai_registry/` package this milestone added on top of `ai_features/` (`docs/architecture/ai_live_feature_parity.md`).

## 1. Model lifecycle (investigation findings)

Before writing any code, the existing lifecycle was traced in full:

- **Training:** `ai_training.experiment.ExperimentRunner.run()` ties `TrainingDataset` → `model_cls.build_table()` → `ai_training.split.make_split()` → `estimator.fit()` → `ai_training.evaluation.evaluate_*()` into one call. Reused unmodified — `ai_registry.training` calls `EvacuationTimeModel.build_table()`/`BottleneckModel.build_table()` directly, exactly as `ExperimentRunner` does, just against a live-compatible `TrainingDataset` instead of the legacy one.
- **Artifact persistence:** `BaseModel.save()`/`.load()` (`ai_training/models/base.py`) persist a `ModelBundle` (estimator + `Preprocessor` + `FeatureSchema` + target metadata) via `joblib`. **`FeatureSchema` (numeric/categorical column split) already survives a save/load round-trip** — `loaded_model.model.feature_schema.columns` gives the exact trained column list, confirmed and reused directly by `ai_features.compatibility.check_model_compatibility()` (previous milestone).
- **Manifest:** `ExperimentRunner.save_result()` additionally writes `manifest.json` (config + metrics + train/test sizes) — **no feature-column list, no schema version, no deployability marker** exists in it. Confirmed via direct inspection: `ModelProvenance` (`ai_inference/loader.py`) carries no schema information either.
- **Schema/version validation:** **None exists anywhere in `ai_inference`/`ai_training`** — confirmed by the previous milestone's own audit (`Predictor._compute_value_and_probability()` hands a row straight to `Preprocessor.transform()`, which silently NaN-fills a missing required column and silently ignores extra keys). `ai_features.compatibility` (previous milestone) is the first schema check this pipeline has ever had.
- **Metrics:** persisted in `manifest.json`, never separately versioned.
- **Splits:** `ai_training.split.make_split()` already supports deterministic, group-aware splitting (`GroupShuffleSplit` under a fixed `random_state`) — reused directly, `groups=scenario_ids`.
- **Research-vs-live-compatible confusion risk:** **Real, and unaddressed before this milestone** — nothing in `manifest.json`/`ModelProvenance` distinguishes a model trained on the full simulation-only scenario-feature set from one trained on a live-compatible subset. This is exactly what `ai_registry.metadata.ModelMetadata`'s `model_deployability` field (§2) now closes.

**Conclusion: no existing infrastructure was duplicated.** `ai_registry` is additive: `ModelMetadata` is saved as a *third* file (`live_metadata.json`) alongside the pre-existing `model.joblib`; nothing about `BaseModel`/`ExperimentRunner`/`ai_inference.Predictor` was modified.

## 2. Metadata contract

`ai_registry.metadata.ModelMetadata` — every field Phase 2 named, at minimum:

`model_id`, `model_type`, `model_version`, `training_timestamp`, `training_dataset_identifier`, `training_seed`, `feature_schema_version`, `ordered_feature_names`, `prediction_target`, `model_deployability` (`Deployability.LIVE_COMPATIBLE` / `Deployability.RESEARCH_ONLY` — never inferred from a filename), `training_metrics`, `validation_metrics`, `missing_data_policy`.

`model_id` is unique per trained artifact (`f"{ModelClass}-{version_stamp}"`); `model_type` (`"evacuation_time"`/`"bottleneck_occurrence"`) is the stable lookup key `ModelRegistry.get_latest_compatible_model()` uses across versions.

**Impossible to accidentally load a RESEARCH_ONLY model live:** `ModelRegistry.get_model()` raises `ResearchOnlyModelError` by default (requires `allow_research_only=True`); `get_latest_compatible_model()` filters to `LIVE_COMPATIBLE` *and* independently re-validates via `ai_features.compatibility.check_model_compatibility()` — a model that merely self-declares `LIVE_COMPATIBLE` but whose real feature columns don't match the canonical schema still gets rejected.

## 3. Training dataset generation

`ai_registry.training_scenario.make_training_building()`/`make_training_definition()` — a substantially richer `Building` (2 floors, 4 zones, 2 doors, 2 exits, 1 stair, 3 cameras, 4 canonical detectors) and `ScenarioDefinition` than the tiny fixture existing `ai_training`/`training_dataset` unit tests use (which remains completely untouched). Varies, via the existing `scenario_definition` distribution vocabulary (`UniformRange`/`WeightedOptions`, nothing new): fire growth rate and ignition-zone weighting, per-zone occupant counts and behaviour-profile mix (5 profiles), door/exit/stair state, camera availability, and firefighter team presence/size/arrival time. `detector_state_distribution` was deliberately **not** used — `EngineeringConstraints` validation only resolves it against legacy `Floor.detectors`, not the canonical `SmokeDetector`/`HeatDetector` assets this building uses (confirmed by `DefinitionValidationReport` rejecting the canonical ids as `unknown_id`); camera/door/exit/stair/occupancy/fire/firefighter variation already gives substantial diversity without it.

Every varied factor above affects the **simulation and its labels** only; `ai_registry.campaign.build_live_compatible_dataset()` re-extracts every scenario's INPUT features through `ai_features.simulation_extractor.extract_canonical_training_row()` — the canonical schema, nothing more — so none of this extra diversity leaks into the model's inputs.

**Target semantics (Phase 3 investigation):** `EvacuationTimeModel_LiveCompatible`'s target remains **`total_evacuation_time`** (not "remaining evacuation time from t") — unchanged from the existing model, confirmed against `dataset_builder/labels.py:extract_simulation_outcome()`. See §9 for why a `RemainingEvacuationTimeModel` was investigated but not built this milestone. `BottleneckOccurrenceModel_LiveCompatible`'s target is **exactly** the existing `bool(GroundTruth.doors_that_became_bottlenecks)` definition (`ground_truth/bottleneck.py`) — no new bottleneck definition was created.

## 4. Scenario split strategy

`ai_training.split.make_split(n_rows, groups=scenario_ids, test_size=0.15, val_size=0.15, random_state=<training_seed>)` — the existing, already group-aware splitter, reused directly (no new splitting logic written). Because this milestone's design captures exactly one canonical-feature row per scenario (the same "T = alarm activation" snapshot the AI Feature Parity milestone established — see §9), grouping by `scenario_id` and grouping by row are numerically identical here; `groups=` is still passed explicitly so the split remains provably leak-free by construction, and so a future multi-row-per-scenario design (§9) would need zero split-logic changes. `tests/test_ai_registry.py::SplitDeterminismAndLeakageTests` proves both the determinism and the zero-overlap guarantee directly. **Exact scenario counts, full 5000-scenario run:** 3500 train / 750 validation / 750 test (see §10).

## 5. Feature schema version

`ai_features.feature_schema.SCHEMA_VERSION` (currently `"1.0"`) is stamped into every trained model's `ModelMetadata.feature_schema_version` and re-validated on every registry lookup (`ModelRegistry.validate_model_compatibility()` also checks `ordered_feature_names` order/membership match the *currently running* `CANONICAL_LIVE_FEATURE_NAMES` exactly, catching a schema drift a bare version string could miss).

## 6. Model registry

`ai_registry.registry.ModelRegistry` — `register_model()`/`register_model_directory()`, `list_models(model_type=, deployability=, feature_schema_version=, model_version=)`, `get_model(model_id, allow_research_only=False)`, `get_latest_compatible_model(model_type, feature_schema_version=SCHEMA_VERSION)`, `validate_model_compatibility(model_id)`. Loads once (either via direct in-memory `register_model()` or `register_model_directory()`), caches in an in-memory dict — `tests/test_ai_registry.py::RegistryCachingTests` proves repeated lookups return the identical cached model object, never a fresh disk load.

## 7. Inference service

`ai_registry.inference_service.LiveAIInferenceService` — `BuildingState → extract_canonical_features() → registry.get_latest_compatible_model() → validate_feature_row() → model.predict()/predict_proba() → structured prediction`. Returns `EvacuationTimePrediction` (`predicted_seconds`, `uncertainty_seconds` — `None` when genuinely unavailable, `model_id`, `model_version`, `feature_schema_version`, `timestamp`) or `BottleneckOccurrencePrediction` (`probability`, `predicted_occurrence`, `threshold`, `model_id`, `model_version`, `feature_schema_version`, `timestamp`). Every failure mode Phase 11 named raises `InferenceUnavailableError` — never a fabricated prediction: no compatible model, missing/unexpected required features, schema mismatch, or a corrupted/raising estimator. **Not imported by `live_system` or `advisory_system` anywhere** — mechanically enforced (`tests/test_ai_registry.py::NoLiveWiringGuardTests`).

## 8. Uncertainty handling

- **Bottleneck occurrence:** raw `predict_proba()` retained as the reported `probability`, unmodified — Phase 9's own explicit instruction. `ai_registry.uncertainty.probability_calibration_report()` additionally computes a Brier score and a binned reliability table as an *evaluation*, not a recalibration — poor calibration found there is the trigger for a future `CalibratedClassifierCV` wrap, not attempted this milestone.
- **Evacuation time:** `ai_registry.uncertainty.regression_ensemble_uncertainty()` — per-row prediction std across the fitted `RandomForestRegressor`'s individual trees (the smallest architecture-compatible method available, since `random_forest` is the default/only algorithm trained here). Returns `None` (never a fabricated number) for any estimator without an `estimators_` ensemble structure — proven by `tests/test_ai_registry.py::UncertaintyHonestyTests`.

## 9. Temporal prediction semantics (Phase 12 investigation)

**Confirmed: the existing target, and this milestone's `EvacuationTimeModel_LiveCompatible`, both predict *total* evacuation time, not *remaining* evacuation time from an arbitrary observation time t.** Both are single-snapshot (one row per scenario, captured at "T = alarm activation" — established by the AI Feature Parity milestone) predictors of one eventual outcome, matching every one of the four original models' own shape.

**Investigated, not built this milestone: `RemainingEvacuationTimeModel`.** A remaining-time target is computable today without new simulation work — `dataset_builder/timeline.py`'s per-tick `people_evacuated`/`people_remaining` and `simulation_time` columns already let `remaining_time(t) = total_evacuation_time - t` be derived directly from existing exported data for any tick `t < total_evacuation_time`. What is genuinely missing is the FEATURE side: `ai_features.simulation_extractor` only builds a single "alarm activation" `BuildingState` snapshot per scenario today; a `RemainingEvacuationTimeModel` needs a `BuildingState` at *each* sampled tick `t` (occupant count declining as people evacuate, detector/FACP state possibly changing), which is a genuine, non-trivial extension of that module (and would reintroduce the multi-row-per-scenario leakage risk §4's `groups=` splitting already anticipates but doesn't yet need). Given this milestone's explicit scope (two named models, `total_evacuation_time` confirmed as the correct, unchanged target) and the instruction to investigate first and not redesign automatically, building this is deferred to a future milestone — recorded here as a scoped, well-understood follow-up, not an unknown unknown.

**For a live Command Center, remaining time would likely be more directly actionable than total time** — this is worth prioritizing whenever that future milestone is picked up.

## 10. Model evaluation, baseline comparisons, production readiness

Full run: `python scripts/train_live_compatible_models.py --count 5000 --seed 2026` — 5000/5000 scenarios accepted (128.8s generation, 25.8ms/scenario), split 3500 train / 750 validation / 750 test (70/15/15, scenario-grouped, deterministic for `training_seed=2026`).

**`EvacuationTimeModel_LiveCompatible` — status: EXPERIMENTAL.**

| | test MAE | test RMSE | test R² |
|---|---|---|---|
| model | 119.55 | 174.81 | **0.088** |
| baseline (mean) | 126.16 | 183.15 | -0.001 |
| baseline (median) | **118.56** | 189.49 | -0.071 |

R² is positive and beats both baselines' R² — but the model's own MAE (119.55) is marginally *worse* than the median baseline's MAE (118.56). `evacuation_time_model_status()` requires beating the best baseline MAE, not merely a positive R², and correctly returns `EXPERIMENTAL` here — a real, un-softened result: occupancy + device-health signal alone captures some, but not most, of what drives `total_evacuation_time` in this campaign. Error grows with occupancy (`low_0_5`: MAE 22.6 over 2 samples; `medium_6_15`: MAE 100.7 over 268; `high_16_plus`: MAE 130.5 over 480) and the worst individual errors exceed 900 seconds on high-occupancy, long-evacuation scenarios — the model systematically under-predicts the longest evacuations.

**`BottleneckOccurrenceModel_LiveCompatible` — status: PRODUCTION_CANDIDATE.**

| | accuracy | precision | recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| model | 0.948 | 0.642 | 0.512 | **0.511** | **0.793** | 0.985 |
| baseline (majority) | 0.949 | 0.475 | 0.500 | 0.487 | 0.500 | — |
| baseline (stratified) | 0.896 | 0.497 | 0.497 | 0.497 | 0.497 | — |

**Class balance is heavily imbalanced** (True/bottleneck-occurred: 95.5% train, 94.9% test) — this building's constrained exit/door layout produces a bottleneck in the overwhelming majority of generated scenarios. Accuracy alone (0.948) is barely distinguishable from the majority-class baseline's accuracy (0.949) *and would be actively misleading reported alone* — the model's real edge is in F1 (0.511 vs 0.487) and especially ROC-AUC (0.793 vs 0.500, i.e. genuine ranking ability the majority baseline has none of). Confusion matrix: TP=710, TN=1, FP=37, FN=2 — the model is good at catching true bottlenecks (recall on the rare False class is the weak point, only 1 of 38 true negatives caught). Probability calibration: Brier score 0.045 (lower is better; a perfectly calibrated model on this class balance would score near the base rate's own variance) — raw `predict_proba()` is retained as the reported probability per Phase 9's instruction; this Brier score is the evaluation that instruction asked for, not a claim of perfect calibration.

Both models are evaluated against real `sklearn.dummy` baselines (`DummyRegressor(strategy="mean"/"median")`, `DummyClassifier(strategy="most_frequent"/"stratified")`) via the existing `ai_training.metrics` scoring functions, and are labeled `PRODUCTION_CANDIDATE` only if they genuinely beat their own baseline on held-out data (`ai_registry.training.evacuation_time_model_status()`/`bottleneck_model_status()`) — never assumed production-ready merely for training successfully.

**Inference latency** (`scripts/train_live_compatible_models.py`'s own benchmark, 20-iteration average): `BuildingState` canonical feature extraction 0.007 ms; cached model registry lookup 0.015 ms; evacuation-time inference (dominated by the 200-tree ensemble-variance uncertainty pass) 61.9 ms; bottleneck inference 32.0 ms; combined (extract + lookup + both predictions) 108.1 ms. Registry lookup being ~4000x cheaper than model inference confirms the registry's in-memory caching (§6) is not the bottleneck — the estimator's own prediction cost (200 trees) is, and would be the first thing to optimize (e.g. fewer trees, or skip the uncertainty pass on cycles that don't need it) before wiring this into any tight live loop.

## 11. What remains explicitly unconnected

Exactly as instructed: `ai_registry` has zero callers from `live_system`, `advisory_system`, or `decision_policy` (mechanically enforced). The future flow this milestone prepares, but does not build:

```
BuildingState -> LiveAIInferenceService -> {EvacuationTimePrediction, BottleneckOccurrencePrediction}
    -> Decision Policy / Advisory System -> Recommendations
    -> Command Center
```

AI predictions are advisory input only in this future design — nothing skips directly from a prediction to a building control action, and nothing in this milestone changes that boundary.
