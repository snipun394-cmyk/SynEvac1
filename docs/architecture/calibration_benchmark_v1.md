# Calibration Benchmark Framework V1

## Objective

Compare a real-world, dataset-derived candidate value for one existing SynEvac parameter against
that parameter's current production default, across an identical, paired scenario campaign, and
produce a publication-quality report with a statistical adopt/reject/inconclusive recommendation.

This is explicitly a **benchmarking** tool, not a **calibration** tool: nothing in this package
writes to `behaviour_profile_resolver/registry.py`, `simulator/capacity.py`,
`simulator/congestion.py`, or `crowd_intelligence/models.py`. Every SynEvac default a candidate is
compared against remains exactly what it was before the benchmark ran. A candidate becomes
"adopted" only when a human reads this framework's report and makes a separate, explicit code
change elsewhere — this framework has no code path capable of making that change itself.

The mapping from external dataset field to SynEvac parameter is taken as already decided (the
external SynEvac Calibration Mapping Report) — this milestone does not re-derive or re-justify
those mappings, only turns a given mapping's candidate value into something a real scenario
campaign can be run against.

## 1. Architecture

```
calibration_benchmark/
    candidates.py        ParameterCandidate + 5 concrete candidates (Walking Model,
                          Pre-movement Model, Capacity Model/Stair Model, Congestion Model,
                          Crowd Intelligence)
    simulation_seam.py    run_with_overrides() -- the one injection seam this milestone needed
    metrics.py            MetricSample + the five headline metrics
    optional_metrics.py   AdditionalMetric interface + Prediction Accuracy / Recommendation
                          Effectiveness reference implementations
    harness.py            run_calibration_benchmark() -- the paired campaign runner
    recommendation.py     recommend() -- per-metric and overall adopt/reject/inconclusive verdict
    report.py             render_markdown_report() / save_report()
```

### Why a new `simulation_seam.py` was needed

`scenario_runner.run(scenario, building)` is SynEvac's one frozen entry point for turning a
`Scenario` into a ready-to-simulate `SimulationContext`, but its own
`occupant_initializer.build_simulation(engine)` hardcodes
`MultiAgentSimulation(engine, capacity_model=StairCapacityModel(), congestion_model=StairAwareCongestionModel())`
with no parameter to override either model. `behaviour_profile_resolver.registrar.register_occupants(context, registry=None)`
already exposes exactly the registry-override seam this milestone needed for Walking Model and
Pre-movement Model candidates — no change was needed there.

`calibration_benchmark/simulation_seam.py::run_with_overrides()` restates `scenario_runner.run()`'s
own composition (every one of its calls is to `scenario_runner`'s own already-public,
already-frozen construction functions) with one additional parameter: an injectable
`capacity_model`/`congestion_model` pair, defaulting to the exact same
`StairCapacityModel()`/`StairAwareCongestionModel()` production defaults `build_simulation()`
itself hardcodes. Calling it with every override left at its default reproduces
`scenario_runner.run()`'s own result exactly — verified directly in
`tests/test_calibration_benchmark_simulation_seam.py::RunWithOverridesReproducesProductionDefaultsTests`.

This restatement pattern is not novel to this milestone: `research_framework/runner.py`'s own
module docstring already restates `designer/campaign/campaign_worker.py`'s
generate→simulate→export sequence for exactly the same reason — "the one seam \[the existing
entry point\] has no hook for."

### Non-mutation guarantee

Every `ParameterCandidate.baseline_*()`/`candidate_*()` method constructs and returns a brand-new
object on every call:

- Walking Model / Pre-movement Model candidates copy `DEFAULT_PROFILE_REGISTRY` into a plain
  `dict` and replace exactly one profile's `BehaviorProfileTemplate` via `dataclasses.replace()`
  (the template is frozen — this can never mutate the shared instance every other caller still
  reads).
- Capacity Model / Congestion Model candidates use runtime subclassing
  (`type("Candidate...", (DefaultCapacityModel,), {"PEOPLE_PER_METER_OF_WIDTH": value})`) to
  override exactly one class constant, never editing the production class itself.
- Crowd Intelligence candidates construct a new `DensityThresholds(...)` dataclass instance.

`tests/test_calibration_benchmark_harness.py::test_never_leaves_any_production_default_mutated`
runs two full benchmarks (one per candidate family) and then asserts every one of
`DEFAULT_PROFILE_REGISTRY`, `DefaultCapacityModel.PEOPLE_PER_METER_OF_WIDTH`,
`StairCapacityModel.PEOPLE_PER_METER_OF_WIDTH`, and `DefaultCongestionModel.MINIMUM_SPEED_FACTOR`
is byte-for-byte unchanged afterward.

## 2. Workflow

```
ParameterCandidate (current value, candidate value, dataset source, rationale)
        |
        v
run_batch_pipeline(definition, definition_id, building, master_seed, n_scenarios)
        |  -- generates N scenarios ONCE (same seeds for both arms: a true paired design)
        v
for each scenario:
    run_with_overrides(scenario, building, baseline_registry/capacity/congestion) -> GroundTruth A
    run_with_overrides(scenario, building, candidate_registry/capacity/congestion) -> GroundTruth B
    extract_metrics(...) on each -> one paired (MetricSample_A, MetricSample_B)
        |
        v
per metric field: paired_comparison() / effect_size_cohens_d() / confidence_interval()
    (all reused directly from research_framework/statistics.py -- not reimplemented)
        |
        v
recommend(result) -> per-metric verdict (ADOPT / REJECT / INCONCLUSIVE / NOT_APPLICABLE)
                      + one overall verdict (any REJECT wins; else any ADOPT; else INCONCLUSIVE)
        |
        v
render_markdown_report(result, recommendation) -> publication-quality Markdown report
```

The design is a **true paired comparison**: `run_batch_pipeline()` generates the scenario batch
exactly once, and each scenario is then run twice (baseline arm, candidate arm) from the same seed.
This is what makes `research_framework.statistics.paired_comparison()` (a paired t-test, already
built and already used elsewhere in this codebase for exactly this shape of problem) the
statistically correct tool, rather than comparing two independently-sampled distributions.

## 3. Supported Parameter Types

| Candidate class | Subsystem | SynEvac parameter overridden | Calibration Mapping Report tier |
|---|---|---|---|
| `WalkingSpeedCandidate` | Walking Model | `DEFAULT_PROFILE_REGISTRY[profile_id].walking_speed` | Tier 2 |
| `PreMovementDelayCandidate` | Pre-movement Model | `DEFAULT_PROFILE_REGISTRY[profile_id].pre_movement_strategy` (`ProbabilisticPreMovementDelay.median_delay`/`spread`) | Tier 2 |
| `CapacityWidthCandidate` | Capacity Model / Stair Model | `DefaultCapacityModel.PEOPLE_PER_METER_OF_WIDTH` or `StairCapacityModel.PEOPLE_PER_METER_OF_WIDTH` (`stair_specific=True`) | Tier 1 |
| `CongestionMinimumSpeedFactorCandidate` | Congestion Model | `DefaultCongestionModel.MINIMUM_SPEED_FACTOR` | Tier 2 |
| `DensityThresholdCandidate` | Crowd Intelligence | `DensityThresholds` (`moderate_at`/`high_at`/`very_high_at`/`critical_at`) | Tier 1 |

Extending to a new parameter means adding one new `ParameterCandidate` subclass overriding only
the `baseline_*()`/`candidate_*()` methods it actually needs to change — every other method
inherits the untouched production default from the base class, so a new candidate type can never
accidentally leave a *different* subsystem's baseline/candidate pair mismatched.

### Metrics

| Metric | Source | Notes |
|---|---|---|
| Evacuation Time | `GroundTruth.total_evacuation_time` | direct |
| Congestion Duration | `GroundTruth.congestion_duration` | direct; `None` whenever the run's own peak-congestion location was a zone rather than a door/exit/stair edge (`ground_truth/bottleneck.py`'s own definition only computes a duration for edge-type locations) |
| Queue Length | `GroundTruth.peak_congestion_value` | direct (peak simultaneous occupants on the single worst edge) |
| Maximum Density | `calibration_benchmark.metrics.compute_peak_occupancy_ratio()` | **explicitly a proxy, not a literal areal people/m² measurement** — GroundTruth has no areal-density field at all; this is the worst (peak occupants on an edge ÷ that edge's own modeled capacity) ratio across every Door/Exit/Stair, using whichever `CapacityModel` the run itself used |
| Exit Utilization | `calibration_benchmark.metrics.compute_exit_utilization_balance()` | `1 - (|underutilized ∪ exceeding| / total_exits)`, derived from `GroundTruth.exits_underutilized`/`exits_exceeding_capacity` |
| Prediction Accuracy *(optional)* | `calibration_benchmark.optional_metrics.PredictionAccuracyMetric` | scoped specifically to `DensityThresholds` candidates: classifies this run's own peak occupancy ratio and checks agreement with the same run's actual bottleneck outcome — no trained model, no feature pipeline |
| Recommendation Effectiveness *(optional)* | `calibration_benchmark.optional_metrics.RecommendationEffectivenessMetric` | fraction of this run's own engineering-findings targets (`doors_that_became_bottlenecks`/`exits_exceeding_capacity`/`stairs_exceeding_capacity`) that `GroundTruth.recommendations` also addressed by `target_id` |

Both optional metrics were deliberately kept self-contained (reusing only data a normal
`GroundTruth` already produces for the same run) rather than reaching into
`predictive_dataset`/`predictive_model`'s own full feature-extraction/training pipeline — doing
that per benchmark scenario would mean training or re-scoring a real ML model inside a
benchmarking tool, which risks drifting into the "automatic calibration" this milestone was
explicitly told not to build. `AdditionalMetric` is an open extension point for a future, heavier
implementation.

## 4. Report Generation

`calibration_benchmark/report.py::render_markdown_report()` produces one self-contained Markdown
document per benchmark run:

1. Parameter under test (subsystem, current value, candidate value, dataset source, rationale).
2. Experiment settings (scenario count, paired design statement, statistical test used).
3. Before/after comparison table (baseline mean, candidate mean, mean difference, Cohen's d,
   p-value, n) for all five headline metrics plus any additional metrics supplied.
4. Statistical significance, per metric, in plain language.
5. A recommendation section with one overall verdict and an explicit reminder that adopting it
   requires a separate, human-made code change.

`save_report()` writes the same Markdown to a given path. Every number in the report is a `float`
already computed by `harness.py`/`recommendation.py` — the report module performs no independent
computation of its own beyond formatting, including an explicit, honest rendering of the
degenerate "every paired scenario produced an identical value" case (scipy's paired t-test returns
`NaN`, not a small p-value, when there is no variance in the differences at all — rendered as
"identical values (no variance)" rather than a bare, confusing "p=nan").

## 5. Test Results

45 new tests across 7 files, all passing:

| File | Tests | Covers |
|---|---|---|
| `test_calibration_benchmark_candidates.py` | 11 | every candidate type's baseline/candidate objects, non-mutation of production defaults, rejection of invalid profile ids/strategy types |
| `test_calibration_benchmark_simulation_seam.py` | 4 | `run_with_overrides()` reproduces `scenario_runner.run()`'s own result byte-for-byte with no overrides; a real override produces a real, direction-correct outcome difference; no mutation |
| `test_calibration_benchmark_metrics.py` | 6 | every `MetricSample` field populated correctly from a real simulation run; `peak_occupancy_ratio`/`exit_utilization_balance` bounds; a building with no measurable edges returns `None`, not a fabricated value |
| `test_calibration_benchmark_harness.py` | 5 | end-to-end paired campaign run; direction-correct effect from a real candidate; additional metrics wired through; global non-mutation across two different candidate families; zero-scenario edge case |
| `test_calibration_benchmark_optional_metrics.py` | 7 | both optional metrics' honest-`None`, correct-positive, correct-negative, and partial-coverage cases |
| `test_calibration_benchmark_recommendation.py` | 5 | ADOPT/REJECT/INCONCLUSIVE/NOT_APPLICABLE verdict logic, including "any regression rejects even if something else improved" and higher-is-better metric direction |
| `test_calibration_benchmark_report.py` | 5 | report names the parameter and both values, states no default was changed, includes experiment settings and a recommendation section, `save_report()` writes identical content to disk |

A demonstration run (`scripts/run_calibration_benchmark_demo.py`) benchmarks the Calibration
Mapping Report's own worked example — `Adult_Default.walking_speed` (current default 1.2 m/s)
against a candidate value (0.65 m/s) derived directly from the already-acquired Jülich Pedestrian
Dynamics Data Archive stair-egress trajectories (free-flow-condition instantaneous speed, mean
0.649 m/s across 1,818 real samples) — across 30 paired scenarios in a single-bottleneck test
building (`tests/calibration_benchmark_fixtures.py`, 25 occupants through one capacity-2 door and
exit). Full report: `docs/architecture/calibration_benchmark_v1_demo_report.md`.

**Result: REJECT.** The candidate speed significantly worsens Evacuation Time (684 s → 1254 s
mean, p<0.0001, Cohen's d=8.3) with no compensating significant improvement on any other metric.
This is disclosed in the demo report's own rationale as an expected, informative finding rather
than a data error: the Jülich candidate value is a real **stair-descent** speed from a calm,
instructed experiment, while SynEvac's `walking_speed` is applied as one flat constant across both
level and stair movement. The benchmark correctly did not — and structurally cannot — adopt this
value; it surfaced the trade-off for a human to weigh, including the genuine follow-on finding that
SynEvac may benefit from a separate stair-specific base speed rather than a single flat
`walking_speed` constant (a Calibration Mapping Report Part 3.1 caveat that this run gives the
first concrete, quantified evidence for).

Full existing repository regression suite (`tests/` directory, pre-existing suite) was re-run
after this milestone's changes to confirm zero regressions from the new package.

## 6. What Was Not Built

Per the milestone's own explicit constraints: no automatic calibration (nothing selects, fits, or
applies a candidate value on its own), no new heavy ML pipeline for Prediction Accuracy (a
lightweight, self-contained proxy was built instead, with the heavier option left as a documented
extension point), and no change to any of `scenario_runner`, `behaviour_profile_resolver`,
`simulator`, or `crowd_intelligence` — every one of those packages is used exactly as it already
existed.
