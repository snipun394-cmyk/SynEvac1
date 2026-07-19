# SynEvac Platform — Full Engineering Validation Report

**Date:** 2026-07-15
**Scope:** End-to-end scientific validation of the completed SynEvac platform (Building Designer → Scenario Generator → Interactive Simulation → Dataset Builder → Ground Truth → Decision Policy → Campaign Analytics → AI Training → AI Explainability → AI Inference → RL Training).
**Nature of this phase:** validation, benchmarking, profiling, and documentation only. No architecture was redesigned and no new platform features were added; the only new artifacts are the validation scripts under `validation/` (used to produce the numbers in this report) and this document itself.

---

## 1. Architecture Overview

SynEvac is organized as a pipeline of independent, single-responsibility packages, each communicating through frozen dataclasses rather than shared mutable state:

```
Scenario Definition ─▶ Scenario Generator ─▶ Scenario Validator ─▶ (scenario_pipeline orchestration)
        │
        ▼
Scenario Runner ─▶ Behaviour Profile Resolver ─▶ Human Behavior Layer ─▶ Multi-Agent Simulation (simulator/)
        │                                                                        │
        ▼                                                                        ▼
Hazard Evolution Engine                                              Simulation Runtime (tick loop)
        │                                                                        │
        └───────────────────────────┬────────────────────────────────────────────┘
                                     ▼
                    Dataset Builder / Ground Truth / Decision Policy
                                     │
              ┌──────────────────────┼──────────────────────┐
              ▼                      ▼                      ▼
      Training Dataset Toolkit   Campaign Analytics    AI Training / Inference / Explainability

                    Interactive Simulation (step-driven wrapper, additive)
                                     │
                                     ▼
                              RL Training (Gymnasium + Stable-Baselines3)
```

Two parallel execution models coexist by design:

- **`simulation_runtime.SimulationRuntime`** — the original, run-to-completion engine. Occupant movement is solved once, synchronously, at construction time (`MultiAgentSimulation.run()`), and `tick()` only replays/observes hazard, occupancy, and decision state against that already-fixed movement result.
- **`simulation_interactive.InteractiveSimulation`** — an additive, step-driven wrapper built in a later phase specifically to make RL honest: it never calls `MultiAgentSimulation.run()` up front, instead draining the same event heap one event at a time (`MovementStepper`), so routes can be recomputed mid-episode in response to door/exit/stair changes, hazard changes, and externally injected recommendations (`RouteManager`, `ActionExecutor`).

Neither engine was modified during this validation phase.

---

## 2. Phase 1 — Large Campaign Validation

**Method.** Rather than only extrapolating from a small sample, campaigns were actually executed at every requested scale (10 / 50 / 100 / 300 / 1,000 / 5,000 / 10,000 scenarios) against a small, fixed 2-zone/2-floor building (mirrors `tests/training_dataset_fixtures.py`'s own shape), running the real pipeline sequence (`scenario_pipeline.run_batch_pipeline` → `scenario_runner.run` → `behaviour_profile_resolver.register_occupants` → `ai_decision.AIDecisionEngine` → `simulation_runtime.SimulationRuntime` → `dataset_builder.DatasetBuilder.export_all` → `ground_truth.analyze` → `decision_policy.generate_policy`), bypassing only the QThread/Qt-signal machinery `CampaignWorker` wraps this same sequence in (pure compute cost, not GUI event-loop overhead). Memory was sampled via `psutil` process RSS before/after each run. Script: `validation/phase1_campaign_benchmark.py`; raw data: `validation/results/phase1_campaign_benchmark.json`.

| Scenarios | Generation (s) | Simulation (s) | Dataset export (s) | Ground truth + Decision policy (s) | Total (s) | Memory Δ (MB) | Failures |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.006 | 0.013 | 0.003 | 0.004 | 0.025 | 0.5 | 0 |
| 50 | 0.026 | 0.054 | 0.006 | 0.016 | 0.101 | 1.8 | 0 |
| 100 | 0.052 | 0.110 | 0.010 | 0.032 | 0.204 | 1.9 | 0 |
| 300 | 0.147 | 0.325 | 0.026 | 0.094 | 0.592 | 8.7 | 0 |
| 1,000 | 0.487 | 1.101 | 0.081 | 0.318 | 1.987 | 35.9 | 0 |
| 5,000 | 2.688 | 5.935 | 0.428 | 1.622 | 10.673 | 193.6 | 0 |
| 10,000 | 6.215 | 12.936 | 0.804 | 3.153 | 23.108 | 380.1 | 0 |

**Findings.**
- **0% failure rate at every scale** — every requested scenario was generated, accepted, simulated, and exported successfully up to 10,000 scenarios.
- **Near-perfectly linear scaling.** Per-scenario cost is stable across three orders of magnitude: generation ≈ 0.5–0.6 ms/scenario, simulation ≈ 1.0–1.3 ms/scenario, at both 10 and 10,000 scenarios. This is strong evidence against algorithmic blow-up (no accidental O(n²) behavior in the batch path).
- **Memory grows linearly, not sub-linearly** (380 MB for 10,000 scenarios vs. 194 MB for 5,000 — almost exactly double). This is because this benchmark (like `DatasetBuilder`'s own API) accumulates every `SimulationRun` in memory before a single batch `export_all()` call, rather than streaming rows to disk incrementally. For campaigns far beyond 10,000 scenarios this is the first resource limit to hit — see §6 (Performance Profiling) and §8 (Future Work).
- These numbers are for a **small validation building** (2 zones, 1 door, 1 exit, ~1–2 occupants/scenario). Real campaigns against large, geometrically complex buildings (see §5's "very large building" stress test — 50 zones, 5 floors) will cost meaningfully more per scenario, dominated by pathfinding search cost (§6). The linear *scaling law* should still hold — only the per-scenario constant changes.

---

## 3. Phase 2 — AI Validation

**Method.** A held-out, `Staff_Default`/`Adult_Default`-mixed, 800-scenario campaign was generated through the real Campaign Studio pipeline (`tests/training_dataset_fixtures.py::make_campaign`, with the occupancy/fire-growth distributions widened from that fixture's own deliberately-narrow unit-test ranges — see below) and loaded via `ai_training.load_campaign_dataset`. An 80/20 train/test split (`ai_training.make_split`, `random_state=0`) was applied. Two model families were trained and evaluated with all three interchangeable estimators the platform already supports (`ai_training.models.base.build_regressor`/`build_classifier`): **Random Forest**, **Gradient Boosting**, **XGBoost**. Script: `validation/phase2_ai_validation.py`; raw data: `validation/results/phase2_ai_validation.json`.

**A note on methodology, stated up front:** an initial run using `training_dataset_fixtures.make_definition()` verbatim (fixed fire-growth parameter, 1–2 occupants/zone) produced **perfect** R²=1.0 and accuracy=1.0 for every algorithm. This is real, but not a meaningful validation result — it reflects that fixture's own near-constant target distribution (it exists for fast, deterministic *unit* tests, not for exercising a regressor). The table below instead uses a deliberately higher-variance definition (fire growth `UniformRange(50, 400)`, occupancy `UniformRange(1, 8)` per zone, mixed behaviour profiles) built from the same existing `FixedValue`/`UniformRange`/`WeightedOptions` distribution primitives — no new distribution kind was added.

**Campaign:** 800/800 scenarios accepted (0 rejected), mean simulated evacuation time 115.2 s.

### 3.1 Regression — `EvacuationTimeModel` (target: `total_evacuation_time`)

| Algorithm | MAE | RMSE | R² | Train time (s) |
|---|---:|---:|---:|---:|
| Random Forest | 13.30 | 18.13 | 0.846 | 0.36 |
| Gradient Boosting | 12.61 | 17.00 | **0.865** | 0.11 |
| XGBoost | 14.05 | 19.61 | 0.820 | 1.45 |

### 3.2 Classification — `BottleneckModel` (target: `occurrence`, i.e. did any door become a bottleneck)

| Algorithm | Accuracy | Precision | Recall | F1 | ROC AUC | Confusion Matrix (rows=true, cols=pred: `[False, True]`) |
|---|---:|---:|---:|---:|---:|---|
| Random Forest | 0.919 | 0.852 | 0.897 | 0.872 | 0.964 | `[[25, 4], [9, 122]]` |
| Gradient Boosting | 0.919 | 0.866 | 0.856 | 0.861 | 0.966 | `[[22, 7], [6, 125]]` |
| XGBoost | 0.919 | 0.855 | 0.883 | 0.868 | 0.960 | `[[24, 5], [8, 123]]` |

**Findings.** All three algorithms perform comparably (within ~2–4 percentage points of each other on every metric) — Gradient Boosting had the best regression R² and trained fastest of the three on this dataset size; XGBoost was consistently the slowest to train (1.4s vs. 0.1–0.4s) without a corresponding accuracy advantage at this scale. Bottleneck classification recall (0.86–0.90) is somewhat below precision, meaning the models under-flag true bottleneck scenarios more often than they false-alarm — worth weighting toward recall (e.g. class weighting) if this model informs safety-critical decisions later.

---

## 4. Phase 3 — RL Validation

**Method.** PPO (Stable-Baselines3, `MlpPolicy`) trained for 50,000 timesteps on `SynEvacGymEnv`, with a `ConvergenceCallback` re-evaluating the current deterministic policy over 20 fixed scenarios every 5,000 timesteps (via the same `rl_training.evaluate_policy` function used for the final comparison, so the convergence curve and the final numbers share one code path). The trained policy was then compared against the no-intervention baseline and the two-pass, deterministic-Decision-Policy baseline (§ design decision in `rl_training`'s plan: pass 1 runs with no intervention to obtain a `GroundTruth`/`DecisionPolicy`, since `generate_policy` structurally requires an already-finished run; pass 2 replays the same scenario with that policy's recommendations actually applied). Script: `validation/phase3_rl_validation.py`; raw data: `validation/results/phase3_rl_validation.json`.

Two building/occupancy configurations were tried:

- **1 occupant** (the `rl_training` test-fixture default): reward converges immediately and stays flat at 10.036 for all 10 checkpoints — the RL policy learns to always choose NOOP, which is *correct*: a single occupant already takes the shortest path, so there is nothing to optimize and no congestion to relieve.
- **6–10 occupants funneled toward one near exit while a second, much farther exit sits unused** (deliberately constructed to give `RECOMMEND_EXIT` a real optimization opportunity):

| Checkpoint (timesteps) | Average reward (20 eval scenarios) |
|---:|---:|
| 5,000 | 11.812 |
| 10,000 | 11.720 |
| 15,000 | 11.837 |
| 20,000 | 11.888 |
| 25,000 | 11.657 |
| 30,000 | 11.758 |
| 35,000 | 11.784 |
| 40,000 | 11.785 |
| 45,000 | 11.939 |
| 50,000 | 11.771 |

Training time: 50.6 s for 50,000 timesteps (CPU).

**Final comparison** (20 held-out scenarios, deterministic policy/action for the RL row):

| Policy | Avg. reward | Avg. evacuation time (s) | Avg. peak congestion | Avg. trapped |
|---|---:|---:|---:|---:|
| RL (PPO) | 11.90 | 89.48 | 8.45 | 0.0 |
| No intervention | 11.80 | 85.65 | 8.05 | 0.0 |
| Decision Policy (compiled, two-pass) | 11.96 | 89.48 | 8.45 | 0.0 |

**Findings.**
- Training is **stable but does not show strong monotonic improvement** — the checkpoint curve oscillates in a narrow band (11.66–11.94) rather than climbing steadily. On a 6–10 occupant, single-building scenario family this is a small, low-dimensional decision problem (discrete action space of 6, short episodes) — there is limited room for PPO to demonstrate a large learning curve, and 50,000 timesteps is a modest budget by RL standards. This is an honest result, not a failure: it shows the training loop is mechanically correct (loss decreases, no crashes, reproducible evaluation), but this validation was not run at a scale (timesteps, building diversity) that would let PPO's advantage compound.
- **RL and the compiled Decision Policy baseline produced numerically identical evacuation times and congestion**, both slightly worse than no-intervention on this scenario family. The most likely explanation (not independently re-verified beyond this observation): both policies converged on recommending the same exit that the default shortest-path routing already uses for this zone, so the "recommendation" was a functional no-op with respect to *movement*, while still costing a small "action taken" component in the reward accounting (hence RL's and Decision Policy's *reward* being marginally higher than no-intervention's despite an *identical or slightly worse* simulated outcome). **This is a genuine tension in the reward design worth flagging**: `evacuation_progress`/`time_penalty` reward components and the `congestion`/`exit_balance` components can pull in opposite directions when the "balanced" exit is geometrically much farther away — a recommendation that reduces peak congestion can still increase total evacuation time if the alternate route is long enough. Future work should either weight these components more carefully per building, or make `RECOMMEND_EXIT` conditional on the alternate exit being within some reasonable relative-distance factor of the default.
- Both RL and Decision Policy left `average_peak_congestion` unchanged from no-intervention (8.45 vs. 8.05 — actually marginally *higher*), meaning **neither approach demonstrably reduced congestion on this scenario family** — a concrete, honestly-reported negative result, not glossed over.

---

## 5. Phase 4 — Engineering Validation

Six targeted, hand-designed checks (known-correct expected answers by construction, the same idiom the existing 1,600+ test pytest/unittest suite already uses) were run against the real pipeline. Script: `validation/phase4_engineering_validation.py`; raw data: `validation/results/phase4_engineering_validation.json`.

| Check | Result |
|---|---|
| Recommended exit matches the geometrically closer exit | **PASS** |
| Recommended stair matches the only viable cross-floor route | **PASS** |
| Bottleneck prediction identifies a deliberately narrow, heavily-used door | **PASS** |
| Hazard spread order starts at the ignition zone | **PASS** |
| Ground truth occupant accounting is internally consistent (`reachable + unreachable == total`; `total_evacuation_time == max(arrival_time)`) | **PASS** |
| Decision Policy's own documented risk thresholds (`CRITICAL_RISK_THRESHOLD=0.75`, hazardous-zone override) are the rule actually applied, across 4 synthetic risk/hazard combinations | **PASS** |

**Finding:** all six engineering-correctness properties hold as documented. No discrepancy was found between the platform's documented decision rules (recommendation logic, bottleneck/hazard detection, ground-truth bookkeeping) and its actual behavior in these targeted cases.

---

## 6. Phase 5 — Robustness Testing

Ten stress scenarios were run end-to-end. Script: `validation/phase5_robustness_tests.py`; raw data: `validation/results/phase5_robustness_tests.json`.

| Stress case | Result | Notes |
|---|---|---|
| Very small building (1 zone, 1 exit, 1 occupant) | **PASS** | Cleared in 2.4 ms |
| Very large building (5 floors × 10 zones = 50 zones, 50 occupants) | **PASS** | 5.1 s, 50/50 reachable |
| Single exit funneling 4 zones (20 occupants) | **PASS** | 20/20 reachable and evacuated |
| Multiple exits (5 zones, 5 exits, 25 occupants) | **PASS** | Building cleared |
| Heavy occupancy (200 occupants, one door + one exit) | **PASS** | 0.08 s, 200/200 reachable, evacuation time 4,470 s (expected — genuine capacity-limited queueing) |
| Detector failure (`DeviceAvailability.FAILED`) | **PASS** | State correctly applied, no crash |
| Camera failure (`DeviceAvailability.FAILED`) | **PASS** | State correctly applied, no crash |
| Blocked exit (one of two exits closed) | **PASS** | Occupants correctly rerouted through the remaining exit |
| Locked door (only path to the only exit) | **PASS*** | See finding below |
| Stair failure (only path to the only exit) | **PASS*** | See finding below |

**Discovered finding (starred rows above):** when a zone has genuinely **no route to any exit** (locked door, closed stair, on the *default* `ShortestRouteChoiceStrategy` path), the affected occupants are classified `OccupantState.STATIONARY`, **not** `OccupantState.UNREACHABLE`, and `GroundTruth.reachable_occupants`/`unreachable_occupants` therefore **do not reflect the true unreachability** of these occupants (they are silently counted as "reachable" even though no path exists). Root cause, traced directly:

- `PathfindingEngine.nearest_exit()` correctly returns `None` (verified directly — `Edge.traversable` is `False` for the locked door/closed stair, and `pathfinding/engine.py:375` does exclude non-traversable edges from the search).
- `ShortestRouteChoiceStrategy.choose()` (`behavior/route_choice.py`) then returns `RouteChoice(goal_id=None, route=None)` when `route is None`.
- `MultiAgentSimulation.submit_decision()` (`simulator/coordinator.py`) treats `goal_id is None and route is None` as "no movement decision" (the same shape a deliberate `WAIT`/`IGNORE` intent produces), registering the occupant as `STATIONARY` — the `UNREACHABLE` state is only reachable through `add_occupant()`'s `reached_goal=False` branch, which this code path never calls.

This is a **real, reproducible gap between the platform's own documented semantics and its behavior** for this specific edge case (a zone becoming provably disconnected from every exit at registration time), surfaced only by deliberately combining a locked door / closed stair with an otherwise-normal scenario. It does not cause a crash, incorrect building-cleared reporting, or data corruption — `building_cleared` correctly remains `False` — but any downstream metric or model that reads `GroundTruth.reachable_occupants`/`unreachable_occupants` specifically (rather than `building_cleared` or per-occupant final state) will under-count truly-unreachable occupants whenever they arise from a *structural* disconnection rather than an *unlucky pathfinding* one. Per this validation phase's explicit rules, this was documented, not fixed. See §8.

---

## 7. Phase 6 — Performance Profiling

`cProfile` + `tracemalloc` were run over each pipeline phase at 500 scenarios (same building as Phase 1). Script: `validation/phase6_profiling.py`; raw data: `validation/results/phase6_profiling.txt`.

| Phase | Wall time | Peak traced memory |
|---|---:|---:|
| Generation | 1.38 s (500 scenarios) | 2.2 MB |
| Simulation | 3.34 s (500 scenarios) | 16.1 MB |
| Dataset export | 0.37 s (500 scenarios) | 1.8 MB |
| Ground truth + Decision policy | 1.06 s (500 scenarios) | 0.06 MB |

**Identified hotspots** (no behavior changed anywhere — profiling only):

1. **Generation:** `scenario_validator.dataset_validation.compute_candidate_content_hash` (JSON-serializing every generated candidate for dedup hashing) accounts for ≈22% of generation wall time (0.30s of 1.38s) — mostly `json.dumps`/`encoder.iterencode`. A hash computed from a smaller canonical projection of the candidate (rather than the full serialized dict) would likely be materially cheaper at very large batch sizes.
2. **Simulation (the dominant cost by far): `ai_decision.engine.AIDecisionEngine.decide()` → `_zone_recommendation` → `PathfindingEngine.nearest_exit()`/`_search`/`_relax`.** This one call chain accounts for **≈72% of all per-tick simulation time** (1.73s of 2.41s spent inside `SimulationRuntime.tick()`), because a full pathfinding search is recomputed **from scratch for every zone, on every tick** (14,051 `nearest_exit` calls across 4,064 ticks in this 500-scenario run), even on ticks where neither hazard nor occupancy changed for a given zone. This is the single clearest, safe (behavior-preserving) optimization target identified: caching or incrementally invalidating a zone's recommended route between hazard-state changes could remove the majority of this cost without changing any observable output.
3. **Ground truth / Decision policy: `dataset_builder.timeline.extract_timeline_rows`** accounts for ≈75% of this phase's time (0.79s of 1.06s), driven by 81,780 calls to `HazardSnapshot.node_state()` — one lookup per zone per tick, recomputed independently for every row rather than sharing a single per-tick hazard read across the zone loop.
4. **Memory** is not a concern at this scale for any individual phase (peaks are all under 17 MB for 500 scenarios) — the Phase 1 finding that *cumulative* memory grows linearly across an entire large batch (§2) is a separate, batch-accumulation concern (holding every `SimulationRun` in memory before one `export_all()` call), not a per-tick leak.

No duplicate-work bugs (identical computation performed twice for no reason) were found beyond the "recomputed every tick instead of cached/invalidated" pattern in (2)/(3) above, which is a genuine optimization opportunity but not a correctness defect — every number produced is still correct, just recomputed more often than strictly necessary.

---

## 8. Limitations, Known Assumptions, and Future Work

**Limitations of this validation itself:**
- Phase 1's 10,000-scenario benchmark and Phase 6's profiling both used a small, fixed validation building (2–3 zones). Absolute per-scenario timings will be materially higher against large, geometrically complex real buildings (Phase 5's 50-zone/5-floor stress building took ~100 ms/scenario rather than ~1–2 ms) — the demonstrated *linear scaling law*, not the absolute constants, is what should be expected to generalize.
- Phase 3's RL training budget (50,000 timesteps, one building/occupancy configuration) is modest by RL standards; the "does not show strong convergence" finding should be read as "not yet demonstrated at this budget/scale," not "the reward design cannot work."
- Phase 2's near-perfect scores on the *fixture's own* default definition (§3) were reported and explained rather than hidden, since silently only reporting the higher-variance run's numbers would obscure how sensitive these metrics are to scenario-definition variance — a reader building on this platform should pick training scenario distributions deliberately, not accept whatever a test fixture happens to produce.

**Known assumptions carried over from the existing platform (not introduced by this validation):**
- Congestion/capacity modeling uses `DefaultCapacityModel`/`DefaultCongestionModel`'s documented linear degradation curve (floor speed factor 0.3× at/over capacity) — a simplification, not a validated crowd-dynamics model. As of the Stair Capacity Modeling fix, Stair edges are further narrowed by `StairCapacityModel` (width-plus-vertical-travel, `simulator/capacity.py`) and slowed by a counter-flow penalty from `StairAwareCongestionModel` (`simulator/congestion.py`) when occupants cross the same stair in opposite directions — both wrap the Default models above and are, like them, documented, disclosed constants rather than validated crowd-dynamics values.
- `HazardSeverity` score cutoffs and `zone_policy`/`stair_policy` risk thresholds are documented, disclosed engineering constants, not values fit to real incident data.
- ~~Stairs have no modeled capacity anywhere in the platform~~ — resolved by the Stair Capacity Modeling fix: `ground_truth/bottleneck.py::stairs_exceeding_capacity` and `ground_truth/risk_analysis.py::compute_stair_risk_scores` now both reuse `simulator.capacity.derive_stair_capacity` (the same formula `StairCapacityModel` applies during simulation) instead of the previous hardcoded-empty/queue-fraction-only behavior.

**Newly discovered in this validation phase (§6, Phase 5):** the `STATIONARY`-vs-`UNREACHABLE` gap for zones structurally disconnected from every exit at registration time. Recommended (not performed, per this phase's rules) future remediation: either (a) have `ShortestRouteChoiceStrategy`/`MultiAgentSimulation._register()` distinguish "chose not to move" from "no route exists" explicitly, or (b) have `GroundTruth`'s reachability accounting independently re-derive reachability from the navigation graph rather than solely from final `OccupantState`.

**Future work suggested by this validation, in priority order:**
1. Cache or incrementally invalidate `AIDecisionEngine`'s per-zone recommended route between hazard-state changes (Phase 6, §7.2) — the single largest identified, safe performance win.
2. Resolve the `STATIONARY`/`UNREACHABLE` ground-truth gap (Phase 5, §6).
3. Stream `DatasetBuilder` output incrementally for very large campaigns instead of accumulating every `SimulationRun` in memory before one `export_all()` call (Phase 1, §2).
4. Re-run Phase 3's RL validation at a larger timestep budget and across multiple building topologies before drawing conclusions about PPO's ceiling on this environment; consider reward-component reweighting or a distance-aware `RECOMMEND_EXIT` guard to resolve the evacuation-time/congestion tension found in §4.
5. Reduce redundant per-row hazard lookups in `dataset_builder.timeline.extract_timeline_rows` (Phase 6, §7.3).

None of the above were implemented in this phase, per its own rules (validate/benchmark/profile/document only).

---

## Appendix — Reproducing this report

All numbers above were produced by the scripts in `validation/` (added this phase, not part of the platform proper):

```
python validation/phase1_campaign_benchmark.py     # -> validation/results/phase1_campaign_benchmark.json
python validation/phase2_ai_validation.py          # -> validation/results/phase2_ai_validation.json
python validation/phase3_rl_validation.py          # -> validation/results/phase3_rl_validation.json
python validation/phase4_engineering_validation.py # -> validation/results/phase4_engineering_validation.json
python validation/phase5_robustness_tests.py       # -> validation/results/phase5_robustness_tests.json
python validation/phase6_profiling.py              # -> validation/results/phase6_profiling.txt
```

Environment: Windows, Python 3.14.0, torch 2.11.0+cpu, gymnasium 1.3.0, stable-baselines3 2.9.0, scikit-learn/xgboost/joblib per `requirements.txt`.
