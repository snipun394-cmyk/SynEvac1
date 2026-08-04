"""
Automatic Calibration Campaign V2 -- first full calibration campaign on
the corrected Flow Region architecture.

Admission Control V8 (research milestone, no code) found that Flow
Region grouping (FlowRegionCapacityModel + FlowRegionCongestionModel,
use_flow_regions=True) is the only architectural change that has
produced a large improvement across all four published NIST buildings
(3.05x-16.44x overprediction -> 0.35x-0.71x underprediction), and that
Admission Control V4 (Discharge)/V7 (Buffer) were only ever evaluated
against the legacy per-edge architecture -- never composed with Flow
Regions. This script does not re-litigate that finding; it acts on it.

This script modifies NO production source file. It is a pure CLIENT of
already-existing, already-tested infrastructure, exactly the discipline
scripts/run_automatic_calibration_walking_speed_campaign.py already
established:

  - automatic_calibration/ (engine, budget, grid search, objectives,
    search space) -- used exactly as-is.
  - calibration_studio/ -- used exactly as-is (CalibrationStudio,
    PublishedBenchmark).
  - calibration_benchmark/ -- used exactly as-is (ParameterCandidate as
    a base class to subclass from OUTSIDE the module, run_with_overrides,
    the statistics/recommendation helpers). candidates.py itself is
    imported from, never edited.
  - simulator/flow_region_congestion.py -- used exactly as-is. See
    _flow_region_congestion_model() below for the one wrinkle this
    required working around from the client side (FlowRegionCongestionModel
    hardcodes self._region_model = DefaultCongestionModel() in its own
    __init__, with no constructor parameter to override it -- the
    MERGED-region code path, which is the one that now matters most,
    has no injection point at all for MINIMUM_SPEED_FACTOR). Worked
    around by constructing the model normally through its public
    constructor and then reassigning that one already-mutable instance
    attribute from here -- no class definition, no production file, and
    no method body is touched anywhere in simulator/.

Two new ParameterCandidate subclasses are defined below, following the
exact shape of the existing candidates in calibration_benchmark/candidates.py
(CongestionMinimumSpeedFactorCandidate, WalkingSpeedCandidate,
FlowRegionCapacityCandidate) but living entirely in this script:

  - FlowRegionCongestionFloorCandidate -- Phase 1. Both arms run with
    Flow Regions on; only DefaultCongestionModel.MINIMUM_SPEED_FACTOR
    (applied uniformly to both the region path and the single-edge
    fallback path) differs between baseline (production default, 0.3)
    and candidate.
  - FlowRegionCalibratedWalkingSpeedCandidate -- Phase 2. Both arms run
    with Flow Regions on AND Phase 1's selected congestion floor fixed
    identically on both sides; only Adult_Default.walking_speed differs.

Phase 3 is a straight 4-arm comparison, run the same way every other
NIST validation script in this repo already runs its comparisons
(scripts/run_admission_control_v7_nist_validation.py,
scripts/run_flow_region_nist_validation.py): a single already-generated,
seeded scenario batch run once per arm via run_with_overrides() directly
-- not through CalibrationStudio, since this is validation of four fixed
configurations, not a search.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from run_nist_10story_validation import (  # noqa: E402
    build_nist_10story_building, build_nist_10story_definition,
    DEFINITION_ID as DEFINITION_ID_10, MASTER_SEED as MASTER_SEED_10,
)
from run_nist_18story_validation import (  # noqa: E402
    build_nist_18story_building, build_nist_18story_definition,
    DEFINITION_ID as DEFINITION_ID_18, MASTER_SEED as MASTER_SEED_18,
)
from run_nist_24story_validation import (  # noqa: E402
    build_nist_24story_building, build_nist_24story_definition,
    DEFINITION_ID as DEFINITION_ID_24, MASTER_SEED as MASTER_SEED_24,
)
from run_nist_31story_validation import (  # noqa: E402
    build_nist_31story_building, build_nist_31story_definition,
    DEFINITION_ID as DEFINITION_ID_31, MASTER_SEED as MASTER_SEED_31,
)

from calibration_benchmark import ParameterCandidate, recommend, run_with_overrides  # noqa: E402
from calibration_benchmark.candidates import _registry_with_walking_speed  # noqa: E402

from calibration_studio.benchmark import BenchmarkType, GeometryVersion, PublishedBenchmark, PublishedValue  # noqa: E402
from calibration_studio.studio import CalibrationStudio  # noqa: E402

from automatic_calibration.budget import AutoCalibrationBudget  # noqa: E402
from automatic_calibration.engine import AutoCalibrationEngine  # noqa: E402
from automatic_calibration.grid_search import GridSearchStrategy  # noqa: E402
from automatic_calibration.objectives import PublishedValueObjective  # noqa: E402
from automatic_calibration.search_space import ParameterDimension, SearchSpace  # noqa: E402

from behaviour_profile_resolver.registry import DEFAULT_PROFILE_REGISTRY  # noqa: E402

from scenario_pipeline import run_batch_pipeline  # noqa: E402

from simulator.capacity import StairCapacityModel  # noqa: E402
from simulator.congestion import DefaultCongestionModel, StairAwareCongestionModel  # noqa: E402
from simulator.flow_region_capacity import FlowRegionCapacityModel  # noqa: E402
from simulator.flow_region_congestion import FlowRegionCongestionModel  # noqa: E402

from research_framework.statistics import (  # noqa: E402
    confidence_interval, effect_size_cohens_d, paired_comparison,
)


PUBLISHED_EVACUATION_TIME_S = {
    "10-story": 1022.0,
    "18-story": 1192.0,
    "24-story": 1090.0,
    "31-story": 1002.0,
}

BUILDINGS = {
    "10-story": (build_nist_10story_building, build_nist_10story_definition, DEFINITION_ID_10, MASTER_SEED_10,
                 "scripts.run_nist_10story_validation.build_nist_10story_building"),
    "18-story": (build_nist_18story_building, build_nist_18story_definition, DEFINITION_ID_18, MASTER_SEED_18,
                 "scripts.run_nist_18story_validation.build_nist_18story_building"),
    "24-story": (build_nist_24story_building, build_nist_24story_definition, DEFINITION_ID_24, MASTER_SEED_24,
                 "scripts.run_nist_24story_validation.build_nist_24story_building"),
    "31-story": (build_nist_31story_building, build_nist_31story_definition, DEFINITION_ID_31, MASTER_SEED_31,
                 "scripts.run_nist_31story_validation.build_nist_31story_building"),
}

# Phase 1 grid. Admission Control V8's own inference (docs/architecture/
# capacity_architecture_investigation_v1.md's congestion formula, and
# this session's V8 report) was that Flow Regions now UNDERpredict
# (0.35x-0.71x) -- too FAST, not too slow -- so the congestion floor
# most likely needs to move DOWN from the production default (more
# degradation under crowding, not less). 0.30 (today's default) is kept
# as a grid point so the campaign reports objectively where it ranks,
# the same discipline the walking-speed campaign already applied to 1.2
# m/s.
CONGESTION_FLOOR_GRID = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)

# Phase 2 grid -- identical to the already-published walking-speed
# campaign's own grid, for direct comparability.
WALKING_SPEED_GRID = (0.6, 0.8, 1.0, 1.2, 1.4, 1.6)

N_SCENARIOS = 8
DT = 1.0


# =====================================================
# Shared construction helper -- NOT a production file. Builds a
# FlowRegionCongestionModel whose MINIMUM_SPEED_FACTOR is
# `minimum_speed_factor` on BOTH code paths FlowRegionCongestionModel
# can take (see this file's own module docstring for why the region
# path needs its own workaround). Runtime subclassing only, the same
# mechanism calibration_benchmark.candidates.CongestionMinimumSpeedFactorCandidate
# already uses one wrapper level up.
# =====================================================


def _flow_region_congestion_model(minimum_speed_factor):

    candidate_default_cls = type(
        "CandidateDefaultCongestionModel", (DefaultCongestionModel,),
        {"MINIMUM_SPEED_FACTOR": minimum_speed_factor},
    )

    model = FlowRegionCongestionModel(
        base_model=StairAwareCongestionModel(base_model=candidate_default_cls()),
    )

    # FlowRegionCongestionModel.__init__ hardcodes
    # self._region_model = DefaultCongestionModel() with no constructor
    # parameter to override it -- the FlowRegion (merged) code path,
    # which is the dominant one for these buildings' grouped
    # stairwells/merge-doors, has no other injection point. Reassigning
    # this already-mutable instance attribute from a client script
    # touches no class definition and no method body in
    # simulator/flow_region_congestion.py.
    model._region_model = candidate_default_cls()

    return model


# =====================================================
# Phase 1 candidate -- congestion-degradation curve, Flow Regions fixed
# ON in both arms (this phase's own "keep every other parameter fixed").
# =====================================================


class FlowRegionCongestionFloorCandidate(ParameterCandidate):

    def __init__(self, candidate_minimum_speed_factor, dataset_source, rationale):

        self.candidate_minimum_speed_factor = candidate_minimum_speed_factor

        super().__init__(
            name="FlowRegionCongestionModel.MINIMUM_SPEED_FACTOR (region + edge-fallback path)",
            subsystem="Congestion Model (Flow Region architecture)",
            calibration_tier="Tier 2",
            dataset_source=dataset_source,
            current_value=DefaultCongestionModel.MINIMUM_SPEED_FACTOR,
            candidate_value=candidate_minimum_speed_factor,
            unit="dimensionless speed factor",
            rationale=rationale,
        )

    def baseline_capacity_model(self):
        return FlowRegionCapacityModel()

    def candidate_capacity_model(self):
        return FlowRegionCapacityModel()

    def baseline_congestion_model(self):
        return FlowRegionCongestionModel()

    def candidate_congestion_model(self):
        return _flow_region_congestion_model(self.candidate_minimum_speed_factor)

    def baseline_use_flow_regions(self):
        return True

    def candidate_use_flow_regions(self):
        return True


# =====================================================
# Phase 2 candidate -- walking speed, Flow Regions ON and Phase 1's
# selected congestion floor FIXED IDENTICALLY on both arms.
# =====================================================


class FlowRegionCalibratedWalkingSpeedCandidate(ParameterCandidate):

    def __init__(self, profile_id, candidate_speed, best_minimum_speed_factor, dataset_source, rationale):

        self.profile_id = profile_id
        self.candidate_speed = candidate_speed
        self.best_minimum_speed_factor = best_minimum_speed_factor

        super().__init__(
            name=f"{profile_id}.walking_speed (Flow Region + calibrated congestion baseline)",
            subsystem="Walking Model",
            calibration_tier="Tier 2",
            dataset_source=dataset_source,
            current_value=DEFAULT_PROFILE_REGISTRY[profile_id].walking_speed,
            candidate_value=candidate_speed,
            unit="m/s",
            rationale=rationale,
        )

    def baseline_registry(self):
        return DEFAULT_PROFILE_REGISTRY

    def candidate_registry(self):
        return _registry_with_walking_speed(self.profile_id, self.candidate_speed)

    def baseline_capacity_model(self):
        return FlowRegionCapacityModel()

    def candidate_capacity_model(self):
        return FlowRegionCapacityModel()

    def baseline_congestion_model(self):
        return _flow_region_congestion_model(self.best_minimum_speed_factor)

    def candidate_congestion_model(self):
        return _flow_region_congestion_model(self.best_minimum_speed_factor)

    def baseline_use_flow_regions(self):
        return True

    def candidate_use_flow_regions(self):
        return True


# =====================================================


def _make_benchmark(building_name, geometry_ref):

    return PublishedBenchmark(
        title=f"NIST {building_name} Office Building Fire Drill Recreation",
        source_citation="NIST published evacuation drill recreation (existing SynEvac validation campaign)",
        dataset=f"nist_{building_name.replace('-', '_')}",
        benchmark_type=BenchmarkType.BUILDING_RECREATION,
        geometry_reference=GeometryVersion(version="v1", ref=geometry_ref),
        published_values={"evacuation_time_s": PublishedValue(value=PUBLISHED_EVACUATION_TIME_S[building_name], unit="s")},
    )


def _run_grid_campaign(building_name, dimension_name, build_candidate, grid_values, project_label):

    build_building, build_definition, definition_id, master_seed, geometry_ref = BUILDINGS[building_name]

    building = build_building()
    definition = build_definition()

    studio = CalibrationStudio()
    project = studio.create_project(name=f"{project_label} -- {building_name}")

    benchmark = _make_benchmark(building_name, geometry_ref)
    studio.benchmarks.register(benchmark)

    objective = PublishedValueObjective(
        benchmark, published_metric_name="evacuation_time_s", result_metric_name="evacuation_time",
    )

    search_space = SearchSpace(dimensions=(
        ParameterDimension(name=dimension_name, bounds=(min(grid_values), max(grid_values)), build=build_candidate),
    ))

    strategy = GridSearchStrategy(values={dimension_name: grid_values})

    engine = AutoCalibrationEngine(
        studio=studio, project=project, building=building, definition=definition,
        definition_id=definition_id, master_seed=master_seed, n_scenarios=N_SCENARIOS, dt=DT,
        search_space=search_space, objective=objective, strategy=strategy,
        budget=AutoCalibrationBudget(max_evaluations=len(grid_values)),
    )

    run = engine.run()

    per_value_results = []

    for session_id in run.session_ids:

        session = studio.get_session(session_id)
        comparison = session.result.comparisons["evacuation_time"] if session.result is not None else None
        adoption = recommend(session.result) if session.result is not None else None

        candidate_mean = comparison.candidate_mean if comparison else None
        overprediction_ratio = (
            candidate_mean / PUBLISHED_EVACUATION_TIME_S[building_name] if candidate_mean is not None else None
        )

        per_value_results.append({
            "grid_value": session.candidate_snapshot["candidate_value"],
            "session_status": session.status.value,
            "candidate_mean_evacuation_time": candidate_mean,
            "baseline_mean_evacuation_time": comparison.baseline_mean if comparison else None,
            "overprediction_ratio": overprediction_ratio,
            "candidate_ci": comparison.candidate_ci.to_dict() if comparison else None,
            "p_value_vs_baseline": comparison.paired.p_value if comparison else None,
            "cohens_d_vs_baseline": comparison.effect_size.cohens_d if comparison else None,
            "score_distance_from_published": objective.score(session),
            "recommendation": adoption.overall_verdict.name if adoption else None,
        })

    return {
        "building": building_name,
        "published_evacuation_time_s": PUBLISHED_EVACUATION_TIME_S[building_name],
        "n_scenarios": N_SCENARIOS,
        "run_status": run.status.value,
        "best_session_id": run.best_session_id,
        "best_score": run.best_score,
        "per_value_results": per_value_results,
    }


def run_phase1():

    results = {}

    for building_name in BUILDINGS:

        print(f"[Phase 1] Congestion-floor grid search -- {building_name} "
              f"({len(CONGESTION_FLOOR_GRID)} grid points x {N_SCENARIOS} seeds)...", flush=True)

        results[building_name] = _run_grid_campaign(
            building_name,
            "FlowRegionCongestionModel.MINIMUM_SPEED_FACTOR",
            lambda v: FlowRegionCongestionFloorCandidate(
                v, "automatic-calibration-campaign-v2-phase1",
                "Phase 1: recalibrate the congestion-degradation floor under the Flow Region architecture.",
            ),
            CONGESTION_FLOOR_GRID,
            "Automatic Calibration Campaign V2 Phase 1 (congestion floor)",
        )

    return results


def run_phase2(best_congestion_floor):

    results = {}

    for building_name in BUILDINGS:

        print(f"[Phase 2] Walking-speed grid search -- {building_name} "
              f"({len(WALKING_SPEED_GRID)} grid points x {N_SCENARIOS} seeds, "
              f"congestion floor fixed at {best_congestion_floor})...", flush=True)

        results[building_name] = _run_grid_campaign(
            building_name,
            "Adult_Default.walking_speed",
            lambda v: FlowRegionCalibratedWalkingSpeedCandidate(
                "Adult_Default", v, best_congestion_floor,
                "automatic-calibration-campaign-v2-phase2",
                "Phase 2: recalibrate walking speed on top of Phase 1's selected congestion floor.",
            ),
            WALKING_SPEED_GRID,
            "Automatic Calibration Campaign V2 Phase 2 (walking speed)",
        )

    return results


# =====================================================
# Phase 3 -- straight 4-arm comparison, same pattern as
# scripts/run_flow_region_nist_validation.py /
# scripts/run_admission_control_v7_nist_validation.py: one already-
# generated, seeded scenario batch, run once per arm via
# run_with_overrides() directly.
# =====================================================


ARM_NAMES = (
    "original_simulator",
    "flow_region_default",
    "flow_region_calibrated_congestion",
    "flow_region_calibrated_congestion_and_speed",
)


def _arm_kwargs(arm_name, best_congestion_floor, best_walking_speed):

    if arm_name == "original_simulator":
        return dict(
            registry=None, capacity_model=StairCapacityModel(), congestion_model=StairAwareCongestionModel(),
            use_flow_regions=False,
        )

    if arm_name == "flow_region_default":
        return dict(
            registry=None, capacity_model=FlowRegionCapacityModel(), congestion_model=FlowRegionCongestionModel(),
            use_flow_regions=True,
        )

    if arm_name == "flow_region_calibrated_congestion":
        return dict(
            registry=None, capacity_model=FlowRegionCapacityModel(),
            congestion_model=_flow_region_congestion_model(best_congestion_floor),
            use_flow_regions=True,
        )

    if arm_name == "flow_region_calibrated_congestion_and_speed":
        return dict(
            registry=_registry_with_walking_speed("Adult_Default", best_walking_speed),
            capacity_model=FlowRegionCapacityModel(),
            congestion_model=_flow_region_congestion_model(best_congestion_floor),
            use_flow_regions=True,
        )

    raise ValueError(arm_name)


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def run_phase3(best_congestion_floor, best_walking_speed, n_scenarios=15):

    results = {}

    for building_name in BUILDINGS:

        build_building, build_definition, definition_id, master_seed, _ = BUILDINGS[building_name]

        building = build_building()
        definition = build_definition()

        print(f"[Phase 3] 4-arm comparison -- {building_name} ({n_scenarios} seeds)...", flush=True)

        batch = run_batch_pipeline(definition, definition_id, building, master_seed, n_scenarios)

        per_arm_times = {arm: [] for arm in ARM_NAMES}

        for scenario in batch.scenarios:

            for arm in ARM_NAMES:

                kwargs = _arm_kwargs(arm, best_congestion_floor, best_walking_speed)
                _movement, gt, _building_copy = run_with_overrides(scenario, building, dt=DT, **kwargs)
                per_arm_times[arm].append(gt.total_evacuation_time)

        published = PUBLISHED_EVACUATION_TIME_S[building_name]

        arm_summaries = {}

        for arm in ARM_NAMES:

            times = per_arm_times[arm]
            mean_time = _mean(times)
            ci = confidence_interval([t for t in times if t is not None])

            arm_summaries[arm] = {
                "mean_evacuation_time": mean_time,
                "overprediction_ratio": (mean_time / published) if mean_time is not None else None,
                "ci": ci.to_dict(),
                "residual_error_s": (mean_time - published) if mean_time is not None else None,
            }

        # Paired significance between each successive arm (each step of
        # the "compare against" chain the milestone itself specifies),
        # same paired times/scenarios/seeds on both sides.
        pairwise = {}

        for a, b in zip(ARM_NAMES, ARM_NAMES[1:]):

            a_vals = [v for v in per_arm_times[a] if v is not None]
            b_vals = [v for v in per_arm_times[b] if v is not None]
            n = min(len(a_vals), len(b_vals))

            paired = paired_comparison(a_vals[:n], b_vals[:n]) if n >= 2 else None
            effect = effect_size_cohens_d(b_vals[:n], a_vals[:n]) if n >= 2 else None

            pairwise[f"{a}_vs_{b}"] = {
                "paired": paired.to_dict() if paired else None,
                "effect_size_cohens_d": effect.to_dict() if effect else None,
            }

        results[building_name] = {
            "building": building_name,
            "published_evacuation_time_s": published,
            "n_scenarios": n_scenarios,
            "arms": arm_summaries,
            "pairwise_significance": pairwise,
        }

    return results


# =====================================================


def _select_best_aggregate(phase_results, grid_values):

    # Aggregation policy for turning 4 independent per-building grid
    # searches (each already using AutoCalibrationEngine/
    # PublishedValueObjective completely unmodified) into ONE global
    # constant to carry forward -- exactly the same "one Adult_Default.
    # walking_speed value used by every building" shape this repo's own
    # production defaults already have. Raw |candidate_mean - published|
    # (the engine's OWN per-building objective) cannot be summed
    # directly across buildings of very different absolute scale (an
    # 18-story building's raw-second gap dwarfs 10-story's) without the
    # sum being dominated by whichever building has the largest
    # buildings/occupant count. Minimizing the SUM of |log(ratio)|
    # instead weighs every building's relative (multiplicative)
    # closeness to its own published value equally -- consistent with
    # how every overprediction ratio in this whole investigation
    # (V5 through V8) has itself always been reported multiplicatively,
    # never as a raw-second difference.

    import math

    best_value, best_total = None, None

    for value in grid_values:

        total = 0.0
        count = 0

        for building_name, result in phase_results.items():

            match = next(
                (r for r in result["per_value_results"] if r["grid_value"] == value), None,
            )

            if match is None or match["overprediction_ratio"] is None or match["overprediction_ratio"] <= 0:
                continue

            total += abs(math.log(match["overprediction_ratio"]))
            count += 1

        if count == 0:
            continue

        if best_total is None or total < best_total:
            best_value, best_total = value, total

    return best_value


def main():

    n_scenarios_phase3 = int(sys.argv[1]) if len(sys.argv) > 1 else 15

    output_dir = os.path.join(os.path.dirname(__file__), "..", "docs", "architecture")
    os.makedirs(output_dir, exist_ok=True)

    print("=== Phase 1: congestion-degradation curve ===", flush=True)
    phase1_results = run_phase1()
    best_congestion_floor = _select_best_aggregate(phase1_results, CONGESTION_FLOOR_GRID)
    print(f"[Phase 1] Selected aggregate-best MINIMUM_SPEED_FACTOR = {best_congestion_floor}", flush=True)

    with open(os.path.join(output_dir, "automatic_calibration_campaign_v2_phase1_raw_results.json"), "w", encoding="utf-8") as handle:
        json.dump({"per_building": phase1_results, "selected_congestion_floor": best_congestion_floor}, handle, indent=2, default=str)

    print("=== Phase 2: walking speed (congestion floor fixed) ===", flush=True)
    phase2_results = run_phase2(best_congestion_floor)
    best_walking_speed = _select_best_aggregate(phase2_results, WALKING_SPEED_GRID)
    print(f"[Phase 2] Selected aggregate-best walking_speed = {best_walking_speed}", flush=True)

    with open(os.path.join(output_dir, "automatic_calibration_campaign_v2_phase2_raw_results.json"), "w", encoding="utf-8") as handle:
        json.dump({"per_building": phase2_results, "selected_walking_speed": best_walking_speed}, handle, indent=2, default=str)

    print("=== Phase 3: four-arm NIST comparison ===", flush=True)
    phase3_results = run_phase3(best_congestion_floor, best_walking_speed, n_scenarios=n_scenarios_phase3)

    with open(os.path.join(output_dir, "automatic_calibration_campaign_v2_phase3_raw_results.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "selected_congestion_floor": best_congestion_floor,
                "selected_walking_speed": best_walking_speed,
                "per_building": phase3_results,
            },
            handle, indent=2, default=str,
        )

    print(json.dumps({
        "selected_congestion_floor": best_congestion_floor,
        "selected_walking_speed": best_walking_speed,
    }, indent=2))


if __name__ == "__main__":
    main()
