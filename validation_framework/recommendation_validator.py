from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from ai_decision.engine import AIDecisionEngine

from behaviour_profile_resolver import register_occupants

from scenario_generator import derive_scenario_seed
from scenario_pipeline import run_pipeline
from scenario_runner import run as run_scenario

from simulation_runtime import SimulationRuntime

from ground_truth.analyzer import SimulationArtifacts, analyze

from dataset_builder.schema import ordered_zones
from dataset_builder.timeline import TimelineRun, extract_timeline_rows
from dataset_builder.timeline_schema import zone_smoke_columns

from decision_policy.policy import DecisionInputs, generate_policy

from advisory_system.orchestrator import AdvisoryOrchestrator
from advisory_system.recommendation_history import RecommendationHistory
from advisory_system.recommendation_models import AdvisoryInputs

from ai_explainability.comparison import decision_policy_flagged


# =====================================================
# Phase 4 -- Recommendation Validation. Runs its own fresh scenarios
# (Scenario/Building objects aren't preserved in exported campaign
# artifacts, so this can't replay a campaign_dir the way prediction_
# validator.py does) through the real production pipeline
# (scenario_pipeline -> scenario_runner -> SimulationRuntime ->
# ground_truth.analyze -> decision_policy.generate_policy), then drives
# one AdvisoryOrchestrator across the timeline tick-by-tick, exactly the
# "successive snapshots of the same scenario over simulation time"
# usage AdvisoryOrchestrator's own docstring describes.
#
# false_redirects is honestly always empty: detecting a redirect that
# later proves unsafe requires a decision_policy recomputed at a LATER
# tick than the one that issued the redirect, but decision_policy is
# (by rl_training.evaluator's own documented finding) a purely post-hoc
# computation over one already-finished GroundTruth -- this module
# computes exactly one decision_policy per scenario, matching that same
# established convention, so there is no later snapshot to compare
# against. Never fabricated as a guessed non-empty result.
# =====================================================

DEFAULT_LATE_REDIRECT_THRESHOLD_SECONDS = 30.0
DEFAULT_SMOKE_HAZARD_THRESHOLD = 0.3


@dataclass(frozen=True)
class RecommendationValidationResult:

    scenarios_evaluated: int
    total_ticks: int
    total_changes: int
    changes_per_zone: Dict[str, int]
    stability_score: float
    consistency_violations: Tuple[Dict[str, Any], ...]
    false_redirects: Tuple[Dict[str, Any], ...]
    late_redirects: Tuple[Dict[str, Any], ...]
    firefighter_priority_deterministic: Optional[bool]

    # First evaluated scenario's own timeline rows / recommendation
    # changes, retained purely so figures.py has a concrete worked
    # example to plot (evacuation curve, occupancy heatmap,
    # recommendation timeline) -- not part of the aggregate statistics
    # above, which are computed across every scenario.
    sample_timeline_rows: Tuple[Dict[str, Any], ...] = ()
    sample_changes: Tuple[Any, ...] = ()

    def to_dict(self) -> Dict[str, Any]:

        return {
            "scenarios_evaluated": self.scenarios_evaluated,
            "total_ticks": self.total_ticks,
            "total_changes": self.total_changes,
            "changes_per_zone": self.changes_per_zone,
            "stability_score": self.stability_score,
            "consistency_violations": list(self.consistency_violations),
            "false_redirects": list(self.false_redirects),
            "late_redirects": list(self.late_redirects),
            "firefighter_priority_deterministic": self.firefighter_priority_deterministic,
        }


def validate_recommendations(
    building,
    definition,
    definition_id: str,
    master_seed: int,
    scenario_count: int,
    *,
    dt: float = 5.0,
    max_steps: int = 500,
    late_redirect_threshold_seconds: float = DEFAULT_LATE_REDIRECT_THRESHOLD_SECONDS,
    smoke_hazard_threshold: float = DEFAULT_SMOKE_HAZARD_THRESHOLD,
) -> RecommendationValidationResult:

    zones = ordered_zones(building)
    smoke_column_by_zone = dict(zip((zone.id for zone in zones), zone_smoke_columns(building)))

    scenarios_evaluated = 0
    total_ticks = 0
    total_changes = 0
    changes_per_zone: Dict[str, int] = {}
    consistency_violations = []
    late_redirects = []
    firefighter_determinism_checks = 0
    firefighter_determinism_failures = 0
    sample_timeline_rows: Tuple[Dict[str, Any], ...] = ()
    sample_changes: Tuple[Any, ...] = ()

    for index in range(scenario_count):

        scenario_seed = derive_scenario_seed(master_seed, index)
        pipeline_result = run_pipeline(definition, definition_id, building, scenario_seed)

        if not pipeline_result.accepted:
            continue

        scenario = pipeline_result.scenario
        scenarios_evaluated += 1

        context = run_scenario(scenario, building)
        register_occupants(context)
        engine = AIDecisionEngine(base_engine=context.engine)
        runtime = SimulationRuntime(context, engine, dt=dt)
        tick_results = runtime.run()

        if len(tick_results) >= max_steps:
            tick_results = tick_results[:max_steps]

        ground_truth = analyze(
            SimulationArtifacts(
                scenario=scenario, building=building,
                movement_result=runtime.movement_result, tick_results=tick_results,
            )
        )
        timeline_rows = extract_timeline_rows(
            TimelineRun(
                scenario=scenario, building=building,
                movement_result=runtime.movement_result, tick_results=tick_results,
            )
        )

        decision_policy = generate_policy(
            DecisionInputs(
                building=building, scenario=scenario, ground_truth=ground_truth,
                timeline_rows=tuple(timeline_rows),
            )
        )

        # Firefighter Intelligence Consistency (determinism half -- see
        # prediction_validator.validate_firefighter_intelligence_consistency's
        # docstring for the monotonicity half and why this check lives
        # here instead, where live Building/Scenario objects are in hand).
        repeat_policy = generate_policy(
            DecisionInputs(
                building=building, scenario=scenario, ground_truth=ground_truth,
                timeline_rows=tuple(timeline_rows),
            )
        )
        firefighter_determinism_checks += 1
        if repeat_policy.firefighter_deployment_priority != decision_policy.firefighter_deployment_priority:
            firefighter_determinism_failures += 1

        history = RecommendationHistory()
        orchestrator = AdvisoryOrchestrator(history)

        first_hazard_tick_by_zone: Dict[str, float] = {}

        for tick_index, row in enumerate(timeline_rows):

            total_ticks += 1
            simulation_time = row.get("simulation_time") or 0.0

            for zone in zones:

                column = smoke_column_by_zone.get(zone.id)
                smoke_level = row.get(column) if column else None

                if (
                    zone.id not in first_hazard_tick_by_zone
                    and smoke_level is not None
                    and smoke_level >= smoke_hazard_threshold
                ):
                    first_hazard_tick_by_zone[zone.id] = simulation_time

            orchestrator.generate_report(
                AdvisoryInputs(
                    building=building, scenario=scenario, ground_truth=ground_truth,
                    decision_policy=decision_policy, timeline_rows=tuple(timeline_rows[: tick_index + 1]),
                    simulation_time=simulation_time,
                )
            )

        if not sample_timeline_rows:
            sample_timeline_rows = tuple(timeline_rows)
            sample_changes = history.changes

        for change in history.changes:

            total_changes += 1
            changes_per_zone[change.zone_id] = changes_per_zone.get(change.zone_id, 0) + 1

            reason = (change.reason_for_change or "").lower()

            if "redirect" in reason:

                hazard_onset_time = first_hazard_tick_by_zone.get(change.zone_id)

                if (
                    hazard_onset_time is not None
                    and (change.timestamp - hazard_onset_time) > late_redirect_threshold_seconds
                ):
                    late_redirects.append({
                        "scenario_id": scenario.metadata.scenario_id,
                        "zone_id": change.zone_id,
                        "redirect_time": change.timestamp,
                        "hazard_onset_time": hazard_onset_time,
                        "delay_seconds": change.timestamp - hazard_onset_time,
                    })

        decision_policy_dict = decision_policy.to_dict()

        for zone_decision in decision_policy.zone_decisions:

            zone_id = zone_decision.get("zone_id")
            recommended_exit = zone_decision.get("recommended_exit")
            recommended_stair = zone_decision.get("recommended_stair")

            if recommended_exit and decision_policy_flagged(decision_policy_dict, recommended_exit, "exit"):
                consistency_violations.append({
                    "scenario_id": scenario.metadata.scenario_id, "zone_id": zone_id,
                    "target_type": "exit", "target_id": recommended_exit,
                })

            if recommended_stair and decision_policy_flagged(decision_policy_dict, recommended_stair, "stair"):
                consistency_violations.append({
                    "scenario_id": scenario.metadata.scenario_id, "zone_id": zone_id,
                    "target_type": "stair", "target_id": recommended_stair,
                })

    stability_score = 1.0 - (total_changes / max(1, total_ticks * max(1, len(zones))))

    firefighter_priority_deterministic = (
        (firefighter_determinism_failures == 0) if firefighter_determinism_checks > 0 else None
    )

    return RecommendationValidationResult(
        scenarios_evaluated=scenarios_evaluated,
        total_ticks=total_ticks,
        total_changes=total_changes,
        changes_per_zone=changes_per_zone,
        stability_score=max(0.0, stability_score),
        consistency_violations=tuple(consistency_violations),
        false_redirects=(),
        late_redirects=tuple(late_redirects),
        firefighter_priority_deterministic=firefighter_priority_deterministic,
        sample_timeline_rows=sample_timeline_rows,
        sample_changes=sample_changes,
    )
