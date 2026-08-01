import hashlib
import json
import os

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

from ai_decision.engine import AIDecisionEngine

from behaviour_profile_resolver.combined_registrar import register_population
from behaviour_profile_resolver.dynamic_registrar import DynamicRegistrationResult, register_population_dynamic
from behaviour_profile_resolver.registrar import register_occupants

from human_decision_engine.events import DecisionEventLog
from human_decision_engine.firefighter_engine import RESCUE_TASKS
from human_decision_engine.groups import compute_group_dissolution, compute_rescue_completion_events

from dataset_builder.builder import DatasetBuilder, SimulationRun
from dataset_builder.behavior_events import BehaviorEventsRun, extract_behavior_event_rows
from dataset_builder.exporters import export_csv
from dataset_builder.timeline import TimelineRun, extract_timeline_rows

from decision_policy import DecisionInputs, generate_policy

from ground_truth import SimulationArtifacts
from ground_truth import analyze as analyze_ground_truth

from scenario_generator import derive_scenario_seed
from scenario_pipeline import DEFAULT_MAX_ATTEMPTS, run_pipeline
from scenario_runner import run as run_scenario
from scenario_storage import load_accepted_hashes, save_scenario
from scenario_validator import compute_candidate_content_hash

from simulation_runtime import SimulationRuntime

from simulation_recording.decision_events import save_decision_events
from simulation_recording.occupant_routes import build_occupant_route_records, save_occupant_routes

from training_dataset import CampaignDataset, load_campaign

from ai_training.dataset import load_campaign_dataset
from ai_training.experiment import ExperimentConfig, ExperimentResult, ExperimentRunner

from rl_training import EvaluationReport, RLTrainer, SynEvacGymEnv, TrainerConfig, evaluate_policy

from research_framework.experiment_spec import Arm, ExperimentSpec


# =====================================================
# Phase 2 -- Automated Campaign Runner. Mirrors designer.campaign.
# campaign_worker.CampaignWorker's own generate -> simulate -> export
# sequence exactly (same packages, same call order, same on-disk
# campaign layout training_dataset.load_campaign() already reads --
# see that worker's own _simulate()/_export_scenario_artifacts()) --
# restated here rather than reused directly for the one seam
# CampaignWorker has no hook for: an injectable registration strategy
# (Arm.registration), needed so an arm can compare "civilians only" /
# "civilians + scenario-authored firefighter rescue" / "civilians +
# dynamic Human Decision Engine" without three different Definitions.
# This is the same "restate an orchestration sequence, never
# reimplement the logic it calls" discipline CampaignWorker's own
# module docstring already establishes for its own restatements of
# scenario_pipeline's attempt loop.
#
# No simulator/architecture change: every call below is to an already-
# public, already-frozen entry point of its own package.
# =====================================================

_REGISTRATION_FUNCTIONS = {
    "civilian": lambda context: register_occupants(context),
    "population": lambda context: register_population(context),
    "dynamic": lambda context: register_population_dynamic(context),
}


def _derive_arm_seed(master_seed: int, arm_name: str) -> int:

    # Same sha256-of-a-stable-key convention every other package in
    # this codebase restates independently (registrar._derive_occupant_seed,
    # ground_truth.human_behavior._derive_behavior_seed, ...) -- one
    # arm's own scenario/RL seeds are fully independent of any other
    # arm's, so two arms never accidentally share a random draw.

    key = f"{master_seed}|research_framework|{arm_name}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()

    return int.from_bytes(digest[:8], "big")


# =====================================================


@dataclass(frozen=True)
class RunOutcome:

    scenario_id: Optional[str]
    accepted: bool
    total_evacuation_time: Optional[float] = None
    rejection_reason: Optional[str] = None


@dataclass(frozen=True)
class ArmRunResult:

    arm: Arm
    campaign_dir: str
    outcomes: Tuple[RunOutcome, ...] = field(default_factory=tuple)
    ai_results: Dict[str, ExperimentResult] = field(default_factory=dict)
    ai_errors: Dict[str, str] = field(default_factory=dict)

    @property
    def accepted_count(self) -> int:

        return sum(1 for outcome in self.outcomes if outcome.accepted)

    def load_dataset(self) -> CampaignDataset:

        return load_campaign(self.campaign_dir, strict=False)

    def evacuation_times(self) -> Tuple[float, ...]:

        return tuple(
            outcome.total_evacuation_time for outcome in self.outcomes
            if outcome.accepted and outcome.total_evacuation_time is not None
        )


@dataclass(frozen=True)
class ExperimentRunResult:

    spec: ExperimentSpec
    arm_results: Tuple[ArmRunResult, ...]

    def arm_result(self, arm_name: str) -> ArmRunResult:

        for result in self.arm_results:
            if result.arm.name == arm_name:
                return result

        raise KeyError(f"No arm named {arm_name!r} in this experiment result.")


# =====================================================


def run_scenario_artifacts(scenario, building, output_directory: str, *, dt: float = 5.0,
                            registration: str = "population") -> Optional[float]:

    # One scenario, start to finish: registration -> SimulationRuntime
    # -> Dataset Builder V1/V2 -> Ground Truth -> Decision Policy, all
    # written to output_directory/{datasets,timelines,ground_truth,
    # decision_policy}/<scenario_id>/ -- byte-for-byte the same layout
    # CampaignWorker._export_scenario_artifacts() writes, so any
    # resulting campaign_dir is directly loadable by training_dataset.
    # load_campaign()/campaign_analytics.analyze_campaign().

    if registration not in _REGISTRATION_FUNCTIONS:

        raise ValueError(
            f"Unknown registration strategy {registration!r} -- choose one of "
            f"{sorted(_REGISTRATION_FUNCTIONS)}",
        )

    context = run_scenario(scenario, building)
    registration_result = _REGISTRATION_FUNCTIONS[registration](context)

    # Production Pipeline Completion -- registration_result's return
    # value used to be discarded outright. "civilian"/"population" still
    # return a plain HumanBehaviorLayer (unchanged); only "dynamic"
    # returns a DynamicRegistrationResult, the one shape that also
    # carries decision events/groups/per-occupant task decisions. Both
    # cases now feed the same post-processing below -- Behavior Events
    # export works for every registration arm (BehaviorDecision exists
    # regardless of which registrar built it); Decision Events/rescue
    # completion are only ever non-empty for the "dynamic" arm, exactly
    # as human_decision_engine/ground_truth's own "empty is honest, only
    # dynamic registration ever produces these" convention already
    # documents.
    if isinstance(registration_result, DynamicRegistrationResult):
        behavior_layer = registration_result.behavior_layer
    else:
        behavior_layer = registration_result

    decision_engine = AIDecisionEngine(base_engine=context.engine)
    runtime = SimulationRuntime(context, decision_engine, dt=dt, perception_provider=None)

    tick_results = runtime.run()

    scenario_id = scenario.metadata.scenario_id
    movement_result = runtime.movement_result

    decision_events: Tuple[Dict[str, Any], ...] = ()

    if isinstance(registration_result, DynamicRegistrationResult):

        completion_log = DecisionEventLog()

        rescue_target_by_firefighter_id = {
            firefighter.firefighter_id: decision.target_occupant_id
            for firefighter in scenario.firefighters
            for decision in [registration_result.decisions_by_occupant_id.get(firefighter.firefighter_id)]
            if decision is not None
            and getattr(decision, "task", None) in RESCUE_TASKS
            and decision.target_occupant_id is not None
        }
        compute_rescue_completion_events(
            rescue_target_by_firefighter_id, movement_result, completion_log,
        )

        compute_group_dissolution(
            registration_result.groups, movement_result, event_log=completion_log,
        )

        decision_events = tuple(
            event.to_dict() for event in (registration_result.events + completion_log.events)
        )

    simulation_run = SimulationRun(
        scenario=scenario, building=building, movement_result=movement_result,
        simulation_finished=runtime.is_finished,
        tick_results=tick_results, decision_events=decision_events,
    )
    dataset_dir = os.path.join(output_directory, "datasets", scenario_id)
    DatasetBuilder([simulation_run]).export_all(dataset_dir)

    behavior_events_run = BehaviorEventsRun(
        scenario=scenario, behavior_decisions=tuple(behavior_layer._decisions_so_far.values()),
    )
    behavior_event_rows = extract_behavior_event_rows(behavior_events_run)

    behavior_events_dir = os.path.join(output_directory, "behavior_events", scenario_id)
    os.makedirs(behavior_events_dir, exist_ok=True)
    export_csv(behavior_event_rows, os.path.join(behavior_events_dir, "behavior_events.csv"))

    timeline_run = TimelineRun(
        scenario=scenario, building=building, movement_result=movement_result, tick_results=tick_results,
    )
    timeline_rows = extract_timeline_rows(timeline_run)

    timeline_dir = os.path.join(output_directory, "timelines", scenario_id)
    os.makedirs(timeline_dir, exist_ok=True)
    export_csv(timeline_rows, os.path.join(timeline_dir, "timeline.csv"))

    # Simulation Replay Studio V1 end-to-end verification, Verification
    # Task 22 -- discovered that this function never wrote the JSON form
    # of the timeline rows, only the CSV. designer/campaign/
    # campaign_worker.py writes both, specifically because
    # command_center.incident_data.load_incident()'s own
    # timeline_rows_path parameter expects a JSON list of row dicts, not
    # a CSV -- without it, a scenario produced by THIS function (the one
    # used for large-scale research campaigns, as opposed to Designer's
    # own interactive Campaign Studio) could never be timeline-scrubbed
    # in Replay Studio at all, silently falling back to a single static
    # baseline frame. A pre-existing gap in this function, not something
    # this milestone's own occupant_routes/decision_events wiring
    # introduced -- fixed here since it directly blocks this milestone's
    # own "any scenario generated by SynEvac can be visually inspected"
    # goal for this producer.
    with open(os.path.join(timeline_dir, "timeline_rows.json"), "w", encoding="utf-8") as handle:
        json.dump(timeline_rows, handle)

    # ---- Simulation Recording -- occupant routes + decision events.
    # Same two artifacts designer/campaign/campaign_worker.py writes,
    # keeping this producer's own "byte-for-byte the same layout" claim
    # (this function's own docstring) true. Neither recomputes anything:
    # occupant routes are extracted straight off this scenario's own
    # already-completed movement_result, and decision_events is the
    # exact same tuple already built above.

    occupant_route_records = build_occupant_route_records(movement_result)

    occupant_routes_dir = os.path.join(output_directory, "occupant_routes", scenario_id)
    os.makedirs(occupant_routes_dir, exist_ok=True)
    save_occupant_routes(
        occupant_route_records, os.path.join(occupant_routes_dir, "occupant_routes.json"),
    )

    decision_events_dir = os.path.join(output_directory, "decision_events", scenario_id)
    os.makedirs(decision_events_dir, exist_ok=True)
    save_decision_events(
        decision_events, os.path.join(decision_events_dir, "decision_events.json"),
    )

    ground_truth = analyze_ground_truth(
        SimulationArtifacts(
            scenario=scenario, building=building, movement_result=movement_result,
            tick_results=tick_results, decision_events=decision_events,
        )
    )
    ground_truth_dir = os.path.join(output_directory, "ground_truth", scenario_id)
    os.makedirs(ground_truth_dir, exist_ok=True)
    with open(os.path.join(ground_truth_dir, "ground_truth.json"), "w", encoding="utf-8") as handle:
        json.dump(ground_truth.to_dict(), handle)

    decision_policy = generate_policy(
        DecisionInputs(
            building=building, scenario=scenario, ground_truth=ground_truth,
            timeline_rows=tuple(timeline_rows),
        )
    )
    decision_policy_dir = os.path.join(output_directory, "decision_policy", scenario_id)
    os.makedirs(decision_policy_dir, exist_ok=True)
    with open(os.path.join(decision_policy_dir, "decision_policy.json"), "w", encoding="utf-8") as handle:
        json.dump(decision_policy.to_dict(), handle)

    return movement_result.total_evacuation_time


# =====================================================


def run_arm(spec: ExperimentSpec, arm: Arm, *, max_attempts: int = DEFAULT_MAX_ATTEMPTS) -> ArmRunResult:

    definition = arm.resolve_definition(spec.base_definition)
    building = spec.building_for(arm)
    campaign_dir = os.path.join(spec.output_root, arm.name)

    arm_master_seed = _derive_arm_seed(spec.master_seed, arm.name)

    accepted_hashes = set(load_accepted_hashes(campaign_dir))
    outcomes = []

    for index in range(spec.samples_per_arm):

        scenario_seed = derive_scenario_seed(arm_master_seed, index)

        result = run_pipeline(
            definition, spec.base_definition_id, building, scenario_seed,
            max_attempts=max_attempts, accepted_hashes=frozenset(accepted_hashes),
        )

        if not result.accepted:

            outcomes.append(RunOutcome(scenario_id=None, accepted=False, rejection_reason=result.failure_reason))
            continue

        scenario = result.scenario

        try:
            save_scenario(scenario, campaign_dir)
        except FileExistsError:
            pass

        accepted_hashes.add(compute_candidate_content_hash(scenario))

        total_evacuation_time = run_scenario_artifacts(
            scenario, building, campaign_dir, dt=spec.dt, registration=arm.registration,
        )

        outcomes.append(
            RunOutcome(
                scenario_id=scenario.metadata.scenario_id, accepted=True,
                total_evacuation_time=total_evacuation_time,
            )
        )

    return ArmRunResult(arm=arm, campaign_dir=campaign_dir, outcomes=tuple(outcomes))


# =====================================================


def run_experiment(
    spec: ExperimentSpec, *,
    ai_experiment_configs: Sequence[ExperimentConfig] = (),
) -> ExperimentRunResult:

    # "Generate datasets, train AI, ... repeat automatically" (Phase 2's
    # own objective) -- one call per arm, each arm producing a fully
    # self-contained campaign directory plus (if ai_experiment_configs
    # is non-empty) one trained model per config, per arm. RL training/
    # evaluation is deliberately NOT run automatically here -- it needs
    # a SynEvacGymEnv per arm (a materially heavier setup than a
    # DatasetBuilder export) and is better driven explicitly per arm via
    # research_framework.runner.train_and_evaluate_rl_for_arm() below,
    # exactly the seam research_framework.ablation's own "No RL"
    # comparison and research_framework.sensitivity's own "RL Reward"
    # sweeps already use.

    arm_results = []

    for arm in spec.arms:

        arm_result = run_arm(spec, arm)

        if ai_experiment_configs:

            arm_result = _train_ai_for_arm(arm_result, ai_experiment_configs)

        arm_results.append(arm_result)

    return ExperimentRunResult(spec=spec, arm_results=tuple(arm_results))


# =====================================================


def _train_ai_for_arm(
    arm_result: ArmRunResult, ai_experiment_configs: Sequence[ExperimentConfig],
) -> ArmRunResult:

    # Failures (e.g. too few accepted scenarios in this arm to split
    # train/test) are recorded in ai_errors rather than raised -- one
    # arm's dataset being too small to train a given model must never
    # abort the whole experiment; the honest result is "this model
    # could not be trained for this arm," not a crash.

    if arm_result.accepted_count == 0:
        return arm_result

    try:
        dataset = load_campaign_dataset(arm_result.campaign_dir, strict=False)
    except Exception as exc:
        return arm_result.__class__(
            arm=arm_result.arm, campaign_dir=arm_result.campaign_dir, outcomes=arm_result.outcomes,
            ai_errors={"__load__": str(exc)},
        )

    runner = ExperimentRunner()
    ai_results: Dict[str, ExperimentResult] = {}
    ai_errors: Dict[str, str] = {}

    for config in ai_experiment_configs:

        try:
            ai_results[config.name] = runner.run(dataset, config)
        except Exception as exc:
            ai_errors[config.name] = str(exc)

    return ArmRunResult(
        arm=arm_result.arm, campaign_dir=arm_result.campaign_dir, outcomes=arm_result.outcomes,
        ai_results=ai_results, ai_errors=ai_errors,
    )


# =====================================================
# RL training/evaluation for one arm -- not invoked automatically by
# run_experiment() (see that function's own docstring for why); called
# explicitly by a caller (or by research_framework.ablation's "No RL"
# comparison / research_framework.sensitivity's "RL Reward" sweeps)
# once per arm it actually needs an RL policy for. Builds one
# SynEvacGymEnv directly from the arm's own resolved ScenarioDefinition
# and Building -- gymnasium.Env.reset() generates a fresh Scenario
# every episode from that Definition (rl_training.environment's own
# established behavior), so no scenario_pipeline/campaign_dir is
# involved here at all.
# =====================================================


def train_and_evaluate_rl_for_arm(
    spec: ExperimentSpec, arm: Arm, *,
    total_timesteps: int = 2000, eval_episode_count: int = 5,
    trainer_config: Optional[TrainerConfig] = None, max_steps: int = 200, dt: float = 1.0,
) -> EvaluationReport:

    definition = arm.resolve_definition(spec.base_definition)
    building = spec.building_for(arm)
    arm_seed = _derive_arm_seed(spec.master_seed, arm.name)

    env = SynEvacGymEnv(
        building=building, definition=definition, definition_id=spec.base_definition_id,
        master_seed=arm_seed, dt=dt, max_steps=max_steps,
    )

    trainer = RLTrainer(env, trainer_config or TrainerConfig())
    trainer.train(total_timesteps)

    # RLTrainer.predict() returns stable-baselines3's own (action, state)
    # pair (see tests.test_rl_training's own "action, _state =
    # trainer.predict(observation)" convention) -- evaluate_policy()'s
    # predict_fn contract is "observation -> action" alone.
    def _predict_action(observation):

        action, _state = trainer.predict(observation)
        return action

    return evaluate_policy(
        env, _predict_action, master_seed=arm_seed, count=eval_episode_count, label=arm.name,
    )
