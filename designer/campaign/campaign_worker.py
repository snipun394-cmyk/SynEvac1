import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import FrozenSet, Optional

from PyQt6.QtCore import QThread, pyqtSignal

from models.building import Building
from models.project import Project

from scenario_definition import ScenarioDefinition
from scenario_generator import GenerationRequest, derive_scenario_seed, generate_scenario
from scenario_pipeline import DEFAULT_MAX_ATTEMPTS, run_pipeline
from scenario_storage import load_accepted_hashes, save_scenario
from scenario_validator import compute_candidate_content_hash, validate

from scenario_runner import run as run_scenario
from behaviour_profile_resolver.dynamic_registrar import register_population_dynamic
from ai_decision.engine import AIDecisionEngine
from simulation_runtime import SimulationRuntime

from human_decision_engine.events import DecisionEventLog
from human_decision_engine.firefighter_engine import RESCUE_TASKS
from human_decision_engine.groups import compute_group_dissolution, compute_rescue_completion_events

from dataset_builder.builder import DatasetBuilder, SimulationRun
from dataset_builder.behavior_events import BehaviorEventsRun, extract_behavior_event_rows
from dataset_builder.exporters import export_csv
from dataset_builder.timeline import TimelineRun, extract_timeline_rows

from ground_truth import SimulationArtifacts
from ground_truth import analyze as analyze_ground_truth

from decision_policy import DecisionInputs, generate_policy

from simulation_recording.decision_events import save_decision_events
from simulation_recording.occupant_routes import build_occupant_route_records, save_occupant_routes

from serialization.serializer import Serializer

from designer.validation import validate_building_authoring

from designer.campaign.campaign_summary import CampaignSummary
from designer.campaign.diagnostics import (
    DiagnosticsCollector,
    PreflightResult,
    ValidationRow,
    explain_total_rejection,
)
from designer.campaign.progress_model import ProgressModel


# The Campaign Studio's only orchestration logic -- calls exactly the
# already-frozen public APIs of scenario_generator/scenario_pipeline/
# scenario_storage/scenario_runner/behaviour_profile_resolver/
# ai_decision/simulation_runtime, in the exact sequence
# docs/architecture pipeline diagrams already fix, and nothing else.
# No generation, validation, simulation, or decision logic is
# reimplemented here -- the only new code is the per-scenario loop
# itself (index -> seed -> run_pipeline(), mirroring scenario_pipeline.
# run_batch_pipeline()'s own internal loop shape, restated here only
# because run_batch_pipeline() has no progress-callback seam to hook
# into) and the wall-clock bookkeeping needed for live progress
# display.
#
# Diagnostics Mode (docs/architecture -- root-cause report accompanying
# this phase) extends this with one more restated-not-reimplemented
# loop: _generate_with_diagnostics() mirrors run_pipeline()'s own
# attempt loop (scenario_pipeline/pipeline.py) exactly -- generate_
# scenario() then validate(), attempt by attempt -- using only those
# two already-public functions, for the same reason the outer
# per-scenario loop already needed restating: run_pipeline() itself
# only ever returns PipelineResult.last_report (the final attempt's
# report), discarding every earlier attempt's ScenarioValidationReport,
# which is by design for that package's own scope but is exactly the
# visibility Requirement 1 needs. This is calling scenario_validator
# more often, not differently -- validate()'s own logic is never
# touched.
#
# perception_provider is deliberately never configured (stays None) --
# docs/architecture/simulation_runtime.md §13: no concrete
# PerceptionProvider composition exists anywhere in this codebase yet.
# SimulationRuntime already treats this as a fully supported
# configuration (Perception is reached, and correctly no-ops), so this
# is calling the existing pipeline honestly, not skipping a stage.
#
# Dataset Intent / Scenario Outcome Labels: neither dataset_intent/ nor
# scenario_outcome/ exists as code (both are architecture-only designs
# from prior phases, and neither is in this phase's own list of
# production-ready packages). CampaignConfig.definition is therefore
# always a directly-authored ScenarioDefinition (CampaignWindow's own
# "Custom" translation, never a resolve_intent() call that does not
# exist), and the only per-scenario summary statistic computed here
# (average evacuation time) is read directly off
# SimulationRuntime.movement_result -- an already-public field of an
# already-frozen package -- rather than calling a record_outcome()
# that does not exist either.
#
# Post-processing (added alongside the pipeline wiring above, same
# "call already-frozen public APIs, nothing reimplemented" rule):
# _export_scenario_artifacts() is the one place per-scenario Runtime
# output (movement_result, and the Tuple[TickResult, ...]
# SimulationRuntime.run() already returns but the pre-existing
# _simulate() used to discard) is handed to dataset_builder/
# ground_truth/decision_policy -- each already its own frozen,
# independently-tested package. No simulation is rerun to produce any
# of this: every artifact is derived from the exact movement_result/
# tick_results this scenario's own SimulationRuntime.run() call already
# produced. See _export_scenario_artifacts() for the exact sequence and
# _ensure_building_project_saved() for the one campaign-wide (not
# per-scenario) artifact, a saved .syn Project -- written once since
# CampaignConfig.building is the same object for every scenario in one
# campaign, needed only so command_center.incident_data.load_incident()
# has a Building to load back for Command Center playback.


@dataclass
class CampaignConfig:

    name: str
    building: Building
    definition: ScenarioDefinition
    definition_id: str

    count: int
    master_seed: int
    output_directory: str

    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    dt: float = 5.0

    accepted_hashes: FrozenSet[str] = field(default_factory=frozenset)

    # Requirements 1-7 -- gated behind this flag so a campaign that
    # doesn't need it keeps using exactly the pre-diagnostics fast path
    # (a single run_pipeline() call per scenario, no per-attempt report
    # capture). Defaults on: the overhead is negligible (§ root-cause
    # report -- DiagnosticsCollector stores only bounded, aggregate
    # data, never one entry per candidate), and this is precisely the
    # capability needed to diagnose exactly the kind of failure that
    # motivated this phase.
    diagnostics_mode: bool = True


class CampaignWorker(QThread):

    # QThread subclass, not QObject + moveToThread -- the simplest
    # correct shape for a single, self-contained, run-to-completion
    # background task with no other work sharing the thread (matches
    # this campaign's own one-worker-per-run lifecycle, §"Execution").
    #
    # The actual algorithm lives in execute() -- a plain method,
    # callable directly and synchronously without ever starting a real
    # QThread, so it can be unit-tested without a Qt event loop.
    # run() (QThread's own entry point) only ever calls execute() and
    # turns its result/exception into a signal.

    progress_updated = pyqtSignal(object)          # ProgressModel
    campaign_finished = pyqtSignal(object)         # CampaignSummary
    error_occurred = pyqtSignal(str)

    preflight_completed = pyqtSignal(object)       # PreflightResult
    first_rejection_detected = pyqtSignal(object)  # ValidationRow
    diagnostics_updated = pyqtSignal(object)       # DiagnosticsSummary

    def __init__(self, config: CampaignConfig, parent=None):
        super().__init__(parent)

        self.config = config

        # set() == not paused (free to proceed); cleared == paused,
        # the loop below blocks on wait() until resume() sets it again.
        self._pause_event = threading.Event()
        self._pause_event.set()

        self._cancel_requested = False

    # =====================================================

    def pause(self):

        self._pause_event.clear()

    # =====================================================

    def resume(self):

        self._pause_event.set()

    # =====================================================

    def cancel(self):

        self._cancel_requested = True

        # Wakes a currently-paused wait() so a cancel issued while
        # paused takes effect immediately rather than waiting for a
        # resume() that may never come.
        self._pause_event.set()

    # =====================================================

    @property
    def is_cancelled(self) -> bool:

        return self._cancel_requested

    # =====================================================

    def run(self):

        # QThread's real entry point -- only ever reached when this
        # worker is actually started on a background thread via
        # .start(). Tests call execute() directly instead, to avoid
        # depending on a running Qt event loop.

        try:
            summary = self.execute()

        except Exception as exc:

            self.error_occurred.emit(str(exc))
            return

        self.campaign_finished.emit(summary)

    # =====================================================

    def execute(self) -> CampaignSummary:

        config = self.config

        collector = DiagnosticsCollector() if config.diagnostics_mode else None

        if config.diagnostics_mode:

            preflight = self._run_preflight_checks(config)
            self.preflight_completed.emit(preflight)

            if preflight.has_errors:

                # Requirements 5/6 -- stop before wasting a single
                # generation attempt against a Building/Definition
                # already known to be invalid, and say so plainly
                # rather than producing an unexplained "0 accepted."
                return CampaignSummary(
                    total_requested=config.count, total_generated=0, accepted=0, rejected=0,
                    average_evacuation_time=None, average_simulation_duration=0.0,
                    output_directory=str(config.output_directory), cancelled=False,
                    rejection_explanation=self._preflight_explanation(preflight),
                )

        start_time = time.perf_counter()

        # Seeded from whatever this output directory already holds --
        # docs/architecture/integration_validation.md §10.1's own
        # documented operational hazard: an orchestrator that skips
        # this risks scenario_storage.save_scenario()'s FileExistsError
        # guard firing mid-campaign against a directory a prior run
        # already populated.
        accepted_hashes = set(config.accepted_hashes) | set(
            load_accepted_hashes(config.output_directory)
        )

        accepted_count = 0
        rejected_count = 0

        evacuation_times = []
        simulation_durations = []

        for index in range(config.count):

            self._pause_event.wait()

            if self._cancel_requested:
                break

            scenario_seed = derive_scenario_seed(config.master_seed, index)

            scenario = self._generate_accepted_scenario(
                config, scenario_seed, accepted_hashes, collector,
            )

            if scenario is not None:

                accepted_count += 1

                try:
                    save_scenario(scenario, config.output_directory)

                except FileExistsError:
                    # Already stored by a prior run against this exact
                    # output directory (same definition_content_hash +
                    # seed) -- not an error, the campaign simply
                    # continues to simulate the in-memory candidate it
                    # already has.
                    pass

                accepted_hashes.add(compute_candidate_content_hash(scenario))

                simulation_start = time.perf_counter()

                evacuation_time = self._simulate(scenario, config)

                simulation_durations.append(time.perf_counter() - simulation_start)

                if evacuation_time is not None:
                    evacuation_times.append(evacuation_time)

                current_label = (
                    f"Scenario {index + 1}/{config.count} ({scenario.metadata.scenario_id})"
                )

            else:

                rejected_count += 1
                current_label = f"Scenario {index + 1}/{config.count} (rejected)"

            self.progress_updated.emit(
                self._build_progress(
                    config, accepted_count, rejected_count, current_label,
                    simulation_durations, start_time,
                )
            )

            if collector is not None:
                self.diagnostics_updated.emit(collector.summary())

        rejection_explanation = None

        if collector is not None and accepted_count == 0 and rejected_count > 0:
            rejection_explanation = explain_total_rejection(collector.summary())

        return CampaignSummary(
            total_requested=config.count,
            total_generated=accepted_count,
            accepted=accepted_count,
            rejected=rejected_count,
            average_evacuation_time=(
                sum(evacuation_times) / len(evacuation_times) if evacuation_times else None
            ),
            average_simulation_duration=(
                sum(simulation_durations) / len(simulation_durations)
                if simulation_durations else 0.0
            ),
            output_directory=str(config.output_directory),
            cancelled=self._cancel_requested,
            rejection_explanation=rejection_explanation,
        )

    # =====================================================
    # Pre-flight -- Requirements 5/6. Building validated once, Definition
    # validated once, both reusing already-existing, already-frozen
    # validators verbatim
    # -- designer.validation.validate_building_authoring() (the same
    # check MainWindow's own "Validate Project" action already runs) and
    # ScenarioDefinition.validate(building=...) (scenario_definition's
    # own frozen self-validation entry point). Neither is reimplemented
    # here.
    # =====================================================

    def _run_preflight_checks(self, config: CampaignConfig) -> PreflightResult:

        building_report = validate_building_authoring(config.building)
        definition_report = config.definition.validate(building=config.building)

        return PreflightResult(
            building_issues=tuple(
                ValidationRow(source="BUILDING", code=issue.code, message=issue.message)
                for issue in building_report.errors
            ),
            definition_issues=tuple(
                ValidationRow(source="DEFINITION", code=issue.code, message=issue.message)
                for issue in definition_report.errors
            ),
        )

    # =====================================================

    def _preflight_explanation(self, preflight: PreflightResult) -> str:

        parts = []

        if preflight.building_issues:

            parts.append(
                f"The Building itself is invalid ({len(preflight.building_issues)} "
                f"issue(s)): " + "; ".join(
                    f"[{row.code}] {row.message}" for row in preflight.building_issues
                )
            )

        if preflight.definition_issues:

            parts.append(
                f"The Scenario Definition is contradictory "
                f"({len(preflight.definition_issues)} issue(s)): " + "; ".join(
                    f"[{row.code}] {row.message}" for row in preflight.definition_issues
                )
            )

        return (
            "Campaign stopped before generating any scenarios. " + " ".join(parts)
        )

    # =====================================================
    # Scenario generation -- Requirement 1/2. In Diagnostics Mode, every
    # attempt's ScenarioValidationReport is captured via the collector
    # (§ module docstring); otherwise, this is exactly the single
    # run_pipeline() call the pre-diagnostics implementation already
    # made.
    # =====================================================

    def _generate_accepted_scenario(self, config, scenario_seed, accepted_hashes, collector):

        if collector is None:

            result = run_pipeline(
                config.definition, config.definition_id, config.building, scenario_seed,
                max_attempts=config.max_attempts, accepted_hashes=frozenset(accepted_hashes),
            )

            return result.scenario if result.accepted else None

        return self._generate_with_diagnostics(config, scenario_seed, accepted_hashes, collector)

    # =====================================================

    def _generate_with_diagnostics(self, config, scenario_seed, accepted_hashes, collector):

        # Mirrors scenario_pipeline.run_pipeline()'s own attempt loop
        # exactly (scenario_pipeline/pipeline.py) -- same two calls,
        # same order, same accepted_hashes threading -- restated only
        # because run_pipeline() itself has no seam to report every
        # attempt's ScenarioValidationReport back to a caller.

        for attempt_index in range(config.max_attempts):

            request = GenerationRequest(
                definition=config.definition, definition_id=config.definition_id,
                building=config.building, seed=scenario_seed, attempt_index=attempt_index,
            )

            candidate = generate_scenario(request)

            report = validate(
                candidate, config.definition, config.building,
                accepted_hashes=frozenset(accepted_hashes),
            )

            first_rejection_row = collector.record_candidate_report(report)

            if first_rejection_row is not None:
                self.first_rejection_detected.emit(first_rejection_row)

            if report.accepted:
                return candidate

        return None

    # =====================================================

    def _simulate(self, scenario, config: CampaignConfig) -> Optional[float]:

        # scenario_runner -> behaviour_profile_resolver ->
        # ai_decision -> simulation_runtime, exactly the sequence
        # docs/architecture/simulation_runtime.md §7 already fixes.
        # Returns MultiAgentSimulationResult.total_evacuation_time --
        # an already-public field, never a re-derivation of it.
        #
        # Production Pipeline Completion -- registration now goes
        # through behaviour_profile_resolver.dynamic_registrar.
        # register_population_dynamic() instead of the civilian-only,
        # firefighter-blind register_occupants() this method used to
        # call: register_occupants() never registered
        # Scenario.firefighters at all, so a firefighter's own rescue
        # target never moved and rescue could never execute in a real
        # campaign. register_population_dynamic() is the same already-
        # existing, already-tested Human Decision Engine path
        # research_framework.runner already exercises via its own
        # "dynamic" registration arm -- calling it here wires the
        # already-built engine into the one caller that never used it,
        # nothing about the engine itself changes. civilian/population
        # registration remain fully available and unaffected via
        # behaviour_profile_resolver.register_occupants/
        # register_population for any other caller that still wants
        # them (e.g. research_framework.runner's own "civilian"/
        # "population" arms).
        context = run_scenario(scenario, config.building)

        registration_result = register_population_dynamic(context)

        decision_engine = AIDecisionEngine(base_engine=context.engine)

        runtime = SimulationRuntime(
            context, decision_engine, dt=config.dt, perception_provider=None,
        )

        # SimulationRuntime.run() already returns the full
        # Tuple[TickResult, ...] for this run -- captured here (the
        # pre-post-processing version of this method discarded it)
        # rather than re-ticking the Runtime a second time to get it.
        tick_results = runtime.run()

        self._export_scenario_artifacts(scenario, config, runtime, tick_results, registration_result)

        return runtime.movement_result.total_evacuation_time

    # =====================================================
    # Post-processing over one already-completed scenario's Runtime
    # output. Every call below is to an already-public, already-tested
    # entry point of its own package (dataset_builder/ground_truth/
    # decision_policy) -- no generation, validation, or simulation logic
    # is reimplemented here, and movement_result/tick_results are never
    # recomputed, only threaded through to each package that needs them.
    # =====================================================

    def _export_scenario_artifacts(self, scenario, config, runtime, tick_results, registration_result):

        building = config.building
        movement_result = runtime.movement_result
        scenario_id = scenario.metadata.scenario_id

        # ---- Decision Events (Human Decision Engine Phase 6) --
        # registration_result.events already carries every Help_Decision/
        # Help_Rejected/Firefighter_Task/Group_Formed event the dynamic
        # registration session itself produced (behaviour_profile_
        # resolver.dynamic_registrar); Rescue_Completed/Group_Dissolved
        # can only be known once movement_result exists (whether the
        # rescued/grouped occupant's own OccupantTimeline actually
        # reached ARRIVED), so those two are computed here, once, from
        # already-existing, already-tested human_decision_engine.groups
        # functions that were previously computed nowhere in production
        # -- no new decision logic, only completing calls into functions
        # that already existed.
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

        # ---- Dataset Builder (V1) -- scenario_features.csv,
        # simulation_outcomes.csv, zone_results.csv ----
        #
        # tick_results/decision_events are both additive, optional
        # SimulationRun fields (see that dataclass's own docstring) --
        # threading them through here is what makes Fallen_Count/
        # Possible_Injury_Count (need tick_results) and Help_Decision_
        # Count/Group_Formed_Count/Rescue_Initiated_Count/... (need
        # decision_events) stop being permanently-zero columns; nothing
        # about how either is computed changes.

        simulation_run = SimulationRun(
            scenario=scenario, building=building, movement_result=movement_result,
            simulation_finished=runtime.is_finished,
            tick_results=tick_results, decision_events=decision_events,
        )

        dataset_dir = os.path.join(config.output_directory, "datasets", scenario_id)
        DatasetBuilder([simulation_run]).export_all(dataset_dir)

        # ---- Behavior Events -- one row per occupant's own final,
        # resolved BehaviorDecision (registration_result.behavior_layer
        # already keeps exactly one per occupant_id, superseded entries
        # replaced in place -- see HumanBehaviorLayer._decisions_so_far's
        # own docstring on why a later decision for the same occupant_id
        # replaces, never appends to, an earlier one). Never fabricated:
        # every row is a decision the Human Behavior Layer already made
        # and submitted to Simulation this run. ----

        behavior_events_run = BehaviorEventsRun(
            scenario=scenario,
            behavior_decisions=tuple(registration_result.behavior_layer._decisions_so_far.values()),
        )
        behavior_event_rows = extract_behavior_event_rows(behavior_events_run)

        behavior_events_dir = os.path.join(config.output_directory, "behavior_events", scenario_id)
        os.makedirs(behavior_events_dir, exist_ok=True)
        export_csv(behavior_event_rows, os.path.join(behavior_events_dir, "behavior_events.csv"))

        # ---- Dataset Builder (V2) -- Timeline Dataset, one row per
        # tick. No TimelineDatasetBuilder class exists (only the
        # TimelineRun/extract_timeline_rows() functions) -- exported via
        # dataset_builder.exporters.export_csv(), the same already-
        # existing, generic CSV writer export_all() itself uses. ----

        timeline_run = TimelineRun(
            scenario=scenario, building=building, movement_result=movement_result,
            tick_results=tick_results,
        )
        timeline_rows = extract_timeline_rows(timeline_run)

        timeline_dir = os.path.join(config.output_directory, "timelines", scenario_id)
        os.makedirs(timeline_dir, exist_ok=True)

        export_csv(timeline_rows, os.path.join(timeline_dir, "timeline.csv"))

        # Also written as JSON -- the shape command_center.incident_data.
        # load_incident()'s timeline_rows_path already expects (a plain
        # JSON list of row dicts), so Command Center playback can be
        # loaded straight from this campaign's output directory without
        # re-parsing the CSV.
        with open(
            os.path.join(timeline_dir, "timeline_rows.json"), "w", encoding="utf-8",
        ) as handle:
            json.dump(timeline_rows, handle)

        # ---- Simulation Recording -- occupant routes + decision events.
        # Simulation Replay Studio V1's one new artifact pair: neither
        # recomputes anything -- occupant routes are extracted straight
        # off this scenario's own already-completed movement_result
        # (simulator.multi_agent_result.MultiAgentSimulationResult, read-
        # only), and decision_events is the exact same tuple already
        # built above for Dataset Builder's own *_Count columns, simply
        # also persisted this time instead of being discarded after. ----

        occupant_route_records = build_occupant_route_records(movement_result)

        occupant_routes_dir = os.path.join(config.output_directory, "occupant_routes", scenario_id)
        os.makedirs(occupant_routes_dir, exist_ok=True)
        save_occupant_routes(
            occupant_route_records, os.path.join(occupant_routes_dir, "occupant_routes.json"),
        )

        decision_events_dir = os.path.join(config.output_directory, "decision_events", scenario_id)
        os.makedirs(decision_events_dir, exist_ok=True)
        save_decision_events(
            decision_events, os.path.join(decision_events_dir, "decision_events.json"),
        )

        # ---- Ground Truth Generator ----

        ground_truth = analyze_ground_truth(
            SimulationArtifacts(
                scenario=scenario, building=building, movement_result=movement_result,
                tick_results=tick_results, decision_events=decision_events,
            )
        )

        ground_truth_dir = os.path.join(config.output_directory, "ground_truth", scenario_id)
        os.makedirs(ground_truth_dir, exist_ok=True)

        with open(
            os.path.join(ground_truth_dir, "ground_truth.json"), "w", encoding="utf-8",
        ) as handle:
            json.dump(ground_truth.to_dict(), handle)

        # ---- Decision Policy Generator -- reuses the GroundTruth just
        # computed above and the Timeline rows just extracted above,
        # never recomputing either. ----

        decision_policy = generate_policy(
            DecisionInputs(
                building=building, scenario=scenario, ground_truth=ground_truth,
                timeline_rows=tuple(timeline_rows),
            )
        )

        decision_policy_dir = os.path.join(config.output_directory, "decision_policy", scenario_id)
        os.makedirs(decision_policy_dir, exist_ok=True)

        with open(
            os.path.join(decision_policy_dir, "decision_policy.json"), "w", encoding="utf-8",
        ) as handle:
            json.dump(decision_policy.to_dict(), handle)

        self._ensure_building_project_saved(config)

    # =====================================================

    def _ensure_building_project_saved(self, config: CampaignConfig):

        # Command Center playback (command_center.incident_data.
        # load_incident()) needs a saved .syn Project to read the
        # Building back from -- written once per campaign output
        # directory (config.building is the same object for every
        # scenario in one campaign run, so re-saving it per scenario
        # would be redundant, not incorrect). Uses Serializer exactly as
        # designer/windows/main_window.py's own "Save Project" action
        # already does -- no new serialization logic.

        project_path = os.path.join(config.output_directory, "building.syn")

        if os.path.exists(project_path):
            return

        os.makedirs(config.output_directory, exist_ok=True)

        Serializer.save(Project(name=config.name, building=config.building), project_path)

    # =====================================================

    def _build_progress(
        self, config, accepted_count, rejected_count, current_label,
        simulation_durations, start_time,
    ) -> ProgressModel:

        processed = accepted_count + rejected_count
        elapsed = time.perf_counter() - start_time

        generation_speed = processed / elapsed if elapsed > 0 else 0.0

        eta_seconds = (
            (config.count - processed) / generation_speed
            if generation_speed > 0 else 0.0
        )

        return ProgressModel(
            total=config.count,
            processed=processed,
            accepted=accepted_count,
            rejected=rejected_count,
            remaining=config.count - processed,
            current_scenario_label=current_label,
            average_simulation_time=(
                sum(simulation_durations) / len(simulation_durations)
                if simulation_durations else 0.0
            ),
            generation_speed=generation_speed,
            eta_seconds=eta_seconds,
            elapsed_seconds=elapsed,
        )
