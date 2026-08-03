from pathlib import Path

from calibration_benchmark import run_with_overrides

from command_center.incident_data import IncidentData, load_incident

from models.project import Project

from replay_studio.session import resolve_scenario_artifacts

from scenario_storage.paths import scenario_json_path
from scenario_storage.storage import save_scenario

from serialization.serializer import Serializer

from simulation_recording.occupant_routes import build_occupant_route_records, save_occupant_routes

from calibration_studio.session import CalibrationSession


# =====================================================
# Calibration Studio Phase 5 -- Replay Studio Integration.
#
# Two functions, two different jobs, deliberately not one:
#
# record_session_replay() PRODUCES the artifacts Replay Studio's own
# existing loader expects, for exactly one (scenario, arm) pair from an
# already-completed session -- using ONLY existing, unmodified
# functions (calibration_benchmark.run_with_overrides(), the same seam
# calibration_benchmark's own harness itself calls; simulation_recording.
# occupant_routes.build_occupant_route_records()/save_occupant_routes(),
# the same functions designer/campaign/campaign_worker.py and
# research_framework/runner.py already use to produce this exact
# artifact for every OTHER scenario source in this codebase;
# scenario_storage.storage.save_scenario(); serialization.serializer.
# Serializer.save()). No new simulation, no new persistence format, no
# visualization -- every one of these calls already existed before this
# phase, doing exactly what it already does for every other caller.
#
# calibration_benchmark.run_calibration_benchmark() itself computes a
# movement_result for every scenario internally and discards it after
# extracting metrics (confirmed by reading calibration_benchmark/
# harness.py: only MetricSample survives the per-scenario loop) --
# there was never anywhere for a movement_result to persist from. This
# function re-runs run_with_overrides() for exactly the one scenario
# being recorded, using the session's own candidate (or plain defaults
# for the 'baseline' arm) so the replay reflects the same conditions
# the original comparison used, not a different run.
#
# open_in_replay_studio() RESOLVES + VALIDATES + DELEGATES -- it never
# runs a simulation, never writes a file, never renders anything.
# replay_studio.session.resolve_scenario_artifacts() (existing,
# unmodified) does the actual path resolution; command_center.
# incident_data.load_incident() (existing, unmodified -- the same
# function Replay Studio's own OpenScenarioDialog flow ultimately
# reaches) does the actual loading. This function's only real
# contribution is validating enough exists to make that call
# meaningful, and raising a clear, specific ReplayArtifactsUnavailableError
# when it doesn't, rather than letting a confusing exception surface
# from three layers down.
# =====================================================


class ReplayArtifactsUnavailableError(Exception):
    pass


_BUILDING_FILENAME = "building.syn"
_OCCUPANT_ROUTES_SUBDIRECTORY = "occupant_routes"
_OCCUPANT_ROUTES_FILENAME = "occupant_routes.json"

_VALID_ARMS = ("baseline", "candidate")


def record_session_replay(
    session: CalibrationSession, scenario, building, output_dir, *, arm: str = "candidate", dt: float = 5.0,
) -> None:

    if arm not in _VALID_ARMS:
        raise ValueError(f"arm must be one of {_VALID_ARMS}, got {arm!r}")

    candidate = session.candidate

    if candidate is None:
        raise ValueError(
            f"Session {session.session_id} has no live ParameterCandidate to record a replay "
            f"from -- `candidate` is None either because none was ever supplied, or because this "
            f"session was reloaded from disk (see CalibrationSession's own docstring: a live "
            f"candidate never survives a save/load round trip). Record replay before saving/"
            f"reloading the session.",
        )

    if arm == "candidate":
        registry = candidate.candidate_registry()
        capacity_model = candidate.candidate_capacity_model()
        congestion_model = candidate.candidate_congestion_model()
        use_flow_regions = candidate.candidate_use_flow_regions()
    else:
        registry = candidate.baseline_registry()
        capacity_model = candidate.baseline_capacity_model()
        congestion_model = candidate.baseline_congestion_model()
        use_flow_regions = candidate.baseline_use_flow_regions()

    movement_result, _ground_truth, building_copy = run_with_overrides(
        scenario, building, registry=registry, capacity_model=capacity_model,
        congestion_model=congestion_model, dt=dt, use_flow_regions=use_flow_regions,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario_id = scenario.metadata.scenario_id

    project = Project(building=building_copy)
    Serializer.save(project, str(output_dir / _BUILDING_FILENAME))

    try:
        save_scenario(scenario, output_dir)
    except FileExistsError:
        # Already recorded -- e.g. a second call recording the other
        # arm of the same scenario, or a caller re-recording after an
        # earlier partial failure. Same scenario_id always means
        # byte-identical content (Scenario is write-once/immutable by
        # its own design), so this is a legitimate no-op, not silently
        # papering over a real conflict.
        pass

    routes_dir = output_dir / _OCCUPANT_ROUTES_SUBDIRECTORY / scenario_id
    routes_dir.mkdir(parents=True, exist_ok=True)

    records = build_occupant_route_records(movement_result)
    save_occupant_routes(records, str(routes_dir / _OCCUPANT_ROUTES_FILENAME))

    # Validates scenario_id actually belongs to this session's own
    # result -- CalibrationSession.set_replay_reference()'s own
    # integrity check, not repeated here.
    session.set_replay_reference(str(output_dir), scenario_id)


def open_in_replay_studio(session: CalibrationSession) -> IncidentData:

    if session.replay_output_dir is None or session.replay_scenario_id is None:
        raise ReplayArtifactsUnavailableError(
            f"Session {session.session_id} has no replay reference recorded -- call "
            f"record_session_replay() for it first.",
        )

    artifact_paths = resolve_scenario_artifacts(session.replay_output_dir, session.replay_scenario_id)

    if artifact_paths["project_path"] is None:
        raise ReplayArtifactsUnavailableError(
            f"No {_BUILDING_FILENAME} found under {session.replay_output_dir!r} -- cannot open "
            f"session {session.session_id} in Replay Studio.",
        )

    scenario_path = scenario_json_path(Path(session.replay_output_dir), session.replay_scenario_id)

    if not scenario_path.exists():
        raise ReplayArtifactsUnavailableError(
            f"No stored scenario found for scenario_id {session.replay_scenario_id!r} under "
            f"{session.replay_output_dir!r} -- cannot open session {session.session_id} in "
            f"Replay Studio.",
        )

    return load_incident(**artifact_paths)
