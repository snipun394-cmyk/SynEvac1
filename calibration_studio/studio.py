from typing import Dict, List, Optional, Tuple

from calibration_studio.project import CalibrationProject
from calibration_studio.session import CalibrationSession


# =====================================================
# Calibration Studio Phase 1 -- Core Architecture.
#
# CalibrationStudio is the single public entry point this milestone's
# own brief requires -- a coordinating facade, never a second
# execution/decision authority, the same rule already proven twice in
# this codebase (Recommendation Layer, Execution Layer) and carried
# forward explicitly from the approved architecture blueprint.
#
# Deliberately holds nothing calibration_benchmark/research_framework/
# Replay Studio/Dataset Builder already own: no simulation composition,
# no statistics, no report rendering, no scenario storage. What it owns
# in this phase is exactly the in-process bookkeeping needed to satisfy
# "CalibrationStudio can create projects" -- a plain dict, not
# persistence (nothing here is written to disk; a restart loses
# everything, precisely because Phase 2 -- durable storage -- is
# explicitly out of scope for this milestone).
#
# run_published_benchmark()/run_parameter_sweep()/open_in_replay_studio()/
# generate_validation_dashboard() are the four orchestration methods the
# approved architecture blueprint names -- present here as placeholders
# only, each documenting which existing system it will delegate to once
# a later phase implements it, so the public API shape is stable before
# any execution logic exists behind it.
# =====================================================


class CalibrationStudio:

    def __init__(self):

        self._projects: Dict[str, CalibrationProject] = {}

    # =====================================================
    # Projects
    # =====================================================

    def create_project(
        self, *, name: str, description: str = "", tags: Tuple[str, ...] = (),
    ) -> CalibrationProject:

        project = CalibrationProject(name=name, description=description, tags=tags)
        self._projects[project.project_id] = project

        return project

    def get_project(self, project_id: str) -> Optional[CalibrationProject]:

        return self._projects.get(project_id)

    def list_projects(self) -> Tuple[CalibrationProject, ...]:

        return tuple(self._projects.values())

    # =====================================================
    # Sessions -- convenience read access across every known project;
    # session creation itself belongs to CalibrationProject (see that
    # class's own docstring: a project's lifecycle gates whether new
    # sessions may be added to it).
    # =====================================================

    def list_sessions(self) -> Tuple[CalibrationSession, ...]:

        sessions: List[CalibrationSession] = []

        for project in self._projects.values():
            sessions.extend(project.sessions)

        return tuple(sessions)

    def get_session(self, session_id: str) -> Optional[CalibrationSession]:

        for project in self._projects.values():

            session = project.get_session(session_id)

            if session is not None:
                return session

        return None

    # =====================================================
    # Orchestration placeholders -- no algorithm lives in this package;
    # each of these will, in a later phase, delegate to the real,
    # already-existing system named in its own docstring. Calling one
    # today raises NotImplementedError rather than silently doing
    # nothing, so a caller can never mistake "not built yet" for "ran
    # and did nothing."
    # =====================================================

    def run_published_benchmark(self, *args, **kwargs):

        # Will delegate to calibration_benchmark.run_calibration_benchmark()
        # (a real, published-value comparison generalization of it, per
        # the approved architecture blueprint's Phase 4/7) -- never a
        # reimplementation of paired-comparison logic.
        raise NotImplementedError(
            "run_published_benchmark() is not implemented until Calibration Studio's "
            "execution phase; it will delegate to calibration_benchmark, never duplicate it.",
        )

    def run_parameter_sweep(self, *args, **kwargs):

        # Will orchestrate multiple CalibrationSession runs, each one
        # still delegating to calibration_benchmark.run_calibration_benchmark()
        # for its own actual comparison.
        raise NotImplementedError(
            "run_parameter_sweep() is not implemented until Calibration Studio's "
            "execution phase; it will orchestrate calibration_benchmark runs, never "
            "duplicate its statistics.",
        )

    def open_in_replay_studio(self, *args, **kwargs):

        # Will resolve a session's recorded scenario artifacts and hand
        # them to replay_studio.session.resolve_scenario_artifacts(),
        # unmodified -- Replay Studio itself is never reimplemented.
        raise NotImplementedError(
            "open_in_replay_studio() is not implemented until Calibration Studio's "
            "replay-integration phase; it will call replay_studio directly, never "
            "reimplement playback.",
        )

    def generate_validation_dashboard(self, *args, **kwargs):

        # Explicitly out of scope for this milestone (Dashboard is not
        # to be implemented) -- present only so the public API's final
        # shape is visible now.
        raise NotImplementedError(
            "generate_validation_dashboard() is out of scope until the Validation "
            "Dashboard milestone.",
        )

    def __repr__(self) -> str:

        return f"CalibrationStudio(projects={len(self._projects)})"
