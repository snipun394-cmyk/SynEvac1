from pathlib import Path
from typing import Dict, List, Optional, Tuple

import calibration_studio.storage as storage
from calibration_studio.project import CalibrationProject
from calibration_studio.session import CalibrationSession


# =====================================================
# Calibration Studio Phase 1 -- Core Architecture; Phase 2 --
# Persistence Layer.
#
# CalibrationStudio is the single public entry point this milestone's
# own brief requires -- a coordinating facade, never a second
# execution/decision authority, the same rule already proven twice in
# this codebase (Recommendation Layer, Execution Layer) and carried
# forward explicitly from the approved architecture blueprint.
#
# Deliberately holds nothing calibration_benchmark/research_framework/
# Replay Studio/Dataset Builder already own: no simulation composition,
# no statistics, no report rendering. In-process bookkeeping (`_projects`)
# and durable storage (`save_project()`/`load_project()`/... below,
# calibration_studio/storage.py) are two separate, explicit layers, not
# one hidden auto-persisting mechanism: create_project() never writes to
# disk on its own, and save_project()/save_session() never happen
# implicitly as a side effect of anything else. A caller decides exactly
# when a project or a session is durable, the same "no hidden magic"
# discipline this codebase already applies to human-approval gates
# elsewhere. `storage_root` is optional -- a CalibrationStudio built
# without one behaves exactly as it did in Phase 1 (in-memory only);
# calling any persistence method on one raises a clear error rather than
# silently doing nothing.
#
# run_published_benchmark()/run_parameter_sweep()/open_in_replay_studio()/
# generate_validation_dashboard() are the four orchestration methods the
# approved architecture blueprint names -- present here as placeholders
# only, each documenting which existing system it will delegate to once
# a later phase implements it, so the public API shape is stable before
# any execution logic exists behind it.
# =====================================================


class CalibrationStudio:

    def __init__(self, *, storage_root=None):

        self._projects: Dict[str, CalibrationProject] = {}
        self._storage_root = Path(storage_root) if storage_root is not None else None

    def _require_storage_root(self) -> Path:

        if self._storage_root is None:
            raise ValueError(
                "This CalibrationStudio has no storage_root configured -- construct it as "
                "CalibrationStudio(storage_root=...) to use persistence methods.",
            )

        return self._storage_root

    # =====================================================
    # Persistence -- thin delegation to calibration_studio/storage.py,
    # which owns the actual file layout/catalog logic; this facade only
    # decides where storage_root comes from and keeps the in-process
    # `_projects` registry in sync with whatever gets loaded, so a
    # reopened project is immediately visible to get_project()/
    # list_projects() too, not just to whoever called load_project()
    # directly.
    # =====================================================

    def save_project(self, project: CalibrationProject) -> Path:

        return storage.save_project(project, self._require_storage_root())

    def load_project(self, project_id: str) -> CalibrationProject:

        project = storage.load_project(project_id, self._require_storage_root())
        self._projects[project.project_id] = project

        return project

    def list_persisted_projects(self) -> Tuple[CalibrationProject, ...]:

        projects = storage.list_projects(self._require_storage_root())

        for project in projects:
            self._projects[project.project_id] = project

        return projects

    def save_session(self, session: CalibrationSession) -> Path:

        return storage.save_session(session, self._require_storage_root())

    def load_session(self, session_id: str) -> CalibrationSession:

        return storage.load_session(session_id, self._require_storage_root())

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
