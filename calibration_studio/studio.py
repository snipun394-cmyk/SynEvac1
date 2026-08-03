from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from calibration_benchmark import CalibrationBenchmarkResult, ParameterCandidate, run_calibration_benchmark
from calibration_benchmark.optional_metrics import AdditionalMetric

import calibration_studio.storage as storage
from calibration_studio.benchmark import PublishedBenchmark
from calibration_studio.benchmark_library import BenchmarkNotFoundError, PublishedBenchmarkLibrary
from calibration_studio.dashboard import generate_validation_dashboard as generate_validation_dashboard_impl
from calibration_studio.geometry_resolution import resolve_geometry_reference
from calibration_studio.project import CalibrationProject
from calibration_studio.replay_integration import (
    open_in_replay_studio as open_in_replay_studio_impl,
    record_session_replay as record_session_replay_impl,
)
from calibration_studio.report import CalibrationReportGenerator
from calibration_studio.session import CalibrationSession


# =====================================================
# Calibration Studio Phase 1 -- Core Architecture; Phase 2 --
# Persistence Layer; Phase 3 -- Published Benchmark Library; Phase 4 --
# Calibration Runner; Phase 5 -- Replay Studio Integration; Phase 6 --
# Validation Dashboard; Phase 7 -- Report Generation (feature-complete).
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
# run_published_benchmark()/run_parameter_sweep() (below) are real now.
# Both delegate every scientific computation to
# calibration_benchmark.run_calibration_benchmark() -- the ONE call
# either method ever makes into calibration_benchmark. Nothing in this
# module reimplements paired comparison, statistics, or recommendation
# logic; grep-testable (see tests/test_calibration_studio_runner_
# architecture.py) -- this file imports `run_calibration_benchmark`
# and calls it exactly once, in `_execute()`, and defines no function
# whose name or body resembles a comparison/statistics primitive.
#
# record_session_replay()/open_in_replay_studio() are real now too --
# thin delegation to calibration_studio/replay_integration.py, which
# owns every actual call into calibration_benchmark/simulation_recording/
# scenario_storage/serialization/replay_studio/command_center. Replay
# Studio remains the sole visualization engine; nothing here renders
# anything.
#
# generate_validation_dashboard() is real now too -- read-only,
# stateless aggregation, thin delegation to calibration_studio/
# dashboard.py. See that module's own docstring for the ONE genuine
# scope gap it discloses: ValidationEvidence/ParameterValidationRecord/
# ExperimentHistory (the ORIGINAL persistent-data-model design's own
# Phases 4/5/6, a different numbering from this implementation's
# Phase 0-5) were never actually built as code anywhere in this
# codebase -- the dashboard computes equivalent summaries directly from
# PublishedBenchmark/CalibrationSession's own already-real fields
# instead, not from stand-in types invented to make the names match.
# =====================================================


class CalibrationStudio:

    def __init__(
        self,
        *,
        storage_root=None,
        benchmark_library: Optional[PublishedBenchmarkLibrary] = None,
        report_generator: Optional[CalibrationReportGenerator] = None,
    ):

        self._projects: Dict[str, CalibrationProject] = {}
        self._storage_root = Path(storage_root) if storage_root is not None else None

        # Composition, not duplication -- CalibrationStudio never
        # re-implements benchmark registration/lookup, it simply owns
        # one PublishedBenchmarkLibrary instance (pointed at the same
        # storage_root by default, so a Studio's data lives under one
        # directory tree) and exposes it directly.
        self.benchmarks = (
            benchmark_library if benchmark_library is not None
            else PublishedBenchmarkLibrary(storage_root=self._storage_root)
        )

        # Same composition pattern, for the same reason -- report
        # generation never happens inline in this class's own methods.
        self.reports = report_generator if report_generator is not None else CalibrationReportGenerator()

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
    # Calibration Runner -- Phase 4.
    #
    # PublishedBenchmark -> CalibrationSession -> calibration_benchmark
    # -> research_framework.statistics -> recommendation ->
    # CalibrationSession Result, exactly the chain this milestone's own
    # brief names. Every step after "look up the benchmark" and "create
    # the session" is calibration_benchmark's own, unmodified code,
    # called through its own public API
    # (calibration_benchmark.run_calibration_benchmark(), which itself
    # already calls research_framework.statistics and
    # calibration_benchmark.recommendation internally -- this module
    # never touches either directly).
    # =====================================================

    def run_published_benchmark(
        self,
        *,
        project: CalibrationProject,
        benchmark_id: str,
        candidate: ParameterCandidate,
        definition,
        definition_id: str,
        master_seed: int,
        n_scenarios: int,
        dt: float = 5.0,
        building=None,
        additional_metrics: Sequence[AdditionalMetric] = (),
    ) -> CalibrationSession:

        benchmark = self.benchmarks.get(benchmark_id)

        if benchmark is None:
            raise BenchmarkNotFoundError(
                f"No benchmark registered at benchmark_id {benchmark_id!r} -- register() or "
                f"load_benchmark() it into this Studio's benchmark library first.",
            )

        # `building` lets a caller always override or supply one
        # directly (required for a DATASET_VALIDATION benchmark, which
        # has no geometry_reference at all -- the same "compared
        # against a synthetic single-bottleneck fixture, not the
        # dataset's own 'geometry'" shape calibration_benchmark's own
        # existing Julich demo already uses). Only resolved from the
        # benchmark's own geometry_reference when the caller didn't
        # supply one.
        resolved_building = building if building is not None else resolve_geometry_reference(benchmark.geometry_reference)

        if resolved_building is None:
            raise ValueError(
                f"Benchmark {benchmark_id!r} has no geometry_reference and no building was "
                f"supplied -- pass building=... explicitly (required for a DATASET_VALIDATION "
                f"benchmark, or any benchmark you want run against a different building).",
            )

        session = project.create_session(benchmark_id=benchmark_id, candidate=candidate, master_seed=master_seed)

        self._execute(session, candidate, resolved_building, definition, definition_id, master_seed, n_scenarios, dt, additional_metrics)

        # Append-only, idempotent (PublishedBenchmark.add_calibration_session()'s
        # own guarantee) -- recorded whether the run completed or
        # failed; a failed attempt is still part of this benchmark's
        # own calibration history, not something to hide. Never
        # auto-saved -- persisting the benchmark's updated history is
        # the caller's own explicit choice, exactly like every other
        # save_*() call in this package.
        benchmark.add_calibration_session(session.session_id)

        return session

    def run_parameter_sweep(
        self,
        *,
        project: CalibrationProject,
        candidates: Sequence[ParameterCandidate],
        building,
        definition,
        definition_id: str,
        master_seed: int,
        n_scenarios: int,
        dt: float = 5.0,
        benchmark_id: Optional[str] = None,
        additional_metrics: Sequence[AdditionalMetric] = (),
    ) -> Tuple[CalibrationSession, ...]:

        # benchmark_id is optional here (a sweep need not reference a
        # real-world benchmark at all -- e.g. "compare five candidate
        # walking speeds against production defaults") but, if given,
        # must resolve -- treated as a real reference, not a free-text
        # label, the same discipline run_published_benchmark() applies.
        benchmark: Optional[PublishedBenchmark] = None

        if benchmark_id is not None:

            benchmark = self.benchmarks.get(benchmark_id)

            if benchmark is None:
                raise BenchmarkNotFoundError(
                    f"No benchmark registered at benchmark_id {benchmark_id!r} -- register() or "
                    f"load_benchmark() it into this Studio's benchmark library first.",
                )

        sessions = []

        # Every candidate runs against the identical building/definition/
        # master_seed -- calibration_benchmark.run_calibration_benchmark()
        # deterministically regenerates the identical scenario batch each
        # call given the same (definition, definition_id, building,
        # master_seed, n_scenarios) (Phase 0's own verified guarantee),
        # so every candidate in the sweep is compared against the same
        # scenarios, not independently-sampled ones.
        for candidate in candidates:

            session = project.create_session(benchmark_id=benchmark_id, candidate=candidate, master_seed=master_seed)

            self._execute(session, candidate, building, definition, definition_id, master_seed, n_scenarios, dt, additional_metrics)

            if benchmark is not None:
                benchmark.add_calibration_session(session.session_id)

            sessions.append(session)

        return tuple(sessions)

    def _execute(
        self,
        session: CalibrationSession,
        candidate: ParameterCandidate,
        building,
        definition,
        definition_id: str,
        master_seed: int,
        n_scenarios: int,
        dt: float,
        additional_metrics: Sequence[AdditionalMetric],
    ) -> None:

        # Queued -> Running -> Completed/Failed, exactly this
        # milestone's own SESSION LIFECYCLE. Progress reporting is
        # necessarily coarse (0/n at the start, n_completed_pairs/n once
        # the call below returns), not per-scenario -- true per-scenario
        # progress would require either modifying calibration_benchmark.
        # harness.run_calibration_benchmark() itself (out of scope: "Do
        # NOT redesign calibration_benchmark") or reimplementing its
        # per-scenario loop out here (calibration_benchmark/harness.py's
        # own comparison/pairing logic -- _compare()/_paired_non_none()
        # -- is private, not exported, and duplicating it is exactly
        # what this milestone's own brief forbids: "No duplicated
        # comparison logic"). Two honest, real progress readings beat
        # a fabricated smooth one.
        session.mark_running(n_scenarios_total=n_scenarios)
        session.update_progress(0)

        try:
            result: CalibrationBenchmarkResult = run_calibration_benchmark(
                candidate, building, definition, definition_id, master_seed, n_scenarios,
                dt=dt, additional_metrics=additional_metrics,
            )
        except Exception as exc:
            # Caught, recorded, not re-raised -- a candidate/scenario
            # combination failing to simulate is a normal, expected
            # experiment outcome (this milestone's own lifecycle diagram
            # shows Completed/Failed as two equally-valid terminal
            # states), not a Calibration Studio bug. A caller who needs
            # to react to failure reads session.status, exactly like
            # they would for a completed run's recommendation.
            session.mark_failed(f"{type(exc).__name__}: {exc}")
            return

        session.update_progress(result.n_completed_pairs)
        session.mark_completed(result)

    # =====================================================
    # Replay Studio Integration -- Phase 5. Thin delegation to
    # calibration_studio/replay_integration.py, which owns every actual
    # call into calibration_benchmark/simulation_recording/
    # scenario_storage/serialization/replay_studio/command_center --
    # this facade only looks the session up. Replay Studio remains the
    # sole visualization engine; nothing here renders anything.
    # =====================================================

    def record_session_replay(
        self, *, session: CalibrationSession, scenario, building, output_dir, arm: str = "candidate", dt: float = 5.0,
    ) -> None:

        record_session_replay_impl(session, scenario, building, output_dir, arm=arm, dt=dt)

    def open_in_replay_studio(self, session_id: str):

        session = self.get_session(session_id)

        if session is None:
            raise ValueError(
                f"No session {session_id!r} is known to this Studio -- it must be reachable "
                f"through one of this Studio's own projects (create_session()'d or loaded via "
                f"load_project()), not just load_session()'d standalone.",
            )

        return open_in_replay_studio_impl(session)

    # =====================================================
    # Validation Dashboard -- Phase 6. Read-only, stateless: every call
    # re-aggregates from whatever this Studio's own benchmark library
    # and in-process sessions currently report -- no cache, no second
    # storage layer, so a session completed after the previous call is
    # simply present the next time this is called, with no separate
    # "refresh" step to remember. To include persisted (not just
    # in-process) data, call list_persisted_projects()/self.benchmarks.
    # list_persisted_benchmarks() first (both already exist, both
    # already merge into this Studio's in-process registries) -- this
    # method itself never reads or writes a file.
    # =====================================================

    def generate_validation_dashboard(self):

        return generate_validation_dashboard_impl(
            benchmarks=self.benchmarks.list_benchmarks(), sessions=self.list_sessions(),
        )

    # =====================================================
    # Report Generation -- Phase 7. Thin delegation to self.reports
    # (calibration_studio/report.py) -- this facade's only real
    # contribution is resolving the session's own project/benchmark
    # context automatically, so a caller doesn't have to look each of
    # them up by hand before every report. include_dashboard=True by
    # default (a fresh, current aggregation, matching Phase 6's own
    # "no cache" design) -- pass False to omit it (e.g. a caller who
    # already knows there is nothing meaningful to show yet).
    # =====================================================

    def _find_project_for_session(self, session: CalibrationSession) -> Optional[CalibrationProject]:

        for project in self._projects.values():
            if project.get_session(session.session_id) is not None:
                return project

        return None

    def generate_session_report(self, session_id: str, *, include_dashboard: bool = True) -> str:

        session = self.get_session(session_id)

        if session is None:
            raise ValueError(f"No session {session_id!r} is known to this Studio.")

        project = self._find_project_for_session(session)
        benchmark = self.benchmarks.get(session.benchmark_id) if session.benchmark_id else None
        dashboard = self.generate_validation_dashboard() if include_dashboard else None

        return self.reports.generate_session_report(
            session=session, project=project, benchmark=benchmark, dashboard=dashboard,
        )

    def save_session_report(self, session_id: str, *, include_dashboard: bool = True) -> Path:

        session = self.get_session(session_id)

        if session is None:
            raise ValueError(f"No session {session_id!r} is known to this Studio.")

        project = self._find_project_for_session(session)
        benchmark = self.benchmarks.get(session.benchmark_id) if session.benchmark_id else None
        dashboard = self.generate_validation_dashboard() if include_dashboard else None

        return self.reports.save_session_report(
            session=session, storage_root=self._require_storage_root(),
            project=project, benchmark=benchmark, dashboard=dashboard,
        )

    def __repr__(self) -> str:

        return f"CalibrationStudio(projects={len(self._projects)})"
