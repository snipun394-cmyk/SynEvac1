import json
from pathlib import Path
from typing import Optional, Tuple

from serialization.json_reader import JsonReader
from serialization.json_writer import JsonWriter

from calibration_studio.benchmark import SUPPORTED_SCHEMA_VERSIONS as SUPPORTED_BENCHMARK_SCHEMA_VERSIONS, PublishedBenchmark
from calibration_studio.catalog import (
    append_benchmark_catalog_row_if_new,
    append_project_catalog_row_if_new,
    append_session_catalog_row_if_new,
    read_benchmark_catalog_rows,
    read_project_catalog_rows,
    read_session_catalog_rows,
)
from calibration_studio.paths import benchmark_json_path, project_json_path, session_json_path
from calibration_studio.project import SUPPORTED_SCHEMA_VERSIONS as SUPPORTED_PROJECT_SCHEMA_VERSIONS, CalibrationProject
from calibration_studio.session import SUPPORTED_SCHEMA_VERSIONS as SUPPORTED_SESSION_SCHEMA_VERSIONS, CalibrationSession


# =====================================================
# Calibration Studio Phase 2 -- Persistence Layer.
#
# Built entirely on serialization/json_writer.py + json_reader.py (the
# same primitives scenario_storage/storage.py itself is built on) --
# no second persistence framework. This module owns exactly what
# scenario_storage/storage.py owns for Scenario: WHERE a record's JSON
# lives on disk and WHEN a catalog row gets written, using
# CalibrationProject.to_dict()/from_dict() and CalibrationSession.
# to_dict()/from_dict() verbatim for the actual shape -- this module
# never re-derives what a project/session "looks like."
#
# Deliberate departure from scenario_storage.save_scenario()'s
# write-once/FileExistsError-on-overwrite semantics: a Scenario is
# immutable once accepted, so overwriting one at the same id would
# either be a no-op or silent corruption. A CalibrationProject/
# CalibrationSession is the opposite -- a living record that changes
# over its own lifetime (a project's status, a session's execution
# state) and MUST be re-saveable at the same id. save_project()/
# save_session() therefore always overwrite; catalog rows are still
# only ever appended once per id (catalog.py's own module docstring).
# =====================================================


class IncompatibleSchemaVersionError(Exception):
    pass


class CorruptedRecordFileError(Exception):
    pass


def _read_record_json(path: Path) -> dict:

    try:
        return JsonReader.read(str(path))
    except json.JSONDecodeError as exc:
        raise CorruptedRecordFileError(f"Corrupted record file at {path}: {exc}") from exc


def _check_schema_version(data: dict, kind: str, supported: frozenset, record_id) -> None:

    found = data.get("schema_version")

    if found not in supported:
        raise IncompatibleSchemaVersionError(
            f"Cannot load {kind} {record_id!r}: schema_version {found!r} is not one of the "
            f"versions this code supports ({sorted(supported)!r}). A newer Calibration Studio "
            f"wrote this file with a schema this version has never heard of, or the file is "
            f"otherwise not a genuine {kind} record.",
        )


# =====================================================
# Projects
# =====================================================


def save_project(project: CalibrationProject, storage_root) -> Path:

    storage_root = Path(storage_root)
    json_path = project_json_path(storage_root, project.project_id)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    JsonWriter.write(str(json_path), project.to_dict())
    append_project_catalog_row_if_new(storage_root, project)

    return json_path


def load_project(project_id: str, storage_root, *, resolve_sessions: bool = True) -> CalibrationProject:

    storage_root = Path(storage_root)
    json_path = project_json_path(storage_root, project_id)

    data = _read_record_json(json_path)
    _check_schema_version(data, "project", SUPPORTED_PROJECT_SCHEMA_VERSIONS, project_id)

    project = CalibrationProject.from_dict(data)

    if resolve_sessions and project.session_ids:

        sessions = []

        for session_id in project.session_ids:

            try:
                sessions.append(load_session(session_id, storage_root))
            except FileNotFoundError:
                # Disclosed, not fatal: a project referencing a session
                # whose own file is missing (e.g. copied/backed-up
                # mid-write) still loads -- with that one session simply
                # absent from `sessions`, honestly reflecting what could
                # actually be resolved, rather than failing the whole
                # project load over one missing sibling file.
                continue

        project._attach_loaded_sessions(sessions)

    return project


def list_projects(storage_root) -> Tuple[CalibrationProject, ...]:

    storage_root = Path(storage_root)
    projects = []

    for row in read_project_catalog_rows(storage_root):

        project_id = row.get("project_id")

        if not project_id:
            continue

        try:
            projects.append(load_project(project_id, storage_root))
        except FileNotFoundError:
            # Catalog row exists but the JSON file it points to doesn't
            # -- same disclosed-not-fatal treatment as a missing
            # referenced session, above.
            continue

    return tuple(projects)


# =====================================================
# Sessions
# =====================================================


def save_session(session: CalibrationSession, storage_root) -> Path:

    storage_root = Path(storage_root)
    json_path = session_json_path(storage_root, session.session_id)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    JsonWriter.write(str(json_path), session.to_dict())
    append_session_catalog_row_if_new(storage_root, session)

    return json_path


def load_session(session_id: str, storage_root) -> CalibrationSession:

    storage_root = Path(storage_root)
    json_path = session_json_path(storage_root, session_id)

    data = _read_record_json(json_path)
    _check_schema_version(data, "session", SUPPORTED_SESSION_SCHEMA_VERSIONS, session_id)

    return CalibrationSession.from_dict(data)


def list_sessions(storage_root, *, project_id: Optional[str] = None) -> Tuple[CalibrationSession, ...]:

    storage_root = Path(storage_root)
    sessions = []

    for row in read_session_catalog_rows(storage_root):

        session_id = row.get("session_id")

        if not session_id:
            continue

        if project_id is not None and row.get("project_id") != project_id:
            continue

        try:
            sessions.append(load_session(session_id, storage_root))
        except FileNotFoundError:
            continue

    return tuple(sessions)


# =====================================================
# Published Benchmarks -- Phase 3. Identical shape/discipline to
# Projects/Sessions above: reuses _read_record_json()/
# _check_schema_version() as-is, overwrite-in-place save, append-if-new
# catalog. Registration/lookup (in-memory, duplicate detection) live on
# PublishedBenchmarkLibrary (calibration_studio/benchmark_library.py),
# not here -- this module only ever knows about disk.
# =====================================================


def save_benchmark(benchmark: PublishedBenchmark, storage_root) -> Path:

    storage_root = Path(storage_root)
    json_path = benchmark_json_path(storage_root, benchmark.benchmark_id)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    JsonWriter.write(str(json_path), benchmark.to_dict())
    append_benchmark_catalog_row_if_new(storage_root, benchmark)

    return json_path


def load_benchmark(benchmark_id: str, storage_root) -> PublishedBenchmark:

    storage_root = Path(storage_root)
    json_path = benchmark_json_path(storage_root, benchmark_id)

    data = _read_record_json(json_path)
    _check_schema_version(data, "benchmark", SUPPORTED_BENCHMARK_SCHEMA_VERSIONS, benchmark_id)

    return PublishedBenchmark.from_dict(data)


def list_benchmarks(storage_root) -> Tuple[PublishedBenchmark, ...]:

    storage_root = Path(storage_root)
    benchmarks = []

    for row in read_benchmark_catalog_rows(storage_root):

        benchmark_id = row.get("benchmark_id")

        if not benchmark_id:
            continue

        try:
            benchmarks.append(load_benchmark(benchmark_id, storage_root))
        except FileNotFoundError:
            continue

    return tuple(benchmarks)
