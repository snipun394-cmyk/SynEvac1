from pathlib import Path


# =====================================================
# Calibration Studio Phase 2 -- Persistence Layer.
#
# Mirrors scenario_storage/paths.py's own role (the one place storage
# layout is decided) and its "one JSON file per record + one running
# catalog" shape, with one deliberate departure: NO sharded
# subdirectories. scenario_storage shards by a fixed prefix of
# scenario_id specifically because a single campaign can accept tens of
# thousands of scenarios (docs/architecture/scenario_engine.md §11) --
# nothing in this milestone's own scale (a handful of projects, at most
# a few hundred sessions per project) approaches that. Following that
# optimization here regardless would be copying a convention past the
# problem it was built to solve; keeping every project/session
# resolution behind this module's own functions (exactly as
# scenario_storage does) means sharding could be added later, inside
# this file alone, without any caller elsewhere needing to change.
# =====================================================

PROJECTS_SUBDIRECTORY = "projects"
SESSIONS_SUBDIRECTORY = "sessions"
BENCHMARKS_SUBDIRECTORY = "benchmarks"

PROJECTS_CATALOG_FILENAME = "projects_catalog.csv"
SESSIONS_CATALOG_FILENAME = "sessions_catalog.csv"
BENCHMARKS_CATALOG_FILENAME = "benchmarks_catalog.csv"


def project_json_filename(project_id: str) -> str:

    return f"{project_id}.json"


def project_json_path(storage_root, project_id: str) -> Path:

    return Path(storage_root) / PROJECTS_SUBDIRECTORY / project_json_filename(project_id)


def session_json_filename(session_id: str) -> str:

    return f"{session_id}.json"


def session_json_path(storage_root, session_id: str) -> Path:

    return Path(storage_root) / SESSIONS_SUBDIRECTORY / session_json_filename(session_id)


def projects_catalog_path(storage_root) -> Path:

    return Path(storage_root) / PROJECTS_CATALOG_FILENAME


def sessions_catalog_path(storage_root) -> Path:

    return Path(storage_root) / SESSIONS_CATALOG_FILENAME


def benchmark_json_filename(benchmark_id: str) -> str:

    return f"{benchmark_id}.json"


def benchmark_json_path(storage_root, benchmark_id: str) -> Path:

    return Path(storage_root) / BENCHMARKS_SUBDIRECTORY / benchmark_json_filename(benchmark_id)


def benchmarks_catalog_path(storage_root) -> Path:

    return Path(storage_root) / BENCHMARKS_CATALOG_FILENAME
