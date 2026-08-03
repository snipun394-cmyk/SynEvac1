import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

from calibration_benchmark import ParameterCandidate

from calibration_studio.session import CalibrationSession


# =====================================================
# Calibration Studio Phase 1 -- Core Architecture; Phase 2 --
# Persistence Layer.
#
# CalibrationProject is the object model for one research thread --
# Phase 2 of the approved persistent data model.
#
# Like CalibrationSession, this is a plain mutable object with an
# explicit `version` counter rather than a frozen dataclass +
# dataclasses.replace() chain -- see session.py's own docstring for the
# identical reasoning. `version` is bumped on every mutation, giving a
# cheap, storage-independent answer to "has anything about this project
# changed since I last looked."
#
# session_ids is tracked as its OWN authoritative list, independent of
# which CalibrationSession objects happen to be resolved in memory
# (`sessions`) -- necessary for persistence: a freshly-loaded project
# knows every session id that belongs to it (persisted directly) before
# calibration_studio/storage.py has necessarily loaded each of those
# sessions' own files. `sessions` may therefore be a strict subset of
# `session_ids` right after from_dict() -- see _attach_loaded_sessions()'s
# own docstring, the one sanctioned way the persistence layer closes
# that gap.
# =====================================================


SCHEMA_VERSION = "calibration_studio_project/1"
SUPPORTED_SCHEMA_VERSIONS = frozenset({SCHEMA_VERSION})


class ProjectStatus(Enum):

    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    CLOSED = "CLOSED"
    ARCHIVED = "ARCHIVED"


class ProjectNotActiveError(Exception):
    pass


class CorruptedProjectRecordError(Exception):
    pass


def _utc_now_iso() -> str:

    return datetime.now(timezone.utc).isoformat()


_KNOWN_TOP_LEVEL_KEYS = frozenset({
    "schema_version", "project_id", "name", "description", "status", "created_at",
    "updated_at", "version", "tags", "benchmark_ids", "session_ids", "extra",
})


class CalibrationProject:

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        tags: Tuple[str, ...] = (),
        extra: Optional[Dict[str, Any]] = None,
        # Restoration-only parameters -- see CalibrationSession.__init__'s
        # identical convention and reasoning.
        project_id: Optional[str] = None,
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None,
        status: ProjectStatus = ProjectStatus.ACTIVE,
        version: int = 1,
        benchmark_ids: Sequence[str] = (),
        session_ids: Sequence[str] = (),
    ):

        self._project_id = project_id if project_id is not None else str(uuid.uuid4())
        self._created_at = created_at if created_at is not None else _utc_now_iso()

        self.name = name
        self.description = description
        self._tags: List[str] = list(tags)
        self._status = status
        self._updated_at = updated_at if updated_at is not None else self._created_at
        self._version = version

        self._benchmark_ids: List[str] = list(benchmark_ids)
        self._session_ids: List[str] = list(session_ids)
        self._sessions: List[CalibrationSession] = []

        self.extra: Dict[str, Any] = dict(extra) if extra else {}

    # =====================================================
    # Identity (read-only -- see session.py's identical convention)
    # =====================================================

    @property
    def project_id(self) -> str:
        return self._project_id

    @property
    def created_at(self) -> str:
        return self._created_at

    # =====================================================
    # Metadata / lifecycle
    # =====================================================

    @property
    def status(self) -> ProjectStatus:
        return self._status

    @property
    def updated_at(self) -> str:
        return self._updated_at

    @property
    def version(self) -> int:
        return self._version

    @property
    def tags(self) -> Tuple[str, ...]:
        return tuple(self._tags)

    def _touch(self) -> None:

        self._updated_at = _utc_now_iso()
        self._version += 1

    def rename(self, name: str) -> None:

        self.name = name
        self._touch()

    def set_description(self, description: str) -> None:

        self.description = description
        self._touch()

    def add_tag(self, tag: str) -> None:

        if tag not in self._tags:
            self._tags.append(tag)
            self._touch()

    def remove_tag(self, tag: str) -> None:

        if tag in self._tags:
            self._tags.remove(tag)
            self._touch()

    def set_status(self, status: ProjectStatus) -> None:

        self._status = status
        self._touch()

    # =====================================================
    # Benchmark references -- plain id strings only; no
    # PublishedBenchmark type exists yet (Benchmark Library is
    # explicitly out of scope for this milestone). append-only by
    # convention (matches scenario_storage.catalog's own "append,
    # never remove" rule for the same kind of history-bearing list).
    # =====================================================

    @property
    def benchmark_ids(self) -> Tuple[str, ...]:
        return tuple(self._benchmark_ids)

    def add_benchmark_id(self, benchmark_id: str) -> None:

        if benchmark_id not in self._benchmark_ids:
            self._benchmark_ids.append(benchmark_id)
            self._touch()

    # =====================================================
    # Sessions
    # =====================================================

    @property
    def sessions(self) -> Tuple[CalibrationSession, ...]:

        # May be a strict subset of session_ids -- see this module's
        # own docstring.
        return tuple(self._sessions)

    @property
    def session_ids(self) -> Tuple[str, ...]:

        # The authoritative list -- independent of how many of those
        # ids currently have a resolved CalibrationSession object in
        # `sessions` (see this module's own docstring).
        return tuple(self._session_ids)

    def create_session(
        self,
        *,
        benchmark_id: Optional[str] = None,
        candidate: Optional[ParameterCandidate] = None,
        master_seed: Optional[int] = None,
    ) -> CalibrationSession:

        # A Project's own lifecycle gates whether new work can be added
        # to it -- creating a session against a CLOSED/ARCHIVED/PAUSED
        # project is a caller error, raised rather than silently
        # allowed (the same "never silently succeed on an invalid
        # operation" discipline session.py's own transition methods
        # enforce).
        if self._status is not ProjectStatus.ACTIVE:
            raise ProjectNotActiveError(
                f"Cannot create a session on project {self._project_id} ({self.name!r}): "
                f"status is {self._status.value}, not ACTIVE",
            )

        session = CalibrationSession(
            project_id=self._project_id, benchmark_id=benchmark_id,
            candidate=candidate, master_seed=master_seed,
        )

        self._sessions.append(session)
        self._session_ids.append(session.session_id)
        self._touch()

        return session

    def get_session(self, session_id: str) -> Optional[CalibrationSession]:

        for session in self._sessions:
            if session.session_id == session_id:
                return session

        return None

    def _attach_loaded_sessions(self, sessions: Sequence[CalibrationSession]) -> None:

        # Persistence-layer use only (calibration_studio/storage.py's
        # own load_project()) -- resolves the gap from_dict() alone
        # cannot close (session_ids is known immediately from this
        # project's own file; the CalibrationSession objects those ids
        # name live in separate files only the storage layer knows how
        # to find). Does not bump version/updated_at -- attaching an
        # already-persisted session's own already-persisted state is
        # not a mutation of this project, it is completing a load.
        # Silently ignores a session whose id isn't actually one of
        # this project's own session_ids, rather than raising -- a
        # defensive guard against a caller wiring the wrong sessions
        # in, not a scenario this phase's own storage.py is expected to
        # trigger.

        by_id = {session.session_id: session for session in sessions if session.session_id in self._session_ids}

        self._sessions = [by_id[session_id] for session_id in self._session_ids if session_id in by_id]

    # =====================================================

    def to_dict(self) -> dict:

        return {
            "schema_version": SCHEMA_VERSION,
            "project_id": self._project_id,
            "name": self.name,
            "description": self.description,
            "status": self._status.value,
            "created_at": self._created_at,
            "updated_at": self._updated_at,
            "version": self._version,
            "tags": list(self._tags),
            "benchmark_ids": list(self._benchmark_ids),
            "session_ids": list(self._session_ids),
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CalibrationProject":

        # No schema_version check here -- see CalibrationSession.
        # from_dict()'s identical comment; that is calibration_studio/
        # storage.py's own job.

        raw_status = data.get("status", ProjectStatus.ACTIVE.value)
        try:
            status = ProjectStatus(raw_status)
        except ValueError:
            raise CorruptedProjectRecordError(
                f"Unrecognised project status {raw_status!r} in stored project "
                f"{data.get('project_id')!r}",
            )

        extra = dict(data.get("extra") or {})

        # Forward compatibility -- identical reasoning to
        # CalibrationSession.from_dict()'s own handling.
        for key, value in data.items():
            if key not in _KNOWN_TOP_LEVEL_KEYS:
                extra[key] = value

        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            tags=tuple(data.get("tags") or ()),
            extra=extra,
            project_id=data.get("project_id"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            status=status,
            version=data.get("version", 1),
            benchmark_ids=tuple(data.get("benchmark_ids") or ()),
            session_ids=tuple(data.get("session_ids") or ()),
        )

    def __repr__(self) -> str:

        return (
            f"CalibrationProject(project_id={self._project_id!r}, name={self.name!r}, "
            f"status={self._status.value}, sessions={len(self._sessions)})"
        )
