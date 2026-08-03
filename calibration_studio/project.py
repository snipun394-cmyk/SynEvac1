import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from calibration_benchmark import ParameterCandidate

from calibration_studio.session import CalibrationSession


# =====================================================
# Calibration Studio Phase 1 -- Core Architecture.
#
# CalibrationProject is the object model for one research thread --
# Phase 2 of the approved persistent data model, minus persistence: a
# Project exists only in the creating process's memory in this phase
# (Phase 2 of the milestone plan, not this document's own Phase 2, adds
# durable storage -- explicitly out of scope here).
#
# Like CalibrationSession, this is a plain mutable object with an
# explicit `version` counter rather than a frozen dataclass +
# dataclasses.replace() chain -- see session.py's own docstring for the
# identical reasoning. `version` is bumped on every mutation, giving a
# cheap, storage-independent answer to "has anything about this project
# changed since I last looked" that a future persistence layer can
# reuse directly for optimistic-concurrency checks, without this phase
# needing to anticipate what that layer's own storage format will be.
# =====================================================


class ProjectStatus(Enum):

    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    CLOSED = "CLOSED"
    ARCHIVED = "ARCHIVED"


class ProjectNotActiveError(Exception):
    pass


def _utc_now_iso() -> str:

    return datetime.now(timezone.utc).isoformat()


class CalibrationProject:

    def __init__(self, *, name: str, description: str = "", tags: Tuple[str, ...] = ()):

        self._project_id = str(uuid.uuid4())
        self._created_at = _utc_now_iso()

        self.name = name
        self.description = description
        self._tags: List[str] = list(tags)
        self._status = ProjectStatus.ACTIVE
        self._updated_at = self._created_at
        self._version = 1

        self._benchmark_ids: List[str] = []
        self._sessions: List[CalibrationSession] = []

        self.extra: Dict[str, Any] = {}

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
        return tuple(self._sessions)

    @property
    def session_ids(self) -> Tuple[str, ...]:
        return tuple(session.session_id for session in self._sessions)

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
        self._touch()

        return session

    def get_session(self, session_id: str) -> Optional[CalibrationSession]:

        for session in self._sessions:
            if session.session_id == session_id:
                return session

        return None

    # =====================================================

    def to_dict(self) -> dict:

        return {
            "project_id": self._project_id,
            "name": self.name,
            "description": self.description,
            "status": self._status.value,
            "created_at": self._created_at,
            "updated_at": self._updated_at,
            "version": self._version,
            "tags": list(self._tags),
            "benchmark_ids": list(self._benchmark_ids),
            "session_ids": list(self.session_ids),
            "extra": dict(self.extra),
        }

    def __repr__(self) -> str:

        return (
            f"CalibrationProject(project_id={self._project_id!r}, name={self.name!r}, "
            f"status={self._status.value}, sessions={len(self._sessions)})"
        )
