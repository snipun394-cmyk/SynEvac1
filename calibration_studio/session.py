import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional

from calibration_benchmark import CalibrationBenchmarkResult, ParameterCandidate

from calibration_studio.git_provenance import GitProvenance, capture_git_provenance


# =====================================================
# Calibration Studio Phase 1 -- Core Architecture; Phase 2 --
# Persistence Layer.
#
# CalibrationSession is the object model for one reproducible unit of
# calibration work -- Phase 3 of the approved persistent data model.
# Nothing in this class calls calibration_benchmark.
# run_calibration_benchmark()/run_with_overrides() or any other
# simulation entry point; `result` is a slot a LATER phase's execution
# code populates by calling this package's own real, unmodified
# calibration_benchmark.CalibrationBenchmarkResult -- never a parallel
# result type.
#
# SNAPSHOT, NOT RECONSTRUCTION -- the one load-bearing constraint this
# phase's persistence design turns on: `calibration_benchmark.
# ParameterCandidate`/`CalibrationBenchmarkResult` have `describe()`/
# `to_dict()` (write-only serialization) but no `from_dict()` anywhere
# in that package (confirmed by direct search). Building one here would
# be adding a genuinely new capability to calibration_benchmark's own
# domain from outside it -- exactly the "duplicate calibration_
# benchmark" this milestone was explicitly told not to do, and squarely
# out of this phase's own scope (persistence for CalibrationProject/
# CalibrationSession, not for calibration_benchmark's types). So a
# session that has been saved and reloaded honestly does NOT get its
# live `candidate`/`result` objects back -- `candidate`/`result` are
# None after a load, and `candidate_snapshot`/`result_snapshot` (plain,
# already-serialized dicts) are what actually survives the round trip.
# `reproducible` reads from whichever of `result`/`result_snapshot` is
# available, so the one fact that matters most for a reopened session
# ("was this run reproducible") is never lost even though the full
# typed result object is.
#
# Deliberately a plain, mutable object with an explicit status state
# machine -- not a frozen dataclass + dataclasses.replace() (this
# codebase's usual convention for VALUE objects like MetricSample).
# mark_running()/mark_completed()/mark_failed()/mark_cancelled() are
# the only way status changes during a LIVE run, each validating the
# transition and raising rather than silently succeeding on an invalid
# one. from_dict() (below) deliberately does NOT go through these --
# it is restoring an already-valid historical state, not performing a
# new live transition, so replaying validation against it would be
# both pointless and actively wrong (a session legitimately reloaded
# straight into COMPLETED never passed through a live mark_running()
# call in this process).
# =====================================================


SCHEMA_VERSION = "calibration_studio_session/1"
SUPPORTED_SCHEMA_VERSIONS = frozenset({SCHEMA_VERSION})


class SessionStatus(Enum):

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


_VALID_TRANSITIONS = {
    SessionStatus.PENDING: frozenset({SessionStatus.RUNNING, SessionStatus.CANCELLED}),
    SessionStatus.RUNNING: frozenset({SessionStatus.COMPLETED, SessionStatus.FAILED, SessionStatus.CANCELLED}),
    SessionStatus.COMPLETED: frozenset(),
    SessionStatus.FAILED: frozenset(),
    SessionStatus.CANCELLED: frozenset(),
}


class InvalidSessionTransitionError(Exception):
    pass


class CorruptedSessionRecordError(Exception):
    pass


def _utc_now_iso() -> str:

    return datetime.now(timezone.utc).isoformat()


# Every top-level key this schema version's to_dict() writes -- used by
# from_dict() to sweep anything ELSE present in a loaded dict (a field
# a newer schema version added, one this version has never heard of)
# into `extra` rather than silently discarding it. Kept as one
# explicit tuple, not derived from to_dict() itself, so adding a new
# recognised field is a deliberate two-place edit (schema + this set),
# never an accidental one.
_KNOWN_TOP_LEVEL_KEYS = frozenset({
    "schema_version", "session_id", "project_id", "benchmark_id", "candidate",
    "master_seed", "simulator_id", "git_provenance", "created_at", "status",
    "n_scenarios_completed", "n_scenarios_total", "progress", "reproducible",
    "result", "failure_reason", "started_at", "completed_at", "extra",
})


class CalibrationSession:

    def __init__(
        self,
        *,
        project_id: Optional[str] = None,
        benchmark_id: Optional[str] = None,
        candidate: Optional[ParameterCandidate] = None,
        master_seed: Optional[int] = None,
        simulator_id: str = "synevac",
        git_provenance: Optional[GitProvenance] = None,
        extra: Optional[Dict[str, Any]] = None,
        # Restoration-only parameters -- a live caller creating a NEW
        # session never supplies any of these; from_dict() is the one
        # caller that does, to restore an already-existing session's
        # identity and execution state exactly as persisted rather than
        # minting a fresh one. Kept on the same __init__ (one
        # construction path, not a second bypass constructor) so there
        # is exactly one place that decides what a valid
        # CalibrationSession looks like.
        session_id: Optional[str] = None,
        created_at: Optional[str] = None,
        candidate_snapshot: Optional[dict] = None,
        status: SessionStatus = SessionStatus.PENDING,
        n_scenarios_completed: int = 0,
        n_scenarios_total: Optional[int] = None,
        result: Optional[CalibrationBenchmarkResult] = None,
        result_snapshot: Optional[dict] = None,
        failure_reason: Optional[str] = None,
        started_at: Optional[str] = None,
        completed_at: Optional[str] = None,
    ):

        # Identity + configuration -- set once, at creation (or restored
        # verbatim by from_dict()), never reassigned afterward. Enforced
        # by convention (no public setter), the same discipline
        # ScenarioMetadata's own frozen dataclass enforces mechanically;
        # a plain object needs the discipline stated, since a status-
        # machine object can't also be frozen.
        self._session_id = session_id if session_id is not None else str(uuid.uuid4())
        self._project_id = project_id
        self._benchmark_id = benchmark_id
        self._candidate = candidate
        self._candidate_snapshot = (
            candidate.describe() if candidate is not None
            else candidate_snapshot
        )
        self._master_seed = master_seed
        self._simulator_id = simulator_id
        self._git_provenance = git_provenance if git_provenance is not None else capture_git_provenance()
        self._created_at = created_at if created_at is not None else _utc_now_iso()
        self.extra: Dict[str, Any] = dict(extra) if extra else {}

        # Execution state.
        self._status = status
        self._n_scenarios_completed = n_scenarios_completed
        self._n_scenarios_total = n_scenarios_total
        self._result = result
        self._result_snapshot = result.to_dict() if result is not None else result_snapshot
        self._failure_reason = failure_reason
        self._started_at = started_at
        self._completed_at = completed_at

    # =====================================================
    # Identity (read-only properties -- see __init__'s own comment)
    # =====================================================

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def project_id(self) -> Optional[str]:
        return self._project_id

    @property
    def benchmark_id(self) -> Optional[str]:
        return self._benchmark_id

    @property
    def candidate(self) -> Optional[ParameterCandidate]:

        # None after a load-from-disk -- see this module's own
        # docstring. candidate_snapshot (below) is what actually
        # survives a round trip.
        return self._candidate

    @property
    def candidate_snapshot(self) -> Optional[dict]:
        return self._candidate_snapshot

    @property
    def created_at(self) -> str:
        return self._created_at

    # =====================================================
    # Reproducibility metadata
    # =====================================================

    @property
    def master_seed(self) -> Optional[int]:
        return self._master_seed

    @property
    def simulator_id(self) -> str:
        return self._simulator_id

    @property
    def git_commit_hash(self) -> Optional[str]:
        return self._git_provenance.commit_hash

    @property
    def git_dirty(self) -> Optional[bool]:
        return self._git_provenance.dirty

    @property
    def reproducible(self) -> Optional[bool]:

        # Prefers the live result's own property (authoritative,
        # freshly computed); falls back to the persisted snapshot's own
        # "reproducible" value after a load, when only the snapshot
        # survived -- see this module's own docstring. Still honestly
        # None when neither exists.
        if self._result is not None:
            return self._result.reproducible

        if self._result_snapshot is not None:
            return self._result_snapshot.get("reproducible")

        return None

    # =====================================================
    # Execution state / progress / session status
    # =====================================================

    @property
    def status(self) -> SessionStatus:
        return self._status

    @property
    def n_scenarios_completed(self) -> int:
        return self._n_scenarios_completed

    @property
    def n_scenarios_total(self) -> Optional[int]:
        return self._n_scenarios_total

    @property
    def progress(self) -> Optional[float]:

        # None (not 0.0) when the total isn't known yet -- an honest
        # "progress cannot be computed", never a misleading 0%.
        if not self._n_scenarios_total:
            return None

        return min(1.0, self._n_scenarios_completed / self._n_scenarios_total)

    @property
    def result(self) -> Optional[CalibrationBenchmarkResult]:

        # None after a load-from-disk -- see this module's own
        # docstring. result_snapshot (below) is what actually survives
        # a round trip.
        return self._result

    @property
    def result_snapshot(self) -> Optional[dict]:
        return self._result_snapshot

    @property
    def failure_reason(self) -> Optional[str]:
        return self._failure_reason

    @property
    def started_at(self) -> Optional[str]:
        return self._started_at

    @property
    def completed_at(self) -> Optional[str]:
        return self._completed_at

    # =====================================================
    # State transitions -- no execution logic behind any of these; a
    # later phase's runner calls them around its own real work, this
    # object only ever validates and records the transition.
    # =====================================================

    def _transition(self, new_status: SessionStatus) -> None:

        allowed = _VALID_TRANSITIONS[self._status]

        if new_status not in allowed:
            raise InvalidSessionTransitionError(
                f"Cannot transition session {self._session_id} from {self._status.value} "
                f"to {new_status.value} (allowed: {sorted(s.value for s in allowed)})",
            )

        self._status = new_status

    def mark_running(self, n_scenarios_total: Optional[int] = None) -> None:

        self._transition(SessionStatus.RUNNING)
        self._n_scenarios_total = n_scenarios_total
        self._started_at = _utc_now_iso()

    def update_progress(self, n_scenarios_completed: int) -> None:

        if self._status is not SessionStatus.RUNNING:
            raise InvalidSessionTransitionError(
                f"Cannot update progress on session {self._session_id}: status is "
                f"{self._status.value}, not RUNNING",
            )

        self._n_scenarios_completed = n_scenarios_completed

    def mark_completed(self, result: CalibrationBenchmarkResult) -> None:

        self._transition(SessionStatus.COMPLETED)
        self._result = result
        self._result_snapshot = result.to_dict()
        self._completed_at = _utc_now_iso()

    def mark_failed(self, reason: str) -> None:

        self._transition(SessionStatus.FAILED)
        self._failure_reason = reason
        self._completed_at = _utc_now_iso()

    def mark_cancelled(self) -> None:

        self._transition(SessionStatus.CANCELLED)
        self._completed_at = _utc_now_iso()

    # =====================================================

    def to_dict(self) -> dict:

        return {
            "schema_version": SCHEMA_VERSION,
            "session_id": self._session_id,
            "project_id": self._project_id,
            "benchmark_id": self._benchmark_id,
            "candidate": self._candidate_snapshot,
            "master_seed": self._master_seed,
            "simulator_id": self._simulator_id,
            "git_provenance": self._git_provenance.to_dict(),
            "created_at": self._created_at,
            "status": self._status.value,
            "n_scenarios_completed": self._n_scenarios_completed,
            "n_scenarios_total": self._n_scenarios_total,
            "progress": self.progress,
            "reproducible": self.reproducible,
            "result": self._result_snapshot,
            "failure_reason": self._failure_reason,
            "started_at": self._started_at,
            "completed_at": self._completed_at,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CalibrationSession":

        # No schema_version check here -- that is the persistence
        # layer's job (calibration_studio/storage.py), which is the one
        # place that knows whether an incompatible version should raise
        # or how to migrate. This method's own job is narrower: given a
        # dict this schema version (or an honest subset of it) can
        # already interpret, reconstruct a session from it, tolerating
        # missing fields (old file, field added later) and unknown
        # extra fields (newer file, field this version has never heard
        # of) without raising on either.

        raw_git = data.get("git_provenance") or {}
        git_provenance = GitProvenance(
            commit_hash=raw_git.get("commit_hash"), dirty=raw_git.get("dirty"),
        )

        raw_status = data.get("status", SessionStatus.PENDING.value)
        try:
            status = SessionStatus(raw_status)
        except ValueError:
            raise CorruptedSessionRecordError(
                f"Unrecognised session status {raw_status!r} in stored session "
                f"{data.get('session_id')!r}",
            )

        extra = dict(data.get("extra") or {})

        # Forward compatibility: fold any top-level key this version
        # doesn't recognise into `extra`, rather than discarding it --
        # a session saved by a future schema version and reloaded by
        # this one keeps that data around (recoverable, re-savable),
        # instead of losing it silently.
        for key, value in data.items():
            if key not in _KNOWN_TOP_LEVEL_KEYS:
                extra[key] = value

        return cls(
            project_id=data.get("project_id"),
            benchmark_id=data.get("benchmark_id"),
            candidate=None,
            candidate_snapshot=data.get("candidate"),
            master_seed=data.get("master_seed"),
            simulator_id=data.get("simulator_id", "synevac"),
            git_provenance=git_provenance,
            extra=extra,
            session_id=data.get("session_id"),
            created_at=data.get("created_at"),
            status=status,
            n_scenarios_completed=data.get("n_scenarios_completed", 0),
            n_scenarios_total=data.get("n_scenarios_total"),
            result=None,
            result_snapshot=data.get("result"),
            failure_reason=data.get("failure_reason"),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
        )

    def __repr__(self) -> str:

        return (
            f"CalibrationSession(session_id={self._session_id!r}, status={self._status.value}, "
            f"benchmark_id={self._benchmark_id!r})"
        )
