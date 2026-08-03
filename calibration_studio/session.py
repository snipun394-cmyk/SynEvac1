import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional

from calibration_benchmark import CalibrationBenchmarkResult, ParameterCandidate

from calibration_studio.git_provenance import GitProvenance, capture_git_provenance


# =====================================================
# Calibration Studio Phase 1 -- Core Architecture.
#
# CalibrationSession is the object model for one reproducible unit of
# calibration work -- Phase 3 of the approved persistent data model,
# minus persistence and execution: no file is ever written by this
# class, and nothing in it calls calibration_benchmark.
# run_calibration_benchmark()/run_with_overrides() or any other
# simulation entry point. `result` exists purely as a slot a LATER
# phase's execution code will populate by calling this package's own
# real, unmodified calibration_benchmark.CalibrationBenchmarkResult --
# never a parallel result type, so `reproducible` (below) can delegate
# to it directly rather than recomputing anything calibration_benchmark
# already computes.
#
# Deliberately a plain, mutable object with an explicit status state
# machine -- not a frozen dataclass + dataclasses.replace() (this
# codebase's usual convention for VALUE objects like MetricSample).
# A Session has identity and a lifecycle spanning many state
# transitions over real wall-clock time (queued, running, reviewed,
# reported); modeling that as a sequence of `replace()`-produced
# siblings would push tracking "which one is current" onto every
# caller. mark_running()/mark_completed()/mark_failed()/
# mark_cancelled() are the only way status changes, each validating the
# transition and raising rather than silently succeeding on an invalid
# one -- the same "human approval is a real two-step gate, never a
# rubber stamp" discipline already enforced elsewhere in this codebase
# (e.g. Execution Layer's own approve()/reject()).
# =====================================================


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


def _utc_now_iso() -> str:

    return datetime.now(timezone.utc).isoformat()


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
    ):

        # Identity + configuration -- set once, at creation, never
        # reassigned. Enforced by convention (no public setter), the
        # same discipline ScenarioMetadata's own frozen dataclass
        # enforces mechanically; a plain object needs the discipline
        # stated, since a status-machine object can't also be frozen.
        self._session_id = str(uuid.uuid4())
        self._project_id = project_id
        self._benchmark_id = benchmark_id
        self._candidate = candidate
        self._master_seed = master_seed
        self._simulator_id = simulator_id
        self._git_provenance = git_provenance if git_provenance is not None else capture_git_provenance()
        self._created_at = _utc_now_iso()
        self.extra: Dict[str, Any] = dict(extra) if extra else {}

        # Execution state -- the only fields a later execution phase is
        # expected to mutate via the transition methods below.
        self._status = SessionStatus.PENDING
        self._n_scenarios_completed = 0
        self._n_scenarios_total: Optional[int] = None
        self._result: Optional[CalibrationBenchmarkResult] = None
        self._failure_reason: Optional[str] = None
        self._started_at: Optional[str] = None
        self._completed_at: Optional[str] = None

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
        return self._candidate

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

        # Delegates to calibration_benchmark.CalibrationBenchmarkResult's
        # own `reproducible` property -- never recomputed here. None
        # (honestly unknown) until a later phase attaches a result via
        # mark_completed().
        if self._result is None:
            return None

        return self._result.reproducible

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
        return self._result

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
            "session_id": self._session_id,
            "project_id": self._project_id,
            "benchmark_id": self._benchmark_id,
            "candidate": self._candidate.describe() if self._candidate is not None else None,
            "master_seed": self._master_seed,
            "simulator_id": self._simulator_id,
            "git_provenance": self._git_provenance.to_dict(),
            "created_at": self._created_at,
            "status": self._status.value,
            "n_scenarios_completed": self._n_scenarios_completed,
            "n_scenarios_total": self._n_scenarios_total,
            "progress": self.progress,
            "reproducible": self.reproducible,
            "failure_reason": self._failure_reason,
            "started_at": self._started_at,
            "completed_at": self._completed_at,
            "extra": dict(self.extra),
        }

    def __repr__(self) -> str:

        return (
            f"CalibrationSession(session_id={self._session_id!r}, status={self._status.value}, "
            f"benchmark_id={self._benchmark_id!r})"
        )
