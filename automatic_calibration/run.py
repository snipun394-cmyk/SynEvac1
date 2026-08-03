import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional, Tuple

from automatic_calibration.budget import AutoCalibrationBudget
from automatic_calibration.search_space import SearchSpace


# =====================================================
# Automatic Calibration Engine, Phase 1 -- Core Architecture.
#
# AutoCalibrationRun is the one genuinely new persistent record type
# this package introduces -- search-level metadata, never a duplicate
# of CalibrationSession's own execution state. Every individual
# evaluation IS already a real CalibrationSession, saved through
# Calibration Studio's own existing save_session() -- this class only
# records WHICH sessions (by id, in the order they were proposed) one
# search run produced, plus the search's own configuration and outcome.
#
# Same mutable-object-with-an-explicit-status-state-machine shape as
# calibration_studio.session.CalibrationSession, deliberately mirrored
# field-for-field where the two classes' concerns overlap (identity/
# configuration set once at construction; execution state changed only
# through validated transition methods; forward-compatible from_dict()
# that folds unrecognised top-level keys into `extra` rather than
# discarding them).
#
# One genuine departure from CalibrationSession, worth stating
# explicitly: EVERY field stored here is already a plain, JSON-safe
# snapshot by construction. search_space_description is plain dicts
# (never the live SearchSpace, whose ParameterDimension.build callables
# can never be reconstructed from JSON -- the same "live object,
# snapshot only" constraint ParameterCandidate itself has). budget,
# however, is a real AutoCalibrationBudget object even after a reload --
# AutoCalibrationBudget (unlike ParameterCandidate) holds no live,
# unreconstructable state at all, so its own to_dict()/from_dict() is a
# genuine inverse pair, and there is no reason to store only a snapshot
# of it. from_dict() below is therefore a full, honest reconstruction
# of an AutoCalibrationRun -- not a partial "candidate/result are None"
# restore the way CalibrationSession's own from_dict() must be.
# =====================================================


SCHEMA_VERSION = "automatic_calibration_run/1"
SUPPORTED_SCHEMA_VERSIONS = frozenset({SCHEMA_VERSION})


class AutoCalibrationRunStatus(Enum):

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


_VALID_TRANSITIONS = {
    AutoCalibrationRunStatus.PENDING: frozenset(
        {AutoCalibrationRunStatus.RUNNING, AutoCalibrationRunStatus.CANCELLED},
    ),
    AutoCalibrationRunStatus.RUNNING: frozenset({
        AutoCalibrationRunStatus.COMPLETED, AutoCalibrationRunStatus.FAILED, AutoCalibrationRunStatus.CANCELLED,
    }),
    AutoCalibrationRunStatus.COMPLETED: frozenset(),
    AutoCalibrationRunStatus.FAILED: frozenset(),
    AutoCalibrationRunStatus.CANCELLED: frozenset(),
}


class InvalidRunTransitionError(Exception):
    pass


class CorruptedRunRecordError(Exception):
    pass


def _utc_now_iso() -> str:

    return datetime.now(timezone.utc).isoformat()


_KNOWN_TOP_LEVEL_KEYS = frozenset({
    "schema_version", "run_id", "project_id", "benchmark_id", "search_space_description",
    "objective_description", "objective_direction", "strategy_description", "budget", "search_seed",
    "created_at", "status", "session_ids", "best_session_id", "best_score", "failure_reason",
    "started_at", "completed_at", "extra",
})


class AutoCalibrationRun:

    def __init__(
        self,
        *,
        project_id: str,
        search_space: Optional[SearchSpace] = None,
        objective_description: Optional[str] = None,
        objective_direction: Optional[str] = None,
        strategy_description: Optional[str] = None,
        budget: Optional[AutoCalibrationBudget] = None,
        benchmark_id: Optional[str] = None,
        search_seed: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
        # Restoration-only parameters -- a live caller creating a NEW
        # run never supplies any of these; from_dict() is the one
        # caller that does. Same convention as CalibrationSession's own
        # constructor.
        run_id: Optional[str] = None,
        created_at: Optional[str] = None,
        search_space_description: Optional[Tuple[dict, ...]] = None,
        status: AutoCalibrationRunStatus = AutoCalibrationRunStatus.PENDING,
        session_ids: Tuple[str, ...] = (),
        best_session_id: Optional[str] = None,
        best_score: Optional[float] = None,
        failure_reason: Optional[str] = None,
        started_at: Optional[str] = None,
        completed_at: Optional[str] = None,
    ):

        self._run_id = run_id if run_id is not None else str(uuid.uuid4())
        self._project_id = project_id
        self._benchmark_id = benchmark_id
        self._search_space_description = (
            search_space.describe() if search_space is not None else tuple(search_space_description or ())
        )
        self._objective_description = objective_description
        self._objective_direction = objective_direction
        self._strategy_description = strategy_description
        self._budget = budget
        self._search_seed = search_seed
        self._created_at = created_at if created_at is not None else _utc_now_iso()
        self.extra: Dict[str, Any] = dict(extra) if extra else {}

        self._status = status
        self._session_ids = tuple(session_ids)
        self._best_session_id = best_session_id
        self._best_score = best_score
        self._failure_reason = failure_reason
        self._started_at = started_at
        self._completed_at = completed_at

    # =====================================================
    # Identity / configuration (read-only -- see __init__'s own comment)
    # =====================================================

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def project_id(self) -> str:
        return self._project_id

    @property
    def benchmark_id(self) -> Optional[str]:
        return self._benchmark_id

    @property
    def search_space_description(self) -> Tuple[dict, ...]:
        return self._search_space_description

    @property
    def objective_description(self) -> Optional[str]:
        return self._objective_description

    @property
    def objective_direction(self) -> Optional[str]:
        return self._objective_direction

    @property
    def strategy_description(self) -> Optional[str]:
        return self._strategy_description

    @property
    def budget(self) -> Optional[AutoCalibrationBudget]:
        return self._budget

    @property
    def search_seed(self) -> Optional[int]:
        return self._search_seed

    @property
    def created_at(self) -> str:
        return self._created_at

    # =====================================================
    # Execution state
    # =====================================================

    @property
    def status(self) -> AutoCalibrationRunStatus:
        return self._status

    @property
    def session_ids(self) -> Tuple[str, ...]:
        return self._session_ids

    @property
    def n_evaluations(self) -> int:
        return len(self._session_ids)

    @property
    def best_session_id(self) -> Optional[str]:
        return self._best_session_id

    @property
    def best_score(self) -> Optional[float]:
        return self._best_score

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
    # State transitions -- no execution logic behind any of these;
    # AutoCalibrationEngine calls them around its own real work, this
    # object only ever validates and records the transition, exactly
    # like CalibrationSession.
    # =====================================================

    def _transition(self, new_status: AutoCalibrationRunStatus) -> None:

        allowed = _VALID_TRANSITIONS[self._status]

        if new_status not in allowed:
            raise InvalidRunTransitionError(
                f"Cannot transition run {self._run_id} from {self._status.value} to "
                f"{new_status.value} (allowed: {sorted(s.value for s in allowed)})",
            )

        self._status = new_status

    def mark_running(self) -> None:

        self._transition(AutoCalibrationRunStatus.RUNNING)
        self._started_at = _utc_now_iso()

    def record_evaluation(self, session_id: str, score: Optional[float]) -> None:

        if self._status is not AutoCalibrationRunStatus.RUNNING:
            raise InvalidRunTransitionError(
                f"Cannot record an evaluation on run {self._run_id}: status is "
                f"{self._status.value}, not RUNNING",
            )

        self._session_ids = self._session_ids + (session_id,)

        if score is None:
            # Not scoreable (a failed session, or a metric this
            # session's result never produced) -- still recorded above
            # as an evaluation that happened, just never a candidate for
            # best_session_id/best_score.
            return

        if self._best_score is None or self._is_better(score, self._best_score):
            self._best_score = score
            self._best_session_id = session_id

    def _is_better(self, score: float, current_best: float) -> bool:

        if self._objective_direction == "maximize":
            return score > current_best

        return score < current_best

    def mark_completed(self) -> None:

        self._transition(AutoCalibrationRunStatus.COMPLETED)
        self._completed_at = _utc_now_iso()

    def mark_failed(self, reason: str) -> None:

        self._transition(AutoCalibrationRunStatus.FAILED)
        self._failure_reason = reason
        self._completed_at = _utc_now_iso()

    def mark_cancelled(self) -> None:

        self._transition(AutoCalibrationRunStatus.CANCELLED)
        self._completed_at = _utc_now_iso()

    # =====================================================

    def to_dict(self) -> dict:

        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": self._run_id,
            "project_id": self._project_id,
            "benchmark_id": self._benchmark_id,
            "search_space_description": [dict(d) for d in self._search_space_description],
            "objective_description": self._objective_description,
            "objective_direction": self._objective_direction,
            "strategy_description": self._strategy_description,
            "budget": self._budget.to_dict() if self._budget is not None else None,
            "search_seed": self._search_seed,
            "created_at": self._created_at,
            "status": self._status.value,
            "session_ids": list(self._session_ids),
            "best_session_id": self._best_session_id,
            "best_score": self._best_score,
            "failure_reason": self._failure_reason,
            "started_at": self._started_at,
            "completed_at": self._completed_at,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AutoCalibrationRun":

        # No schema_version check here -- that is the persistence
        # layer's job (automatic_calibration/storage.py), exactly the
        # same division of responsibility CalibrationSession.from_dict()
        # already establishes.

        raw_status = data.get("status", AutoCalibrationRunStatus.PENDING.value)
        try:
            status = AutoCalibrationRunStatus(raw_status)
        except ValueError:
            raise CorruptedRunRecordError(
                f"Unrecognised run status {raw_status!r} in stored run {data.get('run_id')!r}",
            )

        raw_budget = data.get("budget")
        budget = AutoCalibrationBudget.from_dict(raw_budget) if raw_budget is not None else None

        extra = dict(data.get("extra") or {})

        for key, value in data.items():
            if key not in _KNOWN_TOP_LEVEL_KEYS:
                extra[key] = value

        return cls(
            project_id=data.get("project_id"),
            benchmark_id=data.get("benchmark_id"),
            objective_description=data.get("objective_description"),
            objective_direction=data.get("objective_direction"),
            strategy_description=data.get("strategy_description"),
            budget=budget,
            search_seed=data.get("search_seed"),
            extra=extra,
            run_id=data.get("run_id"),
            created_at=data.get("created_at"),
            search_space_description=tuple(data.get("search_space_description") or ()),
            status=status,
            session_ids=tuple(data.get("session_ids") or ()),
            best_session_id=data.get("best_session_id"),
            best_score=data.get("best_score"),
            failure_reason=data.get("failure_reason"),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
        )

    def __repr__(self) -> str:

        return (
            f"AutoCalibrationRun(run_id={self._run_id!r}, status={self._status.value}, "
            f"n_evaluations={len(self._session_ids)})"
        )
