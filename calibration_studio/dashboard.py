from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from calibration_benchmark import Verdict, recommend

from calibration_studio.benchmark import PublishedBenchmark, ValidationStatus
from calibration_studio.session import CalibrationSession


# =====================================================
# Calibration Studio Phase 6 -- Validation Dashboard.
#
# Read-only, stateless aggregation: generate_validation_dashboard()
# takes whatever PublishedBenchmark/CalibrationSession sequences its
# caller already has (from PublishedBenchmarkLibrary/CalibrationStudio,
# both unchanged) and computes a plain snapshot from them. Nothing here
# ever calls a save_*()/register()/mark_*()/set_*() method on anything
# it reads, never runs a simulation, and holds no state of its own
# between calls -- call it again after a new session exists and the
# summary reflects it, precisely because there is no cache to go stale.
#
# HONEST GAP, disclosed rather than papered over: the approved
# architecture's own ValidationEvidence/ParameterValidationRecord/
# ExperimentHistory types (Phases 4/5/6 of the ORIGINAL persistent-data-
# model design -- a different numbering from this implementation's own
# Phase 0-5) were never actually built as code (confirmed by repository
# search: no such class exists anywhere in calibration_studio/ or
# elsewhere). This module does not invent stand-ins for them. Instead:
# - "Current Parameter Confidence" is computed directly from
#   CalibrationSession.candidate_snapshot/result/reproducible -- the
#   real data ValidationEvidence would have rolled up, just not
#   pre-aggregated into a separate stored type.
# - "Calibration History"/"Evidence Availability" read directly from
#   PublishedBenchmark.calibration_history (Phase 3, already real,
#   already persisted) -- exactly what ExperimentHistory's own
#   "which sessions justify this benchmark" role already reduces to
#   here, since every session already lives in
#   calibration_studio.storage's own session catalog (Phase 2).
# No new storage of any kind is introduced to compensate.
#
# RECOMMENDATION VERDICTS -- never reimplemented, only reused, and only
# when honestly possible: calibration_benchmark.recommend() requires a
# live CalibrationBenchmarkResult (real MetricComparison objects), which
# only exists for a session that has not been through a save/load round
# trip in a different process load bearing on Phase 2's own finding
# that calibration_benchmark has no MetricComparison.from_dict(). A
# reloaded session's verdict is honestly reported as unknown (None) --
# never guessed from the result_snapshot dict, which is exactly the
# "duplicate recommendation logic working off different data" this
# milestone's own brief forbids.
# =====================================================


def _utc_now_iso() -> str:

    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class BenchmarkStatusSummary:

    benchmark_id: str
    title: str
    dataset: str
    benchmark_type: str
    validation_status: str
    tags: Tuple[str, ...]

    # "Recorded" is PublishedBenchmark.calibration_history's own count
    # (already real, already persisted); "resolvable" is how many of
    # those session_ids are actually present in the sessions this
    # dashboard run was given -- a real, honest cross-check ("3
    # recorded, only 1 currently loaded"), not a fabricated number.
    calibration_history_recorded: int
    calibration_history_resolvable: int

    @property
    def has_evidence(self) -> bool:
        return self.calibration_history_recorded > 0

    def to_dict(self) -> dict:

        return {
            "benchmark_id": self.benchmark_id,
            "title": self.title,
            "dataset": self.dataset,
            "benchmark_type": self.benchmark_type,
            "validation_status": self.validation_status,
            "tags": list(self.tags),
            "calibration_history_recorded": self.calibration_history_recorded,
            "calibration_history_resolvable": self.calibration_history_resolvable,
            "has_evidence": self.has_evidence,
        }


@dataclass(frozen=True)
class ParameterConfidenceSummary:

    parameter_name: str
    subsystem: str
    n_sessions: int
    n_reproducible: int
    n_adopt: int
    n_reject: int
    n_inconclusive: int
    n_unknown: int  # verdict not computable -- see this module's own docstring
    benchmark_ids: Tuple[str, ...]
    session_ids: Tuple[str, ...]

    def to_dict(self) -> dict:

        return {
            "parameter_name": self.parameter_name,
            "subsystem": self.subsystem,
            "n_sessions": self.n_sessions,
            "n_reproducible": self.n_reproducible,
            "n_adopt": self.n_adopt,
            "n_reject": self.n_reject,
            "n_inconclusive": self.n_inconclusive,
            "n_unknown": self.n_unknown,
            "benchmark_ids": list(self.benchmark_ids),
            "session_ids": list(self.session_ids),
        }


@dataclass(frozen=True)
class ValidationDashboard:

    generated_at: str
    total_benchmarks: int

    # (validated + known_broken) / total -- "how much of the published-
    # benchmark space has actually been run at all," regardless of
    # outcome. None (not 0.0) when there are no benchmarks to compute a
    # fraction over at all.
    validation_coverage: Optional[float]

    validated_benchmarks: Tuple[BenchmarkStatusSummary, ...]
    pending_benchmarks: Tuple[BenchmarkStatusSummary, ...]
    known_broken_benchmarks: Tuple[BenchmarkStatusSummary, ...]
    benchmark_status: Tuple[BenchmarkStatusSummary, ...]

    parameter_confidence: Tuple[ParameterConfidenceSummary, ...]

    calibration_history: Mapping[str, Tuple[str, ...]]
    evidence_availability: Mapping[str, bool]

    def to_dict(self) -> dict:

        return {
            "generated_at": self.generated_at,
            "total_benchmarks": self.total_benchmarks,
            "validation_coverage": self.validation_coverage,
            "validated_benchmarks": [b.to_dict() for b in self.validated_benchmarks],
            "pending_benchmarks": [b.to_dict() for b in self.pending_benchmarks],
            "known_broken_benchmarks": [b.to_dict() for b in self.known_broken_benchmarks],
            "benchmark_status": [b.to_dict() for b in self.benchmark_status],
            "parameter_confidence": [p.to_dict() for p in self.parameter_confidence],
            "calibration_history": {key: list(value) for key, value in self.calibration_history.items()},
            "evidence_availability": dict(self.evidence_availability),
        }


def generate_validation_dashboard(
    *, benchmarks: Sequence[PublishedBenchmark], sessions: Sequence[CalibrationSession],
) -> ValidationDashboard:

    known_session_ids = {session.session_id for session in sessions}

    benchmark_summaries: List[BenchmarkStatusSummary] = []
    calibration_history: Dict[str, Tuple[str, ...]] = {}

    for benchmark in benchmarks:

        recorded = benchmark.calibration_history
        resolvable = sum(1 for session_id in recorded if session_id in known_session_ids)

        benchmark_summaries.append(BenchmarkStatusSummary(
            benchmark_id=benchmark.benchmark_id,
            title=benchmark.title,
            dataset=benchmark.dataset,
            benchmark_type=benchmark.benchmark_type.value,
            validation_status=benchmark.validation_status.value,
            tags=benchmark.tags,
            calibration_history_recorded=len(recorded),
            calibration_history_resolvable=resolvable,
        ))
        calibration_history[benchmark.benchmark_id] = recorded

    validated = tuple(
        summary for summary in benchmark_summaries
        if summary.validation_status in (ValidationStatus.RUN_WITH_DEFAULTS.value, ValidationStatus.RUN_WITH_CANDIDATES.value)
    )
    pending = tuple(
        summary for summary in benchmark_summaries if summary.validation_status == ValidationStatus.NOT_RUN.value
    )
    known_broken = tuple(
        summary for summary in benchmark_summaries if summary.validation_status == ValidationStatus.KNOWN_BROKEN.value
    )

    total = len(benchmark_summaries)
    coverage = ((len(validated) + len(known_broken)) / total) if total else None

    evidence_availability = {summary.benchmark_id: summary.has_evidence for summary in benchmark_summaries}

    return ValidationDashboard(
        generated_at=_utc_now_iso(),
        total_benchmarks=total,
        validation_coverage=coverage,
        validated_benchmarks=validated,
        pending_benchmarks=pending,
        known_broken_benchmarks=known_broken,
        benchmark_status=tuple(benchmark_summaries),
        parameter_confidence=_aggregate_parameter_confidence(sessions),
        calibration_history=calibration_history,
        evidence_availability=evidence_availability,
    )


def _aggregate_parameter_confidence(sessions: Sequence[CalibrationSession]) -> Tuple[ParameterConfidenceSummary, ...]:

    groups: Dict[str, List[CalibrationSession]] = {}

    for session in sessions:

        snapshot = session.candidate_snapshot

        if snapshot is None:
            # A defaults-only run (no candidate under test) has no
            # parameter to report confidence about -- correctly
            # excluded, not folded into a misleading "unknown parameter"
            # bucket.
            continue

        groups.setdefault(snapshot.get("name", "unknown"), []).append(session)

    summaries = []

    for parameter_name, group_sessions in groups.items():

        subsystem = group_sessions[0].candidate_snapshot.get("subsystem", "")
        benchmark_ids = tuple(sorted({s.benchmark_id for s in group_sessions if s.benchmark_id}))
        session_ids = tuple(s.session_id for s in group_sessions)

        n_reproducible = sum(1 for s in group_sessions if s.reproducible is True)

        n_adopt = n_reject = n_inconclusive = n_unknown = 0

        for session in group_sessions:

            verdict = _verdict_for(session)

            if verdict is None:
                n_unknown += 1
            elif verdict is Verdict.ADOPT:
                n_adopt += 1
            elif verdict is Verdict.REJECT:
                n_reject += 1
            else:
                n_inconclusive += 1

        summaries.append(ParameterConfidenceSummary(
            parameter_name=parameter_name,
            subsystem=subsystem,
            n_sessions=len(group_sessions),
            n_reproducible=n_reproducible,
            n_adopt=n_adopt,
            n_reject=n_reject,
            n_inconclusive=n_inconclusive,
            n_unknown=n_unknown,
            benchmark_ids=benchmark_ids,
            session_ids=session_ids,
        ))

    return tuple(sorted(summaries, key=lambda summary: summary.parameter_name))


def _verdict_for(session: CalibrationSession) -> Optional[Verdict]:

    if session.result is None:
        return None

    return recommend(session.result).overall_verdict
