from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class GenerationStatistics:

    # Operational metrics only -- this implementation phase's own
    # brief ("do not perform analytics beyond these operational
    # metrics"). One shape covers both a single run_pipeline() call and
    # an aggregated run_batch_pipeline() call: for a single scenario,
    # `attempts` is how many generate+validate cycles that one scenario
    # took; for a batch, it is the sum across every scenario in the
    # batch (see aggregate() below) -- the same fields mean the same
    # thing at either granularity, so callers never need two different
    # types to reason about "how much work did this take."

    attempts: int
    accepted_count: int
    rejection_count: int
    acceptance_rate: float

    rejection_categories: Mapping[str, int] = field(default_factory=dict)

    generation_time_seconds: float = 0.0
    validation_time_seconds: float = 0.0
    total_time_seconds: float = 0.0

    # =====================================================

    def __post_init__(self):

        object.__setattr__(
            self, "rejection_categories", MappingProxyType(dict(self.rejection_categories)),
        )


def build_statistics(
    attempts,
    accepted_count,
    rejection_categories,
    generation_time_seconds,
    validation_time_seconds,
    total_time_seconds,
) -> GenerationStatistics:

    # The one place acceptance_rate/rejection_count are derived from
    # the raw counts -- shared by pipeline.py's single-scenario and
    # batch code paths so the derivation can't drift between them.

    rejection_count = attempts - accepted_count
    acceptance_rate = accepted_count / attempts if attempts > 0 else 0.0

    return GenerationStatistics(
        attempts=attempts,
        accepted_count=accepted_count,
        rejection_count=rejection_count,
        acceptance_rate=acceptance_rate,
        rejection_categories=rejection_categories,
        generation_time_seconds=generation_time_seconds,
        validation_time_seconds=validation_time_seconds,
        total_time_seconds=total_time_seconds,
    )


def aggregate_statistics(per_scenario_statistics) -> GenerationStatistics:

    # Combines a batch's per-scenario GenerationStatistics into one --
    # used by run_batch_pipeline(). Rejection categories are summed
    # across every scenario in the batch, giving the same
    # exhaustion-vs-miscalibration diagnostic signal
    # docs/architecture/scenario_engine.md §4.9/§5.8 describes, just
    # computed here by orchestration (exactly where §5.8 says that
    # aggregation belongs) rather than inside the stateless Validator.

    attempts = sum(s.attempts for s in per_scenario_statistics)
    accepted_count = sum(s.accepted_count for s in per_scenario_statistics)

    rejection_categories = {}

    for stats in per_scenario_statistics:
        for category, count in stats.rejection_categories.items():
            rejection_categories[category] = rejection_categories.get(category, 0) + count

    return build_statistics(
        attempts=attempts,
        accepted_count=accepted_count,
        rejection_categories=rejection_categories,
        generation_time_seconds=sum(s.generation_time_seconds for s in per_scenario_statistics),
        validation_time_seconds=sum(s.validation_time_seconds for s in per_scenario_statistics),
        total_time_seconds=sum(s.total_time_seconds for s in per_scenario_statistics),
    )
