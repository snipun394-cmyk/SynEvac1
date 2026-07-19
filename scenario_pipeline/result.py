from dataclasses import dataclass, field
from typing import Optional, Tuple

from scenario import Scenario
from scenario_validator import ScenarioValidationReport

from scenario_pipeline.statistics import GenerationStatistics


@dataclass(frozen=True)
class PipelineResult:

    # The outcome of one run_pipeline() call -- immutable, matching
    # every model in scenario/ and scenario_definition/'s own
    # convention. `scenario` is the accepted Scenario, or None if every
    # attempt up to max_attempts was rejected; `failure_reason` is
    # populated only in that second case ("explaining why generation
    # failed", this phase's own Termination brief) -- never both at
    # once.

    accepted: bool
    scenario: Optional[Scenario]
    statistics: GenerationStatistics
    last_report: Optional[ScenarioValidationReport] = None
    failure_reason: Optional[str] = None

    # =====================================================

    def __post_init__(self):

        if self.accepted and self.scenario is None:
            raise ValueError("PipelineResult.accepted is True but scenario is None.")

        if not self.accepted and self.scenario is not None:
            raise ValueError("PipelineResult.accepted is False but scenario is not None.")


@dataclass(frozen=True)
class BatchPipelineResult:

    # `scenarios` holds only accepted Scenarios, in index order --
    # "rejected candidates must not appear in the final output" (this
    # phase's own Batch Pipeline brief), taken literally: a rejected
    # PipelineResult's own `scenario` field is already None, so
    # nothing rejected is ever reachable through `scenarios`.
    # `results` keeps one PipelineResult per index attempted (accepted
    # or not) purely for diagnostics -- the statistics/rejection
    # information this phase's brief explicitly asks for, distinct
    # from the deliverable Scenario objects themselves.

    scenarios: Tuple[Scenario, ...] = field(default_factory=tuple)
    results: Tuple[PipelineResult, ...] = field(default_factory=tuple)
    statistics: GenerationStatistics = None

    # =====================================================

    def __post_init__(self):

        object.__setattr__(self, "scenarios", tuple(self.scenarios))
        object.__setattr__(self, "results", tuple(self.results))
