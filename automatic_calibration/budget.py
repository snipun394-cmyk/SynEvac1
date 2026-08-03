from dataclasses import dataclass


# =====================================================
# Automatic Calibration Engine, Phase 1 -- Core Architecture.
#
# Deliberately minimal -- "implement only the shared infrastructure"
# this milestone's own brief requires. max_evaluations is the one
# stopping condition every future concrete strategy (Grid/Random/
# Bayesian/Evolutionary/Pareto) needs regardless of its own internal
# convergence logic; nothing else is added speculatively (no wall-clock
# limit, no convergence-delta field) since no strategy exists yet to
# consume it -- the same restraint "Do NOT invent parameters the
# simulator does not already expose" already models elsewhere in this
# codebase, applied here to this package's own new surface instead.
#
# A plain, fully round-trippable value object (unlike ParameterCandidate/
# CalibrationBenchmarkResult, it holds no live, unreconstructable
# state) -- to_dict()/from_dict() are a genuine inverse pair.
# =====================================================


@dataclass(frozen=True)
class AutoCalibrationBudget:

    max_evaluations: int

    def __post_init__(self):

        if self.max_evaluations < 1:
            raise ValueError(f"max_evaluations must be >= 1, got {self.max_evaluations!r}.")

    def to_dict(self) -> dict:

        return {"max_evaluations": self.max_evaluations}

    @classmethod
    def from_dict(cls, data: dict) -> "AutoCalibrationBudget":

        return cls(max_evaluations=data["max_evaluations"])
