from dataclasses import dataclass
from typing import Callable, Tuple

from calibration_benchmark import ParameterCandidate


# =====================================================
# Automatic Calibration Engine, Phase 1 -- Core Architecture.
#
# ParameterDimension is the one genuinely new abstraction this package
# introduces: a thin adapter OVER calibration_benchmark.candidates,
# never a change to it -- the same "wrap, don't touch" discipline
# StairCapacityModel/CapacityWidthCandidate already model. `build` is a
# factory: given one scalar value, it constructs a real, concrete
# ParameterCandidate (e.g. WalkingSpeedCandidate("Adult_Default", value,
# ...)) -- so a search strategy never needs to know which
# ParameterCandidate subclass, constructor signature, or profile_id it
# is actually varying; it only ever sees "propose a value in these
# bounds."
#
# `build` is a live callable and therefore cannot survive persistence --
# the same "live object, snapshot not reconstruction" constraint
# calibration_studio.session.CalibrationSession.candidate already has.
# describe() is the JSON-safe projection AutoCalibrationRun actually
# persists.
# =====================================================


@dataclass(frozen=True)
class ParameterDimension:

    name: str
    bounds: Tuple[float, float]
    build: Callable[[float], ParameterCandidate]

    def build_candidate(self, value: float) -> ParameterCandidate:

        return self.build(value)

    def describe(self) -> dict:

        return {"name": self.name, "bounds": [self.bounds[0], self.bounds[1]]}


@dataclass(frozen=True)
class SearchSpace:

    # An ordered tuple of ParameterDimension -- single-dimension is the
    # common case (most existing ParameterCandidate subclasses vary one
    # scalar, e.g. WalkingSpeedCandidate.candidate_speed). Multi-
    # dimension exists for a future joint/cross-subsystem calibration
    # strategy (see the Automatic Calibration Engine architecture
    # investigation's own Evolutionary Algorithms discussion) -- nothing
    # in this milestone consumes more than one dimension.

    dimensions: Tuple[ParameterDimension, ...]

    def __post_init__(self):

        if not self.dimensions:
            raise ValueError("SearchSpace must have at least one ParameterDimension.")

        names = [d.name for d in self.dimensions]

        if len(names) != len(set(names)):
            raise ValueError(f"SearchSpace dimension names must be unique, got {names!r}.")

    def __iter__(self):

        return iter(self.dimensions)

    def __len__(self) -> int:

        return len(self.dimensions)

    def describe(self) -> Tuple[dict, ...]:

        return tuple(d.describe() for d in self.dimensions)
