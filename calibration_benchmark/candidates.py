from dataclasses import dataclass, replace
from typing import Any, Dict, Optional

from behavior_library.pre_movement_strategies import ProbabilisticPreMovementDelay
from behaviour_profile_resolver.registry import DEFAULT_PROFILE_REGISTRY
from crowd_intelligence.models import DensityThresholds
from simulator.capacity import DefaultCapacityModel, StairCapacityModel
from simulator.congestion import DefaultCongestionModel, StairAwareCongestionModel
from simulator.flow_region_capacity import FlowRegionCapacityModel
from simulator.flow_region_congestion import FlowRegionCongestionModel


# =====================================================
# Calibration Benchmark Framework V1 -- one ParameterCandidate per
# dataset-derived value identified in the (already-completed, external)
# SynEvac Calibration Mapping Report. This module does not decide WHICH
# SynEvac parameter a dataset field maps to -- that mapping is already
# closed, external, and taken as given. This module only ever answers
# "given that mapping, how is the candidate value expressed as a
# concrete, injectable SynEvac object" -- never by mutating a shared
# default (DEFAULT_PROFILE_REGISTRY, DefaultCapacityModel,
# DefaultCongestionModel, DensityThresholds are all read here, never
# written), always by constructing a brand-new object every time
# candidate_*()/baseline_*() is called. baseline_*() and candidate_*()
# are deliberately symmetric -- a caller that only ever calls
# baseline_*() must see byte-for-byte the same production behavior as
# calling scenario_runner.run() directly, never a silently different
# one.
#
# ParameterCandidate is a plain (non-dataclass) base so every subclass
# can freely accept its own constructor arguments and forward the
# common, described fields to it via a single ordinary super().__init__()
# call -- mixing @dataclass's generated __init__ with per-subclass extra
# constructor logic would fight the dataclass machinery for no benefit
# here, since no subclass needs auto-generated __eq__/__repr__ beyond
# what describe()/to_dict() already provide on every candidate uniformly.
# =====================================================


class ParameterCandidate:

    def __init__(
        self, *, name: str, subsystem: str, calibration_tier: str, dataset_source: str,
        current_value: Any, candidate_value: Any, unit: str, rationale: str,
    ):

        self.name = name
        self.subsystem = subsystem
        self.calibration_tier = calibration_tier
        self.dataset_source = dataset_source
        self.current_value = current_value
        self.candidate_value = candidate_value
        self.unit = unit
        self.rationale = rationale

    # =====================================================

    def baseline_registry(self):
        return DEFAULT_PROFILE_REGISTRY

    def candidate_registry(self):
        return DEFAULT_PROFILE_REGISTRY

    def baseline_capacity_model(self):
        return StairCapacityModel()

    def candidate_capacity_model(self):
        return StairCapacityModel()

    def baseline_congestion_model(self):
        return StairAwareCongestionModel()

    def candidate_congestion_model(self):
        return StairAwareCongestionModel()

    def baseline_density_thresholds(self) -> DensityThresholds:
        return DensityThresholds()

    def candidate_density_thresholds(self) -> DensityThresholds:
        return DensityThresholds()

    # Hybrid Flow Regions (Option D), Milestone 4 -- both default False
    # (today's exact per-edge admission control), so none of the five
    # existing candidate classes above need any change at all. Only
    # FlowRegionCapacityCandidate (below) overrides these, always
    # together with matching capacity_model()/congestion_model()
    # overrides -- flow_region_map is meaningless to
    # MultiAgentSimulation without a Flow-Region-aware pair of models
    # to hand it to (see simulator/coordinator.py's own __init__
    # comment), so this framework keeps that pairing entirely within
    # one candidate's own four hooks rather than trying to enforce it
    # generically.
    def baseline_use_flow_regions(self) -> bool:
        return False

    def candidate_use_flow_regions(self) -> bool:
        return False

    # =====================================================

    def describe(self) -> Dict[str, Any]:

        return {
            "name": self.name,
            "subsystem": self.subsystem,
            "calibration_tier": self.calibration_tier,
            "dataset_source": self.dataset_source,
            "current_value": self.current_value,
            "candidate_value": self.candidate_value,
            "unit": self.unit,
            "rationale": self.rationale,
        }


# =====================================================


def _registry_with_walking_speed(profile_id: str, walking_speed: float) -> Dict[str, Any]:

    # Copies DEFAULT_PROFILE_REGISTRY one level deep (Mapping -> plain
    # dict) and replaces exactly one profile's own template via
    # dataclasses.replace() -- BehaviorProfileTemplate is frozen, so
    # this can never mutate the shared template instance every other
    # occupant/candidate/baseline run still reads. register_occupants()
    # (behaviour_profile_resolver/registrar.py) accepts any
    # Mapping[str, BehaviorProfileTemplate] in place of the default
    # registry -- this is that exact, already-supported seam.

    if profile_id not in DEFAULT_PROFILE_REGISTRY:

        raise KeyError(
            f"{profile_id!r} is not a profile in DEFAULT_PROFILE_REGISTRY -- "
            f"choose one of {sorted(DEFAULT_PROFILE_REGISTRY)}.",
        )

    registry = dict(DEFAULT_PROFILE_REGISTRY)
    registry[profile_id] = replace(registry[profile_id], walking_speed=walking_speed)

    return registry


class WalkingSpeedCandidate(ParameterCandidate):

    def __init__(self, profile_id: str, candidate_speed: float, dataset_source: str, rationale: str):

        self.profile_id = profile_id

        super().__init__(
            name=f"{profile_id}.walking_speed",
            subsystem="Walking Model",
            calibration_tier="Tier 2",
            dataset_source=dataset_source,
            current_value=DEFAULT_PROFILE_REGISTRY[profile_id].walking_speed,
            candidate_value=candidate_speed,
            unit="m/s",
            rationale=rationale,
        )

    def baseline_registry(self):
        return DEFAULT_PROFILE_REGISTRY

    def candidate_registry(self):
        return _registry_with_walking_speed(self.profile_id, self.candidate_value)


# =====================================================


class PreMovementDelayCandidate(ParameterCandidate):

    def __init__(
        self, profile_id: str, candidate_median_delay: float, dataset_source: str, rationale: str,
        candidate_spread: Optional[float] = None,
    ):

        current_strategy = DEFAULT_PROFILE_REGISTRY[profile_id].pre_movement_strategy

        if not isinstance(current_strategy, ProbabilisticPreMovementDelay):

            raise TypeError(
                f"{profile_id!r}'s pre_movement_strategy is "
                f"{type(current_strategy).__name__}, not ProbabilisticPreMovementDelay -- "
                f"no median_delay/spread exists on this profile to calibrate.",
            )

        self.profile_id = profile_id
        self.candidate_spread = candidate_spread if candidate_spread is not None else current_strategy.spread

        super().__init__(
            name=f"{profile_id}.pre_movement_strategy.median_delay",
            subsystem="Pre-movement Model",
            calibration_tier="Tier 2",
            dataset_source=dataset_source,
            current_value=current_strategy.median_delay,
            candidate_value=candidate_median_delay,
            unit="seconds",
            rationale=rationale,
        )

    def baseline_registry(self):
        return DEFAULT_PROFILE_REGISTRY

    def candidate_registry(self):

        registry = dict(DEFAULT_PROFILE_REGISTRY)
        registry[self.profile_id] = replace(
            registry[self.profile_id],
            pre_movement_strategy=ProbabilisticPreMovementDelay(
                median_delay=self.candidate_value, spread=self.candidate_spread,
            ),
        )
        return registry


# =====================================================


def _capacity_model_with_width_constant(base_cls, people_per_meter_of_width: float):

    # Runtime subclassing, never editing the production class -- the
    # standard, idiomatic way to override one class-attribute constant
    # on a strategy object without touching (or even importing for
    # mutation) DefaultCapacityModel/StairCapacityModel's own module.
    # base_cls stays exactly one of the two already-existing production
    # classes; only PEOPLE_PER_METER_OF_WIDTH is ever replaced.

    candidate_cls = type(
        f"Candidate{base_cls.__name__}", (base_cls,),
        {"PEOPLE_PER_METER_OF_WIDTH": people_per_meter_of_width},
    )
    return candidate_cls()


class CapacityWidthCandidate(ParameterCandidate):

    def __init__(
        self, candidate_people_per_meter_of_width: float, dataset_source: str, rationale: str,
        stair_specific: bool = False,
    ):

        base_cls = StairCapacityModel if stair_specific else DefaultCapacityModel
        self.stair_specific = stair_specific

        super().__init__(
            name=(
                "StairCapacityModel.PEOPLE_PER_METER_OF_WIDTH" if stair_specific
                else "DefaultCapacityModel.PEOPLE_PER_METER_OF_WIDTH"
            ),
            subsystem="Stair Model" if stair_specific else "Capacity Model",
            calibration_tier="Tier 1",
            dataset_source=dataset_source,
            current_value=base_cls.PEOPLE_PER_METER_OF_WIDTH,
            candidate_value=candidate_people_per_meter_of_width,
            unit="people per meter of width",
            rationale=rationale,
        )

    def baseline_capacity_model(self):
        return StairCapacityModel()

    def candidate_capacity_model(self):

        if self.stair_specific:
            return _capacity_model_with_width_constant(StairCapacityModel, self.candidate_value)

        # A non-stair-specific width candidate must still be wrapped by
        # StairCapacityModel so Stair edges keep their own (unmodified)
        # narrower default -- only Door/Exit capacity changes, exactly
        # mirroring StairCapacityModel's own "wrap the base model,
        # special-case only your own concern" production shape.
        candidate_default = _capacity_model_with_width_constant(DefaultCapacityModel, self.candidate_value)
        return StairCapacityModel(base_model=candidate_default)


# =====================================================


class CongestionMinimumSpeedFactorCandidate(ParameterCandidate):

    def __init__(self, candidate_minimum_speed_factor: float, dataset_source: str, rationale: str):

        super().__init__(
            name="DefaultCongestionModel.MINIMUM_SPEED_FACTOR",
            subsystem="Congestion Model",
            calibration_tier="Tier 2",
            dataset_source=dataset_source,
            current_value=DefaultCongestionModel.MINIMUM_SPEED_FACTOR,
            candidate_value=candidate_minimum_speed_factor,
            unit="dimensionless speed factor",
            rationale=rationale,
        )

    def baseline_congestion_model(self):
        return StairAwareCongestionModel()

    def candidate_congestion_model(self):

        candidate_default_cls = type(
            "CandidateDefaultCongestionModel", (DefaultCongestionModel,),
            {"MINIMUM_SPEED_FACTOR": self.candidate_value},
        )
        return StairAwareCongestionModel(base_model=candidate_default_cls())


# =====================================================


class DensityThresholdCandidate(ParameterCandidate):

    def __init__(self, candidate_thresholds: DensityThresholds, dataset_source: str, rationale: str):

        current = DensityThresholds()
        self._candidate_thresholds = candidate_thresholds

        super().__init__(
            name="DensityThresholds",
            subsystem="Crowd Intelligence",
            calibration_tier="Tier 1",
            dataset_source=dataset_source,
            current_value={
                "moderate_at": current.moderate_at, "high_at": current.high_at,
                "very_high_at": current.very_high_at, "critical_at": current.critical_at,
            },
            candidate_value={
                "moderate_at": candidate_thresholds.moderate_at, "high_at": candidate_thresholds.high_at,
                "very_high_at": candidate_thresholds.very_high_at, "critical_at": candidate_thresholds.critical_at,
            },
            unit="occupancy ratio (dimensionless proxy for people/m^2 -- see metrics.py)",
            rationale=rationale,
        )

    def baseline_density_thresholds(self) -> DensityThresholds:
        return DensityThresholds()

    def candidate_density_thresholds(self) -> DensityThresholds:
        return self._candidate_thresholds


# =====================================================


class FlowRegionCapacityCandidate(ParameterCandidate):

    # Hybrid Flow Regions (Option D), Milestone 4 -- unlike every
    # candidate above, this one is not calibrating a single dataset-
    # derived scalar against a Calibration Mapping Report tier; it
    # compares an ARCHITECTURE (today's independent per-edge admission
    # control) against a candidate one (FlowRegion-aware shared
    # admission control, Milestones 1-3). current_value/candidate_value
    # are therefore short descriptions, not numbers -- render_markdown_report()
    # already formats either the same way (plain str()), so no reporting
    # change was needed for this.
    #
    # The baseline arm is left entirely at the inherited base-class
    # defaults (StairCapacityModel/StairAwareCongestionModel/
    # baseline_use_flow_regions()==False) -- byte-for-byte today's
    # production behavior, exactly like every other candidate's own
    # baseline arm. The candidate arm overrides capacity_model(),
    # congestion_model(), AND use_flow_regions() together, since
    # flow_region_map only does anything when paired with Flow-Region-
    # aware models (see ParameterCandidate's own comment above).

    def __init__(self, dataset_source: str, rationale: str):

        super().__init__(
            name="MultiAgentSimulation.flow_region_map (Hybrid Flow Regions, Option D)",
            subsystem="Admission Control Architecture",
            calibration_tier="Tier 1",
            dataset_source=dataset_source,
            current_value="independent per-edge admission control (today's production default)",
            candidate_value="FlowRegion-aware shared admission control (FlowRegionCapacityModel + FlowRegionCongestionModel)",
            unit="architecture variant (not a scalar parameter)",
            rationale=rationale,
        )

    def candidate_capacity_model(self):
        return FlowRegionCapacityModel()

    def candidate_congestion_model(self):
        return FlowRegionCongestionModel()

    def candidate_use_flow_regions(self) -> bool:
        return True
