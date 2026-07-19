from dataclasses import dataclass, field, replace
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

from models.building import Building

from scenario_definition.definition import ScenarioDefinition
from scenario_definition.distributions import Distribution, FixedValue, WeightedOptions


# =====================================================
# Phase 1 -- Experiment Definition. An "experiment" compares two or
# more *arms* -- each arm a small, named set of overrides applied to
# one shared base ScenarioDefinition -- across the same base Building.
# Every override is a pure function ScenarioDefinition -> ScenarioDefinition
# built with dataclasses.replace(), never a mutation -- the same
# immutable-value-object discipline ScenarioDefinition itself already
# requires ("declares what may be sampled, samples nothing itself").
# This module never samples, generates, simulates, or validates
# anything -- it only describes what a research_framework.runner call
# should compare.
# =====================================================


@dataclass(frozen=True)
class ParameterOverride:

    # A named, pure ScenarioDefinition -> ScenarioDefinition transform.
    # `name` is what appears in reports/figures (Phase 6/7) -- kept
    # separate from the callable itself so two overrides that happen to
    # share an implementation (e.g. two different "set team count to 0"
    # calls from different call sites) can still be labeled distinctly.

    name: str
    apply: Callable[[ScenarioDefinition], ScenarioDefinition]

    def __call__(self, definition: ScenarioDefinition) -> ScenarioDefinition:

        return self.apply(definition)


@dataclass(frozen=True)
class Arm:

    # One experimental condition -- a name plus the overrides that
    # distinguish it from the experiment's base ScenarioDefinition.
    # `building` is None by default (inherit ExperimentSpec.base_building)
    # -- set it to compare *building layouts* as an arm-level variable
    # (research_framework.runner never mutates a Building; a different
    # layout means a genuinely different Building object, supplied
    # here). `registration` selects which behaviour_profile_resolver
    # registration function research_framework.runner uses for this arm
    # -- "population" (civilians + scenario-authored firefighter
    # rescue/assistance, the correct default for any arm involving
    # firefighters) is used unless declared as `"civilian"` (civilians
    # only, behaviour_profile_resolver.registrar.register_occupants) or
    # `"dynamic"` (behaviour_profile_resolver.dynamic_registrar.
    # register_population_dynamic).

    name: str
    overrides: Tuple[ParameterOverride, ...] = ()
    building: Optional[Building] = None
    registration: str = "population"

    def resolve_definition(self, base_definition: ScenarioDefinition) -> ScenarioDefinition:

        return apply_overrides(base_definition, self.overrides)


def make_arm(name: str, *overrides: ParameterOverride, building: Optional[Building] = None,
             registration: str = "population") -> Arm:

    return Arm(name=name, overrides=tuple(overrides), building=building, registration=registration)


def apply_overrides(
    base_definition: ScenarioDefinition, overrides: Sequence[ParameterOverride],
) -> ScenarioDefinition:

    definition = base_definition

    for override in overrides:
        definition = override(definition)

    return definition


# =====================================================


@dataclass(frozen=True)
class ExperimentSpec:

    name: str
    base_definition: ScenarioDefinition
    base_definition_id: str
    base_building: Building
    arms: Tuple[Arm, ...]

    samples_per_arm: int
    master_seed: int
    output_root: str

    dt: float = 5.0

    def __post_init__(self):

        if not self.arms:
            raise ValueError("ExperimentSpec.arms must contain at least one Arm.")

        names = [arm.name for arm in self.arms]

        if len(names) != len(set(names)):
            raise ValueError(f"ExperimentSpec.arms must have unique names, got {names!r}.")

        object.__setattr__(self, "arms", tuple(self.arms))

    def building_for(self, arm: Arm) -> Building:

        return arm.building if arm.building is not None else self.base_building


# =====================================================
# Composable override builders -- one per example knob the phase spec
# names verbatim (occupant populations, firefighter team sizes,
# wheelchair/elderly percentages, helping probabilities, fire growth
# models, detector failures; building layouts are handled via Arm.
# building directly, not an override, since layout is a Building
# choice, not a ScenarioDefinition field -- see scenario_definition's
# own architecture: there is no layout field to override). Every
# builder below returns a ParameterOverride wrapping dataclasses.
# replace(); none of them samples, validates, or interprets a
# behaviour_profile_id -- that stays exclusively scenario_generator's
# and behaviour_profile_resolver's jobs, respectively.
# =====================================================


def override_occupant_population(
    occupancy_distribution: Optional[Mapping[str, Distribution]] = None,
    behaviour_profile_distribution: Optional[Mapping[str, Distribution]] = None,
    name: str = "occupant_population",
) -> ParameterOverride:

    def _apply(definition: ScenarioDefinition) -> ScenarioDefinition:

        occupant = definition.occupant

        if occupancy_distribution is not None:
            occupant = replace(occupant, occupancy_distribution=dict(occupancy_distribution))

        if behaviour_profile_distribution is not None:
            occupant = replace(
                occupant, behaviour_profile_distribution=dict(behaviour_profile_distribution),
            )

        return replace(definition, occupant=occupant)

    return ParameterOverride(name=name, apply=_apply)


def override_profile_mix(
    zone_ids: Sequence[str], percentages: Mapping[str, float], name: str = "profile_mix",
) -> ParameterOverride:

    # percentages: behaviour_profile_id -> relative weight (need not sum
    # to 1.0 -- WeightedOptions treats weights as relative, not
    # normalized probabilities). The same mix is applied to every zone
    # in zone_ids -- a per-zone-varying mix is available via
    # override_occupant_population() directly if needed.

    def _apply(definition: ScenarioDefinition) -> ScenarioDefinition:

        distribution = WeightedOptions(dict(percentages))
        updated = dict(definition.occupant.behaviour_profile_distribution)

        for zone_id in zone_ids:
            updated[zone_id] = distribution

        return replace(definition, occupant=replace(definition.occupant, behaviour_profile_distribution=updated))

    return ParameterOverride(name=name, apply=_apply)


def override_helping_probability(probability: float, name: str = "helping_probability") -> ParameterOverride:

    def _apply(definition: ScenarioDefinition) -> ScenarioDefinition:

        return replace(
            definition,
            occupant=replace(definition.occupant, assistance_pairing_probability=probability),
        )

    return ParameterOverride(name=name, apply=_apply)


def override_firefighter_team_size(
    team_count: Optional[int] = None, team_size: Optional[int] = None,
    entry_zone_ids: Optional[Sequence[str]] = None, name: str = "firefighter_team_size",
) -> ParameterOverride:

    def _apply(definition: ScenarioDefinition) -> ScenarioDefinition:

        firefighter = definition.firefighter

        if team_count is not None:
            firefighter = replace(firefighter, team_count_distribution=FixedValue(team_count))

        if team_size is not None:
            firefighter = replace(firefighter, team_size_distribution=FixedValue(team_size))

        if entry_zone_ids is not None:
            firefighter = replace(firefighter, entry_zone_ids=tuple(entry_zone_ids))

        return replace(definition, firefighter=firefighter)

    return ParameterOverride(name=name, apply=_apply)


def override_fire_growth_time(growth_time: float, name: str = "fire_growth_time") -> ParameterOverride:

    def _apply(definition: ScenarioDefinition) -> ScenarioDefinition:

        return replace(
            definition, fire=replace(definition.fire, growth_parameter_distribution=FixedValue(growth_time)),
        )

    return ParameterOverride(name=name, apply=_apply)


def override_detector_failure_rate(
    detector_ids: Sequence[str], failure_rate: float, name: str = "detector_failure_rate",
) -> ParameterOverride:

    def _apply(definition: ScenarioDefinition) -> ScenarioDefinition:

        distribution = WeightedOptions({"AVAILABLE": 1.0 - failure_rate, "FAILED": failure_rate})
        updated = dict(definition.engineering.detector_state_distribution)

        for detector_id in detector_ids:
            updated[detector_id] = distribution

        return replace(
            definition, engineering=replace(definition.engineering, detector_state_distribution=updated),
        )

    return ParameterOverride(name=name, apply=_apply)
