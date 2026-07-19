from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping


class Distribution:

    # The common, pure-data marker every distribution kind subclasses --
    # architecture doc docs/architecture/scenario_engine.md §3.1 finding
    # 1 / §3.2. Describes an allowed set of values for some field and,
    # optionally, the relative likelihood within that set -- nothing
    # more. No sample() method exists anywhere in this module, and this
    # module never imports random -- the interpreter that turns a
    # Distribution descriptor plus an rng into a concrete value is
    # exclusively the (not-yet-implemented) Scenario Generator's
    # concern, out of scope here.
    #
    # Deliberately not a dataclass itself (it has no fields of its
    # own) -- each concrete kind below is an independent frozen
    # dataclass subclassing this purely as a common type for
    # isinstance checks and Dict[id, Distribution] annotations, the
    # same "plain marker base, concrete dataclass subclasses" shape
    # hazard_evolution.source.HazardSource uses for a behavioral
    # interface (this one for a data-only one instead).

    pass


@dataclass(frozen=True)
class FixedValue(Distribution):

    # Exactly one allowed value -- the degenerate case every other
    # distribution-valued field collapses into for a pinned/always-X
    # rule (§3.1 finding: replaces what used to be separate
    # fixed_occupancy/always_open_door_ids-style sibling fields).
    # `value` is intentionally untyped (Any): this module never
    # interprets what it means (a zone id, a bool, an enum member's
    # name, a float, ...), only carries it.

    value: Any

    # =====================================================

    def to_dict(self) -> dict:

        return {"kind": "FixedValue", "value": self.value}

    # =====================================================

    @classmethod
    def from_dict(cls, data: dict) -> "FixedValue":

        return cls(value=data["value"])


@dataclass(frozen=True)
class UniformRange(Distribution):

    # Every value in [low, high] allowed, no stated preference among
    # them. discrete=True describes an integer-valued range (e.g.
    # occupant counts); discrete=False a continuous one (e.g. fire
    # growth parameters). This module never checks discrete against
    # low/high's own types -- only that low <= high (§3.4).

    low: float
    high: float
    discrete: bool = False

    # =====================================================

    def to_dict(self) -> dict:

        return {
            "kind": "UniformRange",
            "low": self.low,
            "high": self.high,
            "discrete": self.discrete,
        }

    # =====================================================

    @classmethod
    def from_dict(cls, data: dict) -> "UniformRange":

        return cls(
            low=data["low"],
            high=data["high"],
            discrete=data.get("discrete", False),
        )


@dataclass(frozen=True)
class WeightedOptions(Distribution):

    # A finite allowed set, each member with a relative weight (a
    # boolean "open probability" is the two-key case,
    # WeightedOptions({True: p, False: 1 - p})). Weight keys are
    # restricted to plain, already-primitive values (str/bool/int/
    # float) -- never an enum member -- so this module never needs to
    # know what a key *means* (a DoorState, a zone id, a bool) to
    # serialize it. An author describing a door's WeightedOptions
    # writes e.g. {"OPEN": 0.7, "CLOSED": 0.2, "LOCKED": 0.1} using the
    # engineering_constraints.DoorState enum's own .name strings for
    # discoverability -- interpreting those strings back into a
    # DoorState member is exclusively the Generator's job (this
    # prompt's own instruction: "Do NOT interpret the distributions").

    weights: Mapping[Any, float] = field(default_factory=dict)

    # =====================================================

    def __post_init__(self):

        object.__setattr__(self, "weights", MappingProxyType(dict(self.weights)))

    # =====================================================

    def to_dict(self) -> dict:

        return {"kind": "WeightedOptions", "weights": dict(self.weights)}

    # =====================================================

    @classmethod
    def from_dict(cls, data: dict) -> "WeightedOptions":

        return cls(weights=data.get("weights", {}))


# Named and reserved, not implemented (§3.2): GaussianParameters(mean,
# stddev), TimeDependentSpec(...), EmpiricalDatasetRef(...). Each is a
# future *description of an allowed-value rule*, same as the three
# kinds above -- none of them, now or later, gains a method that draws
# a value. Left undefined here rather than stubbed out: TimeDependentSpec
# and EmpiricalDatasetRef don't yet have a committed shape beyond a
# name (the architecture doc itself only writes "(...)" for them), and
# this implementation phase's own brief lists only FixedValue/
# UniformRange/WeightedOptions as in scope.


_DISTRIBUTION_KINDS = {
    "FixedValue": FixedValue,
    "UniformRange": UniformRange,
    "WeightedOptions": WeightedOptions,
}


def distribution_from_dict(data: dict) -> Distribution:

    # The one place that dispatches a serialized Distribution's "kind"
    # discriminator back to its concrete class -- shared by every
    # module in this package that holds a bare Distribution field
    # (FireDefinition.ignition_zone_preference, EventTemplate.occurs/
    # time, ...) or a Dict[id, Distribution] field (see
    # distribution_mapping_to_dict/_from_dict below). This is
    # structural reconstruction only (which dataclass shape does this
    # dict match), never interpretation of what the distribution's
    # values mean.

    kind = data["kind"]

    try:
        distribution_cls = _DISTRIBUTION_KINDS[kind]
    except KeyError:
        raise ValueError(f"Unknown Distribution kind: {kind!r}") from None

    return distribution_cls.from_dict(data)


def distribution_mapping_to_dict(mapping: Mapping[str, Distribution]) -> dict:

    # Shared helper for every Dict[id, Distribution] field
    # (door_state_distribution, exit_state_distribution,
    # occupancy_distribution, ...) across fire_definition.py,
    # occupant_definition.py, and engineering_constraints.py.

    return {key: distribution.to_dict() for key, distribution in mapping.items()}


def distribution_mapping_from_dict(data: Mapping[str, dict]) -> dict:

    return {key: distribution_from_dict(value) for key, value in data.items()}
