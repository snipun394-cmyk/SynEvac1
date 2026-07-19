from dataclasses import dataclass, field
from typing import FrozenSet, Optional

from scenario_definition.distributions import Distribution, distribution_from_dict


@dataclass(frozen=True)
class FireDefinition:

    # Pure configuration for where/how a candidate's fire may be
    # sampled -- architecture doc §3.2's Fire row, §3.1 finding 4.
    # allowed_ignition_zone_ids/forbidden_ignition_zone_ids/
    # allowed_ignition_floor_ids are first-class, independently
    # authorable allow/forbid rules (not folded into an implicit
    # "domain"); ignition_zone_preference is an *optional* likelihood
    # weighting layered on top of, never instead of, those allow/
    # forbid rules -- its absence means no stated preference, not "no
    # fire allowed."
    #
    # growth_parameter_distribution has no default: the frozen schema
    # types it as a bare Distribution[float], not Optional, so a
    # FireDefinition must state some rule for it (a FixedValue is the
    # degenerate "always this exact growth rate" case if the author
    # wants a pinned value).
    #
    # allowed_fire_profiles holds plain, open identifier strings (e.g.
    # "Electrical", "Flaming") -- §9 leaves Fire Profile's own schema
    # unfinalized ("architecture support only, no implementation"), so
    # this module commits to nothing more than a label, exactly like
    # scenario.fire.ScenarioFire.fire_profile does on the resolved
    # side.

    growth_parameter_distribution: Distribution

    allowed_ignition_zone_ids: FrozenSet[str] = field(default_factory=frozenset)
    forbidden_ignition_zone_ids: FrozenSet[str] = field(default_factory=frozenset)
    allowed_ignition_floor_ids: FrozenSet[str] = field(default_factory=frozenset)

    ignition_zone_preference: Optional[Distribution] = None

    allowed_fire_profiles: FrozenSet[str] = field(default_factory=frozenset)

    # =====================================================

    def __post_init__(self):

        object.__setattr__(
            self, "allowed_ignition_zone_ids", frozenset(self.allowed_ignition_zone_ids),
        )
        object.__setattr__(
            self, "forbidden_ignition_zone_ids", frozenset(self.forbidden_ignition_zone_ids),
        )
        object.__setattr__(
            self, "allowed_ignition_floor_ids", frozenset(self.allowed_ignition_floor_ids),
        )
        object.__setattr__(
            self, "allowed_fire_profiles", frozenset(self.allowed_fire_profiles),
        )

    # =====================================================

    def to_dict(self) -> dict:

        return {
            "growth_parameter_distribution": self.growth_parameter_distribution.to_dict(),
            "allowed_ignition_zone_ids": sorted(self.allowed_ignition_zone_ids),
            "forbidden_ignition_zone_ids": sorted(self.forbidden_ignition_zone_ids),
            "allowed_ignition_floor_ids": sorted(self.allowed_ignition_floor_ids),
            "ignition_zone_preference": (
                self.ignition_zone_preference.to_dict()
                if self.ignition_zone_preference is not None
                else None
            ),
            "allowed_fire_profiles": sorted(self.allowed_fire_profiles),
        }

    # =====================================================

    @classmethod
    def from_dict(cls, data: dict) -> "FireDefinition":

        preference_data = data.get("ignition_zone_preference")

        return cls(
            growth_parameter_distribution=distribution_from_dict(
                data["growth_parameter_distribution"],
            ),
            allowed_ignition_zone_ids=data.get("allowed_ignition_zone_ids", []),
            forbidden_ignition_zone_ids=data.get("forbidden_ignition_zone_ids", []),
            allowed_ignition_floor_ids=data.get("allowed_ignition_floor_ids", []),
            ignition_zone_preference=(
                distribution_from_dict(preference_data) if preference_data is not None else None
            ),
            allowed_fire_profiles=data.get("allowed_fire_profiles", []),
        )
