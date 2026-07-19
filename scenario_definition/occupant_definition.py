from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from scenario_definition.distributions import (
    Distribution,
    distribution_mapping_from_dict,
    distribution_mapping_to_dict,
)


@dataclass(frozen=True)
class OccupantDefinition:

    # Pure configuration for how many occupants, and with which
    # behaviour profile, may be sampled per zone -- architecture doc
    # §3.2's Occupants row.
    #
    # Deliberately ONE field for count, not four: "min occupants",
    # "max occupants", "fixed occupants", and "forbidden occupancy
    # zones" (this implementation phase's own worked examples) are all
    # just different Distribution values inside the same
    # occupancy_distribution field, exactly as §3.1's compliance
    # review already settled -- min/max is UniformRange(min, max,
    # discrete=True), fixed is FixedValue(n), forbidden/always-empty
    # is FixedValue(0). Adding separate fields for any of these would
    # reintroduce the "fields that were previously pairs in the schema
    # collapse into one distribution-valued field each" violation §3.1
    # explicitly corrected -- not a new gap this phase is free to
    # reopen.
    #
    # behaviour_profile_distribution values are opaque identifier
    # strings (e.g. "Adult_Default") wrapped in a Distribution --
    # §8: this package never interprets what a profile id means, only
    # carries the rule for which ids may be sampled and how likely
    # each is.

    occupancy_distribution: Mapping[str, Distribution] = field(default_factory=dict)
    behaviour_profile_distribution: Mapping[str, Distribution] = field(default_factory=dict)

    # Human Population & Assisted Evacuation Modeling Phase 5 --
    # additive. The probability that a generated occupant whose
    # OccupantCategory is considered potentially in need of assistance
    # (CHILD/ELDERLY/WHEELCHAIR_USER -- see
    # behaviour_profile_resolver.category.OccupantCategory) is paired,
    # within their own zone, with another generated occupant who then
    # assists them (ScenarioOccupant.assisting_occupant_id/
    # assistance_type -- Phase 3). Defaults to 0.0 (no pairing ever
    # generated), so every existing Definition/generated Scenario is
    # unaffected.
    assistance_pairing_probability: float = 0.0

    # =====================================================

    def __post_init__(self):

        if not (0.0 <= self.assistance_pairing_probability <= 1.0):

            raise ValueError(
                f"OccupantDefinition.assistance_pairing_probability must be in [0.0, 1.0], "
                f"got {self.assistance_pairing_probability!r}."
            )

        object.__setattr__(
            self, "occupancy_distribution", MappingProxyType(dict(self.occupancy_distribution)),
        )
        object.__setattr__(
            self,
            "behaviour_profile_distribution",
            MappingProxyType(dict(self.behaviour_profile_distribution)),
        )

    # =====================================================

    def to_dict(self) -> dict:

        return {
            "occupancy_distribution": distribution_mapping_to_dict(self.occupancy_distribution),
            "behaviour_profile_distribution": distribution_mapping_to_dict(
                self.behaviour_profile_distribution,
            ),
            "assistance_pairing_probability": self.assistance_pairing_probability,
        }

    # =====================================================

    @classmethod
    def from_dict(cls, data: dict) -> "OccupantDefinition":

        return cls(
            occupancy_distribution=distribution_mapping_from_dict(
                data.get("occupancy_distribution", {}),
            ),
            behaviour_profile_distribution=distribution_mapping_from_dict(
                data.get("behaviour_profile_distribution", {}),
            ),
            assistance_pairing_probability=data.get("assistance_pairing_probability", 0.0),
        )
