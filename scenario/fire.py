from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from scenario.serialization import mapping_to_dict


@dataclass(frozen=True)
class ScenarioFire:

    # The resolved fire state inside a generated Scenario --
    # architecture doc §4.2/§7/§9. Plain sampled data only: no
    # fire_growth.model.FireGrowthModel instance, no growth curve
    # object, no HazardSource. §4.2's own reasoning is explicit about
    # why -- FireGrowthModel implements HazardSource and plugs
    # straight into simulation machinery, so constructing one here
    # would be preparing live simulation state, not describing a
    # Scenario. Turning this plain data into a FireGrowthModel is the
    # Simulator's job, at simulation start, not this package's.
    #
    # growth_parameters is deliberately an opaque, generically-named
    # str -> float mapping (e.g. {"growth_time": 300.0}) rather than a
    # fixed set of named fields -- this package has no opinion about
    # which growth model (t-squared or otherwise) a given
    # fire_profile implies; that interpretation belongs entirely to
    # fire_growth/, which this package must never import (§4.2/§12).
    #
    # fire_profile is a plain, open identifier string, not a closed
    # enum -- §9 leaves the Fire Profile schema itself unfinalized
    # ("architecture support only, no implementation"), so this model
    # commits to the least amount of structure that is still faithful
    # to the frozen schema: a label, exactly like behaviour_profile_id
    # (§8) is a label the Scenario Engine carries but never interprets.

    ignition_zone_id: str
    ignition_floor_id: str

    fire_profile: str

    growth_parameters: Mapping[str, float] = field(default_factory=dict)

    # =====================================================

    def __post_init__(self):

        object.__setattr__(
            self, "growth_parameters", MappingProxyType(dict(self.growth_parameters)),
        )

    # =====================================================

    def to_dict(self) -> dict:

        return {
            "ignition_zone_id": self.ignition_zone_id,
            "ignition_floor_id": self.ignition_floor_id,
            "fire_profile": self.fire_profile,
            "growth_parameters": mapping_to_dict(self.growth_parameters),
        }

    # =====================================================

    @classmethod
    def from_dict(cls, data: dict) -> "ScenarioFire":

        return cls(
            ignition_zone_id=data["ignition_zone_id"],
            ignition_floor_id=data["ignition_floor_id"],
            fire_profile=data["fire_profile"],
            growth_parameters=data.get("growth_parameters", {}),
        )
