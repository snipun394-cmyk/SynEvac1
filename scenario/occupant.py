from dataclasses import dataclass
from typing import Optional, Tuple

from scenario.serialization import position_from_list, position_to_list


@dataclass(frozen=True)
class ScenarioOccupant:

    # One resolved occupant record inside a generated Scenario --
    # architecture doc §7/§8. behaviour_profile_id is an opaque
    # identifier string (e.g. "Adult_Default"); this package never
    # resolves, interprets, or validates what it means -- that is
    # exclusively the Human Behavior Layer's job, at simulation time
    # (§8, frozen). Nothing here samples walking speed, panic, route
    # choice, or any other behavioral quantity -- those don't exist
    # until the Behaviour Layer resolves the id.
    #
    # assisting_occupant_id/assistance_type -- Human Population &
    # Assisted Evacuation Modeling Phase 3, additive: both None by
    # default, so every existing occupant/scenario is unaffected.
    # Same "opaque identifier string, never interpreted here" pattern
    # as behaviour_profile_id -- assistance_type is a plain string
    # (e.g. "HELP"/"ESCORT"/"PUSH_WHEELCHAIR"/"DRAG"/"CARRY", the
    # behaviour_library.assistance_strategies vocabulary), resolved
    # only at Behaviour Profile Resolver / Human Behavior Layer time,
    # never here. assisting_occupant_id points from the occupant doing
    # the assisting to the occupant_id of whoever they are assisting
    # (never the reverse direction) -- behaviour_profile_resolver.
    # registrar derives "who is being assisted by whom" for both sides
    # from this one field.

    occupant_id: str

    zone_id: str
    floor_id: str

    position: Tuple[float, float]

    behaviour_profile_id: str

    assisting_occupant_id: Optional[str] = None
    assistance_type: Optional[str] = None

    # =====================================================

    def to_dict(self) -> dict:

        return {
            "occupant_id": self.occupant_id,
            "zone_id": self.zone_id,
            "floor_id": self.floor_id,
            "position": position_to_list(self.position),
            "behaviour_profile_id": self.behaviour_profile_id,
            "assisting_occupant_id": self.assisting_occupant_id,
            "assistance_type": self.assistance_type,
        }

    # =====================================================

    @classmethod
    def from_dict(cls, data: dict) -> "ScenarioOccupant":

        return cls(
            occupant_id=data["occupant_id"],
            zone_id=data["zone_id"],
            floor_id=data["floor_id"],
            position=position_from_list(data["position"]),
            behaviour_profile_id=data["behaviour_profile_id"],
            assisting_occupant_id=data.get("assisting_occupant_id"),
            assistance_type=data.get("assistance_type"),
        )
