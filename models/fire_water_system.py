from dataclasses import dataclass, field
from typing import Optional, Tuple
from uuid import uuid4


@dataclass
class FireWaterSystem:

    # A lightweight LOGICAL grouping of fire-water infrastructure asset
    # ids -- explicitly NOT pipe geometry, NOT a hydraulic network, and
    # NOT inferred from physical proximity (Phase 7 of the Fire Water
    # Supply & Suppression Infrastructure milestone's own explicit
    # instruction). Every relationship here is authored/configured by
    # an operator through the Designer, never guessed from distance or
    # floor adjacency.
    #
    # Lives on Building (models/building.py), not Floor -- a real Fire
    # Water System routinely spans multiple floors (a basement tank/
    # pump room feeding hydrants on every floor above), so it cannot
    # honestly be scoped to one floor's own asset lists the way a Zone
    # or a point device is.
    #
    # Every *_ids field is a plain tuple of already-existing asset ids
    # (FireWaterTank.id / FirePump.id / JockeyPump.id / FireServiceInlet.id
    # / Sprinkler.id / FireHydrant.id / HoseReel.id) -- the same "plain id
    # reference, never resolved or validated here" convention
    # EngineeringAsset.zone_ids/Door.zone_a_id already establish.
    # Dangling references (an id that no longer names a real asset) are
    # reported by designer/validation.py, never silently dropped or
    # crashed on by this class itself.

    name: str

    id: str = field(default_factory=lambda: str(uuid4()))

    tank_ids: Tuple[str, ...] = field(default_factory=tuple)
    pump_ids: Tuple[str, ...] = field(default_factory=tuple)
    jockey_pump_ids: Tuple[str, ...] = field(default_factory=tuple)
    inlet_ids: Tuple[str, ...] = field(default_factory=tuple)

    sprinkler_ids: Tuple[str, ...] = field(default_factory=tuple)
    hydrant_ids: Tuple[str, ...] = field(default_factory=tuple)
    hose_reel_ids: Tuple[str, ...] = field(default_factory=tuple)

    # =====================================================

    def to_dict(self):

        return {
            "id": self.id,
            "name": self.name,
            "tank_ids": list(self.tank_ids),
            "pump_ids": list(self.pump_ids),
            "jockey_pump_ids": list(self.jockey_pump_ids),
            "inlet_ids": list(self.inlet_ids),
            "sprinkler_ids": list(self.sprinkler_ids),
            "hydrant_ids": list(self.hydrant_ids),
            "hose_reel_ids": list(self.hose_reel_ids),
        }

    # =====================================================

    @classmethod
    def from_dict(cls, data):

        return cls(
            id=data["id"],
            name=data.get("name", ""),
            tank_ids=tuple(data.get("tank_ids", ())),
            pump_ids=tuple(data.get("pump_ids", ())),
            jockey_pump_ids=tuple(data.get("jockey_pump_ids", ())),
            inlet_ids=tuple(data.get("inlet_ids", ())),
            sprinkler_ids=tuple(data.get("sprinkler_ids", ())),
            hydrant_ids=tuple(data.get("hydrant_ids", ())),
            hose_reel_ids=tuple(data.get("hose_reel_ids", ())),
        )


# =====================================================
# MEMBERSHIP_FIELDS -- the seven asset-id fields above, named once here
# so every generic membership helper (below) and every Property Panel
# "Fire Water System" combo (designer/widgets/property_panel.py) reads
# the exact same list rather than each re-declaring its own.
# =====================================================

MEMBERSHIP_FIELDS = (
    "tank_ids", "pump_ids", "jockey_pump_ids", "inlet_ids",
    "sprinkler_ids", "hydrant_ids", "hose_reel_ids",
)


def system_containing_asset(building, asset_id: str, field_name: str) -> Optional[FireWaterSystem]:

    # At most one FireWaterSystem should ever contain a given asset id
    # in a given field_name -- a Designer-authored invariant (assign_
    # asset_to_system() below always removes the id from every other
    # system's same field before adding it to the new one) rather than
    # something this function enforces itself. Returns the first match
    # if that invariant were ever violated by hand-edited project data,
    # never raises.

    if building is None:
        return None

    for system in building.fire_water_systems:

        if asset_id in getattr(system, field_name):
            return system

    return None


def assign_asset_to_system(building, asset_id: str, field_name: str, target_system_id: Optional[str]) -> None:

    # Reassignment, not addition: first removes asset_id from every
    # system's own field_name tuple (wherever it currently is), then --
    # only if target_system_id names a real, still-existing system --
    # adds it there. target_system_id=None (or an id naming no system)
    # leaves the asset in no system at all, mirroring exactly how
    # clearing a Zone combo leaves an asset's zone_ids empty rather than
    # raising.

    if building is None:
        return

    for system in building.fire_water_systems:

        current = getattr(system, field_name)

        if asset_id in current:
            setattr(system, field_name, tuple(i for i in current if i != asset_id))

    if not target_system_id:
        return

    for system in building.fire_water_systems:

        if system.id == target_system_id:

            setattr(system, field_name, getattr(system, field_name) + (asset_id,))
            return
