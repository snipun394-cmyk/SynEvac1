from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class FireWaterAssetStatus:

    # One infrastructure asset's current management status -- same
    # shared shape fire_safety_manager.status.FireSafetyAssetStatus
    # already established for its own, different asset family (Sprinkler/
    # FireExtinguisher/FireHydrant/HoseReel), independently restated
    # here rather than imported (this milestone's own asset family,
    # not a redesign of that one). `state` carries whichever vocabulary
    # the underlying asset actually produces (TankOperationalState /
    # PumpOperationalState / PassiveFireSafetyAvailability's value for
    # a FireServiceInlet), as a plain string.

    asset_id: str
    asset_type: str
    name: str
    floor_id: str
    zone_ids: Tuple[str, ...]

    active: bool
    health_status: str

    state: str
