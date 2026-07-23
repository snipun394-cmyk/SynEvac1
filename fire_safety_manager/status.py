from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class FireSafetyAssetStatus:

    # One asset's current management status -- same "derived, never
    # stored, produced fresh by the manager" role as sign_manager.
    # status.SignStatus/speaker_manager.status.SpeakerStatus/
    # emergency_light_manager.status.EmergencyLightStatus. Deliberately
    # ONE shape shared by all four asset types (Sprinkler/
    # FireExtinguisher/FireHydrant/HoseReel) rather than four separate
    # status classes -- `state` carries whichever vocabulary the
    # underlying asset actually produces (SprinklerActivationState's
    # NORMAL/ACTIVATED/FAULT for a Sprinkler; PassiveFireSafetyAvailability's
    # AVAILABLE/UNAVAILABLE/FAULT for the other three), as a plain
    # string -- the same "Source / Type / Zone / State" shape
    # command_center.live_status_panel.LiveStatusPanel.facp_sources_table
    # already established for a completely different asset family,
    # reused here because the display need is genuinely identical.

    asset_id: str
    asset_type: str
    name: str
    floor_id: str
    zone_ids: Tuple[str, ...]

    active: bool
    health_status: str

    state: str
