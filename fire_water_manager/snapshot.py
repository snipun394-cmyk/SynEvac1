from dataclasses import dataclass, field
from typing import Tuple

from fire_water_manager.status import FireWaterAssetStatus


class FireWaterSystemStatus:

    # Deliberately conservative (Phase 11 of the Fire Water Supply &
    # Suppression Infrastructure milestone's own explicit instruction):
    # these four values answer ONLY "does the configured supply
    # infrastructure appear operational", never "is required hydraulic
    # demand satisfied" -- no member of this vocabulary, and nothing
    # that reads it, may be treated as a hydraulic-adequacy guarantee.

    SYSTEM_AVAILABLE = "SYSTEM_AVAILABLE"
    SYSTEM_DEGRADED = "SYSTEM_DEGRADED"
    SYSTEM_UNAVAILABLE = "SYSTEM_UNAVAILABLE"
    UNKNOWN = "UNKNOWN"

    ALL = (SYSTEM_AVAILABLE, SYSTEM_DEGRADED, SYSTEM_UNAVAILABLE, UNKNOWN)


@dataclass(frozen=True)
class FireWaterSystemStatusReport:

    # One FireWaterSystem's (models/fire_water_system.py) computed
    # status -- `reasons` names exactly which configured member(s)
    # degraded it (Phase 11's own "explicitly expose why a system is
    # degraded" instruction), never a bare status code with no
    # explanation. sprinkler_ids/hydrant_ids/hose_reel_ids are copied
    # straight from the FireWaterSystem for traceability only (Phase 15:
    # "SynEvac should be able to trace this dependency") -- their own
    # health is never assessed or folded into `status` here; that
    # remains fire_safety_manager's own, separate concern (Phase 14's
    # own "two independent facts" instruction).

    system_id: str
    name: str

    status: str = FireWaterSystemStatus.UNKNOWN
    reasons: Tuple[str, ...] = field(default_factory=tuple)

    tank_ids: Tuple[str, ...] = field(default_factory=tuple)
    pump_ids: Tuple[str, ...] = field(default_factory=tuple)
    jockey_pump_ids: Tuple[str, ...] = field(default_factory=tuple)
    inlet_ids: Tuple[str, ...] = field(default_factory=tuple)

    sprinkler_ids: Tuple[str, ...] = field(default_factory=tuple)
    hydrant_ids: Tuple[str, ...] = field(default_factory=tuple)
    hose_reel_ids: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class FireWaterInfrastructureSnapshot:

    # BuildingState's additive view over FireWaterInfrastructureManager
    # (see that class's own snapshot() method, which produces this
    # value -- BuildingStateEstimator only ever passes it through,
    # never computes it, the same discipline facp_status/control_status/
    # fire_safety_status already establish). None on BuildingState when
    # no FireWaterInfrastructureManager is configured for this cycle --
    # never fabricated.

    entries: Tuple[FireWaterAssetStatus, ...] = field(default_factory=tuple)
    systems: Tuple[FireWaterSystemStatusReport, ...] = field(default_factory=tuple)
