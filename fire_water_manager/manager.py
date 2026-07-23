from typing import Dict, Optional, Tuple

from fire_water_manager.snapshot import FireWaterInfrastructureSnapshot, FireWaterSystemStatus, FireWaterSystemStatusReport
from fire_water_manager.status import FireWaterAssetStatus

from models.fire_pump import FirePump
from models.fire_service_inlet import FireServiceInlet
from models.fire_safety_asset import PassiveFireSafetyAvailability
from models.fire_water_tank import FireWaterTank, TankOperationalState
from models.jockey_pump import JockeyPump
from models.pump_asset import PumpOperationalState


# A pump reporting either of these is a healthy, normal condition for
# a fire pump's own automatic-standby lifecycle -- STOPPED is exactly
# what a correctly-functioning automatic fire pump looks like between
# demand events, never itself a degradation.
_PUMP_HEALTHY_STATES = (PumpOperationalState.STOPPED, PumpOperationalState.RUNNING)

# LOW_LEVEL is a warning worth surfacing in reasons but not, on its
# own, treated as the tank being unable to supply at all.
_TANK_HEALTHY_STATES = (TankOperationalState.AVAILABLE, TankOperationalState.LOW_LEVEL)

_INLET_HEALTHY_STATES = (PassiveFireSafetyAvailability.AVAILABLE,)


class FireWaterInfrastructureManager:

    # The single manager for FireWaterTank/FirePump/JockeyPump/
    # FireServiceInlet -- ONE coherent manager for all four, mirroring
    # fire_safety_manager.manager.FireSafetyAssetManager's own "genuine
    # coherent family, not four separate managers" reasoning: all four
    # are physical fire-WATER-SUPPLY infrastructure, a different family
    # from Sprinkler/FireExtinguisher/FireHydrant/HoseReel's own
    # "suppression/firefighting resource" family, so this is
    # deliberately its own manager rather than folded into
    # FireSafetyAssetManager.
    #
    # Also computes FireWaterSystem (models/fire_water_system.py)
    # status -- deterministic, conservative, and explicit about WHY a
    # system is degraded (Phase 11's own instruction) -- never a
    # hydraulic calculation of any kind.

    def __init__(self):

        self._tanks: Dict[str, FireWaterTank] = {}
        self._pumps: Dict[str, FirePump] = {}
        self._jockey_pumps: Dict[str, JockeyPump] = {}
        self._inlets: Dict[str, FireServiceInlet] = {}

        self._building = None

    # =====================================================
    # Discovery / Registration
    # =====================================================

    def discover_assets(self, building) -> Tuple[object, ...]:

        self._tanks = {}
        self._pumps = {}
        self._jockey_pumps = {}
        self._inlets = {}

        self._building = building

        if building is None:
            return ()

        for floor in building.ordered_floors():

            for tank in floor.fire_water_tanks:
                self.register_tank(tank)

            for pump in floor.fire_pumps:
                self.register_fire_pump(pump)

            for jockey_pump in floor.jockey_pumps:
                self.register_jockey_pump(jockey_pump)

            for inlet in floor.fire_service_inlets:
                self.register_inlet(inlet)

        return self.all_assets()

    # =====================================================

    def register_tank(self, tank: FireWaterTank) -> None:

        self._tanks[tank.id] = tank

    def register_fire_pump(self, pump: FirePump) -> None:

        self._pumps[pump.id] = pump

    def register_jockey_pump(self, jockey_pump: JockeyPump) -> None:

        self._jockey_pumps[jockey_pump.id] = jockey_pump

    def register_inlet(self, inlet: FireServiceInlet) -> None:

        self._inlets[inlet.id] = inlet

    # =====================================================
    # Lookup
    # =====================================================

    def tanks(self) -> Tuple[FireWaterTank, ...]:

        return tuple(self._tanks.values())

    def fire_pumps(self) -> Tuple[FirePump, ...]:

        return tuple(self._pumps.values())

    def jockey_pumps(self) -> Tuple[JockeyPump, ...]:

        return tuple(self._jockey_pumps.values())

    def inlets(self) -> Tuple[FireServiceInlet, ...]:

        return tuple(self._inlets.values())

    def all_assets(self) -> Tuple[object, ...]:

        return self.tanks() + self.fire_pumps() + self.jockey_pumps() + self.inlets()

    # =====================================================

    def get_asset(self, asset_id: str) -> Optional[object]:

        return (
            self._tanks.get(asset_id)
            or self._pumps.get(asset_id)
            or self._jockey_pumps.get(asset_id)
            or self._inlets.get(asset_id)
        )

    # =====================================================

    def assets_on_floor(self, floor_id: str) -> Tuple[object, ...]:

        return tuple(asset for asset in self.all_assets() if asset.floor_id == floor_id)

    # =====================================================

    def assets_in_zone(self, zone_id: str) -> Tuple[object, ...]:

        return tuple(asset for asset in self.all_assets() if zone_id in asset.zone_ids)

    # =====================================================
    # Enable / Disable
    # =====================================================

    def enable_asset(self, asset_id: str) -> None:

        self._require_asset(asset_id).active = True

    def disable_asset(self, asset_id: str) -> None:

        self._require_asset(asset_id).active = False

    # =====================================================
    # Asset Status
    # =====================================================

    def status_of(self, asset_id: str) -> FireWaterAssetStatus:

        return self._status_for(self._require_asset(asset_id))

    # =====================================================

    def all_statuses(self) -> Tuple[FireWaterAssetStatus, ...]:

        return tuple(self._status_for(asset) for asset in self.all_assets())

    # =====================================================
    # System Status
    # =====================================================

    def system_status(self, system) -> FireWaterSystemStatusReport:

        reasons = []

        tank_states = self._member_states(system.tank_ids, self._tanks, reasons, "tank")
        pump_states = self._member_states(system.pump_ids, self._pumps, reasons, "pump")
        jockey_states = self._member_states(system.jockey_pump_ids, self._jockey_pumps, reasons, "jockey pump")
        inlet_states = self._member_states(system.inlet_ids, self._inlets, reasons, "inlet")

        has_any_supply = bool(system.tank_ids or system.pump_ids or system.jockey_pump_ids or system.inlet_ids)

        if not has_any_supply:

            return FireWaterSystemStatusReport(
                system_id=system.id, name=system.name, status=FireWaterSystemStatus.UNKNOWN,
                reasons=("no supply assets configured",),
                sprinkler_ids=system.sprinkler_ids, hydrant_ids=system.hydrant_ids, hose_reel_ids=system.hose_reel_ids,
            )

        self._collect_degraded_reasons(tank_states, _TANK_HEALTHY_STATES, "tank", reasons)
        self._collect_degraded_reasons(pump_states, _PUMP_HEALTHY_STATES, "pump", reasons)
        self._collect_degraded_reasons(jockey_states, _PUMP_HEALTHY_STATES, "jockey pump", reasons)
        self._collect_degraded_reasons(inlet_states, _INLET_HEALTHY_STATES, "inlet", reasons)

        healthy_flags = (
            [state in _TANK_HEALTHY_STATES for state, _ in tank_states]
            + [state in _PUMP_HEALTHY_STATES for state, _ in pump_states]
            + [state in _PUMP_HEALTHY_STATES for state, _ in jockey_states]
            + [state in _INLET_HEALTHY_STATES for state, _ in inlet_states]
        )

        if healthy_flags and all(not healthy for healthy in healthy_flags):
            status = FireWaterSystemStatus.SYSTEM_UNAVAILABLE
        elif reasons:
            status = FireWaterSystemStatus.SYSTEM_DEGRADED
        else:
            status = FireWaterSystemStatus.SYSTEM_AVAILABLE

        return FireWaterSystemStatusReport(
            system_id=system.id, name=system.name, status=status, reasons=tuple(reasons),
            tank_ids=system.tank_ids, pump_ids=system.pump_ids,
            jockey_pump_ids=system.jockey_pump_ids, inlet_ids=system.inlet_ids,
            sprinkler_ids=system.sprinkler_ids, hydrant_ids=system.hydrant_ids, hose_reel_ids=system.hose_reel_ids,
        )

    # =====================================================

    def snapshot(self) -> FireWaterInfrastructureSnapshot:

        systems = (
            tuple(self.system_status(system) for system in self._building.fire_water_systems)
            if self._building is not None else ()
        )

        return FireWaterInfrastructureSnapshot(entries=self.all_statuses(), systems=systems)

    # =====================================================
    # Internals
    # =====================================================

    def _member_states(self, member_ids, registry, reasons, label):

        states = []

        for member_id in member_ids:

            asset = registry.get(member_id)

            if asset is None:
                reasons.append(f"{label} {member_id} referenced but not found")
                continue

            state = asset.compute_state() if hasattr(asset, "compute_state") else asset.compute_availability()
            states.append((state, asset))

        return states

    # =====================================================

    def _collect_degraded_reasons(self, states, healthy_states, label, reasons):

        for state, asset in states:

            if state not in healthy_states:
                reasons.append(f"{label} {asset.name or asset.id} {state.lower()}")

    # =====================================================

    def _status_for(self, asset) -> FireWaterAssetStatus:

        state = asset.compute_state() if hasattr(asset, "compute_state") else asset.compute_availability()

        return FireWaterAssetStatus(
            asset_id=asset.id, asset_type=asset.object_type, name=asset.name,
            floor_id=asset.floor_id, zone_ids=asset.zone_ids,
            active=asset.active, health_status=asset.health_status,
            state=state,
        )

    # =====================================================

    def _require_asset(self, asset_id: str):

        asset = self.get_asset(asset_id)

        if asset is None:
            raise KeyError(f"FireWaterInfrastructureManager has no asset registered with id {asset_id!r}.")

        return asset
