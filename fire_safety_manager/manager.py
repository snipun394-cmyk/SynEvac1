from typing import Dict, Mapping, Optional, Tuple

from fire_safety_manager.snapshot import FireSafetyStatusSnapshot, PassiveAssetCounts, SprinklerCounts
from fire_safety_manager.status import FireSafetyAssetStatus

from models.fire_extinguisher import FireExtinguisher
from models.fire_hydrant import FireHydrant
from models.fire_safety_asset import PassiveFireSafetyAvailability
from models.hose_reel import HoseReel
from models.sprinkler import Sprinkler, SprinklerActivationState


class FireSafetyAssetManager:

    # The single manager for every Fire Suppression & Water-Based
    # Safety asset (Sprinkler, FireExtinguisher, FireHydrant, HoseReel)
    # -- ONE coherent manager for all four, not four separate ones and
    # not folded into SensorManager/EmergencyLightManager. Investigated
    # deliberately (Phase 10 of this milestone's own brief): none of
    # the four fits an EXISTING manager's own semantics --
    # SensorManager exists specifically to aggregate ALARM/FAULT
    # conditions for FACP (Sprinkler is deliberately kept OUT of that
    # aggregation, see models.sprinkler.Sprinkler's own docstring, and
    # FireExtinguisher/FireHydrant/HoseReel have no alarm concept at
    # all), and EmergencyLightManager's own docstring already commits
    # to being specifically about lighting, not a general "misc safety
    # asset" dumping ground. All four of THESE assets, in contrast,
    # genuinely share one identity: physical fire-suppression/
    # firefighting RESOURCES a building has installed -- the same
    # coherent-family reasoning that justifies one manager rather than
    # four (Phase 10's own "avoid manager proliferation without
    # reason", read the other way: don't avoid a shared manager when
    # the assets genuinely belong together either).
    #
    # Same "never decides what should happen, only reports what
    # already exists" discipline every sibling manager already holds --
    # this class performs no suppression physics and no hazard
    # computation of any kind.

    def __init__(self):

        self._sprinklers: Dict[str, Sprinkler] = {}
        self._fire_extinguishers: Dict[str, FireExtinguisher] = {}
        self._fire_hydrants: Dict[str, FireHydrant] = {}
        self._hose_reels: Dict[str, HoseReel] = {}

    # =====================================================
    # Discovery / Registration
    # =====================================================

    def discover_assets(self, building) -> Tuple[object, ...]:

        self._sprinklers = {}
        self._fire_extinguishers = {}
        self._fire_hydrants = {}
        self._hose_reels = {}

        if building is None:
            return ()

        for floor in building.ordered_floors():

            for sprinkler in floor.sprinklers:
                self.register_sprinkler(sprinkler)

            for fire_extinguisher in floor.fire_extinguishers:
                self.register_fire_extinguisher(fire_extinguisher)

            for fire_hydrant in floor.fire_hydrants:
                self.register_fire_hydrant(fire_hydrant)

            for hose_reel in floor.hose_reels:
                self.register_hose_reel(hose_reel)

        return self.all_assets()

    # =====================================================

    def register_sprinkler(self, sprinkler: Sprinkler) -> None:

        self._sprinklers[sprinkler.id] = sprinkler

    def register_fire_extinguisher(self, fire_extinguisher: FireExtinguisher) -> None:

        self._fire_extinguishers[fire_extinguisher.id] = fire_extinguisher

    def register_fire_hydrant(self, fire_hydrant: FireHydrant) -> None:

        self._fire_hydrants[fire_hydrant.id] = fire_hydrant

    def register_hose_reel(self, hose_reel: HoseReel) -> None:

        self._hose_reels[hose_reel.id] = hose_reel

    # =====================================================
    # Lookup
    # =====================================================

    def sprinklers(self) -> Tuple[Sprinkler, ...]:

        return tuple(self._sprinklers.values())

    def fire_extinguishers(self) -> Tuple[FireExtinguisher, ...]:

        return tuple(self._fire_extinguishers.values())

    def fire_hydrants(self) -> Tuple[FireHydrant, ...]:

        return tuple(self._fire_hydrants.values())

    def hose_reels(self) -> Tuple[HoseReel, ...]:

        return tuple(self._hose_reels.values())

    def all_assets(self) -> Tuple[object, ...]:

        return self.sprinklers() + self.fire_extinguishers() + self.fire_hydrants() + self.hose_reels()

    # =====================================================

    def get_asset(self, asset_id: str) -> Optional[object]:

        return (
            self._sprinklers.get(asset_id)
            or self._fire_extinguishers.get(asset_id)
            or self._fire_hydrants.get(asset_id)
            or self._hose_reels.get(asset_id)
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
    # Status
    # =====================================================

    def status_of(
        self, asset_id: str, *, sprinkler_temperatures: Optional[Mapping[str, float]] = None, time: Optional[float] = None,
    ) -> FireSafetyAssetStatus:

        asset = self._require_asset(asset_id)

        return self._status_for(asset, sprinkler_temperatures=sprinkler_temperatures, time=time)

    # =====================================================

    def all_statuses(
        self, *, sprinkler_temperatures: Optional[Mapping[str, float]] = None, time: Optional[float] = None,
    ) -> Tuple[FireSafetyAssetStatus, ...]:

        return tuple(
            self._status_for(asset, sprinkler_temperatures=sprinkler_temperatures, time=time)
            for asset in self.all_assets()
        )

    # =====================================================

    def snapshot(
        self, *, sprinkler_temperatures: Optional[Mapping[str, float]] = None, time: Optional[float] = None,
    ) -> FireSafetyStatusSnapshot:

        entries = self.all_statuses(sprinkler_temperatures=sprinkler_temperatures, time=time)

        sprinkler_entries = tuple(entry for entry in entries if entry.asset_type == "Sprinkler")
        extinguisher_entries = tuple(entry for entry in entries if entry.asset_type == "FireExtinguisher")
        hydrant_entries = tuple(entry for entry in entries if entry.asset_type == "FireHydrant")
        hose_reel_entries = tuple(entry for entry in entries if entry.asset_type == "HoseReel")

        return FireSafetyStatusSnapshot(
            entries=entries,
            sprinklers=SprinklerCounts(
                total=len(sprinkler_entries),
                normal=sum(1 for e in sprinkler_entries if e.state == SprinklerActivationState.NORMAL.name),
                activated=sum(1 for e in sprinkler_entries if e.state == SprinklerActivationState.ACTIVATED.name),
                fault=sum(1 for e in sprinkler_entries if e.state == SprinklerActivationState.FAULT.name),
            ),
            fire_extinguishers=self._passive_counts(extinguisher_entries),
            fire_hydrants=self._passive_counts(hydrant_entries),
            hose_reels=self._passive_counts(hose_reel_entries),
        )

    # =====================================================
    # Internals
    # =====================================================

    def _passive_counts(self, entries: Tuple[FireSafetyAssetStatus, ...]) -> PassiveAssetCounts:

        return PassiveAssetCounts(
            total=len(entries),
            available=sum(1 for e in entries if e.state == PassiveFireSafetyAvailability.AVAILABLE),
            unavailable=sum(1 for e in entries if e.state == PassiveFireSafetyAvailability.UNAVAILABLE),
            fault=sum(1 for e in entries if e.state == PassiveFireSafetyAvailability.FAULT),
        )

    # =====================================================

    def _status_for(
        self, asset, *, sprinkler_temperatures: Optional[Mapping[str, float]], time: Optional[float],
    ) -> FireSafetyAssetStatus:

        if isinstance(asset, Sprinkler):

            # No temperature reading available this cycle is the
            # honest default (never a fabricated ACTIVATED) -- the same
            # "provider left unset means nothing available yet"
            # discipline every reading-driven asset in this codebase
            # already follows (see live_system.facp_gateway.
            # EngineFACPGateway's own smoke/heat reading providers).
            temperature = (
                sprinkler_temperatures.get(asset.id) if sprinkler_temperatures is not None else None
            )
            state = asset.compute_state(temperature, time).name

        else:

            state = asset.compute_availability()

        return FireSafetyAssetStatus(
            asset_id=asset.id, asset_type=asset.object_type, name=asset.name,
            floor_id=asset.floor_id, zone_ids=asset.zone_ids,
            active=asset.active, health_status=asset.health_status,
            state=state,
        )

    # =====================================================

    def _require_asset(self, asset_id: str):

        asset = self.get_asset(asset_id)

        if asset is None:
            raise KeyError(f"FireSafetyAssetManager has no asset registered with id {asset_id!r}.")

        return asset
