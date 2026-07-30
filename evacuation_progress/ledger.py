import dataclasses
from typing import Dict, FrozenSet, Optional, Set

from event_bus.bus import EventBus, EventType

from live_occupants import events as occupant_events
from live_occupants.state import OccupantStatus

from crowd_intelligence.flow import distance_to_asset, exit_sides


# =====================================================
# Live Evacuation Progress, Flow & Clearance Intelligence milestone,
# Phase 3/8/9 -- the ONE durable, event-driven memory this package
# needs. live_occupants.manager.LiveOccupantManager's own store is
# DELIBERATELY transient for EXPIRED occupants (they are removed from
# it entirely -- see LiveOccupantManager._expire()), so a caller that
# only ever re-queries the manager's CURRENT state each cycle would
# silently lose every occupant the moment they expire, permanently
# undercounting "known exited" the longer a session runs. This ledger
# subscribes to the SAME live_occupants event stream every other
# consumer already uses (OCCUPANT_CREATED/UPDATED/EXITED/EXPIRED),
# recording durable facts BEFORE an occupant vanishes from the manager.
#
# CRITICAL, investigated finding this milestone had to fix as a
# prerequisite: live_runtime.factory.build_live_runtime() previously
# constructed LiveOccupantManager() with NEITHER an event_bus NOR an
# exits= list -- meaning, in production, NO occupant lifecycle event
# was ever published at all, AND OccupantStatus.EXITED was completely
# unreachable (live_occupants.lifecycle.is_near_exit() always returns
# False against an empty exits list). Both gaps are fixed in
# live_runtime/factory.py as part of this milestone (see that file's
# own comment) -- this ledger is the reason the fix was needed.
#
# Reappearance correction (Phase 9) falls out of this design for free:
# an occupant recorded here as durably exited is REMOVED the moment
# OCCUPANT_UPDATED reports them ACTIVE again (a real, current sighting
# always wins over an earlier uncertain disappearance) -- the ledger
# never needs to distinguish "was this a correction" from "was this
# always right," it just always reflects the LATEST evidence.
# =====================================================


@dataclasses.dataclass(frozen=True)
class ExitedRecord:

    occupant_id: str
    exit_id: Optional[str]  # None -- exited near an exit, but WHICH one is unknown (no world_position)
    zone_id: Optional[str]
    timestamp: float


class EvacuationLedger:

    def __init__(self, event_bus: EventBus, building):

        self.event_bus = event_bus
        self.building = building

        self._ever_seen_ids: Set[str] = set()
        self._ever_seen_in_zone: Dict[str, Set[str]] = {}
        self._durable_exited: Dict[str, ExitedRecord] = {}

        self._exit_sides_by_id = self._index_exit_geometry()

        event_bus.subscribe(EventType.OCCUPANT_CREATED, self._on_created)
        event_bus.subscribe(EventType.OCCUPANT_UPDATED, self._on_updated)
        event_bus.subscribe(EventType.OCCUPANT_ZONE_CHANGED, self._on_zone_changed)
        event_bus.subscribe(EventType.OCCUPANT_EXITED, self._on_exited)

    # =====================================================

    def _index_exit_geometry(self):

        index = {}

        for floor in self.building.ordered_floors():
            for exit_obj in floor.exits:
                index[exit_obj.id] = exit_sides(exit_obj)

        return index

    # =====================================================
    # Event handlers -- each does the smallest possible update; none of
    # them ever calls back into LiveOccupantManager (a subscriber never
    # queries the thing it's subscribed to mid-callback, avoiding any
    # ordering coupling with sweep_missing()'s own in-progress loop).
    # =====================================================

    def _on_created(self, event) -> None:

        occupant = event.payload.occupant

        self._ever_seen_ids.add(occupant.occupant_id)

        if occupant.current_zone_id is not None:
            self._ever_seen_in_zone.setdefault(occupant.current_zone_id, set()).add(occupant.occupant_id)

    # =====================================================

    def _on_zone_changed(self, event) -> None:

        payload = event.payload

        if payload.to_zone_id is not None:
            self._ever_seen_in_zone.setdefault(payload.to_zone_id, set()).add(payload.occupant_id)

    # =====================================================

    def _on_updated(self, event) -> None:

        occupant = event.payload.occupant

        # A real, current sighting -- the occupant is unambiguously
        # ACTIVE right now. If they were previously recorded as
        # durably exited, that earlier guess is corrected immediately
        # (Phase 9). Never removes a NEW/ACTIVE occupant that was never
        # in the exited ledger to begin with (a plain no-op dict pop).
        if occupant.status in (OccupantStatus.ACTIVE, OccupantStatus.NEW):
            self._durable_exited.pop(occupant.occupant_id, None)

    # =====================================================

    def _on_exited(self, event) -> None:

        occupant = event.payload.occupant

        exit_id = self._nearest_exit_id(occupant)

        self._durable_exited[occupant.occupant_id] = ExitedRecord(
            occupant_id=occupant.occupant_id, exit_id=exit_id,
            zone_id=occupant.current_zone_id, timestamp=occupant.last_seen,
        )

    # =====================================================

    def _nearest_exit_id(self, occupant) -> Optional[str]:

        # Honest degradation (Phase 16 test 16): no world_position means
        # no honest basis to attribute this exit crossing to any ONE
        # specific exit, even though live_occupants.lifecycle.
        # is_near_exit() already confirmed (using world_position, before
        # it went missing) that SOME exit was nearby. The occupant still
        # counts toward known_exited_occupants overall -- only the
        # per-exit attribution is None.

        if occupant.world_position is None:
            return None

        best_exit_id = None
        best_distance = None

        for exit_id, sides in sorted(self._exit_sides_by_id.items()):

            distance = distance_to_asset(occupant.world_position, occupant.current_floor_id, sides)

            if distance is None:
                continue

            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_exit_id = exit_id

        return best_exit_id

    # =====================================================
    # Queries
    # =====================================================

    def ever_seen_ids(self) -> FrozenSet[str]:

        return frozenset(self._ever_seen_ids)

    def ever_seen_in_zone(self, zone_id: str) -> FrozenSet[str]:

        return frozenset(self._ever_seen_in_zone.get(zone_id, ()))

    def known_exited_occupant_ids(self) -> FrozenSet[str]:

        return frozenset(self._durable_exited.keys())

    def exited_records_for_exit(self, exit_id: str):

        return tuple(record for record in self._durable_exited.values() if record.exit_id == exit_id)

    def recent_exit_count(self, exit_id: str, since_time: float) -> int:

        return sum(
            1 for record in self._durable_exited.values()
            if record.exit_id == exit_id and record.timestamp >= since_time
        )

    def any_exit_position_available(self, exit_id: str) -> bool:

        # Whether at least one occupant WHO EXITED at this exit had a
        # known position (i.e. this exit's own attribution has ever
        # been possible) -- used only as one honest input among several
        # to ExitFlow.position_available; the engine also folds in
        # crowd_intelligence's own live position_available signal for
        # occupants still approaching.

        return any(record.exit_id == exit_id for record in self._durable_exited.values())
