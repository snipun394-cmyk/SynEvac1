from typing import Tuple

from facp.models import PanelEvent, PanelEventType


class PanelEventLog:

    # An append-only store for one panel's PanelEvent history (Phase 4)
    # -- deliberately its own small class, independent of SimulatedFACP,
    # so a future Live/Replay adapter that produces canonical PanelEvents
    # some other way (see facp/provider.py's FACPEventProvider seam) can
    # reuse the exact same query surface without depending on
    # SimulatedFACP's own state machine at all.

    def __init__(self):

        self._events: list = []

    # =====================================================

    def append(self, event: PanelEvent) -> None:

        self._events.append(event)

    # =====================================================

    def all_events(self) -> Tuple[PanelEvent, ...]:

        return tuple(self._events)

    # =====================================================

    def events_of_type(self, event_type: PanelEventType) -> Tuple[PanelEvent, ...]:

        return tuple(event for event in self._events if event.event_type == event_type)

    # =====================================================

    def events_for_asset(self, asset_id: str) -> Tuple[PanelEvent, ...]:

        return tuple(event for event in self._events if event.source_asset_id == asset_id)

    # =====================================================

    def events_for_floor(self, floor_id: str) -> Tuple[PanelEvent, ...]:

        return tuple(event for event in self._events if event.floor_id == floor_id)

    # =====================================================

    def events_for_zone(self, zone_id: str) -> Tuple[PanelEvent, ...]:

        return tuple(event for event in self._events if zone_id in event.zone_ids)

    # =====================================================

    def events_between(self, start_time: float, end_time: float) -> Tuple[PanelEvent, ...]:

        return tuple(
            event for event in self._events if start_time <= event.timestamp <= end_time
        )

    # =====================================================

    def recent(self, count: int) -> Tuple[PanelEvent, ...]:

        return tuple(self._events[-count:]) if count > 0 else ()

    # =====================================================

    def __len__(self) -> int:

        return len(self._events)
