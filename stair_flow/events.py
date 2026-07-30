from typing import Tuple

from stair_flow.models import StairFlowEvent, StairFlowEventType, TrafficDirection


# =====================================================
# Live Stair Flow & Movement Direction Intelligence milestone, Phase 1/6
# -- the ONE place ENTERED_STAIR/EXITED_STAIR events are derived from
# already-recorded evidence. Reads live_occupants.history.OccupantHistory.
# stair_transitions directly (already written by live_occupants.manager.
# LiveOccupantManager.update() on every genuine current_stair_id change --
# see that method's own `if stair_id != existing.current_stair_id:
# history = history.with_stair_transition(...)`) -- this module never
# re-derives, re-scans raw detections, or runs its own identity
# resolution. `occupant` is duck-typed (only `.occupant_id`, `.first_seen`,
# `.history.stair_transitions`, `.history.position_samples` are read).
#
# Multi-camera dedup (Phase 3) is INHERITED, not reimplemented: by the
# time a StairTransitionRecord exists in `occupant.history`, cross-camera
# identity resolution and multi-camera fusion have already collapsed
# every camera's own observation of this SAME physical person into ONE
# canonical occupant_id and ONE current_stair_id per LiveOccupantManager.
# update() call (see that method's own "there is exactly one spatial
# lookup per occupant per cycle" discipline, documented on
# live_occupants.occupant.LiveOccupant.current_stair_id itself). Two
# cameras seeing the same occupant enter the same stair in the same
# cycle produce, at most, ONE genuine value change and therefore ONE
# StairTransitionRecord -- a second update() call this same cycle with
# the SAME stair_id is a no-op on history (see docs/architecture/
# stair_flow_intelligence.md Sec "Transition identity" for the traced
# proof).
# =====================================================


def extract_stair_flow_events(occupant, window_start: float, window_end: float) -> Tuple[StairFlowEvent, ...]:

    position_by_timestamp = {
        sample.timestamp: sample.floor_id for sample in occupant.history.position_samples
    }

    events = []

    for record in occupant.history.stair_transitions:

        if record.timestamp <= window_start or record.timestamp > window_end:
            continue

        # Phase 6 edge case -- "occupant first appearing already on
        # stair": LiveOccupantManager.update() unconditionally writes
        # `with_stair_transition(timestamp, None, stair_id)` the very
        # first time ANY occupant_id is ever seen, regardless of whether
        # stair_id is None. When it is NOT None, this record LOOKS
        # exactly like a genuine None -> stair_id entry, but there is no
        # honest evidence of an actual entry EVENT -- tracking simply
        # began mid-traversal (the physical entry may have happened
        # seconds, minutes, or hours before this camera network ever saw
        # them). Excluded from entries entirely -- never counted as a
        # confirmed entry, even though the occupant genuinely IS on the
        # stair (that presence is already captured honestly elsewhere,
        # via observed_occupant_count -- see docs/architecture/
        # stair_flow_intelligence.md Sec "Phase 6"). A later, genuine
        # mid-stream transition can never share this occupant's
        # first_seen timestamp (timestamps only advance), so this check
        # can never misclassify a real entry.
        first_ever_observation = record.timestamp == occupant.first_seen and record.from_stair_id is None

        if record.from_stair_id is not None:

            events.append(StairFlowEvent(
                occupant_id=occupant.occupant_id, stair_id=record.from_stair_id,
                event_type=StairFlowEventType.EXITED_STAIR, direction=TrafficDirection.UNKNOWN,
                timestamp=record.timestamp, floor_id=position_by_timestamp.get(record.timestamp),
                provenance="confirmed current_stair_id -> None/other-stair transition",
            ))

        if record.to_stair_id is not None and not first_ever_observation:

            events.append(StairFlowEvent(
                occupant_id=occupant.occupant_id, stair_id=record.to_stair_id,
                event_type=StairFlowEventType.ENTERED_STAIR, direction=TrafficDirection.UNKNOWN,
                timestamp=record.timestamp, floor_id=position_by_timestamp.get(record.timestamp),
                provenance="confirmed None/other-stair -> current_stair_id transition",
            ))

    return tuple(events)
