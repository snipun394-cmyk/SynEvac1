from dataclasses import dataclass
from typing import Optional

from behavior_recognition.observation import RecognizedBehavior

from perception.models.human_observation import HumanClassification, HumanState

from event_bus.bus import Event, EventBus, EventType

from live_occupants.occupant import LiveOccupant


# Live Occupant Digital Twin milestone, Phase 9 -- "use existing event
# infrastructure if present, do not invent another event bus":
# live_system.event_bus.EventBus/EventType already exist and are
# already the one internal pub/sub mechanism every other live_system
# gateway uses; this module ADDS the 7 new EventType members this
# milestone needs (live_system/event_bus.py itself, whose own docstring
# already invites exactly this: "extending this list is the natural
# extension point for a future event") and defines the PAYLOAD shapes
# those event types carry -- Event.payload is deliberately typed Any at
# the EventBus level, so payload shapes live here, one layer up, the
# same way every other event's payload shape lives with its own
# publisher rather than inside event_bus.py itself.


@dataclass(frozen=True)
class OccupantCreatedPayload:
    occupant: LiveOccupant


@dataclass(frozen=True)
class OccupantUpdatedPayload:
    occupant: LiveOccupant


@dataclass(frozen=True)
class BehaviorChangedPayload:
    occupant_id: str
    from_behavior: Optional[RecognizedBehavior]
    to_behavior: Optional[RecognizedBehavior]


@dataclass(frozen=True)
class ZoneChangedPayload:
    occupant_id: str
    from_zone_id: Optional[str]
    to_zone_id: Optional[str]


@dataclass(frozen=True)
class CameraChangedPayload:
    occupant_id: str
    from_camera_id: Optional[str]
    to_camera_id: Optional[str]


@dataclass(frozen=True)
class OccupantExitedPayload:
    occupant: LiveOccupant


@dataclass(frozen=True)
class OccupantExpiredPayload:
    occupant: LiveOccupant


# Live Human State & Assistance Perception Bridge milestone, Phase 11 --
# mirrors BehaviorChangedPayload exactly. `to_classification`/`to_state`
# are the value AFTER reconciliation (human_evidence.reconciliation),
# never a raw, unreconciled per-camera reading.


@dataclass(frozen=True)
class ClassificationUpdatedPayload:
    occupant_id: str
    from_classification: HumanClassification
    to_classification: HumanClassification


@dataclass(frozen=True)
class StateChangedPayload:
    occupant_id: str
    from_state: Optional[HumanState]
    to_state: Optional[HumanState]


@dataclass(frozen=True)
class PossibleAssistanceRequiredPayload:
    occupant_id: str
    zone_id: Optional[str]


@dataclass(frozen=True)
class ConfirmedAssistanceRequiredPayload:
    occupant_id: str
    zone_id: Optional[str]
    human_state: HumanState


def publish(event_bus: Optional[EventBus], event_type: EventType, payload, timestamp: float) -> Optional[Event]:

    # A caller (live_occupants.manager.LiveOccupantManager) is always
    # constructed with an OPTIONAL event_bus -- publishing an occupant
    # lifecycle event is a side effect a caller may not want at all
    # (e.g. a unit test exercising the manager in isolation); None here
    # is simply "no bus configured", never an error.

    if event_bus is None:
        return None

    return event_bus.emit(event_type, payload, timestamp)
