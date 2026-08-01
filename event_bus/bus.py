from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List
from uuid import uuid4


# =====================================================
# Shadow-Mode Predictive AI Integration milestone, Phase 1 -- moved
# verbatim from live_system/event_bus.py (the Core Architecture Freeze
# Review's own Finding 1: this type has zero project imports of its own
# -- a genuinely leaf, dependency-free pub-sub mechanism -- but was
# physically nested inside live_system, whose own __init__.py eagerly
# imports the ENTIRE package, including ai_registry/orchestrator/
# sensor_registry. Every consumer of this trivial type (live_occupants,
# evacuation_progress, ...) was transitively forced to load the whole
# orchestration stack just to get EventBus/EventType, a real package-
# level circular dependency confirmed by that review's own AST-based
# import scan.
#
# ZERO behavior change: every class/method below is byte-identical to
# its live_system/event_bus.py predecessor. live_system/event_bus.py
# itself now re-exports these same names (see that file's own docstring)
# so `from live_system.event_bus import EventBus` keeps working
# unchanged for every existing caller -- this move only changes WHERE
# the real implementation lives, never what it does.
# =====================================================


class EventType(Enum):

    # The complete, exhaustive vocabulary this internal bus carries --
    # deliberately not open to ad-hoc string topics, so every publisher
    # and subscriber shares one typo-proof vocabulary. Mirrors
    # simulator.decision.ActionType's "a fixed enum, not a free string"
    # convention. Extending this list is the natural extension point
    # for a future event; nothing about EventBus itself needs to change
    # to add one.

    SENSOR_UPDATE = auto()
    ALARM_ACTIVATED = auto()
    HAZARD_UPDATED = auto()
    OCCUPANCY_UPDATED = auto()
    BUILDING_STATE_UPDATED = auto()
    AI_PREDICTION_UPDATED = auto()
    ADVISORY_REPORT_UPDATED = auto()
    CROWD_INTELLIGENCE_UPDATED = auto()
    EVACUATION_PROGRESS_UPDATED = auto()
    ZONE_CLEARANCE_STALLED = auto()
    EXIT_FLOW_STALLED = auto()
    ZONE_OBSERVED_CLEAR = auto()
    RESPONSE_PRIORITY_UPDATED = auto()
    ZONE_RESPONSE_ESCALATED = auto()
    ZONE_RESPONSE_DEESCALATED = auto()
    POSSIBLE_ASSISTANCE_DETECTED = auto()
    RECOMMENDATION_UPDATED = auto()
    OPERATOR_ACKNOWLEDGEMENT = auto()

    # Live Occupant Digital Twin milestone -- published by
    # live_occupants.manager.LiveOccupantManager, payloads defined in
    # live_occupants.events (this module deliberately never imports
    # live_occupants itself, keeping payload shapes typed Any at this
    # level, same as every other event above).
    OCCUPANT_CREATED = auto()
    OCCUPANT_UPDATED = auto()
    OCCUPANT_BEHAVIOR_CHANGED = auto()
    OCCUPANT_ZONE_CHANGED = auto()
    OCCUPANT_CAMERA_CHANGED = auto()
    OCCUPANT_EXITED = auto()
    OCCUPANT_EXPIRED = auto()

    # Live Human State & Assistance Perception Bridge milestone --
    # published by live_occupants.manager.LiveOccupantManager, payloads
    # defined in live_occupants.events (same "this module never imports
    # live_occupants itself" convention as the seven members above).
    OCCUPANT_CLASSIFICATION_UPDATED = auto()
    OCCUPANT_STATE_CHANGED = auto()
    POSSIBLE_ASSISTANCE_REQUIRED = auto()
    CONFIRMED_ASSISTANCE_REQUIRED = auto()

    # Live Occupant Trajectory, Movement Anomaly & Route-Deviation
    # Intelligence milestone, Phase 25 -- published by live_system.
    # orchestrator.LiveOrchestrator, transition-only (fired the cycle a
    # condition NEWLY becomes/stops being true, never every cycle it
    # merely continues to hold -- mirrors ZONE_CLEARANCE_STALLED/
    # ZONE_RESPONSE_ESCALATED's own established discipline).
    TRAJECTORY_INTELLIGENCE_UPDATED = auto()
    OCCUPANT_ROUTE_DEVIATION_DETECTED = auto()
    OCCUPANT_ROUTE_RECOVERED = auto()
    OCCUPANT_MOVEMENT_STALLED = auto()
    OCCUPANT_MOVEMENT_RESUMED = auto()
    OCCUPANT_ENTERED_HAZARDOUS_ZONE = auto()
    OCCUPANT_EXITED_HAZARDOUS_ZONE = auto()
    SHARED_ROUTE_DEVIATION_DETECTED = auto()

    # Live Dynamic Evacuation Recommendation Engine milestone --
    # published by live_system.orchestrator.LiveOrchestrator,
    # transition-only (fired the cycle a zone's own recommendation
    # genuinely changes, or the building-wide safe-exit set genuinely
    # changes -- never every cycle it merely continues to hold, mirrors
    # every prior milestone's own established discipline).
    EVACUATION_RECOMMENDATION_UPDATED = auto()
    RECOMMENDATION_CHANGED = auto()
    SAFE_EXIT_CHANGED = auto()
    NO_SAFE_EXIT = auto()
    RECOVERY_OF_SAFE_EXIT = auto()

    # Live Evacuation Guidance & Zoned Message Planning milestone --
    # published by live_system.orchestrator.LiveOrchestrator,
    # transition-only (fired the cycle a zone's own guidance genuinely
    # changes/becomes unavailable/recovers, or its delivery coverage
    # genuinely changes -- never every cycle it merely continues to
    # hold, mirrors every prior milestone's own established discipline).
    EVACUATION_GUIDANCE_UPDATED = auto()
    EVACUATION_ROUTE_CHANGED = auto()
    EVACUATION_GUIDANCE_UNAVAILABLE = auto()
    EVACUATION_GUIDANCE_RECOVERED = auto()
    GUIDANCE_DELIVERY_UNAVAILABLE = auto()

    # Live Dynamic Evacuation Signage milestone, Phase 23 -- published
    # by live_system.orchestrator.LiveOrchestrator, transition-only
    # (fired the cycle a sign's own effective instruction genuinely
    # changes/becomes unavailable/recovers, or a conflict is newly
    # detected/cleared -- never every cycle it merely continues to
    # hold, mirrors every prior milestone's own established discipline).
    # The approval-workflow events (APPROVED/REJECTED) are instead
    # published by the Command Center's own operator-action seam, the
    # one place an explicit operator action happens.
    SIGNAGE_PLAN_UPDATED = auto()
    SIGNAGE_INSTRUCTION_CHANGED = auto()
    SIGNAGE_CONFLICT_DETECTED = auto()
    SIGNAGE_CONFLICT_CLEARED = auto()
    SIGNAGE_UNAVAILABLE = auto()
    SIGNAGE_RECOVERED = auto()
    SIGNAGE_INSTRUCTION_APPROVED = auto()
    SIGNAGE_INSTRUCTION_REJECTED = auto()

    # Expose Real Live Camera Occupant State In SynEvac UI milestone --
    # published by live_system.orchestrator.LiveOrchestrator once per
    # cycle a live_occupants_gateway is configured and succeeds, mirroring
    # CROWD_INTELLIGENCE_UPDATED's own "fires every cycle it is computed,
    # not transition-only" discipline (a live occupant table is meant to
    # refresh every tick, not only on a lifecycle transition).
    LIVE_OCCUPANTS_UPDATED = auto()

    # Shadow-Mode Predictive AI Integration milestone -- published by
    # live_system.orchestrator.LiveOrchestrator once per cycle a Shadow-
    # Mode prediction gateway is configured, mirroring LIVE_OCCUPANTS_
    # UPDATED's own "fires every cycle it is computed, not transition-
    # only" discipline (a shadow prediction log is meant to grow every
    # tick a prediction was genuinely attempted, whether it succeeded,
    # was skipped, or errored). Never consumed by Recommendation/
    # Guidance/Voice/Signage/Building Control/Simulation -- see
    # docs/architecture/shadow_mode_prediction.md.
    SHADOW_PREDICTION_RECORDED = auto()

    # The Recommendation Layer milestone -- published by live_system.
    # orchestrator.LiveOrchestrator once per cycle a recommendation_
    # layer_gateway is configured and succeeds. Fires every cycle, not
    # transition-only, since RecommendationManager itself already owns
    # the create/update/expire lifecycle for individual recommendations.
    RECOMMENDATION_SET_UPDATED = auto()


@dataclass(frozen=True)
class Event:

    # One immutable fact published on the bus -- payload is
    # deliberately typed Any (not a per-event-type union) so this
    # module stays free of a hard import dependency on every package
    # that might ever publish something; same "typed Any, not imported
    # for its own sake" convention decision_policy.policy.DecisionInputs.
    # ground_truth already uses. Handlers that care about a payload's
    # shape are the ones that know it, not this type.

    event_type: EventType
    payload: Any
    timestamp: float
    event_id: str = field(default_factory=lambda: str(uuid4()))


Handler = Callable[[Event], None]


class EventBus:

    # An internal, in-process publish/subscribe system -- no
    # networking, no threads, no async: publish() dispatches to every
    # subscribed handler synchronously, in subscription order, on the
    # caller's own thread. This is what keeps the whole live_system
    # deterministic under a driven (not real-time) update loop: a test
    # that calls publish() directly sees every handler run to
    # completion before publish() returns, with no scheduling
    # nondeterminism to account for.

    def __init__(self):

        self._handlers: Dict[EventType, List[Handler]] = {}
        self._wildcard_handlers: List[Handler] = []

        # Every event ever published, in publish order -- an always-on
        # record a test (or a future replay/audit consumer) can inspect
        # without having to subscribe before the fact. Mirrors
        # HumanBehaviorLayer._decisions_so_far's "keep everything
        # resolved so far" convention, applied to events instead of
        # decisions.
        self._history: List[Event] = []

    # =====================================================

    def subscribe(self, event_type: EventType, handler: Handler) -> None:

        self._handlers.setdefault(event_type, []).append(handler)

    # =====================================================

    def subscribe_all(self, handler: Handler) -> None:

        # A wildcard subscriber, notified of every event regardless of
        # type -- the seam a Command Center notification bridge or a
        # test's own event recorder is meant to use, rather than
        # subscribing to each of the six EventType values individually.

        self._wildcard_handlers.append(handler)

    # =====================================================

    def unsubscribe(self, event_type: EventType, handler: Handler) -> None:

        handlers = self._handlers.get(event_type)

        if handlers is not None and handler in handlers:
            handlers.remove(handler)

    # =====================================================

    def unsubscribe_all(self, handler: Handler) -> None:

        if handler in self._wildcard_handlers:
            self._wildcard_handlers.remove(handler)

    # =====================================================

    def publish(self, event: Event) -> None:

        self._history.append(event)

        for handler in list(self._handlers.get(event.event_type, ())):
            handler(event)

        for handler in list(self._wildcard_handlers):
            handler(event)

    # =====================================================

    def emit(self, event_type: EventType, payload: Any, time: float) -> Event:

        # Convenience wrapper around publish() -- constructs the Event
        # itself so a caller (Orchestrator/UpdateLoop) never has to
        # import Event/uuid machinery just to publish one.

        event = Event(event_type=event_type, payload=payload, timestamp=time)
        self.publish(event)

        return event

    # =====================================================

    @property
    def history(self) -> List[Event]:

        return list(self._history)

    # =====================================================

    def history_of(self, event_type: EventType) -> List[Event]:

        return [event for event in self._history if event.event_type == event_type]
