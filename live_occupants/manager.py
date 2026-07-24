import dataclasses
from typing import Dict, Optional, Sequence, Set, Tuple

from behavior_recognition.observation import RecognizedBehavior

from perception.models.human_observation import HumanClassification, HumanState

from human_evidence.reconciliation import (
    HumanEvidenceConfig, apply_classification_staleness, apply_state_staleness,
    reconcile_classification, reconcile_state,
)

from live_system.event_bus import EventBus, EventType

from live_occupants import events
from live_occupants.events import (
    BehaviorChangedPayload, CameraChangedPayload, ClassificationUpdatedPayload,
    ConfirmedAssistanceRequiredPayload, OccupantCreatedPayload,
    OccupantExitedPayload, OccupantExpiredPayload, OccupantUpdatedPayload,
    PossibleAssistanceRequiredPayload, StateChangedPayload, ZoneChangedPayload,
)
from live_occupants.history import OccupantHistory
from live_occupants.lifecycle import (
    DEFAULT_EXIT_PROXIMITY_THRESHOLD, DEFAULT_EXPIRE_AFTER_SECONDS,
    is_expired, is_near_exit, next_status_on_missing, next_status_on_update,
)
from live_occupants.occupant import LiveOccupant
from live_occupants.state import OccupantStatus


# Live Human State & Assistance Perception Bridge milestone, Phase 6 --
# the ONE hard safety boundary this manager enforces mechanically:
# RecognizedBehavior.POSSIBLY_FALLEN (a hedged geometric heuristic) must
# NEVER be treated as, or silently promoted into, HumanState.FALLEN (a
# stronger, unhedged evidence claim) anywhere in this class. `behavior`
# and `human_state` remain two entirely separate fields/parameters,
# reconciled independently, with no code path here that ever reads one
# to set the other (see tests/test_human_evidence.py::
# PossiblyFallenSeparationTests).
_CONFIRMED_ASSISTANCE_STATES = frozenset({HumanState.FALLEN, HumanState.CRAWLING, HumanState.BEING_ASSISTED})


class LiveOccupantManager:

    # Live Occupant Digital Twin milestone, Phase 4 -- the single owner
    # of runtime LiveOccupant state, keyed by the SAME global occupant_id
    # cross_camera_identity.identity_registry.IdentityRegistry already
    # mints (this class never mints an occupant_id itself -- a caller,
    # the pipeline glue, always supplies one it already resolved
    # upstream). One instance is meant to be SHARED across every camera
    # in a deployment, exactly the same "one shared instance, not one
    # per camera" convention cross_camera_identity's own resolver
    # already established.
    #
    # O(1) lookup by occupant_id (a plain dict) -- Phase 4's own
    # explicit requirement. Secondary indices (by zone/floor/behavior/
    # camera) are maintained incrementally alongside the primary dict on
    # every store/remove, so queries never need a full scan.

    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        history_length: int = 30,
        exit_proximity_threshold: float = DEFAULT_EXIT_PROXIMITY_THRESHOLD,
        expire_after_seconds: float = DEFAULT_EXPIRE_AFTER_SECONDS,
        exits: Sequence[object] = (),
        human_evidence_config: Optional[HumanEvidenceConfig] = None,
    ):

        self.event_bus = event_bus
        self.history_length = history_length
        self.exit_proximity_threshold = exit_proximity_threshold
        self.expire_after_seconds = expire_after_seconds
        self.exits = tuple(exits)
        self.human_evidence_config = human_evidence_config if human_evidence_config is not None else HumanEvidenceConfig()

        self._occupants: Dict[str, LiveOccupant] = {}
        self._near_exit: Dict[str, bool] = {}

        self._by_zone: Dict[str, Set[str]] = {}
        self._by_floor: Dict[str, Set[str]] = {}
        self._by_behavior: Dict[RecognizedBehavior, Set[str]] = {}
        self._by_camera: Dict[str, Set[str]] = {}

    # =====================================================
    # Update (Phase 4/6/9)
    # =====================================================

    def update(
        self,
        occupant_id: str,
        camera_id: Optional[str],
        track_id: Optional[str],
        zone_id: Optional[str],
        floor_id: Optional[str],
        world_position: Optional[Tuple[float, float]],
        world_velocity: Optional[float],
        behavior: Optional[RecognizedBehavior],
        confidence: float,
        timestamp: float,
        *,
        classification_evidence: HumanClassification = HumanClassification.UNKNOWN,
        classification_confidence: Optional[float] = None,
        state_evidence: Optional[HumanState] = None,
        state_confidence: Optional[float] = None,
        world_position_provenance: Optional[str] = None,
    ) -> LiveOccupant:

        existing = self._occupants.get(occupant_id)

        near_exit = is_near_exit(world_position, self.exits, self.exit_proximity_threshold)

        if existing is None:

            history = OccupantHistory(max_length=self.history_length)
            history = history.with_camera_transition(timestamp, None, camera_id)
            history = history.with_zone_transition(timestamp, None, zone_id)

            if behavior is not None:
                history = history.with_behavior_change(timestamp, None, behavior)

            if world_position is not None:
                history = history.with_position_sample(timestamp, world_position)

            if world_velocity is not None:
                history = history.with_velocity_sample(timestamp, world_velocity)

            classification, classification_conf, classification_source, classification_seen_at = reconcile_classification(
                existing_classification=HumanClassification.UNKNOWN, existing_confidence=None,
                existing_source=None, existing_last_observed_at=None,
                new_classification=classification_evidence, new_confidence=classification_confidence,
                new_source=camera_id, timestamp=timestamp,
            )
            human_state, state_conf, state_source, state_seen_at = reconcile_state(
                existing_state=None, existing_confidence=None, existing_source=None, existing_last_observed_at=None,
                new_state=state_evidence, new_confidence=state_confidence, new_source=camera_id, timestamp=timestamp,
            )

            if classification != HumanClassification.UNKNOWN:
                history = history.with_classification_change(timestamp, HumanClassification.UNKNOWN, classification)

            if human_state is not None:
                history = history.with_state_change(timestamp, None, human_state)

            occupant = LiveOccupant(
                occupant_id=occupant_id, current_camera_id=camera_id, current_track_id=track_id,
                current_zone_id=zone_id, current_floor_id=floor_id, world_position=world_position,
                world_velocity=world_velocity, behavior=behavior, confidence=confidence,
                first_seen=timestamp, last_seen=timestamp, status=OccupantStatus.NEW, history=history,
                human_classification=classification, human_classification_confidence=classification_conf,
                human_classification_source=classification_source,
                human_classification_last_observed_at=classification_seen_at,
                human_state=human_state, human_state_confidence=state_conf,
                human_state_source=state_source, human_state_last_observed_at=state_seen_at,
                world_position_provenance=world_position_provenance,
            )

            self._store(occupant, near_exit)
            events.publish(self.event_bus, EventType.OCCUPANT_CREATED, OccupantCreatedPayload(occupant), timestamp)

            if behavior == RecognizedBehavior.POSSIBLY_FALLEN:
                events.publish(
                    self.event_bus, EventType.POSSIBLE_ASSISTANCE_REQUIRED,
                    PossibleAssistanceRequiredPayload(occupant_id, zone_id), timestamp,
                )

            if human_state in _CONFIRMED_ASSISTANCE_STATES:
                events.publish(
                    self.event_bus, EventType.CONFIRMED_ASSISTANCE_REQUIRED,
                    ConfirmedAssistanceRequiredPayload(occupant_id, zone_id, human_state), timestamp,
                )

            return occupant

        history = existing.history

        if camera_id != existing.current_camera_id:
            history = history.with_camera_transition(timestamp, existing.current_camera_id, camera_id)

        if zone_id != existing.current_zone_id:
            history = history.with_zone_transition(timestamp, existing.current_zone_id, zone_id)

        if behavior != existing.behavior:
            history = history.with_behavior_change(timestamp, existing.behavior, behavior)

        if world_position is not None:
            history = history.with_position_sample(timestamp, world_position)

        if world_velocity is not None:
            history = history.with_velocity_sample(timestamp, world_velocity)

        # Live Human State & Assistance Perception Bridge milestone --
        # staleness is applied to the EXISTING, stored evidence first
        # (decaying it back to UNKNOWN/None if it has genuinely gone too
        # long without reconfirmation, Phase 8), and only THEN is this
        # cycle's new reading reconciled against whatever remains --
        # never the other way around (which would let a stale value
        # outrank fresh evidence).
        decayed_classification, decayed_classification_conf, decayed_classification_source, decayed_classification_seen_at = (
            apply_classification_staleness(
                existing.human_classification, existing.human_classification_confidence,
                existing.human_classification_source, existing.human_classification_last_observed_at,
                now=timestamp, config=self.human_evidence_config,
            )
        )
        decayed_state, decayed_state_conf, decayed_state_source, decayed_state_seen_at = apply_state_staleness(
            existing.human_state, existing.human_state_confidence, existing.human_state_source,
            existing.human_state_last_observed_at, now=timestamp, config=self.human_evidence_config,
        )

        classification, classification_conf, classification_source, classification_seen_at = reconcile_classification(
            existing_classification=decayed_classification, existing_confidence=decayed_classification_conf,
            existing_source=decayed_classification_source, existing_last_observed_at=decayed_classification_seen_at,
            new_classification=classification_evidence, new_confidence=classification_confidence,
            new_source=camera_id, timestamp=timestamp,
        )
        human_state, state_conf, state_source, state_seen_at = reconcile_state(
            existing_state=decayed_state, existing_confidence=decayed_state_conf,
            existing_source=decayed_state_source, existing_last_observed_at=decayed_state_seen_at,
            new_state=state_evidence, new_confidence=state_confidence, new_source=camera_id, timestamp=timestamp,
        )

        if classification != existing.human_classification:
            history = history.with_classification_change(timestamp, existing.human_classification, classification)

        if human_state != existing.human_state:
            history = history.with_state_change(timestamp, existing.human_state, human_state)

        updated = dataclasses.replace(
            existing,
            current_camera_id=camera_id, current_track_id=track_id, current_zone_id=zone_id,
            current_floor_id=floor_id, world_position=world_position, world_velocity=world_velocity,
            behavior=behavior, confidence=confidence, last_seen=timestamp,
            status=next_status_on_update(existing.status), history=history,
            human_classification=classification, human_classification_confidence=classification_conf,
            human_classification_source=classification_source,
            human_classification_last_observed_at=classification_seen_at,
            human_state=human_state, human_state_confidence=state_conf,
            human_state_source=state_source, human_state_last_observed_at=state_seen_at,
            world_position_provenance=world_position_provenance,
        )

        self._store(updated, near_exit)

        # Event ordering (Phase 9/10): OCCUPANT_UPDATED always fires
        # first (it is the general-purpose "something about this
        # occupant changed this cycle" signal every consumer can
        # subscribe to alone), followed by any more specific field-
        # change events, camera/zone/behavior, in that fixed order --
        # deterministic regardless of which fields actually changed.
        events.publish(self.event_bus, EventType.OCCUPANT_UPDATED, OccupantUpdatedPayload(updated), timestamp)

        if camera_id != existing.current_camera_id:
            events.publish(
                self.event_bus, EventType.OCCUPANT_CAMERA_CHANGED,
                CameraChangedPayload(occupant_id, existing.current_camera_id, camera_id), timestamp,
            )

        if zone_id != existing.current_zone_id:
            events.publish(
                self.event_bus, EventType.OCCUPANT_ZONE_CHANGED,
                ZoneChangedPayload(occupant_id, existing.current_zone_id, zone_id), timestamp,
            )

        if behavior != existing.behavior:
            events.publish(
                self.event_bus, EventType.OCCUPANT_BEHAVIOR_CHANGED,
                BehaviorChangedPayload(occupant_id, existing.behavior, behavior), timestamp,
            )

        # Live Human State & Assistance Perception Bridge milestone --
        # Phase 11's own "only add events justified by actual
        # transitions... no repeated event spam every cycle": every
        # event below fires on a GENUINE change only (never every cycle
        # the same reconciled value happens to still hold).
        if classification != existing.human_classification:
            events.publish(
                self.event_bus, EventType.OCCUPANT_CLASSIFICATION_UPDATED,
                ClassificationUpdatedPayload(occupant_id, existing.human_classification, classification), timestamp,
            )

        if human_state != existing.human_state:
            events.publish(
                self.event_bus, EventType.OCCUPANT_STATE_CHANGED,
                StateChangedPayload(occupant_id, existing.human_state, human_state), timestamp,
            )

        if behavior == RecognizedBehavior.POSSIBLY_FALLEN and existing.behavior != RecognizedBehavior.POSSIBLY_FALLEN:
            events.publish(
                self.event_bus, EventType.POSSIBLE_ASSISTANCE_REQUIRED,
                PossibleAssistanceRequiredPayload(occupant_id, zone_id), timestamp,
            )

        if human_state in _CONFIRMED_ASSISTANCE_STATES and existing.human_state not in _CONFIRMED_ASSISTANCE_STATES:
            events.publish(
                self.event_bus, EventType.CONFIRMED_ASSISTANCE_REQUIRED,
                ConfirmedAssistanceRequiredPayload(occupant_id, zone_id, human_state), timestamp,
            )

        return updated

    # =====================================================
    # Missing / expiry sweep (Phase 6/8) -- called ONCE per overall
    # pipeline cycle (see live_camera_pipeline.pipeline.LiveCameraPipeline.
    # run_cycle()), after every camera this cycle has already called
    # update() for whoever it actually saw. Comparing the full active
    # occupant set against `seen_occupant_ids` is what lets this class
    # detect "missing" without needing to know anything about
    # tracking.tracker.SingleCameraTracker's own MISSING/EXPIRED states,
    # or cross_camera_identity's own registry internals, directly.
    # =====================================================

    def sweep_missing(self, timestamp: float, seen_occupant_ids: Set[str]) -> None:

        for occupant_id in list(self._occupants.keys()):

            if occupant_id in seen_occupant_ids:
                continue

            occupant = self._occupants[occupant_id]

            if is_expired(occupant.last_seen, timestamp, self.expire_after_seconds):
                self._expire(occupant_id, timestamp)
                continue

            near_exit = self._near_exit.get(occupant_id, False)
            new_status = next_status_on_missing(occupant.status, near_exit)

            # Live Human State & Assistance Perception Bridge milestone,
            # Phase 8 -- staleness is checked EVERY cycle an occupant is
            # missing, not only on their next actual sighting: real wall-
            # clock time keeps passing for someone not currently
            # observed by any camera, and stale evidence must still
            # honestly expire rather than being frozen at whatever it
            # last read.
            classification, classification_conf, classification_source, classification_seen_at = (
                apply_classification_staleness(
                    occupant.human_classification, occupant.human_classification_confidence,
                    occupant.human_classification_source, occupant.human_classification_last_observed_at,
                    now=timestamp, config=self.human_evidence_config,
                )
            )
            human_state, state_conf, state_source, state_seen_at = apply_state_staleness(
                occupant.human_state, occupant.human_state_confidence, occupant.human_state_source,
                occupant.human_state_last_observed_at, now=timestamp, config=self.human_evidence_config,
            )

            classification_changed = classification != occupant.human_classification
            state_changed = human_state != occupant.human_state

            if new_status == occupant.status and not classification_changed and not state_changed:
                continue

            history = occupant.history

            if classification_changed:
                history = history.with_classification_change(timestamp, occupant.human_classification, classification)

            if state_changed:
                history = history.with_state_change(timestamp, occupant.human_state, human_state)

            updated = dataclasses.replace(
                occupant, status=new_status, history=history,
                human_classification=classification, human_classification_confidence=classification_conf,
                human_classification_source=classification_source,
                human_classification_last_observed_at=classification_seen_at,
                human_state=human_state, human_state_confidence=state_conf,
                human_state_source=state_source, human_state_last_observed_at=state_seen_at,
            )
            self._store(updated, near_exit)

            if new_status == OccupantStatus.EXITED:
                events.publish(self.event_bus, EventType.OCCUPANT_EXITED, OccupantExitedPayload(updated), timestamp)

            if classification_changed:
                events.publish(
                    self.event_bus, EventType.OCCUPANT_CLASSIFICATION_UPDATED,
                    ClassificationUpdatedPayload(occupant_id, occupant.human_classification, classification), timestamp,
                )

            if state_changed:
                events.publish(
                    self.event_bus, EventType.OCCUPANT_STATE_CHANGED,
                    StateChangedPayload(occupant_id, occupant.human_state, human_state), timestamp,
                )

    # =====================================================

    def _expire(self, occupant_id: str, timestamp: float) -> None:

        occupant = self._occupants.get(occupant_id)

        if occupant is None:
            return

        expired = dataclasses.replace(occupant, status=OccupantStatus.EXPIRED)

        events.publish(self.event_bus, EventType.OCCUPANT_EXPIRED, OccupantExpiredPayload(expired), timestamp)

        self._remove(occupant_id)

    # =====================================================
    # Storage + secondary indices
    # =====================================================

    def _store(self, occupant: LiveOccupant, near_exit: bool) -> None:

        previous = self._occupants.get(occupant.occupant_id)

        if previous is not None:
            self._unindex(previous)

        self._occupants[occupant.occupant_id] = occupant
        self._near_exit[occupant.occupant_id] = near_exit

        self._index(occupant)

    # =====================================================

    def _remove(self, occupant_id: str) -> None:

        occupant = self._occupants.pop(occupant_id, None)
        self._near_exit.pop(occupant_id, None)

        if occupant is not None:
            self._unindex(occupant)

    # =====================================================

    def _index(self, occupant: LiveOccupant) -> None:

        if occupant.current_zone_id is not None:
            self._by_zone.setdefault(occupant.current_zone_id, set()).add(occupant.occupant_id)

        if occupant.current_floor_id is not None:
            self._by_floor.setdefault(occupant.current_floor_id, set()).add(occupant.occupant_id)

        if occupant.behavior is not None:
            self._by_behavior.setdefault(occupant.behavior, set()).add(occupant.occupant_id)

        if occupant.current_camera_id is not None:
            self._by_camera.setdefault(occupant.current_camera_id, set()).add(occupant.occupant_id)

    # =====================================================

    def _unindex(self, occupant: LiveOccupant) -> None:

        for index, key in (
            (self._by_zone, occupant.current_zone_id),
            (self._by_floor, occupant.current_floor_id),
            (self._by_behavior, occupant.behavior),
            (self._by_camera, occupant.current_camera_id),
        ):

            if key is None:
                continue

            members = index.get(key)

            if members is None:
                continue

            members.discard(occupant.occupant_id)

            if not members:
                del index[key]

    # =====================================================
    # Queries (Phase 4)
    # =====================================================

    def get(self, occupant_id: str) -> Optional[LiveOccupant]:

        return self._occupants.get(occupant_id)

    # =====================================================

    def active_occupants(self) -> Tuple[LiveOccupant, ...]:

        return tuple(
            occupant for occupant in self._occupants.values()
            if occupant.status in (OccupantStatus.NEW, OccupantStatus.ACTIVE)
        )

    # =====================================================

    def all_occupants(self) -> Tuple[LiveOccupant, ...]:

        return tuple(self._occupants.values())

    # =====================================================

    def occupants_in_zone(self, zone_id: str) -> Tuple[LiveOccupant, ...]:

        return tuple(self._occupants[occupant_id] for occupant_id in self._by_zone.get(zone_id, ()))

    # =====================================================

    def occupants_on_floor(self, floor_id: str) -> Tuple[LiveOccupant, ...]:

        return tuple(self._occupants[occupant_id] for occupant_id in self._by_floor.get(floor_id, ()))

    # =====================================================

    def occupants_with_behavior(self, behavior: RecognizedBehavior) -> Tuple[LiveOccupant, ...]:

        return tuple(self._occupants[occupant_id] for occupant_id in self._by_behavior.get(behavior, ()))

    # =====================================================

    def occupants_on_camera(self, camera_id: str) -> Tuple[LiveOccupant, ...]:

        return tuple(self._occupants[occupant_id] for occupant_id in self._by_camera.get(camera_id, ()))

    # =====================================================

    def __len__(self) -> int:

        return len(self._occupants)
