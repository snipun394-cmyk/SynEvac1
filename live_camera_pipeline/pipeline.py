import dataclasses
from typing import Mapping, Optional

from perception.models.human_observation import HumanClassification, HumanState

from live_camera_pipeline.detection_provider import LiveCameraPipelineDetectionProvider
from live_camera_pipeline.frame_source import CameraFrameSource
from live_camera_pipeline.human_detector import HumanDetector
from live_camera_pipeline.identity_resolver import IdentityResolver

from tracking.tracker import SingleCameraTracker

from behavior_recognition.observation import RecognizedBehavior
from behavior_recognition.recognizer import BehaviorRecognizer

from cross_camera_identity.resolver import CrossCameraIdentityResolver

from camera_calibration.projection import WorldProjector

from live_occupants.manager import LiveOccupantManager


def _map_behavior_to_human_state(behavior: RecognizedBehavior) -> Optional[HumanState]:

    # Human Behavior Recognition Framework milestone, Phase 8 -- the
    # ONLY two RecognizedBehavior values honestly equivalent to an
    # EXISTING HumanState member: WALKING and RUNNING both describe the
    # same plain, observable "currently in motion at roughly this pace"
    # fact a rule-based velocity threshold can honestly assert, and
    # HumanState.WALKING/RUNNING carry no stronger claim than that.
    #
    # STATIONARY and UNKNOWN deliberately map to None (RawHumanDetection.
    # state_evidence's own "no evidence" convention) rather than any of
    # HumanState.STANDING/WAITING/NEVER_MOVING_YET -- none of those is a
    # genuinely equivalent claim to "velocity measured near zero" (each
    # implies something about WHY the person stopped that pure geometry
    # cannot honestly assert).
    #
    # POSSIBLY_FALLEN deliberately NEVER maps to HumanState.FALLEN --
    # command_center.building_view/incident_data already treat
    # HumanState.FALLEN as a confident, operator-facing signal (a
    # "fallen_count" tooltip); remapping a hedged, disabled-by-default
    # geometric heuristic onto that same value would misrepresent a
    # guess as a confirmed observation. See docs/architecture/
    # behavior_recognition.md Sec 4 for the full reasoning.

    if behavior == RecognizedBehavior.WALKING:
        return HumanState.WALKING

    if behavior == RecognizedBehavior.RUNNING:
        return HumanState.RUNNING

    return None


class LiveCameraPipeline:

    # Phase 12's orchestrator seam: wires
    # CameraFrameSource -> HumanDetector -> IdentityResolver ->
    # LiveCameraPipelineDetectionProvider, entirely via dependency
    # injection, so this class runs today with fake/in-memory
    # implementations (see tests/test_live_camera_pipeline.py) with
    # zero real camera. It deliberately stops there -- it never calls
    # CameraManager, MultiCameraFusionEngine, or BuildingStateEstimator
    # itself. A caller registers the detection_provider it was built
    # with into CameraManager (register_detection_provider(DeviceMode.
    # LIVE, provider)) and continues with the exact same
    # CameraManager -> MultiCameraFusionEngine -> BuildingStateEstimator
    # chain Simulation already uses -- "everything downstream remains
    # unchanged" (Phase 12's own requirement).
    #
    # When real CCTV access exists, only frame_sources' values change
    # (SimulationFrameSource/fakes -> RTSPFrameSource) and human_detector
    # changes (a fake -> real YOLO+tracker). Nothing about this class,
    # or anything it is wired into, needs to change.
    #
    # Single-Camera Tracking Framework milestone, Phase 7: `tracker` is
    # an OPTIONAL, additive seam between human_detector.detect() and
    # identity_resolver.resolve() -- omitting it (the default,
    # unchanged behavior for every existing caller/test) reproduces
    # this class's exact pre-tracking behavior. When supplied, each
    # frame's raw detections are run through tracker.update(camera_id,
    # frame.timestamp, raw) (tracking/tracker.SingleCameraTracker),
    # and only its own per-detection positional output (never the
    # trailing MISSING/EXPIRED remainder -- there is no real
    # observation to hand IdentityResolver for a track not seen this
    # cycle) is used to replace each RawHumanDetection's own
    # per-frame-only local_track_id with the tracker's now
    # frame-to-frame-STABLE track_id. IdentityResolver.resolve() itself
    # is never modified -- it still receives plain RawHumanDetection
    # objects, just ones whose local_track_id is more stable input
    # than an unstabilized detector could honestly provide alone
    # (Phase 8's identity boundary: a tracking id is still local and
    # temporary, never a global occupant_id -- that remains entirely
    # IdentityResolver's own job, untouched here).

    # Human Behavior Recognition Framework milestone, Phase 8:
    # `behavior_recognizer` is a further OPTIONAL, additive seam, only
    # ever consulted when `tracker` is also supplied (behavior
    # recognition inherently needs tracking history -- supplying
    # behavior_recognizer without a tracker is simply a no-op). When
    # both are supplied, each cycle's FULL tracker output (including
    # MISSING/EXPIRED entries -- see behavior_recognition.recognizer.
    # BehaviorRecognizer's own contract) is handed to behavior_
    # recognizer.recognize(), and the resulting BehaviorObservation for
    # each currently-observed person additionally sets that detection's
    # state_evidence via _map_behavior_to_human_state() above --
    # IdentityResolver.resolve() again receives only plain
    # RawHumanDetection objects, never a BehaviorObservation directly,
    # keeping IdentityResolver's own global-identity responsibility
    # completely untouched (Phase 8's own "without changing global
    # identity responsibilities" requirement).

    # Cross-Camera Identity Resolution (ReID Framework) milestone,
    # Phase 7: `cross_camera_identity_resolver` is a further OPTIONAL,
    # additive seam, only ever consulted when `tracker` is also
    # supplied (cross-camera resolution inherently needs tracking
    # output -- supplying it without a tracker is simply a no-op, same
    # constraint as behavior_recognizer). When supplied, each cycle's
    # FULL tracker output is handed to cross_camera_identity_resolver.
    # resolve() (cross_camera_identity.resolver.
    # CrossCameraIdentityResolver's own NEW/TRACKED-prefix contract),
    # and the resulting GLOBAL occupant id REPLACES the tracker's own
    # local_track_id in the RawHumanDetection handed to
    # IdentityResolver.resolve() -- IdentityResolver itself is never
    # modified. IMPORTANT: pair this with an identity_resolver that
    # treats local_track_id as an already-resolved identity (e.g.
    # live_camera_pipeline.identity_resolver.SimulationIdentityResolver,
    # whose own docstring already documents exactly this "local_track_id
    # IS already the global identity" strategy) -- NOT
    # MappingIdentityResolver, which would re-namespace an already-
    # global id by camera_id and defeat cross-camera unification (see
    # docs/architecture/cross_camera_identity.md Sec 5).

    # Camera Calibration & World Coordinate Projection milestone, Phase
    # 6/7: `world_projector` is a further OPTIONAL, additive seam, only
    # ever consulted when `tracker` is also supplied (world projection
    # needs a TrackedHuman's own bounding_box). Desired order per this
    # milestone: Tracker -> WorldProjection -> BehaviorRecognizer ->
    # CrossCameraIdentity -- each matched TrackedHuman's bounding_box is
    # projected FIRST (camera_calibration.projection.WorldProjector.
    # project()), and only successfully-projected world positions (a
    # camera with no calibration, or a geometrically undefined
    # projection, contributes nothing) are passed into behavior_
    # recognizer.recognize()'s own world_positions_by_track_id
    # parameter -- "operate using world-space motion instead of
    # pixel-space whenever calibration is available, gracefully fall
    # back to image-space if calibration is unavailable" (Phase 6) is
    # therefore satisfied automatically by BehaviorRecognizer's own
    # documented fallback, not by any special-casing here.
    #
    # The projection's floor_id/zone_id/world_position/
    # projection_confidence, and (once behavior recognition has run)
    # world_velocity, are all set on the final RawHumanDetection via
    # dataclasses.replace() -- IdentityResolver.resolve() still only
    # ever receives plain RawHumanDetection objects (Phase 7's "augment
    # Detection with optional world-space information" reaches
    # Detection through live_camera_pipeline.identity_resolver.
    # _to_detection(), never by modifying IdentityResolver itself).

    # Live Occupant Digital Twin milestone, Phase 7 (refined by Live
    # Perception -> BuildingState Integration Bridge milestone, Phase 8
    # -- see below): `live_occupant_manager` is a further OPTIONAL,
    # additive seam, only ever consulted when `tracker` is also
    # supplied. It never changes what RawHumanDetection/Detection/
    # BuildingState themselves contain (Phase 7's own "Detection
    # remains immutable... do not redesign downstream systems").
    #
    # CORRECTED in the Live Perception -> BuildingState Integration
    # Bridge milestone: LiveOccupantManager.update() is now called
    # AFTER identity_resolver.resolve() runs, keyed by the FINAL
    # Detection.occupant_id -- NOT the tracker's own per-camera-local
    # track_id, and not only the cross_camera_identity_resolver's own
    # global id. The earlier version keyed it by local_track_id
    # (replaced by cross_camera_identity_resolver's global id ONLY when
    # that seam was configured), which silently DISAGREED with
    # BuildingState.occupant_tracks whenever a deployment resolved
    # cross-camera identity a different way -- e.g. via
    # MappingIdentityResolver (the existing, already-proven mechanism
    # for reconciling two cameras that see the SAME person
    # SIMULTANEOUSLY, which cross_camera_identity_resolver's own
    # sequential departure/arrival matcher cannot do -- see
    # docs/architecture/live_perception_building_state_integration.md
    # Sec 8). Proven directly in tests/test_live_perception_double_
    # counting.py: 2 cameras, 3 physical occupants, 4 raw detections ->
    # exactly 3 in BuildingState.occupant_tracks, LiveOccupantManager,
    # and BuildingState.zone_occupancy alike -- never 4, 6, or 7.
    #
    # _process_camera_cycle() below no longer calls live_occupant_
    # manager.update() itself -- it returns a `pending_updates` list,
    # positionally aligned with its own `raw` return value (one entry
    # per detection, None wherever live_occupant_manager is not
    # configured), carrying exactly the fields Detection itself cannot
    # (RecognizedBehavior, camera_id, tracker-local track_id,
    # confidence -- Detection only ever gets the LOSSY HumanState
    # mapping, never the rich RecognizedBehavior enum). run_cycle()
    # zips this list against identity_resolver.resolve()'s own output
    # (relying on the same one-Detection-per-RawHumanDetection,
    # same-order guarantee every existing IdentityResolver
    # implementation already provides) to call live_occupant_manager.
    # update() with the FINAL occupant_id, alongside Detection's own
    # already-correct zone_id/floor_id/position/world_velocity (passed
    # through unchanged from what this method already computed).
    # LiveOccupantManager.sweep_missing() still runs exactly ONCE per
    # overall cycle, now driven by the FINAL resolved occupant_ids.

    def __init__(
        self,
        frame_sources: Mapping[str, CameraFrameSource],
        human_detector: HumanDetector,
        identity_resolver: IdentityResolver,
        detection_provider: LiveCameraPipelineDetectionProvider,
        tracker: Optional[SingleCameraTracker] = None,
        behavior_recognizer: Optional[BehaviorRecognizer] = None,
        cross_camera_identity_resolver: Optional[CrossCameraIdentityResolver] = None,
        world_projector: Optional[WorldProjector] = None,
        live_occupant_manager: Optional[LiveOccupantManager] = None,
    ):

        self.frame_sources = dict(frame_sources)
        self.human_detector = human_detector
        self.identity_resolver = identity_resolver
        self.detection_provider = detection_provider
        self.tracker = tracker
        self.behavior_recognizer = behavior_recognizer
        self.cross_camera_identity_resolver = cross_camera_identity_resolver
        self.world_projector = world_projector
        self.live_occupant_manager = live_occupant_manager

    # =====================================================

    def run_cycle(self, time: float) -> None:

        raw_detections = []
        pending_occupant_updates = []

        for camera_id, frame_source in self.frame_sources.items():

            frame = frame_source.read_frame()

            if frame is None:
                continue

            raw = self.human_detector.detect(frame)

            if self.tracker is not None:
                raw, camera_pending_updates = self._process_camera_cycle(camera_id, frame.timestamp, raw)
            else:
                camera_pending_updates = [None] * len(raw)

            raw_detections.extend(raw)
            pending_occupant_updates.extend(camera_pending_updates)

        resolved = self.identity_resolver.resolve(raw_detections, time)

        if self.live_occupant_manager is not None:

            seen_occupant_ids = set()

            # Relies on identity_resolver.resolve() returning exactly
            # one Detection per RawHumanDetection, in the same order --
            # true of every existing IdentityResolver implementation
            # (SimulationIdentityResolver/MappingIdentityResolver both
            # build their result via a single `for raw in
            # raw_detections` comprehension) and the only way to
            # correlate a resolved Detection back to the rich,
            # pre-resolution data pending_occupant_updates carries.
            for detection, pending in zip(resolved, pending_occupant_updates):

                if pending is None:
                    continue

                self.live_occupant_manager.update(
                    detection.occupant_id, pending["camera_id"], pending["track_id"],
                    detection.zone_id, detection.floor_id, detection.position, detection.world_velocity,
                    pending["behavior"], pending["confidence"], time,
                    classification_evidence=pending["classification_evidence"],
                    classification_confidence=pending["classification_confidence"],
                    state_evidence=pending["state_evidence"],
                    state_confidence=pending["state_confidence"],
                    world_position_provenance=detection.world_position_provenance,
                    stair_id=detection.stair_id,
                )
                seen_occupant_ids.add(detection.occupant_id)

            self.live_occupant_manager.sweep_missing(time, seen_occupant_ids)

        self.detection_provider.publish(self.frame_sources.keys(), resolved)

    # =====================================================

    def _process_camera_cycle(self, camera_id, timestamp, raw):

        tracked = self.tracker.update(camera_id, timestamp, raw)
        matched = tracked[:len(raw)]

        projections_by_track_id = {}
        if self.world_projector is not None:

            for tracked_human in matched:
                projections_by_track_id[tracked_human.track_id] = self.world_projector.project(
                    camera_id, tracked_human.bounding_box, tracked_human.confidence,
                )

        world_positions_by_track_id = {
            track_id: projection.world_position
            for track_id, projection in projections_by_track_id.items()
            if projection.world_position is not None
        }

        observations = ()
        if self.behavior_recognizer is not None:
            observations = self.behavior_recognizer.recognize(
                camera_id, timestamp, tracked, world_positions_by_track_id or None,
            )

        behaviors_by_track_id = {observation.track_id: observation for observation in observations}

        global_id_by_track_id = None
        if self.cross_camera_identity_resolver is not None:

            resolved_identities = self.cross_camera_identity_resolver.resolve(
                camera_id, timestamp, tracked, behaviors_by_track_id,
            )
            global_id_by_track_id = {
                identity.track_id: identity.global_id for identity in resolved_identities
            }

        results = []
        pending_updates = []

        for detection, tracked_human in zip(raw, matched):

            local_track_id = tracked_human.track_id
            if global_id_by_track_id is not None:
                local_track_id = global_id_by_track_id.get(tracked_human.track_id, tracked_human.track_id)

            behavior_observation = behaviors_by_track_id.get(tracked_human.track_id)
            recognized_behavior = behavior_observation.recognized_behavior if behavior_observation is not None else None

            # Live Human State & Assistance Perception Bridge milestone,
            # Phase 1/15 -- a genuine bug fix: `detection.state_evidence`
            # (whatever the DETECTOR itself already supplied -- e.g. a
            # future fall-detection-capable HumanDetector genuinely
            # reporting FALLEN/CRAWLING) is AUTHORITATIVE and must never
            # be silently discarded. Previously this method unconditionally
            # overwrote it with the behavior recognizer's own coarse
            # WALKING/RUNNING-or-None mapping below, discarding any
            # richer evidence a real detector might already have
            # supplied -- the behavior-derived value now only fills in
            # when the detector itself reported no state evidence at all.
            state_evidence = detection.state_evidence
            if state_evidence is None and recognized_behavior is not None:
                state_evidence = _map_behavior_to_human_state(recognized_behavior)

            world_velocity = (
                behavior_observation.world_metrics.world_velocity
                if behavior_observation is not None and behavior_observation.world_metrics is not None
                else None
            )

            projection = projections_by_track_id.get(tracked_human.track_id)
            floor_id = projection.floor_id if projection is not None else detection.floor_id
            zone_id = projection.zone_id if projection is not None else detection.zone_id
            world_position = projection.world_position if projection is not None else None
            projection_confidence = projection.projection_confidence if projection is not None else None
            world_position_provenance = projection.provenance if projection is not None else None
            stair_id = projection.stair_id if projection is not None else None
            stair_localization_ambiguous = projection.stair_localization_ambiguous if projection is not None else False

            results.append(
                dataclasses.replace(
                    detection,
                    local_track_id=local_track_id,
                    state_evidence=state_evidence,
                    floor_id=floor_id,
                    zone_id=zone_id,
                    world_position=world_position,
                    world_velocity=world_velocity,
                    projection_confidence=projection_confidence,
                    world_position_provenance=world_position_provenance,
                    stair_id=stair_id,
                    stair_localization_ambiguous=stair_localization_ambiguous,
                )
            )

            if self.live_occupant_manager is not None:

                pending_updates.append({
                    "camera_id": camera_id,
                    "track_id": tracked_human.track_id,
                    "behavior": recognized_behavior,
                    "confidence": tracked_human.confidence,
                    # Live Human State & Assistance Perception Bridge
                    # milestone -- reuses the SAME "detector's own
                    # presence confidence" value _to_detection() already
                    # falls back to for classification (Detection has no
                    # separate classification-specific/state-specific
                    # confidence signal anywhere in this codebase, see
                    # human_evidence.models.HumanEvidence's own
                    # docstring), never a fabricated new number.
                    "classification_evidence": detection.classification_evidence or HumanClassification.UNKNOWN,
                    "classification_confidence": detection.confidence,
                    "state_evidence": state_evidence,
                    "state_confidence": detection.confidence,
                })

            else:

                pending_updates.append(None)

        return tuple(results), pending_updates
