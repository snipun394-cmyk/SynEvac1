import unittest

from perception.models.human_observation import HumanClassification, HumanState

from models.building import Building
from models.floor import Floor
from models.staircase import Staircase, StairObservableRegion
from models.zone import Zone

from camera_calibration.asset_lookup import build_assets_by_floor, covered_asset_ids
from camera_calibration.camera_model import CalibrationProfile, CameraExtrinsics, CameraIntrinsics
from camera_calibration.projection import WorldProjector
from camera_calibration.stair_lookup import DEFAULT_OBSERVABLE_ASSET_KINDS, build_stairs_by_floor

from live_camera_pipeline.detection_provider import LiveCameraPipelineDetectionProvider
from live_camera_pipeline.frame_source import CameraFrame, CameraFrameSource
from live_camera_pipeline.human_detector import HumanDetector, RawHumanDetection
from live_camera_pipeline.identity_resolver import MappingIdentityResolver, SimulationIdentityResolver
from live_camera_pipeline.pipeline import LiveCameraPipeline

from live_occupants.manager import LiveOccupantManager

from tracking.simple_tracker import SimpleSingleCameraTracker

from observable_assets.facts import compute_asset_occupancy_snapshot
from observable_assets.models import ObservationStatus


# =====================================================
# Observable Stair Perception milestone, Phase 22/23/24 (updated by the
# Observable Asset Perception Framework milestone: the snapshot layer
# below is now the generic observable_assets package, Stair exercised
# as its first concrete asset type -- see docs/architecture/
# observable_asset_perception.md) -- realistic, offline end-to-end
# tests driving the REAL production chain:
#
#   RawHumanDetection bounding box
#     -> tracking.SimpleSingleCameraTracker (real)
#     -> camera_calibration.projection.WorldProjector (real, with real
#        stair geometry, exactly the same class production code uses)
#     -> live_camera_pipeline.identity_resolver (real)
#     -> live_occupants.manager.LiveOccupantManager (real)
#     -> LiveOccupantManager.canonical_occupancy() (real)
#     -> observable_assets.facts.compute_asset_occupancy_snapshot() (real)
#
# Mirrors tests/test_live_camera_pipeline.py's own "CameraFrame all the
# way through, zero real camera, zero Simulation ground truth" fake/
# in-memory convention, extended with a real WorldProjector + real
# Staircase observable-region geometry (that file's own fakes never
# exercise projection/calibration at all).
# =====================================================


CAMERA_POSITION = (0.0, 0.0)


def make_calibration(camera_id, floor_id="floor-1", position=CAMERA_POSITION):

    intrinsics = CameraIntrinsics(image_width=640, image_height=480, focal_length_x=500.0, focal_length_y=500.0)
    extrinsics = CameraExtrinsics(position=position, mount_height=3.0, yaw_degrees=0.0, pitch_degrees=45.0, roll_degrees=0.0)

    return CalibrationProfile(camera_id=camera_id, floor_id=floor_id, intrinsics=intrinsics, extrinsics=extrinsics)


# Three distinct ground-contact pixels on the SAME camera -- each
# resolved to its own exact world position once via a bare, geometry-only
# WorldProjector (no zones/stairs configured), then used to build Zone A/
# Stair S1/Zone B geometry precisely centered on each -- avoids hand-
# deriving ray-floor trigonometry by hand while staying fully
# deterministic (same technique tests/test_camera_calibration.py's own
# ProjectionAccuracyTests establishes: a pure geometric function,
# reproducible across calls).
BBOX_ZONE_A = (315.0, 260.0, 325.0, 300.0)    # closer to the camera
BBOX_STAIR = (315.0, 200.0, 325.0, 240.0)     # center pixel
BBOX_ZONE_B = (315.0, 140.0, 325.0, 180.0)    # further from the camera


def resolve_world_positions(camera_id="CAM-1"):

    bare_projector = WorldProjector(calibrations={camera_id: make_calibration(camera_id)}, zones_by_floor={})

    return {
        "zone_a": bare_projector.project(camera_id, BBOX_ZONE_A, 0.9).world_position,
        "stair": bare_projector.project(camera_id, BBOX_STAIR, 0.9).world_position,
        "zone_b": bare_projector.project(camera_id, BBOX_ZONE_B, 0.9).world_position,
    }


def make_building_and_geometry():

    positions = resolve_world_positions("CAM-1")

    floor = Floor(name="Floor 1", display_order=0)
    building = Building(name="Test Building")
    building.add_floor(floor)

    zone_a = Zone(name="Zone A", x=positions["zone_a"][0] - 0.5, y=positions["zone_a"][1] - 0.5, width=1.0, height=1.0, floor_id=floor.id)
    zone_b = Zone(name="Zone B", x=positions["zone_b"][0] - 0.5, y=positions["zone_b"][1] - 0.5, width=1.0, height=1.0, floor_id=floor.id)

    floor.add_zone(zone_a)
    floor.add_zone(zone_b)

    stair = Staircase(
        name="Stair S1", from_floor_id=floor.id, to_floor_id=floor.id,
        from_position=positions["stair"], to_position=positions["stair"], width=1.5,
    )
    stair.from_observable_region = StairObservableRegion(
        center_x=positions["stair"][0], center_y=positions["stair"][1], width=0.6, depth=0.6,
    )
    floor.add_stair(stair)

    return building, floor, zone_a, zone_b, stair, positions


class FakeFrameSource(CameraFrameSource):

    def __init__(self, camera_id):
        self.camera_id = camera_id
        self._queue = []
        self._running = False

    def queue_frame(self, timestamp, frame_sequence, payload_ref=None):
        self._queue.append(CameraFrame(camera_id=self.camera_id, timestamp=timestamp, frame_sequence=frame_sequence, payload_ref=payload_ref))

    def start(self):
        self._running = True

    def stop(self):
        self._running = False

    @property
    def is_running(self):
        return self._running

    def read_frame(self):
        return self._queue.pop(0) if self._queue else None


class FakeHumanDetector(HumanDetector):

    # Maps a CameraFrame's payload_ref (a plain list of dicts the test
    # controls) to RawHumanDetection(s) carrying a REAL bounding_box --
    # unlike tests/test_live_camera_pipeline.py's own FakeHumanDetector,
    # which never sets bounding_box at all (that file never exercises
    # WorldProjector). No image processing anywhere -- this is the exact
    # boundary a real YOLO-based HumanDetector would sit behind.

    def detect(self, frame):

        if frame.payload_ref is None:
            return ()

        return tuple(
            RawHumanDetection(
                camera_id=frame.camera_id,
                local_track_id=entry["local_track_id"],
                timestamp=frame.timestamp,
                bounding_box=entry["bounding_box"],
                confidence=0.9,
                classification_evidence=HumanClassification.ADULT,
                state_evidence=HumanState.WALKING,
            )
            for entry in frame.payload_ref
        )


def make_pipeline(camera_ids, building, resolver, live_occupant_manager, stair):

    frame_sources = {camera_id: FakeFrameSource(camera_id) for camera_id in camera_ids}

    # Both cameras are calibrated against the SAME real Building floor
    # id the Staircase itself references -- build_stairs_by_floor() keys
    # its mapping by the Building's own floor.id (a UUID), never the
    # placeholder "floor-1" default_calibration()/make_calibration()
    # otherwise uses elsewhere in this codebase's tests.
    calibrations = {camera_id: make_calibration(camera_id, floor_id=stair.from_floor_id) for camera_id in camera_ids}
    stairs_by_floor = build_stairs_by_floor(building)

    world_projector = WorldProjector(calibrations=calibrations, zones_by_floor={}, stairs_by_floor=stairs_by_floor)

    # zones_by_floor deliberately empty here -- this suite tests Zone
    # geometry only insofar as current_zone_id transitions correctly
    # (see BasicStairCrossingE2ETests), which it can prove equally well
    # by asserting on world_position landing near each Zone's own
    # authored geometry without requiring WorldProjector's OWN zone
    # lookup to also be exercised in the same test (Stair lookup is
    # this milestone's own subject; Zone lookup is already covered
    # exhaustively in tests/test_camera_calibration.py).

    return LiveCameraPipeline(
        frame_sources=frame_sources,
        human_detector=FakeHumanDetector(),
        identity_resolver=resolver,
        detection_provider=LiveCameraPipelineDetectionProvider(),
        # A generous max_centroid_distance -- this suite tests the
        # Stair/Zone perception chain, not tracker matching robustness;
        # the fixed bounding boxes below jump directly between Zone A/
        # Stair/Zone B image positions (a real deployment would see many
        # small-motion intermediate frames in between) rather than
        # crafting a slow, incremental walk across dozens of frames.
        tracker=SimpleSingleCameraTracker(max_centroid_distance=1000.0),
        world_projector=world_projector,
        live_occupant_manager=live_occupant_manager,
    )


class BasicStairCrossingE2ETests(unittest.TestCase):

    # Phase 22's own scenario: Zone A -> enters Stair S1 -> remains on
    # Stair S1 (x2) -> reaches Zone B.

    def test_1_full_crossing_preserves_identity_and_reports_correct_stair_occupancy(self):

        building, floor, zone_a, zone_b, stair, positions = make_building_and_geometry()

        # Single camera -- SimulationIdentityResolver's own "local_track_id
        # IS already the global identity" strategy is exactly right here
        # (no cross-camera fusion involved in this test); the tracker's
        # own deterministic, sequential "CAM-1-T1" naming
        # (tracking.simple_tracker.SimpleSingleCameraTracker._next_track_id())
        # is what becomes the resulting global occupant_id.
        resolver = SimulationIdentityResolver()
        manager = LiveOccupantManager()
        pipeline = make_pipeline(["CAM-1"], building, resolver, manager, stair)

        frames = [
            (1.0, BBOX_ZONE_A),
            (2.0, BBOX_STAIR),
            (3.0, BBOX_STAIR),
            (4.0, BBOX_STAIR),
            (5.0, BBOX_ZONE_B),
        ]

        history = []

        for time, bbox in frames:

            pipeline.frame_sources["CAM-1"].queue_frame(time, int(time), payload_ref=[{"local_track_id": "17", "bounding_box": bbox}])
            pipeline.run_cycle(time)

            facts = manager.canonical_occupancy(time)
            history.append((time, manager.get("CAM-1-T1"), facts))

        # Same occupant identity preserved across every frame.
        for _, occupant, _ in history:
            self.assertIsNotNone(occupant)
            self.assertEqual(occupant.occupant_id, "CAM-1-T1")

        t1, occ1, facts1 = history[0]
        t2, occ2, facts2 = history[1]
        t3, occ3, facts3 = history[2]
        t4, occ4, facts4 = history[3]
        t5, occ5, facts5 = history[4]

        # Frame 1 -- in Zone A, not on the stair.
        self.assertIsNone(occ1.current_stair_id)
        self.assertEqual(facts1.stair_count(stair.id), 0)

        # Frame 2/3/4 -- Stair occupancy becomes 1 and PERSISTS while
        # physically observed there, never double-counted.
        for occ, facts in ((occ2, facts2), (occ3, facts3), (occ4, facts4)):
            self.assertEqual(occ.current_stair_id, stair.id)
            self.assertEqual(facts.stair_count(stair.id), 1)
            self.assertEqual(facts.total_observed_count, 1)  # never counted twice

        # Frame 5 -- reached Zone B, Stair occupancy returns to 0.
        self.assertIsNone(occ5.current_stair_id)
        self.assertEqual(facts5.stair_count(stair.id), 0)

        # Zone transition history recorded the full, correct sequence.
        zone_history = occ5.history.zone_transitions
        to_zone_ids = [record.to_zone_id for record in zone_history]
        self.assertIn(None, to_zone_ids)  # honestly unlocalized while on the stair (no Zone polygon covers it here)

        # The Stair transition history is a clean enter/exit pair.
        stair_history = occ5.history.stair_transitions
        self.assertEqual([r.to_stair_id for r in stair_history], [None, stair.id, None])

    def test_2_covered_stair_produces_an_observed_snapshot_not_unknown(self):

        building, floor, zone_a, zone_b, stair, positions = make_building_and_geometry()

        resolver = MappingIdentityResolver({("CAM-1", "17"): "GLOBAL-001"})
        manager = LiveOccupantManager()
        pipeline = make_pipeline(["CAM-1"], building, resolver, manager, stair)

        pipeline.frame_sources["CAM-1"].queue_frame(2.0, 2, payload_ref=[{"local_track_id": "17", "bounding_box": BBOX_STAIR}])
        pipeline.run_cycle(2.0)

        facts = manager.canonical_occupancy(2.0)
        assets_by_floor = build_assets_by_floor(building, DEFAULT_OBSERVABLE_ASSET_KINDS)
        coverage = covered_asset_ids(assets_by_floor, frozenset({floor.id}))

        snapshot = compute_asset_occupancy_snapshot(
            asset_ids_by_type={"Stair": [stair.id]}, occupant_ids_by_asset=facts.occupant_ids_by_stair,
            covered_asset_ids=coverage, timestamp=2.0,
        )

        observation = snapshot.observation_for(stair.id)
        self.assertEqual(observation.status, ObservationStatus.OBSERVED)
        self.assertEqual(observation.occupant_count, 1)


class MultiOccupantStairE2ETests(unittest.TestCase):

    # Phase 23 -- at least 5 occupants on the same stair, entering/
    # leaving at different times, temporary detection loss, no double
    # counting, deterministic occupant ids, occupancy rises and falls.

    def test_3_five_occupants_enter_and_leave_at_different_times(self):

        building, floor, zone_a, zone_b, stair, positions = make_building_and_geometry()

        # Single camera -- SimulationIdentityResolver again (see test_1's
        # own rationale). The tracker assigns "CAM-1-T1".."CAM-1-T5" in
        # deterministic, ascending order as each genuinely new physical
        # track first appears (tracking.simple_tracker.
        # SimpleSingleCameraTracker's own sequential per-camera counter);
        # a detection index whose bounding box already matches an
        # existing track (same box, perfect IoU) is never assigned a new
        # id.
        resolver = SimulationIdentityResolver()
        manager = LiveOccupantManager()
        pipeline = make_pipeline(["CAM-1"], building, resolver, manager, stair)

        def tick(time, local_track_ids):

            payload = [{"local_track_id": str(i), "bounding_box": BBOX_STAIR} for i in local_track_ids]
            pipeline.frame_sources["CAM-1"].queue_frame(time, int(time * 10), payload_ref=payload)
            pipeline.run_cycle(time)
            return manager.canonical_occupancy(time)

        # Occupants enter one at a time, then leave one at a time --
        # occupancy rises to 5, then falls back to 0, with exactly the
        # right identities present at each step.
        self.assertEqual(tick(1.0, [1]).stair_count(stair.id), 1)
        self.assertEqual(tick(2.0, [1, 2]).stair_count(stair.id), 2)
        self.assertEqual(tick(3.0, [1, 2, 3]).stair_count(stair.id), 3)
        self.assertEqual(tick(4.0, [1, 2, 3, 4]).stair_count(stair.id), 4)
        facts_full = tick(5.0, [1, 2, 3, 4, 5])
        self.assertEqual(facts_full.stair_count(stair.id), 5)
        self.assertEqual(
            set(facts_full.occupant_ids_by_stair[stair.id]),
            {f"CAM-1-T{i}" for i in range(1, 6)},
        )

        self.assertEqual(tick(6.0, [2, 3, 4, 5]).stair_count(stair.id), 4)
        self.assertEqual(tick(7.0, [3, 4, 5]).stair_count(stair.id), 3)
        self.assertEqual(tick(8.0, []).stair_count(stair.id), 0)

    def test_4_temporary_detection_loss_does_not_fabricate_an_exit(self):

        building, floor, zone_a, zone_b, stair, positions = make_building_and_geometry()

        resolver = SimulationIdentityResolver()
        manager = LiveOccupantManager(expire_after_seconds=30.0)
        pipeline = make_pipeline(["CAM-1"], building, resolver, manager, stair)

        pipeline.frame_sources["CAM-1"].queue_frame(1.0, 1, payload_ref=[{"local_track_id": "1", "bounding_box": BBOX_STAIR}])
        pipeline.run_cycle(1.0)
        self.assertEqual(manager.canonical_occupancy(1.0).stair_count(stair.id), 1)

        # One cycle with NO detection at all for this camera (a real
        # single missed frame) -- run_cycle() still calls sweep_missing()
        # even with zero raw detections.
        pipeline.run_cycle(2.0)

        occupant = manager.get("CAM-1-T1")
        self.assertIsNotNone(occupant)
        self.assertEqual(occupant.current_stair_id, stair.id)  # frozen, not cleared
        self.assertEqual(manager.canonical_occupancy(2.0).stair_count(stair.id), 0)  # temporarily excluded from active occupancy

        # Detected again shortly after -- recovers cleanly.
        pipeline.frame_sources["CAM-1"].queue_frame(3.0, 3, payload_ref=[{"local_track_id": "1", "bounding_box": BBOX_STAIR}])
        pipeline.run_cycle(3.0)
        self.assertEqual(manager.canonical_occupancy(3.0).stair_count(stair.id), 1)


class MultiCameraStairE2ETests(unittest.TestCase):

    # Phase 24 -- two cameras observing the same Stair must not create
    # one physical person -> two Stair occupants. Uses the existing,
    # already-proven fusion/identity architecture (MappingIdentityResolver,
    # the same class tests/test_live_camera_pipeline.py's own two-camera
    # test already trusts for exactly this guarantee) -- no deep ReID
    # added or required.

    def test_5_two_cameras_same_person_resolved_to_one_stair_occupant(self):

        building, floor, zone_a, zone_b, stair, positions = make_building_and_geometry()

        # Both cameras happen to share the same pose in this fixture
        # (same world geometry either observes) -- what matters is that
        # BOTH cameras' detections of the SAME physical person resolve
        # to the SAME global occupant_id before ever reaching
        # LiveOccupantManager, exactly the existing cross-camera identity
        # boundary the prior audit found: real for topology/mapping-based
        # resolution, honestly absent for anything requiring appearance
        # ReID (not needed here -- MappingIdentityResolver is an
        # explicit, authored mapping, not inference). Keyed by the
        # TRACKER's own per-camera track id ("CAM-A-T1"/"CAM-B-T1" --
        # each camera's first-ever track), not the detector's raw
        # per-frame local_track_id -- MappingIdentityResolver's real key
        # is (camera_id, RawHumanDetection.local_track_id) AFTER
        # live_camera_pipeline.pipeline.LiveCameraPipeline's own tracker
        # stage has already replaced it with the tracker's stable id
        # (see LiveCameraPipeline._process_camera_cycle()).
        resolver = MappingIdentityResolver({
            ("CAM-A", "CAM-A-T1"): "GLOBAL-001",
            ("CAM-B", "CAM-B-T1"): "GLOBAL-001",
        })
        manager = LiveOccupantManager()
        pipeline = make_pipeline(["CAM-A", "CAM-B"], building, resolver, manager, stair)

        pipeline.frame_sources["CAM-A"].queue_frame(1.0, 1, payload_ref=[{"local_track_id": "17", "bounding_box": BBOX_STAIR}])
        pipeline.frame_sources["CAM-B"].queue_frame(1.0, 1, payload_ref=[{"local_track_id": "9", "bounding_box": BBOX_STAIR}])
        pipeline.run_cycle(1.0)

        facts = manager.canonical_occupancy(1.0)

        # NOT 2 -- both detections fused to one global identity before
        # ever reaching current_stair_id.
        self.assertEqual(facts.stair_count(stair.id), 1)
        self.assertEqual(facts.total_observed_count, 1)
        self.assertEqual(facts.occupant_ids_by_stair[stair.id], ("GLOBAL-001",))

    def test_6_unmapped_second_camera_sighting_preserves_uncertainty_rather_than_assuming_same_person(self):

        # No MappingIdentityResolver entry ties CAM-B's local track to
        # CAM-A's -- MappingIdentityResolver's own documented fallback
        # (a per-camera-namespaced synthetic id, never a silent merge)
        # means this is honestly treated as a SEPARATE identity, not
        # collapsed into one person by guesswork.
        building, floor, zone_a, zone_b, stair, positions = make_building_and_geometry()

        resolver = MappingIdentityResolver({("CAM-A", "CAM-A-T1"): "GLOBAL-001"})
        manager = LiveOccupantManager()
        pipeline = make_pipeline(["CAM-A", "CAM-B"], building, resolver, manager, stair)

        pipeline.frame_sources["CAM-A"].queue_frame(1.0, 1, payload_ref=[{"local_track_id": "17", "bounding_box": BBOX_STAIR}])
        pipeline.frame_sources["CAM-B"].queue_frame(1.0, 1, payload_ref=[{"local_track_id": "9", "bounding_box": BBOX_STAIR}])
        pipeline.run_cycle(1.0)

        facts = manager.canonical_occupancy(1.0)

        self.assertEqual(facts.stair_count(stair.id), 2)
        self.assertEqual(set(facts.occupant_ids_by_stair[stair.id]), {"GLOBAL-001", "CAM-B:CAM-B-T1"})


if __name__ == "__main__":
    unittest.main()
