import unittest

from perception.models.human_observation import HumanClassification, HumanState

from live_camera_pipeline.human_detector import RawHumanDetection
from live_camera_pipeline.identity_resolver import (
    MappingIdentityResolver,
    SimulationIdentityResolver,
)

from multi_camera_fusion.engine import MultiCameraFusionEngine


def make_raw(camera_id, local_track_id, timestamp=0.0, **overrides):

    fields = dict(
        camera_id=camera_id,
        local_track_id=local_track_id,
        timestamp=timestamp,
        confidence=0.9,
        classification_evidence=HumanClassification.ADULT,
        state_evidence=HumanState.WALKING,
        floor_id="floor-1",
        zone_id="zone-1",
    )
    fields.update(overrides)

    return RawHumanDetection(**fields)


class SimulationIdentityResolverTests(unittest.TestCase):

    def test_local_track_id_becomes_the_resolved_occupant_id(self):

        resolver = SimulationIdentityResolver()

        raw = make_raw("CAM-1", "occupant-42")
        detections = resolver.resolve([raw], time=0.0)

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].occupant_id, "occupant-42")
        self.assertEqual(detections[0].camera_id, "CAM-1")


class MappingIdentityResolverScenarioATests(unittest.TestCase):

    # Phase 7 A: same person, seen by 2 cameras, mapped to one global
    # id -> occupancy 1.

    def test_same_person_two_cameras_mapped_fuses_to_one_track(self):

        resolver = MappingIdentityResolver({
            ("CAM-A", "17"): "GLOBAL-001",
            ("CAM-B", "9"): "GLOBAL-001",
        })

        raw = [
            make_raw("CAM-A", "17", timestamp=1.0),
            make_raw("CAM-B", "9", timestamp=1.0),
        ]

        detections = resolver.resolve(raw, time=1.0)

        engine = MultiCameraFusionEngine()
        result = engine.fuse(detections, time=1.0)

        self.assertEqual(len(result.tracks), 1)
        self.assertEqual(result.tracks[0].track_id, "GLOBAL-001")
        self.assertEqual(set(result.tracks[0].source_camera_ids), {"CAM-A", "CAM-B"})


class MappingIdentityResolverScenarioBTests(unittest.TestCase):

    # Phase 7 B: two different people, seen by 2 different cameras,
    # no relationship declared -> occupancy 2.

    def test_two_different_people_stay_two_tracks(self):

        resolver = MappingIdentityResolver()

        raw = [
            make_raw("CAM-A", "1", timestamp=1.0),
            make_raw("CAM-B", "2", timestamp=1.0),
        ]

        detections = resolver.resolve(raw, time=1.0)

        engine = MultiCameraFusionEngine()
        result = engine.fuse(detections, time=1.0)

        self.assertEqual(len(result.tracks), 2)


class MappingIdentityResolverScenarioCTests(unittest.TestCase):

    # Phase 7 C: the exact collision case -- CAM-A track_5 and CAM-B
    # track_5 must NOT automatically imply the same person, even
    # though the literal local_track_id string is identical.

    def test_same_local_track_id_on_different_cameras_does_not_auto_fuse(self):

        resolver = MappingIdentityResolver()

        raw = [
            make_raw("CAM-A", "5", timestamp=1.0),
            make_raw("CAM-B", "5", timestamp=1.0),
        ]

        detections = resolver.resolve(raw, time=1.0)

        self.assertNotEqual(detections[0].occupant_id, detections[1].occupant_id)

        engine = MultiCameraFusionEngine()
        result = engine.fuse(detections, time=1.0)

        self.assertEqual(len(result.tracks), 2)

    def test_synthesized_ids_are_namespaced_by_camera(self):

        resolver = MappingIdentityResolver()

        raw = make_raw("CAM-A", "5", timestamp=1.0)
        detection = resolver.resolve([raw], time=1.0)[0]

        self.assertTrue(detection.occupant_id.startswith("CAM-A"))
        self.assertIn("5", detection.occupant_id)


class MappingIdentityResolverScenarioDTests(unittest.TestCase):

    # Phase 7 D / Phase 8: a future resolver must be able to
    # EXPLICITLY state that two distinct (camera, local_track_id)
    # pairs are the same global person -- this is exactly what
    # set_mapping()/the constructor mapping argument does.

    def test_explicit_mapping_declares_cross_camera_identity(self):

        resolver = MappingIdentityResolver()

        raw_a = make_raw("CAM-A", "5", timestamp=1.0)
        raw_b = make_raw("CAM-B", "17", timestamp=1.0)

        # Before any mapping is declared, they resolve to different ids.
        before = resolver.resolve([raw_a, raw_b], time=1.0)
        self.assertNotEqual(before[0].occupant_id, before[1].occupant_id)

        resolver.set_mapping("CAM-A", "5", "GLOBAL-001")
        resolver.set_mapping("CAM-B", "17", "GLOBAL-001")

        after = resolver.resolve([raw_a, raw_b], time=1.0)
        self.assertEqual(after[0].occupant_id, "GLOBAL-001")
        self.assertEqual(after[1].occupant_id, "GLOBAL-001")

        engine = MultiCameraFusionEngine()
        result = engine.fuse(after, time=1.0)
        self.assertEqual(len(result.tracks), 1)


class CameraHandoverTests(unittest.TestCase):

    # Phase 13's handover scenario, exercised directly at the
    # resolver+fusion level (the full fake pipeline version lives in
    # tests/test_live_camera_pipeline.py): t=0 CAM-A sees A-17; t=1
    # both cameras see the same person; t=2 only CAM-B sees B-9.
    # A resolver that knows the handover keeps it one persistent
    # track throughout.

    def test_handover_between_two_cameras_stays_one_track_when_mapped(self):

        resolver = MappingIdentityResolver({
            ("CAM-A", "17"): "GLOBAL-001",
            ("CAM-B", "9"): "GLOBAL-001",
        })
        engine = MultiCameraFusionEngine()

        t0 = resolver.resolve([make_raw("CAM-A", "17", timestamp=0.0)], time=0.0)
        result_t0 = engine.fuse(t0, time=0.0)
        self.assertEqual(len(result_t0.tracks), 1)
        self.assertEqual(result_t0.tracks[0].track_id, "GLOBAL-001")

        t1 = resolver.resolve(
            [make_raw("CAM-A", "17", timestamp=1.0), make_raw("CAM-B", "9", timestamp=1.0)],
            time=1.0,
        )
        result_t1 = engine.fuse(t1, time=1.0)
        self.assertEqual(len(result_t1.tracks), 1)
        self.assertEqual(result_t1.tracks[0].track_id, "GLOBAL-001")
        self.assertEqual(set(result_t1.tracks[0].source_camera_ids), {"CAM-A", "CAM-B"})

        t2 = resolver.resolve([make_raw("CAM-B", "9", timestamp=2.0)], time=2.0)
        result_t2 = engine.fuse(t2, time=2.0)
        self.assertEqual(len(result_t2.tracks), 1)
        self.assertEqual(result_t2.tracks[0].track_id, "GLOBAL-001")

        # The engine's own TrackHistory recorded the handover.
        history = engine.track_history("GLOBAL-001")
        self.assertEqual(history.previous_camera_id, "CAM-A")
        self.assertEqual(history.current_camera_id, "CAM-B")


if __name__ == "__main__":
    unittest.main()
