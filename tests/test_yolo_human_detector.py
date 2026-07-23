import unittest

from perception.models.human_observation import HumanClassification

from live_camera_pipeline.frame_source import CameraFrame

from human_detection.yolo_backend import ModelWeightsNotFoundError, UltralyticsYOLOBackend
from human_detection.yolo_human_detector import YOLOHumanDetector

from tests.human_detection_fixtures import FakeYOLOBackend, non_person, person


# =====================================================
# Real Human Detection Pipeline milestone, Phase 7 -- deterministic,
# fully offline tests. Zero network access, zero CCTV access anywhere
# in this file; no YOLO weights are downloaded or required.
# =====================================================


def make_frame(camera_id="CAM-001", timestamp=1.0, frame_sequence=0, payload_ref="fake-image"):

    return CameraFrame(camera_id=camera_id, timestamp=timestamp, frame_sequence=frame_sequence, payload_ref=payload_ref)


class ZeroAndSinglePersonTests(unittest.TestCase):

    def setUp(self):

        self.backend = FakeYOLOBackend()
        self.detector = YOLOHumanDetector(self.backend)

    def test_1_frame_with_zero_people(self):

        self.backend.queue_result()  # empty result

        detections = self.detector.detect(make_frame())

        self.assertEqual(detections, ())

    def test_2_frame_with_one_person(self):

        self.backend.queue_result(person(confidence=0.87, box=(10.0, 20.0, 30.0, 80.0)))

        detections = self.detector.detect(make_frame())

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].bounding_box, (10.0, 20.0, 30.0, 80.0))
        self.assertAlmostEqual(detections[0].confidence, 0.87)


class MultiplePeopleTests(unittest.TestCase):

    def test_3_frame_with_multiple_people(self):

        backend = FakeYOLOBackend()
        backend.queue_result(
            person(confidence=0.9, box=(0.0, 0.0, 10.0, 10.0)),
            person(confidence=0.8, box=(20.0, 0.0, 30.0, 10.0)),
            person(confidence=0.7, box=(40.0, 0.0, 50.0, 10.0)),
        )
        detector = YOLOHumanDetector(backend)

        detections = detector.detect(make_frame())

        self.assertEqual(len(detections), 3)
        # Distinct detector-local indices -- required so IdentityResolver
        # never collapses simultaneous people in one frame into one
        # identity (see YOLOHumanDetector's own docstring).
        self.assertEqual(
            {d.local_track_id for d in detections}, {"0", "1", "2"},
        )


class ConfidenceAndBoundingBoxPreservationTests(unittest.TestCase):

    def test_4_detector_confidence_preserved_exactly(self):

        backend = FakeYOLOBackend()
        backend.queue_result(person(confidence=0.6234))
        detector = YOLOHumanDetector(backend)

        detections = detector.detect(make_frame())

        self.assertEqual(detections[0].confidence, 0.6234)

    def test_5_bounding_boxes_preserved_exactly(self):

        backend = FakeYOLOBackend()
        box = (12.5, 34.75, 56.0, 98.25)
        backend.queue_result(person(confidence=0.5, box=box))
        detector = YOLOHumanDetector(backend)

        detections = detector.detect(make_frame())

        self.assertEqual(detections[0].bounding_box, box)


class ClassFilteringTests(unittest.TestCase):

    def test_6_only_person_class_accepted(self):

        backend = FakeYOLOBackend()
        backend.queue_result(person(confidence=0.9))
        detector = YOLOHumanDetector(backend)

        detections = detector.detect(make_frame())

        self.assertEqual(len(detections), 1)

    def test_7_non_person_classes_ignored(self):

        backend = FakeYOLOBackend()
        backend.queue_result(
            non_person("chair"), non_person("backpack"), non_person("dog"),
        )
        detector = YOLOHumanDetector(backend)

        detections = detector.detect(make_frame())

        self.assertEqual(detections, ())

    def test_mixed_person_and_non_person_only_person_survives(self):

        backend = FakeYOLOBackend()
        backend.queue_result(
            non_person("chair"), person(confidence=0.9), non_person("backpack"),
        )
        detector = YOLOHumanDetector(backend)

        detections = detector.detect(make_frame())

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].confidence, 0.9)


class FailureModeTests(unittest.TestCase):

    def test_8_malformed_frame_returns_empty_never_raises(self):

        backend = FakeYOLOBackend()
        detector = YOLOHumanDetector(backend)

        malformed_frame = make_frame(payload_ref=None)

        self.assertEqual(detector.detect(malformed_frame), ())
        self.assertEqual(backend.infer_calls, [])  # never even attempted inference on None

    def test_9_missing_model_weights_fails_honestly_at_construction(self):

        with self.assertRaises(ModelWeightsNotFoundError):
            UltralyticsYOLOBackend("definitely/does/not/exist.pt")

    def test_10_detector_inference_exception_never_crashes_the_pipeline(self):

        backend = FakeYOLOBackend()
        backend.fail_next_with(RuntimeError("simulated corrupted model / bad frame"))
        detector = YOLOHumanDetector(backend)

        detections = detector.detect(make_frame())  # must not raise

        self.assertEqual(detections, ())

    def test_11_empty_model_result_is_honest_not_an_error(self):

        backend = FakeYOLOBackend()
        backend.set_default_result()  # every call returns ()
        detector = YOLOHumanDetector(backend)

        self.assertEqual(detector.detect(make_frame()), ())
        self.assertEqual(detector.detect(make_frame()), ())


class RepeatedFramesAndMultiCameraTests(unittest.TestCase):

    def test_12_repeated_frames_produce_independent_results(self):

        backend = FakeYOLOBackend()
        backend.queue_result(person(confidence=0.9))
        backend.queue_result()  # second frame -- nobody
        backend.queue_result(person(confidence=0.4), person(confidence=0.5))
        detector = YOLOHumanDetector(backend)

        first = detector.detect(make_frame(frame_sequence=0))
        second = detector.detect(make_frame(frame_sequence=1))
        third = detector.detect(make_frame(frame_sequence=2))

        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 0)
        self.assertEqual(len(third), 2)

    def test_13_two_independent_cameras_never_cross_contaminate(self):

        backend = FakeYOLOBackend()
        backend.queue_result(person(confidence=0.9))
        backend.queue_result(person(confidence=0.8), person(confidence=0.7))
        detector = YOLOHumanDetector(backend)

        cam_a = detector.detect(make_frame(camera_id="CAM-A"))
        cam_b = detector.detect(make_frame(camera_id="CAM-B"))

        self.assertEqual(len(cam_a), 1)
        self.assertEqual(len(cam_b), 2)

    def test_14_camera_id_preserved(self):

        backend = FakeYOLOBackend()
        backend.queue_result(person(confidence=0.9))
        detector = YOLOHumanDetector(backend)

        detections = detector.detect(make_frame(camera_id="CAM-XYZ"))

        self.assertEqual(detections[0].camera_id, "CAM-XYZ")

    def test_15_timestamp_preserved(self):

        backend = FakeYOLOBackend()
        backend.queue_result(person(confidence=0.9))
        detector = YOLOHumanDetector(backend)

        detections = detector.detect(make_frame(timestamp=42.5))

        self.assertEqual(detections[0].timestamp, 42.5)

    def test_16_frame_sequence_does_not_confuse_detection_across_frames(self):

        backend = FakeYOLOBackend()
        backend.queue_result(person(confidence=0.9))
        backend.queue_result(person(confidence=0.8))
        detector = YOLOHumanDetector(backend)

        first = detector.detect(make_frame(frame_sequence=5, timestamp=5.0))
        second = detector.detect(make_frame(frame_sequence=6, timestamp=6.0))

        self.assertEqual(first[0].timestamp, 5.0)
        self.assertEqual(second[0].timestamp, 6.0)
        self.assertEqual(first[0].confidence, 0.9)
        self.assertEqual(second[0].confidence, 0.8)


class HonestClassificationAndStateTests(unittest.TestCase):

    # Phase 3's own hard requirement -- never fabricate Adult/Child/
    # Elderly/Wheelchair/Firefighter/Fire Warden or Walking/Running/
    # Fallen/... from a single generic bounding box.

    def test_classification_evidence_is_always_unknown(self):

        backend = FakeYOLOBackend()
        backend.queue_result(person(confidence=0.9))
        detector = YOLOHumanDetector(backend)

        detections = detector.detect(make_frame())

        self.assertEqual(detections[0].classification_evidence, HumanClassification.UNKNOWN)

    def test_state_evidence_is_always_none(self):

        backend = FakeYOLOBackend()
        backend.queue_result(person(confidence=0.9))
        detector = YOLOHumanDetector(backend)

        detections = detector.detect(make_frame())

        self.assertIsNone(detections[0].state_evidence)

    def test_is_never_marked_a_false_positive_by_the_detector_itself(self):

        # is_false_positive is a Simulation-only camera-imperfection
        # concept (virtual_camera.imperfections) -- a real detector has
        # no honest basis to assert it either way; the default (False)
        # is the correct, unmodified value.

        backend = FakeYOLOBackend()
        backend.queue_result(person(confidence=0.9))
        detector = YOLOHumanDetector(backend)

        detections = detector.detect(make_frame())

        self.assertFalse(detections[0].is_false_positive)


if __name__ == "__main__":
    unittest.main()
