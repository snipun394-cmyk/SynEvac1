from perception.models.human_observation import HumanClassification, HumanState

from live_camera_pipeline.human_detector import HumanDetector, RawHumanDetection


class MockHumanDetector(HumanDetector):

    # A deterministic stand-in for a real vision model (YOLO + a local
    # tracker) -- CCTV Pipeline End-to-End Offline Validation milestone's
    # own "do not implement YOLO, use deterministic mocks" requirement.
    # A CameraFrame's payload_ref is a plain, test-controlled sequence
    # of dicts (no image processing of any kind); each entry becomes one
    # RawHumanDetection, still namespaced by frame.camera_id exactly the
    # way a real per-camera local tracker would produce it. Reuses
    # HumanClassification/HumanState as-is -- no new vocabulary.

    def detect(self, frame):

        if frame.payload_ref is None:
            return ()

        return tuple(
            RawHumanDetection(
                camera_id=frame.camera_id,
                local_track_id=entry["local_track_id"],
                timestamp=frame.timestamp,
                confidence=entry.get("confidence", 0.9),
                classification_evidence=entry.get(
                    "classification_evidence", HumanClassification.ADULT,
                ),
                state_evidence=entry.get("state_evidence", HumanState.WALKING),
                floor_id=entry.get("floor_id", "floor-1"),
                zone_id=entry.get("zone_id", "zone-1"),
            )
            for entry in frame.payload_ref
        )
