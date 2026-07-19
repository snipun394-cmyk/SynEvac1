from typing import Dict, Tuple

from virtual_camera.detection import Detection
from virtual_camera.provider import DetectionProvider


class LiveCameraPipelineDetectionProvider(DetectionProvider):

    # The future Live-mode counterpart to
    # virtual_camera.provider.SimulatedDetectionProvider -- same
    # DetectionProvider interface, so CameraManager routes to it
    # exactly the way it already routes to SimulatedDetectionProvider
    # (register_detection_provider(DeviceMode.LIVE, this), zero
    # CameraManager change needed). Holds no camera/vision logic of
    # its own -- it is a thin buffer LiveCameraPipeline publishes
    # resolved Detections into each cycle, and CameraManager reads
    # back out via detections_at().

    def __init__(self):

        self._latest: Dict[str, Tuple[Detection, ...]] = {}

    # =====================================================

    def publish(self, camera_ids, detections) -> None:

        # `camera_ids` is every camera this cycle actually attempted a
        # read for -- a camera with zero current detections (the
        # person walked out of frame) must clear to (), never keep
        # serving a stale detection from a previous cycle forever.
        # Cameras NOT in `camera_ids` (not part of this cycle at all)
        # are left untouched.

        grouped: Dict[str, list] = {camera_id: [] for camera_id in camera_ids}

        for detection in detections:
            grouped.setdefault(detection.camera_id, []).append(detection)

        for camera_id, group in grouped.items():
            self._latest[camera_id] = tuple(group)

    # =====================================================

    def detections_at(self, camera_id: str, time: float) -> Tuple[Detection, ...]:

        return self._latest.get(camera_id, ())
