from dataclasses import dataclass
from typing import Any, Optional, Tuple

from camera_manager.connection_status import CameraConnectionState


# =====================================================
# Live CCTV Dashboard milestone -- the read-only counterpart to
# command_center/live_operator_action_gateway.py's own precedent:
# deliberately the one place in `command_center` allowed to import
# `camera_manager.manager`/`live_camera_pipeline` directly, because
# unlike every panel file tests/test_live_command_center.py::
# CommandCenterLiveIntegrationGuardTests scans, this module's entire
# purpose is to be the bridge from an already-running LiveRuntime's
# camera state into Command Center's display-only Live CCTV grid.
# Never listed in that guard's own _LIVE_INTEGRATION_FILES tuple --
# exclusion by omission, the exact mechanism live_operator_action_
# gateway.py already relies on.
#
# Strictly read-only: no method here ever starts/stops a frame source,
# constructs a decoder, or runs detection -- it only ever reads
# already-computed state off CameraManager/LiveCameraPipeline, the
# SAME "reuse the exact production RTSP pipeline, never a second
# decode/detector" discipline every prior CCTV milestone established.
#
# `configured` below means "this camera has an active production frame
# source this session" (camera_id in frame_sources) -- distinct from
# Camera.active (a Designer-authored, persisted flag) and from
# CameraConnectionState (a moment-to-moment runtime signal); a camera
# can be `active=True` in the Building yet `configured=False` here
# simply because it has no RTSP address / Live mode / credential set.
# =====================================================


@dataclass(frozen=True)
class CameraTileData:

    camera_id: str
    name: str
    configured: bool
    connection_status: CameraConnectionState

    # Deliberately untyped Any, never live_camera_pipeline.frame_source.
    # CameraFrame / .human_detector.RawHumanDetection imported here --
    # the same "typed Any, not imported for its own sake" convention
    # every CommandCenterSnapshot passthrough field already follows
    # (command_center/data_source.py). This module is the one place
    # allowed to import those real types (see header) -- callers
    # (CameraTileWidget) stay opaque consumers.
    frame: Optional[Any] = None
    detections: Tuple[Any, ...] = ()


class LiveCameraViewGateway:

    def __init__(self, camera_manager, camera_pipeline, frame_sources):

        self._camera_manager = camera_manager
        self._camera_pipeline = camera_pipeline
        self._frame_sources = frame_sources

    # =====================================================

    def camera_tiles(self) -> Tuple[CameraTileData, ...]:

        # One entry per camera the Building has, configured or not --
        # a stable name-sorted order so the grid's tile positions don't
        # jitter between refreshes for reasons unrelated to the actual
        # camera set changing.

        cameras = sorted(self._camera_manager.all_cameras(), key=lambda camera: camera.name)

        tiles = []

        for camera in cameras:

            configured = camera.id in self._frame_sources

            frame = (
                self._camera_pipeline.latest_frame(camera.id)
                if configured and self._camera_pipeline is not None else None
            )
            detections = (
                self._camera_pipeline.latest_detections(camera.id)
                if configured and self._camera_pipeline is not None else ()
            )

            tiles.append(
                CameraTileData(
                    camera_id=camera.id,
                    name=camera.name,
                    configured=configured,
                    connection_status=self._camera_manager.connection_status(camera.id),
                    frame=frame,
                    detections=detections,
                )
            )

        return tuple(tiles)
