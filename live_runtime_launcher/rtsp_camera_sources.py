from typing import Callable, Dict, Optional

from models.engineering_asset import DeviceMode

from camera_manager.connection_status import CameraConnectionState
from camera_manager.manager import CameraManager

from credential_store.store import CredentialStore

from live_camera_pipeline.rtsp_backend import FrameDecoderBackend
from live_camera_pipeline.rtsp_frame_source import RTSPFrameSource

from human_detection.opencv_decoder_backend import OpenCVFrameDecoderBackend


# =====================================================
# CP PLUS NVR -> SynEvac1 Live Runtime Integration milestone -- the
# composition-root seam named (but never built) by every prior CCTV
# readiness milestone: turning a Designer-configured Camera Asset into
# a real RTSPFrameSource. Lives in live_runtime_launcher/, not
# live_camera_pipeline/ or camera_manager/ -- both of those packages
# are forbidden from importing each other directly (tests/
# test_rtsp_camera_manager_status_integration.py's own
# test_camera_manager_status_survives_camera_manager_never_understanding_rtsp),
# so the bridge between them has to live at the composition root, one
# layer up, exactly like that test's own `_bridge()` helper already
# demonstrates.
#
# One configured Camera -> one RTSPFrameSource (never a batch/shared
# connection) -- the same granularity RTSPFrameSource/
# OpenCVFrameDecoderBackend already assume one level down. A camera
# contributes a source here only when BOTH are true:
#   - camera.mode == DeviceMode.LIVE (set via the Camera Manager Panel
#     mode dropdown -- Simulation/Replay cameras are untouched)
#   - camera.connection.rtsp_address is actually configured (set via
#     the Property Panel's RTSP Address field)
# Neither condition failing is an error -- exactly zero, some, or all
# configured cameras may qualify, and build_live_runtime() already
# treats an empty frame_sources mapping as the honest "no cameras"
# case (Phase 8 item 1 there).
# =====================================================


def build_rtsp_frame_sources(
    building,
    camera_manager: CameraManager,
    credential_store: Optional[CredentialStore] = None,
    decoder_backend_factory: Callable[[], FrameDecoderBackend] = OpenCVFrameDecoderBackend,
) -> Dict[str, object]:

    frame_sources: Dict[str, object] = {}

    if building is None:
        return frame_sources

    def on_status_changed(camera_id, status, detail):

        # The exact bridge tests/test_rtsp_camera_manager_status_
        # integration.py's own _bridge() establishes as the correct
        # composition-root pattern -- RTSPFrameSource's plain status
        # strings match CameraConnectionState's `.value` strings
        # character-for-character by construction (rtsp_frame_source.py's
        # own docstring), so this conversion never fails.
        camera_manager.set_connection_status(camera_id, CameraConnectionState(status))

    for floor in building.ordered_floors():
        for camera in floor.cameras:

            if camera.mode != DeviceMode.LIVE:
                continue

            connection = camera.connection

            if not connection.rtsp_address:
                continue

            frame_sources[camera.id] = RTSPFrameSource(
                camera_id=camera.id,
                endpoint=connection.rtsp_address,
                decoder_backend=decoder_backend_factory(),
                username=connection.username or None,
                credential_ref=connection.credential_ref,
                credential_store=credential_store,
                status_callback=on_status_changed,
            )

    return frame_sources
