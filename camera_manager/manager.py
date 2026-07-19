from typing import Dict, Optional, Tuple

from models.engineering_asset import DeviceMode

from camera_manager.connection_status import CameraConnectionState
from camera_manager.status import CameraStatus


class CameraManager:

    # The single place cameras are managed -- Phase 2/3/4's own stated
    # requirement, restated as this class's one responsibility. Two
    # kinds of state, and nothing else:
    #
    #   1. Which Camera Assets exist (self._cameras: camera_id ->
    #      Camera) -- discovered from a Building, or registered one at
    #      a time.
    #   2. Which DetectionProvider currently answers for each Mode
    #      (self._detection_providers: mode string -> DetectionProvider)
    #      -- Phase 6's registration seam. A camera's OWN mode
    #      (Camera.mode, already a field on the Camera Asset since the
    #      Camera Coverage milestone -- see models/engineering_asset.
    #      DeviceMode) decides which registered provider answers for
    #      it; CameraManager never constructs a provider itself and
    #      never imports a concrete implementation (SimulatedDetection
    #      Provider, a future LiveDetectionProvider, ...) -- only the
    #      DetectionProvider *interface* it is handed. This is what
    #      "the Camera Manager must never perform computer vision" and
    #      "it only receives DetectionProviders" both mean in practice:
    #      there is no geometry, no classification, no vision logic
    #      anywhere in this file.
    #
    # Discovery is a full, idempotent rescan (see discover_cameras()),
    # the same "rebuild rather than incrementally patch" convention
    # NavigationGraphGenerator.build() already establishes for Building
    # edits -- simpler and more robust than trying to hook every
    # possible camera-add/camera-remove call site in the Designer.

    def __init__(self):

        self._cameras: Dict[str, object] = {}
        self._detection_providers: Dict[str, object] = {}

        # Runtime-only, in-memory, never persisted onto the Camera
        # Asset itself -- see camera_manager/connection_status.py for
        # why this is deliberately separate from Camera.active.
        self._connection_states: Dict[str, CameraConnectionState] = {}

    # =====================================================
    # Discovery / Registration (Phase 2)
    # =====================================================

    def discover_cameras(self, building) -> Tuple[object, ...]:

        # Walks every floor of `building` -- a Camera Asset is
        # discoverable regardless of which floor is currently shown in
        # the Designer, matching the Building Digital Twin's own
        # building-wide (not per-floor) scope. None (no Building yet,
        # e.g. before a project is loaded) clears the registry rather
        # than raising.

        self._cameras = {}

        if building is None:
            return ()

        for floor in building.ordered_floors():
            for camera in floor.cameras:
                self.register_camera(camera)

        return self.all_cameras()

    # =====================================================

    def register_camera(self, camera) -> None:

        self._cameras[camera.id] = camera

    # =====================================================

    def remove_camera(self, camera_id: str) -> None:

        self._cameras.pop(camera_id, None)

    # =====================================================

    def all_cameras(self) -> Tuple[object, ...]:

        return tuple(self._cameras.values())

    # =====================================================
    # Lookup (Phase 2)
    # =====================================================

    def get_camera(self, camera_id: str) -> Optional[object]:

        return self._cameras.get(camera_id)

    # =====================================================

    def cameras_on_floor(self, floor_id: str) -> Tuple[object, ...]:

        return tuple(
            camera for camera in self._cameras.values() if camera.floor_id == floor_id
        )

    # =====================================================

    def cameras_in_zone(self, zone_id: str) -> Tuple[object, ...]:

        # Camera.zone_ids (models/engineering_asset.py) is the
        # Assigned Zone(s) field from the Camera Asset milestone --
        # reused as-is, never re-derived from geometry here (this
        # class does no visibility computation of any kind).

        return tuple(
            camera for camera in self._cameras.values() if zone_id in camera.zone_ids
        )

    # =====================================================
    # Enable / Disable (Phase 2)
    # =====================================================

    def enable_camera(self, camera_id: str) -> None:

        self._require_camera(camera_id).active = True

    # =====================================================

    def disable_camera(self, camera_id: str) -> None:

        self._require_camera(camera_id).active = False

    # =====================================================
    # Mode Switching (Phase 4)
    #
    # Only ever mutates Camera.mode in place -- the same Camera object
    # stays registered under the same id throughout, never replaced.
    # =====================================================

    def set_camera_mode(self, camera_id: str, mode: str) -> None:

        if mode not in DeviceMode.ALL:

            raise ValueError(
                f"Unknown camera mode {mode!r}; expected one of {DeviceMode.ALL}."
            )

        self._require_camera(camera_id).mode = mode

    # =====================================================

    def camera_mode(self, camera_id: str) -> str:

        return self._require_camera(camera_id).mode

    # =====================================================
    # Status (Phase 2)
    # =====================================================

    def camera_status(self, camera_id: str) -> CameraStatus:

        camera = self._require_camera(camera_id)

        return CameraStatus(
            camera_id=camera.id,
            name=camera.name,
            floor_id=camera.floor_id,
            zone_ids=camera.zone_ids,
            active=camera.active,
            mode=camera.mode,
            has_detection_provider=camera.mode in self._detection_providers,
        )

    # =====================================================

    def all_statuses(self) -> Tuple[CameraStatus, ...]:

        return tuple(self.camera_status(camera.id) for camera in self.all_cameras())

    # =====================================================
    # Connection Status (runtime-only; see connection_status.py)
    # =====================================================

    def set_connection_status(self, camera_id: str, state: CameraConnectionState) -> None:

        self._require_camera(camera_id)
        self._connection_states[camera_id] = state

    # =====================================================

    def connection_status(self, camera_id: str) -> CameraConnectionState:

        self._require_camera(camera_id)
        return self._connection_states.get(camera_id, CameraConnectionState.CONFIGURED)

    # =====================================================
    # Detection Provider Registration (Phase 6)
    #
    # The seam a future Simulation/Replay/Live adapter registers
    # itself against -- one DetectionProvider per Mode, not per
    # camera, since a real deployment's Live adapter, Replay adapter,
    # etc. each serve every camera currently in that mode. Registering
    # a provider for a mode never touches any Camera object.
    # =====================================================

    def register_detection_provider(self, mode: str, provider) -> None:

        if mode not in DeviceMode.ALL:

            raise ValueError(
                f"Unknown camera mode {mode!r}; expected one of {DeviceMode.ALL}."
            )

        self._detection_providers[mode] = provider

    # =====================================================

    def unregister_detection_provider(self, mode: str) -> None:

        self._detection_providers.pop(mode, None)

    # =====================================================

    def has_detection_provider(self, mode: str) -> bool:

        return mode in self._detection_providers

    # =====================================================
    # Detection Routing (Phase 3)
    #
    # No downstream package talks to an individual VirtualCamera/
    # DetectionProvider directly -- every consumer goes through one of
    # these four methods instead.
    # =====================================================

    def detections_for_camera(self, camera_id: str, time: float) -> Tuple[object, ...]:

        camera = self._require_camera(camera_id)

        provider = self._detection_providers.get(camera.mode)

        if provider is None:

            # Replay/Live remain architecture placeholders in this
            # milestone (Phase 4) -- no registered provider means "no
            # detections available yet", never an error. A camera
            # simply produces nothing until its mode's adapter is
            # registered (Phase 6).
            return ()

        return provider.detections_at(camera_id, time)

    # =====================================================

    def all_detections(self, time: float) -> Tuple[object, ...]:

        detections = []

        for camera in self.all_cameras():
            detections.extend(self.detections_for_camera(camera.id, time))

        return tuple(detections)

    # =====================================================

    def detections_by_floor(self, time: float) -> Dict[str, Tuple[object, ...]]:

        return self._group_by(self.all_detections(time), key=lambda detection: detection.floor_id)

    # =====================================================

    def detections_by_zone(self, time: float) -> Dict[str, Tuple[object, ...]]:

        return self._group_by(self.all_detections(time), key=lambda detection: detection.zone_id)

    # =====================================================

    def detections_by_camera(self, time: float) -> Dict[str, Tuple[object, ...]]:

        return self._group_by(self.all_detections(time), key=lambda detection: detection.camera_id)

    # =====================================================
    # Internals
    # =====================================================

    def _group_by(self, detections, key) -> Dict[str, Tuple[object, ...]]:

        grouped: Dict[str, list] = {}

        for detection in detections:
            grouped.setdefault(key(detection), []).append(detection)

        return {group_key: tuple(items) for group_key, items in grouped.items()}

    # =====================================================

    def _require_camera(self, camera_id: str):

        camera = self._cameras.get(camera_id)

        if camera is None:

            raise KeyError(
                f"CameraManager has no camera registered with id {camera_id!r}."
            )

        return camera
