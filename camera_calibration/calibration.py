from typing import Dict, Optional

from camera_calibration.camera_model import CalibrationProfile


class CalibrationRegistry:

    # Camera Calibration & World Coordinate Projection milestone,
    # Phase 3 -- "store calibration separately from Camera Assets",
    # applied the same way cross_camera_identity.identity_registry.
    # IdentityRegistry stores global identities separately from
    # tracking.tracked_human.TrackedHuman: a plain camera_id -> profile
    # map, owned entirely by this package, never written onto
    # models.camera.Camera itself.

    def __init__(self):

        self._profiles: Dict[str, CalibrationProfile] = {}

    # =====================================================

    def set(self, profile: CalibrationProfile) -> None:

        self._profiles[profile.camera_id] = profile

    # =====================================================

    def get(self, camera_id: str) -> Optional[CalibrationProfile]:

        return self._profiles.get(camera_id)

    # =====================================================

    def remove(self, camera_id: str) -> None:

        self._profiles.pop(camera_id, None)

    # =====================================================

    def has(self, camera_id: str) -> bool:

        return camera_id in self._profiles

    # =====================================================

    def all_profiles(self):

        return tuple(self._profiles.values())

    # =====================================================

    def __len__(self) -> int:

        return len(self._profiles)
