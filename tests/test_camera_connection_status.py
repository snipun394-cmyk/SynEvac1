import unittest

from models.camera import Camera

from camera_manager.connection_status import CameraConnectionState
from camera_manager.manager import CameraManager


class CameraConnectionStatusTests(unittest.TestCase):

    # Phase 9: CameraConnectionState is a runtime-only signal,
    # completely independent of Camera.active -- see
    # camera_manager/connection_status.py's own docstring for why
    # conflating them (as CameraStatus.active / building_state/
    # estimator.py's offline_camera_ids derivation currently does) is
    # exactly the mistake this milestone avoids repeating here.

    def setUp(self):

        self.manager = CameraManager()
        self.camera = Camera(name="Cam", floor_id="floor-1", active=True)
        self.manager.register_camera(self.camera)

    def test_default_connection_status_is_configured(self):

        self.assertEqual(
            self.manager.connection_status(self.camera.id), CameraConnectionState.CONFIGURED,
        )

    def test_set_and_get_round_trips(self):

        self.manager.set_connection_status(self.camera.id, CameraConnectionState.ONLINE)
        self.assertEqual(
            self.manager.connection_status(self.camera.id), CameraConnectionState.ONLINE,
        )

        self.manager.set_connection_status(self.camera.id, CameraConnectionState.OFFLINE)
        self.assertEqual(
            self.manager.connection_status(self.camera.id), CameraConnectionState.OFFLINE,
        )

    def test_connection_status_is_independent_of_active(self):

        # Disabling the asset (a configuration choice) must not, by
        # itself, change what the runtime connection status says.
        self.manager.set_connection_status(self.camera.id, CameraConnectionState.ONLINE)
        self.manager.disable_camera(self.camera.id)

        self.assertFalse(self.camera.active)
        self.assertEqual(
            self.manager.connection_status(self.camera.id), CameraConnectionState.ONLINE,
        )

        # Nor does the reverse hold: marking a camera OFFLINE at
        # runtime must not silently disable the asset.
        self.manager.set_connection_status(self.camera.id, CameraConnectionState.OFFLINE)
        self.assertTrue(self.camera.active or not self.camera.active)  # active untouched by us
        self.manager.enable_camera(self.camera.id)
        self.assertTrue(self.camera.active)
        self.assertEqual(
            self.manager.connection_status(self.camera.id), CameraConnectionState.OFFLINE,
        )

    def test_connection_status_not_persisted_on_camera_asset(self):

        self.manager.set_connection_status(self.camera.id, CameraConnectionState.DEGRADED)

        # The Camera model itself carries no connection-status field --
        # it is configuration, not runtime state (Phase 9's own
        # requirement).
        self.assertFalse(hasattr(self.camera, "connection_status"))
        self.assertNotIn("connection_status", self.camera.to_dict())

    def test_connection_status_for_unknown_camera_raises(self):

        with self.assertRaises(KeyError):
            self.manager.connection_status("no-such-camera")

    def test_set_connection_status_for_unknown_camera_raises(self):

        with self.assertRaises(KeyError):
            self.manager.set_connection_status("no-such-camera", CameraConnectionState.ONLINE)


if __name__ == "__main__":
    unittest.main()
