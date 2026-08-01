import sys
import unittest

import numpy as np

from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

from camera_manager.connection_status import CameraConnectionState

from designer.widgets.live_camera_view_panel import LiveCameraViewPanel

from live_camera_pipeline.frame_source import CameraFrame


def make_frame(width=64, height=48):

    payload = np.zeros((height, width, 3), dtype=np.uint8)
    return CameraFrame(camera_id="CAM-1", timestamp=1.0, frame_sequence=1, payload_ref=payload)


class LiveCameraViewPanelTests(unittest.TestCase):

    def test_online_with_a_real_frame_sets_a_pixmap(self):

        panel = LiveCameraViewPanel()

        panel.refresh("Camera 1", CameraConnectionState.ONLINE, make_frame())

        self.assertFalse(panel.video_label.pixmap().isNull())
        self.assertIn("Camera 1", panel.camera_name_label.text())

    def test_offline_status_clears_any_previously_displayed_frame(self):

        panel = LiveCameraViewPanel()
        panel.refresh("Camera 1", CameraConnectionState.ONLINE, make_frame())
        self.assertFalse(panel.video_label.pixmap().isNull())

        panel.refresh("Camera 1", CameraConnectionState.OFFLINE, None)

        pixmap = panel.video_label.pixmap()
        self.assertTrue(pixmap is None or pixmap.isNull())

    def test_online_status_with_no_frame_shows_no_signal_without_crashing(self):

        panel = LiveCameraViewPanel()

        panel.refresh("Camera 1", CameraConnectionState.ONLINE, None)

        pixmap = panel.video_label.pixmap()
        self.assertTrue(pixmap is None or pixmap.isNull())

    def test_no_session_state_shows_placeholder_without_crashing(self):

        panel = LiveCameraViewPanel()

        panel.refresh(None, None, None)

        self.assertEqual(panel.camera_name_label.text(), "No camera configured")

    def test_malformed_payload_is_handled_gracefully(self):

        panel = LiveCameraViewPanel()
        frame = CameraFrame(camera_id="CAM-1", timestamp=1.0, frame_sequence=1, payload_ref="not-an-image")

        panel.refresh("Camera 1", CameraConnectionState.ONLINE, frame)

        pixmap = panel.video_label.pixmap()
        self.assertTrue(pixmap is None or pixmap.isNull())


if __name__ == "__main__":
    unittest.main()
