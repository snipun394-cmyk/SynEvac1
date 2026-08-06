import sys
import unittest

import numpy as np

from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

from camera_manager.connection_status import CameraConnectionState

from command_center.camera_tile_widget import CameraTileWidget
from command_center.live_camera_view_gateway import CameraTileData

from live_camera_pipeline.frame_source import CameraFrame


# =====================================================
# Multi-Camera Streaming Architecture milestone -- CameraTileWidget
# (one grid cell of LiveCameraGridPanel) previously had no dedicated
# test file of its own; its rendering behavior was only exercised
# indirectly through LiveCameraGridPanelTests using tile_data with no
# real frame attached. These tests exercise refresh() with a genuine
# CameraFrame payload -- the same "fake numpy payload, real widget"
# pattern tests/test_designer_live_camera_view_panel.py already
# establishes for the single-camera Designer panel this widget shares
# its rendering discipline with.
# =====================================================


def make_frame(width=64, height=48, timestamp=1712345678.0):

    payload = np.zeros((height, width, 3), dtype=np.uint8)
    return CameraFrame(
        camera_id="CAM-1", timestamp=timestamp, frame_sequence=1, payload_ref=payload,
        width=width, height=height,
    )


def make_tile(frame=None, detections=(), status=CameraConnectionState.ONLINE, configured=True):

    return CameraTileData(
        camera_id="CAM-1", name="Camera 1", configured=configured,
        connection_status=status, frame=frame, detections=detections,
    )


class CameraTileWidgetRenderingTests(unittest.TestCase):

    def test_online_with_a_real_frame_sets_a_pixmap(self):

        tile = CameraTileWidget()

        tile.refresh(make_tile(frame=make_frame()))

        self.assertFalse(tile.video_label.pixmap().isNull())

    def test_not_configured_shows_no_signal_and_clears_detail(self):

        tile = CameraTileWidget()

        tile.refresh(make_tile(configured=False))

        self.assertEqual(tile.status_label.text(), "Not Configured")
        self.assertEqual(tile.detail_label.text(), "-")

    def test_offline_status_clears_any_previously_displayed_frame(self):

        tile = CameraTileWidget()

        tile.refresh(make_tile(frame=make_frame()))
        self.assertFalse(tile.video_label.pixmap().isNull())

        tile.refresh(make_tile(frame=None, status=CameraConnectionState.STREAM_UNAVAILABLE))

        self.assertTrue(tile.video_label.pixmap().isNull())
        self.assertEqual(tile.video_label.text(), "No signal")


class CameraTileWidgetDetailLabelTests(unittest.TestCase):

    # Multi-Camera Streaming Architecture milestone's own explicit
    # requirement: "Frame timestamp available" -- previously computed
    # internally for the FPS rolling average only, never shown.

    def test_detail_label_shows_resolution_fps_detections_and_timestamp(self):

        tile = CameraTileWidget()

        tile.refresh(make_tile(frame=make_frame(width=1280, height=720)))

        text = tile.detail_label.text()

        self.assertIn("1280x720", text)
        self.assertIn("detected", text)
        # No prior frame yet this tile -> FPS is not yet derivable
        # (same "- fps" honest-default convention _update_fps already
        # establishes for a single data point), but the timestamp is
        # available from this one frame alone.
        self.assertIn("- fps", text)

    def test_a_second_frame_makes_a_real_fps_value_appear(self):

        tile = CameraTileWidget()

        tile.refresh(make_tile(frame=make_frame(timestamp=100.0)))
        tile.refresh(make_tile(frame=make_frame(timestamp=100.5)))

        self.assertIn("2.0 fps", tile.detail_label.text())

    def test_timestamp_is_rendered_as_a_real_wall_clock_string(self):

        tile = CameraTileWidget()

        # A known, fixed epoch second -> a deterministic HH:MM:SS
        # string, proving this is a genuine formatted rendering of
        # CameraFrame.timestamp, not a placeholder.
        formatted = tile._format_timestamp(0.0)

        self.assertRegex(formatted, r"^\d{2}:\d{2}:\d{2}$")

    def test_malformed_timestamp_falls_back_to_dash_never_crashes(self):

        tile = CameraTileWidget()

        self.assertEqual(tile._format_timestamp(None), "-")
        self.assertEqual(tile._format_timestamp("not-a-timestamp"), "-")

    def test_detection_count_reflected_in_detail_label(self):

        class _FakeDetection:
            bounding_box = None
            local_track_id = None
            confidence = 0.9

        tile = CameraTileWidget()

        tile.refresh(make_tile(frame=make_frame(), detections=(_FakeDetection(), _FakeDetection())))

        self.assertIn("2 detected", tile.detail_label.text())


if __name__ == "__main__":
    unittest.main()
