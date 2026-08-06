import sys
import unittest

from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

from camera_manager.connection_status import CameraConnectionState

from command_center.live_camera_grid_panel import LiveCameraGridPanel
from command_center.live_camera_view_gateway import CameraTileData


class _FakeGateway:

    # A test-only stand-in exposing just camera_tiles() -- never a real
    # LiveCameraViewGateway/CameraManager/LiveCameraPipeline.

    def __init__(self, tiles):
        self._tiles = tuple(tiles)

    def camera_tiles(self):
        return self._tiles


def make_tile(camera_id, name="Camera", configured=True, status=CameraConnectionState.ONLINE):

    return CameraTileData(
        camera_id=camera_id, name=name, configured=configured, connection_status=status,
    )


class LiveCameraGridPanelTests(unittest.TestCase):

    # Live CCTV Dashboard milestone -- proves the grid is genuinely
    # dynamic (tile count and layout both derive from however many
    # tiles the gateway returns, never a fixed number) and that an
    # unchanged camera set never rebuilds already-existing tile widgets.

    def test_no_gateway_shows_the_empty_state_without_crashing(self):

        panel = LiveCameraGridPanel()

        panel.show_live(None)

        self.assertFalse(panel._empty_label.isHidden())
        self.assertTrue(panel._scroll_area.isHidden())

    def test_zero_cameras_shows_the_empty_state(self):

        panel = LiveCameraGridPanel()

        panel.show_live(_FakeGateway([]))

        self.assertFalse(panel._empty_label.isHidden())
        self.assertTrue(panel._scroll_area.isHidden())

    def test_a_single_configured_camera_produces_exactly_one_tile(self):

        panel = LiveCameraGridPanel()

        panel.show_live(_FakeGateway([make_tile("CAM-1")]))

        self.assertEqual(len(panel._tiles), 1)
        self.assertIn("CAM-1", panel._tiles)
        self.assertFalse(panel._scroll_area.isHidden())

    def test_growing_the_camera_set_grows_the_grid(self):

        panel = LiveCameraGridPanel()

        panel.show_live(_FakeGateway([make_tile("CAM-1")]))
        self.assertEqual(len(panel._tiles), 1)

        panel.show_live(_FakeGateway(
            [make_tile("CAM-1"), make_tile("CAM-2"), make_tile("CAM-3"), make_tile("CAM-4")]
        ))

        self.assertEqual(len(panel._tiles), 4)
        self.assertEqual(set(panel._tiles.keys()), {"CAM-1", "CAM-2", "CAM-3", "CAM-4"})
        self.assertEqual(panel._grid_layout.count(), 4)

    def test_the_same_camera_set_does_not_rebuild_existing_tile_widgets(self):

        panel = LiveCameraGridPanel()

        panel.show_live(_FakeGateway([make_tile("CAM-1")]))
        first_tile_widget = panel._tiles["CAM-1"]

        panel.show_live(_FakeGateway([make_tile("CAM-1")]))
        second_tile_widget = panel._tiles["CAM-1"]

        self.assertIs(first_tile_widget, second_tile_widget)

    def test_an_unconfigured_camera_tile_shows_not_configured_without_crashing(self):

        panel = LiveCameraGridPanel()

        panel.show_live(_FakeGateway([make_tile("CAM-1", configured=False)]))

        tile = panel._tiles["CAM-1"]
        self.assertEqual(tile.status_label.text(), "Not Configured")


class LiveCameraGridPanelOverflowTests(unittest.TestCase):

    # Live CCTV Layout milestone -- root-cause fix for a real laboratory
    # building (32 cameras, a 6x6 grid) clipping tiles instead of
    # scrolling to them: the grid container's own natural size (driven
    # by however many tiles x each tile's own minimum size) can exceed
    # the panel's available on-screen space well before it exceeds any
    # single tile's minimum size -- a QScrollArea is the fix, not a
    # change to tile size or the grid's own column math (see
    # LiveCameraGridPanel's own class docstring).

    def test_scroll_area_hosts_the_grid_container_and_is_resizable(self):

        panel = LiveCameraGridPanel()

        self.assertIs(panel._scroll_area.widget(), panel._grid_container)
        self.assertTrue(panel._scroll_area.widgetResizable())

    def test_many_tiles_do_not_crash_and_grid_container_may_exceed_the_scroll_areas_own_size(self):

        # Regression guard for the exact overflow scenario that caused
        # today's clipping: constrain the scroll area's own viewport to
        # something deliberately smaller than 32 tiles' natural size,
        # and confirm the CONTENT (grid_container) is still allowed to
        # be taller than the VIEWPORT -- that gap is exactly what
        # scrolling now covers, instead of Qt clipping it.

        panel = LiveCameraGridPanel()
        panel.resize(300, 200)
        panel._scroll_area.resize(300, 200)

        tiles = [make_tile(f"CAM-{i}") for i in range(32)]
        panel.show_live(_FakeGateway(tiles))

        self.assertEqual(len(panel._tiles), 32)
        self.assertEqual(panel._grid_layout.count(), 32)

        # Force the container to compute its real layout-driven size
        # (each CameraTileWidget's own video_label.setMinimumSize(240,
        # 180) alone puts a 6x6 grid's natural height well past 200px).
        panel._grid_container.adjustSize()

        self.assertGreater(panel._grid_container.sizeHint().height(), panel._scroll_area.height())
        self.assertTrue(panel._scroll_area.widgetResizable())


if __name__ == "__main__":
    unittest.main()
