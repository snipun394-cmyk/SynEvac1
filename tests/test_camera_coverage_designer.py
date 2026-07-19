import sys
import unittest

from PyQt6.QtWidgets import QApplication, QGraphicsPolygonItem, QGraphicsRectItem

_app = QApplication.instance() or QApplication(sys.argv)

from designer.items.camera_item import CameraItem
from designer.scene.graphics_scene import GraphicsScene

from models.camera import Camera
from models.zone import Zone


def make_zone(name, **kwargs):

    fields = dict(x=0.0, y=0.0, width=2.0, height=2.0)
    fields.update(kwargs)

    return Zone(name=name, **fields)


class CameraCoverageOverlayTests(unittest.TestCase):

    def setUp(self):

        self.scene = GraphicsScene()
        self.floor = self.scene.current_floor

        self.zone_a = make_zone("Zone A", x=0.0, y=0.0, width=5.0, height=5.0, floor_id=self.floor.id)
        self.zone_b = make_zone("Zone B", x=5.0, y=0.0, width=5.0, height=5.0, floor_id=self.floor.id)

        self.floor.add_zone(self.zone_a)
        self.floor.add_zone(self.zone_b)

        self.camera = Camera(
            name="Cam", position=(1.0, 2.5), floor_id=self.floor.id,
            rotation=0.0, horizontal_fov=360.0, max_range=20.0,
        )
        self.floor.add_camera(self.camera)

        self.scene.rebuild_scene()

    def test_coverage_disabled_by_default_and_no_overlays_present(self):

        self.assertFalse(self.scene.show_camera_coverage)
        self.assertEqual(self.scene._coverage_overlay_items, [])

    def test_toggling_coverage_on_adds_overlays(self):

        self.scene.set_show_camera_coverage(True)

        self.assertTrue(self.scene.show_camera_coverage)
        self.assertGreater(len(self.scene._coverage_overlay_items), 0)

        # Zone A is covered (camera stands in it); Zone B is not (no
        # door between them) -- expect at least one blind-spot tint
        # rect and at least one camera visibility polygon.
        self.assertTrue(
            any(isinstance(item, QGraphicsRectItem) for item in self.scene._coverage_overlay_items)
        )
        self.assertTrue(
            any(isinstance(item, QGraphicsPolygonItem) for item in self.scene._coverage_overlay_items)
        )

    def test_toggling_coverage_off_removes_overlays(self):

        self.scene.set_show_camera_coverage(True)
        self.assertGreater(len(self.scene._coverage_overlay_items), 0)

        self.scene.set_show_camera_coverage(False)

        self.assertEqual(self.scene._coverage_overlay_items, [])
        self.assertFalse(self.scene.show_camera_coverage)

    def test_overlay_items_are_not_selectable_or_mouse_interactive(self):

        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import QGraphicsItem

        self.scene.set_show_camera_coverage(True)

        for item in self.scene._coverage_overlay_items:

            self.assertFalse(
                item.flags() & QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            )
            self.assertEqual(item.acceptedMouseButtons(), Qt.MouseButton.NoButton)

    def test_rebuild_scene_recomputes_overlays_when_coverage_is_enabled(self):

        self.scene.set_show_camera_coverage(True)

        overlay_count_before = len(self.scene._coverage_overlay_items)
        self.assertGreater(overlay_count_before, 0)

        # rebuild_scene() (e.g. switching floors and back) must not
        # leave stale overlays or duplicate them.
        self.scene.rebuild_scene()

        self.assertEqual(len(self.scene._coverage_overlay_items), overlay_count_before)

    def test_refresh_camera_coverage_is_a_noop_when_disabled(self):

        self.scene.refresh_camera_coverage()

        self.assertEqual(self.scene._coverage_overlay_items, [])

    def test_moving_the_camera_and_refreshing_changes_the_overlay(self):

        self.scene.set_show_camera_coverage(True)

        camera_item = next(
            item for item in self.scene.items() if isinstance(item, CameraItem)
        )

        # Extract plain (x, y) floats immediately -- item.polygon()'s
        # QPointF elements are only guaranteed valid while their
        # owning QGraphicsPolygonItem is still alive; refresh_camera_
        # coverage() below destroys and replaces that item, so holding
        # onto the QPointF objects themselves (rather than plain
        # floats) risks comparing against freed/reused Qt memory.
        first_polygon_points = [
            tuple((p.x(), p.y()) for p in item.polygon())
            for item in self.scene._coverage_overlay_items
            if isinstance(item, QGraphicsPolygonItem)
        ]

        # Move the camera deep into Zone B instead, and confirm the
        # overlay is rebuilt (not just left stale) when told to
        # refresh -- mirrors what MainWindow.refresh_ui() does on every
        # drag via CameraItem.geometry_changed_callback.
        self.camera.position = (9.0, 2.5)
        self.scene.refresh_camera_coverage()

        second_polygon_points = [
            tuple((p.x(), p.y()) for p in item.polygon())
            for item in self.scene._coverage_overlay_items
            if isinstance(item, QGraphicsPolygonItem)
        ]

        self.assertNotEqual(first_polygon_points, second_polygon_points)


if __name__ == "__main__":
    unittest.main()
