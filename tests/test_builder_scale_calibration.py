import os
import shutil
import sys
import tempfile
import unittest

from PyQt6.QtCore import QPointF
from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

from builder.scale_calibration import (
    ScaleCalibrationController, ScaleCalibrationError, compute_scale_pixels_per_meter,
)
from builder.windows.builder_main_window import BuilderMainWindow

from models.floor import Floor
from models.project import Project

from serialization.serializer import Serializer


class ComputeScaleTests(unittest.TestCase):

    def test_horizontal_calibration_line(self):

        pixels_per_meter = compute_scale_pixels_per_meter((0, 0), (100, 0), 5.0)

        self.assertAlmostEqual(pixels_per_meter, 20.0)

    def test_diagonal_calibration_line(self):

        pixels_per_meter = compute_scale_pixels_per_meter((0, 0), (30, 40), 5.0)

        self.assertAlmostEqual(pixels_per_meter, 10.0)

    def test_coincident_points_raise(self):

        with self.assertRaises(ScaleCalibrationError):
            compute_scale_pixels_per_meter((10, 10), (10, 10), 5.0)

    def test_zero_distance_raises(self):

        with self.assertRaises(ScaleCalibrationError):
            compute_scale_pixels_per_meter((0, 0), (100, 0), 0.0)

    def test_negative_distance_raises(self):

        with self.assertRaises(ScaleCalibrationError):
            compute_scale_pixels_per_meter((0, 0), (100, 0), -5.0)


class FloorCalibrationModelTests(unittest.TestCase):

    def test_new_floor_is_not_calibrated(self):

        floor = Floor(name="Ground Floor")

        self.assertFalse(floor.is_scale_calibrated)
        self.assertEqual(floor.floor_plan_scale, 0.0)

    def test_set_scale_calibration_marks_floor_calibrated(self):

        floor = Floor(name="Ground Floor")

        floor.set_scale_calibration((0, 0), (100, 0), 5.0, 20.0)

        self.assertTrue(floor.is_scale_calibrated)
        self.assertEqual(floor.floor_plan_scale, 20.0)
        self.assertEqual(floor.floor_plan_calibration_point_a, [0, 0])
        self.assertEqual(floor.floor_plan_calibration_point_b, [100, 0])
        self.assertEqual(floor.floor_plan_calibration_distance_m, 5.0)

    def test_calibration_round_trips_through_to_dict_from_dict(self):

        floor = Floor(name="Ground Floor")
        floor.set_scale_calibration((1, 2), (3, 4), 2.5, 8.0)

        reloaded = Floor.from_dict(floor.to_dict())

        self.assertTrue(reloaded.is_scale_calibrated)
        self.assertEqual(reloaded.floor_plan_scale, 8.0)
        self.assertEqual(reloaded.floor_plan_calibration_point_a, [1, 2])
        self.assertEqual(reloaded.floor_plan_calibration_point_b, [3, 4])
        self.assertEqual(reloaded.floor_plan_calibration_distance_m, 2.5)

    def test_legacy_floor_dict_without_calibration_fields_loads_uncalibrated(self):

        # Simulates a .syn file saved before this milestone -- no
        # floor_plan_scale key at all. Must not raise, must default to
        # "not calibrated", never a fabricated scale.
        legacy_data = Floor(name="Ground Floor").to_dict()

        for key in (
            "floor_plan_scale", "floor_plan_calibration_point_a",
            "floor_plan_calibration_point_b", "floor_plan_calibration_distance_m",
        ):
            legacy_data.pop(key, None)

        reloaded = Floor.from_dict(legacy_data)

        self.assertFalse(reloaded.is_scale_calibrated)
        self.assertIsNone(reloaded.floor_plan_calibration_point_a)


class GraphicsSceneFloorPlanScaleTests(unittest.TestCase):

    # designer/scene/graphics_scene.py's _display_floor_plan() is
    # shared with Studio -- this confirms the guarded scale-transform
    # addition behaves correctly both when calibrated and (the
    # existing, must-stay-identical behavior) when not.

    def setUp(self):

        self.window = BuilderMainWindow()

    # =====================================================

    def test_uncalibrated_floor_plan_renders_at_scale_one(self):

        image_path = self._make_test_image()

        try:
            self.window.canvas.load_floor_plan(image_path)

            item = self.window.canvas.scene_obj.floor_plan_item

            self.assertIsNotNone(item)
            self.assertAlmostEqual(item.scale(), 1.0)

        finally:
            os.remove(image_path)

    # =====================================================

    def test_calibrated_floor_plan_is_scaled_to_match_grid(self):

        image_path = self._make_test_image()

        try:
            self.window.canvas.load_floor_plan(image_path)

            floor = self.window.canvas.scene_obj.current_floor
            floor.set_scale_calibration((0, 0), (100, 0), 5.0, 20.0)

            self.window.canvas.scene_obj.rebuild_scene()

            item = self.window.canvas.scene_obj.floor_plan_item

            expected_scale = self.window.canvas.scene_obj.GRID_SIZE / 20.0

            self.assertAlmostEqual(item.scale(), expected_scale)

        finally:
            os.remove(image_path)

    # =====================================================

    def _make_test_image(self):

        from PyQt6.QtGui import QImage

        path = os.path.join(tempfile.gettempdir(), "builder_test_floor_plan.png")

        image = QImage(200, 100, QImage.Format.Format_RGB32)
        image.fill(0xFFFFFF)
        image.save(path)

        return path


class ScaleCalibrationControllerTests(unittest.TestCase):

    def setUp(self):

        self.window = BuilderMainWindow()
        self.controller = ScaleCalibrationController(self.window.canvas)

    # =====================================================

    def test_start_installs_event_filter_and_sets_cross_cursor(self):

        from PyQt6.QtCore import Qt

        self.assertFalse(self.controller.active)

        self.controller.start()

        self.assertTrue(self.controller.active)
        self.assertEqual(self.window.canvas.viewport().cursor().shape(), Qt.CursorShape.CrossCursor)

        self.controller.cancel()

    # =====================================================

    def test_two_clicks_report_points_and_stop_listening(self):

        from PyQt6.QtCore import QEvent
        from PyQt6.QtGui import QMouseEvent
        from PyQt6.QtCore import Qt as QtNS

        received = []

        self.controller.on_points_chosen = lambda a, b: received.append((a, b))

        self.controller.start()

        viewport = self.window.canvas.viewport()

        for point in (QPointF(10, 10), QPointF(50, 50)):

            event = QMouseEvent(
                QEvent.Type.MouseButtonPress,
                point,
                QtNS.MouseButton.LeftButton,
                QtNS.MouseButton.LeftButton,
                QtNS.KeyboardModifier.NoModifier,
            )

            self.controller.eventFilter(viewport, event)

        self.assertEqual(len(received), 1)
        self.assertFalse(self.controller.active)

    # =====================================================

    def test_cancel_before_second_click_fires_on_cancelled(self):

        cancelled = []

        self.controller.on_cancelled = lambda: cancelled.append(True)

        self.controller.start()
        self.controller.cancel()

        self.assertEqual(cancelled, [True])
        self.assertFalse(self.controller.active)


class BuilderMainWindowCalibrationFlowTests(unittest.TestCase):

    def setUp(self):

        self.window = BuilderMainWindow()
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # =====================================================

    def test_calibrate_scale_warns_when_no_floor_plan_imported(self):

        from unittest.mock import patch

        with patch("builder.windows.builder_main_window.QMessageBox.warning") as warning_mock:

            self.window.calibrate_scale()

            warning_mock.assert_called_once()

        self.assertFalse(self.window._calibration_controller.active)

    # =====================================================

    def test_on_calibration_points_chosen_updates_floor_and_info_bar(self):

        from unittest.mock import patch

        image_path = os.path.join(tempfile.gettempdir(), "builder_calib_flow.png")

        from PyQt6.QtGui import QImage

        QImage(200, 100, QImage.Format.Format_RGB32).save(image_path)

        try:

            self.window.canvas.load_floor_plan(image_path)

            floor_plan_item = self.window.canvas.scene_obj.floor_plan_item

            scene_point_a = floor_plan_item.mapToScene(QPointF(0, 0))
            scene_point_b = floor_plan_item.mapToScene(QPointF(100, 0))

            with patch(
                "builder.windows.builder_main_window.QInputDialog.getDouble",
                return_value=(5.0, True),
            ):

                self.window._on_calibration_points_chosen(scene_point_a, scene_point_b)

            floor = self.window.canvas.scene_obj.current_floor

            self.assertTrue(floor.is_scale_calibrated)
            self.assertAlmostEqual(floor.floor_plan_scale, 20.0)

            self.assertIn("px = 1 m", self.window.info_bar.scale_label.text())

        finally:

            os.remove(image_path)

    # =====================================================

    def test_calibration_persists_through_save_and_load(self):

        image_path = os.path.join(tempfile.gettempdir(), "builder_calib_persist.png")

        from PyQt6.QtGui import QImage

        QImage(200, 100, QImage.Format.Format_RGB32).save(image_path)

        try:

            self.window.canvas.load_floor_plan(image_path)

            floor = self.window.canvas.scene_obj.current_floor
            floor.set_scale_calibration((0, 0), (100, 0), 5.0, 20.0)

            path = os.path.join(self.tmpdir, "calibrated.syn")

            self.assertTrue(self.window._save_to(path))

            reloaded = Serializer.load(path)
            reloaded_floor = reloaded.building.ordered_floors()[0]

            self.assertTrue(reloaded_floor.is_scale_calibrated)
            self.assertAlmostEqual(reloaded_floor.floor_plan_scale, 20.0)

        finally:

            os.remove(image_path)
