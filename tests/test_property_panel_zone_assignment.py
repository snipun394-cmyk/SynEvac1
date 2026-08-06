import sys
import unittest

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

from designer.items.camera_item import CameraItem
from designer.items.heat_detector_item import HeatDetectorItem
from designer.items.smoke_detector_item import SmokeDetectorItem
from designer.items.speaker_item import SpeakerItem
from designer.windows.main_window import MainWindow

from models.camera import Camera
from models.heat_detector import HeatDetector
from models.smoke_detector import SmokeDetector
from models.speaker import Speaker
from models.zone import Zone


# =====================================================
# Digital Twin Asset -> Zone Assignment & Live FACP Runtime milestone,
# Phase 17 -- Property Panel zone-assignment UI tests.
# =====================================================


def _make_window_with_two_zones():

    window = MainWindow()
    floor = window.canvas.scene_obj.current_floor

    zone_a = Zone(id="Z-A", name="Zone A", floor_id=floor.id, x=0.0, y=0.0, width=10.0, height=10.0)
    zone_b = Zone(id="Z-B", name="Zone B", floor_id=floor.id, x=20.0, y=0.0, width=10.0, height=10.0)
    floor.add_zone(zone_a)
    floor.add_zone(zone_b)

    window.property_panel.building = window.canvas.scene_obj.project.building

    return window, floor, zone_a, zone_b


class SmokeDetectorZoneUITests(unittest.TestCase):

    def test_current_assignment_visible_on_selection(self):

        window, floor, zone_a, zone_b = _make_window_with_two_zones()

        model = SmokeDetector(id="SD-1", name="SD-1", floor_id=floor.id, zone_ids=("Z-A",))
        item = SmokeDetectorItem(0, 0, model=model)

        window.property_panel.show_smoke_detector(item)

        self.assertEqual(window.property_panel.smoke_detector_zone.itemData(
            window.property_panel.smoke_detector_zone.currentIndex()
        ), "Z-A")
        self.assertTrue(window.property_panel.smoke_detector_zone_warning.isHidden())

    def test_editable_and_persists_immediately_to_model(self):

        window, floor, zone_a, zone_b = _make_window_with_two_zones()

        model = SmokeDetector(id="SD-1", name="SD-1", floor_id=floor.id)
        item = SmokeDetectorItem(0, 0, model=model)

        window.property_panel.show_smoke_detector(item)

        index = window.property_panel.smoke_detector_zone.findData("Z-B")
        window.property_panel.smoke_detector_zone.setCurrentIndex(index)

        self.assertEqual(model.zone_ids, ("Z-B",))

    def test_warning_visible_when_unassigned(self):

        window, floor, zone_a, zone_b = _make_window_with_two_zones()

        model = SmokeDetector(id="SD-1", name="SD-1", floor_id=floor.id)
        item = SmokeDetectorItem(0, 0, model=model)

        window.property_panel.show_smoke_detector(item)

        self.assertFalse(window.property_panel.smoke_detector_zone_warning.isHidden())

    def test_clearing_assignment_restores_warning(self):

        window, floor, zone_a, zone_b = _make_window_with_two_zones()

        model = SmokeDetector(id="SD-1", name="SD-1", floor_id=floor.id, zone_ids=("Z-A",))
        item = SmokeDetectorItem(0, 0, model=model)

        window.property_panel.show_smoke_detector(item)
        self.assertTrue(window.property_panel.smoke_detector_zone_warning.isHidden())

        window.property_panel.smoke_detector_zone.setCurrentIndex(0)  # "None"

        self.assertEqual(model.zone_ids, ())
        self.assertFalse(window.property_panel.smoke_detector_zone_warning.isHidden())


class HeatDetectorZoneUITests(unittest.TestCase):

    def test_editable_and_persists_immediately_to_model(self):

        window, floor, zone_a, zone_b = _make_window_with_two_zones()

        model = HeatDetector(id="HD-1", name="HD-1", floor_id=floor.id)
        item = HeatDetectorItem(0, 0, model=model)

        window.property_panel.show_heat_detector(item)

        index = window.property_panel.heat_detector_zone.findData("Z-B")
        window.property_panel.heat_detector_zone.setCurrentIndex(index)

        self.assertEqual(model.zone_ids, ("Z-B",))

    def test_warning_visible_when_unassigned(self):

        window, floor, zone_a, zone_b = _make_window_with_two_zones()

        model = HeatDetector(id="HD-1", name="HD-1", floor_id=floor.id)
        item = HeatDetectorItem(0, 0, model=model)

        window.property_panel.show_heat_detector(item)

        self.assertFalse(window.property_panel.heat_detector_zone_warning.isHidden())


class SpeakerMultiZoneUITests(unittest.TestCase):

    def test_supports_multiple_checked_zones(self):

        window, floor, zone_a, zone_b = _make_window_with_two_zones()

        model = Speaker(id="SP-1", name="SP-1", floor_id=floor.id, zone_ids=("Z-A", "Z-B"))
        item = SpeakerItem(0, 0, model=model)

        window.property_panel.show_speaker(item)

        checked = {
            window.property_panel.speaker_zones.item(row).data(Qt.ItemDataRole.UserRole)
            for row in range(window.property_panel.speaker_zones.count())
            if window.property_panel.speaker_zones.item(row).checkState() == Qt.CheckState.Checked
        }
        self.assertEqual(checked, {"Z-A", "Z-B"})

    def test_checking_two_zones_persists_both_to_model(self):

        window, floor, zone_a, zone_b = _make_window_with_two_zones()

        model = Speaker(id="SP-1", name="SP-1", floor_id=floor.id)
        item = SpeakerItem(0, 0, model=model)

        window.property_panel.show_speaker(item)

        list_widget = window.property_panel.speaker_zones
        for row in range(list_widget.count()):
            list_widget.item(row).setCheckState(Qt.CheckState.Checked)

        self.assertEqual(set(model.zone_ids), {"Z-A", "Z-B"})

    def test_never_silently_reduced_to_a_single_zone(self):

        # Explicit regression for Phase 3's own "do not silently reduce
        # the model to one zone merely because a single QComboBox is
        # easier" requirement -- the widget must be a QListWidget
        # (multi-capable), never a QComboBox (single-select only).
        from PyQt6.QtWidgets import QComboBox, QListWidget

        window, floor, zone_a, zone_b = _make_window_with_two_zones()

        self.assertIsInstance(window.property_panel.speaker_zones, QListWidget)
        self.assertNotIsInstance(window.property_panel.speaker_zones, QComboBox)

    def test_warning_visible_when_no_zone_checked(self):

        window, floor, zone_a, zone_b = _make_window_with_two_zones()

        model = Speaker(id="SP-1", name="SP-1", floor_id=floor.id)
        item = SpeakerItem(0, 0, model=model)

        window.property_panel.show_speaker(item)

        self.assertFalse(window.property_panel.speaker_zone_warning.isHidden())

    def test_warning_hidden_once_one_zone_checked(self):

        window, floor, zone_a, zone_b = _make_window_with_two_zones()

        model = Speaker(id="SP-1", name="SP-1", floor_id=floor.id)
        item = SpeakerItem(0, 0, model=model)

        window.property_panel.show_speaker(item)

        window.property_panel.speaker_zones.item(0).setCheckState(Qt.CheckState.Checked)

        self.assertTrue(window.property_panel.speaker_zone_warning.isHidden())

    def test_unchecking_a_zone_removes_it(self):

        window, floor, zone_a, zone_b = _make_window_with_two_zones()

        model = Speaker(id="SP-1", name="SP-1", floor_id=floor.id, zone_ids=("Z-A", "Z-B"))
        item = SpeakerItem(0, 0, model=model)

        window.property_panel.show_speaker(item)

        list_widget = window.property_panel.speaker_zones
        for row in range(list_widget.count()):
            if list_widget.item(row).data(Qt.ItemDataRole.UserRole) == "Z-A":
                list_widget.item(row).setCheckState(Qt.CheckState.Unchecked)

        self.assertEqual(model.zone_ids, ("Z-B",))


class CameraMultiZoneUITests(unittest.TestCase):

    # Camera -> Zone Assignment milestone -- the exact same coverage
    # SpeakerMultiZoneUITests above already proves for Speaker.zone_ids,
    # now proved for Camera.zone_ids: a camera's field of view routinely
    # reaches more than one zone, so the widget must genuinely support
    # checking more than one, never silently cap at one the way the old
    # QComboBox did.

    def test_current_assignment_visible_on_selection(self):

        window, floor, zone_a, zone_b = _make_window_with_two_zones()

        model = Camera(id="CAM-1", name="CAM-1", floor_id=floor.id, zone_ids=("Z-A", "Z-B"))
        item = CameraItem(0, 0, model=model)

        window.property_panel.show_camera(item)

        checked = {
            window.property_panel.camera_zone.item(row).data(Qt.ItemDataRole.UserRole)
            for row in range(window.property_panel.camera_zone.count())
            if window.property_panel.camera_zone.item(row).checkState() == Qt.CheckState.Checked
        }
        self.assertEqual(checked, {"Z-A", "Z-B"})

    def test_checking_two_zones_persists_both_to_model(self):

        window, floor, zone_a, zone_b = _make_window_with_two_zones()

        model = Camera(id="CAM-1", name="CAM-1", floor_id=floor.id)
        item = CameraItem(0, 0, model=model)

        window.property_panel.show_camera(item)

        list_widget = window.property_panel.camera_zone
        for row in range(list_widget.count()):
            list_widget.item(row).setCheckState(Qt.CheckState.Checked)

        self.assertEqual(set(model.zone_ids), {"Z-A", "Z-B"})

    def test_never_silently_reduced_to_a_single_zone(self):

        # Explicit regression: the widget must be a QListWidget
        # (multi-capable), never a QComboBox (single-select only) --
        # the exact gap this milestone closes.
        from PyQt6.QtWidgets import QComboBox, QListWidget

        window, floor, zone_a, zone_b = _make_window_with_two_zones()

        self.assertIsInstance(window.property_panel.camera_zone, QListWidget)
        self.assertNotIsInstance(window.property_panel.camera_zone, QComboBox)

    def test_warning_visible_when_no_zone_checked(self):

        window, floor, zone_a, zone_b = _make_window_with_two_zones()

        model = Camera(id="CAM-1", name="CAM-1", floor_id=floor.id)
        item = CameraItem(0, 0, model=model)

        window.property_panel.show_camera(item)

        self.assertFalse(window.property_panel.camera_zone_warning.isHidden())

    def test_warning_hidden_once_one_zone_checked(self):

        window, floor, zone_a, zone_b = _make_window_with_two_zones()

        model = Camera(id="CAM-1", name="CAM-1", floor_id=floor.id)
        item = CameraItem(0, 0, model=model)

        window.property_panel.show_camera(item)

        window.property_panel.camera_zone.item(0).setCheckState(Qt.CheckState.Checked)

        self.assertTrue(window.property_panel.camera_zone_warning.isHidden())

    def test_unchecking_a_zone_removes_it(self):

        window, floor, zone_a, zone_b = _make_window_with_two_zones()

        model = Camera(id="CAM-1", name="CAM-1", floor_id=floor.id, zone_ids=("Z-A", "Z-B"))
        item = CameraItem(0, 0, model=model)

        window.property_panel.show_camera(item)

        list_widget = window.property_panel.camera_zone
        for row in range(list_widget.count()):
            if list_widget.item(row).data(Qt.ItemDataRole.UserRole) == "Z-A":
                list_widget.item(row).setCheckState(Qt.CheckState.Unchecked)

        self.assertEqual(model.zone_ids, ("Z-B",))

    def test_zero_hard_coded_zone_ids(self):

        # No zone appears as a checklist choice unless it genuinely
        # exists on the camera's own floor -- the checklist is always
        # populated from the real Building, never a fixed list.
        window, floor, zone_a, zone_b = _make_window_with_two_zones()

        model = Camera(id="CAM-1", name="CAM-1", floor_id=floor.id)
        item = CameraItem(0, 0, model=model)

        window.property_panel.show_camera(item)

        list_widget = window.property_panel.camera_zone
        offered = {list_widget.item(row).data(Qt.ItemDataRole.UserRole) for row in range(list_widget.count())}
        self.assertEqual(offered, {"Z-A", "Z-B"})


class DeletedZoneReferenceTests(unittest.TestCase):

    def test_deleted_zone_still_shown_but_not_selectable(self):

        # A zone referenced by an asset's zone_ids may later be deleted
        # from the floor -- the combo must never crash, and the stale
        # id simply no longer appears as a choice (findData returns -1,
        # _populate_zone_combo falls back to index 0 / "None").
        window, floor, zone_a, zone_b = _make_window_with_two_zones()

        model = SmokeDetector(id="SD-1", name="SD-1", floor_id=floor.id, zone_ids=("Z-GONE",))
        item = SmokeDetectorItem(0, 0, model=model)

        window.property_panel.show_smoke_detector(item)

        self.assertEqual(window.property_panel.smoke_detector_zone.currentIndex(), 0)


if __name__ == "__main__":
    unittest.main()
