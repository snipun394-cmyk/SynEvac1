import sys
import unittest

from PyQt6.QtCore import QPointF
from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

from designer.windows.main_window import MainWindow

from models.zone import Zone


# =====================================================
# Digital Twin Asset -> Zone Assignment & Live FACP Runtime milestone,
# Phase 17 -- automatic detector zone assignment on placement (Phase 4).
# =====================================================


class _FakeSceneMouseEvent:

    def __init__(self, x, y):
        self._pos = QPointF(x, y)

    def scenePos(self):
        return self._pos


class SmokeHeatAutoAssignmentTests(unittest.TestCase):

    def setUp(self):

        self.window = MainWindow()
        self.scene = self.window.canvas.scene_obj
        self.floor = self.scene.current_floor

        self.floor.add_zone(
            Zone(id="Z-INSIDE", name="Inside", floor_id=self.floor.id, x=0.0, y=0.0, width=10.0, height=10.0)
        )

    def test_smoke_detector_auto_assigned_inside_a_single_zone(self):

        self.window.toolbar.smoke_detector_action.trigger()
        # (5, 5) meters -> (250, 250) px at GRID_SIZE=50, inside Z-INSIDE (0..10, 0..10).
        self.scene.mousePressEvent(_FakeSceneMouseEvent(250, 250))

        self.assertEqual(self.floor.smoke_detector_count, 1)
        self.assertEqual(self.floor.smoke_detectors[0].zone_ids, ("Z-INSIDE",))

    def test_heat_detector_auto_assigned_inside_a_single_zone(self):

        self.window.toolbar.heat_detector_action.trigger()
        self.scene.mousePressEvent(_FakeSceneMouseEvent(250, 250))

        self.assertEqual(self.floor.heat_detector_count, 1)
        self.assertEqual(self.floor.heat_detectors[0].zone_ids, ("Z-INSIDE",))

    def test_smoke_detector_outside_every_zone_stays_unassigned(self):

        self.window.toolbar.smoke_detector_action.trigger()
        # (100, 100) meters -> far outside Z-INSIDE.
        self.scene.mousePressEvent(_FakeSceneMouseEvent(5000, 5000))

        self.assertEqual(self.floor.smoke_detector_count, 1)
        self.assertEqual(self.floor.smoke_detectors[0].zone_ids, ())

    def test_ambiguous_overlapping_zones_stay_unassigned(self):

        # A second zone fully overlapping Z-INSIDE -- the placement
        # point now falls inside BOTH, which must never be silently
        # resolved by picking whichever zone is first in the list.
        self.floor.add_zone(
            Zone(id="Z-OVERLAP", name="Overlap", floor_id=self.floor.id, x=0.0, y=0.0, width=10.0, height=10.0)
        )

        self.window.toolbar.smoke_detector_action.trigger()
        self.scene.mousePressEvent(_FakeSceneMouseEvent(250, 250))

        self.assertEqual(self.floor.smoke_detectors[0].zone_ids, ())

    def test_manual_reassignment_still_possible_after_auto_assignment(self):

        self.floor.add_zone(
            Zone(id="Z-OTHER", name="Other", floor_id=self.floor.id, x=20.0, y=0.0, width=10.0, height=10.0)
        )

        self.window.toolbar.smoke_detector_action.trigger()
        self.scene.mousePressEvent(_FakeSceneMouseEvent(250, 250))

        model = self.floor.smoke_detectors[0]
        self.assertEqual(model.zone_ids, ("Z-INSIDE",))

        # Manual override, exactly as Property Panel editing would do.
        model.zone_ids = ("Z-OTHER",)
        self.assertEqual(model.zone_ids, ("Z-OTHER",))


class SpeakerNeverAutoAssignedTests(unittest.TestCase):

    def test_speaker_placed_inside_a_zone_stays_unassigned(self):

        # Phase 4's own explicit "do not automatically assign Speaker
        # coverage based solely on position" requirement -- a speaker
        # mounted in one zone may legitimately serve a completely
        # different one, so position alone must never seed zone_ids.
        window = MainWindow()
        scene = window.canvas.scene_obj
        floor = scene.current_floor

        floor.add_zone(Zone(id="Z-INSIDE", name="Inside", floor_id=floor.id, x=0.0, y=0.0, width=10.0, height=10.0))

        window.toolbar.speaker_action.trigger()
        scene.mousePressEvent(_FakeSceneMouseEvent(250, 250))

        self.assertEqual(floor.speaker_count, 1)
        self.assertEqual(floor.speakers[0].zone_ids, ())


if __name__ == "__main__":
    unittest.main()
