import sys
import unittest

from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

from designer.items.fire_extinguisher_item import FireExtinguisherItem
from designer.items.fire_hydrant_item import FireHydrantItem
from designer.items.hose_reel_item import HoseReelItem
from designer.items.sprinkler_item import SprinklerItem
from designer.validation import validate_building_authoring
from designer.windows.main_window import MainWindow

from fire_safety_manager.manager import FireSafetyAssetManager

from models.building import Building
from models.fire_extinguisher import FireExtinguisher
from models.fire_hydrant import FireHydrant
from models.floor import Floor
from models.hose_reel import HoseReel
from models.project import Project
from models.smoke_detector import SmokeDetector
from models.sprinkler import Sprinkler, SprinklerActivationState
from models.zone import Zone


# =====================================================
# Fire Suppression & Water-Based Safety Asset Digital Twin milestone,
# Phase 17 -- failure/degradation cases genuinely not already covered
# by tests.test_fire_safety_asset_models / tests.test_fire_safety_
# asset_designer / tests.test_fire_safety_asset_manager (unassigned,
# outside-zone, ambiguous-zone, inactive, fault, threshold boundaries,
# and legacy-project-without-lists are already covered there).
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


class DeletedZoneReferenceTests(unittest.TestCase):

    def test_sprinkler_deleted_zone_still_shown_but_not_selectable(self):

        window, floor, zone_a, zone_b = _make_window_with_two_zones()

        model = Sprinkler(id="S1", name="S1", floor_id=floor.id, zone_ids=("Z-GONE",))
        item = SprinklerItem(0, 0, model=model)

        window.property_panel.show_sprinkler(item)

        self.assertEqual(window.property_panel.sprinkler_zone.currentIndex(), 0)

    def test_fire_extinguisher_deleted_zone_still_shown_but_not_selectable(self):

        window, floor, zone_a, zone_b = _make_window_with_two_zones()

        model = FireExtinguisher(id="E1", name="E1", floor_id=floor.id, zone_ids=("Z-GONE",))
        item = FireExtinguisherItem(0, 0, model=model)

        window.property_panel.show_fire_extinguisher(item)

        self.assertEqual(window.property_panel.fire_extinguisher_zone.currentIndex(), 0)

    def test_fire_hydrant_deleted_zone_still_shown_but_not_selectable(self):

        window, floor, zone_a, zone_b = _make_window_with_two_zones()

        model = FireHydrant(id="H1", name="H1", floor_id=floor.id, zone_ids=("Z-GONE",))
        item = FireHydrantItem(0, 0, model=model)

        window.property_panel.show_fire_hydrant(item)

        self.assertEqual(window.property_panel.fire_hydrant_zone.currentIndex(), 0)

    def test_hose_reel_deleted_zone_still_shown_but_not_selectable(self):

        window, floor, zone_a, zone_b = _make_window_with_two_zones()

        model = HoseReel(id="HR1", name="HR1", floor_id=floor.id, zone_ids=("Z-GONE",))
        item = HoseReelItem(0, 0, model=model)

        window.property_panel.show_hose_reel(item)

        self.assertEqual(window.property_panel.hose_reel_zone.currentIndex(), 0)

    def test_validation_never_crashes_once_the_referenced_zone_is_gone(self):

        window, floor, zone_a, zone_b = _make_window_with_two_zones()
        floor.remove_zone(zone_a)

        sprinkler = Sprinkler(id="S1", name="S1", floor_id=floor.id, zone_ids=("Z-A",))
        floor.add_sprinkler(sprinkler)

        report = validate_building_authoring(window.canvas.scene_obj.project.building)

        self.assertFalse(any(w.code == "sprinkler_missing_zone" for w in report.warnings))
        self.assertNotIn(sprinkler.zone_ids[0], {z.id for z in floor.zones})

    def test_sprinkler_activation_still_honestly_reported_with_deleted_zone(self):

        sprinkler = Sprinkler(id="S1", name="S1", floor_id="f1", zone_ids=("Z-GONE",))

        self.assertEqual(sprinkler.compute_state(200.0), SprinklerActivationState.ACTIVATED)

    def test_manager_reports_deleted_zone_reference_honestly_no_crash(self):

        floor = Floor(id="f1", name="F1")
        floor.add_zone(Zone(id="Z-A", name="Zone A", floor_id="f1"))
        sprinkler = Sprinkler(id="S1", name="S1", floor_id="f1", zone_ids=("Z-A",))
        floor.add_sprinkler(sprinkler)
        building = Building(id="b1", name="B", floors=[floor])

        floor.remove_zone(floor.zones[0])

        manager = FireSafetyAssetManager()
        manager.discover_assets(building)

        status = manager.status_of("S1")

        # Zone reference is neither dropped nor fabricated as valid --
        # it is reported exactly as stored, honestly.
        self.assertEqual(status.zone_ids, ("Z-A",))


class DuplicateIdTests(unittest.TestCase):

    def test_duplicate_id_across_floors_does_not_crash_discovery(self):

        floor1 = Floor(id="f1", name="F1")
        floor1.add_sprinkler(Sprinkler(id="DUP", name="First", floor_id="f1"))

        floor2 = Floor(id="f2", name="F2")
        floor2.add_sprinkler(Sprinkler(id="DUP", name="Second", floor_id="f2"))

        building = Building(id="b1", name="B", floors=[floor1, floor2])

        manager = FireSafetyAssetManager()
        manager.discover_assets(building)

        # No crash -- the last-discovered asset with that id wins (plain
        # dict-keyed registration, the same behavior every sibling
        # manager's own id-keyed registry already has).
        self.assertEqual(manager.get_asset("DUP").name, "Second")


class EmptyBuildingTests(unittest.TestCase):

    def test_empty_building_produces_empty_snapshot(self):

        building = Building(id="b1", name="B", floors=[Floor(id="f1", name="F1")])

        manager = FireSafetyAssetManager()
        manager.discover_assets(building)
        snapshot = manager.snapshot()

        self.assertEqual(snapshot.entries, ())
        self.assertEqual(snapshot.sprinklers.total, 0)
        self.assertEqual(snapshot.fire_extinguishers.total, 0)
        self.assertEqual(snapshot.fire_hydrants.total, 0)
        self.assertEqual(snapshot.hose_reels.total, 0)


class MixedOldAndNewProjectTests(unittest.TestCase):

    def test_save_reload_with_both_legacy_and_new_asset_types(self):

        floor = Floor(id="f1", name="F1")
        floor.add_zone(Zone(id="Z-A", name="Zone A", floor_id="f1"))
        floor.add_smoke_detector(SmokeDetector(id="SD-1", name="SD-1", floor_id="f1", zone_ids=("Z-A",)))
        floor.add_sprinkler(Sprinkler(id="SPR-1", name="SPR-1", floor_id="f1", zone_ids=("Z-A",)))
        floor.add_fire_extinguisher(FireExtinguisher(id="FE-1", name="FE-1", floor_id="f1", zone_ids=("Z-A",)))

        building = Building(id="b1", name="B", floors=[floor])
        project = Project(building=building)

        restored = Project.from_dict(project.to_dict())
        restored_floor = restored.building.get_floor("f1")

        self.assertEqual(restored_floor.smoke_detector_count, 1)
        self.assertEqual(restored_floor.sprinkler_count, 1)
        self.assertEqual(restored_floor.fire_extinguisher_count, 1)

        self.assertEqual(restored_floor.smoke_detectors[0].zone_ids, ("Z-A",))
        self.assertEqual(restored_floor.sprinklers[0].zone_ids, ("Z-A",))
        self.assertEqual(restored_floor.fire_extinguishers[0].zone_ids, ("Z-A",))


if __name__ == "__main__":
    unittest.main()
