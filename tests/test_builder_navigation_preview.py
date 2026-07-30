import sys
import unittest

from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

from builder.widgets.navigation_preview_panel import NavigationPreviewPanel

from models.building import Building
from models.door import Door
from models.exit import Exit
from models.zone import Zone


class NavigationPreviewPanelTests(unittest.TestCase):

    def setUp(self):

        self.panel = NavigationPreviewPanel()

    # =====================================================

    def test_none_building_shows_no_project_message(self):

        self.panel.refresh(None, None)

        self.assertIn("No project loaded", self.panel.info_label.text())
        self.assertEqual(len(self.panel.scene.items()), 0)

    # =====================================================

    def test_zone_with_exit_is_reachable(self):

        building = Building(name="B")
        floor = building.create_floor(name="Ground Floor")

        zone = Zone(id="Z-1", name="Zone 1", x=0, y=0, width=5, height=5, floor_id=floor.id)
        floor.add_zone(zone)

        floor.add_exit(Exit(
            id="E-1", name="Exit 1", start_point=(0, 0), end_point=(1, 0),
            floor_id=floor.id, zone_id=zone.id,
        ))

        self.panel.refresh(building, floor)

        self.assertIn("1 reachable", self.panel.info_label.text())

    # =====================================================

    def test_isolated_zone_is_not_reachable(self):

        building = Building(name="B")
        floor = building.create_floor(name="Ground Floor")

        zone = Zone(id="Z-1", name="Zone 1", x=0, y=0, width=5, height=5, floor_id=floor.id)
        floor.add_zone(zone)

        self.panel.refresh(building, floor)

        self.assertIn("1 space(s)", self.panel.info_label.text())
        self.assertIn("0 reachable", self.panel.info_label.text())

    # =====================================================

    def test_zones_connected_only_through_a_locked_door_are_unreachable(self):

        building = Building(name="B")
        floor = building.create_floor(name="Ground Floor")

        zone_a = Zone(id="Z-A", name="Zone A", x=0, y=0, width=5, height=5, floor_id=floor.id)
        zone_b = Zone(id="Z-B", name="Zone B", x=6, y=0, width=5, height=5, floor_id=floor.id)
        floor.add_zone(zone_a)
        floor.add_zone(zone_b)

        floor.add_exit(Exit(
            id="E-1", name="Exit 1", start_point=(0, 0), end_point=(1, 0),
            floor_id=floor.id, zone_id=zone_a.id,
        ))

        floor.add_door(Door(
            id="D-1", name="Door 1", start_point=(5, 2), end_point=(6, 2),
            floor_id=floor.id, zone_a_id=zone_a.id, zone_b_id=zone_b.id,
            locked=True,
        ))

        self.panel.refresh(building, floor)

        self.assertIn("2 space(s)", self.panel.info_label.text())
        self.assertIn("1 reachable", self.panel.info_label.text())

    # =====================================================

    def test_draws_a_scene_item_per_reachable_and_unreachable_node(self):

        building = Building(name="B")
        floor = building.create_floor(name="Ground Floor")

        zone = Zone(id="Z-1", name="Zone 1", x=0, y=0, width=5, height=5, floor_id=floor.id)
        floor.add_zone(zone)

        self.panel.refresh(building, floor)

        self.assertGreater(len(self.panel.scene.items()), 0)

    # =====================================================

    def test_stair_to_another_floor_is_drawn_as_a_stub_not_a_crash(self):

        from models.staircase import Staircase

        building = Building(name="B")
        floor_a = building.create_floor(name="Ground Floor")
        floor_b = building.create_floor(name="First Floor")

        zone_a = Zone(id="Z-A", name="Zone A", x=0, y=0, width=5, height=5, floor_id=floor_a.id)
        zone_b = Zone(id="Z-B", name="Zone B", x=0, y=0, width=5, height=5, floor_id=floor_b.id)
        floor_a.add_zone(zone_a)
        floor_b.add_zone(zone_b)

        floor_a.add_stair(Staircase(
            id="S-1", name="Stair 1", from_position=(2, 2), to_position=(1, 1),
            from_floor_id=floor_a.id, to_floor_id=floor_b.id,
            from_zone_id=zone_a.id, to_zone_id=zone_b.id,
        ))

        # Should not raise even though the Stair's other end (Zone B)
        # has no position on THIS floor.
        self.panel.refresh(building, floor_a)

        self.assertIn("1 space(s)", self.panel.info_label.text())
