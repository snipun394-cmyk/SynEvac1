import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import tempfile
import unittest

from PyQt6.QtWidgets import QApplication

from models.building import Building
from models.door import Door
from models.project import Project
from models.zone import Zone

from serialization.serializer import Serializer

# A QApplication is required before constructing any QGraphicsItem or
# QWidget (ZoneRectangle, DoorItem, PropertyPanel below) -- one is
# created once for the whole test process, same as every other Qt-
# backed test module in this repo.
_app = QApplication.instance() or QApplication([])

from designer.items.door_item import DoorItem
from designer.items.zone_rectangle import ZoneRectangle
from designer.widgets.property_panel import PropertyPanel


def make_zone(name, **kwargs):

    return Zone(
        name=name,
        x=0.0,
        y=0.0,
        width=2.0,
        height=2.0,
        **kwargs,
    )


class ZoneRectangleIdentityTests(unittest.TestCase):

    def test_model_passed_to_constructor_is_reflected_immediately(self):

        zone = make_zone("Lobby")

        rect = ZoneRectangle(0, 0, 100, 100, model=zone)

        self.assertEqual(rect.zone_id, zone.id)
        self.assertEqual(rect.zone_name, "Lobby")

    def test_model_assigned_after_construction_is_still_reflected(self):

        # Regression guard for the actual bug: both call sites in
        # GraphicsScene used to construct with model=None and set
        # .model afterward, which left zone_id/zone_name frozen at
        # "" forever because they were plain attributes copied only
        # once, at __init__ time.
        zone = make_zone("Lobby")

        rect = ZoneRectangle(0, 0, 100, 100)
        rect.model = zone

        self.assertEqual(rect.zone_id, zone.id)
        self.assertEqual(rect.zone_name, "Lobby")

    def test_rename_is_reflected_live(self):

        zone = make_zone("Lobby")
        rect = ZoneRectangle(0, 0, 100, 100, model=zone)

        rect.rename("Conference Room")

        self.assertEqual(zone.name, "Conference Room")
        self.assertEqual(rect.zone_name, "Conference Room")

    def test_no_model_yields_empty_strings_not_a_crash(self):

        rect = ZoneRectangle(0, 0, 100, 100)

        self.assertEqual(rect.zone_id, "")
        self.assertEqual(rect.zone_name, "")


class PropertyPanelZoneIdTests(unittest.TestCase):

    def test_zone_id_is_visible_and_copyable_in_property_panel(self):

        zone = make_zone("Lobby")

        # Constructed exactly as GraphicsScene now does (model=
        # passed directly), not patched in afterward.
        rect = ZoneRectangle(0, 0, 100, 100, model=zone)

        panel = PropertyPanel()
        panel.show_rectangle(rect)

        self.assertEqual(panel.object_id.text(), zone.id)
        self.assertNotEqual(panel.object_id.text(), "")

        panel.copy_object_id()

        self.assertEqual(
            QApplication.clipboard().text(),
            zone.id,
        )


class ZoneAutoNamingTests(unittest.TestCase):

    def test_first_zone_is_named_zone_1(self):

        building = Building(name="B")
        building.create_floor(name="Ground Floor")

        self.assertEqual(building.next_zone_name(), "Zone 1")

    def test_sequential_naming(self):

        building = Building(name="B")
        floor = building.create_floor(name="Ground Floor")

        floor.add_zone(make_zone(building.next_zone_name()))
        floor.add_zone(make_zone(building.next_zone_name()))
        floor.add_zone(make_zone(building.next_zone_name()))

        names = [zone.name for zone in floor.zones]

        self.assertEqual(names, ["Zone 1", "Zone 2", "Zone 3"])

    def test_naming_is_unique_across_floors_not_per_floor(self):

        building = Building(name="B")
        ground = building.create_floor(name="Ground Floor")
        floor1 = building.create_floor(name="Floor 1")

        ground.add_zone(make_zone(building.next_zone_name()))
        ground.add_zone(make_zone(building.next_zone_name()))

        # Without project-wide uniqueness this would come back as
        # "Zone 1" again (a second floor's zone_count is also 0).
        next_name = building.next_zone_name()
        floor1.add_zone(make_zone(next_name))

        self.assertEqual(next_name, "Zone 3")

        all_names = [z.name for z in ground.zones] + [
            z.name for z in floor1.zones
        ]

        self.assertEqual(len(all_names), len(set(all_names)))

    def test_naming_remains_stable_after_deleting_a_zone(self):

        building = Building(name="B")
        floor = building.create_floor(name="Ground Floor")

        z1 = make_zone(building.next_zone_name())
        floor.add_zone(z1)

        z2 = make_zone(building.next_zone_name())
        floor.add_zone(z2)

        z3 = make_zone(building.next_zone_name())
        floor.add_zone(z3)

        floor.remove_zone(z2)  # delete "Zone 2"

        # The next new zone must not silently reuse "Zone 2" while
        # "Zone 3" still exists in the project.
        self.assertEqual(building.next_zone_name(), "Zone 4")

    def test_naming_accounts_for_manually_renamed_zones(self):

        building = Building(name="B")
        floor = building.create_floor(name="Ground Floor")

        zone = make_zone("Zone 1")
        floor.add_zone(zone)

        zone.name = "Zone 10"  # user typed this in by hand

        self.assertEqual(building.next_zone_name(), "Zone 11")

    def test_naming_ignores_custom_non_matching_names(self):

        building = Building(name="B")
        floor = building.create_floor(name="Ground Floor")

        floor.add_zone(make_zone("Lobby"))
        floor.add_zone(make_zone("Conference Room"))

        self.assertEqual(building.next_zone_name(), "Zone 1")

    def test_naming_is_stable_after_save_load(self):

        project = Project.new_default()
        ground = project.building.ordered_floors()[0]

        ground.add_zone(make_zone(project.building.next_zone_name()))
        ground.add_zone(make_zone(project.building.next_zone_name()))

        with tempfile.TemporaryDirectory() as tmp_dir:

            path = os.path.join(tmp_dir, "roundtrip.syn")

            Serializer.save(project, path)
            reloaded = Serializer.load(path)

        self.assertEqual(
            [z.name for z in reloaded.building.ordered_floors()[0].zones],
            ["Zone 1", "Zone 2"],
        )

        self.assertEqual(
            reloaded.building.next_zone_name(),
            "Zone 3",
        )


class DoorDropdownUsabilityTests(unittest.TestCase):

    def setUp(self):

        self.building = Building(name="B")
        self.floor = self.building.create_floor(name="Ground Floor")

        self.zone_a = make_zone("Zone 1")
        self.zone_b = make_zone("Zone 2")

        self.floor.add_zone(self.zone_a)
        self.floor.add_zone(self.zone_b)

        self.door = Door(name="D1", floor_id=self.floor.id)
        self.floor.add_door(self.door)

        self.door_item = DoorItem(0, 0, 100, 0, model=self.door)

        self.panel = PropertyPanel()
        self.panel.set_building(self.building)

    def combo_entries(self, combo):

        return [
            (combo.itemText(i), combo.itemData(i))
            for i in range(combo.count())
        ]

    def test_dropdown_shows_names_not_uuids(self):

        self.panel.show_door(self.door_item)

        texts = [
            text
            for text, _ in self.combo_entries(self.panel.door_zone_a)
        ]

        self.assertIn("Zone 1", texts)
        self.assertIn("Zone 2", texts)

        for text in texts:
            self.assertNotEqual(text, self.zone_a.id)
            self.assertNotEqual(text, self.zone_b.id)

    def test_dropdown_data_still_carries_the_real_uuid(self):

        self.panel.show_door(self.door_item)

        entries = dict(self.combo_entries(self.panel.door_zone_a))

        self.assertEqual(entries["Zone 1"], self.zone_a.id)
        self.assertEqual(entries["Zone 2"], self.zone_b.id)

    def test_rename_is_reflected_in_dropdown_on_next_refresh(self):

        self.panel.show_door(self.door_item)

        self.zone_a.name = "Conference Room"

        self.panel.show_door(self.door_item)

        texts = [
            text
            for text, _ in self.combo_entries(self.panel.door_zone_a)
        ]

        self.assertIn("Conference Room", texts)
        self.assertNotIn("Zone 1", texts)


if __name__ == "__main__":
    unittest.main()
