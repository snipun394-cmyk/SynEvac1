import sys
import unittest

from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

from designer.widgets.camera_validation_panel import CameraValidationPanel

from models.building import Building
from models.camera import Camera
from models.exit import Exit
from models.zone import Zone


def make_zone(name, **kwargs):

    fields = dict(x=0.0, y=0.0, width=10.0, height=10.0)
    fields.update(kwargs)

    return Zone(name=name, **fields)


class CameraValidationPanelTests(unittest.TestCase):

    def setUp(self):

        self.building = Building(name="B")
        self.floor = self.building.create_floor(name="Ground Floor")

        self.zone = make_zone("Room", floor_id=self.floor.id)
        self.floor.add_zone(self.zone)

        self.floor.add_exit(Exit(name="Ex", zone_id=self.zone.id, floor_id=self.floor.id))

        self.good_camera = Camera(
            name="Good Cam", floor_id=self.floor.id, position=(0.5, 5.0),
            rotation=0.0, horizontal_fov=170.0, max_range=20.0, zone_ids=(self.zone.id,),
        )
        self.floor.add_camera(self.good_camera)

        self.panel = CameraValidationPanel()

    def test_refresh_with_no_building_shows_placeholder_state(self):

        self.panel.refresh(None)

        self.assertEqual(self.panel.network_score_label.text(), "-")
        self.assertEqual(self.panel.camera_table.rowCount(), 0)
        self.assertEqual(self.panel.recommendation_table.rowCount(), 0)

    def test_refresh_populates_the_camera_table(self):

        self.panel.refresh(self.building)

        self.assertEqual(self.panel.camera_table.rowCount(), 1)
        self.assertEqual(self.panel.camera_table.item(0, 0).text(), "Good Cam")
        self.assertEqual(self.panel.camera_table.item(0, 1).text(), "Ground Floor")

        placement_score = float(self.panel.camera_table.item(0, 2).text())
        self.assertGreater(placement_score, 0.0)

    def test_refresh_populates_the_network_score(self):

        self.panel.refresh(self.building)

        score_text = self.panel.network_score_label.text()
        self.assertNotEqual(score_text, "-")
        self.assertIn("/ 100", score_text)

    def test_poorly_placed_camera_produces_recommendations(self):

        bad_building = Building(name="Bad")
        bad_floor = bad_building.create_floor(name="Ground Floor")
        zone = make_zone("Room", floor_id=bad_floor.id)
        bad_floor.add_zone(zone)
        bad_floor.add_exit(Exit(name="Ex", zone_id=zone.id, floor_id=bad_floor.id))

        bad_camera = Camera(
            name="Bad Cam", floor_id=bad_floor.id, position=(0.5, 5.0),
            rotation=180.0, horizontal_fov=60.0, max_range=20.0, zone_ids=(zone.id,),
        )
        bad_floor.add_camera(bad_camera)

        self.panel.refresh(bad_building)

        self.assertGreater(self.panel.recommendation_table.rowCount(), 0)

    def test_run_validation_button_recomputes_after_a_camera_moves(self):

        self.panel.refresh(self.building)

        first_score = float(self.panel.camera_table.item(0, 2).text())

        self.good_camera.rotation = 180.0
        self.panel._run_validation()

        second_score = float(self.panel.camera_table.item(0, 2).text())

        self.assertLess(second_score, first_score)

    def test_multiple_floors_are_all_represented(self):

        floor_2 = self.building.create_floor(name="Floor 1", height=3.0)
        zone_2 = make_zone("Zone 2", floor_id=floor_2.id)
        floor_2.add_zone(zone_2)

        camera_2 = Camera(name="Cam 2", floor_id=floor_2.id)
        floor_2.add_camera(camera_2)

        self.panel.refresh(self.building)

        self.assertEqual(self.panel.camera_table.rowCount(), 2)

        floor_names = {
            self.panel.camera_table.item(row, 1).text()
            for row in range(self.panel.camera_table.rowCount())
        }
        self.assertEqual(floor_names, {"Ground Floor", "Floor 1"})


if __name__ == "__main__":
    unittest.main()
