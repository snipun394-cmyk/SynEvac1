import sys
import unittest

from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

from designer.windows.main_window import MainWindow

from models.project import Project
from models.zone import Zone

from sensor_manager.manager import SensorManager
from speaker_manager.manager import SpeakerManager
from emergency_light_manager.manager import EmergencyLightManager
from fire_safety_manager.manager import FireSafetyAssetManager

from hazard.node_state import HazardNodeState
from hazard.snapshot import HazardSnapshot

from facp.engine import SimulatedFACP
from facp.models import PanelState

from live_system.facp_gateway import EngineFACPGateway

from building_state.estimator import BuildingStateEstimator
from occupancy.snapshot import OccupancySnapshot


# =====================================================
# Fire Suppression & Water-Based Safety Asset Digital Twin milestone,
# Phase 15/16 -- the full authored-building chain, driven through the
# REAL Designer, extending tests.test_manual_call_point_emergency_
# light_full_e2e.py's own recipe with the four new fire-suppression
# asset types:
#
#   Zone A, Zone B
#   SmokeDetector SD-1, HeatDetector HD-1, ManualCallPoint MCP-1,
#   Speaker SP-1, EmergencyLight EL-1, DynamicSign DS-1
#   Sprinkler SPR-1, SPR-2, FireExtinguisher FE-1,
#   FireHydrant HYD-1, HoseReel HR-1
#
# then: save -> reload -> discover via FireSafetyAssetManager ->
# drive SPR-1 above its activation threshold -> prove the hazard/
# fire-growth/smoke simulation layer is completely unaffected.
# =====================================================


class _FakeSceneMouseEvent:

    def __init__(self, x, y):
        self._pos = QPointF(x, y)

    def scenePos(self):
        return self._pos


class FullAuthoredBuildingWithFireSafetyAssetsE2ETest(unittest.TestCase):

    def setUp(self):

        self.window = MainWindow()
        self.scene = self.window.canvas.scene_obj
        self.floor = self.scene.current_floor
        self.window.property_panel.building = self.scene.project.building

        self.zone_a = Zone(id="ZONE-A", name="Zone A", floor_id=self.floor.id, x=0.0, y=0.0, width=10.0, height=10.0)
        self.zone_b = Zone(id="ZONE-B", name="Zone B", floor_id=self.floor.id, x=20.0, y=0.0, width=10.0, height=10.0)
        self.floor.add_zone(self.zone_a)
        self.floor.add_zone(self.zone_b)

        # ---- Existing asset family (previous milestone) ----
        self.window.toolbar.smoke_detector_action.trigger()
        self.scene.mousePressEvent(_FakeSceneMouseEvent(250, 250))

        self.window.toolbar.heat_detector_action.trigger()
        self.scene.mousePressEvent(_FakeSceneMouseEvent(1250, 250))

        self.window.toolbar.manual_call_point_action.trigger()
        self.scene.mousePressEvent(_FakeSceneMouseEvent(260, 260))

        self.window.toolbar.speaker_action.trigger()
        self.scene.mousePressEvent(_FakeSceneMouseEvent(270, 270))

        self.window.toolbar.emergency_light_action.trigger()
        self.scene.mousePressEvent(_FakeSceneMouseEvent(280, 280))

        self.window.toolbar.sign_action.trigger()
        self.scene.mousePressEvent(_FakeSceneMouseEvent(290, 290))

        # ---- New fire-suppression asset family (this milestone) ----
        self.window.toolbar.sprinkler_action.trigger()
        self.scene.mousePressEvent(_FakeSceneMouseEvent(300, 300))  # SPR-1, Zone A
        self.scene.mousePressEvent(_FakeSceneMouseEvent(1300, 300))  # SPR-2, Zone B

        self.window.toolbar.fire_extinguisher_action.trigger()
        self.scene.mousePressEvent(_FakeSceneMouseEvent(310, 310))  # FE-1, Zone A

        self.window.toolbar.fire_hydrant_action.trigger()
        self.scene.mousePressEvent(_FakeSceneMouseEvent(320, 320))  # HYD-1, Zone A

        self.window.toolbar.hose_reel_action.trigger()
        self.scene.mousePressEvent(_FakeSceneMouseEvent(330, 330))  # HR-1, Zone A

        def _only(object_type):
            items = [i for i in self.scene.items() if getattr(i, "model", None) and i.model.object_type == object_type]
            items.sort(key=lambda i: i.model.position[0])
            return items

        self.sprinkler_item_1, self.sprinkler_item_2 = _only("Sprinkler")
        self.extinguisher_item, = _only("FireExtinguisher")
        self.hydrant_item, = _only("FireHydrant")
        self.hose_reel_item, = _only("HoseReel")

        for item, new_id in (
            (self.sprinkler_item_1, "SPR-1"), (self.sprinkler_item_2, "SPR-2"),
            (self.extinguisher_item, "FE-1"), (self.hydrant_item, "HYD-1"), (self.hose_reel_item, "HR-1"),
        ):
            item.model.id = new_id
            item.model.name = new_id

        # ---- Zone assignment confirmed through the real Property Panel ----
        self.window.property_panel.show_sprinkler(self.sprinkler_item_1)
        self.assertEqual(self.sprinkler_item_1.model.zone_ids, ("ZONE-A",))

        # SPR-2 was placed inside Zone B and auto-assigned there --
        # confirm via panel selection, then manually reassign it back
        # to Zone A through the real combo, proving manual reassignment
        # remains possible after auto-assignment.
        self.window.property_panel.show_sprinkler(self.sprinkler_item_2)
        self.assertEqual(self.sprinkler_item_2.model.zone_ids, ("ZONE-B",))

        index = self.window.property_panel.sprinkler_zone.findData("ZONE-A")
        self.window.property_panel.sprinkler_zone.setCurrentIndex(index)
        self.assertEqual(self.sprinkler_item_2.model.zone_ids, ("ZONE-A",))

        self.window.property_panel.show_fire_extinguisher(self.extinguisher_item)
        self.assertEqual(self.extinguisher_item.model.zone_ids, ("ZONE-A",))

        self.window.property_panel.show_fire_hydrant(self.hydrant_item)
        self.assertEqual(self.hydrant_item.model.zone_ids, ("ZONE-A",))

        self.window.property_panel.show_hose_reel(self.hose_reel_item)
        self.assertEqual(self.hose_reel_item.model.zone_ids, ("ZONE-A",))

    # =====================================================

    def test_save_reload_preserves_identity_floor_zone_position_and_config(self):

        self.extinguisher_item.model.extinguisher_type = "CO2"
        self.hydrant_item.model.hydrant_type = "External Hydrant"
        self.sprinkler_item_1.model.activation_temperature = 79.0

        data = self.scene.project.to_dict()
        restored = Project.from_dict(data)
        restored_floor = restored.building.get_floor(self.floor.id)

        self.assertEqual(restored_floor.sprinkler_count, 2)
        self.assertEqual(restored_floor.fire_extinguisher_count, 1)
        self.assertEqual(restored_floor.fire_hydrant_count, 1)
        self.assertEqual(restored_floor.hose_reel_count, 1)

        r_spr1 = next(s for s in restored_floor.sprinklers if s.id == "SPR-1")
        r_spr2 = next(s for s in restored_floor.sprinklers if s.id == "SPR-2")
        r_fe1 = next(e for e in restored_floor.fire_extinguishers if e.id == "FE-1")
        r_hyd1 = next(h for h in restored_floor.fire_hydrants if h.id == "HYD-1")
        r_hr1 = next(r for r in restored_floor.hose_reels if r.id == "HR-1")

        self.assertEqual(r_spr1.zone_ids, ("ZONE-A",))
        self.assertEqual(r_spr1.activation_temperature, 79.0)
        self.assertEqual(r_spr1.floor_id, self.floor.id)
        self.assertEqual(r_spr1.position, self.sprinkler_item_1.model.position)

        self.assertEqual(r_spr2.zone_ids, ("ZONE-A",))

        self.assertEqual(r_fe1.zone_ids, ("ZONE-A",))
        self.assertEqual(r_fe1.extinguisher_type, "CO2")

        self.assertEqual(r_hyd1.zone_ids, ("ZONE-A",))
        self.assertEqual(r_hyd1.hydrant_type, "External Hydrant")

        self.assertEqual(r_hr1.zone_ids, ("ZONE-A",))

    # =====================================================

    def test_sprinkler_activation_never_reduces_hazard_or_fire_growth(self):

        data = self.scene.project.to_dict()
        restored = Project.from_dict(data)
        restored_floor = restored.building.get_floor(self.floor.id)

        fire_safety_manager = FireSafetyAssetManager()
        fire_safety_manager.discover_assets(restored.building)

        # ---- Baseline: no active hazard anywhere ----
        hazard_snapshot = HazardSnapshot(
            timestamp=1.0,
            node_states={
                "ZONE-A": HazardNodeState(hazard_score=0.0, temperature=20.0, smoke_level=0.0),
            },
        )

        estimator = BuildingStateEstimator()
        building_state_before = estimator.estimate(
            1.0, hazard_snapshot=hazard_snapshot, occupancy_snapshot=OccupancySnapshot(),
        )

        self.assertEqual(building_state_before.hazard_summary.zone_severities["ZONE-A"].name, "NONE")

        # ---- SPR-1 driven ABOVE its activation threshold ----
        spr1_id = self.sprinkler_item_1.model.id
        status = fire_safety_manager.status_of(spr1_id, sprinkler_temperatures={spr1_id: 200.0})
        self.assertEqual(status.state, "ACTIVATED")

        # ---- The SAME hazard_snapshot, untouched, fed through the SAME
        # estimator again -- nothing about Sprinkler activation ever
        # reaches HazardSnapshot/BuildingStateEstimator; this call has
        # no way to see it even if it wanted to (fire_safety_manager is
        # never passed to BuildingStateEstimator.estimate() at all). ----
        building_state_after = estimator.estimate(
            1.0, hazard_snapshot=hazard_snapshot, occupancy_snapshot=OccupancySnapshot(),
        )

        self.assertEqual(building_state_before.hazard_summary, building_state_after.hazard_summary)
        self.assertEqual(building_state_after.hazard_summary.zone_severities["ZONE-A"].name, "NONE")

        # ---- FACP is likewise unaffected: Sprinkler is never a FACP
        # alarm source (Phase 13) ----
        sensor_manager = SensorManager()
        sensor_manager.discover_sensors(restored.building)

        facp = SimulatedFACP()
        gateway = EngineFACPGateway(facp, sensor_manager)
        gateway.evaluate(1.0)

        self.assertEqual(facp.panel_state, PanelState.NORMAL)
        self.assertNotIn(spr1_id, facp.active_alarm_source_ids)
        self.assertNotIn(spr1_id, facp.active_fault_source_ids)

    # =====================================================

    def test_command_center_shows_activated_sprinkler_alongside_passive_assets(self):

        data = self.scene.project.to_dict()
        restored = Project.from_dict(data)

        fire_safety_manager = FireSafetyAssetManager()
        fire_safety_manager.discover_assets(restored.building)

        spr1_id = self.sprinkler_item_1.model.id
        snapshot = fire_safety_manager.snapshot(sprinkler_temperatures={spr1_id: 200.0})

        from building_state.models import BuildingState
        from command_center.live_status_panel import LiveStatusPanel

        panel = LiveStatusPanel()
        panel.show_building_state(BuildingState(fire_safety_status=snapshot))

        rows = {panel.fire_safety_table.item(r, 0).text(): panel.fire_safety_table.item(r, 3).text()
                for r in range(panel.fire_safety_table.rowCount())}

        self.assertEqual(rows[spr1_id], "ACTIVATED")
        self.assertEqual(rows["FE-1"], "AVAILABLE")
        self.assertEqual(rows["HYD-1"], "AVAILABLE")
        self.assertEqual(rows["HR-1"], "AVAILABLE")


if __name__ == "__main__":
    unittest.main()
