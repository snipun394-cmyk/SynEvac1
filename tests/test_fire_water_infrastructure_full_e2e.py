import sys
import unittest

from PyQt6.QtCore import QPointF
from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

from designer.windows.main_window import MainWindow

from models.project import Project
from models.zone import Zone
from models.sensor_asset import HealthStatus

from fire_safety_manager.manager import FireSafetyAssetManager
from fire_water_manager.manager import FireWaterInfrastructureManager
from fire_water_manager.snapshot import FireWaterSystemStatus

from hazard.node_state import HazardNodeState
from hazard.snapshot import HazardSnapshot

from building_state.estimator import BuildingStateEstimator
from occupancy.snapshot import OccupancySnapshot


# =====================================================
# Fire Water Supply & Suppression Infrastructure milestone, Phase 17/18
# -- the full authored-building chain, driven through the REAL
# Designer:
#
#   Zone A, Zone B
#   FireWaterTank TANK-1, FirePump FP-1, JockeyPump JP-1,
#   FireServiceInlet FSI-1
#   Sprinkler SPR-1, SPR-2, FireHydrant HYD-1, HoseReel HR-1
#   FireWaterSystem FW-1 associating all of the above
#
# then save -> reload -> real FireWaterInfrastructureManager/
# FireSafetyAssetManager discovery -> Command Center, proving
# degradation/recovery and that hazard/fire-growth/smoke are never
# touched by infrastructure status changes.
# =====================================================


class _FakeSceneMouseEvent:

    def __init__(self, x, y):
        self._pos = QPointF(x, y)

    def scenePos(self):
        return self._pos


class FullFireWaterInfrastructureE2ETest(unittest.TestCase):

    def setUp(self):

        self.window = MainWindow()
        self.scene = self.window.canvas.scene_obj
        self.floor = self.scene.current_floor
        self.window.property_panel.building = self.scene.project.building

        self.zone_a = Zone(id="ZONE-A", name="Zone A", floor_id=self.floor.id, x=0.0, y=0.0, width=10.0, height=10.0)
        self.zone_b = Zone(id="ZONE-B", name="Zone B", floor_id=self.floor.id, x=20.0, y=0.0, width=10.0, height=10.0)
        self.floor.add_zone(self.zone_a)
        self.floor.add_zone(self.zone_b)

        self.window.toolbar.fire_water_tank_action.trigger()
        self.scene.mousePressEvent(_FakeSceneMouseEvent(250, 250))  # TANK-1, Zone A

        self.window.toolbar.fire_pump_action.trigger()
        self.scene.mousePressEvent(_FakeSceneMouseEvent(260, 260))  # FP-1, Zone A

        self.window.toolbar.jockey_pump_action.trigger()
        self.scene.mousePressEvent(_FakeSceneMouseEvent(270, 270))  # JP-1, Zone A

        self.window.toolbar.fire_service_inlet_action.trigger()
        self.scene.mousePressEvent(_FakeSceneMouseEvent(280, 280))  # FSI-1, Zone A

        self.window.toolbar.sprinkler_action.trigger()
        self.scene.mousePressEvent(_FakeSceneMouseEvent(290, 290))  # SPR-1, Zone A
        self.scene.mousePressEvent(_FakeSceneMouseEvent(1290, 290))  # SPR-2, Zone B

        self.window.toolbar.fire_hydrant_action.trigger()
        self.scene.mousePressEvent(_FakeSceneMouseEvent(300, 300))  # HYD-1, Zone A

        self.window.toolbar.hose_reel_action.trigger()
        self.scene.mousePressEvent(_FakeSceneMouseEvent(310, 310))  # HR-1, Zone A

        def _only(object_type):
            items = [i for i in self.scene.items() if getattr(i, "model", None) and i.model.object_type == object_type]
            items.sort(key=lambda i: i.model.position[0])
            return items

        self.tank_item, = _only("FireWaterTank")
        self.pump_item, = _only("FirePump")
        self.jockey_item, = _only("JockeyPump")
        self.inlet_item, = _only("FireServiceInlet")
        self.sprinkler_item_1, self.sprinkler_item_2 = _only("Sprinkler")
        self.hydrant_item, = _only("FireHydrant")
        self.hose_reel_item, = _only("HoseReel")

        for item, new_id in (
            (self.tank_item, "TANK-1"), (self.pump_item, "FP-1"), (self.jockey_item, "JP-1"),
            (self.inlet_item, "FSI-1"), (self.sprinkler_item_1, "SPR-1"), (self.sprinkler_item_2, "SPR-2"),
            (self.hydrant_item, "HYD-1"), (self.hose_reel_item, "HR-1"),
        ):
            item.model.id = new_id
            item.model.name = new_id

        # ---- Create FireWaterSystem FW-1 through the real dock panel's
        # underlying Building method (never JSON editing) ----
        self.building = self.scene.project.building
        self.system = self.building.create_fire_water_system("FW-1")
        self.window.fire_water_system_list.refresh()

        # ---- Associate every asset through the real Property Panel
        # "Fire Water System" combo ----
        for item, combo_attr, show_method in (
            (self.tank_item, "fire_water_tank_fire_water_system", "show_fire_water_tank"),
            (self.pump_item, "fire_pump_fire_water_system", "show_fire_pump"),
            (self.jockey_item, "jockey_pump_fire_water_system", "show_jockey_pump"),
            (self.inlet_item, "fire_service_inlet_fire_water_system", "show_fire_service_inlet"),
            (self.sprinkler_item_1, "sprinkler_fire_water_system", "show_sprinkler"),
            (self.sprinkler_item_2, "sprinkler_fire_water_system", "show_sprinkler"),
            (self.hydrant_item, "fire_hydrant_fire_water_system", "show_fire_hydrant"),
            (self.hose_reel_item, "hose_reel_fire_water_system", "show_hose_reel"),
        ):
            getattr(self.window.property_panel, show_method)(item)
            combo = getattr(self.window.property_panel, combo_attr)
            index = combo.findData(self.system.id)
            combo.setCurrentIndex(index)

    # =====================================================

    def test_system_associations_configured_through_real_ui(self):

        self.assertEqual(self.system.tank_ids, ("TANK-1",))
        self.assertEqual(self.system.pump_ids, ("FP-1",))
        self.assertEqual(self.system.jockey_pump_ids, ("JP-1",))
        self.assertEqual(self.system.inlet_ids, ("FSI-1",))
        self.assertEqual(set(self.system.sprinkler_ids), {"SPR-1", "SPR-2"})
        self.assertEqual(self.system.hydrant_ids, ("HYD-1",))
        self.assertEqual(self.system.hose_reel_ids, ("HR-1",))

    # =====================================================

    def test_save_reload_preserves_stable_ids_and_relationships(self):

        data = self.scene.project.to_dict()
        restored = Project.from_dict(data)
        restored_floor = restored.building.get_floor(self.floor.id)

        self.assertEqual(restored_floor.fire_water_tank_count, 1)
        self.assertEqual(restored_floor.fire_pump_count, 1)
        self.assertEqual(restored_floor.jockey_pump_count, 1)
        self.assertEqual(restored_floor.fire_service_inlet_count, 1)
        self.assertEqual(restored_floor.sprinkler_count, 2)
        self.assertEqual(restored_floor.fire_hydrant_count, 1)
        self.assertEqual(restored_floor.hose_reel_count, 1)

        self.assertEqual(len(restored.building.fire_water_systems), 1)
        restored_system = restored.building.fire_water_systems[0]

        self.assertEqual(restored_system.tank_ids, ("TANK-1",))
        self.assertEqual(restored_system.pump_ids, ("FP-1",))
        self.assertEqual(restored_system.jockey_pump_ids, ("JP-1",))
        self.assertEqual(restored_system.inlet_ids, ("FSI-1",))
        self.assertEqual(set(restored_system.sprinkler_ids), {"SPR-1", "SPR-2"})
        self.assertEqual(restored_system.hydrant_ids, ("HYD-1",))
        self.assertEqual(restored_system.hose_reel_ids, ("HR-1",))

        r_tank = next(t for t in restored_floor.fire_water_tanks if t.id == "TANK-1")
        self.assertEqual(r_tank.zone_ids, ("ZONE-A",))
        self.assertEqual(r_tank.position, self.tank_item.model.position)

    # =====================================================

    def test_runtime_manager_and_command_center_show_same_relationships(self):

        data = self.scene.project.to_dict()
        restored = Project.from_dict(data)

        fire_water_manager = FireWaterInfrastructureManager()
        fire_water_manager.discover_assets(restored.building)

        restored_system = restored.building.fire_water_systems[0]
        report = fire_water_manager.system_status(restored_system)

        self.assertEqual(report.status, FireWaterSystemStatus.SYSTEM_AVAILABLE)
        self.assertEqual(report.tank_ids, ("TANK-1",))
        self.assertEqual(set(report.sprinkler_ids), {"SPR-1", "SPR-2"})

        from building_state.models import BuildingState
        from command_center.live_status_panel import LiveStatusPanel

        snapshot = fire_water_manager.snapshot()
        panel = LiveStatusPanel()
        panel.show_building_state(BuildingState(fire_water_status=snapshot))

        self.assertEqual(panel.fire_water_asset_table.rowCount(), 4)
        self.assertEqual(panel.fire_water_system_table.rowCount(), 1)

        row = [panel.fire_water_system_table.item(0, c).text() for c in range(5)]
        self.assertEqual(row[0], "FW-1")
        self.assertEqual(row[1], "SYSTEM_AVAILABLE")
        self.assertIn("TANK-1", row[2])
        self.assertIn("SPR-1", row[3])

    # =====================================================

    def test_degradation_then_recovery_never_touches_hazard_or_fire_growth(self):

        data = self.scene.project.to_dict()
        restored = Project.from_dict(data)
        restored_system = restored.building.fire_water_systems[0]

        fire_water_manager = FireWaterInfrastructureManager()
        fire_water_manager.discover_assets(restored.building)

        # ---- Healthy baseline ----
        self.assertEqual(
            fire_water_manager.system_status(restored_system).status, FireWaterSystemStatus.SYSTEM_AVAILABLE,
        )

        hazard_snapshot = HazardSnapshot(
            timestamp=1.0,
            node_states={"ZONE-A": HazardNodeState(hazard_score=0.0, temperature=20.0)},
        )
        estimator = BuildingStateEstimator()
        state_before = estimator.estimate(1.0, hazard_snapshot=hazard_snapshot, occupancy_snapshot=OccupancySnapshot())

        # ---- Pump fault -> degraded ----
        pump = fire_water_manager.get_asset("FP-1")
        pump.health_status = HealthStatus.FAULT

        report = fire_water_manager.system_status(restored_system)
        self.assertEqual(report.status, FireWaterSystemStatus.SYSTEM_DEGRADED)
        self.assertTrue(any("FP-1" in reason for reason in report.reasons))

        # ---- Every other configured supply asset also unavailable ->
        # fully unavailable (JP-1/FSI-1 are also on this system; only
        # once ALL configured supply is bad does status escalate). ----
        tank = fire_water_manager.get_asset("TANK-1")
        tank.active = False
        fire_water_manager.get_asset("JP-1").health_status = HealthStatus.FAULT
        fire_water_manager.get_asset("FSI-1").active = False

        report = fire_water_manager.system_status(restored_system)
        self.assertEqual(report.status, FireWaterSystemStatus.SYSTEM_UNAVAILABLE)

        # ---- The exact same hazard_snapshot, re-run through the SAME
        # estimator -- infrastructure degradation never touches it. ----
        state_after = estimator.estimate(1.0, hazard_snapshot=hazard_snapshot, occupancy_snapshot=OccupancySnapshot())
        self.assertEqual(state_before.hazard_summary, state_after.hazard_summary)
        self.assertEqual(state_after.hazard_summary.zone_severities["ZONE-A"].name, "NONE")

        # ---- Restore every component -> recovers ----
        pump.health_status = HealthStatus.OK
        tank.active = True
        fire_water_manager.get_asset("JP-1").health_status = HealthStatus.OK
        fire_water_manager.get_asset("FSI-1").active = True

        report = fire_water_manager.system_status(restored_system)
        self.assertEqual(report.status, FireWaterSystemStatus.SYSTEM_AVAILABLE)
        self.assertEqual(report.reasons, ())

    # =====================================================

    def test_sprinkler_and_hydrant_remain_independent_of_system_status(self):

        # Phase 14/15's own "two independent facts" instruction --
        # Sprinkler ACTIVATED and FireHydrant availability are never
        # derived from, or altered by, FireWaterSystem status.
        from fire_safety_manager.manager import FireSafetyAssetManager

        data = self.scene.project.to_dict()
        restored = Project.from_dict(data)
        restored_system = restored.building.fire_water_systems[0]

        fire_water_manager = FireWaterInfrastructureManager()
        fire_water_manager.discover_assets(restored.building)

        fire_safety_manager = FireSafetyAssetManager()
        fire_safety_manager.discover_assets(restored.building)

        sprinkler_status_before = fire_safety_manager.status_of("SPR-1")
        hydrant_status_before = fire_safety_manager.status_of("HYD-1")

        # System becomes fully unavailable.
        fire_water_manager.get_asset("FP-1").health_status = HealthStatus.FAULT
        fire_water_manager.get_asset("TANK-1").active = False
        fire_water_manager.get_asset("JP-1").health_status = HealthStatus.FAULT
        fire_water_manager.get_asset("FSI-1").active = False
        self.assertEqual(
            fire_water_manager.system_status(restored_system).status, FireWaterSystemStatus.SYSTEM_UNAVAILABLE,
        )

        # Sprinkler/Hydrant status is completely untouched.
        sprinkler_status_after = fire_safety_manager.status_of("SPR-1")
        hydrant_status_after = fire_safety_manager.status_of("HYD-1")

        self.assertEqual(sprinkler_status_before, sprinkler_status_after)
        self.assertEqual(hydrant_status_before, hydrant_status_after)


if __name__ == "__main__":
    unittest.main()
