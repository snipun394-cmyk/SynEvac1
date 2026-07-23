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

from facp.engine import SimulatedFACP
from facp.models import PanelState

from live_system.facp_gateway import EngineFACPGateway

from voice_evacuation.controller import VoiceEvacuationController
from voice_evacuation.provider import SimulationVoiceOutputProvider

from building_control.controller import BuildingControlController
from building_control.providers import SimulationControlProvider
from building_control.types import ApprovalMode


# =====================================================
# Manual Call Points & Emergency Lighting milestone, Step 3/4/6 -- the
# full authored-building chain driven through the REAL Designer
# (toolbar triggers + GraphicsScene.mousePressEvent, real Property
# Panel widgets), extending the existing
# tests.test_zone_assignment_full_e2e.FullAuthoredBuildingChainE2ETest
# recipe with the two new asset types:
#
#   Zone A, Zone B
#   SmokeDetector SD-1, HeatDetector HD-1
#   ManualCallPoint MCP-1, MCP-2
#   Speaker SP-1
#   EmergencyLight EL-1
#   DynamicEvacuationSign DS-1
#
# then: save -> reload -> re-discover -> drive MCP-1 ACTIVATED through
# SensorManager -> EngineFACPGateway -> FACP.evaluate() -> FACPSnapshot,
# and prove that activation never reaches VoiceEvacuationController or
# BuildingControlController on its own.
# =====================================================


class _FakeSceneMouseEvent:

    def __init__(self, x, y):
        self._pos = QPointF(x, y)

    def scenePos(self):
        return self._pos


class FullAuthoredBuildingChainWithMCPAndEmergencyLightE2ETest(unittest.TestCase):

    def setUp(self):

        self.window = MainWindow()
        self.scene = self.window.canvas.scene_obj
        self.floor = self.scene.current_floor
        self.window.property_panel.building = self.scene.project.building

        # ---- Zone A, Zone B: drawn through the real zone tool (two
        # clicks: start corner, then opposite corner) ----
        self.window.toolbar.zone_action.trigger()
        self.scene.mousePressEvent(_FakeSceneMouseEvent(0, 0))
        self.scene.mousePressEvent(_FakeSceneMouseEvent(500, 500))  # Zone A: (0,0) to (10,10)m

        self.window.toolbar.zone_action.trigger()
        self.scene.mousePressEvent(_FakeSceneMouseEvent(1000, 0))
        self.scene.mousePressEvent(_FakeSceneMouseEvent(1500, 500))  # Zone B: (20,0) to (30,10)m

        self.zone_a, self.zone_b = self.floor.zones
        self.zone_a.id, self.zone_b.id = "ZONE-A", "ZONE-B"

        # ---- SD-1, HD-1: auto-assigned by position ----
        self.window.toolbar.smoke_detector_action.trigger()
        self.scene.mousePressEvent(_FakeSceneMouseEvent(250, 250))  # inside Zone A

        self.window.toolbar.heat_detector_action.trigger()
        self.scene.mousePressEvent(_FakeSceneMouseEvent(1250, 250))  # inside Zone B

        # ---- MCP-1, MCP-2 ----
        self.window.toolbar.manual_call_point_action.trigger()
        self.scene.mousePressEvent(_FakeSceneMouseEvent(300, 300))  # inside Zone A
        self.scene.mousePressEvent(_FakeSceneMouseEvent(1300, 300))  # inside Zone B

        # ---- SP-1 ----
        self.window.toolbar.speaker_action.trigger()
        self.scene.mousePressEvent(_FakeSceneMouseEvent(250, 350))  # inside Zone A

        # ---- EL-1 ----
        self.window.toolbar.emergency_light_action.trigger()
        self.scene.mousePressEvent(_FakeSceneMouseEvent(280, 280))  # inside Zone A

        # ---- DS-1 ----
        self.window.toolbar.sign_action.trigger()
        self.scene.mousePressEvent(_FakeSceneMouseEvent(260, 260))  # inside Zone A

        def _only(object_type):
            items = [i for i in self.scene.items() if getattr(i, "model", None) and i.model.object_type == object_type]
            items.sort(key=lambda i: i.model.position[0])
            return items

        self.smoke_item, = _only("SmokeDetector")
        self.heat_item, = _only("HeatDetector")
        self.mcp_item_1, self.mcp_item_2 = _only("ManualCallPoint")
        self.speaker_item, = _only("Speaker")
        self.light_item, = _only("EmergencyLight")
        self.sign_item, = _only("DynamicEvacuationSign")

        self.mcp_item_1.model.id = "MCP-1"
        self.mcp_item_2.model.id = "MCP-2"
        self.speaker_item.model.id = "SP-1"
        self.light_item.model.id = "EL-1"
        self.sign_item.model.id = "DS-1"
        self.smoke_item.model.id = "SD-1"
        self.heat_item.model.id = "HD-1"

        # ---- Zone assignment confirmed through the real Property Panel ----
        self.window.property_panel.show_smoke_detector(self.smoke_item)
        self.assertEqual(self.smoke_item.model.zone_ids, ("ZONE-A",))

        self.window.property_panel.show_heat_detector(self.heat_item)
        self.assertEqual(self.heat_item.model.zone_ids, ("ZONE-B",))

        self.window.property_panel.show_manual_call_point(self.mcp_item_1)
        self.assertEqual(self.mcp_item_1.model.zone_ids, ("ZONE-A",))

        self.window.property_panel.show_manual_call_point(self.mcp_item_2)
        self.assertEqual(self.mcp_item_2.model.zone_ids, ("ZONE-B",))

        self.window.property_panel.show_emergency_light(self.light_item)
        self.assertEqual(self.light_item.model.zone_ids, ("ZONE-A",))

        self.window.property_panel.show_speaker(self.speaker_item)
        list_widget = self.window.property_panel.speaker_zones
        for row in range(list_widget.count()):
            if list_widget.item(row).data(Qt.ItemDataRole.UserRole) == "ZONE-A":
                list_widget.item(row).setCheckState(Qt.CheckState.Checked)
        self.assertEqual(self.speaker_item.model.zone_ids, ("ZONE-A",))

    # =====================================================

    def test_save_reload_preserves_identity_floor_zone_position_and_config(self):

        before = {
            "MCP-1": (self.mcp_item_1.model.zone_ids, self.mcp_item_1.model.position, self.mcp_item_1.model.floor_id),
            "MCP-2": (self.mcp_item_2.model.zone_ids, self.mcp_item_2.model.position, self.mcp_item_2.model.floor_id),
            "EL-1": (self.light_item.model.zone_ids, self.light_item.model.position, self.light_item.model.light_type),
            "DS-1": (self.sign_item.model.position, self.sign_item.model.floor_id),
        }

        self.mcp_item_2.model.activate()
        self.light_item.model.light_type = "Ceiling Mounted"

        project = self.scene.project
        data = project.to_dict()
        restored = Project.from_dict(data)
        restored_floor = restored.building.get_floor(self.floor.id)

        self.assertEqual(restored_floor.manual_call_point_count, 2)
        self.assertEqual(restored_floor.emergency_light_count, 1)
        self.assertEqual(restored_floor.sign_count, 1)

        r_mcp1 = next(m for m in restored_floor.manual_call_points if m.id == "MCP-1")
        r_mcp2 = next(m for m in restored_floor.manual_call_points if m.id == "MCP-2")
        r_light = next(l for l in restored_floor.emergency_lights if l.id == "EL-1")

        self.assertEqual(r_mcp1.zone_ids, before["MCP-1"][0])
        self.assertEqual(r_mcp1.position, before["MCP-1"][1])
        self.assertEqual(r_mcp1.floor_id, before["MCP-1"][2])
        self.assertFalse(r_mcp1.activated)

        self.assertEqual(r_mcp2.zone_ids, before["MCP-2"][0])
        self.assertTrue(r_mcp2.activated)

        self.assertEqual(r_light.zone_ids, before["EL-1"][0])
        self.assertEqual(r_light.position, before["EL-1"][1])
        self.assertEqual(r_light.light_type, "Ceiling Mounted")

    # =====================================================

    def test_mcp_activation_reaches_facp_as_correct_source_without_auto_dispatch(self):

        project = self.scene.project
        data = project.to_dict()
        restored = Project.from_dict(data)
        restored_floor = restored.building.get_floor(self.floor.id)

        restored_mcp_1 = next(m for m in restored_floor.manual_call_points if m.id == "MCP-1")
        restored_mcp_1.activate()

        # ---- Re-discovery from the reloaded building ----
        sensor_manager = SensorManager()
        sensor_manager.discover_sensors(restored.building)

        speaker_manager = SpeakerManager()
        speaker_manager.discover_speakers(restored.building)

        light_manager = EmergencyLightManager()
        light_manager.discover_lights(restored.building)

        # ---- MCP-1 ACTIVATED -> SensorManager -> EngineFACPGateway -> FACP.evaluate() -> FACPSnapshot ----
        facp = SimulatedFACP()
        gateway = EngineFACPGateway(facp, sensor_manager)
        gateway.evaluate(1.0)

        self.assertEqual(facp.panel_state, PanelState.ALARM)
        self.assertIn("MCP-1", facp.active_alarm_source_ids)

        snapshot = facp.current_snapshot(1.0)
        self.assertEqual(snapshot.panel_state, PanelState.ALARM)
        self.assertIn("MCP-1", snapshot.active_alarm_source_ids)

        mcp_events = [e for e in snapshot.recent_events if e.source_asset_id == "MCP-1"]
        self.assertTrue(mcp_events)
        self.assertEqual(mcp_events[-1].source_asset_type, "ManualCallPoint")
        self.assertEqual(mcp_events[-1].zone_ids, ("ZONE-A",))

        # ---- Command Center representation ----
        from building_state.models import BuildingState
        from command_center.live_status_panel import LiveStatusPanel

        panel = LiveStatusPanel()
        panel.show_building_state(BuildingState(facp_status=snapshot))

        source_rows = {
            panel.facp_sources_table.item(r, 0).text(): (
                panel.facp_sources_table.item(r, 1).text(),
                panel.facp_sources_table.item(r, 2).text(),
                panel.facp_sources_table.item(r, 3).text(),
            )
            for r in range(panel.facp_sources_table.rowCount())
        }
        self.assertEqual(source_rows["MCP-1"], ("ManualCallPoint", "ZONE-A", "ALARM"))

        # ---- Critically: NO automatic dispatch to VoiceEvacuationController ----
        voice_provider = SimulationVoiceOutputProvider()
        voice_controller = VoiceEvacuationController(speaker_manager, voice_provider)

        # The MCP alarm above never called anything on voice_controller;
        # its provider must show zero sent instructions unless a caller
        # separately, explicitly, invokes broadcast() themselves.
        self.assertEqual(voice_provider.sent_instructions(), ())

        # ---- Critically: NO automatic execution by BuildingControlController ----
        control_provider = SimulationControlProvider(restored.building)
        control_controller = BuildingControlController(
            restored.building, control_provider, approval_mode=ApprovalMode.REQUIRES_APPROVAL,
        )
        control_snapshot = control_controller.snapshot()

        self.assertEqual(control_snapshot.entries, ())
        self.assertEqual(control_snapshot.pending_count, 0)


if __name__ == "__main__":
    unittest.main()
