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

from facp.engine import SimulatedFACP
from facp.models import DetectorConditionReport, PanelState

from perception.models.smoke_detector_observation import SmokeDetectorReading
from perception.models.heat_detector_observation import HeatDetectorReading

from voice_evacuation.controller import VoiceEvacuationController
from voice_evacuation.models import VoiceMessage
from voice_evacuation.provider import SimulationVoiceOutputProvider


# =====================================================
# Digital Twin Asset -> Zone Assignment & Live FACP Runtime milestone,
# Phase 6/7/8/13 -- the full authored-building chain, driven through
# the REAL Designer (toolbar triggers + GraphicsScene.mousePressEvent,
# real Property Panel widgets), never hand-constructed fixtures:
#
#   1. Create two zones.
#   2. Place SmokeDetector, HeatDetector, two Speakers.
#   3. Configure zone assignment through the Property Panel.
#   4. Save the project (Project.to_dict()).
#   5. Reload it (Project.from_dict()).
#   6. Discover the RELOADED assets via SensorManager/SpeakerManager.
#   7. Drive detector conditions -> FACP.evaluate() -> BuildingState-shaped snapshot.
#   8. Send a zone-targeted voice message through VoiceEvacuationController
#      -> SimulationVoiceOutputProvider, proving only the correctly-
#      assigned speaker(s) receive it.
# =====================================================


class _FakeSceneMouseEvent:

    def __init__(self, x, y):
        self._pos = QPointF(x, y)

    def scenePos(self):
        return self._pos


class FullAuthoredBuildingChainE2ETest(unittest.TestCase):

    def setUp(self):

        self.window = MainWindow()
        self.scene = self.window.canvas.scene_obj
        self.floor = self.scene.current_floor
        self.window.property_panel.building = self.scene.project.building

        # ---- Step 1: two zones ----
        self.zone_a = Zone(id="ZONE-A", name="Zone A", floor_id=self.floor.id, x=0.0, y=0.0, width=10.0, height=10.0)
        self.zone_b = Zone(id="ZONE-B", name="Zone B", floor_id=self.floor.id, x=20.0, y=0.0, width=10.0, height=10.0)
        self.floor.add_zone(self.zone_a)
        self.floor.add_zone(self.zone_b)

        # ---- Step 2: SmokeDetector + HeatDetector (auto-assigned by
        # position, per Phase 4), two Speakers (never auto-assigned) ----
        self.window.toolbar.smoke_detector_action.trigger()
        self.scene.mousePressEvent(_FakeSceneMouseEvent(250, 250))  # (5,5)m -> inside Zone A

        self.window.toolbar.heat_detector_action.trigger()
        self.scene.mousePressEvent(_FakeSceneMouseEvent(1250, 250))  # (25,5)m -> inside Zone B

        self.window.toolbar.speaker_action.trigger()
        self.scene.mousePressEvent(_FakeSceneMouseEvent(250, 300))  # SP-1, near Zone A
        self.scene.mousePressEvent(_FakeSceneMouseEvent(1250, 300))  # SP-2, near Zone B

        self.smoke_item = next(i for i in self.scene.items() if getattr(i, "model", None) and i.model.object_type == "SmokeDetector")
        self.heat_item = next(i for i in self.scene.items() if getattr(i, "model", None) and i.model.object_type == "HeatDetector")
        speaker_items = [i for i in self.scene.items() if getattr(i, "model", None) and i.model.object_type == "Speaker"]
        speaker_items.sort(key=lambda i: i.model.position[0])
        self.speaker_item_a, self.speaker_item_b = speaker_items

        # ---- Step 3: zone assignment through the real Property Panel ----
        # SD-1/HD-1 already auto-assigned -- confirm via panel selection.
        self.window.property_panel.show_smoke_detector(self.smoke_item)
        self.assertEqual(self.smoke_item.model.zone_ids, ("ZONE-A",))

        self.window.property_panel.show_heat_detector(self.heat_item)
        self.assertEqual(self.heat_item.model.zone_ids, ("ZONE-B",))

        # SP-1 -> Zone A only.
        self.window.property_panel.show_speaker(self.speaker_item_a)
        list_widget = self.window.property_panel.speaker_zones
        for row in range(list_widget.count()):
            if list_widget.item(row).data(Qt.ItemDataRole.UserRole) == "ZONE-A":
                list_widget.item(row).setCheckState(Qt.CheckState.Checked)

        # SP-2 -> BOTH Zone A and Zone B (multi-zone coverage).
        self.window.property_panel.show_speaker(self.speaker_item_b)
        list_widget = self.window.property_panel.speaker_zones
        for row in range(list_widget.count()):
            list_widget.item(row).setCheckState(Qt.CheckState.Checked)

        self.assertEqual(self.speaker_item_a.model.zone_ids, ("ZONE-A",))
        self.assertEqual(set(self.speaker_item_b.model.zone_ids), {"ZONE-A", "ZONE-B"})

    # =====================================================

    def test_full_chain(self):

        smoke_id = self.smoke_item.model.id
        heat_id = self.heat_item.model.id
        speaker_a_id = self.speaker_item_a.model.id
        speaker_b_id = self.speaker_item_b.model.id

        # ---- Step 4/5: save + reload ----
        project = self.scene.project
        data = project.to_dict()
        restored = Project.from_dict(data)
        restored_floor = restored.building.get_floor(self.floor.id)

        self.assertEqual(restored_floor.smoke_detector_count, 1)
        self.assertEqual(restored_floor.heat_detector_count, 1)
        self.assertEqual(restored_floor.speaker_count, 2)

        restored_smoke = next(s for s in restored_floor.smoke_detectors if s.id == smoke_id)
        restored_heat = next(h for h in restored_floor.heat_detectors if h.id == heat_id)
        restored_speaker_a = next(s for s in restored_floor.speakers if s.id == speaker_a_id)
        restored_speaker_b = next(s for s in restored_floor.speakers if s.id == speaker_b_id)

        self.assertEqual(restored_smoke.zone_ids, ("ZONE-A",))
        self.assertEqual(restored_heat.zone_ids, ("ZONE-B",))
        self.assertEqual(restored_speaker_a.zone_ids, ("ZONE-A",))
        self.assertEqual(set(restored_speaker_b.zone_ids), {"ZONE-A", "ZONE-B"})

        # ---- Step 6: discovery from the RELOADED building ----
        sensor_manager = SensorManager()
        sensor_manager.discover_sensors(restored.building)

        speaker_manager = SpeakerManager()
        speaker_manager.discover_speakers(restored.building)

        self.assertEqual({s.id for s in sensor_manager.sensors_in_zone("ZONE-A")}, {smoke_id})
        self.assertEqual({s.id for s in sensor_manager.sensors_in_zone("ZONE-B")}, {heat_id})

        self.assertEqual({s.id for s in speaker_manager.speakers_in_zone("ZONE-A")}, {speaker_a_id, speaker_b_id})
        self.assertEqual({s.id for s in speaker_manager.speakers_in_zone("ZONE-B")}, {speaker_b_id})

        # ---- Step 7: drive detector conditions -> FACP -> snapshot ----
        smoke_status = next(s for s in sensor_manager.all_statuses() if s.sensor_id == smoke_id)
        heat_status = next(s for s in sensor_manager.all_statuses() if s.sensor_id == heat_id)

        smoke_reading = SmokeDetectorReading(detector_id=smoke_id, timestamp=1.0, alarm_active=True)
        heat_reading = HeatDetectorReading(detector_id=heat_id, timestamp=1.0, alarm_active=False)

        conditions = {
            smoke_id: DetectorConditionReport.from_status_and_reading(smoke_status, smoke_reading),
            heat_id: DetectorConditionReport.from_status_and_reading(heat_status, heat_reading),
        }

        facp = SimulatedFACP()
        facp.evaluate(conditions, 1.0)

        self.assertEqual(facp.panel_state, PanelState.ALARM)
        self.assertIn(smoke_id, facp.active_alarm_source_ids)
        self.assertNotIn(heat_id, facp.active_alarm_source_ids)

        snapshot = facp.current_snapshot(1.0)
        self.assertEqual(snapshot.panel_state, PanelState.ALARM)

        # Zone consistency now works for Designer-authored detectors --
        # the alarm source's own zone (via SensorStatus.zone_ids, IDs
        # preserved through the whole save/reload/discovery chain) is
        # Zone A, exactly where SD-1 was placed and assigned.
        self.assertEqual(smoke_status.zone_ids, ("ZONE-A",))

        # ---- Step 8: zone-targeted voice message ----
        voice_provider = SimulationVoiceOutputProvider()
        voice_controller = VoiceEvacuationController(speaker_manager, voice_provider)

        message_a = VoiceMessage(target_zone_ids=("ZONE-A",), message_text="Evacuate Zone A.")
        instructions_a = voice_controller.broadcast(message_a, 2.0)

        self.assertEqual(len(instructions_a), 1)
        self.assertEqual(set(instructions_a[0].speaker_ids), {speaker_a_id, speaker_b_id})

        message_b = VoiceMessage(target_zone_ids=("ZONE-B",), message_text="Evacuate Zone B.")
        instructions_b = voice_controller.broadcast(message_b, 3.0)

        self.assertEqual(len(instructions_b), 1)
        self.assertEqual(set(instructions_b[0].speaker_ids), {speaker_b_id})
        self.assertNotIn(speaker_a_id, instructions_b[0].speaker_ids)

        sent = voice_provider.sent_instructions()
        self.assertEqual(len(sent), 2)

    def test_unassigned_speaker_never_receives_a_zone_targeted_broadcast(self):

        # A third speaker, deliberately left with zone_ids=() (never
        # configured through the Property Panel) -- proving the
        # audit's own NO_SPEAKERS_AVAILABLE-for-everyone problem is
        # actually fixed: an unassigned speaker is now the EXCEPTION,
        # not the universal default, and it correctly never receives a
        # zone-targeted broadcast for a zone it was never assigned to.
        self.window.toolbar.speaker_action.trigger()
        self.scene.mousePressEvent(_FakeSceneMouseEvent(250, 400))

        unassigned_item = next(
            i for i in self.scene.items()
            if getattr(i, "model", None) and i.model.object_type == "Speaker" and not i.model.zone_ids
        )
        self.assertEqual(unassigned_item.model.zone_ids, ())

        speaker_manager = SpeakerManager()
        speaker_manager.discover_speakers(self.scene.project.building)

        voice_provider = SimulationVoiceOutputProvider()
        voice_controller = VoiceEvacuationController(speaker_manager, voice_provider)

        message = VoiceMessage(target_zone_ids=("ZONE-A",), message_text="Evacuate Zone A.")
        instructions = voice_controller.broadcast(message, 1.0)

        speaker_ids_notified = instructions[0].speaker_ids
        self.assertNotIn(unassigned_item.model.id, speaker_ids_notified)


if __name__ == "__main__":
    unittest.main()
