import sys
import unittest

from PyQt6.QtWidgets import QApplication, QPushButton

_app = QApplication.instance() or QApplication(sys.argv)

from models.building import Building
from models.door import Door

from advisory_system.recommendation_models import AdvisoryReport, BuildingRecommendation, CivilianAnnouncement

from voice_evacuation.controller import VoiceEvacuationController
from voice_evacuation.provider import SimulationVoiceOutputProvider

from speaker_manager.manager import SpeakerManager
from models.speaker import Speaker

from building_control.controller import BuildingControlController
from building_control.providers import SimulationControlProvider

from command_center.dashboard import Dashboard
from command_center.data_source import CommandCenterMode
from command_center.live_operator_action_gateway import LiveOperatorActionGateway


# =====================================================
# Live Operator Action Routing milestone -- Phase 6's Command Center
# safety-UX proof, exercised through the real Qt widgets (Dashboard ->
# RecommendationCenter -> VoiceEvacuationPanel/BuildingControlsPanel),
# not just the gateway in isolation (already covered by tests/
# test_live_operator_action_gateway.py). Every test here clicks a real
# QPushButton and reads the resulting table cell back, the same
# "drive the widget directly" convention tests/test_command_center.py
# and tests/test_live_command_center.py already use for the pre-
# existing Replay-path Approve/Reject buttons.
# =====================================================


def make_voice_report(zone_id="zone-1", zone_name="Zone 1", text="Evacuate now"):

    announcement = CivilianAnnouncement(
        zone_id=zone_id, zone_name=zone_name, announcement=text, reason="fire detected",
        confidence=0.9, predicted_rset_improvement_seconds=30.0, confidence_source=("ai",),
    )
    return AdvisoryReport(
        scenario_id="s", simulation_time=1.0, civilian_announcements=(announcement,),
        firefighter_intelligence=None, building_recommendations=(), commander_dashboard=None,
        recommendation_history=(),
    )


def make_control_report(door_id, action="Open Door"):

    recommendation = BuildingRecommendation(
        action=action, target_type="door", target_id=door_id, reason="egress",
        confidence=0.8, expected_engineering_benefit="faster egress",
    )
    return AdvisoryReport(
        scenario_id="s", simulation_time=1.0, civilian_announcements=(),
        firefighter_intelligence=None, building_recommendations=(recommendation,),
        commander_dashboard=None, recommendation_history=(),
    )


def make_dashboard_in_live_mode(gateway=None):

    dashboard = Dashboard()
    dashboard.set_operator_action_gateway(gateway)
    dashboard.set_mode(CommandCenterMode.LIVE)
    return dashboard


def buttons_in(table, row, column):

    widget = table.cellWidget(row, column)
    return widget.findChildren(QPushButton) if widget is not None else []


class VoicePanelOperatorActionTests(unittest.TestCase):

    def setUp(self):

        manager = SpeakerManager()
        manager.register_speaker(Speaker(name="SPK-1", floor_id="floor-1", zone_ids=("zone-1",)))

        self.voice_controller = VoiceEvacuationController(manager, SimulationVoiceOutputProvider())
        self.gateway = LiveOperatorActionGateway(voice_controller=self.voice_controller)
        self.dashboard = make_dashboard_in_live_mode(self.gateway)
        self.panel = self.dashboard.recommendation_center.voice_evacuation_panel

    def test_recommended_row_shows_approve_and_reject_buttons(self):

        self.panel.show_live(make_voice_report(), self.gateway)

        buttons = buttons_in(self.panel.active_table, 0, 6)
        self.assertEqual([b.text() for b in buttons], ["Approve / Send", "Reject"])

    def test_clicking_approve_sends_through_the_real_controller(self):

        self.panel.show_live(make_voice_report(), self.gateway)

        buttons_in(self.panel.active_table, 0, 6)[0].click()

        self.assertEqual(self.panel.active_table.item(0, 5).text(), "SENT")
        self.assertEqual(len(self.voice_controller.broadcast_log.all_instructions()), 1)
        self.assertEqual(self.panel.history_table.rowCount(), 1)

    def test_clicking_reject_never_touches_the_controller(self):

        self.panel.show_live(make_voice_report(), self.gateway)

        buttons_in(self.panel.active_table, 0, 6)[1].click()

        self.assertEqual(self.panel.active_table.item(0, 5).text(), "REJECTED")
        self.assertEqual(len(self.voice_controller.broadcast_log.all_instructions()), 0)

    def test_no_provider_disables_approve_but_not_reject(self):

        no_provider_gateway = LiveOperatorActionGateway()
        self.panel.show_live(make_voice_report(), no_provider_gateway)

        buttons = buttons_in(self.panel.active_table, 0, 6)
        approve_button, reject_button = buttons[0], buttons[1]

        self.assertFalse(approve_button.isEnabled())
        self.assertTrue(reject_button.isEnabled())

    def test_no_gateway_at_all_renders_honest_recommended_with_no_working_buttons(self):

        self.panel.show_live(make_voice_report(), None)

        self.assertEqual(self.panel.active_table.item(0, 5).text(), "RECOMMENDED")
        buttons = buttons_in(self.panel.active_table, 0, 6)
        self.assertFalse(buttons[0].isEnabled())
        self.assertFalse(buttons[1].isEnabled())

    def test_message_text_rendered_matches_advisory_verbatim(self):

        report = make_voice_report(text="Attention: proceed to the nearest marked exit.")
        self.panel.show_live(report, self.gateway)

        self.assertEqual(
            self.panel.active_table.item(0, 1).text(), "Attention: proceed to the nearest marked exit.",
        )


class ControlsPanelOperatorActionTests(unittest.TestCase):

    def setUp(self):

        self.building = Building(name="B")
        floor = self.building.create_floor(name="Ground Floor")
        self.door = Door(name="D1", floor_id=floor.id)
        floor.add_door(self.door)

        self.control_controller = BuildingControlController(self.building, SimulationControlProvider(self.building))
        self.gateway = LiveOperatorActionGateway(control_controller=self.control_controller)
        self.dashboard = make_dashboard_in_live_mode(self.gateway)
        self.panel = self.dashboard.recommendation_center.building_controls_panel

    def test_pending_row_shows_approve_and_reject_buttons(self):

        self.panel.show_live(make_control_report(self.door.id), self.gateway)

        buttons = buttons_in(self.panel.pending_table, 0, 6)
        self.assertEqual([b.text() for b in buttons], ["Approve", "Reject"])

    def test_clicking_reject_removes_it_from_pending_and_never_confirms(self):

        self.panel.show_live(make_control_report(self.door.id), self.gateway)

        buttons_in(self.panel.pending_table, 0, 6)[1].click()

        self.assertEqual(self.panel.pending_table.rowCount(), 0)
        self.assertEqual(self.panel.active_table.rowCount(), 0)

    def test_clicking_approve_dispatches_through_the_real_controller(self):

        self.panel.show_live(make_control_report(self.door.id), self.gateway)

        buttons_in(self.panel.pending_table, 0, 6)[0].click()

        # No action_executor configured -- honest failure, never a
        # fabricated confirmation (Phase 6's own "never display
        # CONFIRMED if the provider failed" requirement).
        self.assertEqual(self.panel.active_table.rowCount(), 0)
        self.assertGreater(self.panel.history_table.rowCount(), 0)

    def test_no_provider_preserves_the_original_inert_display(self):

        no_provider_gateway = LiveOperatorActionGateway()
        self.panel.show_live(make_control_report(self.door.id), no_provider_gateway)

        self.assertEqual(self.panel.pending_table.rowCount(), 1)
        self.assertEqual(self.panel.pending_table.item(0, 5).text(), "RECOMMENDED (not submitted)")
        self.assertEqual(self.panel.pending_table.item(0, 6).text(), "Execution Provider: Not Connected")
        self.assertIsNone(self.panel.pending_table.cellWidget(0, 6))

    def test_no_gateway_at_all_preserves_the_original_inert_display(self):

        self.panel.show_live(make_control_report(self.door.id), None)

        self.assertEqual(self.panel.pending_table.item(0, 6).text(), "Execution Provider: Not Connected")


if __name__ == "__main__":
    unittest.main()
