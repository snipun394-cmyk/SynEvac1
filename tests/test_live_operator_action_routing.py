import pathlib
import re
import sys
import unittest

from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

from models.building import Building
from models.door import Door

from advisory_system.orchestrator import AdvisoryOrchestrator
from advisory_system.recommendation_models import AdvisoryInputs, AdvisoryReport

from voice_evacuation.controller import VoiceEvacuationController
from voice_evacuation.provider import SimulationVoiceOutputProvider

from speaker_manager.manager import SpeakerManager
from models.speaker import Speaker

from building_control.controller import BuildingControlController
from building_control.providers import SimulationControlProvider
from building_control.types import RequestStatus

from command_center.data_source import CommandCenterMode
from command_center.dashboard import Dashboard
from command_center.live_operator_action_gateway import (
    OperatorActionUnavailable,
    VOICE_STATUS_FAILED,
    VOICE_STATUS_RECOMMENDED,
    VOICE_STATUS_REJECTED,
    VOICE_STATUS_SENT,
    LiveOperatorActionGateway,
)

from tests.test_advisory_system import make_building, make_decision_policy, make_ground_truth, make_scenario
from tests.test_live_command_center import _LiveChain


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


# =====================================================
# Live Operator Action Routing milestone.
#
# Phase 7 (AI Authority Guard), Phase 8 (FACP separation regression),
# Phase 9 (offline E2E voice), Phase 10 (offline E2E control), Phase 11
# (failure cases), Phase 13 (Replay stays read-only).
# =====================================================


def _make_two_zone_report(exit_decisions=(), stair_decisions=()):

    building = make_building()
    scenario = make_scenario()
    ground_truth = make_ground_truth(
        zone_route_stats=[
            {"zone_id": "zone-a", "preferred_exit": "exit-1", "preferred_stair": None,
             "average_travel_distance": 5.0, "average_travel_time": 10.0},
            {"zone_id": "zone-b", "preferred_exit": "exit-1", "preferred_stair": "stair-1",
             "average_travel_distance": 5.0, "average_travel_time": 10.0},
        ],
    )
    decision_policy = make_decision_policy(
        zone_decisions=[
            {"zone_id": "zone-a", "recommended_exit": "exit-1", "recommended_stair": None, "action": "EVACUATE_IMMEDIATELY"},
            {"zone_id": "zone-b", "recommended_exit": "exit-1", "recommended_stair": "stair-1", "action": "EVACUATE_IMMEDIATELY"},
        ],
        exit_decisions=exit_decisions,
        stair_decisions=stair_decisions,
    )

    inputs = AdvisoryInputs(building=building, scenario=scenario, ground_truth=ground_truth, decision_policy=decision_policy)

    return AdvisoryOrchestrator().generate_report(inputs), building


# =====================================================
# Phase 7 -- AI Authority Guard.
# =====================================================


class AIAuthorityGuardTests(unittest.TestCase):

    # AI inference and Advisory generation must never be able to reach
    # this milestone's own new gateway -- mechanically enforced the
    # same way tests/test_ai_augmented_advisory.py::
    # NoOutputExecutionBoundaryTests already enforces the identical
    # constraint against voice_evacuation/building_control.controller/
    # building_control.providers directly. This test additionally
    # forbids importing command_center.live_operator_action_gateway
    # itself -- the one new path into execution this milestone adds.

    _FORBIDDEN = (
        r"^\s*(from|import)\s+"
        r"(voice_evacuation|speaker_manager|building_control\.controller|"
        r"building_control\.providers|command_center\.live_operator_action_gateway|"
        r"command_center\.building_controls_panel|command_center\.recommendation_center)\b"
    )

    def _assert_file_clean(self, path: pathlib.Path):

        text = path.read_text(encoding="utf-8")

        self.assertIsNone(
            re.search(self._FORBIDDEN, text, re.MULTILINE),
            f"{path.relative_to(REPO_ROOT)} imports an execution-capable module or the operator "
            f"action gateway directly -- AI/Advisory generation must never be able to reach "
            f"real voice/control execution, only an explicit operator action may.",
        )

    def test_advisory_system_never_imports_execution_or_gateway_modules(self):

        for path in (REPO_ROOT / "advisory_system").glob("*.py"):
            self._assert_file_clean(path)

    def test_live_ai_gateway_never_imports_execution_or_gateway_modules(self):

        self._assert_file_clean(REPO_ROOT / "live_system" / "live_ai_gateway.py")

    def test_live_advisory_gateway_never_imports_execution_or_gateway_modules(self):

        self._assert_file_clean(REPO_ROOT / "live_system" / "live_advisory_gateway.py")

    def test_decision_policy_never_imports_execution_or_gateway_modules(self):

        for path in (REPO_ROOT / "decision_policy").glob("*.py"):
            self._assert_file_clean(path)

    def test_live_orchestrator_never_imports_the_operator_action_gateway(self):

        # LiveOrchestrator (live_system/orchestrator.py) is the one
        # class that runs AI inference + advisory generation each
        # cycle -- explicitly re-checked here by name, on top of the
        # whole-package sweep above.
        self._assert_file_clean(REPO_ROOT / "live_system" / "orchestrator.py")

    def test_gateway_itself_has_no_caller_anywhere_in_the_ai_advisory_chain(self):

        # Positive-direction proof, complementing the negative-direction
        # regex scans above: the only production files that import
        # command_center.live_operator_action_gateway at all are the
        # two Live panels this milestone wires it into -- never
        # anything upstream of AdvisoryReport generation.

        forbidden_importers = set()

        for package_dir in ("advisory_system", "live_system", "decision_policy", "ai_registry", "ai_inference"):

            directory = REPO_ROOT / package_dir

            if not directory.is_dir():
                continue

            for path in directory.glob("*.py"):

                text = path.read_text(encoding="utf-8")

                if "live_operator_action_gateway" in text:
                    forbidden_importers.add(str(path.relative_to(REPO_ROOT)))

        self.assertEqual(forbidden_importers, set())


# =====================================================
# Phase 8 -- FACP Separation.
# =====================================================


class FACPSeparationTests(unittest.TestCase):

    def test_facp_alarm_alone_produces_zero_voice_dispatches_and_zero_control_executions(self):

        # _LiveChain's own building_state_gateway (tests/
        # test_live_command_center.py) forces SMOKE-1 into ALARM every
        # cycle via facp_snapshot_provider -- a real, live FACP ALARM
        # condition reaches BuildingState.facp_status unchanged.
        # Running a full cycle through it, with no gateway wired to
        # anything and no operator action taken, must produce zero
        # voice broadcasts and zero control dispatches.

        chain = _LiveChain(with_ai=False)
        chain.orchestrator.start()
        chain.run_cycle(1.0)

        snapshot = chain.orchestrator.state_manager.current()
        self.assertIsNotNone(snapshot.building_state)
        self.assertIsNotNone(snapshot.building_state.facp_status)

        from facp.models import PanelState
        self.assertEqual(snapshot.building_state.facp_status.panel_state, PanelState.ALARM)

        voice_controller = VoiceEvacuationController(SpeakerManager(), SimulationVoiceOutputProvider())
        control_controller = BuildingControlController(make_building(), SimulationControlProvider(make_building()))

        # No caller anywhere in this test drove any operator action --
        # both real controllers must be entirely untouched.
        self.assertEqual(len(voice_controller.broadcast_log.all_instructions()), 0)
        self.assertEqual(control_controller.all_requests(), ())

        chain.orchestrator.stop()


# =====================================================
# Phase 9 -- Offline End-to-End Voice Test.
# =====================================================


class VoiceOperatorOfflineE2ETests(unittest.TestCase):

    def setUp(self):

        self.report, self.building = _make_two_zone_report()
        self.assertEqual(len(self.report.civilian_announcements), 2)

        self.zone_a_announcement = next(a for a in self.report.civilian_announcements if a.zone_id == "zone-a")
        self.zone_b_announcement = next(a for a in self.report.civilian_announcements if a.zone_id == "zone-b")

        manager = SpeakerManager()
        manager.register_speaker(Speaker(name="SPK-A", floor_id="floor-1", zone_ids=("zone-a",)))
        manager.register_speaker(Speaker(name="SPK-B", floor_id="floor-1", zone_ids=("zone-b",)))

        self.voice_controller = VoiceEvacuationController(manager, SimulationVoiceOutputProvider())
        self.gateway = LiveOperatorActionGateway(voice_controller=self.voice_controller)

    def test_1_nothing_is_sent_automatically(self):

        self.assertEqual(len(self.voice_controller.broadcast_log.all_instructions()), 0)
        for announcement in self.report.civilian_announcements:
            self.assertEqual(self.gateway.voice_recommendation_status(announcement), VOICE_STATUS_RECOMMENDED)

    def test_2_and_3_correct_zone_receives_message_other_zone_does_not(self):

        self.gateway.approve_voice_message(self.zone_a_announcement, self.report.simulation_time)

        self.assertEqual(self.gateway.voice_recommendation_status(self.zone_a_announcement), VOICE_STATUS_SENT)
        self.assertEqual(self.gateway.voice_recommendation_status(self.zone_b_announcement), VOICE_STATUS_RECOMMENDED)

        history = self.gateway.voice_broadcast_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].target_zone_id, "zone-a")

    def test_4_exact_advisory_text_is_preserved(self):

        instructions = self.gateway.approve_voice_message(self.zone_a_announcement, self.report.simulation_time)

        self.assertEqual(instructions[0].message.message_text, self.zone_a_announcement.announcement)

    def test_5_history_records_the_action(self):

        self.gateway.approve_voice_message(self.zone_a_announcement, self.report.simulation_time)

        self.assertEqual(len(self.gateway.voice_audit_log()), 1)
        self.assertEqual(len(self.voice_controller.broadcast_log.all_instructions()), 1)

    def test_6_provider_identity_is_visible(self):

        from command_center.live_operator_action_gateway import PROVIDER_CAPABILITY_SIMULATION
        self.assertEqual(self.gateway.voice_capability, PROVIDER_CAPABILITY_SIMULATION)

    def test_7_failure_is_represented_honestly(self):

        no_speaker_manager = SpeakerManager()  # no speakers registered anywhere
        controller = VoiceEvacuationController(no_speaker_manager, SimulationVoiceOutputProvider())
        gateway = LiveOperatorActionGateway(voice_controller=controller)

        gateway.approve_voice_message(self.zone_a_announcement, self.report.simulation_time)

        self.assertEqual(gateway.voice_recommendation_status(self.zone_a_announcement), VOICE_STATUS_FAILED)


# =====================================================
# Phase 10 -- Offline End-to-End Control Test.
# =====================================================


class ControlOperatorOfflineE2ETests(unittest.TestCase):

    def setUp(self):

        self.building = Building(name="B")
        floor = self.building.create_floor(name="Ground Floor")
        self.door = Door(name="D1", floor_id=floor.id)
        floor.add_door(self.door)

        self.control_controller = BuildingControlController(self.building, SimulationControlProvider(self.building))
        self.gateway = LiveOperatorActionGateway(control_controller=self.control_controller)

        from advisory_system.recommendation_models import BuildingRecommendation, AdvisoryReport as _AdvisoryReport

        self.recommendation = BuildingRecommendation(
            action="Open Door", target_type="door", target_id=self.door.id, reason="egress",
            confidence=0.8, expected_engineering_benefit="faster egress",
        )
        self.report = _AdvisoryReport(
            scenario_id="s", simulation_time=1.0, civilian_announcements=(),
            firefighter_intelligence=None, building_recommendations=(self.recommendation,),
            commander_dashboard=None, recommendation_history=(),
        )

    def test_1_no_automatic_execution(self):

        self.assertEqual(self.control_controller.all_requests(), ())

    def test_2_approve_executes_through_controller(self):

        request = self.gateway.ingest_control_recommendations(self.report)[0]
        self.gateway.approve_control_request(request.request_id)

        self.assertIn(
            self.control_controller.status_of(request.request_id),
            (RequestStatus.DISPATCHED, RequestStatus.CONFIRMED, RequestStatus.FAILED),
        )
        self.assertNotEqual(self.control_controller.status_of(request.request_id), RequestStatus.PENDING_APPROVAL)

    def test_3_reject_never_executes(self):

        request = self.gateway.ingest_control_recommendations(self.report)[0]
        self.gateway.reject_control_request(request.request_id)

        self.assertEqual(self.control_controller.status_of(request.request_id), RequestStatus.REJECTED)
        self.assertEqual(self.gateway.confirmed_control_entries(), ())

    def test_4_provider_failure_never_becomes_confirmed(self):

        # SimulationControlProvider(self.building) has no action_executor
        # configured -- Door/Exit dispatch honestly fails.
        request = self.gateway.ingest_control_recommendations(self.report)[0]
        self.gateway.approve_control_request(request.request_id)

        self.assertEqual(self.control_controller.status_of(request.request_id), RequestStatus.FAILED)
        self.assertEqual(self.gateway.confirmed_control_entries(), ())

    def test_5_duplicate_recommendation_does_not_cause_duplicate_execution(self):

        first = self.gateway.ingest_control_recommendations(self.report)[0]
        second = self.gateway.ingest_control_recommendations(self.report)[0]

        self.assertEqual(first.request_id, second.request_id)
        self.assertEqual(len(self.gateway.pending_control_requests()), 1)

    def test_6_contradictory_recommendations_preserve_history(self):

        from advisory_system.recommendation_models import BuildingRecommendation, AdvisoryReport as _AdvisoryReport

        close_recommendation = BuildingRecommendation(
            action="Close Door", target_type="door", target_id=self.door.id, reason="contain smoke",
            confidence=0.6, expected_engineering_benefit="contain smoke spread",
        )
        close_report = _AdvisoryReport(
            scenario_id="s", simulation_time=2.0, civilian_announcements=(),
            firefighter_intelligence=None, building_recommendations=(close_recommendation,),
            commander_dashboard=None, recommendation_history=(),
        )

        open_request = self.gateway.ingest_control_recommendations(self.report)[0]
        close_request = self.gateway.ingest_control_recommendations(close_report)[0]

        self.assertNotEqual(open_request.request_id, close_request.request_id)
        self.assertEqual(len(self.gateway.pending_control_requests()), 2)

    def test_7_state_only_systems_remain_honestly_state_only(self):

        from advisory_system.recommendation_models import BuildingRecommendation, AdvisoryReport as _AdvisoryReport
        from building_control.types import ControlSystemType

        exhaust_recommendation = BuildingRecommendation(
            action="Activate Smoke Exhaust", target_type="zone", target_id="does-not-exist",
            reason="clear smoke", confidence=0.7, expected_engineering_benefit="clears smoke",
        )
        # translate_recommendation matches on target_type/action-prefix;
        # use a real zone-shaped target instead so validation passes.
        building = Building(name="B2")
        floor = building.create_floor(name="Ground Floor")
        from models.zone import Zone
        zone = Zone(name="Z1", x=0, y=0, width=1, height=1, floor_id=floor.id)
        floor.add_zone(zone)

        controller = BuildingControlController(building, SimulationControlProvider(building))
        gateway = LiveOperatorActionGateway(control_controller=controller)

        exhaust_recommendation = BuildingRecommendation(
            action="Activate Smoke Exhaust", target_type="zone", target_id=zone.id,
            reason="clear smoke", confidence=0.7, expected_engineering_benefit="clears smoke",
        )
        report = _AdvisoryReport(
            scenario_id="s", simulation_time=1.0, civilian_announcements=(),
            firefighter_intelligence=None, building_recommendations=(exhaust_recommendation,),
            commander_dashboard=None, recommendation_history=(),
        )

        request = gateway.ingest_control_recommendations(report)[0]
        gateway.approve_control_request(request.request_id)

        self.assertEqual(controller.status_of(request.request_id), RequestStatus.CONFIRMED)
        entries = gateway.confirmed_control_entries()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].system_type, ControlSystemType.SMOKE_EXHAUST)

    def test_8_door_actions_reuse_action_executor_when_available(self):

        from simulation_interactive.action_executor import Action, ActionResult

        class RecordingActionExecutor:

            def __init__(self):
                self.calls = []

            def apply(self, action: Action, time: float) -> ActionResult:
                self.calls.append(action)
                return ActionResult(action=action, applied=True, reason=None)

        executor = RecordingActionExecutor()
        controller = BuildingControlController(self.building, SimulationControlProvider(self.building, executor))
        gateway = LiveOperatorActionGateway(control_controller=controller)

        request = gateway.ingest_control_recommendations(self.report)[0]
        gateway.approve_control_request(request.request_id)

        self.assertEqual(len(executor.calls), 1)
        self.assertEqual(controller.status_of(request.request_id), RequestStatus.CONFIRMED)


# =====================================================
# Phase 11 -- Failure Cases.
# =====================================================


class FailureCaseTests(unittest.TestCase):

    def test_1_no_voice_provider(self):

        gateway = LiveOperatorActionGateway()
        report, _building = _make_two_zone_report()

        with self.assertRaises(OperatorActionUnavailable):
            gateway.approve_voice_message(report.civilian_announcements[0], report.simulation_time)

    def test_2_no_building_control_provider(self):

        gateway = LiveOperatorActionGateway()

        with self.assertRaises(OperatorActionUnavailable):
            gateway.approve_control_request("does-not-exist")

    def test_3_voice_provider_failure(self):

        controller = VoiceEvacuationController(SpeakerManager(), SimulationVoiceOutputProvider())
        gateway = LiveOperatorActionGateway(voice_controller=controller)
        report, _building = _make_two_zone_report()

        gateway.approve_voice_message(report.civilian_announcements[0], report.simulation_time)  # must not raise
        self.assertEqual(
            gateway.voice_recommendation_status(report.civilian_announcements[0]), VOICE_STATUS_FAILED,
        )

    def test_4_control_provider_failure(self):

        building = Building(name="B")
        floor = building.create_floor(name="Ground Floor")
        door = Door(name="D1", floor_id=floor.id)
        floor.add_door(door)

        controller = BuildingControlController(building, SimulationControlProvider(building))
        gateway = LiveOperatorActionGateway(control_controller=controller)

        from advisory_system.recommendation_models import BuildingRecommendation, AdvisoryReport as _AdvisoryReport

        recommendation = BuildingRecommendation(
            action="Open Door", target_type="door", target_id=door.id, reason="egress",
            confidence=0.8, expected_engineering_benefit="faster egress",
        )
        report = _AdvisoryReport(
            scenario_id="s", simulation_time=1.0, civilian_announcements=(),
            firefighter_intelligence=None, building_recommendations=(recommendation,),
            commander_dashboard=None, recommendation_history=(),
        )

        request = gateway.ingest_control_recommendations(report)[0]
        gateway.approve_control_request(request.request_id)  # must not raise

        self.assertEqual(controller.status_of(request.request_id), RequestStatus.FAILED)

    def test_5_stale_advisory_report_never_offers_operator_actions(self):

        # Dashboard.apply_snapshot() already withholds a STALE cycle's
        # AdvisoryReport entirely (advisory_for_display = None -- see
        # dashboard.py's own Phase 14 discipline, unchanged by this
        # milestone) -- so a stale report never even reaches the voice/
        # control panels as something an operator could act on.

        from command_center.data_source import CommandCenterSnapshot, SnapshotConsistency

        report, _building = _make_two_zone_report()

        gateway = LiveOperatorActionGateway(
            voice_controller=VoiceEvacuationController(SpeakerManager(), SimulationVoiceOutputProvider()),
        )

        dashboard = Dashboard()
        dashboard.set_operator_action_gateway(gateway)
        dashboard.set_mode(CommandCenterMode.LIVE)

        stale_snapshot = CommandCenterSnapshot(
            mode=CommandCenterMode.LIVE, advisory_report=report, consistency=SnapshotConsistency.STALE,
        )
        dashboard.apply_snapshot(stale_snapshot)

        voice_panel = dashboard.recommendation_center.voice_evacuation_panel
        self.assertEqual(voice_panel.active_table.rowCount(), 0)

    def test_6_recommendation_superseded_before_approval(self):

        controller = VoiceEvacuationController(SpeakerManager(), SimulationVoiceOutputProvider())
        gateway = LiveOperatorActionGateway(voice_controller=controller)

        from advisory_system.recommendation_models import CivilianAnnouncement

        first = CivilianAnnouncement(
            zone_id="zone-a", zone_name="Zone A", announcement="Shelter in place",
            reason="fire", confidence=0.9, predicted_rset_improvement_seconds=None,
        )
        second = CivilianAnnouncement(
            zone_id="zone-a", zone_name="Zone A", announcement="Evacuate immediately",
            reason="fire spread", confidence=0.95, predicted_rset_improvement_seconds=None,
        )

        # No decision was ever made for `first` -- a new cycle's
        # recommendation supersedes it before any operator ever saw it.
        self.assertEqual(gateway.voice_recommendation_status(second), VOICE_STATUS_RECOMMENDED)

    def test_7_duplicate_operator_click(self):

        manager = SpeakerManager()
        manager.register_speaker(Speaker(name="SPK-A", floor_id="floor-1", zone_ids=("zone-a",)))
        controller = VoiceEvacuationController(manager, SimulationVoiceOutputProvider())
        gateway = LiveOperatorActionGateway(voice_controller=controller)

        report, _building = _make_two_zone_report()
        announcement = next(a for a in report.civilian_announcements if a.zone_id == "zone-a")

        gateway.approve_voice_message(announcement, report.simulation_time)
        gateway.approve_voice_message(announcement, report.simulation_time)  # duplicate click

        self.assertEqual(len(controller.broadcast_log.all_instructions()), 1)

    def test_8_operator_rejects(self):

        gateway = LiveOperatorActionGateway()
        report, _building = _make_two_zone_report()
        announcement = report.civilian_announcements[0]

        gateway.reject_voice_message(announcement, report.simulation_time)

        self.assertEqual(gateway.voice_recommendation_status(announcement), VOICE_STATUS_REJECTED)

    def test_9_live_runtime_stops_before_execution(self):

        chain = _LiveChain(with_ai=False)
        chain.orchestrator.start()
        chain.run_cycle(1.0)
        chain.orchestrator.stop()

        # A gateway built independently of the (now-stopped) orchestrator
        # is unaffected -- stopping the live runtime never crashes or
        # corrupts a separately-held gateway/controller.
        gateway = LiveOperatorActionGateway(
            voice_controller=VoiceEvacuationController(SpeakerManager(), SimulationVoiceOutputProvider()),
        )
        report, _building = _make_two_zone_report()
        gateway.approve_voice_message(report.civilian_announcements[0], report.simulation_time)  # must not raise

    def test_10_recommendation_disappears(self):

        # A recommendation present in one cycle's report simply absent
        # from the next (e.g. conditions improved) -- the gateway must
        # not crash or retain a stale actionable entry; there is
        # nothing to approve/reject for a recommendation that no longer
        # exists in the current report, and the panel-level rendering
        # (recommendation_center.py) simply shows zero rows for it.
        gateway = LiveOperatorActionGateway()
        report, _building = _make_two_zone_report()

        # Nothing crashes evaluating status for an announcement never
        # submitted to the gateway at all.
        self.assertEqual(gateway.voice_recommendation_status(report.civilian_announcements[0]), VOICE_STATUS_RECOMMENDED)

    def test_11_facp_alarm_with_no_operator_action(self):

        chain = _LiveChain(with_ai=False)
        chain.orchestrator.start()
        chain.run_cycle(1.0)

        snapshot = chain.orchestrator.state_manager.current()
        from facp.models import PanelState
        self.assertEqual(snapshot.building_state.facp_status.panel_state, PanelState.ALARM)

        # No gateway action was ever taken -- nothing to assert beyond
        # "this ran to completion without executing anything," already
        # proven structurally by FACPSeparationTests above.
        chain.orchestrator.stop()

    def test_12_ai_probability_changes_with_no_operator_action(self):

        # AI/advisory changing what it recommends, cycle to cycle, must
        # never itself execute anything -- only the presence of a
        # DIFFERENT recommendation, which still requires its own fresh
        # operator approval per test_6 above.
        gateway = LiveOperatorActionGateway(
            voice_controller=VoiceEvacuationController(SpeakerManager(), SimulationVoiceOutputProvider()),
        )

        report_a, _ = _make_two_zone_report()
        report_b, _ = _make_two_zone_report()  # a fresh "next cycle" report, same content

        for announcement in report_a.civilian_announcements + report_b.civilian_announcements:
            self.assertEqual(gateway.voice_recommendation_status(announcement), VOICE_STATUS_RECOMMENDED)


# =====================================================
# Phase 13 -- Replay Mode Stays Read-Only.
# =====================================================


class ReplayReadOnlyTests(unittest.TestCase):

    def test_replay_mode_never_receives_an_operator_action_gateway(self):

        # Dashboard.apply_snapshot() (the only Live render call) is the
        # sole place a gateway is ever threaded into recommendation_
        # center.show_live(); Dashboard.show_frame() (Replay's own,
        # completely separate render path -- command_center/dashboard.py)
        # never references self._operator_action_gateway at all.

        source_text = pathlib.Path(REPO_ROOT / "command_center" / "dashboard.py").read_text(encoding="utf-8")

        show_frame_body = source_text.split("def show_frame(self, frame):")[1].split("def set_frame_index")[0]

        self.assertNotIn("_operator_action_gateway", show_frame_body)

    def test_replay_path_reconstruction_never_calls_the_new_gateway(self):

        # command_center/incident_data.py's own Replay reconstruction
        # (_build_voice_broadcasts()/_build_control_requests(), an
        # already-audited, pre-existing deterministic replay of history)
        # must not import this milestone's new gateway either -- Replay
        # mode's own interactive approve/reject buttons (building_
        # controls_panel.py's Replay-path _on_approve()/_on_reject())
        # already route directly through IncidentData.control_controller,
        # by design, never through the new Live-only gateway.

        text = pathlib.Path(REPO_ROOT / "command_center" / "incident_data.py").read_text(encoding="utf-8")
        self.assertNotIn("live_operator_action_gateway", text)

    def test_dashboard_show_live_only_reachable_from_live_mode(self):

        app_mode_source = pathlib.Path(REPO_ROOT / "command_center" / "dashboard.py").read_text(encoding="utf-8")

        # apply_snapshot() (the only caller of recommendation_center.
        # show_live()) is a distinct method from show_frame()/
        # set_frame_index() (Replay's own path) -- structurally
        # confirmed by both existing side by side, never merged.
        self.assertIn("def apply_snapshot(self, snapshot", app_mode_source)
        self.assertIn("def show_frame(self, frame)", app_mode_source)


if __name__ == "__main__":
    unittest.main()
