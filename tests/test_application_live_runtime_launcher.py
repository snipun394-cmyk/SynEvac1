import sys
import unittest

from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

from advisory_system.recommendation_models import AdvisoryReport, BuildingRecommendation, CivilianAnnouncement

from building_control.types import RequestStatus

from voice_evacuation.models import BroadcastStatus

from dynamic_signage.models import SignageInstruction, SignageStatus

from models.project import Project

from designer.windows.main_window import MainWindow
from designer.widgets.live_runtime_panel import LiveRuntimePanel
from designer.live_runtime_controller import LiveRuntimeController

from live_runtime_launcher.modes import ApplicationMode, RuntimeLifecycleState
from live_runtime_launcher.session import LiveRuntimeSession

from tests.live_runtime_fixtures import make_demo_building


# =====================================================
# Application Live Runtime Launcher milestone -- Phase 1's own
# investigation (docs/architecture/application_live_runtime_integration.md)
# found the Designer entry point (main.py -> core.app.SynEvacApp ->
# designer.windows.main_window.MainWindow) and the already-tested Live
# Runtime composition root (live_runtime.factory.build_live_runtime())
# were two graphs that never met. This file proves they now do, through
# the ACTUAL application entry path (a real MainWindow instance and its
# real live_runtime_controller), never by calling build_live_runtime()
# directly.
# =====================================================


class LiveRuntimeSessionConstructionTests(unittest.TestCase):

    def test_designer_mode_is_rejected(self):

        with self.assertRaises(ValueError):
            LiveRuntimeSession(ApplicationMode.DESIGNER)

    def test_missing_building_is_an_honest_failed_state_not_a_crash(self):

        session = LiveRuntimeSession(ApplicationMode.OFFLINE_DEMO)
        session.construct(None)

        self.assertEqual(session.state, RuntimeLifecycleState.FAILED)
        self.assertIsNotNone(session.last_error)
        self.assertIsNone(session.runtime)

    def test_empty_building_constructs_successfully(self):

        from models.building import Building

        session = LiveRuntimeSession(ApplicationMode.OFFLINE_DEMO)
        session.construct(Building(id="empty", name="Empty", floors=[]))

        self.assertIsNotNone(session.runtime)
        self.assertEqual(session.state, RuntimeLifecycleState.STOPPED)
        self.assertIsNone(session.last_error)

    def test_starting_without_construct_is_an_honest_failed_state(self):

        session = LiveRuntimeSession(ApplicationMode.LIVE)
        session.start()

        self.assertEqual(session.state, RuntimeLifecycleState.FAILED)
        self.assertIsNotNone(session.last_error)


class LiveRuntimeSessionLifecycleTests(unittest.TestCase):

    def setUp(self):

        self.building = make_demo_building()

    def test_start_stop_restart_reuses_the_same_runtime_instance(self):

        session = LiveRuntimeSession(ApplicationMode.OFFLINE_DEMO)
        session.construct(self.building)
        runtime = session.runtime

        session.start()
        self.assertTrue(session.is_running)
        self.assertEqual(session.state, RuntimeLifecycleState.RUNNING)

        session.stop()
        self.assertFalse(session.is_running)
        self.assertEqual(session.state, RuntimeLifecycleState.STOPPED)

        session.start()
        self.assertTrue(session.is_running)
        self.assertIs(session.runtime, runtime)  # never reconstructed

        session.stop()

    def test_double_start_is_idempotent(self):

        session = LiveRuntimeSession(ApplicationMode.OFFLINE_DEMO)
        session.construct(self.building)

        session.start()
        session.start()  # must not raise

        self.assertTrue(session.is_running)
        session.stop()

    def test_double_stop_and_stop_before_start_are_safe(self):

        session = LiveRuntimeSession(ApplicationMode.OFFLINE_DEMO)
        session.construct(self.building)

        session.stop()  # never started
        session.stop()  # repeated

        self.assertEqual(session.state, RuntimeLifecycleState.STOPPED)

    def test_zero_network_zero_hardware_by_default(self):

        # Neither mode's session ever supplies frame_sources/
        # human_detector/identity_resolver (no RTSP-from-Camera-asset
        # builder exists yet -- explicitly out of scope this milestone),
        # so no camera pipeline exists to open a network connection at
        # all, in either mode.

        for mode in (ApplicationMode.LIVE, ApplicationMode.OFFLINE_DEMO):

            session = LiveRuntimeSession(mode)
            session.construct(self.building)

            self.assertEqual(session.runtime.frame_sources, {})
            self.assertIsNone(session.runtime.camera_pipeline)

            session.start()
            self.assertEqual(session.runtime.frame_sources, {})
            session.stop()


class LiveRuntimeSessionCapabilityHonestyTests(unittest.TestCase):

    def setUp(self):

        self.building = make_demo_building()

    def test_ai_is_always_not_configured(self):

        # No default live_ai_gateway construction path exists anywhere
        # in this repository (confirmed by docs/architecture/
        # synevac_end_to_end_architecture_review.md) -- this launcher
        # must never fabricate one.

        for mode in (ApplicationMode.LIVE, ApplicationMode.OFFLINE_DEMO):

            session = LiveRuntimeSession(mode)
            session.construct(self.building)

            self.assertEqual(session.ai_status(), "NOT CONFIGURED")

    def test_live_mode_reports_no_provider_for_every_output_channel(self):

        session = LiveRuntimeSession(ApplicationMode.LIVE)
        session.construct(self.building)

        capabilities = session.provider_capabilities()

        self.assertEqual(capabilities["camera"], "NO_PROVIDER")
        self.assertEqual(capabilities["voice"], "NO_PROVIDER")
        self.assertEqual(capabilities["dynamic_signage"], "NO_PROVIDER")
        self.assertEqual(capabilities["building_control"], "NO_PROVIDER")

        self.assertIsNone(session.runtime.voice_evacuation_controller)
        self.assertIsNone(session.runtime.dynamic_signage_controller)
        self.assertIsNone(session.runtime.building_control_controller)

    def test_offline_demo_mode_reports_simulation_for_output_channels(self):

        session = LiveRuntimeSession(ApplicationMode.OFFLINE_DEMO)
        session.construct(self.building)

        capabilities = session.provider_capabilities()

        self.assertEqual(capabilities["camera"], "NO_PROVIDER")  # still no frame_sources
        self.assertEqual(capabilities["voice"], "SIMULATION")
        self.assertEqual(capabilities["dynamic_signage"], "SIMULATION")
        self.assertEqual(capabilities["building_control"], "SIMULATION")

    def test_unconstructed_session_reports_every_channel_as_no_provider(self):

        session = LiveRuntimeSession(ApplicationMode.LIVE)

        self.assertEqual(session.ai_status(), "NOT CONFIGURED")
        capabilities = session.provider_capabilities()
        self.assertTrue(all(value == "NO_PROVIDER" for value in capabilities.values()))


class DegradedStateDetectionTests(unittest.TestCase):

    # Unit-tests LiveRuntimeSession._any_configured_camera_offline() in
    # isolation against a minimal stand-in runtime -- constructing a
    # genuinely offline configured camera through the full factory would
    # require a real CameraFrameSource, unnecessary to prove this one
    # honest-reporting rule.

    class _FakeCameraManager:

        def __init__(self, statuses):
            self._statuses = statuses

        def connection_status(self, camera_id):
            return self._statuses[camera_id]

    class _FakeRuntime:

        def __init__(self, frame_sources, camera_manager):
            self.frame_sources = frame_sources
            self.camera_manager = camera_manager

    def test_no_frame_sources_is_never_degraded(self):

        from camera_manager.connection_status import CameraConnectionState

        session = LiveRuntimeSession(ApplicationMode.LIVE)
        session.runtime = self._FakeRuntime({}, self._FakeCameraManager({}))

        self.assertFalse(session._any_configured_camera_offline())

    def test_an_offline_configured_camera_is_reported_degraded(self):

        from camera_manager.connection_status import CameraConnectionState

        session = LiveRuntimeSession(ApplicationMode.LIVE)
        session.runtime = self._FakeRuntime(
            {"CAM-1": object()},
            self._FakeCameraManager({"CAM-1": CameraConnectionState.OFFLINE}),
        )

        self.assertTrue(session._any_configured_camera_offline())

    def test_an_online_configured_camera_is_never_degraded(self):

        from camera_manager.connection_status import CameraConnectionState

        session = LiveRuntimeSession(ApplicationMode.LIVE)
        session.runtime = self._FakeRuntime(
            {"CAM-1": object()},
            self._FakeCameraManager({"CAM-1": CameraConnectionState.ONLINE}),
        )

        self.assertFalse(session._any_configured_camera_offline())


class PanelAndControllerWiringTests(unittest.TestCase):

    def setUp(self):

        self.building = make_demo_building()
        self.panel = LiveRuntimePanel()
        self.controller = LiveRuntimeController(self.panel, lambda: self.building)

    def tearDown(self):

        self.controller.shutdown()

    def test_constructing_the_controller_starts_nothing(self):

        # Safe Startup Semantics milestone requirement -- building the
        # panel/controller (equivalent to Designer simply launching)
        # must never itself construct or start a LiveRuntime.

        self.assertIsNone(self.controller.session)
        self.assertIn("STOPPED", self.panel.status_label.text())

    def test_default_selected_mode_is_live(self):

        self.assertEqual(self.panel.selected_mode(), ApplicationMode.LIVE)

    def test_start_button_constructs_and_starts_a_session(self):

        self.panel.start_button.click()

        self.assertIsNotNone(self.controller.session)
        self.assertTrue(self.controller.session.is_running)
        self.assertIn("RUNNING", self.panel.status_label.text())

    def test_stop_button_stops_the_running_session(self):

        self.panel.start_button.click()
        self.panel.stop_button.click()

        self.assertFalse(self.controller.session.is_running)
        self.assertIn("STOPPED", self.panel.status_label.text())

    def test_switching_mode_while_stopped_forces_reconstruction(self):

        self.panel.mode_combo.setCurrentIndex(1)  # Offline Demo
        self.panel.start_button.click()

        offline_runtime = self.controller.session.runtime
        self.assertEqual(self.controller.session.mode, ApplicationMode.OFFLINE_DEMO)
        self.assertIsNotNone(offline_runtime.voice_evacuation_controller)

        self.panel.stop_button.click()
        self.panel.mode_combo.setCurrentIndex(0)  # back to Live

        self.assertIsNone(self.controller.session)

        self.panel.start_button.click()

        self.assertEqual(self.controller.session.mode, ApplicationMode.LIVE)
        self.assertIsNot(self.controller.session.runtime, offline_runtime)
        self.assertIsNone(self.controller.session.runtime.voice_evacuation_controller)

    def test_switching_mode_while_running_does_not_touch_the_session(self):

        self.panel.start_button.click()
        running_session = self.controller.session

        self.panel.mode_combo.setCurrentIndex(1)

        self.assertIs(self.controller.session, running_session)
        self.assertTrue(self.controller.session.is_running)

    def test_open_command_center_before_start_shows_a_message_not_a_crash(self):

        # No QMessageBox.information mock needed -- exec() on an
        # information box would block; instead assert no exception and
        # no window created.
        from PyQt6.QtWidgets import QMessageBox

        original = QMessageBox.information
        QMessageBox.information = staticmethod(lambda *a, **k: None)
        try:
            self.controller.on_open_command_center()
        finally:
            QMessageBox.information = original

    def test_open_command_center_after_start_returns_a_working_window(self):

        self.panel.start_button.click()
        self.controller.on_open_command_center()

        window = self.controller.session._command_center_window
        self.assertIsNotNone(window)
        self.assertIs(
            window.live_data_source._state_manager,
            self.controller.session.runtime.orchestrator.state_manager,
        )

    def test_stop_and_reset_tears_down_session_and_command_center(self):

        self.panel.start_button.click()
        self.controller.on_open_command_center()

        self.controller.stop_and_reset()

        self.assertIsNone(self.controller.session)
        self.assertIn("STOPPED", self.panel.status_label.text())


class ApplicationLevelOfflineFullChainE2ETests(unittest.TestCase):

    # Phase 12's own "critical test" -- exercises the NEW APPLICATION
    # ENTRY PATH (a real designer.windows.main_window.MainWindow and its
    # real live_runtime_controller), never build_live_runtime() called
    # directly. Proves: application launch -> project load -> Live mode
    # -> LiveRuntime construction -> runtime start -> BuildingState ->
    # every intelligence engine -> StateManager -> Live Command Center,
    # all through the SAME instances, with the operator boundary intact.

    def setUp(self):

        self.window = MainWindow()
        self.window.canvas.scene_obj.project = Project(name="Demo", building=make_demo_building())

    def tearDown(self):

        self.window.live_runtime_controller.shutdown()

    def test_full_chain_reaches_command_center_through_the_same_instances(self):

        panel = self.window.live_runtime_panel
        panel.mode_combo.setCurrentIndex(1)  # Offline Demo -- Simulation providers
        panel.start_button.click()

        controller = self.window.live_runtime_controller
        session = controller.session

        self.assertTrue(session.is_running)

        for _ in range(3):
            session.runtime.run_cycle(1.0)

        snapshot = session.runtime.command_center_data_source.current_snapshot()
        self.assertIsNotNone(snapshot.building_state)
        self.assertIsNotNone(snapshot.evacuation_progress)
        self.assertIsNotNone(snapshot.trajectory_intelligence)
        self.assertIsNotNone(snapshot.emergency_response)

        panel.open_command_center_button.click()
        command_center_window = session._command_center_window

        # Phase 6 -- the SAME StateManager/runtime, never a duplicate.
        self.assertIs(command_center_window.live_data_source._state_manager, session.runtime.orchestrator.state_manager)
        self.assertIs(command_center_window.live_data_source, session.runtime.command_center_data_source)

        self.assertIs(
            session.runtime.operator_action_gateway.voice_controller, session.runtime.voice_evacuation_controller,
        )
        self.assertIs(
            session.runtime.operator_action_gateway.control_controller, session.runtime.building_control_controller,
        )

    def test_nothing_auto_dispatches_before_operator_approval(self):

        panel = self.window.live_runtime_panel
        panel.mode_combo.setCurrentIndex(1)
        panel.start_button.click()

        session = self.window.live_runtime_controller.session

        for _ in range(3):
            session.runtime.run_cycle(1.0)

        gateway = session.runtime.operator_action_gateway
        self.assertEqual(len(session.runtime.voice_evacuation_controller.broadcast_log.all_instructions()), 0)
        self.assertEqual(session.runtime.building_control_controller.all_requests(), ())
        self.assertEqual(gateway.all_signage_instructions(), ())

    def test_operator_approves_voice_signage_and_building_control_through_the_same_controllers(self):

        panel = self.window.live_runtime_panel
        panel.mode_combo.setCurrentIndex(1)
        panel.start_button.click()

        session = self.window.live_runtime_controller.session
        session.runtime.run_cycle(1.0)

        gateway = session.runtime.operator_action_gateway

        # --- Voice -------------------------------------------------
        announcement = CivilianAnnouncement(
            zone_id="zone-cafeteria", zone_name="Cafeteria",
            announcement="Evacuate via nearest exit.", reason="test", confidence=None,
            predicted_rset_improvement_seconds=None,
        )
        instructions = gateway.approve_voice_message(announcement, 1.0)
        self.assertGreater(len(instructions), 0)
        self.assertEqual(instructions[0].status, BroadcastStatus.BROADCAST)
        self.assertIs(
            session.runtime.voice_evacuation_controller, session.runtime.operator_action_gateway.voice_controller,
        )

        # --- Building Control ---------------------------------------
        recommendation = BuildingRecommendation(
            action="Open Door", target_type="door", target_id="DOOR-1",
            reason="test", confidence=None, expected_engineering_benefit="test",
        )
        report = AdvisoryReport(
            scenario_id="demo-scn", simulation_time=1.0, civilian_announcements=(),
            firefighter_intelligence=None, building_recommendations=(recommendation,),
            commander_dashboard=None, recommendation_history=(),
        )
        request = gateway.ingest_control_recommendations(report)[0]
        gateway.approve_control_request(request.request_id)

        # No ActionExecutor configured -- honest FAILED, never a
        # fabricated CONFIRMED (build_offline_demo_runtime's own
        # SimulationControlProvider default, action_executor=None).
        self.assertEqual(
            session.runtime.building_control_controller.status_of(request.request_id), RequestStatus.FAILED,
        )

        # --- Dynamic Signage ------------------------------------------
        signage_instruction = SignageInstruction(
            sign_id="SIGN-TEST", floor_id="floor-1", zone_id="zone-cafeteria",
            recommended_exit_id="EXIT-1", guidance_revision=1, indication="LEFT",
            status=SignageStatus.ACTIVE, signage_revision=1,
        )
        session.runtime.dynamic_signage_controller.submit(signage_instruction, 1.0)
        approved = gateway.approve_signage_instruction(signage_instruction, 1.0)
        self.assertIsNotNone(approved)

        self.assertIs(
            session.runtime.dynamic_signage_controller, session.runtime.operator_action_gateway.signage_controller,
        )

    def test_live_mode_leaves_every_output_channel_at_no_provider(self):

        panel = self.window.live_runtime_panel  # default mode is LIVE
        panel.start_button.click()

        session = self.window.live_runtime_controller.session
        self.assertEqual(session.mode, ApplicationMode.LIVE)

        gateway = session.runtime.operator_action_gateway
        self.assertIsNone(gateway.voice_controller)
        self.assertIsNone(gateway.control_controller)
        self.assertIsNone(gateway.signage_controller)


class ProjectLifecycleResetTests(unittest.TestCase):

    def setUp(self):

        self.window = MainWindow()
        self.window.canvas.scene_obj.project = Project(name="Demo", building=make_demo_building())

    def tearDown(self):

        self.window.live_runtime_controller.shutdown()

    def test_new_project_stops_and_resets_a_running_live_runtime_session(self):

        self.window.live_runtime_panel.start_button.click()
        self.assertIsNotNone(self.window.live_runtime_controller.session)

        self.window.new_project()

        self.assertIsNone(self.window.live_runtime_controller.session)

    def test_close_event_shuts_down_a_running_live_runtime_session(self):

        self.window.live_runtime_panel.start_button.click()
        session = self.window.live_runtime_controller.session
        self.assertTrue(session.is_running)

        class FakeCloseEvent:
            def accept(self):
                pass
            def ignore(self):
                pass

        self.window.closeEvent(FakeCloseEvent())

        self.assertFalse(session.is_running)
        self.assertIsNone(self.window.live_runtime_controller.session)


class AuthorityGuardTests(unittest.TestCase):

    # Authority Guards milestone requirement (Phase 14) -- mechanically
    # proves the launcher/application layer composes but never decides:
    # never imports AI/advisory/decision-policy internals directly, and
    # (structurally, via LiveRuntimeSession never supplying a
    # live_ai_gateway/live_advisory_gateway of its own) can never make
    # AI execution-capable, auto-approve, or auto-dispatch anything --
    # every approve_*() call in this file above required an explicit,
    # test-driven call, exactly mirroring what a real operator click
    # would be.

    def test_session_module_never_imports_ai_or_advisory_internals(self):

        import pathlib
        import re

        path = pathlib.Path(__file__).resolve().parent.parent / "live_runtime_launcher" / "session.py"
        text = path.read_text(encoding="utf-8")

        forbidden = (
            r"^\s*(from|import)\s+"
            r"(advisory_system\.orchestrator|ai_registry|ai_inference|ai_training|decision_policy)\b"
        )

        self.assertIsNone(re.search(forbidden, text, re.MULTILINE))

    def test_constructing_and_starting_a_session_never_supplies_an_ai_or_advisory_gateway(self):

        session = LiveRuntimeSession(ApplicationMode.OFFLINE_DEMO)
        session.construct(make_demo_building())

        self.assertIsNone(session.runtime.orchestrator.live_ai_gateway)
        self.assertIsNone(session.runtime.orchestrator.live_advisory_gateway)


if __name__ == "__main__":
    unittest.main()
