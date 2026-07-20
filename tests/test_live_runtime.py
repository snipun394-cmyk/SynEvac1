import unittest

from models.building import Building

from camera_manager.manager import CameraManager
from sensor_manager.manager import SensorManager
from multi_camera_fusion.engine import MultiCameraFusionEngine
from speaker_manager.manager import SpeakerManager

from facp.engine import SimulatedFACP

from voice_evacuation.provider import SimulationVoiceOutputProvider
from building_control.providers import SimulationControlProvider

from live_system.orchestrator import LiveSystemAlreadyRunningError

from live_runtime.factory import build_live_runtime, build_offline_demo_runtime

from tests.live_runtime_fixtures import (
    make_demo_building,
    make_offline_frame_sources,
    make_offline_human_detector,
    make_offline_identity_resolver,
)


# =====================================================
# Production Live Runtime Composition Root milestone.
#
# Phase 4 (shared-instance ownership) and Phase 5 (lifecycle) core
# tests. Phase 7 (offline E2E demo) and Phase 8 (failure/degradation)
# live in their own dedicated test files.
# =====================================================


class NoAutomaticConnectionTests(unittest.TestCase):

    # Phase 3's own hard requirement: constructing (or building via the
    # factory) a LiveRuntime must never itself connect to anything --
    # only an explicit runtime.start() may.

    def test_building_the_runtime_never_starts_any_frame_source(self):

        building = make_demo_building()
        frame_sources = make_offline_frame_sources(building)

        runtime = build_offline_demo_runtime(
            building, frame_sources=frame_sources,
            human_detector=make_offline_human_detector(),
            identity_resolver=make_offline_identity_resolver(building),
        )

        self.assertFalse(runtime.is_running)
        for source in runtime.frame_sources.values():
            self.assertFalse(source.is_running)

    def test_building_the_runtime_never_starts_the_orchestrator(self):

        building = Building(name="B")
        building.create_floor(name="F1")

        runtime = build_offline_demo_runtime(building)

        self.assertFalse(runtime.orchestrator.is_running)


class SharedInstanceOwnershipTests(unittest.TestCase):

    # Phase 4: one live session must use the SAME shared instances
    # everywhere identity/state matters -- never a Command Center
    # controller A vs. a Live Runtime controller B.

    def setUp(self):

        self.building = make_demo_building()

        self.runtime = build_offline_demo_runtime(
            self.building,
            frame_sources=make_offline_frame_sources(self.building),
            human_detector=make_offline_human_detector(),
            identity_resolver=make_offline_identity_resolver(self.building),
        )

    def test_operator_action_gateway_shares_the_runtime_voice_controller(self):

        self.assertIsNotNone(self.runtime.voice_evacuation_controller)
        self.assertIs(
            self.runtime.operator_action_gateway.voice_controller,
            self.runtime.voice_evacuation_controller,
        )

    def test_operator_action_gateway_shares_the_runtime_control_controller(self):

        self.assertIsNotNone(self.runtime.building_control_controller)
        self.assertIs(
            self.runtime.operator_action_gateway.control_controller,
            self.runtime.building_control_controller,
        )

    def test_command_center_data_source_shares_the_orchestrator_state_manager(self):

        self.assertIs(
            self.runtime.command_center_data_source._state_manager,
            self.runtime.orchestrator.state_manager,
        )

    def test_command_center_data_source_shares_the_same_building(self):

        self.assertIs(self.runtime.command_center_data_source._building, self.building)

    def test_voice_controller_shares_the_runtime_speaker_manager(self):

        self.assertIs(self.runtime.voice_evacuation_controller._speaker_manager, self.runtime.speaker_manager)

    def test_control_controller_shares_the_runtime_building(self):

        self.assertIs(self.runtime.building_control_controller._building, self.building)

    def test_camera_pipeline_and_camera_manager_share_the_same_frame_sources(self):

        self.assertEqual(
            set(self.runtime.camera_pipeline.frame_sources.keys()), set(self.runtime.frame_sources.keys()),
        )
        for camera_id, source in self.runtime.frame_sources.items():
            self.assertIs(self.runtime.camera_pipeline.frame_sources[camera_id], source)

    def test_a_caller_supplied_camera_manager_is_reused_not_duplicated(self):

        shared_camera_manager = CameraManager()
        shared_camera_manager.discover_cameras(self.building)

        runtime = build_offline_demo_runtime(
            self.building, camera_manager=shared_camera_manager,
        )

        self.assertIs(runtime.camera_manager, shared_camera_manager)

    def test_a_caller_supplied_sensor_manager_is_reused_not_duplicated(self):

        shared_sensor_manager = SensorManager()
        shared_sensor_manager.discover_sensors(self.building)

        runtime = build_offline_demo_runtime(self.building, sensor_manager=shared_sensor_manager)

        self.assertIs(runtime.sensor_manager, shared_sensor_manager)

    def test_a_caller_supplied_fusion_engine_is_reused_not_duplicated(self):

        shared_fusion_engine = MultiCameraFusionEngine()

        runtime = build_offline_demo_runtime(self.building, fusion_engine=shared_fusion_engine)

        self.assertIs(runtime.fusion_engine, shared_fusion_engine)

    def test_fusion_engine_state_persists_across_cycles_same_instance(self):

        # MultiCameraFusionEngine holds cross-cycle TrackHistory state
        # (handover tracking) -- it must never be silently reconstructed
        # per cycle. Two consecutive run_cycle() calls must observe the
        # identical fusion_engine object throughout.

        fusion_engine_before = self.runtime.fusion_engine

        self.runtime.start()
        self.runtime.run_cycle(1.0)
        self.runtime.run_cycle(2.0)

        self.assertIs(self.runtime.fusion_engine, fusion_engine_before)

        self.runtime.stop()


class LifecycleTests(unittest.TestCase):

    def setUp(self):

        self.building = make_demo_building()
        self.runtime = build_offline_demo_runtime(
            self.building,
            frame_sources=make_offline_frame_sources(self.building),
            human_detector=make_offline_human_detector(),
            identity_resolver=make_offline_identity_resolver(self.building),
        )

    def test_start_then_is_running_true(self):

        self.runtime.start()
        self.assertTrue(self.runtime.is_running)

        for source in self.runtime.frame_sources.values():
            self.assertTrue(source.is_running)

        self.assertTrue(self.runtime.orchestrator.is_running)

    def test_repeated_start_is_safe(self):

        self.runtime.start()
        self.runtime.start()  # must not raise LiveSystemAlreadyRunningError

        self.assertTrue(self.runtime.is_running)

    def test_repeated_stop_is_safe(self):

        self.runtime.start()
        self.runtime.stop()
        self.runtime.stop()  # must not raise

        self.assertFalse(self.runtime.is_running)

    def test_stop_before_start_is_safe(self):

        self.runtime.stop()  # must not raise

        self.assertFalse(self.runtime.is_running)

    def test_stop_stops_every_frame_source(self):

        self.runtime.start()
        self.runtime.stop()

        for source in self.runtime.frame_sources.values():
            self.assertFalse(source.is_running)

        self.assertFalse(self.runtime.orchestrator.is_running)

    def test_run_cycle_after_stop_raises_the_orchestrators_own_error(self):

        from live_system.orchestrator import LiveSystemNotRunningError

        self.runtime.start()
        self.runtime.stop()

        with self.assertRaises(LiveSystemNotRunningError):
            self.runtime.run_cycle(1.0)

    def test_one_failed_camera_does_not_prevent_others_or_the_orchestrator_from_starting(self):

        class RaisingFrameSource:

            def start(self):
                raise RuntimeError("simulated camera start failure")

            def stop(self):
                pass

            @property
            def is_running(self):
                return False

            def read_frame(self):
                return None

        camera_ids = list(self.runtime.frame_sources.keys())
        self.runtime.frame_sources[camera_ids[0]] = RaisingFrameSource()

        self.runtime.start()  # must not raise

        self.assertTrue(self.runtime.is_running)
        self.assertTrue(self.runtime.frame_sources[camera_ids[1]].is_running)
        self.assertTrue(self.runtime.orchestrator.is_running)

        self.runtime.stop()

    def test_partial_startup_failure_rolls_back_and_leaves_is_running_false(self):

        # Simulate the orchestrator itself failing to start (already
        # running from outside LiveRuntime's own bookkeeping) -- every
        # frame source that WAS started must be rolled back, and
        # is_running must end up honestly False, never true.

        self.runtime.orchestrator.start()  # pre-empt it, so LiveRuntime's own orchestrator.start() call raises

        with self.assertRaises(LiveSystemAlreadyRunningError):
            self.runtime.start()

        self.assertFalse(self.runtime.is_running)
        for source in self.runtime.frame_sources.values():
            self.assertFalse(source.is_running)

        self.runtime.orchestrator.stop()

    def test_component_failing_during_shutdown_does_not_block_the_rest(self):

        class RaisingStopFrameSource:

            def __init__(self):
                self._running = False

            def start(self):
                self._running = True

            def stop(self):
                raise RuntimeError("simulated stop failure")

            @property
            def is_running(self):
                return self._running

            def read_frame(self):
                return None

        camera_ids = list(self.runtime.frame_sources.keys())
        other_source = self.runtime.frame_sources[camera_ids[1]]
        self.runtime.frame_sources[camera_ids[0]] = RaisingStopFrameSource()

        self.runtime.start()
        self.runtime.stop()  # must not raise despite one source's stop() raising

        self.assertFalse(other_source.is_running)
        self.assertFalse(self.runtime.is_running)


if __name__ == "__main__":
    unittest.main()
