import unittest

from hazard.snapshot import HazardSnapshot

from occupancy.snapshot import OccupancySnapshot

from models.building import Building
from models.floor import Floor
from models.zone import Zone
from models.camera import Camera
from models.smoke_detector import SmokeDetector
from models.sensor_asset import DetectorState, HealthStatus

from camera_manager.manager import CameraManager
from camera_manager.status import CameraStatus

from sensor_manager.status import SensorStatus

from live_camera_pipeline.replay_frame_source import ReplayFrameSource
from live_camera_pipeline.detection_provider import LiveCameraPipelineDetectionProvider
from live_camera_pipeline.identity_resolver import MappingIdentityResolver
from live_camera_pipeline.pipeline import LiveCameraPipeline

from tests.live_camera_pipeline_fixtures import MockHumanDetector

from multi_camera_fusion.engine import MultiCameraFusionEngine
from multi_camera_fusion.track import FusionResult

from facp.engine import SimulatedFACP
from facp.models import DetectorConditionReport, PanelState

from building_control.controller import BuildingControlController
from building_control.providers import SimulationControlProvider
from building_control.requests import ControlRequest
from building_control.types import ApprovalMode, ControlAction, ControlSystemType, RequestSource

from building_state.estimator import BuildingStateEstimator
from building_state.models import BuildingState

from live_system.building_state_gateway import BuildingStateGateway, EstimatorBuildingStateGateway
from live_system.event_bus import EventBus, EventType
from live_system.orchestrator import LiveOrchestrator, LiveSystemNotRunningError
from live_system.state_manager import StateManager


# =====================================================
# Canonical Live BuildingState Runtime Assembly milestone -- Phase 9/12.
#
# Every test below proves the NEW seam (EstimatorBuildingStateGateway /
# StateManager.update_building_state / LiveOrchestrator.building_state_
# gateway); none of them re-prove fusion, identity resolution, FACP, or
# Building Control correctness themselves -- those already have their
# own dedicated test suites this file never duplicates.
#
# No network access, no real CCTV, no AI inference, no advisory
# generation appears anywhere in this file.
# =====================================================


def _make_camera_status(camera_id, active=True, zone_ids=("zone-a",)):

    return CameraStatus(
        camera_id=camera_id, name=camera_id, floor_id="floor-1",
        zone_ids=zone_ids, active=active, mode="Simulation", has_detection_provider=True,
    )


def _make_sensor_status(sensor_id, sensor_type="SmokeDetector", active=True, health_status=HealthStatus.OK):

    return SensorStatus(
        sensor_id=sensor_id, sensor_type=sensor_type, name=sensor_id, floor_id="floor-1",
        zone_ids=("zone-a",), active=active, mode="Simulation", health_status=health_status,
    )


# =====================================================
# EstimatorBuildingStateGateway -- unit level
# =====================================================


class EstimatorBuildingStateGatewayEmptyProvidersTests(unittest.TestCase):

    # Phase 9 items 6/14/15/16/17 -- a gateway with every provider left
    # unset (no cameras, no sensors, no FACP, no control provider
    # configured) must still produce a valid, non-crashing BuildingState.

    def test_collect_with_no_providers_returns_a_valid_empty_building_state(self):

        gateway = EstimatorBuildingStateGateway()

        state = gateway.collect(5.0)

        self.assertIsInstance(state, BuildingState)
        self.assertEqual(state.timestamp, 5.0)
        self.assertEqual(dict(state.occupant_tracks), {})
        self.assertEqual(dict(state.camera_observations), {})
        self.assertEqual(dict(state.smoke_detector_states), {})
        self.assertEqual(dict(state.heat_detector_states), {})
        self.assertIsNone(state.facp_status)
        self.assertIsNone(state.control_status)
        self.assertEqual(state.building_alarm_status, DetectorState.NORMAL)

    def test_missing_cameras_alone_is_safe_with_other_inputs_present(self):

        gateway = EstimatorBuildingStateGateway(
            smoke_detector_status_provider=lambda t: (_make_sensor_status("SMOKE-1"),),
        )

        state = gateway.collect(1.0)

        self.assertEqual(dict(state.camera_observations), {})
        self.assertEqual(len(state.smoke_detector_states), 1)

    def test_missing_sensors_alone_is_safe_with_other_inputs_present(self):

        gateway = EstimatorBuildingStateGateway(
            camera_status_provider=lambda t: (_make_camera_status("CAM-A"),),
        )

        state = gateway.collect(1.0)

        self.assertEqual(len(state.camera_observations), 1)
        self.assertEqual(dict(state.smoke_detector_states), {})
        self.assertEqual(dict(state.heat_detector_states), {})

    def test_missing_facp_alone_is_safe(self):

        gateway = EstimatorBuildingStateGateway(
            camera_status_provider=lambda t: (_make_camera_status("CAM-A"),),
        )

        state = gateway.collect(1.0)

        self.assertIsNone(state.facp_status)

    def test_missing_control_provider_alone_is_safe(self):

        gateway = EstimatorBuildingStateGateway(
            camera_status_provider=lambda t: (_make_camera_status("CAM-A"),),
        )

        state = gateway.collect(1.0)

        self.assertIsNone(state.control_status)


class EstimatorBuildingStateGatewayInputSeamTests(unittest.TestCase):

    # Phase 9 items 7/9/10/12/13 -- each individually-configured provider
    # reaches the right BuildingState field, and BuildingStateEstimator's
    # own aggregation logic is reused verbatim (not duplicated).

    def test_fusion_result_reaches_occupant_tracks(self):

        detections = _make_two_camera_four_detection_fixture()
        fusion_result = MultiCameraFusionEngine().fuse(detections, time=0.0)

        gateway = EstimatorBuildingStateGateway(
            fusion_result_provider=lambda t: fusion_result,
        )

        state = gateway.collect(0.0)

        self.assertEqual(len(state.occupant_tracks), 3)

    def test_four_raw_detections_fuse_to_three_building_state_occupants(self):

        # Phase 9 item 8 -- restates the CCTV milestone's own proven
        # "2 cameras, 4 raw detections, 3 physical occupants" scenario
        # (tests.test_cctv_offline_pipeline_validation.
        # BuildingStateNoDoubleCountingTests) through this new seam
        # specifically, not through BuildingStateEstimator directly.

        detections = _make_two_camera_four_detection_fixture()
        self.assertEqual(len(detections), 4)

        fusion_result = MultiCameraFusionEngine().fuse(detections, time=0.0)
        self.assertEqual(len(fusion_result.tracks), 3)

        gateway = EstimatorBuildingStateGateway(fusion_result_provider=lambda t: fusion_result)
        state = gateway.collect(0.0)

        self.assertEqual(len(state.occupant_tracks), 3)

    def test_camera_statuses_reach_camera_observations_and_active_assets(self):

        gateway = EstimatorBuildingStateGateway(
            camera_status_provider=lambda t: (
                _make_camera_status("CAM-A", active=True),
                _make_camera_status("CAM-B", active=False),
            ),
        )

        state = gateway.collect(0.0)

        self.assertEqual(set(state.camera_observations.keys()), {"CAM-A", "CAM-B"})
        self.assertIn("CAM-A", state.active_assets.active_camera_ids)
        self.assertIn("CAM-B", state.active_assets.offline_camera_ids)

    def test_sensor_statuses_reach_detector_states_and_active_assets(self):

        gateway = EstimatorBuildingStateGateway(
            smoke_detector_status_provider=lambda t: (_make_sensor_status("SMOKE-1", active=True),),
            heat_detector_status_provider=lambda t: (
                _make_sensor_status("HEAT-1", sensor_type="HeatDetector", active=False),
            ),
        )

        state = gateway.collect(0.0)

        self.assertIn("SMOKE-1", state.smoke_detector_states)
        self.assertIn("HEAT-1", state.heat_detector_states)
        self.assertIn("SMOKE-1", state.active_assets.active_sensor_ids)
        self.assertIn("HEAT-1", state.active_assets.offline_sensor_ids)

    def test_detector_alarm_state_reaches_building_alarm_status(self):

        from perception.models.smoke_detector_observation import SmokeDetectorReading

        gateway = EstimatorBuildingStateGateway(
            smoke_detector_status_provider=lambda t: (_make_sensor_status("SMOKE-1"),),
            smoke_detector_reading_provider=lambda t: (
                SmokeDetectorReading(detector_id="SMOKE-1", timestamp=t, alarm_active=True),
            ),
        )

        state = gateway.collect(0.0)

        self.assertEqual(state.building_alarm_status, DetectorState.ALARM)

    def test_facp_snapshot_reaches_facp_status(self):

        facp = SimulatedFACP(panel_id="FACP-TEST")
        facp.evaluate(
            {"SMOKE-1": DetectorConditionReport(
                asset_id="SMOKE-1", asset_type="SmokeDetector", state=DetectorState.ALARM,
            )},
            time=0.0,
        )

        gateway = EstimatorBuildingStateGateway(
            facp_snapshot_provider=lambda t: facp.current_snapshot(t),
        )

        state = gateway.collect(0.0)

        self.assertIsNotNone(state.facp_status)
        self.assertEqual(state.facp_status.panel_state, PanelState.ALARM)

    def test_control_snapshot_reaches_control_status_when_supplied(self):

        building = Building(id="b1", name="Test", floors=[
            Floor(id="floor-1", name="Ground", zones=[
                Zone(id="zone-a", name="Zone A", floor_id="floor-1", x=0, y=0, width=4, height=4),
            ]),
        ])

        controller = BuildingControlController(
            building, SimulationControlProvider(building), approval_mode=ApprovalMode.AUTO_APPROVE_SIMULATION,
        )
        controller.submit(ControlRequest(
            request_id="req-1", system_type=ControlSystemType.SMOKE_EXHAUST,
            target_id="zone-a", requested_action=ControlAction.ACTIVATE, timestamp=0.0,
            reason="smoke test", source=RequestSource.OPERATOR,
        ))

        gateway = EstimatorBuildingStateGateway(
            control_snapshot_provider=lambda t: controller.snapshot(),
        )

        state = gateway.collect(0.0)

        self.assertIsNotNone(state.control_status)
        self.assertEqual(len(state.control_status.entries), 1)


def _make_two_camera_four_detection_fixture():

    source_a = ReplayFrameSource(
        camera_id="CAM-A",
        frames=[(0.0, [{"local_track_id": "a1"}, {"local_track_id": "shared"}])],
    )
    source_b = ReplayFrameSource(
        camera_id="CAM-B",
        frames=[(0.0, [{"local_track_id": "b1"}, {"local_track_id": "shared"}])],
    )
    source_a.start()
    source_b.start()

    resolver = MappingIdentityResolver({
        ("CAM-A", "a1"): "P1",
        ("CAM-B", "b1"): "P2",
        ("CAM-A", "shared"): "P3",
        ("CAM-B", "shared"): "P3",
    })

    detection_provider = LiveCameraPipelineDetectionProvider()

    pipeline = LiveCameraPipeline(
        frame_sources={"CAM-A": source_a, "CAM-B": source_b},
        human_detector=MockHumanDetector(),
        identity_resolver=resolver,
        detection_provider=detection_provider,
    )

    pipeline.run_cycle(0.0)

    detections = []
    for camera_id in ("CAM-A", "CAM-B"):
        detections.extend(detection_provider.detections_at(camera_id, 0.0))

    return tuple(detections)


# =====================================================
# StateManager -- canonical state storage
# =====================================================


class StateManagerBuildingStateTests(unittest.TestCase):

    def test_starts_with_no_building_state(self):

        manager = StateManager()

        self.assertIsNone(manager.latest_building_state())
        self.assertIsNone(manager.current().building_state)

    def test_update_building_state_stores_the_latest_canonical_state(self):

        manager = StateManager()
        state = BuildingStateEstimator().estimate(
            1.0, hazard_snapshot=HazardSnapshot(), occupancy_snapshot=OccupancySnapshot(),
        )

        snapshot = manager.update_building_state(state, time=1.0)

        self.assertIs(manager.latest_building_state(), state)
        self.assertIs(snapshot.building_state, state)
        self.assertEqual(snapshot.component_timestamps["building_state"], 1.0)

    def test_multiple_cycles_replace_the_canonical_snapshot(self):

        manager = StateManager()
        estimator = BuildingStateEstimator()

        first = estimator.estimate(1.0, hazard_snapshot=HazardSnapshot(), occupancy_snapshot=OccupancySnapshot())
        second = estimator.estimate(2.0, hazard_snapshot=HazardSnapshot(), occupancy_snapshot=OccupancySnapshot())

        manager.update_building_state(first, time=1.0)
        manager.update_building_state(second, time=2.0)

        self.assertIs(manager.latest_building_state(), second)
        self.assertIsNot(manager.latest_building_state(), first)

    def test_no_parallel_independently_computed_state_exists(self):

        # Phase 9 item 4 -- updating building_state must never touch
        # building_observation (Live Perception's own, separate,
        # pre-existing compatibility field), and vice versa. There is
        # exactly one canonical BuildingState field on LiveBuildingSnapshot,
        # never a second, independently-recomputed one.

        from perception.models.building_observation import BuildingObservation

        manager = StateManager()

        state = BuildingStateEstimator().estimate(
            1.0, hazard_snapshot=HazardSnapshot(), occupancy_snapshot=OccupancySnapshot(),
        )
        manager.update_building_state(state, time=1.0)

        self.assertIsNone(manager.current().building_observation)

        observation = BuildingObservation()
        manager.update_perception(observation, time=2.0)

        # The earlier building_state update must survive an unrelated
        # update_perception() call untouched -- StateManager.replace()
        # carries every field forward except the one just changed.
        self.assertIs(manager.current().building_state, state)
        self.assertIs(manager.current().building_observation, observation)

    def test_existing_perception_based_consumers_are_unaffected(self):

        # Phase 9 item 5 -- the pre-existing building_observation/
        # ai_predictions/decision_policy/recommendations fields and
        # their own update_*() methods must keep working exactly as
        # before this milestone.

        from perception.models.building_observation import BuildingObservation

        manager = StateManager()
        observation = BuildingObservation()

        snapshot = manager.update_perception(observation, time=3.0)

        self.assertIs(snapshot.building_observation, observation)
        self.assertIsNone(snapshot.building_state)


# =====================================================
# LiveOrchestrator -- assembly wiring
# =====================================================


class _StubBuildingStateGateway:

    def __init__(self, state_factory):
        self.calls = []
        self._state_factory = state_factory

    def collect(self, time: float) -> BuildingState:
        self.calls.append(time)
        return self._state_factory(time)


class LiveOrchestratorBuildingStateAssemblyTests(unittest.TestCase):

    def test_run_cycle_produces_a_building_state_when_gateway_configured(self):

        gateway = _StubBuildingStateGateway(
            lambda t: BuildingStateEstimator().estimate(
                t, hazard_snapshot=HazardSnapshot(), occupancy_snapshot=OccupancySnapshot(),
            ),
        )
        orchestrator = LiveOrchestrator(building_state_gateway=gateway)
        orchestrator.start()

        snapshot = orchestrator.run_cycle(5.0)

        self.assertIsInstance(snapshot.building_state, BuildingState)
        self.assertEqual(gateway.calls, [5.0])
        self.assertIs(orchestrator.latest_building_state, snapshot.building_state)

    def test_run_cycle_without_a_gateway_leaves_building_state_none(self):

        orchestrator = LiveOrchestrator()
        orchestrator.start()

        snapshot = orchestrator.run_cycle(0.0)

        self.assertIsNone(snapshot.building_state)
        self.assertIsNone(orchestrator.latest_building_state)

    def test_building_state_updated_event_is_published_when_gateway_configured(self):

        bus = EventBus()
        received = []
        bus.subscribe(EventType.BUILDING_STATE_UPDATED, received.append)

        gateway = _StubBuildingStateGateway(
            lambda t: BuildingStateEstimator().estimate(
                t, hazard_snapshot=HazardSnapshot(), occupancy_snapshot=OccupancySnapshot(),
            ),
        )
        orchestrator = LiveOrchestrator(event_bus=bus, building_state_gateway=gateway)
        orchestrator.start()
        orchestrator.run_cycle(0.0)

        self.assertEqual(len(received), 1)
        self.assertIsInstance(received[0].payload, BuildingState)

    def test_building_state_updated_event_is_not_published_without_a_gateway(self):

        bus = EventBus()
        received = []
        bus.subscribe(EventType.BUILDING_STATE_UPDATED, received.append)

        orchestrator = LiveOrchestrator(event_bus=bus)
        orchestrator.start()
        orchestrator.run_cycle(0.0)

        self.assertEqual(received, [])

    def test_building_state_gateway_is_independent_of_perception_gateway(self):

        # Either, both, or neither may be configured -- this test proves
        # a building_state_gateway alone (no perception_gateway) still
        # runs cleanly.

        gateway = _StubBuildingStateGateway(
            lambda t: BuildingStateEstimator().estimate(
                t, hazard_snapshot=HazardSnapshot(), occupancy_snapshot=OccupancySnapshot(),
            ),
        )
        orchestrator = LiveOrchestrator(building_state_gateway=gateway)
        orchestrator.start()

        snapshot = orchestrator.run_cycle(0.0)

        self.assertIsNone(snapshot.building_observation)
        self.assertIsNotNone(snapshot.building_state)

    def test_repeated_deterministic_cycles_produce_deterministic_building_state_content(self):

        # Phase 9 item 18 -- aside from BuildingState.state_id (a fresh
        # uuid4 every estimate() call, by that type's own explicit
        # design -- see building_state/models.py), repeated cycles over
        # identical inputs must produce identical BuildingState content.

        def build_and_run():

            gateway = EstimatorBuildingStateGateway(
                camera_status_provider=lambda t: (_make_camera_status("CAM-A"),),
            )
            orchestrator = LiveOrchestrator(building_state_gateway=gateway, interval_seconds=1.0)
            orchestrator.start()

            return orchestrator.update_loop.run_for(3, start_time=0.0)

        first_run = build_and_run()
        second_run = build_and_run()

        self.assertEqual(
            [tuple(s.building_state.camera_observations.keys()) for s in first_run],
            [tuple(s.building_state.camera_observations.keys()) for s in second_run],
        )
        self.assertEqual(
            [s.building_state.timestamp for s in first_run],
            [s.building_state.timestamp for s in second_run],
        )

    def test_run_cycle_still_refuses_before_start_with_a_gateway_configured(self):

        # Phase 9 item 19 -- unchanged lifecycle discipline.

        gateway = _StubBuildingStateGateway(lambda t: BuildingStateEstimator().estimate(
            t, hazard_snapshot=HazardSnapshot(), occupancy_snapshot=OccupancySnapshot(),
        ))
        orchestrator = LiveOrchestrator(building_state_gateway=gateway)

        with self.assertRaises(LiveSystemNotRunningError):
            orchestrator.run_cycle(0.0)

        self.assertEqual(gateway.calls, [])

    def test_stop_then_run_cycle_raises_again_with_a_gateway_configured(self):

        # Phase 9 item 20 -- unchanged lifecycle discipline.

        gateway = _StubBuildingStateGateway(lambda t: BuildingStateEstimator().estimate(
            t, hazard_snapshot=HazardSnapshot(), occupancy_snapshot=OccupancySnapshot(),
        ))
        orchestrator = LiveOrchestrator(building_state_gateway=gateway)
        orchestrator.start()
        orchestrator.run_cycle(0.0)
        orchestrator.stop()

        with self.assertRaises(LiveSystemNotRunningError):
            orchestrator.run_cycle(1.0)


# =====================================================
# Phase 12 -- full deterministic integration smoke test
# =====================================================


class LiveBuildingStateSmokeTest(unittest.TestCase):

    # No network access. No real CCTV. No AI inference. No advisory
    # generation. Every collaborator below is real (CameraManager,
    # MultiCameraFusionEngine, SimulatedFACP, the offline-proven
    # live_camera_pipeline chain) except the physical camera feed
    # itself, which remains ReplayFrameSource -- exactly the CCTV
    # milestone's own "offline, deterministic" scenario, this time
    # driven through a real, started LiveOrchestrator instead of being
    # called directly.

    def setUp(self):

        self.building = Building(id="b1", name="Smoke Test Building", floors=[
            Floor(id="floor-1", name="Ground", zones=[
                Zone(id="zone-a", name="Zone A", floor_id="floor-1", x=0, y=0, width=4, height=4),
            ], cameras=[
                Camera(id="CAM-A", name="A", floor_id="floor-1", zone_ids=("zone-a",)),
                Camera(id="CAM-B", name="B", floor_id="floor-1", zone_ids=("zone-a",)),
            ], smoke_detectors=[
                SmokeDetector(id="SMOKE-1", name="Smoke 1", floor_id="floor-1", zone_ids=("zone-a",)),
            ]),
        ])

        self.camera_manager = CameraManager()
        self.camera_manager.discover_cameras(self.building)

        self.source_a = ReplayFrameSource(
            camera_id="CAM-A",
            frames=[(0.0, [{"local_track_id": "a1"}, {"local_track_id": "shared"}])],
        )
        self.source_b = ReplayFrameSource(
            camera_id="CAM-B",
            frames=[(0.0, [{"local_track_id": "b1"}, {"local_track_id": "shared"}])],
        )
        self.source_a.start()
        self.source_b.start()

        self.resolver = MappingIdentityResolver({
            ("CAM-A", "a1"): "P1",
            ("CAM-B", "b1"): "P2",
            ("CAM-A", "shared"): "P3",
            ("CAM-B", "shared"): "P3",
        })

        self.detection_provider = LiveCameraPipelineDetectionProvider()

        self.pipeline = LiveCameraPipeline(
            frame_sources={"CAM-A": self.source_a, "CAM-B": self.source_b},
            human_detector=MockHumanDetector(),
            identity_resolver=self.resolver,
            detection_provider=self.detection_provider,
        )

        self.fusion_engine = MultiCameraFusionEngine()
        self.facp = SimulatedFACP(panel_id="FACP-SMOKE-TEST")

        self._last_raw_detection_count = None

        def fusion_result_provider(time: float) -> FusionResult:

            self.pipeline.run_cycle(time)

            detections = []
            for camera_id in ("CAM-A", "CAM-B"):
                detections.extend(self.detection_provider.detections_at(camera_id, time))

            self._last_raw_detection_count = len(detections)

            return self.fusion_engine.fuse(detections, time)

        def facp_snapshot_provider(time: float):

            report = DetectorConditionReport(
                asset_id="SMOKE-1", asset_type="SmokeDetector", state=DetectorState.ALARM,
                floor_id="floor-1", zone_ids=("zone-a",),
            )
            self.facp.evaluate({"SMOKE-1": report}, time)

            return self.facp.current_snapshot(time)

        self.gateway = EstimatorBuildingStateGateway(
            camera_status_provider=lambda t: self.camera_manager.all_statuses(),
            fusion_result_provider=fusion_result_provider,
            facp_snapshot_provider=facp_snapshot_provider,
        )

        self.orchestrator = LiveOrchestrator(building_state_gateway=self.gateway)

    def test_deterministic_end_to_end_smoke(self):

        self.orchestrator.start()

        states = []

        for cycle_index in range(3):

            if cycle_index > 0:
                # ReplayFrameSource is single-pass by design -- reset()
                # is this test's own repeat-scenario harness, not a
                # change to that class (see its own docstring).
                self.source_a.reset()
                self.source_b.reset()

            snapshot = self.orchestrator.run_cycle(float(cycle_index))
            states.append(snapshot.building_state)

        # 7. Verify canonical BuildingState updates each cycle.
        for cycle_index, state in enumerate(states):
            self.assertIsInstance(state, BuildingState)
            self.assertEqual(state.timestamp, float(cycle_index))

        # 8. Verify fused occupant count is not double-counted.
        self.assertEqual(self._last_raw_detection_count, 4, "raw detections this cycle")
        self.assertEqual(len(states[-1].occupant_tracks), 3, "unique BuildingState occupants")

        # Camera asset status reaches BuildingState.
        self.assertEqual(set(states[-1].camera_observations.keys()), {"CAM-A", "CAM-B"})

        # 9. Verify FACP state is present.
        self.assertIsNotNone(states[-1].facp_status)
        self.assertEqual(states[-1].facp_status.panel_state, PanelState.ALARM)
        self.assertIn("SMOKE-1", states[-1].facp_status.active_alarm_source_ids)

        # 10. Stop the orchestrator cleanly.
        self.orchestrator.stop()
        self.assertFalse(self.orchestrator.is_running)

        with self.assertRaises(LiveSystemNotRunningError):
            self.orchestrator.run_cycle(3.0)


if __name__ == "__main__":
    unittest.main()
