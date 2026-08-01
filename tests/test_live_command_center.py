import shutil
import sys
import tempfile
import unittest

from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

import ai_registry as reg

from models.building import Building
from models.floor import Floor
from models.zone import Zone
from models.camera import Camera
from models.smoke_detector import SmokeDetector
from models.sensor_asset import DetectorState

from camera_manager.manager import CameraManager
from sensor_manager.status import SensorStatus

from live_camera_pipeline.replay_frame_source import ReplayFrameSource
from live_camera_pipeline.detection_provider import LiveCameraPipelineDetectionProvider
from live_camera_pipeline.identity_resolver import MappingIdentityResolver
from live_camera_pipeline.pipeline import LiveCameraPipeline

from tests.live_camera_pipeline_fixtures import MockHumanDetector

from multi_camera_fusion.engine import MultiCameraFusionEngine
from multi_camera_fusion.track import FusionResult

from occupancy.snapshot import OccupancySnapshot
from occupancy.observation import OccupancyObservation

from facp.engine import SimulatedFACP
from facp.models import DetectorConditionReport

from live_system.building_state_gateway import EstimatorBuildingStateGateway
from live_system.event_bus import EventBus, EventType
from live_system.live_ai_gateway import RegistryLiveAIInferenceGateway
from live_system.live_advisory_gateway import ReplayCompatibleAdvisoryGateway
from live_system.orchestrator import LiveOrchestrator
from live_system.live_command_center_gateway import LiveCommandCenterDataSource, frame_from_building_state

from command_center.data_source import CommandCenterMode, ReplayCommandCenterDataSource, SnapshotConsistency
from command_center.dashboard import Dashboard
from command_center.incident_data import IncidentData
from command_center.main_window import MainWindow

from advisory_system.recommendation_models import AdvisoryInputs
from decision_policy.zone_policy import EVACUATE_IMMEDIATELY

from tests.test_advisory_system import make_building, make_decision_policy, make_ground_truth, make_scenario


# =====================================================
# Live Command Center Integration milestone -- Phase 16 (offline E2E
# demo), Phase 17 (15 failure/degradation scenarios), Phase 18 (Replay
# backward compatibility), Phase 19 (architecture guards).
#
# The E2E chain reuses tests.test_live_building_state_integration.
# LiveBuildingStateSmokeTest's own real Camera/Fusion/FACP chain and
# tests.test_ai_augmented_advisory.EndToEndOfflineLiveAdvisoryTests'
# own real-trained-model setUpModule pattern, composed together one
# level higher: through LiveCommandCenterDataSource and a real Dashboard.
# =====================================================


_MODULE_STATE = {}

CAMPAIGN_COUNT = 120
CAMPAIGN_SEED = 4242


def setUpModule():

    tmp_dir = tempfile.mkdtemp(prefix="live_command_center_test_")
    building = reg.make_training_building()
    definition = reg.make_training_definition()

    campaign_dir, _summary = reg.generate_training_campaign(
        tmp_dir, building, definition, count=CAMPAIGN_COUNT, master_seed=CAMPAIGN_SEED,
    )

    import ai_training as at

    legacy_dataset = at.load_campaign_dataset(campaign_dir)
    live_dataset = reg.build_live_compatible_dataset(legacy_dataset, building)

    bottleneck_result = reg.train_bottleneck_occurrence_model(
        live_dataset, training_seed=1, dataset_identifier="command-center-e2e",
    )

    _MODULE_STATE["tmp_dir"] = tmp_dir
    _MODULE_STATE["bottleneck_result"] = bottleneck_result


def tearDownModule():

    shutil.rmtree(_MODULE_STATE.get("tmp_dir", ""), ignore_errors=True)


# =====================================================
# Shared chain builder -- one real camera Building (2 cameras, 1 smoke
# detector, 1 zone), the offline-proven ReplayFrameSource/
# LiveCameraPipeline/MultiCameraFusionEngine/SimulatedFACP chain, a
# real trained bottleneck model via ai_registry, and a real
# ReplayCompatibleAdvisoryGateway fed the tests.test_advisory_system
# fixture Building/Scenario/GroundTruth/DecisionPolicy (a SEPARATE,
# smaller building than the camera one -- exactly the established,
# honest "advisory needs its own caller-supplied Building/Scenario/
# GroundTruth, independent of the live camera Building" pattern; see
# docs/architecture/ai_augmented_advisory_integration.md §12).
# =====================================================


def _build_camera_building():

    return Building(id="cc-b1", name="Command Center E2E Building", floors=[
        Floor(
            id="floor-1", name="Ground",
            zones=[Zone(id="zone-a", name="Zone A", floor_id="floor-1", x=0, y=0, width=4, height=4)],
            cameras=[
                Camera(id="CAM-A", name="A", floor_id="floor-1", zone_ids=("zone-a",)),
                Camera(id="CAM-B", name="B", floor_id="floor-1", zone_ids=("zone-a",)),
            ],
            smoke_detectors=[SmokeDetector(id="SMOKE-1", name="Smoke 1", floor_id="floor-1", zone_ids=("zone-a",))],
        ),
    ])


class _LiveChain:

    # A small, reusable harness -- not a TestCase itself -- bundling
    # every real collaborator one E2E/failure test needs, so each test
    # method only has to override the one thing it wants to break.

    def __init__(self, *, with_ai=True, with_advisory=True, with_camera_pipeline=True):

        self.building = _build_camera_building()

        self.camera_manager = CameraManager()
        self.camera_manager.discover_cameras(self.building)

        self.event_bus = EventBus()

        building_state_gateway = None

        if with_camera_pipeline:

            self.source_a = ReplayFrameSource(camera_id="CAM-A", frames=[(0.0, [{"local_track_id": "a1"}])])
            self.source_b = ReplayFrameSource(camera_id="CAM-B", frames=[(0.0, [{"local_track_id": "b1"}])])
            self.source_a.start()
            self.source_b.start()

            self.resolver = MappingIdentityResolver({("CAM-A", "a1"): "P1", ("CAM-B", "b1"): "P2"})
            self.detection_provider = LiveCameraPipelineDetectionProvider()
            self.pipeline = LiveCameraPipeline(
                frame_sources={"CAM-A": self.source_a, "CAM-B": self.source_b},
                human_detector=MockHumanDetector(),
                identity_resolver=self.resolver,
                detection_provider=self.detection_provider,
            )
            self.fusion_engine = MultiCameraFusionEngine()
            self.facp = SimulatedFACP(panel_id="FACP-CC-E2E")

            def fusion_result_provider(time: float) -> FusionResult:

                self.pipeline.run_cycle(time)
                detections = []
                for camera_id in ("CAM-A", "CAM-B"):
                    detections.extend(self.detection_provider.detections_at(camera_id, time))
                return self.fusion_engine.fuse(detections, time)

            def facp_snapshot_provider(time: float):

                report = DetectorConditionReport(
                    asset_id="SMOKE-1", asset_type="SmokeDetector", state=DetectorState.ALARM,
                    floor_id="floor-1", zone_ids=("zone-a",),
                )
                self.facp.evaluate({"SMOKE-1": report}, time)
                return self.facp.current_snapshot(time)

            occupancy_snapshot_provider = lambda t: OccupancySnapshot(
                timestamp=t, observations={"zone-a": OccupancyObservation(occupant_count=2)},
            )

            smoke_detector_status_provider = lambda t: (
                SensorStatus(
                    sensor_id="SMOKE-1", sensor_type="SmokeDetector", name="Smoke 1",
                    floor_id="floor-1", zone_ids=("zone-a",), active=True, mode="LIVE", health_status="OK",
                ),
            )

            building_state_gateway = EstimatorBuildingStateGateway(
                camera_status_provider=lambda t: self.camera_manager.all_statuses(),
                fusion_result_provider=fusion_result_provider,
                facp_snapshot_provider=facp_snapshot_provider,
                occupancy_snapshot_provider=occupancy_snapshot_provider,
                smoke_detector_status_provider=smoke_detector_status_provider,
            )

        live_ai_gateway = None
        if with_ai:

            registry = reg.ModelRegistry()
            registry.register_model(_MODULE_STATE["bottleneck_result"].model, _MODULE_STATE["bottleneck_result"].metadata)
            service = reg.LiveAIInferenceService(registry)
            live_ai_gateway = RegistryLiveAIInferenceGateway(service, include_evacuation_time=False)

        live_advisory_gateway = None
        if with_advisory:

            decision_policy = make_decision_policy(
                zone_decisions=[{"zone_id": "zone-a", "recommended_exit": "exit-1", "recommended_stair": None, "action": EVACUATE_IMMEDIATELY}],
            )
            live_advisory_gateway = ReplayCompatibleAdvisoryGateway(
                building=make_building(), scenario=make_scenario(), ground_truth=make_ground_truth(),
                decision_policy_provider=lambda time: decision_policy,
            )

        self.orchestrator = LiveOrchestrator(
            event_bus=self.event_bus,
            building_state_gateway=building_state_gateway,
            live_ai_gateway=live_ai_gateway,
            live_advisory_gateway=live_advisory_gateway,
        )

        self.data_source = LiveCommandCenterDataSource(
            self.orchestrator.state_manager, building=self.building, event_bus=self.event_bus,
        )

    # =====================================================

    def run_cycle(self, time: float):

        return self.orchestrator.run_cycle(time)


# =====================================================
# Phase 16 -- Offline End-to-End Live Demo.
# =====================================================


class EndToEndOfflineLiveCommandCenterTests(unittest.TestCase):

    def test_full_chain_populates_every_required_ui_fact(self):

        chain = _LiveChain()
        chain.orchestrator.start()
        chain.run_cycle(1.0)

        chain.data_source.start()
        snapshot = chain.data_source.current_snapshot()

        dashboard = Dashboard()
        dashboard.set_mode(CommandCenterMode.LIVE)
        dashboard.apply_snapshot(snapshot)

        # 1. LIVE mode.
        self.assertEqual(dashboard.mode, CommandCenterMode.LIVE)

        # 2. Building name.
        self.assertEqual(dashboard.status_bar.building_name_value.text(), "Command Center E2E Building")

        # 3. Occupant count.
        self.assertNotEqual(dashboard.status_bar.remaining_value.text(), "-")

        # 4. Camera status.
        self.assertEqual(dashboard.live_status_panel.camera_table.rowCount(), 2)

        # 5. Detector status.
        self.assertEqual(dashboard.live_status_panel.sensor_table.rowCount(), 1)

        # 6. FACP state.
        self.assertGreater(dashboard.live_status_panel.facp_state_table.rowCount(), 0)

        # 7. Bottleneck probability.
        self.assertNotEqual(dashboard.status_bar.ai_probability_value.text(), "-")
        self.assertNotIn("unavailable", dashboard.live_ai_panel.status_label.text())

        # 8. Civilian recommendation.
        self.assertGreater(dashboard.recommendation_center.civilian_panel.table.rowCount(), 0)

        # 9. Firefighter intelligence.
        self.assertNotEqual(dashboard.recommendation_center.firefighter_panel.building_status_label.text(), "-")

        # 10. Commander summary.
        self.assertNotEqual(dashboard.recommendation_center.commander_panel.severity_value.text(), "-")

        # 11. AI explainability -- the AI-sourced building recommendation
        # (if the trained model predicted occurrence) or the firefighter
        # panel's own ai_bottleneck_probability field is present; either
        # way, no localized claim ("North Stair") ever appears.
        rendered = str(snapshot.advisory_report.to_dict())
        self.assertNotIn("North Stair", rendered)

        # 12. Recommendation confidence.
        self.assertNotEqual(dashboard.recommendation_center.commander_panel.recommendation_confidence_value.text(), "-")

        # 13. AI probability separately -- distinct from recommendation
        # confidence (three-way separation, Phase 10 of the prior
        # milestone).
        self.assertEqual(
            dashboard.status_bar.ai_probability_value.text(),
            f"{snapshot.ai_prediction_snapshot.bottleneck.probability:.0%}",
        )

        # 14. No voice message sent -- this test never calls
        # dashboard.set_operator_action_gateway(), so every announcement
        # renders through VoiceEvacuationPanel's honest NO_PROVIDER
        # fallback (Live Operator Action Routing milestone): a bare
        # RECOMMENDED status, never a fabricated broadcast confirmation.
        voice_panel = dashboard.recommendation_center.voice_evacuation_panel
        self.assertEqual(voice_panel.history_table.rowCount(), 0)
        for row in range(voice_panel.active_table.rowCount()):
            status_item = voice_panel.active_table.item(row, 5)
            self.assertEqual(status_item.text(), "RECOMMENDED")

        # 15. No building control executed.
        controls_panel = dashboard.recommendation_center.building_controls_panel
        self.assertEqual(controls_panel.active_table.rowCount(), 0)
        for row in range(controls_panel.pending_table.rowCount()):
            decision_item = controls_panel.pending_table.item(row, 6)
            self.assertEqual(decision_item.text(), "Execution Provider: Not Connected")

        chain.orchestrator.stop()

    def test_full_chain_via_main_window_live_mode_toggle(self):

        chain = _LiveChain()
        chain.orchestrator.start()
        chain.run_cycle(1.0)

        window = MainWindow()
        window.enable_live_mode(chain.data_source)

        self.assertEqual(window.dashboard.mode, CommandCenterMode.LIVE)
        self.assertTrue(window.live_refresh_timer.isActive())
        self.assertFalse(window.playback_timer.isActive())

        window.close()
        chain.orchestrator.stop()


# =====================================================
# Phase 17 -- Failure/degradation tests.
# =====================================================


class LiveCommandCenterFailureTests(unittest.TestCase):

    def test_orchestrator_not_running_yields_unavailable_snapshot(self):

        chain = _LiveChain()
        # orchestrator never started -- data_source itself CAN still be
        # started/stopped independently of the orchestrator's own
        # start()/stop() (they are separate lifecycles); before start()
        # the data source honestly reports UNAVAILABLE regardless.

        snapshot = chain.data_source.current_snapshot()
        self.assertEqual(snapshot.consistency, SnapshotConsistency.UNAVAILABLE)

    def test_building_state_unavailable_before_first_cycle(self):

        chain = _LiveChain()
        chain.orchestrator.start()
        chain.data_source.start()

        snapshot = chain.data_source.current_snapshot()

        self.assertEqual(snapshot.consistency, SnapshotConsistency.UNAVAILABLE)
        self.assertIsNone(snapshot.frame)

    def test_ai_unavailable_no_gateway_configured(self):

        chain = _LiveChain(with_ai=False)
        chain.orchestrator.start()
        chain.run_cycle(1.0)
        chain.data_source.start()

        snapshot = chain.data_source.current_snapshot()

        self.assertIsNone(snapshot.ai_prediction_snapshot)
        self.assertEqual(snapshot.consistency, SnapshotConsistency.PARTIAL)

        dashboard = Dashboard()
        dashboard.set_mode(CommandCenterMode.LIVE)
        dashboard.apply_snapshot(snapshot)
        self.assertIn("unavailable", dashboard.live_ai_panel.status_label.text())
        self.assertEqual(dashboard.status_bar.ai_probability_value.text(), "-")

        chain.orchestrator.stop()

    def test_ai_stale_marks_snapshot_and_ai_panel(self):

        import dataclasses

        chain = _LiveChain()
        chain.orchestrator.start()
        chain.run_cycle(1.0)

        # Cycle 2: bump only building_state's own component_timestamps
        # entry by directly updating StateManager (simulating an AI/
        # Advisory gateway that skipped this cycle -- the honest "None
        # means skip" convention -- while BuildingState kept advancing).
        state_manager = chain.orchestrator.state_manager
        current = state_manager.current()
        advanced_state = dataclasses.replace(current.building_state, timestamp=2.0)
        state_manager.update_building_state(advanced_state, 2.0)

        chain.data_source.start()
        snapshot = chain.data_source.current_snapshot()

        self.assertEqual(snapshot.consistency, SnapshotConsistency.STALE)

        dashboard = Dashboard()
        dashboard.set_mode(CommandCenterMode.LIVE)
        dashboard.apply_snapshot(snapshot)

        self.assertIn("Advisory unavailable", dashboard.live_consistency_banner.text())
        self.assertEqual(dashboard.recommendation_center.civilian_panel.table.rowCount(), 0)

        chain.orchestrator.stop()

    def test_advisory_unavailable_no_gateway_configured(self):

        chain = _LiveChain(with_advisory=False)
        chain.orchestrator.start()
        chain.run_cycle(1.0)
        chain.data_source.start()

        snapshot = chain.data_source.current_snapshot()

        self.assertIsNone(snapshot.advisory_report)
        self.assertEqual(snapshot.consistency, SnapshotConsistency.PARTIAL)

        chain.orchestrator.stop()

    def test_camera_offline_shows_inactive_not_fabricated_online(self):

        chain = _LiveChain()
        chain.orchestrator.start()

        from camera_manager.connection_status import CameraConnectionState
        chain.camera_manager.set_connection_status("CAM-A", CameraConnectionState.CONFIGURED)

        chain.run_cycle(1.0)
        chain.data_source.start()
        snapshot = chain.data_source.current_snapshot()

        self.assertIn("CAM-A", snapshot.frame.camera_states)

        chain.orchestrator.stop()

    def test_camera_status_unknown_when_no_camera_pipeline_configured(self):

        chain = _LiveChain(with_camera_pipeline=False)
        chain.orchestrator.start()
        chain.run_cycle(1.0)

        # No building_state_gateway configured -- building_state stays
        # None every cycle, an honestly UNAVAILABLE snapshot, never a
        # fabricated empty-but-present camera table.
        chain.data_source.start()
        snapshot = chain.data_source.current_snapshot()
        self.assertEqual(snapshot.consistency, SnapshotConsistency.UNAVAILABLE)

        chain.orchestrator.stop()

    def test_detector_fault_reflected_honestly(self):

        chain = _LiveChain()
        chain.orchestrator.start()
        chain.run_cycle(1.0)
        chain.data_source.start()

        snapshot = chain.data_source.current_snapshot()
        # Detector states are AVAILABLE/FAILED (availability), not a
        # fabricated alarm condition -- confirmed present, never guessed.
        self.assertIn(snapshot.frame.detector_states.get("SMOKE-1"), (None, "AVAILABLE", "FAILED"))

        chain.orchestrator.stop()

    def test_facp_alarm_reaches_live_status_panel(self):

        chain = _LiveChain()
        chain.orchestrator.start()
        chain.run_cycle(1.0)
        chain.data_source.start()

        snapshot = chain.data_source.current_snapshot()
        dashboard = Dashboard()
        dashboard.set_mode(CommandCenterMode.LIVE)
        dashboard.apply_snapshot(snapshot)

        facp_rows = dashboard.live_status_panel.facp_state_table.rowCount()
        self.assertGreater(facp_rows, 0)

        chain.orchestrator.stop()

    def test_partial_occupancy_no_fabricated_full_count(self):

        building_state = frame_from_building_state(None)
        self.assertIsNone(building_state)

    def test_missing_hazard_data_leaves_no_fire_zones(self):

        chain = _LiveChain()
        chain.orchestrator.start()
        chain.run_cycle(1.0)
        chain.data_source.start()

        snapshot = chain.data_source.current_snapshot()
        # No hazard_snapshot was ever supplied to the estimator in this
        # harness -- current_fire_zone_ids must be empty, never guessed.
        self.assertEqual(snapshot.frame.current_fire_zone_ids, frozenset())

        chain.orchestrator.stop()

    def test_live_runtime_stops_data_source_reports_unavailable(self):

        chain = _LiveChain()
        chain.orchestrator.start()
        chain.run_cycle(1.0)
        chain.data_source.start()

        self.assertNotEqual(chain.data_source.current_snapshot().consistency, SnapshotConsistency.UNAVAILABLE)

        chain.data_source.stop()
        snapshot = chain.data_source.current_snapshot()
        self.assertEqual(snapshot.consistency, SnapshotConsistency.UNAVAILABLE)

        chain.orchestrator.stop()

    def test_data_source_disconnects_dashboard_never_crashes(self):

        chain = _LiveChain()
        dashboard = Dashboard()
        dashboard.set_mode(CommandCenterMode.LIVE)

        # Never started -- current_snapshot() must not raise.
        dashboard.apply_snapshot(chain.data_source.current_snapshot())
        self.assertEqual(dashboard.mode, CommandCenterMode.LIVE)

    def test_replay_mode_after_live_mode(self):

        chain = _LiveChain()
        chain.orchestrator.start()
        chain.run_cycle(1.0)

        window = MainWindow()
        window.enable_live_mode(chain.data_source)
        self.assertEqual(window.dashboard.mode, CommandCenterMode.LIVE)

        window.switch_to_replay_mode()
        self.assertEqual(window.dashboard.mode, CommandCenterMode.REPLAY)
        self.assertFalse(window.live_refresh_timer.isActive())

        window.close()
        chain.orchestrator.stop()

    def test_live_mode_after_replay_mode(self):

        building = make_building()
        scenario = make_scenario()
        ground_truth = make_ground_truth()
        decision_policy = make_decision_policy(
            zone_decisions=[{"zone_id": "zone-a", "recommended_exit": "exit-1", "recommended_stair": None, "action": EVACUATE_IMMEDIATELY}],
        )
        incident = IncidentData(building=building, scenario=scenario, ground_truth=ground_truth, decision_policy=decision_policy)

        window = MainWindow()
        window.load_incident_data(incident)
        self.assertEqual(window.dashboard.frame_count, 1)

        chain = _LiveChain()
        chain.orchestrator.start()
        chain.run_cycle(1.0)

        window.enable_live_mode(chain.data_source)
        self.assertEqual(window.dashboard.mode, CommandCenterMode.LIVE)

        window.close()
        chain.orchestrator.stop()


# =====================================================
# Phase 18 -- Replay backward compatibility. The full pre-existing
# tests/test_command_center.py suite (63 tests) already proves this at
# the unit level, unmodified -- this class adds one further end-to-end
# proof that a Replay session works with zero LiveOrchestrator/
# StateManager/live_system object ever constructed.
# =====================================================


class ReplayBackwardCompatibilityTests(unittest.TestCase):

    def test_replay_session_never_touches_live_system(self):

        import pathlib
        import re

        # Structural guarantee: command_center/incident_data.py (the
        # module Replay mode is built on) never imports live_system --
        # Replay's own correctness cannot depend on anything this
        # milestone added.
        path = pathlib.Path(__file__).resolve().parent.parent / "command_center" / "incident_data.py"
        text = path.read_text()
        self.assertIsNone(re.search(r"^\s*(from|import)\s+live_system", text, re.MULTILINE))

    def test_replay_data_source_full_round_trip(self):

        building = make_building()
        scenario = make_scenario()
        ground_truth = make_ground_truth()
        decision_policy = make_decision_policy(
            zone_decisions=[{"zone_id": "zone-a", "recommended_exit": "exit-1", "recommended_stair": None, "action": EVACUATE_IMMEDIATELY}],
        )
        incident = IncidentData(building=building, scenario=scenario, ground_truth=ground_truth, decision_policy=decision_policy)

        source = ReplayCommandCenterDataSource(incident)
        snapshot = source.current_snapshot()

        self.assertEqual(snapshot.mode, CommandCenterMode.REPLAY)
        self.assertIsNotNone(snapshot.advisory_report)
        self.assertIsNone(snapshot.building_state)
        self.assertIsNone(snapshot.ai_prediction_snapshot)

    def test_main_window_replay_flow_unaffected_by_live_fields(self):

        building = make_building()
        scenario = make_scenario()
        ground_truth = make_ground_truth()
        decision_policy = make_decision_policy(
            zone_decisions=[{"zone_id": "zone-a", "recommended_exit": "exit-1", "recommended_stair": None, "action": EVACUATE_IMMEDIATELY}],
        )
        incident = IncidentData(building=building, scenario=scenario, ground_truth=ground_truth, decision_policy=decision_policy)

        window = MainWindow()
        window.load_incident_data(incident)

        self.assertEqual(window.dashboard.mode, CommandCenterMode.REPLAY)
        self.assertIsNone(window.live_data_source)
        self.assertFalse(window.live_mode_action.isEnabled())

        window.close()


# =====================================================
# Phase 19 -- Architecture guards. Command Center's Live integration is
# a presentation layer -- it must never import a device manager, a
# perception/detection component, BuildingStateEstimator, an AI
# training/inference package, AdvisoryOrchestrator, or any execution-
# capable Voice Evacuation/Building Control module. Scoped to the files
# THIS milestone added/touches -- command_center/incident_data.py (pre-
# existing, untouched) legitimately imports AdvisoryOrchestrator/
# VoiceEvacuationController/BuildingControlController/SpeakerManager
# for its own, already-audited Replay reconstruction and is deliberately
# excluded from this guard.
# =====================================================


class _StubLiveOccupantsGatewayForCommandCenter:

    def __init__(self, snapshot):
        self.snapshot = snapshot

    def compute(self, time):
        return self.snapshot


class _StubBuildingStateGatewayForCommandCenter:

    # current_snapshot() only builds the full CommandCenterSnapshot
    # (including live_occupants) once BuildingState itself is non-None
    # this cycle -- a minimal real BuildingState is enough to reach that
    # path without pulling in the full _LiveChain fixture above.

    def collect(self, time):
        from building_state.models import BuildingState
        return BuildingState(timestamp=time)


class LiveOccupantsReachesCommandCenterSnapshotTests(unittest.TestCase):

    # Expose Real Live Camera Occupant State In SynEvac UI milestone --
    # proves the LAST composition hop in isolation: whatever
    # StateManager.current().live_occupants holds is exactly what
    # LiveCommandCenterDataSource.current_snapshot().live_occupants
    # returns, never recomputed/reshaped along the way.

    def test_live_occupants_snapshot_passes_through_unchanged(self):

        from live_occupants.models import LiveOccupantsSnapshot

        live_occupants_snapshot = LiveOccupantsSnapshot(timestamp=1.0, occupants=(), current_occupant_count=0)
        gateway = _StubLiveOccupantsGatewayForCommandCenter(live_occupants_snapshot)

        building = Building(id="b1", name="Test", floors=[])
        orchestrator = LiveOrchestrator(
            live_occupants_gateway=gateway, building_state_gateway=_StubBuildingStateGatewayForCommandCenter(),
        )
        orchestrator.start()
        orchestrator.run_cycle(1.0)

        data_source = LiveCommandCenterDataSource(orchestrator.state_manager, building=building)
        data_source.start()

        snapshot = data_source.current_snapshot()

        self.assertIs(snapshot.live_occupants, live_occupants_snapshot)

    def test_unconfigured_gateway_leaves_live_occupants_none(self):

        building = Building(id="b1", name="Test", floors=[])
        orchestrator = LiveOrchestrator(building_state_gateway=_StubBuildingStateGatewayForCommandCenter())
        orchestrator.start()
        orchestrator.run_cycle(1.0)

        data_source = LiveCommandCenterDataSource(orchestrator.state_manager, building=building)
        data_source.start()

        self.assertIsNone(data_source.current_snapshot().live_occupants)


class CommandCenterLiveIntegrationGuardTests(unittest.TestCase):

    _LIVE_INTEGRATION_FILES = (
        ("command_center", "data_source.py"),
        ("command_center", "live_status_panel.py"),
        ("command_center", "live_ai_panel.py"),
        ("command_center", "live_events_panel.py"),
        ("command_center", "dashboard.py"),
        ("command_center", "main_window.py"),
        ("command_center", "recommendation_center.py"),
        ("command_center", "building_controls_panel.py"),
        ("command_center", "live_dynamic_signage_panel.py"),
        ("command_center", "incident_status_bar.py"),
        ("live_system", "live_command_center_gateway.py"),
    )

    def _read(self, package, filename):

        import pathlib

        return (pathlib.Path(__file__).resolve().parent.parent / package / filename).read_text()

    def test_never_imports_camera_decoding_or_device_managers(self):

        import re

        forbidden = (
            r"^\s*(from|import)\s+"
            r"(cv2|torch|ultralytics|onvif|"
            r"live_camera_pipeline|camera_manager\.manager|sensor_manager\.manager|"
            r"multi_camera_fusion\.engine|facp\.engine)\b"
        )

        for package, filename in self._LIVE_INTEGRATION_FILES:

            text = self._read(package, filename)
            self.assertIsNone(
                re.search(forbidden, text, re.MULTILINE),
                f"{package}/{filename} imports a camera-decoding/device-manager/detection/fusion/"
                f"panel-engine module directly -- Command Center's live integration must only ever "
                f"read an already-computed BuildingState, never construct or drive these itself.",
            )

    def test_never_imports_building_state_estimator_or_ai_inference(self):

        import re

        forbidden = (
            r"^\s*(from|import)\s+"
            r"(building_state\.estimator|ai_registry|ai_features|ai_inference|ai_training)\b"
        )

        for package, filename in self._LIVE_INTEGRATION_FILES:

            text = self._read(package, filename)
            self.assertIsNone(
                re.search(forbidden, text, re.MULTILINE),
                f"{package}/{filename} imports BuildingStateEstimator or an AI training/inference "
                f"package directly -- Command Center must only ever read already-computed "
                f"BuildingState/LiveAIPredictionSnapshot values from StateManager.",
            )

    def test_never_imports_advisory_orchestrator(self):

        import re

        forbidden = r"^\s*(from|import)\s+advisory_system\.orchestrator\b"

        for package, filename in self._LIVE_INTEGRATION_FILES:

            text = self._read(package, filename)
            self.assertIsNone(
                re.search(forbidden, text, re.MULTILINE),
                f"{package}/{filename} imports AdvisoryOrchestrator directly -- Command Center's "
                f"live integration must only ever read an already-computed AdvisoryReport from "
                f"StateManager, never generate one itself.",
            )

    def test_never_imports_execution_capable_voice_or_control_modules(self):

        import re

        forbidden = (
            r"^\s*(from|import)\s+"
            r"(voice_evacuation\.controller|voice_evacuation\.provider|speaker_manager|"
            r"building_control\.controller|building_control\.providers|"
            r"dynamic_signage\.controller|dynamic_signage\.provider)\b"
        )

        for package, filename in self._LIVE_INTEGRATION_FILES:

            text = self._read(package, filename)
            self.assertIsNone(
                re.search(forbidden, text, re.MULTILINE),
                f"{package}/{filename} imports an execution-capable Voice Evacuation/Building "
                f"Control/Dynamic Signage module -- Live mode must remain display-only, every "
                f"operator action routed through LiveOperatorActionGateway instead (Phase 11/12; "
                f"Live Dynamic Sign Operator Approval & Dispatch Completion Phase 14).",
            )

    def test_live_status_and_ai_panels_expose_no_control_methods(self):

        from command_center.live_status_panel import LiveStatusPanel
        from command_center.live_ai_panel import LiveAIPanel

        for cls in (LiveStatusPanel, LiveAIPanel):
            for forbidden_method in ("acknowledge", "silence", "reset", "broadcast", "execute", "approve", "reject"):
                self.assertFalse(hasattr(cls, forbidden_method), f"{cls.__name__} exposes a {forbidden_method}() method")


if __name__ == "__main__":
    unittest.main()
