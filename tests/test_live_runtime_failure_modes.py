import unittest

from models.building import Building
from models.floor import Floor
from models.zone import Zone
from models.camera import Camera

from live_camera_pipeline.replay_frame_source import ReplayFrameSource
from live_camera_pipeline.identity_resolver import MappingIdentityResolver

from live_system.live_ai_gateway import AISystemStatus, RegistryLiveAIInferenceGateway
from live_system.live_advisory_gateway import ReplayCompatibleAdvisoryGateway

from ai_registry.registry import ModelRegistry
from ai_registry.inference_service import LiveAIInferenceService

from voice_evacuation.models import BroadcastStatus

from building_control.types import RequestStatus

from command_center.live_operator_action_gateway import (
    OperatorActionUnavailable,
    PROVIDER_CAPABILITY_NO_PROVIDER,
    VOICE_STATUS_FAILED,
    VOICE_STATUS_REJECTED,
)

from live_runtime.factory import build_live_runtime, build_offline_demo_runtime

from tests.live_runtime_fixtures import (
    make_demo_building,
    make_offline_frame_sources,
    make_offline_human_detector,
    make_offline_identity_resolver,
)
from tests.live_camera_pipeline_fixtures import MockHumanDetector


# =====================================================
# Production Live Runtime Composition Root milestone, Phase 8 -- the
# system must degrade honestly. Never fabricate Online, Sent,
# Confirmed, or AI Available states.
#
# Items 14-17 (runtime start called twice / stop called twice /
# component fails during startup / component fails during shutdown)
# are covered by tests/test_live_runtime.py::LifecycleTests and are not
# duplicated here.
# =====================================================


class NoCamerasTests(unittest.TestCase):

    def test_1_no_cameras_building_state_generation_still_succeeds(self):

        building = Building(name="B")
        building.create_floor(name="F1")

        runtime = build_offline_demo_runtime(building)
        runtime.start()

        snapshot = runtime.run_cycle(1.0)

        self.assertIsNotNone(snapshot.building_state)
        self.assertEqual(snapshot.building_state.camera_observations, {})
        self.assertIsNone(runtime.camera_pipeline)

        runtime.stop()


class CameraConfiguredButNoFrameSourceTests(unittest.TestCase):

    def test_2_camera_configured_but_no_frame_source_produces_no_detections_honestly(self):

        building = Building(id="b1", name="B", floors=[
            Floor(id="floor-1", name="F1", zones=[
                Zone(id="zone-a", name="Zone A", x=0, y=0, width=4, height=4, floor_id="floor-1"),
            ], cameras=[
                Camera(id="CAM-A", name="A", floor_id="floor-1", zone_ids=("zone-a",)),
                Camera(id="CAM-B", name="B", floor_id="floor-1", zone_ids=("zone-a",)),
            ]),
        ])

        # Only CAM-A gets a frame source -- CAM-B is a configured Camera
        # Asset with nothing behind it, an honest, common real-world gap.
        frame_sources = {"CAM-A": ReplayFrameSource(camera_id="CAM-A", frames=[(0.0, [{"local_track_id": "1"}])])}

        runtime = build_offline_demo_runtime(
            building, frame_sources=frame_sources, human_detector=MockHumanDetector(),
            identity_resolver=MappingIdentityResolver({("CAM-A", "1"): "P1"}),
        )
        runtime.start()

        snapshot = runtime.run_cycle(1.0)

        # Both cameras are still honestly reported as Digital Twin
        # assets (CameraManager discovers every Camera regardless of
        # whether a frame source backs it) -- but only CAM-A, the one
        # actually wired to LIVE mode with a provider, ever contributes
        # a detection. CAM-B stays Simulation-mode/no-provider, exactly
        # reflecting that nothing ever configured it for Live ingestion.
        cam_a_status = snapshot.building_state.camera_observations["CAM-A"].status
        cam_b_status = snapshot.building_state.camera_observations["CAM-B"].status

        self.assertEqual(cam_a_status.mode, "Live")
        self.assertTrue(cam_a_status.has_detection_provider)

        self.assertEqual(cam_b_status.mode, "Simulation")
        self.assertFalse(cam_b_status.has_detection_provider)

        self.assertEqual(len(snapshot.building_state.occupant_tracks), 1)

        runtime.stop()


class CameraOfflineTests(unittest.TestCase):

    def test_3_one_camera_offline_others_still_produce_detections(self):

        building = make_demo_building()
        frame_sources = make_offline_frame_sources(building)

        # CAM-CAFETERIA's own source has no frames at all -- "offline"
        # rendered as an honest, permanently-empty stream, never a crash.
        frame_sources["CAM-CAFETERIA"] = ReplayFrameSource(camera_id="CAM-CAFETERIA", frames=[])

        runtime = build_offline_demo_runtime(
            building, frame_sources=frame_sources, human_detector=make_offline_human_detector(),
            identity_resolver=make_offline_identity_resolver(building),
        )
        runtime.start()

        snapshot = runtime.run_cycle(1.0)

        self.assertEqual(len(snapshot.building_state.occupant_tracks), 1)  # only the Lobby camera's occupant

        runtime.stop()

    def test_4_all_cameras_offline_building_state_still_generates(self):

        building = make_demo_building()
        frame_sources = {
            camera_id: ReplayFrameSource(camera_id=camera_id, frames=[])
            for camera_id in make_offline_frame_sources(building)
        }

        runtime = build_offline_demo_runtime(
            building, frame_sources=frame_sources, human_detector=make_offline_human_detector(),
            identity_resolver=make_offline_identity_resolver(building),
        )
        runtime.start()

        snapshot = runtime.run_cycle(1.0)

        self.assertIsNotNone(snapshot.building_state)
        self.assertEqual(len(snapshot.building_state.occupant_tracks), 0)
        # Camera assets are still discovered/reported, just with no detections.
        self.assertEqual(set(snapshot.building_state.camera_observations.keys()), set(frame_sources.keys()))

        runtime.stop()


class NoDetectorsTests(unittest.TestCase):

    def test_5_no_detectors_building_state_generation_still_succeeds(self):

        building = Building(name="B")
        building.create_floor(name="F1")

        runtime = build_offline_demo_runtime(building)
        runtime.start()

        snapshot = runtime.run_cycle(1.0)

        self.assertEqual(snapshot.building_state.smoke_detector_states, {})
        self.assertEqual(snapshot.building_state.heat_detector_states, {})

        runtime.stop()


class FACPUnavailableTests(unittest.TestCase):

    def test_6_facp_unavailable_building_state_generation_still_succeeds(self):

        building = Building(name="B")
        building.create_floor(name="F1")

        runtime = build_offline_demo_runtime(building)  # facp=None (default)
        runtime.start()

        snapshot = runtime.run_cycle(1.0)

        self.assertIsNotNone(snapshot.building_state)
        self.assertIsNone(snapshot.building_state.facp_status)

        runtime.stop()


class AIUnavailableOrFailingTests(unittest.TestCase):

    def test_7_ai_registry_has_no_model_building_state_still_generates_honestly(self):

        building = Building(name="B")
        building.create_floor(name="F1")

        empty_registry = ModelRegistry()
        service = LiveAIInferenceService(empty_registry)
        live_ai_gateway = RegistryLiveAIInferenceGateway(service, include_evacuation_time=False)

        runtime = build_offline_demo_runtime(building, live_ai_gateway=live_ai_gateway)
        runtime.start()

        snapshot = runtime.run_cycle(1.0)

        self.assertIsNotNone(snapshot.building_state)  # unaffected by AI unavailability
        prediction = snapshot.ai_prediction_snapshot
        self.assertIsNotNone(prediction)
        self.assertEqual(prediction.system_status, AISystemStatus.UNAVAILABLE)
        self.assertIsNone(prediction.bottleneck)

        runtime.stop()

    def test_8_ai_inference_raises_internally_never_crashes_the_cycle(self):

        building = Building(name="B")
        building.create_floor(name="F1")

        class _BrokenModel:

            label_encoder = type("LabelEncoder", (), {"classes_": [False, True]})()

            def predict_proba(self, rows):
                raise RuntimeError("simulated corrupted model artifact")

        class _BrokenRegistry:

            def get_latest_compatible_model(self, model_type, feature_schema_version=None):

                if model_type != "bottleneck_occurrence":
                    return None

                from ai_registry.metadata import Deployability

                metadata = type("Metadata", (), {
                    "model_id": "broken-model", "model_type": "bottleneck_occurrence",
                    "model_version": "v0", "training_timestamp": "2026-01-01T00:00:00",
                    "feature_schema_version": feature_schema_version,
                    "ordered_feature_names": (), "model_deployability": Deployability.LIVE_COMPATIBLE,
                })()

                return (_BrokenModel(), metadata)

            def list_models(self, model_type=None, deployability=None):
                return []

        service = LiveAIInferenceService(_BrokenRegistry())
        live_ai_gateway = RegistryLiveAIInferenceGateway(service, include_evacuation_time=False)

        runtime = build_offline_demo_runtime(building, live_ai_gateway=live_ai_gateway)
        runtime.start()

        snapshot = runtime.run_cycle(1.0)  # must not raise

        self.assertIsNotNone(snapshot.building_state)
        self.assertIsNotNone(snapshot.ai_prediction_snapshot)
        self.assertIn(
            snapshot.ai_prediction_snapshot.system_status,
            (AISystemStatus.ERROR, AISystemStatus.INCOMPATIBLE),
        )
        self.assertIsNone(snapshot.ai_prediction_snapshot.bottleneck)

        runtime.stop()


class AdvisoryGenerationFailureTests(unittest.TestCase):

    def test_9_advisory_generation_failure_never_crashes_the_cycle(self):

        building = Building(name="B")
        building.create_floor(name="F1")

        def raising_decision_policy_provider(time):
            raise RuntimeError("simulated decision policy computation failure")

        live_advisory_gateway = ReplayCompatibleAdvisoryGateway(
            building=building, scenario=None, ground_truth=None,
            decision_policy_provider=raising_decision_policy_provider,
        )

        runtime = build_offline_demo_runtime(building, live_advisory_gateway=live_advisory_gateway)
        runtime.start()

        snapshot = runtime.run_cycle(1.0)  # must not raise

        self.assertIsNotNone(snapshot.building_state)  # BuildingState assembly unaffected
        self.assertIsNone(snapshot.advisory_report)  # honestly absent, never fabricated

        runtime.stop()


class NoOutputProviderTests(unittest.TestCase):

    def setUp(self):

        self.building = Building(name="B")
        self.building.create_floor(name="F1")

    def test_10_no_voice_provider_recommendations_still_displayed_execution_fails_honestly(self):

        runtime = build_live_runtime(self.building)  # no voice_output_provider, no building_control_provider

        self.assertEqual(runtime.operator_action_gateway.voice_capability, PROVIDER_CAPABILITY_NO_PROVIDER)
        self.assertIsNone(runtime.voice_evacuation_controller)

        from advisory_system.recommendation_models import CivilianAnnouncement

        announcement = CivilianAnnouncement(
            zone_id="zone-a", zone_name="Zone A", announcement="Evacuate now", reason="fire",
            confidence=0.9, predicted_rset_improvement_seconds=None,
        )

        # Reviewing the recommendation itself never requires a provider.
        self.assertEqual(
            runtime.operator_action_gateway.voice_recommendation_status(announcement), "RECOMMENDED",
        )

        with self.assertRaises(OperatorActionUnavailable):
            runtime.operator_action_gateway.approve_voice_message(announcement, 1.0)

    def test_11_no_building_control_provider_recommendations_still_displayed_execution_fails_honestly(self):

        runtime = build_live_runtime(self.building)

        self.assertEqual(runtime.operator_action_gateway.control_capability, PROVIDER_CAPABILITY_NO_PROVIDER)
        self.assertIsNone(runtime.building_control_controller)

        from advisory_system.recommendation_models import AdvisoryReport, BuildingRecommendation

        recommendation = BuildingRecommendation(
            action="Open Door", target_type="door", target_id="does-not-matter", reason="egress",
            confidence=0.8, expected_engineering_benefit="faster egress",
        )
        report = AdvisoryReport(
            scenario_id="s", simulation_time=1.0, civilian_announcements=(),
            firefighter_intelligence=None, building_recommendations=(recommendation,),
            commander_dashboard=None, recommendation_history=(),
        )

        # ingest is a safe no-op with no controller -- nothing is submitted.
        submitted = runtime.operator_action_gateway.ingest_control_recommendations(report)
        self.assertEqual(submitted, ())

        with self.assertRaises(OperatorActionUnavailable):
            runtime.operator_action_gateway.approve_control_request("does-not-matter")


class OperatorDecisionFailureModeTests(unittest.TestCase):

    def setUp(self):

        self.building = make_demo_building()
        self.runtime = build_offline_demo_runtime(
            self.building,
            frame_sources=make_offline_frame_sources(self.building),
            human_detector=make_offline_human_detector(),
            identity_resolver=make_offline_identity_resolver(self.building),
        )
        self.runtime.start()

    def tearDown(self):

        self.runtime.stop()

    def test_12_operator_rejects_never_executes(self):

        from advisory_system.recommendation_models import CivilianAnnouncement

        announcement = CivilianAnnouncement(
            zone_id="zone-lobby", zone_name="Lobby", announcement="Evacuate now", reason="fire",
            confidence=0.9, predicted_rset_improvement_seconds=None,
        )

        self.runtime.operator_action_gateway.reject_voice_message(announcement, 1.0)

        self.assertEqual(
            self.runtime.operator_action_gateway.voice_recommendation_status(announcement), VOICE_STATUS_REJECTED,
        )
        self.assertEqual(len(self.runtime.voice_evacuation_controller.broadcast_log.all_instructions()), 0)

    def test_13_duplicate_operator_approval_never_double_executes(self):

        from advisory_system.recommendation_models import CivilianAnnouncement

        announcement = CivilianAnnouncement(
            zone_id="zone-lobby", zone_name="Lobby", announcement="Evacuate now", reason="fire",
            confidence=0.9, predicted_rset_improvement_seconds=None,
        )

        first = self.runtime.operator_action_gateway.approve_voice_message(announcement, 1.0)
        second = self.runtime.operator_action_gateway.approve_voice_message(announcement, 2.0)

        self.assertEqual(first, second)
        self.assertEqual(len(self.runtime.voice_evacuation_controller.broadcast_log.all_instructions()), 1)


if __name__ == "__main__":
    unittest.main()
