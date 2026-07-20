import shutil
import sys
import tempfile
import unittest

from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

import ai_training as at
import ai_registry as reg
import ai_features as af

from building_state.models import BuildingState

from live_system.event_bus import EventBus, EventType
from live_system.live_ai_gateway import (
    AISystemStatus,
    LiveAIPredictionSnapshot,
    RegistryLiveAIInferenceGateway,
    ThrottledLiveAIInferenceGateway,
)
from live_system.orchestrator import LiveOrchestrator
from live_system.state_manager import StateManager


# =====================================================
# Live AI Inference Runtime Integration milestone -- Phase 14 tests.
# One real, shared, moderate-scale campaign + trained models
# (setUpModule -- same convention tests.test_ai_registry already
# establishes) rather than the full 5000-scenario production run,
# which lives only in scripts/train_live_compatible_models.py.
# =====================================================


_MODULE_STATE = {}

CAMPAIGN_COUNT = 150
CAMPAIGN_SEED = 909


def setUpModule():

    tmp_dir = tempfile.mkdtemp(prefix="live_ai_runtime_test_")
    building = reg.make_training_building()
    definition = reg.make_training_definition()

    campaign_dir, _summary = reg.generate_training_campaign(
        tmp_dir, building, definition, count=CAMPAIGN_COUNT, master_seed=CAMPAIGN_SEED,
    )

    legacy_dataset = at.load_campaign_dataset(campaign_dir)
    live_dataset = reg.build_live_compatible_dataset(legacy_dataset, building)

    evac_result = reg.train_evacuation_time_model(live_dataset, training_seed=1, dataset_identifier="rt-test")
    bottleneck_result = reg.train_bottleneck_occurrence_model(live_dataset, training_seed=1, dataset_identifier="rt-test")

    _MODULE_STATE["tmp_dir"] = tmp_dir
    _MODULE_STATE["building"] = building
    _MODULE_STATE["evac_result"] = evac_result
    _MODULE_STATE["bottleneck_result"] = bottleneck_result


def tearDownModule():

    shutil.rmtree(_MODULE_STATE.get("tmp_dir", ""), ignore_errors=True)


def _make_registry(*, with_bottleneck=True, with_evacuation_time=True):

    registry = reg.ModelRegistry()

    if with_bottleneck:
        registry.register_model(_MODULE_STATE["bottleneck_result"].model, _MODULE_STATE["bottleneck_result"].metadata)

    if with_evacuation_time:
        registry.register_model(_MODULE_STATE["evac_result"].model, _MODULE_STATE["evac_result"].metadata)

    return registry


def _make_gateway(*, with_bottleneck=True, with_evacuation_time=True, include_evacuation_time=True):

    registry = _make_registry(with_bottleneck=with_bottleneck, with_evacuation_time=with_evacuation_time)
    service = reg.LiveAIInferenceService(registry)

    return RegistryLiveAIInferenceGateway(service, include_evacuation_time=include_evacuation_time)


def _real_building_state(total_occupants=6, ignition_zone_id="zone-lobby"):

    return af.build_building_state_at_alarm_activation(
        _MODULE_STATE["building"], total_occupants=total_occupants, ignition_zone_id=ignition_zone_id,
    )


def _canonical_feature_schema():

    # Splits CANONICAL_LIVE_SCHEMA's own fields into numeric/categorical
    # exactly the way a model genuinely trained on canonical rows would
    # (ai_training.preprocessing.FeatureSchema.infer()'s own int/float
    # -is-numeric, everything else (bool/str) -is-categorical rule) --
    # a stub whose schema doesn't match this would fail the real
    # dtype-consistency check in ai_features.compatibility for reasons
    # unrelated to whatever failure mode a given test wants to exercise.

    from ai_training.preprocessing import FeatureSchema

    numeric = tuple(f.name for f in af.CANONICAL_LIVE_SCHEMA if f.dtype in ("int", "float"))
    categorical = tuple(f.name for f in af.CANONICAL_LIVE_SCHEMA if f.dtype not in ("int", "float"))

    return FeatureSchema(numeric_columns=numeric, categorical_columns=categorical)


# =====================================================
# 1/2 -- LiveOrchestrator works without AI; the gateway is optional.
# =====================================================


class NoAIGatewayTests(unittest.TestCase):

    def test_run_cycle_without_a_live_ai_gateway_leaves_ai_prediction_none(self):

        orchestrator = LiveOrchestrator()
        orchestrator.start()

        snapshot = orchestrator.run_cycle(0.0)

        self.assertIsNone(snapshot.ai_prediction_snapshot)
        self.assertIsNone(orchestrator.latest_ai_prediction)

    def test_existing_building_state_only_cycle_still_works_with_no_ai_gateway(self):

        gateway = None
        orchestrator = LiveOrchestrator(live_ai_gateway=gateway)
        orchestrator.start()

        snapshot = orchestrator.run_cycle(0.0)

        self.assertIsNone(snapshot.building_state)
        self.assertIsNone(snapshot.ai_prediction_snapshot)


# =====================================================
# 3/4 -- BuildingState reaches AI inference; bottleneck model runs.
# =====================================================


class _StubBuildingStateGateway:

    def __init__(self, state):
        self._state = state

    def collect(self, time):
        return self._state


class BottleneckLiveInferenceTests(unittest.TestCase):

    def test_building_state_reaches_the_ai_gateway(self):

        received = []

        class _CapturingGateway:
            def predict(self, state, time):
                received.append(state)
                return LiveAIPredictionSnapshot(
                    timestamp=time, building_state_timestamp=state.timestamp if state else None,
                    feature_schema_version=None, system_status=AISystemStatus.UNAVAILABLE,
                )

        state = _real_building_state()
        orchestrator = LiveOrchestrator(
            building_state_gateway=_StubBuildingStateGateway(state), live_ai_gateway=_CapturingGateway(),
        )
        orchestrator.start()
        orchestrator.run_cycle(5.0)

        self.assertEqual(len(received), 1)
        self.assertIs(received[0], state)

    def test_bottleneck_production_candidate_model_runs_end_to_end(self):

        state = _real_building_state()
        gateway = _make_gateway()

        orchestrator = LiveOrchestrator(
            building_state_gateway=_StubBuildingStateGateway(state), live_ai_gateway=gateway,
        )
        orchestrator.start()
        snapshot = orchestrator.run_cycle(0.0)

        prediction = snapshot.ai_prediction_snapshot
        self.assertIsNotNone(prediction)
        self.assertIn(prediction.system_status, (AISystemStatus.AVAILABLE, AISystemStatus.PARTIAL))
        self.assertIsNotNone(prediction.bottleneck)
        self.assertGreaterEqual(prediction.bottleneck.probability, 0.0)
        self.assertLessEqual(prediction.bottleneck.probability, 1.0)
        self.assertEqual(
            prediction.bottleneck.model_id, _MODULE_STATE["bottleneck_result"].metadata.model_id,
        )


# =====================================================
# 5/6 -- Probability vs confidence; experimental labeling.
# =====================================================


class ProbabilityAndExperimentalLabelingTests(unittest.TestCase):

    def test_bottleneck_probability_field_is_not_named_or_treated_as_confidence(self):

        import dataclasses

        from ai_registry.inference_service import BottleneckOccurrencePrediction

        pred_fields = {f.name for f in dataclasses.fields(BottleneckOccurrencePrediction)}
        self.assertIn("probability", pred_fields)
        self.assertNotIn("confidence", pred_fields)

    def test_evacuation_time_field_is_named_experimental_not_predicted_rset(self):

        import dataclasses

        snapshot_fields = {f.name for f in dataclasses.fields(LiveAIPredictionSnapshot)}

        self.assertIn("evacuation_time_experimental", snapshot_fields)
        self.assertNotIn("predicted_rset", snapshot_fields)
        self.assertNotIn("rset", snapshot_fields)

    def test_experimental_evacuation_prediction_is_present_when_included(self):

        state = _real_building_state()
        gateway = _make_gateway()

        snapshot = gateway.predict(state, 0.0)

        self.assertIsNotNone(snapshot.evacuation_time_experimental)
        self.assertEqual(
            snapshot.evacuation_time_experimental.model_id, _MODULE_STATE["evac_result"].metadata.model_id,
        )


# =====================================================
# 7/8 -- Experimental evacuation prediction cannot enter Decision
# Policy / Advisory System (structural, mechanical guard).
# =====================================================


class ExperimentalModelDoesNotReachPolicyOrAdvisoryTests(unittest.TestCase):

    def test_decision_policy_never_imports_live_ai_gateway_or_prediction_snapshot(self):

        import pathlib
        import re

        package_dir = pathlib.Path(__file__).resolve().parent.parent / "decision_policy"
        forbidden = r"^\s*(from|import)\s+(live_system\.live_ai_gateway|ai_registry|ai_features)\b"

        for path in package_dir.glob("*.py"):

            self.assertIsNone(
                re.search(forbidden, path.read_text(), re.MULTILINE),
                f"decision_policy/{path.name} imports the live AI inference boundary -- "
                f"AI must not influence Decision Policy in this milestone.",
            )

    def test_advisory_system_never_imports_live_ai_gateway_or_prediction_snapshot(self):

        import pathlib
        import re

        package_dir = pathlib.Path(__file__).resolve().parent.parent / "advisory_system"
        forbidden = r"^\s*(from|import)\s+(live_system\.live_ai_gateway|ai_registry|ai_features)\b"

        for path in package_dir.glob("*.py"):

            self.assertIsNone(
                re.search(forbidden, path.read_text(), re.MULTILINE),
                f"advisory_system/{path.name} imports the live AI inference boundary -- "
                f"AI must not influence Advisory System in this milestone.",
            )


# =====================================================
# 9/10/11/12/13 -- Safe failure modes.
# =====================================================


class SafeFailureModeTests(unittest.TestCase):

    def test_missing_model_fails_safely_as_unavailable(self):

        state = _real_building_state()
        gateway = _make_gateway(with_bottleneck=False, with_evacuation_time=False)

        snapshot = gateway.predict(state, 0.0)

        self.assertEqual(snapshot.system_status, AISystemStatus.UNAVAILABLE)
        self.assertIsNone(snapshot.bottleneck)
        self.assertTrue(snapshot.errors)

    def test_schema_incompatible_registered_model_fails_safely_as_incompatible(self):

        from ai_training.preprocessing import FeatureSchema

        class _StubIncompatibleModel:
            feature_schema = FeatureSchema(numeric_columns=("not_a_canonical_column",), categorical_columns=())
            label_encoder = None
            def predict_proba(self, X_rows):
                raise AssertionError("must never be called -- registry should reject before inference")

        bad_metadata = reg.ModelMetadata(
            model_id="bad-schema-1", model_type="bottleneck_occurrence", model_version="v1",
            training_timestamp="2026-01-01T00:00:00", training_dataset_identifier="test", training_seed=1,
            feature_schema_version=af.SCHEMA_VERSION, ordered_feature_names=af.CANONICAL_LIVE_FEATURE_NAMES,
            prediction_target="bottleneck_occurrence", model_deployability=reg.Deployability.LIVE_COMPATIBLE,
        )

        registry = reg.ModelRegistry()
        registry.register_model(_StubIncompatibleModel(), bad_metadata)
        service = reg.LiveAIInferenceService(registry)
        gateway = RegistryLiveAIInferenceGateway(service)

        snapshot = gateway.predict(_real_building_state(), 0.0)

        self.assertEqual(snapshot.system_status, AISystemStatus.INCOMPATIBLE)
        self.assertIsNone(snapshot.bottleneck)

    def test_missing_building_state_fields_do_not_crash_inference(self):

        gateway = _make_gateway()

        # An empty BuildingState -- every canonical field None/False --
        # must still produce a real (if less certain) prediction, never
        # a crash and never a fabricated zero-filled value being hidden.
        snapshot = gateway.predict(BuildingState(), 0.0)

        self.assertIn(snapshot.system_status, (AISystemStatus.AVAILABLE, AISystemStatus.PARTIAL))
        self.assertIsNotNone(snapshot.bottleneck)

    def test_corrupted_model_artifact_fails_safely_as_error(self):

        class _CorruptedModel:
            feature_schema = _canonical_feature_schema()
            label_encoder = None
            def predict_proba(self, X_rows):
                raise RuntimeError("simulated corrupted model artifact")

        metadata = reg.ModelMetadata(
            model_id="corrupted-1", model_type="bottleneck_occurrence", model_version="v1",
            training_timestamp="2026-01-01T00:00:00", training_dataset_identifier="test", training_seed=1,
            feature_schema_version=af.SCHEMA_VERSION, ordered_feature_names=af.CANONICAL_LIVE_FEATURE_NAMES,
            prediction_target="bottleneck_occurrence", model_deployability=reg.Deployability.LIVE_COMPATIBLE,
        )

        registry = reg.ModelRegistry()
        registry.register_model(_CorruptedModel(), metadata)
        service = reg.LiveAIInferenceService(registry)
        gateway = RegistryLiveAIInferenceGateway(service)

        snapshot = gateway.predict(_real_building_state(), 0.0)  # must not raise

        self.assertEqual(snapshot.system_status, AISystemStatus.ERROR)
        self.assertIsNone(snapshot.bottleneck)
        self.assertTrue(snapshot.errors)

    def test_inference_exception_does_not_stop_the_live_cycle(self):

        class _CorruptedModel:
            feature_schema = _canonical_feature_schema()
            label_encoder = None
            def predict_proba(self, X_rows):
                raise RuntimeError("boom")

        metadata = reg.ModelMetadata(
            model_id="corrupted-2", model_type="bottleneck_occurrence", model_version="v1",
            training_timestamp="2026-01-01T00:00:00", training_dataset_identifier="test", training_seed=1,
            feature_schema_version=af.SCHEMA_VERSION, ordered_feature_names=af.CANONICAL_LIVE_FEATURE_NAMES,
            prediction_target="bottleneck_occurrence", model_deployability=reg.Deployability.LIVE_COMPATIBLE,
        )
        registry = reg.ModelRegistry()
        registry.register_model(_CorruptedModel(), metadata)
        service = reg.LiveAIInferenceService(registry)
        gateway = RegistryLiveAIInferenceGateway(service)

        state = _real_building_state()
        orchestrator = LiveOrchestrator(
            building_state_gateway=_StubBuildingStateGateway(state), live_ai_gateway=gateway,
        )
        orchestrator.start()

        snapshot = orchestrator.run_cycle(0.0)  # must not raise

        self.assertEqual(snapshot.ai_prediction_snapshot.system_status, AISystemStatus.ERROR)
        self.assertTrue(orchestrator.is_running)


# =====================================================
# 14 -- Stale prediction protection.
# =====================================================


class StalePredictionProtectionTests(unittest.TestCase):

    def test_failed_inference_on_a_later_cycle_replaces_the_earlier_success_not_hides_behind_it(self):

        state = _real_building_state()
        good_gateway = _make_gateway()

        orchestrator = LiveOrchestrator(
            building_state_gateway=_StubBuildingStateGateway(state), live_ai_gateway=good_gateway,
        )
        orchestrator.start()

        first = orchestrator.run_cycle(0.0)
        self.assertIn(first.ai_prediction_snapshot.system_status, (AISystemStatus.AVAILABLE, AISystemStatus.PARTIAL))

        # Swap in a failing gateway for the next cycle -- simulates the
        # model becoming unavailable mid-run.
        orchestrator.live_ai_gateway = _make_gateway(with_bottleneck=False, with_evacuation_time=False)

        second = orchestrator.run_cycle(1.0)

        self.assertEqual(second.ai_prediction_snapshot.system_status, AISystemStatus.UNAVAILABLE)
        self.assertEqual(second.ai_prediction_snapshot.timestamp, 1.0)
        self.assertNotEqual(second.ai_prediction_snapshot, first.ai_prediction_snapshot)


# =====================================================
# 15/16 -- StateManager storage and cycle correspondence.
# =====================================================


class StateManagerAIStorageTests(unittest.TestCase):

    def test_state_manager_stores_the_latest_ai_prediction_snapshot(self):

        manager = StateManager()
        snapshot = LiveAIPredictionSnapshot(
            timestamp=3.0, building_state_timestamp=3.0, feature_schema_version="1.0",
            system_status=AISystemStatus.AVAILABLE,
        )

        result = manager.update_ai_prediction(snapshot, time=3.0)

        self.assertIs(manager.latest_ai_prediction(), snapshot)
        self.assertIs(result.ai_prediction_snapshot, snapshot)
        self.assertEqual(result.component_timestamps["ai_prediction_snapshot"], 3.0)

    def test_ai_prediction_snapshot_references_the_correct_building_state_timestamp(self):

        state = _real_building_state()
        gateway = _make_gateway()

        snapshot = gateway.predict(state, 100.0)

        self.assertEqual(snapshot.timestamp, 100.0)
        self.assertEqual(snapshot.building_state_timestamp, state.timestamp)


# =====================================================
# 17/18 -- Replay-origin and Simulation-origin BuildingState both work.
# =====================================================


class OriginIndependenceTests(unittest.TestCase):

    def test_simulation_origin_building_state_works(self):

        gateway = _make_gateway()
        state = _real_building_state()

        snapshot = gateway.predict(state, 0.0)

        self.assertIsNotNone(snapshot.bottleneck)

    def test_replay_origin_building_state_works(self):

        from camera_manager.manager import CameraManager
        from live_camera_pipeline.replay_frame_source import ReplayFrameSource
        from live_camera_pipeline.detection_provider import LiveCameraPipelineDetectionProvider
        from live_camera_pipeline.identity_resolver import MappingIdentityResolver
        from live_camera_pipeline.pipeline import LiveCameraPipeline
        from tests.live_camera_pipeline_fixtures import MockHumanDetector
        from multi_camera_fusion.engine import MultiCameraFusionEngine
        from building_state.estimator import BuildingStateEstimator
        from hazard.snapshot import HazardSnapshot
        from occupancy.snapshot import OccupancySnapshot

        building = _MODULE_STATE["building"]
        camera_manager = CameraManager()
        camera_manager.discover_cameras(building)

        source = ReplayFrameSource(
            camera_id="cam-lobby", frames=[(0.0, [{"local_track_id": "1"}, {"local_track_id": "2"}])],
        )
        source.start()

        resolver = MappingIdentityResolver({("cam-lobby", "1"): "P1", ("cam-lobby", "2"): "P2"})
        provider = LiveCameraPipelineDetectionProvider()
        pipeline = LiveCameraPipeline(
            frame_sources={"cam-lobby": source}, human_detector=MockHumanDetector(),
            identity_resolver=resolver, detection_provider=provider,
        )
        pipeline.run_cycle(0.0)

        detections = provider.detections_at("cam-lobby", 0.0)
        fusion_result = MultiCameraFusionEngine().fuse(detections, 0.0)

        replay_state = BuildingStateEstimator().estimate(
            0.0, hazard_snapshot=HazardSnapshot(), occupancy_snapshot=OccupancySnapshot(),
            camera_statuses=camera_manager.all_statuses(), fusion_result=fusion_result,
        )

        gateway = _make_gateway()
        snapshot = gateway.predict(replay_state, 0.0)

        self.assertIsNotNone(snapshot.bottleneck)


# =====================================================
# 19/20 -- No ScenarioDefinition/GroundTruth enters AI inference.
# =====================================================


class NoScenarioOrGroundTruthLeakageTests(unittest.TestCase):

    def test_live_ai_gateway_module_never_imports_scenario_or_ground_truth(self):

        import pathlib
        import re

        path = pathlib.Path(__file__).resolve().parent.parent / "live_system" / "live_ai_gateway.py"
        forbidden = r"^\s*(from|import)\s+(scenario_definition|scenario_generator|scenario_runner|ground_truth)\b"

        self.assertIsNone(re.search(forbidden, path.read_text(), re.MULTILINE))

    def test_no_building_control_or_voice_evacuation_import_in_live_ai_gateway(self):

        import pathlib
        import re

        path = pathlib.Path(__file__).resolve().parent.parent / "live_system" / "live_ai_gateway.py"
        forbidden = r"^\s*(from|import)\s+(building_control\.controller|building_control\.providers|voice_evacuation|speaker_manager)\b"

        self.assertIsNone(re.search(forbidden, path.read_text(), re.MULTILINE))


# =====================================================
# 21 -- AI models are not reloaded every cycle.
# =====================================================


class ModelCachingAcrossCyclesTests(unittest.TestCase):

    def test_the_same_model_object_answers_every_cycle(self):

        state = _real_building_state()
        gateway = _make_gateway()

        orchestrator = LiveOrchestrator(
            building_state_gateway=_StubBuildingStateGateway(state), live_ai_gateway=gateway,
        )
        orchestrator.start()

        first = orchestrator.run_cycle(0.0).ai_prediction_snapshot
        second = orchestrator.run_cycle(1.0).ai_prediction_snapshot

        self.assertEqual(first.bottleneck.model_id, second.bottleneck.model_id)
        self.assertEqual(first.bottleneck.model_version, second.bottleneck.model_version)


# =====================================================
# 24/25/26/27 -- No Decision Policy/Advisory/control/voice changes.
# =====================================================


class NoDownstreamChangesIntroducedTests(unittest.TestCase):

    def test_decision_policy_package_has_no_ai_registry_dependency(self):

        import pathlib
        import re

        package_dir = pathlib.Path(__file__).resolve().parent.parent / "decision_policy"
        forbidden = r"^\s*(from|import)\s+ai_registry\b"

        for path in package_dir.glob("*.py"):
            self.assertIsNone(re.search(forbidden, path.read_text(), re.MULTILINE))

    def test_advisory_system_package_has_no_ai_registry_dependency(self):

        import pathlib
        import re

        package_dir = pathlib.Path(__file__).resolve().parent.parent / "advisory_system"
        forbidden = r"^\s*(from|import)\s+ai_registry\b"

        for path in package_dir.glob("*.py"):
            self.assertIsNone(re.search(forbidden, path.read_text(), re.MULTILINE))

    def test_live_ai_gateway_never_imports_building_control_execution_or_voice_broadcast(self):

        import pathlib
        import re

        path = pathlib.Path(__file__).resolve().parent.parent / "live_system" / "live_ai_gateway.py"
        forbidden = r"^\s*(from|import)\s+(building_control\.controller|building_control\.providers|voice_evacuation)\b"

        self.assertIsNone(re.search(forbidden, path.read_text(), re.MULTILINE))


# =====================================================
# Throttling (Phase 9).
# =====================================================


class ThrottledGatewayTests(unittest.TestCase):

    def test_throttled_gateway_skips_within_the_interval_and_runs_after_it(self):

        inner = _make_gateway()
        throttled = ThrottledLiveAIInferenceGateway(inner, min_interval_seconds=5.0)

        state = _real_building_state()

        first = throttled.predict(state, 0.0)
        self.assertIsNotNone(first)

        skipped = throttled.predict(state, 2.0)
        self.assertIsNone(skipped)

        third = throttled.predict(state, 5.0)
        self.assertIsNotNone(third)

    def test_orchestrator_leaves_the_previous_snapshot_in_place_when_throttled_gateway_skips(self):

        inner = _make_gateway()
        throttled = ThrottledLiveAIInferenceGateway(inner, min_interval_seconds=100.0)

        state = _real_building_state()
        orchestrator = LiveOrchestrator(
            building_state_gateway=_StubBuildingStateGateway(state), live_ai_gateway=throttled,
        )
        orchestrator.start()

        first = orchestrator.run_cycle(0.0)
        second = orchestrator.run_cycle(1.0)

        self.assertIs(first.ai_prediction_snapshot, second.ai_prediction_snapshot)
        self.assertEqual(second.component_timestamps["ai_prediction_snapshot"], 0.0)


# =====================================================
# Phase 10 -- the full, dedicated live runtime end-to-end test:
# ReplayFrameSource -> HumanDetector -> IdentityResolver -> CameraManager
# -> MultiCameraFusionEngine -> BuildingState -> Live AI Inference ->
# LiveAIPredictionSnapshot -> StateManager -> LiveOrchestrator, run for
# multiple real cycles through the orchestrator itself (not just calling
# the gateway directly, unlike the origin-independence tests above).
# =====================================================


class LiveRuntimeEndToEndTest(unittest.TestCase):

    def setUp(self):

        from camera_manager.manager import CameraManager
        from live_camera_pipeline.replay_frame_source import ReplayFrameSource
        from live_camera_pipeline.detection_provider import LiveCameraPipelineDetectionProvider
        from live_camera_pipeline.identity_resolver import MappingIdentityResolver
        from live_camera_pipeline.pipeline import LiveCameraPipeline
        from tests.live_camera_pipeline_fixtures import MockHumanDetector
        from multi_camera_fusion.engine import MultiCameraFusionEngine
        from building_state.estimator import BuildingStateEstimator
        from hazard.snapshot import HazardSnapshot
        from occupancy.snapshot import OccupancySnapshot

        self.building = _MODULE_STATE["building"]

        self.camera_manager = CameraManager()
        self.camera_manager.discover_cameras(self.building)

        self.source = ReplayFrameSource(
            camera_id="cam-lobby", frames=[(0.0, [{"local_track_id": "1"}, {"local_track_id": "2"}])],
        )
        self.source.start()

        self.resolver = MappingIdentityResolver({("cam-lobby", "1"): "P1", ("cam-lobby", "2"): "P2"})
        self.provider = LiveCameraPipelineDetectionProvider()
        self.pipeline = LiveCameraPipeline(
            frame_sources={"cam-lobby": self.source}, human_detector=MockHumanDetector(),
            identity_resolver=self.resolver, detection_provider=self.provider,
        )
        self.fusion_engine = MultiCameraFusionEngine()
        self.estimator = BuildingStateEstimator()
        self.hazard_snapshot = HazardSnapshot()
        self.occupancy_snapshot = OccupancySnapshot()

        class _ReplayBuildingStateGateway:
            def __init__(self, outer):
                self._outer = outer

            def collect(self, time):

                self._outer.pipeline.run_cycle(time)
                detections = self._outer.provider.detections_at("cam-lobby", time)
                fusion_result = self._outer.fusion_engine.fuse(detections, time)

                return self._outer.estimator.estimate(
                    time, hazard_snapshot=self._outer.hazard_snapshot,
                    occupancy_snapshot=self._outer.occupancy_snapshot,
                    camera_statuses=self._outer.camera_manager.all_statuses(), fusion_result=fusion_result,
                )

        self.building_state_gateway = _ReplayBuildingStateGateway(self)
        self.ai_gateway = _make_gateway()

        self.orchestrator = LiveOrchestrator(
            building_state_gateway=self.building_state_gateway, live_ai_gateway=self.ai_gateway,
        )
        self.orchestrator.start()

    # =====================================================

    def test_full_replay_to_ai_prediction_chain_across_multiple_cycles_with_an_induced_failure(self):

        # Cycle 1 -- full chain succeeds.
        first = self.orchestrator.run_cycle(0.0)

        self.assertIsNotNone(first.building_state, "1. BuildingState is produced")
        self.assertEqual(len(first.building_state.occupant_tracks), 2)

        prediction_1 = first.ai_prediction_snapshot
        self.assertIsNotNone(prediction_1, "2. Bottleneck AI inference runs")
        self.assertIn(prediction_1.system_status, (AISystemStatus.AVAILABLE, AISystemStatus.PARTIAL))

        self.assertIs(
            self.orchestrator.state_manager.latest_ai_prediction(), prediction_1, "3. Prediction is stored in StateManager",
        )
        self.assertEqual(
            prediction_1.bottleneck.model_id, _MODULE_STATE["bottleneck_result"].metadata.model_id,
            "4. Prediction references the correct model",
        )
        self.assertEqual(
            prediction_1.feature_schema_version, af.SCHEMA_VERSION, "5. Prediction uses the canonical feature schema",
        )

        # 6/7/8 -- no fabricated/leaked inputs: the row extracted from
        # this cycle's real BuildingState carries none of the forbidden
        # scenario/ground-truth-only keys, and every canonical field is
        # either a real observed value or an honest None.
        row = af.extract_canonical_features(first.building_state)
        for forbidden_key in ("ignition_zone", "total_evacuation_time", "bottleneck_occurrence", "Adult_Count"):
            self.assertNotIn(forbidden_key, row)
        self.assertEqual(row["total_occupant_count"], 2)  # real, not fabricated

        # 9 -- a second cycle updates the prediction again (fresh
        # timestamp, same underlying replay-derived occupant count).
        self.source.reset()
        second = self.orchestrator.run_cycle(1.0)

        self.assertEqual(second.ai_prediction_snapshot.timestamp, 1.0)
        self.assertIsNot(second.ai_prediction_snapshot, prediction_1)

        # 10 -- AI failure on one cycle does not stop the next cycle:
        # swap in a broken gateway for cycle 3, then a working one again
        # for cycle 4.
        self.orchestrator.live_ai_gateway = _make_gateway(with_bottleneck=False, with_evacuation_time=False)
        self.source.reset()
        third = self.orchestrator.run_cycle(2.0)  # must not raise
        self.assertEqual(third.ai_prediction_snapshot.system_status, AISystemStatus.UNAVAILABLE)
        self.assertTrue(self.orchestrator.is_running)

        self.orchestrator.live_ai_gateway = self.ai_gateway
        self.source.reset()
        fourth = self.orchestrator.run_cycle(3.0)
        self.assertIn(fourth.ai_prediction_snapshot.system_status, (AISystemStatus.AVAILABLE, AISystemStatus.PARTIAL))


if __name__ == "__main__":
    unittest.main()
