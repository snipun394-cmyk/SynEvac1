import shutil
import sys
import tempfile
import unittest

from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

import ai_training as at

from ai_features.feature_schema import CANONICAL_LIVE_FEATURE_NAMES, SCHEMA_VERSION

from building_state.models import BuildingState

import ai_registry as reg


# =====================================================
# Live-Compatible AI Model Training & Model Registry milestone -- Phase
# 16 tests. One real, shared, moderate-scale campaign (setUpClass,
# reused read-only across every test method -- same convention
# tests.ai_training_fixtures.RealCampaignTestCase already establishes)
# rather than the full 5000-scenario production run, which lives only
# in scripts/train_live_compatible_models.py's own report.
# =====================================================


_MODULE_STATE = {}

CAMPAIGN_COUNT = 200
CAMPAIGN_SEED = 4242


def setUpModule():

    # Generated ONCE for the entire test module (unittest's own
    # module-level fixture hook) and shared read-only across every
    # TestCase below -- each of those subclasses' own setUpClass()
    # otherwise reruns independently per class, which would regenerate
    # this same 200-scenario campaign (and retrain both models) a dozen
    # times over for no benefit, since nothing here is ever mutated.

    tmp_dir = tempfile.mkdtemp(prefix="ai_registry_test_")
    building = reg.make_training_building()
    definition = reg.make_training_definition()

    campaign_dir, campaign_summary = reg.generate_training_campaign(
        tmp_dir, building, definition, count=CAMPAIGN_COUNT, master_seed=CAMPAIGN_SEED,
    )

    legacy_dataset = at.load_campaign_dataset(campaign_dir)
    live_dataset = reg.build_live_compatible_dataset(legacy_dataset, building)

    _MODULE_STATE["tmp_dir"] = tmp_dir
    _MODULE_STATE["building"] = building
    _MODULE_STATE["campaign_dir"] = campaign_dir
    _MODULE_STATE["campaign_summary"] = campaign_summary
    _MODULE_STATE["legacy_dataset"] = legacy_dataset
    _MODULE_STATE["live_dataset"] = live_dataset
    _MODULE_STATE["evac_result"] = reg.train_evacuation_time_model(
        live_dataset, training_seed=1, dataset_identifier="test-campaign",
    )
    _MODULE_STATE["bottleneck_result"] = reg.train_bottleneck_occurrence_model(
        live_dataset, training_seed=1, dataset_identifier="test-campaign",
    )


def tearDownModule():

    shutil.rmtree(_MODULE_STATE.get("tmp_dir", ""), ignore_errors=True)


class _SharedCampaignTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        super().setUpClass()

        cls.building = _MODULE_STATE["building"]
        cls.campaign_dir = _MODULE_STATE["campaign_dir"]
        cls.campaign_summary = _MODULE_STATE["campaign_summary"]
        cls.legacy_dataset = _MODULE_STATE["legacy_dataset"]
        cls.live_dataset = _MODULE_STATE["live_dataset"]
        cls.evac_result = _MODULE_STATE["evac_result"]
        cls.bottleneck_result = _MODULE_STATE["bottleneck_result"]


# =====================================================
# 1 -- Live-compatible models contain metadata.
# =====================================================


class ModelMetadataTests(_SharedCampaignTestCase):

    def test_evacuation_model_metadata_is_fully_populated(self):

        metadata = self.evac_result.metadata

        self.assertTrue(metadata.model_id)
        self.assertEqual(metadata.model_type, "evacuation_time")
        self.assertTrue(metadata.model_version)
        self.assertTrue(metadata.training_timestamp)
        self.assertEqual(metadata.training_dataset_identifier, "test-campaign")
        self.assertEqual(metadata.training_seed, 1)
        self.assertEqual(metadata.feature_schema_version, SCHEMA_VERSION)
        self.assertEqual(metadata.ordered_feature_names, CANONICAL_LIVE_FEATURE_NAMES)
        self.assertIn("total_evacuation_time", metadata.prediction_target)
        self.assertEqual(metadata.model_deployability, reg.Deployability.LIVE_COMPATIBLE)
        self.assertTrue(metadata.training_metrics)
        self.assertTrue(metadata.missing_data_policy)

    def test_metadata_round_trips_through_disk(self):

        directory = tempfile.mkdtemp(prefix="ai_registry_metadata_")

        try:

            reg.save_live_model(self.evac_result.model, self.evac_result.metadata, directory)
            loaded = reg.load_live_metadata(directory)

            self.assertEqual(loaded, self.evac_result.metadata)

        finally:

            shutil.rmtree(directory, ignore_errors=True)


# =====================================================
# 2/14 -- Research-only models cannot be loaded through the live
# registry / registry refuses incompatible models.
# =====================================================


class ResearchOnlyRefusalTests(_SharedCampaignTestCase):

    def _research_only_metadata(self):

        return reg.ModelMetadata(
            model_id="fake-research-only-1",
            model_type="evacuation_time",
            model_version="v1",
            training_timestamp="2026-01-01T00:00:00",
            training_dataset_identifier="test",
            training_seed=1,
            feature_schema_version=SCHEMA_VERSION,
            ordered_feature_names=CANONICAL_LIVE_FEATURE_NAMES,
            prediction_target="total_evacuation_time",
            model_deployability=reg.Deployability.RESEARCH_ONLY,
        )

    def test_get_model_refuses_research_only_by_default(self):

        registry = reg.ModelRegistry()
        registry.register_model(self.evac_result.model, self._research_only_metadata())

        with self.assertRaises(reg.ResearchOnlyModelError):
            registry.get_model("fake-research-only-1")

    def test_get_model_allows_research_only_with_explicit_override(self):

        registry = reg.ModelRegistry()
        registry.register_model(self.evac_result.model, self._research_only_metadata())

        model, metadata = registry.get_model("fake-research-only-1", allow_research_only=True)
        self.assertEqual(metadata.model_deployability, reg.Deployability.RESEARCH_ONLY)

    def test_get_latest_compatible_model_never_returns_research_only(self):

        registry = reg.ModelRegistry()
        registry.register_model(self.evac_result.model, self._research_only_metadata())

        self.assertIsNone(registry.get_latest_compatible_model("evacuation_time"))

    def test_validate_model_compatibility_marks_research_only_incompatible(self):

        registry = reg.ModelRegistry()
        registry.register_model(self.evac_result.model, self._research_only_metadata())

        report = registry.validate_model_compatibility("fake-research-only-1")

        self.assertFalse(report.compatible)
        self.assertTrue(any(issue.kind == "research_only_deployability" for issue in report.issues))


# =====================================================
# 3/4 -- Feature schema / ordering mismatch rejected.
# =====================================================


class SchemaCompatibilityTests(_SharedCampaignTestCase):

    def test_wrong_feature_schema_version_finds_no_compatible_model(self):

        registry = reg.ModelRegistry()
        registry.register_model(self.evac_result.model, self.evac_result.metadata)

        self.assertIsNone(
            registry.get_latest_compatible_model("evacuation_time", feature_schema_version="99.9-does-not-exist"),
        )

    def test_reordered_feature_names_is_rejected(self):

        import dataclasses

        reordered = tuple(reversed(CANONICAL_LIVE_FEATURE_NAMES))
        bad_metadata = dataclasses.replace(self.evac_result.metadata, ordered_feature_names=reordered, model_id="reordered-1")

        registry = reg.ModelRegistry()
        registry.register_model(self.evac_result.model, bad_metadata)

        report = registry.validate_model_compatibility("reordered-1")

        self.assertFalse(report.compatible)
        self.assertTrue(any(issue.kind == "feature_ordering_mismatch" for issue in report.issues))


# =====================================================
# 5/6 -- Missing required features fail safely; no zero fabrication.
# =====================================================


class SafeInferenceFailureTests(_SharedCampaignTestCase):

    def setUp(self):

        self.registry = reg.ModelRegistry()
        self.registry.register_model(self.evac_result.model, self.evac_result.metadata)
        self.registry.register_model(self.bottleneck_result.model, self.bottleneck_result.metadata)
        self.service = reg.LiveAIInferenceService(self.registry)

    def test_no_compatible_model_raises_inference_unavailable(self):

        empty_registry = reg.ModelRegistry()
        service = reg.LiveAIInferenceService(empty_registry)

        with self.assertRaises(reg.InferenceUnavailableError):
            service.predict_evacuation_time(BuildingState(), timestamp=0.0)

    def test_empty_building_state_never_fabricates_a_prediction_from_zero_filled_features(self):

        # Every field of an empty BuildingState's canonical row is
        # None/False -- the model still predicts (that is a legitimate,
        # imputed prediction, exactly like every existing model already
        # tolerates missing scenario columns), but the SERVICE itself
        # must never silently substitute a 0 for a missing REQUIRED
        # column -- proven by the row always containing every required
        # column (as None, not 0) and validate_feature_row() accepting
        # exactly that shape.
        from ai_features.building_state_extractor import extract_canonical_features

        row = extract_canonical_features(BuildingState())

        self.assertIsNone(row["total_occupant_count"])
        self.assertNotEqual(row["total_occupant_count"], 0)

        prediction = self.service.predict_evacuation_time(BuildingState(), timestamp=0.0)
        self.assertIsInstance(prediction.predicted_seconds, float)


# =====================================================
# 7/8 -- Deterministic, leak-free scenario-level splitting.
# =====================================================


class SplitDeterminismAndLeakageTests(_SharedCampaignTestCase):

    def test_training_split_is_deterministic_for_a_fixed_seed(self):

        first = reg.train_evacuation_time_model(self.live_dataset, training_seed=99, dataset_identifier="d")
        second = reg.train_evacuation_time_model(self.live_dataset, training_seed=99, dataset_identifier="d")

        self.assertEqual(first.scenario_split_counts, second.scenario_split_counts)
        self.assertEqual(first.extra_evaluation["test_metrics"], second.extra_evaluation["test_metrics"])

    def test_scenario_level_leakage_between_train_and_test_is_impossible(self):

        from ai_training.models.evacuation_time_model import EvacuationTimeModel
        from ai_training.split import make_split

        X_rows, y, _extra = EvacuationTimeModel.build_table(self.live_dataset)
        groups = list(self.live_dataset.scenario_ids)

        split = make_split(len(X_rows), groups=groups, test_size=0.15, val_size=0.15, random_state=1)

        train_scenarios = {groups[i] for i in split.train_indices}
        val_scenarios = {groups[i] for i in split.val_indices}
        test_scenarios = {groups[i] for i in split.test_indices}

        self.assertEqual(train_scenarios & test_scenarios, set())
        self.assertEqual(train_scenarios & val_scenarios, set())
        self.assertEqual(val_scenarios & test_scenarios, set())

        total = self.evac_result.scenario_split_counts
        self.assertEqual(total.train + total.validation + total.test, len(self.live_dataset))


# =====================================================
# 9/10 -- Baseline comparison and imbalance-aware metrics.
# =====================================================


class BaselineAndImbalanceTests(_SharedCampaignTestCase):

    def test_evacuation_model_reports_a_status_derived_from_baseline_comparison(self):

        status = self.evac_result.extra_evaluation["model_status"]
        self.assertIn(status, (reg.ModelStatus.PRODUCTION_CANDIDATE, reg.ModelStatus.EXPERIMENTAL))

        self.assertIn("mean", self.evac_result.baseline_metrics)
        self.assertIn("median", self.evac_result.baseline_metrics)

    def test_bottleneck_evaluation_includes_imbalance_aware_metrics_and_class_balance(self):

        test_metrics = self.bottleneck_result.extra_evaluation["test_metrics"]

        self.assertIn("pr_auc", test_metrics)
        self.assertIn("roc_auc", test_metrics)
        self.assertIn("train_class_balance", self.bottleneck_result.extra_evaluation)
        self.assertIn("confusion_matrix", self.bottleneck_result.extra_evaluation)

        # Accuracy alone must never be the only thing reported.
        self.assertGreater(len(test_metrics), 1)


# =====================================================
# 11 -- Model probability is not mislabeled as recommendation confidence.
# =====================================================


class ProbabilityNamingTests(unittest.TestCase):

    def test_bottleneck_prediction_field_is_named_probability_not_confidence(self):

        import dataclasses

        field_names = {f.name for f in dataclasses.fields(reg.BottleneckOccurrencePrediction)}

        self.assertIn("probability", field_names)
        self.assertNotIn("confidence", field_names)


# =====================================================
# 12 -- Evacuation uncertainty is unavailable unless genuinely computed.
# =====================================================


class UncertaintyHonestyTests(_SharedCampaignTestCase):

    def test_ensemble_uncertainty_is_available_for_random_forest(self):

        self.assertTrue(self.evac_result.extra_evaluation["uncertainty_available"])

    def test_uncertainty_is_none_for_a_non_ensemble_estimator(self):

        from ai_registry.uncertainty import regression_ensemble_uncertainty

        class _NotAnEnsemble:
            def predict(self, X):
                return [0.0] * len(X)

        class _FakeModel:
            estimator = _NotAnEnsemble()
            preprocessor = self.evac_result.model.preprocessor

        result = regression_ensemble_uncertainty(_FakeModel(), [{"total_occupant_count": 1}])
        self.assertIsNone(result)


# =====================================================
# 13/15 -- Registry finds compatible models; models are cached.
# =====================================================


class RegistryCachingTests(_SharedCampaignTestCase):

    def test_registry_finds_a_compatible_model(self):

        registry = reg.ModelRegistry()
        registry.register_model(self.evac_result.model, self.evac_result.metadata)

        found = registry.get_latest_compatible_model("evacuation_time")

        self.assertIsNotNone(found)
        model, metadata = found
        self.assertEqual(metadata.model_id, self.evac_result.metadata.model_id)

    def test_repeated_lookups_return_the_identical_cached_model_object(self):

        registry = reg.ModelRegistry()
        registry.register_model(self.evac_result.model, self.evac_result.metadata)

        first_model, _ = registry.get_latest_compatible_model("evacuation_time")
        second_model, _ = registry.get_latest_compatible_model("evacuation_time")

        self.assertIs(first_model, second_model)
        self.assertIs(first_model, self.evac_result.model)


# =====================================================
# 16/17/18 -- BuildingState-only inference; no GroundTruth/
# ScenarioDefinition can enter inference features.
# =====================================================


class InferenceInputBoundaryTests(_SharedCampaignTestCase):

    def test_inference_service_never_imports_ground_truth_or_scenario_definition(self):

        import pathlib
        import re

        path = pathlib.Path(__file__).resolve().parent.parent / "ai_registry" / "inference_service.py"
        text = path.read_text()

        forbidden = r"^\s*(from|import)\s+(ground_truth|scenario_definition|scenario)\b"

        self.assertIsNone(re.search(forbidden, text, re.MULTILINE))

    def test_building_state_extractor_never_imports_ground_truth_or_scenario_definition(self):

        import pathlib
        import re

        path = pathlib.Path(__file__).resolve().parent.parent / "ai_features" / "building_state_extractor.py"
        text = path.read_text()

        forbidden = r"^\s*(from|import)\s+(ground_truth|scenario_definition|scenario)\b"

        self.assertIsNone(re.search(forbidden, text, re.MULTILINE))


# =====================================================
# 19/20 -- Offline end-to-end inference test (Phase 13): simulation
# path and replay path both reach the same inference service.
# =====================================================


class OfflineEndToEndInferenceTests(_SharedCampaignTestCase):

    def setUp(self):

        self.registry = reg.ModelRegistry()
        self.registry.register_model(self.evac_result.model, self.evac_result.metadata)
        self.registry.register_model(self.bottleneck_result.model, self.bottleneck_result.metadata)
        self.service = reg.LiveAIInferenceService(self.registry)

    def test_simulation_pipeline_inference_works(self):

        import ai_features as af

        state = af.build_building_state_at_alarm_activation(
            self.building, total_occupants=6, ignition_zone_id="zone-lobby",
        )

        prediction = self.service.predict_evacuation_time(state, timestamp=0.0)
        self.assertIsInstance(prediction.predicted_seconds, float)
        self.assertEqual(prediction.feature_schema_version, SCHEMA_VERSION)

        bottleneck_prediction = self.service.predict_bottleneck_occurrence(state, timestamp=0.0)
        self.assertIsInstance(bottleneck_prediction.probability, float)
        self.assertGreaterEqual(bottleneck_prediction.probability, 0.0)
        self.assertLessEqual(bottleneck_prediction.probability, 1.0)

    def test_replay_pipeline_inference_works_through_the_same_service(self):

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

        camera_manager = CameraManager()
        camera_manager.discover_cameras(self.building)

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

        prediction = self.service.predict_evacuation_time(replay_state, timestamp=0.0)
        self.assertIsInstance(prediction.predicted_seconds, float)

    def test_no_scenario_definition_or_ground_truth_data_reaches_the_extracted_row(self):

        import ai_features as af

        state = af.build_building_state_at_alarm_activation(
            self.building, total_occupants=6, ignition_zone_id="zone-lobby",
        )
        row = af.extract_canonical_features(state)

        for forbidden_key in (
            "ignition_zone", "fire_profile", "growth_time", "total_evacuation_time",
            "bottleneck_occurrence", "Adult_Count", "Mean_Walking_Speed_Multiplier",
        ):
            self.assertNotIn(forbidden_key, row)


# =====================================================
# 21/22 -- Existing research workflows and ai_training stay unaffected.
# =====================================================


class ExistingResearchWorkflowTests(_SharedCampaignTestCase):

    def test_existing_evacuation_time_model_still_trains_via_experiment_runner(self):

        from ai_training.experiment import ExperimentConfig, ExperimentRunner

        result = ExperimentRunner().run(
            self.legacy_dataset, ExperimentConfig(name="research-exp", model_name="evacuation_time"),
        )

        self.assertIn("r2", result.metrics)

    def test_existing_scenario_feature_columns_are_still_the_full_legacy_set(self):

        row = self.legacy_dataset.scenario_feature_rows()[0]

        self.assertIn("ignition_zone", row)
        self.assertIn("Adult_Count", row)
        self.assertIn("Mean_Walking_Speed_Multiplier", row)


# =====================================================
# 23/24 -- LiveOrchestrator/Advisory System remain unwired and unchanged.
# =====================================================


class NoLiveWiringGuardTests(unittest.TestCase):

    def test_live_system_never_imports_ai_registry(self):

        import pathlib
        import re

        package_dir = pathlib.Path(__file__).resolve().parent.parent / "live_system"
        forbidden = r"^\s*(from|import)\s+ai_registry\b"

        for path in package_dir.glob("*.py"):

            self.assertIsNone(
                re.search(forbidden, path.read_text(), re.MULTILINE),
                f"live_system/{path.name} imports ai_registry -- AI must not be wired into "
                f"LiveOrchestrator by this milestone.",
            )

    def test_advisory_system_never_imports_ai_registry(self):

        import pathlib
        import re

        package_dir = pathlib.Path(__file__).resolve().parent.parent / "advisory_system"
        forbidden = r"^\s*(from|import)\s+ai_registry\b"

        for path in package_dir.glob("*.py"):

            self.assertIsNone(
                re.search(forbidden, path.read_text(), re.MULTILINE),
                f"advisory_system/{path.name} imports ai_registry -- Advisory System behavior "
                f"must remain unchanged by this milestone.",
            )

    def test_ai_registry_never_imports_live_system_or_advisory_system(self):

        import pathlib
        import re

        package_dir = pathlib.Path(__file__).resolve().parent.parent / "ai_registry"
        forbidden = r"^\s*(from|import)\s+(live_system|advisory_system)\b"

        for path in package_dir.glob("*.py"):

            self.assertIsNone(
                re.search(forbidden, path.read_text(), re.MULTILINE),
                f"ai_registry/{path.name} imports live_system/advisory_system.",
            )


if __name__ == "__main__":
    unittest.main()
