import shutil
import sys
import tempfile
import unittest

from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

import ai_training as at
import ai_registry as reg
import ai_features as af

from ai_features.feature_schema import CANONICAL_LIVE_FEATURE_NAMES, SCHEMA_VERSION
from ai_registry.metadata import Deployability, ModelMetadata
from ai_registry.registry import ModelRegistry
from ai_registry.inference_service import LiveAIInferenceService
from ai_training.models.bottleneck_model import BottleneckModel
from ai_training.models.evacuation_time_model import EvacuationTimeModel
from ai_training.split import apply_split, make_split

from live_system.live_ai_gateway import RegistryLiveAIInferenceGateway


# =====================================================
# Predictive Model Development & Benchmark Campaign milestone, Phase 9 --
# proves a model trained with a NON-DEFAULT algorithm (this milestone's
# own extension to ai_training.models.base.build_classifier/
# build_regressor) registers and serves through the EXISTING Model
# Registry / Shadow-Mode gateway exactly like the default random_forest
# path tests/test_live_ai_runtime_integration.py already covers -- no
# runtime wiring is touched, only proven. Uses a small (150-scenario)
# campaign, same convention as that existing test file, never the full
# 11,997-scenario benchmark campaign (which lives entirely outside the
# repo -- regenerable via scripts/run_model_benchmark_campaign.py).
# =====================================================


_MODULE_STATE = {}

CAMPAIGN_COUNT = 150
CAMPAIGN_SEED = 20260730


def setUpModule():

    tmp_dir = tempfile.mkdtemp(prefix="model_benchmark_registration_test_")
    building = reg.make_training_building()
    definition = reg.make_training_definition()

    campaign_dir, _summary = reg.generate_training_campaign(
        tmp_dir, building, definition, count=CAMPAIGN_COUNT, master_seed=CAMPAIGN_SEED,
    )

    legacy_dataset = at.load_campaign_dataset(campaign_dir)
    live_dataset = reg.build_live_compatible_dataset(legacy_dataset, building)

    X_bottleneck, y_bottleneck, extra = BottleneckModel.build_table(live_dataset, target="occurrence")
    X_evac, y_evac, _ = EvacuationTimeModel.build_table(live_dataset)

    groups = extra.get("groups") or list(live_dataset.scenario_ids)
    split = make_split(len(X_bottleneck), groups=groups, test_size=0.2, random_state=1)

    Xb_train, _, Xb_test = apply_split(X_bottleneck, split)
    yb_train, _, yb_test = apply_split(list(y_bottleneck), split)
    Xe_train, _, Xe_test = apply_split(X_evac, split)
    ye_train, _, ye_test = apply_split(list(y_evac), split)

    bottleneck_model = BottleneckModel(config={"algorithm": "gradient_boosting"}, target="occurrence")
    bottleneck_model.fit(Xb_train, yb_train)

    evac_model = EvacuationTimeModel(config={"algorithm": "linear_regression"})
    evac_model.fit(Xe_train, ye_train)

    _MODULE_STATE["tmp_dir"] = tmp_dir
    _MODULE_STATE["building"] = building
    _MODULE_STATE["bottleneck_model"] = bottleneck_model
    _MODULE_STATE["evac_model"] = evac_model
    _MODULE_STATE["Xb_test"] = Xb_test
    _MODULE_STATE["yb_test"] = yb_test


def tearDownModule():

    shutil.rmtree(_MODULE_STATE.get("tmp_dir", ""), ignore_errors=True)


def _make_metadata(model_type, prediction_target, algorithm):

    return ModelMetadata(
        model_id=f"{model_type}-benchmark-{algorithm}-test",
        model_type=model_type,
        model_version="test-version",
        training_timestamp="2026-07-30T00:00:00+00:00",
        training_dataset_identifier="model-benchmark-registration-test",
        training_seed=1,
        feature_schema_version=SCHEMA_VERSION,
        ordered_feature_names=CANONICAL_LIVE_FEATURE_NAMES,
        prediction_target=prediction_target,
        model_deployability=Deployability.LIVE_COMPATIBLE,
    )


class NonDefaultAlgorithmRegistrationTests(unittest.TestCase):

    def test_gradient_boosting_bottleneck_model_registers_and_predicts(self):

        registry = ModelRegistry()
        metadata = _make_metadata("bottleneck_occurrence", "bool(GroundTruth.doors_that_became_bottlenecks)", "gradient_boosting")

        registry.register_model(_MODULE_STATE["bottleneck_model"], metadata)

        model, returned_metadata = registry.get_model(metadata.model_id)

        self.assertIs(model, _MODULE_STATE["bottleneck_model"])
        self.assertEqual(returned_metadata.model_id, metadata.model_id)

        predictions = model.predict(_MODULE_STATE["Xb_test"])
        self.assertEqual(len(predictions), len(_MODULE_STATE["yb_test"]))

    def test_get_latest_compatible_model_returns_the_registered_gradient_boosting_model(self):

        registry = ModelRegistry()
        metadata = _make_metadata("bottleneck_occurrence", "bool(GroundTruth.doors_that_became_bottlenecks)", "gradient_boosting")
        registry.register_model(_MODULE_STATE["bottleneck_model"], metadata)

        result = registry.get_latest_compatible_model("bottleneck_occurrence")

        self.assertIsNotNone(result)
        model, returned_metadata = result
        self.assertEqual(returned_metadata.model_id, metadata.model_id)

    def test_linear_regression_evacuation_time_model_registers_and_predicts(self):

        registry = ModelRegistry()
        metadata = _make_metadata("evacuation_time", "total_evacuation_time (seconds)", "linear_regression")

        registry.register_model(_MODULE_STATE["evac_model"], metadata)

        model, _ = registry.get_model(metadata.model_id)
        predictions = model.predict(_MODULE_STATE["Xb_test"][:5])

        self.assertEqual(len(predictions), 5)

    def test_shadow_mode_gateway_serves_both_registered_benchmark_models(self):

        registry = ModelRegistry()
        registry.register_model(
            _MODULE_STATE["bottleneck_model"],
            _make_metadata("bottleneck_occurrence", "bool(GroundTruth.doors_that_became_bottlenecks)", "gradient_boosting"),
        )
        registry.register_model(
            _MODULE_STATE["evac_model"],
            _make_metadata("evacuation_time", "total_evacuation_time (seconds)", "linear_regression"),
        )

        service = LiveAIInferenceService(registry)
        gateway = RegistryLiveAIInferenceGateway(service, include_evacuation_time=True)

        state = af.build_building_state_at_alarm_activation(_MODULE_STATE["building"], total_occupants=8, timestamp=0.0)
        snapshot = gateway.predict(state, 0.0)

        self.assertEqual(snapshot.errors, ())
        self.assertIsNotNone(snapshot.bottleneck)
        self.assertIsNotNone(snapshot.evacuation_time_experimental)
        self.assertGreaterEqual(snapshot.bottleneck.probability, 0.0)
        self.assertLessEqual(snapshot.bottleneck.probability, 1.0)
        self.assertGreater(snapshot.evacuation_time_experimental.predicted_seconds, 0.0)

    def test_incompatible_ordered_feature_names_are_rejected_by_compatibility_check(self):

        # A model claiming the current schema VERSION but a stale/wrong
        # feature ORDER must be rejected -- proves the registration this
        # milestone performs is genuinely schema-checked, not merely
        # "registered because nothing crashed" (ai_registry.registry.
        # validate_model_compatibility's own documented guarantee,
        # reused here rather than re-asserted blindly).

        registry = ModelRegistry()
        metadata = ModelMetadata(
            model_id="bottleneck-bad-order-test", model_type="bottleneck_occurrence",
            model_version="test-version", training_timestamp="2026-07-30T00:00:00+00:00",
            training_dataset_identifier="model-benchmark-registration-test", training_seed=1,
            feature_schema_version=SCHEMA_VERSION,
            ordered_feature_names=tuple(reversed(CANONICAL_LIVE_FEATURE_NAMES)),
            prediction_target="bool(GroundTruth.doors_that_became_bottlenecks)",
            model_deployability=Deployability.LIVE_COMPATIBLE,
        )
        registry.register_model(_MODULE_STATE["bottleneck_model"], metadata)

        report = registry.validate_model_compatibility(metadata.model_id)

        self.assertFalse(report.compatible)


if __name__ == "__main__":
    unittest.main()
