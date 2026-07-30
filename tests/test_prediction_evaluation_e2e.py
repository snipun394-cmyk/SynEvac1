import shutil
import tempfile
import time
import unittest

import ai_features as af
import ai_registry as reg

from prediction_evaluation.registry import PredictionRegistry
from prediction_evaluation.ground_truth import extract_ground_truth_sample
from prediction_evaluation.timeline import GroundTruthTimeline
from prediction_evaluation.models import GroundTruthSample
from prediction_evaluation.evaluator import evaluate
from prediction_evaluation.comparison import compare_models
from prediction_evaluation.timeline import match_predictions

from live_system.live_ai_gateway import RegistryLiveAIInferenceGateway

from crowd_intelligence.models import BuildingCrowdSummary, CrowdIntelligenceSnapshot


# =====================================================
# Prediction vs Reality Evaluation Framework milestone -- end-to-end
# proof against a REAL, small, trained model (mirrors tests/test_
# shadow_mode_prediction.py's own setUpModule convention exactly) --
# every prediction recorded below is a genuine RegistryLiveAIInferenceGateway
# output, not a hand-built stub.
# =====================================================

_MODULE_STATE = {}

CAMPAIGN_COUNT = 30
CAMPAIGN_SEED = 424


def setUpModule():

    tmp_dir = tempfile.mkdtemp(prefix="pred_eval_e2e_")
    building = reg.make_training_building()
    definition = reg.make_training_definition()

    campaign_dir, _summary = reg.generate_training_campaign(
        tmp_dir, building, definition, count=CAMPAIGN_COUNT, master_seed=CAMPAIGN_SEED,
    )

    import ai_training as at
    legacy_dataset = at.load_campaign_dataset(campaign_dir)
    live_dataset = reg.build_live_compatible_dataset(legacy_dataset, building)

    bottleneck_result = reg.train_bottleneck_occurrence_model(live_dataset, training_seed=1, dataset_identifier="pred-eval-test")

    _MODULE_STATE["tmp_dir"] = tmp_dir
    _MODULE_STATE["building"] = building
    _MODULE_STATE["bottleneck_result"] = bottleneck_result


def tearDownModule():

    shutil.rmtree(_MODULE_STATE.get("tmp_dir", ""), ignore_errors=True)


def _make_gateway():

    registry = reg.ModelRegistry()
    registry.register_model(_MODULE_STATE["bottleneck_result"].model, _MODULE_STATE["bottleneck_result"].metadata)
    service = reg.LiveAIInferenceService(registry)

    return RegistryLiveAIInferenceGateway(service, include_evacuation_time=False)


class EndToEndRealModelTests(unittest.TestCase):

    def test_full_pipeline_with_real_predictions_across_all_horizons(self):

        gateway = _make_gateway()
        eval_registry = PredictionRegistry()
        gt_samples = []

        horizons = (5.0, 10.0, 20.0, 30.0, 60.0)

        for i in range(20):

            t = float(i * 15)
            horizon = horizons[i % len(horizons)]

            state = af.build_building_state_at_alarm_activation(
                _MODULE_STATE["building"], total_occupants=2 + (i % 6), timestamp=t,
            )

            snapshot = gateway.predict(state, t)

            eval_registry.record_from_snapshot(
                snapshot, prediction_horizon_seconds=horizon, source="live" if i % 2 == 0 else "simulation",
                scenario_id=f"scenario-{i % 3}", building_id="building-1",
                context_tags={"occupancy_level": "high" if (2 + (i % 6)) >= 5 else "low"},
            )

            gt_samples.append(GroundTruthSample(
                timestamp=t + horizon, source="live" if i % 2 == 0 else "simulation",
                congestion_detected=(i % 3 == 0),
            ))

        timeline = GroundTruthTimeline(gt_samples)

        start = time.perf_counter()
        report = evaluate(eval_registry.all(), timeline, tolerance_seconds=1.0)
        elapsed = time.perf_counter() - start

        print(f"\n[prediction evaluation perf] evaluate() over {len(eval_registry)} predictions took {elapsed * 1000:.2f} ms")

        # Phase 2 -- every prediction found ITS OWN correctly-aligned match.
        self.assertEqual(len(report.matched_evaluations), 20)
        self.assertEqual(report.unmatched_predictions, ())

        for matched in report.matched_evaluations:
            self.assertEqual(matched.target_timestamp, matched.prediction.timestamp + matched.prediction.prediction_horizon_seconds)
            self.assertLessEqual(matched.match_time_delta_seconds, 1.0)

        # Phase 5 -- all 5 configured horizons represented (4 predictions each).
        for horizon in horizons:
            self.assertEqual(report.by_horizon[horizon].evaluation_count, 4)

        # Phase 8 -- per-scenario/per-building statistics non-empty.
        self.assertEqual(len(report.by_scenario), 3)
        self.assertIn("building-1", report.by_building)

        # Phase 6 -- condition breakdown available via the raw matched list.
        from prediction_evaluation.condition_analysis import analyze_by_condition
        by_condition = analyze_by_condition(report.matched_evaluations, "occupancy_level")
        self.assertIn("high", by_condition)
        self.assertIn("low", by_condition)

        # Phase 4 -- real classification metrics were genuinely computed
        # (not honestly-empty, since real probabilities/occurrences and
        # real ground truth congestion values were both supplied).
        self.assertGreater(report.overall.evaluation_count, 0)
        self.assertIsNotNone(report.overall.classification.precision)

        self.assertLess(elapsed, 2.0)

    def test_simulation_and_live_sources_use_identical_extraction_logic(self):

        # Phase 0's own "must support both Simulation and Live Runtime
        # using identical evaluation logic" requirement -- the SAME
        # CrowdIntelligenceSnapshot-shaped input, extracted through the
        # SAME function, differing ONLY in the `source` label a caller
        # supplies.
        snapshot = CrowdIntelligenceSnapshot(
            timestamp=10.0,
            building_summary=BuildingCrowdSummary(total_observed_occupants=7, congested_stairs=("S1",)),
        )

        sim_sample = extract_ground_truth_sample(10.0, "simulation", crowd_snapshot=snapshot)
        live_sample = extract_ground_truth_sample(10.0, "live", crowd_snapshot=snapshot)

        self.assertEqual(sim_sample.total_occupant_count, live_sample.total_occupant_count)
        self.assertEqual(sim_sample.congestion_detected, live_sample.congestion_detected)
        self.assertEqual(sim_sample.congested_asset_ids, live_sample.congested_asset_ids)
        self.assertEqual(sim_sample.source, "simulation")
        self.assertEqual(live_sample.source, "live")

    def test_model_comparison_with_two_real_registries(self):

        # Phase 9 -- two independently-registered models (here, the SAME
        # trained artifact registered twice under different labels,
        # standing in for "model A vs model B" -- the comparison logic
        # itself is model-identity-agnostic) evaluated against the exact
        # SAME scenarios.
        gateway_a = _make_gateway()
        gateway_b = _make_gateway()

        registry_a = PredictionRegistry()
        registry_b = PredictionRegistry()
        gt_samples = []

        for i in range(6):

            t = float(i * 10)
            state = af.build_building_state_at_alarm_activation(_MODULE_STATE["building"], total_occupants=3, timestamp=t)

            registry_a.record_from_snapshot(gateway_a.predict(state, t), prediction_horizon_seconds=20.0, source="simulation")
            registry_b.record_from_snapshot(gateway_b.predict(state, t), prediction_horizon_seconds=20.0, source="simulation")

            gt_samples.append(GroundTruthSample(timestamp=t + 20.0, source="simulation", congestion_detected=(i % 2 == 0)))

        timeline = GroundTruthTimeline(gt_samples)

        matched_a = match_predictions(registry_a.all(), timeline, tolerance_seconds=1.0)
        matched_b = match_predictions(registry_b.all(), timeline, tolerance_seconds=1.0)

        report = compare_models({"model-a": matched_a, "model-b": matched_b})

        self.assertEqual(report.results_by_model["model-a"].evaluation_count, 6)
        self.assertEqual(report.results_by_model["model-b"].evaluation_count, 6)
        # Identical model/data -> identical metrics (not a stronger claim
        # than that -- this proves comparison plumbing works, not that
        # two DIFFERENT models were compared).
        self.assertEqual(
            report.results_by_model["model-a"].classification.confusion_matrix,
            report.results_by_model["model-b"].classification.confusion_matrix,
        )


if __name__ == "__main__":
    unittest.main()
