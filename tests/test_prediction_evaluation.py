import unittest
from dataclasses import FrozenInstanceError

from crowd_intelligence.models import AssetApproachMetrics, BuildingCrowdSummary, CrowdIntelligenceSnapshot, TrendDirection
from stair_flow.models import StairFlowMetrics
from observable_assets.models import ObservationStatus

from hazard.severity import HazardSeverity
from building_state.models import HazardSummary

from prediction_evaluation.registry import PredictionRegistry
from prediction_evaluation.ground_truth import extract_ground_truth_sample
from prediction_evaluation.timeline import GroundTruthTimeline, match_predictions, unmatched_predictions
from prediction_evaluation.models import GroundTruthSample, PredictionRecord
from prediction_evaluation.pairs import bottleneck_classification_pairs, evacuation_time_regression_pairs
from prediction_evaluation.classification_metrics import compute_classification_metrics
from prediction_evaluation.regression_metrics import compute_regression_metrics
from prediction_evaluation.horizon_analysis import analyze_by_horizon
from prediction_evaluation.condition_analysis import analyze_by_condition
from prediction_evaluation.statistics import overall_statistics, per_building_statistics, per_scenario_statistics
from prediction_evaluation.comparison import compare_models


# =====================================================
# Prediction vs Reality Evaluation Framework milestone -- Phases 1-6,
# 8, 9 unit tests. Deterministic, offline, no real trained model
# required (see tests/test_prediction_evaluation_e2e.py for the full
# real-model proof) -- these tests exercise the framework's own logic
# against hand-built PredictionRecord/GroundTruthSample fixtures with
# known, independently-verifiable expected metric values.
# =====================================================


class _FakeBottleneck:
    def __init__(self, probability, predicted_occurrence, model_id="M1", model_version="v1"):
        self.probability = probability
        self.predicted_occurrence = predicted_occurrence
        self.model_id = model_id
        self.model_version = model_version


class _FakeEvacuationTime:
    def __init__(self, predicted_seconds, uncertainty_seconds=None):
        self.predicted_seconds = predicted_seconds
        self.uncertainty_seconds = uncertainty_seconds


class _FakePredictionSnapshot:
    def __init__(self, timestamp, bottleneck=None, evacuation_time_experimental=None, feature_schema_version="1.0"):
        self.timestamp = timestamp
        self.bottleneck = bottleneck
        self.evacuation_time_experimental = evacuation_time_experimental
        self.feature_schema_version = feature_schema_version


# =====================================================
# Phase 1 -- Prediction Registry
# =====================================================


class PredictionRegistryTests(unittest.TestCase):

    def test_record_is_immutable(self):

        registry = PredictionRegistry()
        record = registry.record(
            timestamp=10.0, prediction_horizon_seconds=20.0,
            payload=_FakePredictionSnapshot(10.0), model_id="M1", source="simulation",
        )

        with self.assertRaises(FrozenInstanceError):
            record.timestamp = 999.0

    def test_every_required_field_present(self):

        registry = PredictionRegistry()
        record = registry.record(
            timestamp=10.0, model_id="M1", model_version="v1", feature_schema_version="1.0",
            prediction_horizon_seconds=20.0, payload=_FakePredictionSnapshot(10.0), source="live",
        )

        self.assertIsNotNone(record.prediction_id)
        self.assertEqual(record.timestamp, 10.0)
        self.assertEqual(record.model_version, "v1")
        self.assertEqual(record.feature_schema_version, "1.0")
        self.assertEqual(record.prediction_horizon_seconds, 20.0)
        self.assertIsNotNone(record.payload)

    def test_registry_queries(self):

        registry = PredictionRegistry()
        registry.record(timestamp=0.0, prediction_horizon_seconds=5.0, payload=None, model_id="A", source="simulation", scenario_id="S1")
        registry.record(timestamp=1.0, prediction_horizon_seconds=5.0, payload=None, model_id="B", source="live", scenario_id="S2")

        self.assertEqual(len(registry), 2)
        self.assertEqual(len(registry.by_source("simulation")), 1)
        self.assertEqual(len(registry.by_model("B")), 1)
        self.assertEqual(len(registry.by_scenario("S1")), 1)

    def test_record_from_snapshot_extracts_model_metadata(self):

        registry = PredictionRegistry()
        snapshot = _FakePredictionSnapshot(5.0, bottleneck=_FakeBottleneck(0.7, True, model_id="M2", model_version="v3"))

        record = registry.record_from_snapshot(snapshot, prediction_horizon_seconds=20.0, source="live")

        self.assertEqual(record.timestamp, 5.0)
        self.assertEqual(record.model_id, "M2")
        self.assertEqual(record.model_version, "v3")
        self.assertIs(record.payload, snapshot)


# =====================================================
# Phase 2/3 -- Ground truth matching
# =====================================================


def _make_crowd_snapshot(congested_stairs=(), queue_by_stair=None, occupants=5):

    stair_metrics = {}
    for stair_id, queue in (queue_by_stair or {}).items():
        stair_metrics[stair_id] = AssetApproachMetrics(
            asset_id=stair_id, asset_type="Stair", position_available=True, queue_candidate_count=queue,
        )

    return CrowdIntelligenceSnapshot(
        timestamp=0.0,
        stair_metrics=stair_metrics,
        building_summary=BuildingCrowdSummary(total_observed_occupants=occupants, congested_stairs=tuple(congested_stairs)),
    )


class GroundTruthExtractionTests(unittest.TestCase):

    def test_congestion_detected_true_when_any_asset_congested(self):

        snapshot = _make_crowd_snapshot(congested_stairs=("S1",))
        sample = extract_ground_truth_sample(10.0, "live", crowd_snapshot=snapshot)

        self.assertTrue(sample.congestion_detected)
        self.assertIn("S1", sample.congested_asset_ids)

    def test_congestion_detected_false_when_nothing_congested(self):

        snapshot = _make_crowd_snapshot()
        sample = extract_ground_truth_sample(10.0, "simulation", crowd_snapshot=snapshot)

        self.assertFalse(sample.congestion_detected)
        self.assertEqual(sample.congested_asset_ids, ())

    def test_congestion_detected_none_without_crowd_snapshot(self):

        sample = extract_ground_truth_sample(10.0, "live")
        self.assertIsNone(sample.congestion_detected)
        self.assertIsNone(sample.total_occupant_count)

    def test_queue_lengths_extracted(self):

        snapshot = _make_crowd_snapshot(queue_by_stair={"S1": 3})
        sample = extract_ground_truth_sample(10.0, "live", crowd_snapshot=snapshot)

        self.assertEqual(sample.queue_lengths.get("S1"), 3)

    def test_hazard_severity_extracted(self):

        summary = HazardSummary(overall_severity=HazardSeverity.HIGH)
        sample = extract_ground_truth_sample(10.0, "simulation", hazard_summary=summary)

        self.assertEqual(sample.hazard_overall_severity, "HIGH")

    def test_evacuation_time_from_simulation_total(self):

        sample = extract_ground_truth_sample(10.0, "simulation", total_evacuation_time=142.5)

        self.assertTrue(sample.evacuation_complete)
        self.assertEqual(sample.evacuation_time_seconds, 142.5)

    def test_no_fabricated_evacuation_completion_without_evidence(self):

        sample = extract_ground_truth_sample(10.0, "live")
        self.assertIsNone(sample.evacuation_complete)
        self.assertIsNone(sample.evacuation_time_seconds)


class GroundTruthTimelineTests(unittest.TestCase):

    def test_resolves_nearest_sample_within_tolerance(self):

        timeline = GroundTruthTimeline([
            GroundTruthSample(timestamp=10.0, source="live"),
            GroundTruthSample(timestamp=20.0, source="live"),
            GroundTruthSample(timestamp=30.0, source="live"),
        ])

        resolved = timeline.resolve(21.0, tolerance_seconds=2.0)
        self.assertEqual(resolved.timestamp, 20.0)

    def test_returns_none_outside_tolerance(self):

        timeline = GroundTruthTimeline([GroundTruthSample(timestamp=10.0, source="live")])
        self.assertIsNone(timeline.resolve(50.0, tolerance_seconds=2.0))

    def test_empty_timeline_returns_none(self):

        timeline = GroundTruthTimeline([])
        self.assertIsNone(timeline.resolve(10.0, tolerance_seconds=5.0))

    def test_add_keeps_timeline_sorted(self):

        timeline = GroundTruthTimeline([GroundTruthSample(timestamp=10.0, source="live")])
        timeline.add(GroundTruthSample(timestamp=5.0, source="live"))
        timeline.add(GroundTruthSample(timestamp=15.0, source="live"))

        self.assertEqual([s.timestamp for s in timeline.all()], [5.0, 10.0, 15.0])

    def test_match_predictions_uses_timestamp_plus_horizon(self):

        registry = PredictionRegistry()
        prediction = registry.record(
            timestamp=10.0, prediction_horizon_seconds=20.0, payload=_FakePredictionSnapshot(10.0), source="live",
        )

        timeline = GroundTruthTimeline([GroundTruthSample(timestamp=30.0, source="live", congestion_detected=True)])

        matched = match_predictions([prediction], timeline, tolerance_seconds=1.0)

        self.assertEqual(len(matched), 1)
        self.assertEqual(matched[0].target_timestamp, 30.0)
        self.assertEqual(matched[0].match_time_delta_seconds, 0.0)

    def test_unmatched_predictions_reported(self):

        registry = PredictionRegistry()
        prediction = registry.record(
            timestamp=10.0, prediction_horizon_seconds=20.0, payload=None, source="live",
        )

        timeline = GroundTruthTimeline([])

        self.assertEqual(match_predictions([prediction], timeline, 1.0), ())
        self.assertEqual(unmatched_predictions([prediction], timeline, 1.0), (prediction,))


# =====================================================
# Phase 4 -- error metrics, hand-computed expected values
# =====================================================


class ClassificationMetricsTests(unittest.TestCase):

    def test_perfect_classifier(self):

        y_true = [True, False, True, False]
        y_pred = [True, False, True, False]
        y_proba = [0.9, 0.1, 0.8, 0.2]

        metrics = compute_classification_metrics(y_true, y_pred, y_proba)

        self.assertEqual(metrics.precision, 1.0)
        self.assertEqual(metrics.recall, 1.0)
        self.assertEqual(metrics.f1, 1.0)
        self.assertEqual(metrics.confusion_matrix, (2, 0, 0, 2))  # tn, fp, fn, tp
        self.assertEqual(metrics.roc_auc, 1.0)

    def test_known_confusion_matrix(self):

        # Pairs: (T,T)=TP (T,F)=FN (F,T)=FP (F,F)=TN (T,T)=TP (F,F)=TN
        # -> TP=2, FN=1, FP=1, TN=2.
        y_true = [True, True, False, False, True, False]
        y_pred = [True, False, True, False, True, False]

        metrics = compute_classification_metrics(y_true, y_pred)

        tn, fp, fn, tp = metrics.confusion_matrix
        self.assertEqual((tn, fp, fn, tp), (2, 1, 1, 2))
        self.assertAlmostEqual(metrics.precision, 2 / 3)
        self.assertAlmostEqual(metrics.recall, 2 / 3)

    def test_empty_input_returns_honest_none(self):

        metrics = compute_classification_metrics([], [])
        self.assertEqual(metrics.sample_count, 0)
        self.assertIsNone(metrics.precision)
        self.assertIsNone(metrics.confusion_matrix)

    def test_roc_auc_none_without_both_classes(self):

        metrics = compute_classification_metrics([True, True], [True, False], [0.9, 0.6])
        self.assertIsNone(metrics.roc_auc)

    def test_calibration_bias_sign(self):

        # Model consistently over-confident: predicts high probability
        # but actual outcome is False every time.
        y_true = [False] * 10
        y_pred = [True] * 10
        y_proba = [0.9] * 10

        metrics = compute_classification_metrics(y_true, y_pred, y_proba)

        self.assertGreater(metrics.confidence_bias, 0.0)  # over-confident toward "occurs"


class RegressionMetricsTests(unittest.TestCase):

    def test_known_mae_rmse(self):

        y_true = [100.0, 200.0, 300.0]
        y_pred = [110.0, 190.0, 320.0]

        metrics = compute_regression_metrics(y_true, y_pred)

        # errors: +10, -10, +20 -> MAE = (10+10+20)/3 = 13.333
        self.assertAlmostEqual(metrics.mae, 40 / 3)
        self.assertAlmostEqual(metrics.bias, 20 / 3)  # mean(pred - actual)

    def test_perfect_regression_zero_error(self):

        metrics = compute_regression_metrics([50.0, 60.0], [50.0, 60.0])
        self.assertEqual(metrics.mae, 0.0)
        self.assertEqual(metrics.rmse, 0.0)
        self.assertEqual(metrics.worst_case_error, 0.0)

    def test_mape_none_when_any_true_value_zero(self):

        metrics = compute_regression_metrics([0.0, 10.0], [1.0, 11.0])
        self.assertIsNone(metrics.mape)

    def test_mape_computed_when_no_zero_true_values(self):

        metrics = compute_regression_metrics([100.0, 200.0], [110.0, 190.0])
        self.assertIsNotNone(metrics.mape)

    def test_empty_input_honest_none(self):

        metrics = compute_regression_metrics([], [])
        self.assertEqual(metrics.sample_count, 0)
        self.assertIsNone(metrics.mae)

    def test_single_sample_no_ci(self):

        metrics = compute_regression_metrics([100.0], [110.0])
        self.assertIsNone(metrics.std_error)
        self.assertIsNone(metrics.error_ci_95_low)


# =====================================================
# Phase 5/6/8/9 -- horizon, condition, statistics, comparison
# =====================================================


def _matched_evaluation(timestamp, horizon, actual_congestion, predicted_occurrence, probability, tags=None, scenario_id=None, building_id=None):

    registry = PredictionRegistry()
    prediction = registry.record(
        timestamp=timestamp, prediction_horizon_seconds=horizon,
        payload=_FakePredictionSnapshot(timestamp, bottleneck=_FakeBottleneck(probability, predicted_occurrence)),
        source="simulation", context_tags=tags or {}, scenario_id=scenario_id, building_id=building_id,
    )
    timeline = GroundTruthTimeline([
        GroundTruthSample(timestamp=timestamp + horizon, source="simulation", congestion_detected=actual_congestion),
    ])
    return match_predictions([prediction], timeline, tolerance_seconds=0.5)[0]


class HorizonAnalysisTests(unittest.TestCase):

    def test_groups_by_configured_horizon_buckets(self):

        evaluations = [
            _matched_evaluation(0.0, 5.0, True, True, 0.9),
            _matched_evaluation(10.0, 20.0, False, False, 0.1),
            _matched_evaluation(20.0, 20.0, True, True, 0.8),
        ]

        results = analyze_by_horizon(evaluations)

        self.assertEqual(results[5.0].evaluation_count, 1)
        self.assertEqual(results[20.0].evaluation_count, 2)
        self.assertEqual(results[10.0].evaluation_count, 0)

    def test_accuracy_can_differ_across_horizons(self):

        evaluations = [
            _matched_evaluation(0.0, 5.0, True, True, 0.9),   # correct at 5s
            _matched_evaluation(10.0, 60.0, True, False, 0.2),  # wrong at 60s
        ]

        results = analyze_by_horizon(evaluations)

        self.assertEqual(results[5.0].classification.confusion_matrix, (0, 0, 0, 1))
        self.assertEqual(results[60.0].classification.confusion_matrix, (0, 0, 1, 0))


class ConditionAnalysisTests(unittest.TestCase):

    def test_groups_by_tag_value(self):

        evaluations = [
            _matched_evaluation(0.0, 20.0, True, True, 0.9, tags={"occupancy_level": "high"}),
            _matched_evaluation(10.0, 20.0, False, False, 0.1, tags={"occupancy_level": "low"}),
        ]

        results = analyze_by_condition(evaluations, "occupancy_level")

        self.assertEqual(set(results.keys()), {"high", "low"})
        self.assertEqual(results["high"].evaluation_count, 1)

    def test_evaluations_missing_the_tag_are_excluded(self):

        evaluations = [_matched_evaluation(0.0, 20.0, True, True, 0.9)]  # no tags
        results = analyze_by_condition(evaluations, "occupancy_level")
        self.assertEqual(results, {})


class StatisticsTests(unittest.TestCase):

    def test_per_scenario_and_per_building_grouping(self):

        evaluations = [
            _matched_evaluation(0.0, 20.0, True, True, 0.9, scenario_id="S1", building_id="B1"),
            _matched_evaluation(10.0, 20.0, False, False, 0.1, scenario_id="S2", building_id="B1"),
        ]

        by_scenario = per_scenario_statistics(evaluations)
        by_building = per_building_statistics(evaluations)

        self.assertEqual(set(by_scenario.keys()), {"S1", "S2"})
        self.assertEqual(by_building["B1"].evaluation_count, 2)

    def test_overall_statistics(self):

        evaluations = [_matched_evaluation(0.0, 20.0, True, True, 0.9)]
        overall = overall_statistics(evaluations)
        self.assertEqual(overall.evaluation_count, 1)


class ModelComparisonTests(unittest.TestCase):

    def test_compare_two_models(self):

        model_a_evals = [_matched_evaluation(0.0, 20.0, True, True, 0.9), _matched_evaluation(10.0, 20.0, False, False, 0.1)]
        model_b_evals = [_matched_evaluation(0.0, 20.0, True, False, 0.3), _matched_evaluation(10.0, 20.0, False, True, 0.6)]

        report = compare_models({"model-a": model_a_evals, "model-b": model_b_evals})

        self.assertEqual(report.results_by_model["model-a"].classification.f1, 1.0)
        self.assertEqual(report.results_by_model["model-b"].classification.f1, 0.0)
        self.assertEqual(report.better_classifier(), "model-a")

    def test_no_better_classifier_with_fewer_than_two_models(self):

        report = compare_models({"only-model": [_matched_evaluation(0.0, 20.0, True, True, 0.9)]})
        self.assertIsNone(report.better_classifier())


if __name__ == "__main__":
    unittest.main()
