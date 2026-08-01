from typing import Optional

from crowd_intelligence.models import DensityThresholds, IntensityLevel


# =====================================================
# Calibration Benchmark Framework V1 -- "Prediction Accuracy" and
# "Recommendation Effectiveness" are the two metrics this milestone's
# own worked example names that no ground-truth field maps to directly.
# Both are deliberately kept lightweight and self-contained rather than
# reaching into predictive_dataset/predictive_model's own full
# feature-extraction/training pipeline -- doing that per benchmark
# scenario would mean training or re-scoring a real ML model inside a
# benchmarking tool, which risks drifting into exactly the "automatic
# calibration" this milestone is explicitly told not to build. Both
# reference implementations below reuse ONLY data a normal
# GroundTruth/MetricSample already produces for this same run -- no new
# pipeline, no trained model, no feature schema dependency.
#
# AdditionalMetric is an explicit extension point (not a closed set):
# a future, heavier Prediction Accuracy metric that does reuse
# predictive_model's own trained baselines can be added later as a
# second implementation of this same interface without touching
# harness.py.
# =====================================================


class AdditionalMetric:

    name: str

    def compute(
        self, ground_truth, movement_result, building, peak_occupancy_ratio, density_thresholds,
    ) -> Optional[float]:

        raise NotImplementedError


# =====================================================


class RecommendationEffectivenessMetric(AdditionalMetric):

    # Fraction of this run's own engineering-findings targets (doors
    # that became bottlenecks, exits/stairs exceeding capacity -- all
    # three already computed by ground_truth.bottleneck.
    # compute_engineering_findings()) that GroundTruth.recommendations
    # (ground_truth/recommendations.py, deterministic, rule-based) also
    # produced a matching, target_id-addressed recommendation for.
    # None (not 0.0) when a scenario had no engineering problem to
    # address at all -- an "honestly not applicable" result, never a
    # fabricated perfect score.

    name = "recommendation_effectiveness"

    def compute(self, ground_truth, movement_result, building, peak_occupancy_ratio, density_thresholds):

        problem_targets = (
            set(ground_truth.doors_that_became_bottlenecks)
            | set(ground_truth.exits_exceeding_capacity)
            | set(ground_truth.stairs_exceeding_capacity)
        )

        if not problem_targets:
            return None

        addressed_targets = {
            recommendation["target_id"]
            for recommendation in ground_truth.recommendations
            if recommendation.get("target_id") in problem_targets
        }

        return len(addressed_targets) / len(problem_targets)


# =====================================================


_CONGESTED_LEVELS = (IntensityLevel.HIGH, IntensityLevel.VERY_HIGH, IntensityLevel.CRITICAL)


class PredictionAccuracyMetric(AdditionalMetric):

    # Scoped specifically to DensityThresholds-type candidates (an
    # honestly-disclosed limitation, not a generic prediction-accuracy
    # score): classifies THIS run's own peak_occupancy_ratio (already
    # computed by metrics.compute_peak_occupancy_ratio()) using
    # whichever DensityThresholds this arm (baseline or candidate) was
    # actually given, and checks whether that classification (congested
    # vs. not) agrees with what the SAME run's GroundTruth already
    # determined actually happened (a real exceeded-capacity/bottleneck
    # outcome). No trained model, no feature vector, no held-out data --
    # a single deterministic rule scored against the one scenario that
    # produced it. Returns None whenever there is nothing to classify
    # (peak_occupancy_ratio itself is None) rather than fabricating an
    # accuracy value from missing data.

    name = "prediction_accuracy"

    def compute(self, ground_truth, movement_result, building, peak_occupancy_ratio, density_thresholds):

        if peak_occupancy_ratio is None or density_thresholds is None:
            return None

        actually_congested = bool(
            ground_truth.exits_exceeding_capacity
            or ground_truth.stairs_exceeding_capacity
            or ground_truth.doors_that_became_bottlenecks
        )

        predicted_level = density_thresholds.classify(peak_occupancy_ratio)
        predicted_congested = predicted_level in _CONGESTED_LEVELS

        return 1.0 if predicted_congested == actually_congested else 0.0
