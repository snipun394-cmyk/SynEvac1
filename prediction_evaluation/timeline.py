import bisect
from typing import Optional, Sequence, Tuple

from prediction_evaluation.models import GroundTruthSample, MatchedEvaluation, PredictionRecord


# =====================================================
# Prediction vs Reality Evaluation Framework milestone, Phase 2 -- the
# GroundTruthResolver. Given a prediction made at time T with horizon H,
# automatically locates the actual GroundTruthSample nearest T + H,
# honoring a configurable tolerance -- exactly Phase 2's own wording.
# Purely a SEARCH over an already-recorded sequence of GroundTruthSamples
# a caller has already produced (via prediction_evaluation.ground_truth.
# extract_ground_truth_sample(), once per cycle, in EITHER a Simulation
# or a Live Runtime context) -- never re-derives, never re-simulates,
# never queries a live system itself.
# =====================================================


class GroundTruthTimeline:

    def __init__(self, samples: Sequence[GroundTruthSample] = ()):

        self._samples = sorted(samples, key=lambda s: s.timestamp)
        self._timestamps = [s.timestamp for s in self._samples]

    # =====================================================

    def add(self, sample: GroundTruthSample) -> None:

        # Sorted insert (bisect) -- callers append roughly in time
        # order, but this stays correct even if a caller supplies
        # samples out of order (e.g. combining two sources after the
        # fact).
        index = bisect.bisect_left(self._timestamps, sample.timestamp)
        self._timestamps.insert(index, sample.timestamp)
        self._samples.insert(index, sample)

    # =====================================================

    def resolve(self, target_timestamp: float, tolerance_seconds: float) -> Optional[GroundTruthSample]:

        # Nearest-timestamp match within tolerance -- Phase 2's own
        # "support configurable tolerances" requirement. None (never an
        # arbitrarily-chosen sample) when nothing in this timeline falls
        # within [target - tolerance, target + tolerance].

        if not self._samples:
            return None

        index = bisect.bisect_left(self._timestamps, target_timestamp)

        candidates = []

        if index < len(self._samples):
            candidates.append(self._samples[index])

        if index > 0:
            candidates.append(self._samples[index - 1])

        if not candidates:
            return None

        best = min(candidates, key=lambda s: abs(s.timestamp - target_timestamp))

        if abs(best.timestamp - target_timestamp) > tolerance_seconds:
            return None

        return best

    # =====================================================

    def __len__(self) -> int:

        return len(self._samples)

    def all(self) -> Tuple[GroundTruthSample, ...]:

        return tuple(self._samples)


DEFAULT_TOLERANCE_SECONDS = 2.0


def match_predictions(
    predictions: Sequence[PredictionRecord],
    timeline: GroundTruthTimeline,
    tolerance_seconds: float = DEFAULT_TOLERANCE_SECONDS,
) -> Tuple[MatchedEvaluation, ...]:

    # Phase 2's own top-level entry point -- every prediction that finds
    # an honest ground-truth match within tolerance becomes exactly one
    # MatchedEvaluation; a prediction with no match within tolerance is
    # silently excluded (never forced into a fabricated match) -- see
    # unmatched_predictions() below for the complementary "which ones
    # were dropped, and why" query.

    matched = []

    for prediction in predictions:

        target_timestamp = prediction.timestamp + prediction.prediction_horizon_seconds
        ground_truth = timeline.resolve(target_timestamp, tolerance_seconds)

        if ground_truth is None:
            continue

        matched.append(MatchedEvaluation(
            prediction=prediction,
            ground_truth=ground_truth,
            target_timestamp=target_timestamp,
            match_time_delta_seconds=abs(ground_truth.timestamp - target_timestamp),
        ))

    return tuple(matched)


def unmatched_predictions(
    predictions: Sequence[PredictionRecord],
    timeline: GroundTruthTimeline,
    tolerance_seconds: float = DEFAULT_TOLERANCE_SECONDS,
) -> Tuple[PredictionRecord, ...]:

    matched_ids = {
        m.prediction.prediction_id
        for m in match_predictions(predictions, timeline, tolerance_seconds)
    }

    return tuple(p for p in predictions if p.prediction_id not in matched_ids)
