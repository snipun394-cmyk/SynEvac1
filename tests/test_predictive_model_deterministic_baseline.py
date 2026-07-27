import ast
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from predictive_model.baselines import DeterministicCurrentStateBaseline
from predictive_model.feature_prep_v2_1 import build_experimental_feature_matrix


# =====================================================
# Localized Predictive Model V3 milestone, Task #37/#44 -- tests for
# DeterministicCurrentStateBaseline, the "how well would an operator do
# using ONLY SynEvac's already-existing current-state read, with no
# learning at all" baseline this milestone's central question (does ML
# beat SynEvac's existing deterministic intelligence, not just random)
# depends on. Purely a fixed rule over two already-one-hot-encoded
# columns -- fit() must be a genuine no-op (no parameters learned), and
# predict_proba() must be a deterministic, monotonic function of
# congestion_level rank and trend, never touching y.
# =====================================================


def _synthetic_frame(rows):

    base = {
        "scenario_id": "scn-0", "candidate_id": "door-0", "candidate_type": "Door",
        "total_active_occupant_count": 10, "candidate_capacity": 1, "candidate_walking_distance": 5.0,
        "candidate_traversable": True, "candidate_adjacent_zone_occupancy": 5.0,
        "candidate_queue_length": 0, "candidate_approaching_count": 0,
        "candidate_recent_flow_rate": 0.0, "candidate_alternative_route_count": 1,
        "currently_congested": False, "had_any_activity_in_window": True, "target": False,
    }
    frame_rows = []
    for overrides in rows:
        row = dict(base)
        row.update(overrides)
        frame_rows.append(row)
    return pd.DataFrame(frame_rows)


class DeterministicCurrentStateBaselineTests(unittest.TestCase):

    def test_fit_is_a_genuine_no_op(self):
        """No parameter may change as a result of fit() -- score for a
        FIXED input row must be identical before and after fit() is
        called with arbitrary/adversarial labels."""

        frame = _synthetic_frame([
            {"candidate_congestion_level": "CRITICAL", "candidate_congestion_trend": "RISING"},
            {"candidate_congestion_level": "LOW", "candidate_congestion_trend": "STABLE"},
        ])
        prepared = build_experimental_feature_matrix(frame)
        model = DeterministicCurrentStateBaseline(prepared.feature_names)

        before = model.predict_proba(prepared.X).copy()
        model.fit(prepared.X, np.array([1, 1]))  # adversarial: opposite of what the rule would predict
        after = model.predict_proba(prepared.X)

        np.testing.assert_array_equal(before, after)

    def test_score_is_monotonic_in_congestion_level_rank(self):
        """Holding trend fixed (STABLE, +0 contribution), score must
        strictly increase LOW < MODERATE < HIGH < VERY_HIGH < CRITICAL."""

        levels = ["LOW", "MODERATE", "HIGH", "VERY_HIGH", "CRITICAL"]
        frame = _synthetic_frame([
            {"candidate_congestion_level": level, "candidate_congestion_trend": "STABLE"} for level in levels
        ])
        prepared = build_experimental_feature_matrix(frame)
        model = DeterministicCurrentStateBaseline(prepared.feature_names)
        model.fit(prepared.X, prepared.y)

        scores = model.predict_proba(prepared.X)
        self.assertTrue(np.all(np.diff(scores) > 0), f"expected strictly increasing scores, got {scores}")

    def test_rising_trend_scores_higher_than_stable_or_falling(self):

        frame = _synthetic_frame([
            {"candidate_congestion_level": "MODERATE", "candidate_congestion_trend": "RISING"},
            {"candidate_congestion_level": "MODERATE", "candidate_congestion_trend": "STABLE"},
            {"candidate_congestion_level": "MODERATE", "candidate_congestion_trend": "FALLING"},
        ])
        prepared = build_experimental_feature_matrix(frame)
        model = DeterministicCurrentStateBaseline(prepared.feature_names)
        model.fit(prepared.X, prepared.y)

        rising, stable, falling = model.predict_proba(prepared.X)
        self.assertGreater(rising, stable)
        self.assertGreater(rising, falling)

    def test_unknown_trend_scores_between_stable_and_rising(self):

        frame = _synthetic_frame([
            {"candidate_congestion_level": "MODERATE", "candidate_congestion_trend": "RISING"},
            {"candidate_congestion_level": "MODERATE", "candidate_congestion_trend": "UNKNOWN"},
            {"candidate_congestion_level": "MODERATE", "candidate_congestion_trend": "STABLE"},
        ])
        prepared = build_experimental_feature_matrix(frame)
        model = DeterministicCurrentStateBaseline(prepared.feature_names)
        model.fit(prepared.X, prepared.y)

        rising, unknown, stable = model.predict_proba(prepared.X)
        self.assertGreater(rising, unknown)
        self.assertGreater(unknown, stable)

    def test_output_range_and_extremes(self):
        """CRITICAL+RISING must hit exactly 1.0 (the theoretical max);
        LOW+STABLE/FALLING must hit exactly 0.0 (the theoretical min);
        every score in between must lie in [0, 1]."""

        frame = _synthetic_frame([
            {"candidate_congestion_level": "CRITICAL", "candidate_congestion_trend": "RISING"},
            {"candidate_congestion_level": "LOW", "candidate_congestion_trend": "STABLE"},
            {"candidate_congestion_level": "MODERATE", "candidate_congestion_trend": "FALLING"},
        ])
        prepared = build_experimental_feature_matrix(frame)
        model = DeterministicCurrentStateBaseline(prepared.feature_names)
        model.fit(prepared.X, prepared.y)

        scores = model.predict_proba(prepared.X)
        self.assertAlmostEqual(scores[0], 1.0)
        self.assertAlmostEqual(scores[1], 0.0)
        self.assertTrue(np.all(scores >= 0.0) and np.all(scores <= 1.0))

    def test_missing_congestion_columns_do_not_crash(self):
        """If a feature_names list happens not to include one of the
        one-hot congestion_level/trend columns (e.g. a degenerate
        synthetic split where a category never appears), the baseline
        must degrade gracefully (that contribution treated as 0), not
        raise."""

        frame = _synthetic_frame([
            {"candidate_congestion_level": "LOW", "candidate_congestion_trend": "STABLE"},
        ])
        prepared = build_experimental_feature_matrix(frame)
        # simulate a feature_names list missing the CRITICAL column entirely
        pruned_names = [n for n in prepared.feature_names if n != "candidate_congestion_level=LOW"]
        model = DeterministicCurrentStateBaseline(pruned_names)
        # X still has all real columns -- model just won't look at the pruned one
        model.fit(prepared.X, prepared.y)
        scores = model.predict_proba(prepared.X)
        self.assertEqual(scores.shape, (1,))

    def test_predict_proba_shape_matches_row_count(self):

        frame = _synthetic_frame([
            {"candidate_congestion_level": level, "candidate_congestion_trend": "STABLE"}
            for level in ["LOW", "MODERATE", "HIGH", "VERY_HIGH", "CRITICAL"]
        ])
        prepared = build_experimental_feature_matrix(frame)
        model = DeterministicCurrentStateBaseline(prepared.feature_names)
        model.fit(prepared.X, prepared.y)
        scores = model.predict_proba(prepared.X)
        self.assertEqual(scores.shape, (len(frame),))


class DeterministicBaselineLeakageBoundaryGuardTests(unittest.TestCase):
    """predictive_model/baselines.py only ever consumes already-extracted
    feature columns -- it must never import target-generation or
    leakage-adjacent modules, the same leakage-boundary discipline every
    extractor/target module in predictive_dataset already enforces."""

    def test_baselines_module_never_imports_target_generation_modules(self):

        baselines_path = Path(__file__).resolve().parent.parent / "predictive_model" / "baselines.py"
        tree = ast.parse(baselines_path.read_text(encoding="utf-8"))

        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module)

        self.assertFalse(any("target_generator" in name for name in names))
        self.assertFalse(any("target_semantics_analysis" in name for name in names))


if __name__ == "__main__":
    unittest.main()
