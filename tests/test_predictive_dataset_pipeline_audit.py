import csv
import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from predictive_dataset.campaign_config_v4 import build_campaign_config_v4
from predictive_dataset.campaign_runner_v4 import CSV_COLUMNS, run_campaign_v4
from predictive_dataset.quality_checks import zero_walking_distance_candidates
from predictive_dataset.schema_v4 import CANDIDATE_FEATURE_NAMES_V4
from predictive_dataset.topologies_v3 import with_scenario_count
from predictive_dataset.topologies_v4 import all_structural_variants_v4

from predictive_model.feature_prep_v2_1 import build_experimental_feature_matrix, trainable_rows
from predictive_model.tree_models import build_tree_models


# =====================================================
# Synthetic Dataset Generation Stress Test & ML Compatibility Audit
# milestone -- permanent regression coverage for the properties this
# audit verified empirically at scale (10-5000+ scenarios): schema
# stability, zero row-level corruption, zero reintroduction of the
# historical zero-walking-distance Stair bug at the DATASET level (02b958b
# added the guard at the simulation-engine level; this is the dataset-
# generation-level companion), and unmodified downstream trainability.
# A single tiny campaign (1 scenario per variant, 24 total) is generated
# once for the whole module -- fast (a few seconds, matching the audit's
# own measured throughput), never a multi-thousand-scenario run in CI.
# =====================================================

_MODULE_STATE = {}


def setUpModule():

    tmp_dir = Path(tempfile.mkdtemp(prefix="pred_dataset_audit_"))

    variants = all_structural_variants_v4()
    tiny_variants = tuple(
        type(v)(v.family, v.variant_id, v.variant_label, with_scenario_count(v.topology, 1))
        for v in variants
    )
    config = build_campaign_config_v4(tiny_variants, campaign_version="pipeline-audit-regression-test")

    result = run_campaign_v4(tiny_variants, config, tmp_dir)

    _MODULE_STATE["tmp_dir"] = tmp_dir
    _MODULE_STATE["result"] = result
    _MODULE_STATE["frame"] = pd.read_csv(result["csv_path"])


def tearDownModule():
    shutil.rmtree(_MODULE_STATE.get("tmp_dir", ""), ignore_errors=True)


class DatasetGenerationSucceedsTests(unittest.TestCase):

    def test_every_variant_produces_at_least_one_accepted_scenario(self):

        result = _MODULE_STATE["result"]
        self.assertEqual(result["failed_scenarios"], 0)
        self.assertGreaterEqual(result["accepted_scenarios"], 24)

    def test_csv_columns_exactly_match_declared_schema(self):

        with open(_MODULE_STATE["result"]["csv_path"], newline="", encoding="utf-8") as f:
            header = next(csv.reader(f))

        self.assertEqual(tuple(header), CSV_COLUMNS)

        declared_features = set(name for name in CANDIDATE_FEATURE_NAMES_V4 if name != "candidate_type")
        self.assertTrue(declared_features.issubset(set(header)))


class DatasetIntegrityTests(unittest.TestCase):

    def test_no_duplicate_scenario_tick_candidate_rows(self):

        frame = _MODULE_STATE["frame"]
        key_cols = ["scenario_id", "observation_time", "candidate_id"]
        self.assertEqual(frame.duplicated(subset=key_cols).sum(), 0)

    def test_no_negative_times_or_counts(self):

        frame = _MODULE_STATE["frame"]
        numeric_nonnegative_cols = [
            "observation_time", "total_active_occupant_count", "candidate_capacity",
            "candidate_walking_distance", "candidate_queue_length", "candidate_approaching_count",
            "candidate_recent_flow_rate", "candidate_alternative_route_count",
            "candidate_upstream_catchment_count",
        ]
        for col in numeric_nonnegative_cols:
            coerced = pd.to_numeric(frame[col], errors="coerce")
            self.assertEqual(int((coerced < 0).sum()), 0, f"{col} has negative values")

    def test_lead_time_only_populated_when_target_true(self):

        frame = _MODULE_STATE["frame"]
        violating = frame.loc[frame["target_v2"].astype(str) != "True", "lead_time_seconds_v2"].notna().sum()
        self.assertEqual(violating, 0)

    def test_scenario_metadata_json_count_matches_accepted_scenarios(self):

        import json
        with open(_MODULE_STATE["tmp_dir"] / "scenario_metadata.json", encoding="utf-8") as f:
            metadata = json.load(f)

        self.assertEqual(len(metadata), _MODULE_STATE["result"]["accepted_scenarios"])

    def test_every_scenario_id_in_csv_exists_in_metadata(self):

        import json
        with open(_MODULE_STATE["tmp_dir"] / "scenario_metadata.json", encoding="utf-8") as f:
            metadata = json.load(f)

        meta_ids = {m["scenario_id"] for m in metadata}
        csv_ids = set(_MODULE_STATE["frame"]["scenario_id"].unique())
        self.assertTrue(csv_ids.issubset(meta_ids))


class StairRegressionGuardTests(unittest.TestCase):
    """Dataset-generation-level companion to 02b958b's simulation-engine-
    level guard: proves a freshly generated candidate dataset never
    contains a Stair candidate whose walking_distance is constant-zero
    across every row (the historical bug's dataset-visible symptom)."""

    def test_no_zero_walking_distance_stair_candidates(self):

        frame = _MODULE_STATE["frame"]
        stair_rows = frame[frame["candidate_type"] == "Stair"]

        self.assertGreater(len(stair_rows), 0, "campaign produced zero Stair rows -- test cannot verify anything")

        check = zero_walking_distance_candidates(stair_rows.to_dict("records"))
        self.assertEqual(check["flagged_zero_distance_candidate_ids"], [])

    def test_all_stair_candidates_traversable(self):

        frame = _MODULE_STATE["frame"]
        stair_rows = frame[frame["candidate_type"] == "Stair"]
        self.assertTrue((stair_rows["candidate_traversable"] == True).all())  # noqa: E712


class TrainingPipelineCompatibilityTests(unittest.TestCase):
    """Proves predictive_model's existing research training scripts still
    consume a freshly generated V4 dataset without modification -- the
    audit's Phase 6 finding, pinned as a regression test."""

    def test_freshly_generated_dataset_trains_without_modification(self):

        frame = _MODULE_STATE["frame"].rename(columns={
            "target_v2": "target", "currently_congested_v2": "currently_congested",
            "had_any_activity_in_window_v2": "had_any_activity_in_window",
            "lead_time_seconds_v2": "lead_time_seconds",
        })

        trainable = trainable_rows(frame)
        self.assertGreater(len(trainable), 0)

        prepared = build_experimental_feature_matrix(trainable)
        self.assertEqual(prepared.X.shape[0], len(trainable))
        self.assertGreater(prepared.X.shape[1], 0)

        if len(set(prepared.y.tolist())) < 2:
            self.skipTest("tiny 24-scenario campaign happened to produce a single-class target -- "
                           "not enough data for a meaningful fit, schema/shape checks above already passed")

        models = build_tree_models(seed=1, n_jobs=1)
        model = models["xgboost"]
        model.fit(prepared.X, prepared.y)
        proba = model.predict_proba(prepared.X)

        self.assertEqual(len(proba), len(prepared.y))
        self.assertTrue(((proba >= 0.0) & (proba <= 1.0)).all())


if __name__ == "__main__":
    unittest.main()
