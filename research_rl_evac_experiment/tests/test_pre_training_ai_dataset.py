"""Tests for scripts/audit_pre_training_ai_dataset.py -- the READ-ONLY
pre-training data / feature / target integrity audit for the proposed
consequence-prediction experiment. Nothing here trains a model, implements
the selector, or writes to the simulator, observation.py, the controller,
the frozen RL branch, or the existing candidate dataset."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import audit_pre_training_ai_dataset as audit  # noqa: E402

from observation import OBSERVATION_INDEX  # noqa: E402  (importable via audit's DATA_ROOT path setup)


@pytest.fixture(scope="module")
def hashes_before():
    return audit.protected_hashes()


@pytest.fixture(scope="module")
def records(hashes_before):
    return audit.load_records()


@pytest.fixture(scope="module")
def rows(records):
    return audit.flatten_tuples(records)


@pytest.fixture(scope="module")
def states(records):
    return audit.replay_all_states(records)


@pytest.fixture(scope="module")
def manifest():
    return audit.build_feature_manifest()


@pytest.fixture(scope="module")
def adm(manifest):
    return audit.admissible_state_feature_names(manifest)


# ---------------------------------------------------------------------------
# Read-only guarantees.
# ---------------------------------------------------------------------------

class TestReadOnly:
    def test_no_protected_file_missing(self, hashes_before):
        assert "MISSING" not in hashes_before.values()

    def test_safe_write_rejects_non_deliverable_names(self, tmp_path):
        with pytest.raises(PermissionError):
            audit._safe_write("simulator.py", f"x {audit.AUDIT_MARKER}", set(), out_dir=str(tmp_path))

    def test_safe_write_refuses_to_overwrite_foreign_file(self, tmp_path):
        name = audit.OUTPUT_FILES[1]
        (tmp_path / name).write_text("{}", encoding="utf-8")
        with pytest.raises(PermissionError):
            audit._safe_write(name, {"audit": audit.AUDIT_MARKER}, set(), out_dir=str(tmp_path))
        assert (tmp_path / name).read_text(encoding="utf-8") == "{}"

    def test_safe_write_requires_marker(self, tmp_path):
        with pytest.raises(ValueError):
            audit._safe_write(audit.OUTPUT_FILES[1], {"x": 1}, set(), out_dir=str(tmp_path))

    def test_deliverable_names_do_not_collide_with_prior_design_artifacts(self):
        prior = {os.path.basename(p) for p in audit.PROTECTED_RELATIVE_PATHS}
        assert not (set(audit.OUTPUT_FILES) & prior)

    def test_frozen_controller_and_checkpoints_unchanged(self):
        assert audit.design.verify_controller_hash()["match"] is True
        ck = audit.design.verify_checkpoint_hashes()
        assert ck["n_checkpoints"] == 15 and ck["all_match"] is True


# ---------------------------------------------------------------------------
# Section 1: dataset structure.
# ---------------------------------------------------------------------------

class TestDatasetStructure:
    def test_exact_counts(self, records, rows):
        s = audit.dataset_structure(records, rows)
        assert s["n_records_decision_states"] == 450
        assert s["n_total_state_action_outcome_tuples"] == 9450
        assert s["n_candidate_tuples_GLOBAL_and_NO_INTERVENTION"] == 900
        assert s["n_pairwise_tuples"] == 8550
        assert s["n_unique_states"] == 450
        assert s["n_exact_duplicate_state_action_rows"] == 0
        assert s["every_tuple_has_simulator_outcome"] is True

    def test_no_feature_vector_is_stored(self, records, rows):
        assert audit.dataset_structure(records, rows)["stores_observation_or_feature_vector"] is False

    def test_scenarios_seeds_ticks(self, records, rows):
        s = audit.dataset_structure(records, rows)
        assert s["scenarios"] == {"S1": 150, "S2": 150, "S3": 150}
        assert s["seeds"]["n_unique"] == 50
        assert s["decision_ticks"] == {"0": 150, "5": 150, "10": 150}
        assert s["compliance_probabilities_represented"] == [0.9]
        assert s["hazard_timings_represented"] == {"S1": None, "S2": 7, "S3": 5}

    def test_candidate_set_is_unequal_across_states(self, records, rows):
        s = audit.dataset_structure(records, rows)
        assert s["n_states_with_zero_pairwise_rows"] == 250
        assert set(s["global_action_variants"]) <= {"GLOBAL:0", "GLOBAL:1", "GLOBAL:2"}


# ---------------------------------------------------------------------------
# Section 2: replay + leakage.
# ---------------------------------------------------------------------------

class TestReplayAndLeakage:
    def test_all_450_states_replay_consistently(self, records, states):
        r = audit.replay_consistency(records, states)
        assert r["passed"] and r["all_reached"] and r["n_replay_mismatches"] == 0

    def test_temporal_features_leak_future_onset(self, states):
        t = audit.temporal_leakage_check(states)
        assert t["leak_confirmed"] is True
        assert t["n_pre_onset_states_where_feature29_recovers_exact_future_t_h"] == t["n_pre_onset_hazard_states"] > 0

    def test_s2_tick0_feature29_recovers_t_h_7(self, states):
        s = states[("S2", 10000, 0)]
        i29 = audit.OBSERVATION_INDEX_TEMPORAL.index("time_since_hazard_onset_normalized")
        assert round(0 - s["obs31"][i29] * audit.MAX_STEPS) == 7

    def test_residual_capacity_does_not_leak(self, states):
        r = audit.residual_capacity_timing_check(states)
        assert r["passed"] is True and r["reveals_future_onset"] is False

    def test_q_labels_track_hidden_compliance(self, states):
        q = audit.q_label_compliance_leak_check(states)
        assert q["n_states_prior_action_favor_B_with_nonempty_hub"] > 0
        assert q["leaks_hidden_compliance"] is True


class TestFeatureManifest:
    def test_every_feature_has_exactly_one_valid_class(self, manifest):
        for m in manifest:
            assert m["classification"] in audit.CLASS_LABELS

    def test_every_baseline_and_temporal_index_is_classified(self, manifest):
        names = {m["name"] for m in manifest}
        assert set(audit.OBSERVATION_INDEX_TEMPORAL) <= names

    def test_leaky_and_hidden_features_classified(self, manifest):
        cls = {m["name"]: m["classification"] for m in manifest}
        assert cls["time_since_hazard_onset_normalized"] == "F"
        assert cls["hazard_phase"] == "F"
        assert cls["Q_A_at_H"] == "D" and cls["Q_B_at_H"] == "D"
        assert cls["residual_cap_H_CE1_normalized"] == "C"

    def test_admissible_set_is_clean(self, manifest, adm):
        cls = {m["name"]: m["classification"] for m in manifest}
        assert len(adm) == 26
        assert all(cls[n] in ("A", "B", "C") for n in adm)
        forbidden = {"Q_A_at_H", "Q_B_at_H", "time_since_hazard_onset_normalized", "hazard_phase",
                     "normalized_sim_time_temporal", "normalized_decision_index"}
        assert not (set(adm) & forbidden)
        assert not any(n.startswith("delta_") for n in adm)

    def test_admissible_vector_matches_observation(self, states, adm):
        s = states[("S3", 10000, 5)]
        v = audit.admissible_vector(s, adm)
        assert v[adm.index("waiting_H_total")] == s["obs27"][8] + s["obs27"][9]
        assert v[adm.index("waiting_R1")] == s["obs27"][OBSERVATION_INDEX.index("waiting_R1")]

    def test_trend_features_undefined_at_tick0(self, states):
        assert np.isnan(audit.trend_vector(states[("S2", 10000, 0)], None)).all()


# ---------------------------------------------------------------------------
# Section 3: action independence.
# ---------------------------------------------------------------------------

class TestActionIndependence:
    def test_only_hidden_label_features_are_action_sensitive(self, records, states):
        subset = [r for r in records if r["seed"] in (10000, 10001)]
        a = audit.action_independence_check(subset, states)
        assert a["observation_recomputed_on_unmodified_state_is_identical"] is True
        assert set(a["action_sensitive_features"]) <= {"Q_A_at_H", "Q_B_at_H"}

    def test_admissible_vector_unchanged_by_any_relabel(self, records, states, adm):
        import copy
        r = next(r for r in records if (r["scenario"], r["seed"], r["decision_tick"]) == ("S2", 10000, 0))
        s = states[audit.state_key(r)]
        base = audit.admissible_vector(s, adm)
        for key in list(r["pairwise_results"])[:10]:
            fork = copy.deepcopy(s["sim"])
            audit.apply_differential_action(fork, audit._parse_pair(key))
            after = dict(s, obs27=np.asarray(audit.compute_observation(fork), dtype=float))
            assert np.array_equal(audit.admissible_vector(after, adm), base)


# ---------------------------------------------------------------------------
# Section 4: labels.
# ---------------------------------------------------------------------------

class TestLabels:
    def test_stored_labels_reproduce_on_sample(self, records, states):
        subset = [r for r in records if r["seed"] in (10000, 10049)]
        rep = audit.reproduce_labels(subset, states, pairwise_stride=7)
        assert rep["passed"] is True
        assert rep["n_label_mismatches"] == 0
        assert rep["n_rollouts_not_starting_from_identical_state"] == 0

    def test_global_rollout_shared_within_trajectory(self, records, states):
        t = audit.target_structure_checks(records, states)
        assert t["n_trajectories_where_GLOBAL_outcome_identical_at_all_3_ticks"] == t["n_trajectories"] == 150

    def test_prior_unmet_is_embedded_in_stored_targets(self, records, states):
        t = audit.target_structure_checks(records, states)
        assert t["n_states_with_nonzero_prior_total_unmet"] > 0
        assert t["S1_hazard_edge_unmet_label_values"] == {0: 150}

    def test_target_audit_lists_required_issues(self):
        ta = audit.target_generation_audit(dict(n_rollouts_not_starting_from_identical_state=0, passed=True,
                                                hazard_unmet_post_diagnostic_vs_realized_tick_log={}), {})
        assert {i["id"] for i in ta["contamination_and_definition_issues"]} == {"T1", "T2", "T3", "T4", "T5", "T6"}


# ---------------------------------------------------------------------------
# Sections 5-8: dependence, split, identifiability.
# ---------------------------------------------------------------------------

class TestDependenceSplitIdentifiability:
    def test_states_collapse_to_few_distinct_inputs(self, records, rows, states, adm):
        td = audit.trajectory_dependence(records, rows, states, adm)
        rs = td["repeated_state_frequency"]
        assert rs["n_distinct_admissible_feature_vectors"] < 20
        assert td["states_per_trajectory"] == {"3": 150}

    def test_existing_grid_cannot_support_joint_split(self, records):
        sp = audit.split_design(records, 0.001)
        assert sp["n_existing_nonempty_cells"] == 3
        assert sp["joint_timing_x_compliance_split_possible_with_existing_data"] is False
        assert sp["heldout_compliance_split_possible_with_existing_data"] is False

    def test_proposed_primary_cells_are_jointly_held_out(self, records):
        nd = audit.split_design(records, 0.001)["proposed_new_data_design"]
        train_t, train_p = set(nd["axes"]["hazard_timing"]["train"]), set(nd["axes"]["compliance_p"]["train"])
        prim = [k for k, v in nd["cell_roles"].items() if v == "TEST_PRIMARY_JOINT"]
        assert prim
        for k in prim:
            th, p = k.split("|")
            assert th.split("=")[1] not in train_t and float(p.split("=")[1]) not in train_p
        assert all(v != "TRAIN" or k.split("|")[0].split("=")[1] in train_t for k, v in nd["cell_roles"].items())

    def test_s1_s2_tick0_same_input_different_outcome(self, records, rows, states, adm):
        table = audit.build_modeling_table(rows, states, adm)
        ti = audit.target_identifiability(table, records, states, adm)
        s = ti["S1_vs_S2_tick0"]
        assert s["identical_admissible_observation_across_S1_and_S2"] is True
        g = s["T100_by_action_and_scenario"]["NO_INTERVENTION"]
        assert g["S1"]["mean"] != g["S2"]["mean"]
        full = ti["variance_decomposition"]["T100"]["plus_full_compliance_vector_eq_trajectory_id"]
        assert full["irreducible_share"] == 0.0  # simulator determinism
        assert ti["variance_decomposition"]["T100"]["admissible_features_plus_action"]["irreducible_share"] > 0.0


# ---------------------------------------------------------------------------
# Sections 10-12: definitions + decision.
# ---------------------------------------------------------------------------

class TestDefinitionsAndDecision:
    def test_selector_preserves_established_ordering(self):
        sel = audit.selector_definition()
        assert sel["established_ordering"]["order"][0].startswith("T90")
        assert sel["established_ordering"]["order"][1:] == ["T100", "total_unmet_demand", "hazard_edge_unmet_demand"]
        assert [s["step"] for s in sel["steps"]] == [1, 2, 3, 4, 5, 6]

    def test_success_criteria_cover_all_five_families(self):
        c = audit.preregistered_success_criteria()
        for k in ("prediction_success", "recommendation_success", "safety_success", "generalization_success",
                  "closed_loop_success"):
            assert k in c
        assert c["generalization_success"]["families"]["D_joint_timing_x_compliance"] == "PRIMARY"

    def test_final_decision_is_requires_new_data(self, records, rows, states):
        s = audit.dataset_structure(records, rows)
        sp = audit.split_design(records, 0.001)
        d = audit.final_decision(s, sp, audit.temporal_leakage_check(states))
        assert d["classification"] == "C"


# ---------------------------------------------------------------------------
# Generated deliverables (only when the script has been run).
# ---------------------------------------------------------------------------

outputs_present = pytest.mark.skipif(
    not all(os.path.isfile(os.path.join(audit.OUT_DIR, n)) for n in audit.OUTPUT_FILES),
    reason="audit script not run yet",
)


@outputs_present
class TestDeliverables:
    def test_every_deliverable_carries_marker_and_parses(self):
        for n in audit.OUTPUT_FILES:
            text = open(os.path.join(audit.OUT_DIR, n), encoding="utf-8").read()
            assert audit.AUDIT_MARKER in text
            if n.endswith(".json"):
                json.loads(text)

    def test_manifest_and_decision_recorded(self):
        m = json.load(open(os.path.join(audit.OUT_DIR, "exact_feature_manifest.json"), encoding="utf-8"))
        assert m["n_admissible_v1_state_features"] == 26
        assert m["prior_design_feature_count_claim"]["consistent"] is False
        sp = json.load(open(os.path.join(audit.OUT_DIR, "split_design.json"), encoding="utf-8"))
        assert sp["final_decision"]["classification"] == "C"

    def test_recorded_integrity_and_label_reproduction(self):
        ds = json.load(open(os.path.join(audit.OUT_DIR, "dataset_structure.json"), encoding="utf-8"))
        assert ds["integrity"]["all_protected_unchanged"] is True
        assert ds["file_sha256"] == audit.sha256_file(audit.DATASET_PATH)
        tg = json.load(open(os.path.join(audit.OUT_DIR, "target_generation_audit.json"), encoding="utf-8"))
        assert tg["label_reproduction"]["n_rollouts_rerun"] == 9450
        assert tg["label_reproduction"]["n_label_mismatches"] == 0


def test_protected_files_unchanged_after_tests(hashes_before, states):
    assert audit.protected_hashes() == hashes_before
