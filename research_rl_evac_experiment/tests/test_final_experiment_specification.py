"""Tests for scripts/audit_final_experiment_specification.py -- the READ-ONLY
final experiment-specification / data-generation design audit. Nothing here
generates a dataset or pilot, trains a model, implements the selector, or
writes to the simulator, observation.py, the controller, the frozen RL
branch, or the existing 9,450-row dataset."""
from __future__ import annotations

import copy
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import audit_final_experiment_specification as spec  # noqa: E402
import audit_pre_training_ai_dataset as prior  # noqa: E402


@pytest.fixture(scope="module")
def hashes_before():
    return spec.protected_hashes()


@pytest.fixture(scope="module")
def s2_tick0(hashes_before):
    return prior.replay_state("S2", 10000, 0)


@pytest.fixture(scope="module")
def s3_tick5(hashes_before):
    return prior.replay_state("S3", 10000, 5)


# ---------------------------------------------------------------------------
# Read-only guarantees.
# ---------------------------------------------------------------------------

class TestReadOnly:
    def test_nothing_protected_is_missing(self, hashes_before):
        assert "MISSING" not in hashes_before.values()

    def test_safe_write_rejects_non_deliverables(self, tmp_path):
        with pytest.raises(PermissionError):
            spec._safe_write("observation.py", f"x {spec.MARKER}", set(), out_dir=str(tmp_path))

    def test_safe_write_refuses_foreign_file(self, tmp_path):
        (tmp_path / "scenario_space.json").write_text("{}", encoding="utf-8")
        with pytest.raises(PermissionError):
            spec._safe_write("scenario_space.json", {"audit": spec.MARKER}, set(), out_dir=str(tmp_path))

    def test_safe_write_may_supersede_only_the_prior_preregistration(self, tmp_path):
        (tmp_path / "preregistered_success_criteria.json").write_text(
            json.dumps({"audit": prior.AUDIT_MARKER}), encoding="utf-8")
        spec._safe_write("preregistered_success_criteria.json", {"audit": spec.MARKER}, set(), out_dir=str(tmp_path))
        (tmp_path / "split_specification.json").write_text(json.dumps({"audit": prior.AUDIT_MARKER}), encoding="utf-8")
        with pytest.raises(PermissionError):
            spec._safe_write("split_specification.json", {"audit": spec.MARKER}, set(), out_dir=str(tmp_path))

    def test_prior_audit_skips_superseded_file(self, tmp_path):
        (tmp_path / "preregistered_success_criteria.json").write_text(json.dumps({"audit": spec.MARKER}), encoding="utf-8")
        outputs = {n: ({"audit": prior.AUDIT_MARKER} if n.endswith(".json") else f"<!-- {prior.AUDIT_MARKER} -->")
                   for n in prior.OUTPUT_FILES}
        written = prior.write_outputs(dict(outputs=outputs), out_dir=str(tmp_path))
        assert "preregistered_success_criteria.json" not in written
        assert spec.MARKER in (tmp_path / "preregistered_success_criteria.json").read_text(encoding="utf-8")

    def test_prior_audit_decision_is_c(self):
        assert spec.load_prior_audit()["split"]["final_decision"]["classification"] == "C"


# ---------------------------------------------------------------------------
# Candidate action set.
# ---------------------------------------------------------------------------

class TestCandidateFamilies:
    def test_base2_structure(self):
        fam = spec.FIXED_FAMILIES["BASE2_298"]
        assert len(fam) == 298 == 1 + 3 * (1 + 7 * 2 + 21 * 4)
        names = list(fam)
        assert names[0] == "NO_INTERVENTION" and len(set(names)) == 298
        for n, a in fam.items():
            assert set(a) == set(spec.ELIGIBLE_ZONES)
            if n.startswith("BASE="):
                base = {"A": 0, "P": 1, "B": 2}[n[5]]
                overrides = [v for v in a.values() if v != base]
                assert len(overrides) <= 2

    def test_all_three_globals_present(self):
        fam = spec.FIXED_FAMILIES["BASE2_298"]
        for v in (0, 1, 2):
            assert any(set(a.values()) == {v} for a in fam.values())

    def test_absolute_representation_reproduces_env_semantics(self, s2_tick0):
        """DifferentialRLGymEnv: heuristic globally, then override. Absolute:
        BASE=h with the same override. Label maps must be identical."""
        sim = s2_tick0["sim"]
        h = int(spec.HEURISTIC(spec.compute_observation(sim)))
        for action_id in (6, 102, 150):
            asg = spec.decode_action(action_id)
            if not {z for z, _ in asg} <= set(spec.nonempty(sim)):
                continue
            env_fork = copy.deepcopy(sim)
            spec.apply_action(env_fork, h)
            spec.apply_differential_action(env_fork, asg)
            absolute = {z: h for z in spec.nonempty(sim)}
            absolute.update(dict(asg))
            assert spec.label_map_after(sim, absolute) == tuple(o["target_exit"] for _, o in sorted(env_fork.occupants.items()))

    def test_semantics_inventory_confirms_conflict(self):
        inv = spec.semantics_inventory()
        assert inv["S3_global_then_override"]["code_confirms"] is True
        assert inv["S2_apply_high_level_action"]["code_confirms"] is True


class TestRolloutAndTargets:
    def test_measurement_matches_simulator_tick_log(self, s2_tick0, s3_tick5):
        for s in (s2_tick0, s3_tick5):
            sim = s["sim"]
            for asg in list(spec._restrict(sim, spec.FIXED_FAMILIES["BASE2_298"]).values())[:15]:
                o = spec.rollout(sim, asg, s["has_hazard"], s["t_h"])
                assert o["check_hazard_all_ticks_equals_tick_log"] and o["check_hazard_onset_equals_tick_log"]

    def test_future_targets_are_relative_to_decision(self, s3_tick5):
        sim = s3_tick5["sim"]
        o = spec.rollout(sim, {z: spec.U for z in spec.nonempty(sim)}, s3_tick5["has_hazard"], s3_tick5["t_h"])
        assert o["future_T100"] == o["T100_abs"] - 5
        assert o["future_T90"] == o["T90_abs"] - 5
        assert o["T90_reached_before_decision"] is False

    def test_rollout_does_not_mutate_state(self, s3_tick5):
        fp = prior._state_fingerprint(s3_tick5["sim"])
        spec.rollout(s3_tick5["sim"], {z: 2 for z in spec.nonempty(s3_tick5["sim"])}, True, 5)
        assert prior._state_fingerprint(s3_tick5["sim"]) == fp

    def test_hazard_target_is_zero_without_hazard(self):
        s = prior.replay_state("S1", 10000, 0)
        o = spec.rollout(s["sim"], {z: 0 for z in spec.nonempty(s["sim"])}, s["has_hazard"], s["t_h"])
        assert o["future_hazard_unmet"] == 0

    def test_base2_matches_unrestricted_best_on_sample(self):
        states = [st for st in spec.existing_states() if st["key"] in ("S2|10000|0", "S3|10001|0", "S3|10000|5")]
        suff = spec.candidate_sufficiency(states)
        assert suff["per_candidate_set"]["BASE2_298"]["share_states_where_set_best_equals_unrestricted_best"] == 1.0
        assert suff["measurement_checks"]["hazard_all_ticks_matches_tick_log"] == suff["measurement_checks"]["n"]

    def test_known_interventions_reached_by_base2_not_side28(self):
        known = spec.known_interventions_expressible()
        assert all(v["by_family"]["BASE2_298"]["best_at_least_as_good"] for v in known.values())
        assert not all(v["by_family"]["SIDE28"]["best_at_least_as_good"] for v in known.values())

    def test_selection_rule(self):
        per = {f: dict(share_states_where_set_best_equals_unrestricted_best=1.0 if f == "BASE2_298" else 0.8,
                       max_T100_regret_ticks=0.0 if f == "BASE2_298" else 1.0) for f in spec.FIXED_FAMILIES}
        known = {"k": dict(by_family={f: dict(best_at_least_as_good=(f == "BASE2_298")) for f in spec.FIXED_FAMILIES})}
        assert spec.choose_candidate_set(dict(per_candidate_set=per), known)["chosen"] == "BASE2_298"


# ---------------------------------------------------------------------------
# Features, scenario space, split.
# ---------------------------------------------------------------------------

class TestFeaturesScenarioSplit:
    def test_feature_manifest(self):
        f = spec.final_feature_manifest_spec()
        assert f["n_features"] == 26
        names = {x["name"] for x in f["features"]}
        for bad in ("time_since_hazard_onset_normalized", "hazard_phase", "Q_A_at_H", "Q_B_at_H"):
            assert bad not in names and bad in f["excluded"]
        assert all(x["leakage_class"] in ("A", "B", "C") for x in f["features"])

    def test_compliance_vectors_nested_and_timing_independent(self):
        n = spec.compliance_nesting_check()
        assert n["vectors_nested_across_p_for_same_key_and_seed"] is True
        assert n["vector_independent_of_t_h"] is True

    def test_split_axes_are_disjoint(self):
        assert not (set(spec.T_TRAIN) & set(spec.T_TEST)) and not (set(spec.P_TRAIN) & set(spec.P_TEST))
        assert not (set(spec.TRAIN_TEMPLATES) & set(spec.ZERO_SHOT_TEMPLATES))
        assert spec.cell_role("4", 0.6) == "TEST_PRIMARY_JOINT"
        assert spec.cell_role("4", 0.5) == "TEST_HELDOUT_TIMING"
        assert spec.cell_role("5", 0.6) == "TEST_HELDOUT_COMPLIANCE"
        assert spec.cell_role("5", 0.5) == "TRAIN"

    def test_timing_not_tied_to_template(self):
        roles = spec.split_specification_spec()["matrix"]
        assert len(roles) == 9 and all(len(v) == 7 for v in roles.values())

    def test_compliance_keys_differ_by_role(self):
        assert spec.compliance_key("D0_sym", "TRAIN") != spec.compliance_key("D0_sym", "TEST_PRIMARY_JOINT")

    def test_tick0_input_identical_across_timing_and_compliance(self):
        xs = {next(spec.reference_decision_states(spec.DISTRIBUTIONS["D1_east_heavy"], th, p, 3, "k", 0.0))["x"]
              for th in ("none", "4", "8") for p in (0.5, 0.95)}
        assert len(xs) == 1

    def test_epsilon_increases_distinct_inputs(self):
        def count(eps):
            return len({d["x"] for seed in range(4) for th in ("5", "9") for p in (0.5, 0.9)
                        for d in spec.reference_decision_states(spec.DISTRIBUTIONS["D0_sym"], th, p, seed, "k", eps)})
        assert count(0.5) > count(0.0)


# ---------------------------------------------------------------------------
# Decision.
# ---------------------------------------------------------------------------

class TestDecision:
    def test_open_decisions_block_generation(self):
        suff = dict(per_candidate_set={"BASE2_298": dict(share_states_where_set_best_equals_unrestricted_best=1.0,
                                                         mean_T100_regret_ticks=0.0)})
        d = spec.go_no_go(suff, {}, dict(chosen="BASE2_298"))
        assert d["classification"] == "REQUIRES ONE MORE DESIGN DECISION"
        assert {x["id"] for x in d["open_decisions"]} == {"D1", "D2"}
        d0 = spec.go_no_go(suff, {}, dict(chosen=None))
        assert "D0" in {x["id"] for x in d0["open_decisions"]}


# ---------------------------------------------------------------------------
# Generated deliverables.
# ---------------------------------------------------------------------------

outputs_present = pytest.mark.skipif(
    not all(os.path.isfile(os.path.join(spec.OUT_DIR, n)) for n in spec.OUTPUT_FILES),
    reason="final specification audit not run yet")


@outputs_present
class TestDeliverables:
    def _j(self, n):
        with open(os.path.join(spec.OUT_DIR, n), encoding="utf-8") as f:
            return json.load(f)

    def test_all_deliverables_marked_and_parse(self):
        for n in spec.OUTPUT_FILES:
            with open(os.path.join(spec.OUT_DIR, n), encoding="utf-8") as f:
                text = f.read()
            assert spec.MARKER in text
            if n.endswith(".json"):
                json.loads(text)

    def test_candidate_set_recorded(self):
        c = self._j("candidate_action_set.json")
        assert c["name"] == "BASE2_298" and c["size"] == 298 and len(c["candidates"]) == 298
        assert [x["id"] for x in c["candidates"]] == list(range(298))

    def test_status_and_preregistration(self):
        s = self._j("split_specification.json")
        assert s["final_go_no_go"]["classification"] == "REQUIRES ONE MORE DESIGN DECISION"
        p = self._j("preregistered_success_criteria.json")
        prev = p["previous_draft_superseded"]
        assert prev and prev["content"]["audit"] == prior.AUDIT_MARKER and len(prev["sha256"]) == 64

    def test_integrity_recorded(self):
        d = self._j("dataset_design.json")
        assert d["integrity"]["all_unchanged"] is True
        assert d["integrity"]["controller"]["match"] is True


def test_protected_files_unchanged_after_tests(hashes_before, s2_tick0):
    after = spec.protected_hashes()
    # The prior audit script is edited by this milestone only between runs, never during a test session.
    assert after == hashes_before
