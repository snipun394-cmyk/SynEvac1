"""Tests for the Phase-1 AI-learning pilot: the label-free default (D2), the
specification v2 update (D1/D2), the pilot generator, and the pilot audit.
Nothing here trains a model, builds a selector, or generates the full
dataset. Generation tests use a few trajectories in memory and never write."""
from __future__ import annotations

import copy
import gzip
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import ai_learning_label_free_default as lfd  # noqa: E402
import run_ai_learning_pilot as P  # noqa: E402
import audit_ai_learning_pilot as A  # noqa: E402
import audit_label_free_default as lfa  # noqa: E402
import audit_final_experiment_specification as spec  # noqa: E402
import audit_pre_training_ai_dataset as prior  # noqa: E402
import update_experiment_specification_v2 as v2  # noqa: E402

from compliance import generate_compliance_vector  # noqa: E402
from differential_targeting import apply_differential_action  # noqa: E402
from observation import OBSERVATION_INDEX, compute_observation  # noqa: E402


@pytest.fixture(scope="module")
def hashes_before():
    return A.protected_hashes()


@pytest.fixture(scope="module")
def fs(hashes_before):
    return P.load_frozen_spec()


@pytest.fixture(scope="module")
def some_states(fs):
    trajs = P.trajectories(fs["pilot"])
    out = []
    for t in (trajs[0], trajs[37], trajs[101], trajs[199]):
        out.extend(P.pre_decision_states(t))
    return out


# ---------------------------------------------------------------------------
# D2: label-free default.
# ---------------------------------------------------------------------------

class TestLabelFreeDefault:
    def _x(self, h, degraded):
        x = np.zeros(26)
        x[lfd.ADMISSIBLE_FEATURES.index("waiting_H_total")] = h
        x[lfd.ADMISSIBLE_FEATURES.index("residual_cap_H_CE1_normalized")] = 2 / 6 if degraded else 1.0
        return x

    def test_closed_form(self):
        for h in range(0, 30):
            assert lfd.label_free_default_action(self._x(h, False)) == 1
            assert lfd.label_free_default_action(self._x(h, True)) == (2 if h >= 1 else 1)

    def test_only_26_admissible_inputs(self):
        with pytest.raises(ValueError):
            lfd.label_free_default_action(np.zeros(27))
        assert "Q_A_at_H" not in lfd.ADMISSIBLE_FEATURES and "Q_B_at_H" not in lfd.ADMISSIBLE_FEATURES
        assert lfd.ADMISSIBLE_FEATURES == prior.admissible_state_feature_names(prior.build_feature_manifest())

    def test_rule_input_splits_hub_evenly(self):
        obs = lfd.rebuild_rule_input(self._x(7, True))
        assert obs[OBSERVATION_INDEX.index("Q_A_at_H")] == obs[OBSERVATION_INDEX.index("Q_B_at_H")] == 3.5

    def test_matches_audited_margin0_policy(self, some_states):
        for s in some_states:
            obs = compute_observation(s["sim"])
            assert lfd.label_free_default_action(s["x"]) == lfa.POLICIES["SYMMETRIC_SPLIT_DEADBAND (margin 0)"](obs)


# ---------------------------------------------------------------------------
# Specification v2.
# ---------------------------------------------------------------------------

class TestSpecificationV2:
    def _rec(self):
        with open(os.path.join(prior.OUT_DIR, "spec_version_record.json"), encoding="utf-8") as f:
            return json.load(f)

    def test_changed_and_unchanged_files(self):
        rec = self._rec()
        for n, v in rec["files"].items():
            assert v["v2_sha256"] == prior.sha256_file(os.path.join(prior.OUT_DIR, n))
            assert v["v1_sha256"] == prior.sha256_file(os.path.join(v2.ARCHIVE_DIR, n))
            assert v["changed"] == (v["v1_sha256"] != v["v2_sha256"])
        assert all(rec["unchanged_verified"].values())
        assert not rec["files"]["candidate_action_set.json"]["changed"]
        assert not rec["files"]["final_feature_manifest.json"]["changed"]

    def test_d1_and_d2_applied(self):
        td = P._read_json("target_definitions.json")
        hz = td["targets"]["future_hazard_edge_unmet_demand"]
        assert "alternative" not in hz and "max(t_d, t_h)" in hz["definition"]
        assert td["continuation"]["policy_id"] == lfd.POLICY_ID
        sel = P._read_json("selector_design.json")
        assert "caveat" not in sel
        for k in ("selector_fallback", "abstention_fallback", "out_of_distribution_fallback"):
            assert sel["default_and_fallbacks"][k] == "d(s)"
        cl = P._read_json("closed_loop_evaluation.json")["policies"]
        assert "P2b_hidden_label_heuristic_REFERENCE" in cl and "P2_label_free_default" in cl

    def test_preserved_decisions(self):
        dd = P._read_json("dataset_design.json")
        assert dd["phases"]["PHASE_1_PILOT"]["exact_specification"]["n_trajectories"] == 240
        assert P._read_json("candidate_action_set.json")["size"] == 298
        assert P._read_json("final_feature_manifest.json")["n_features"] == 26

    def test_v1_spec_script_cannot_clobber_v2(self, tmp_path):
        (tmp_path / "scenario_space.json").write_text(json.dumps({"audit": v2.V2_MARKER}), encoding="utf-8")
        outputs = {n: ({"audit": spec.MARKER} if n.endswith(".json") else f"<!-- {spec.MARKER} -->")
                   for n in spec.OUTPUT_FILES}
        written = spec.write_outputs(dict(outputs=outputs), out_dir=str(tmp_path))
        assert "scenario_space.json" not in written

    def test_generator_refuses_without_v2(self, monkeypatch):
        real = P._read_json

        def fake(name):
            d = real(name)
            if name == "dataset_design.json":
                d = dict(d, spec_version="v1")
            return d
        monkeypatch.setattr(P, "_read_json", fake)
        with pytest.raises(RuntimeError):
            P.load_frozen_spec()


# ---------------------------------------------------------------------------
# Generator.
# ---------------------------------------------------------------------------

class TestGenerator:
    def test_trajectory_grid(self, fs):
        t = P.trajectories(fs["pilot"])
        assert len(t) == 240
        assert [x["seed"] for x in t] == list(range(190000, 190240))
        cells = {(x["template"], x["t_h"], x["p"]) for x in t}
        assert len(cells) == 8 * 5 * 3

    def test_features_are_label_free(self, some_states):
        for s in some_states:
            obs = compute_observation(s["sim"])
            h = s["x"][lfd.ADMISSIBLE_FEATURES.index("waiting_H_total")]
            assert h == len(s["sim"].zone_waiting["H"]) == obs[OBSERVATION_INDEX.index("Q_A_at_H")] + obs[OBSERVATION_INDEX.index("Q_B_at_H")]

    def test_pre_decision_generation_is_deterministic(self, fs):
        t = P.trajectories(fs["pilot"])[55]
        a, b = P.pre_decision_states(t), P.pre_decision_states(t)
        assert [s["fp"] for s in a] == [s["fp"] for s in b]
        assert all(np.array_equal(x["x"], y["x"]) for x, y in zip(a, b))

    def test_fast_label_map_equals_apply(self, some_states, fs):
        for s in some_states:
            for c in fs["candidates"][::17]:
                asg = P.restrict(s["sim"], c["asg"])
                fork = copy.deepcopy(s["sim"])
                pairs = [(z, v) for z, v in asg.items() if v is not None]
                if pairs:
                    apply_differential_action(fork, pairs)
                assert P.label_map(s["sim"], asg) == tuple(o["target_exit"] for _, o in sorted(fork.occupants.items()))

    def test_rollout_matches_independent_implementation(self, some_states, fs):
        cont = lfa.POLICIES["SYMMETRIC_SPLIT_DEADBAND (margin 0)"]
        for s in some_states[:6]:
            t_h = s["traj"]["t_h_int"]
            for c in fs["candidates"][::37]:
                asg = P.restrict(s["sim"], c["asg"])
                o = P.rollout(s, asg)
                assert o["_same_start"] and o["_hazard_matches_tick_log"]
                ref = lfa.rollout_with(s["sim"], {z: (spec.U if v is None else v) for z, v in asg.items()},
                                       t_h <= 180, t_h, cont)
                assert (o["future_T90"], o["future_T100"], o["future_unmet"], o["future_hazard_unmet"]) == lfa.key4(ref)

    def test_generate_small_in_memory(self):
        r = P.generate(limit_trajectories=2)
        assert len(r["rows"]) == r["n_states"] * 298
        assert r["checks"]["same_start"] == r["checks"]["n_classes"] == r["checks"]["hazard_tick_log"]
        feat_cols = P.HEADER[len(P.META_COLUMNS): len(P.META_COLUMNS) + 26]
        assert feat_cols == lfd.ADMISSIBLE_FEATURES


# ---------------------------------------------------------------------------
# Generated pilot + audit deliverables.
# ---------------------------------------------------------------------------

pilot_present = pytest.mark.skipif(
    not all(os.path.isfile(os.path.join(P.PILOT_DIR, n)) for n in A.OUTPUT_FILES + [P.ROWS_FILE]),
    reason="pilot not generated / audited yet")


@pilot_present
class TestPilotDeliverables:
    def _j(self, n):
        with open(os.path.join(P.PILOT_DIR, n), encoding="utf-8") as f:
            return json.load(f)

    def test_rows_file_matches_manifest(self):
        m = self._j("pilot_dataset_manifest.json")
        assert prior.sha256_file(os.path.join(P.PILOT_DIR, P.ROWS_FILE)) == m["rows_file_sha256"]
        with gzip.open(os.path.join(P.PILOT_DIR, P.ROWS_FILE), "rt", encoding="utf-8") as f:
            header = f.readline().strip().split(",")
        assert header == P.HEADER and m["n_rows"] == m["n_decision_states"] * 298 and m["n_trajectories"] == 240

    def test_all_marked_and_parse(self):
        for n in A.OUTPUT_FILES:
            with open(os.path.join(P.PILOT_DIR, n), encoding="utf-8") as f:
                text = f.read()
            assert A.MARKER in text
            if n.endswith(".json"):
                json.loads(text)

    def test_criteria_applied_unchanged(self):
        g = self._j("pilot_go_no_go.json")
        with open(os.path.join(prior.OUT_DIR, "diversity_requirements.json"), encoding="utf-8") as f:
            assert g["criteria_text"] == json.load(f)["go_no_go_after_pilot"] == v2.GO_CRITERIA
        assert set(g["results"]) == {f"G{i}" for i in range(1, 8)} | {f"X{i}" for i in range(1, 11)}
        assert g["decision"] == ("PILOT GO" if not g["failed"] else "PILOT NO-GO")

    def test_no_training_artifacts(self):
        names = set(os.listdir(P.PILOT_DIR))
        assert not any(n.endswith((".pkl", ".joblib", ".pt", ".zip", ".onnx")) for n in names)

    def test_continuation_audit(self):
        c = self._j("pilot_continuation_audit.json")
        assert c["label_free_passed"] and c["regeneration"]["identical_rows"]
        assert c["hidden_heuristic_booby_trap"]["regeneration_completed_with_hidden_heuristic_disabled"]


def test_protected_files_unchanged_after_tests(hashes_before):
    assert A.protected_hashes() == hashes_before
