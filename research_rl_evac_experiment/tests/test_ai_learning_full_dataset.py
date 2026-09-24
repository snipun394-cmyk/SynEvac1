"""Tests for the Phase-3 full-dataset generator invariants and its audit.
No model is trained; generation tests run a handful of trajectories in memory."""
from __future__ import annotations

import json
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import run_ai_learning_full_dataset as F  # noqa: E402
import run_ai_learning_pilot as P  # noqa: E402
import audit_ai_learning_full_dataset as FA  # noqa: E402
import audit_pre_training_ai_dataset as prior  # noqa: E402
import ai_learning_label_free_default as lfd  # noqa: E402


@pytest.fixture(scope="module")
def trajs():
    return F.full_trajectories(F.APPROVED_SEED_RULE)


class TestAllocationAndSeeds:
    def test_spec_v2_verified(self):
        assert len(F.verify_spec_v2()) == 11

    def test_spec_mismatch_stops(self, monkeypatch):
        monkeypatch.setattr(F.paths, "sha256_file", lambda p: "0" * 64)
        with pytest.raises(F.InvariantError):
            F.verify_spec_v2()

    def test_counts_match_frozen_allocation(self, trajs):
        dd = P._read_json("dataset_design.json")
        assert len(trajs) == dd["trajectories_FULL_total"] == 5544
        from collections import Counter
        c = Counter(t["split_role"] for t in trajs)
        for role, n in dd["trajectories_FULL_by_role"].items():
            assert c.get(role, 0) == n

    def test_unique_seeds_and_non_overlapping_ranges(self, trajs):
        seeds = [t["seed"] for t in trajs]
        assert len(set(seeds)) == 5544
        rep = F.seed_rule_report(trajs)
        assert rep["cross_role_seed_overlap"] == {}
        train = [t["seed"] for t in trajs if t["split_role"] == "TRAIN"]
        assert (min(train), max(train)) == (100000, 101799)
        assert not set(seeds) & set(range(125000, 126000))   # Test A reserved
        assert not set(seeds) & set(range(190000, 191000))   # pilot
        for role, r in rep["per_role"].items():
            if role != "TRAIN":
                assert r["inside_documented_range"]

    def test_roles_keys_and_templates(self, trajs):
        dd = P._read_json("dataset_design.json")
        zs = set(dd["factors"]["templates_zero_shot_only"])
        for t in trajs:
            assert t["compliance_key"] == f"AILX_v1|{t['split_role']}|{t['template']}"
            assert (t["template"] in zs) == (t["split_role"] == F.ZERO_SHOT_ROLE)
        prim = [t for t in trajs if t["split_role"] == "TEST_PRIMARY_JOINT"]
        train = [t for t in trajs if t["split_role"] == "TRAIN"]
        assert not ({t["t_h"] for t in prim} & {t["t_h"] for t in train})
        assert not ({t["p"] for t in prim} & {t["p"] for t in train})

    def test_seed_rule_must_be_explicit_and_approved(self, tmp_path, monkeypatch):
        with pytest.raises(F.InvariantError):
            F.full_trajectories("default")
        with pytest.raises(F.InvariantError):
            F.write_full("replicate_index_per_cell")

    def test_decisions_and_criteria_recorded(self):
        assert F.USER_DECISIONS["seed_rule"]["no_seed_reused"] is True
        assert "125000-125999" in F.USER_DECISIONS["test_A"]["reserved_range"]
        assert "LITERALLY" in F.CRITERIA_APPLICATION["X7_association"]


class TestGenerationInvariants:
    def test_smoke_generation(self):
        rows = []
        r = F.generate(F.APPROVED_SEED_RULE, limit_per_role=1, sink=rows.extend)
        assert r["n_rows"] == len(rows) == r["n_states"] * 298
        assert r["checks"]["same_start"] == r["checks"]["n_classes"] == r["checks"]["hazard_tick_log"]
        assert F.HEADER[18:44] == lfd.ADMISSIBLE_FEATURES
        ix = {c: i for i, c in enumerate(F.HEADER)}
        assert {row[ix["seed_rule"]] for row in rows} == {F.APPROVED_SEED_RULE}
        assert {row[ix["continuation_hash"]] for row in rows} == {lfd.policy_hash()}

    def test_targets_equal_pilot_functions(self, trajs):
        fs = P.load_frozen_spec()
        prov = dict(policy_id=lfd.POLICY_ID, policy_hash=lfd.policy_hash(),
                    dataset_design_sha256=fs["hashes"]["dataset_design.json"])
        st = P.pre_decision_states(trajs[900])[1]
        full_rows, _ = F.evaluate_state(st, fs["candidates"], prov, F.APPROVED_SEED_RULE)
        pilot_rows, _ = P.evaluate_state(st, fs["candidates"], prov)
        fi = {c: i for i, c in enumerate(F.HEADER)}
        pi = {c: i for i, c in enumerate(P.HEADER)}
        for fr, pr in zip(full_rows, pilot_rows):
            for k in P.TARGET_COLUMNS + ["candidate_id", "equivalence_class"]:
                assert fr[fi[k]] == pr[pi[k]]

    @pytest.mark.parametrize("bad", [dict(_same_start=False), dict(_hazard_matches_tick_log=False),
                                     dict(T100_censored=True), dict(future_unmet=float("inf")), dict(future_T100=-1)])
    def test_invariant_failures_stop(self, bad):
        o = dict(_same_start=True, _hazard_matches_tick_log=True, T100_censored=False, future_T90=5, future_T100=6,
                 future_unmet=3, future_hazard_unmet=0)
        o.update(bad)
        with pytest.raises(F.InvariantError):
            F._check_state([0] * 298, 298, {"c": o}, {})

    def test_share_helper(self):
        assert FA._shares([(2, 2.0, 2.0), (2, 4.0, 8.0)]) == 0.0
        assert math.isclose(FA._shares([(2, 1.0, 1.0)]), 1.0)


full_present = pytest.mark.skipif(
    not all(os.path.isfile(os.path.join(F.FULL_DIR, n)) for n in FA.OUTPUT_FILES + [F.ROWS_FILE]),
    reason="full dataset not generated / audited yet")


@full_present
class TestFullDeliverables:
    def _j(self, n):
        with open(os.path.join(F.FULL_DIR, n), encoding="utf-8") as f:
            return json.load(f)

    def test_generation_record_complete_and_hash(self):
        g = self._j(F.GENERATION_RECORD)
        assert g["status"] == "COMPLETE" and g["n_trajectories"] == 5544
        assert g["n_rows"] == g["n_decision_states"] * 298
        assert prior.sha256_file(os.path.join(F.FULL_DIR, F.ROWS_FILE)) == g["rows_file_sha256"]
        assert "seed_rule" in g["user_decisions"] and "test_A" in g["user_decisions"]

    def test_marked_and_parse(self):
        for n in FA.OUTPUT_FILES:
            with open(os.path.join(F.FULL_DIR, n), encoding="utf-8") as f:
                text = f.read()
            assert FA.MARKER in text
            if n.endswith(".json"):
                json.loads(text)

    def test_decision_consistent(self):
        d = self._j("full_go_no_go.json")
        assert set(d["results"]) == {f"G{i}" for i in range(1, 8)} | {f"X{i}" for i in range(1, 11)}
        assert d["decision"] == ("FULL DATASET GO" if not d["failed"] else "FULL DATASET NO-GO")

    def test_pilot_untouched(self):
        m = json.load(open(os.path.join(P.PILOT_DIR, "pilot_dataset_manifest.json"), encoding="utf-8"))
        assert prior.sha256_file(os.path.join(P.PILOT_DIR, P.ROWS_FILE)) == m["rows_file_sha256"]
