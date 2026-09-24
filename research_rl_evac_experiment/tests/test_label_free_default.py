"""Tests for scripts/audit_label_free_default.py -- the READ-ONLY audit that
selects a label-free continuation / fallback default after decisions D1 and
D2. No pilot, no training data, no model, no policy module, and no frozen
specification file is written by this audit or these tests."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import audit_label_free_default as lf  # noqa: E402
import audit_final_experiment_specification as spec  # noqa: E402
import audit_pre_training_ai_dataset as prior  # noqa: E402

M0 = "SYMMETRIC_SPLIT_DEADBAND (margin 0)"
M75 = "SYMMETRIC_SPLIT_DEADBAND (margin 0.75)"


@pytest.fixture(scope="module")
def hashes_before():
    return lf.protected_hashes()


def _obs(h_total: float, qa: float, degraded: bool) -> np.ndarray:
    o = np.zeros(27)
    o[lf.IQA], o[lf.IQB] = qa, h_total - qa
    o[lf.IRES] = 2.0 / 6.0 if degraded else 1.0
    return o


class TestReadOnly:
    def test_nothing_protected_missing(self, hashes_before):
        assert "MISSING" not in hashes_before.values()

    def test_frozen_spec_files_are_not_deliverables(self):
        assert not (set(lf.OUTPUT_FILES) & set(spec.OUTPUT_FILES))
        for n in spec.OUTPUT_FILES:
            assert f"OUT_DIR/{n}" in lf.protected_hashes()

    def test_safe_write_guards(self, tmp_path):
        with pytest.raises(PermissionError):
            lf._safe_write("candidate_action_set.json", {"audit": lf.MARKER}, set(), out_dir=str(tmp_path))
        (tmp_path / "fallback_validity.json").write_text("{}", encoding="utf-8")
        with pytest.raises(PermissionError):
            lf._safe_write("fallback_validity.json", {"audit": lf.MARKER}, set(), out_dir=str(tmp_path))


class TestLabelFreePolicies:
    @pytest.mark.parametrize("degraded", [False, True])
    def test_margin0_closed_form(self, degraded):
        for h in range(0, 21):
            expected = 2 if (degraded and h >= 1) else 1
            assert lf.POLICIES[M0](_obs(h, 0.0, degraded)) == expected

    @pytest.mark.parametrize("degraded", [False, True])
    def test_margin075_closed_form(self, degraded):
        for h in range(0, 21):
            expected = 2 if (degraded and h >= 5) else 1
            assert lf.POLICIES[M75](_obs(h, 0.0, degraded)) == expected

    def test_label_free_invariant_to_hidden_split(self):
        for name in lf.LABEL_FREE:
            for h in range(0, 15):
                for degraded in (False, True):
                    acts = {lf.POLICIES[name](_obs(h, qa, degraded)) for qa in np.linspace(0, h, 5)}
                    assert len(acts) == 1, name

    def test_hidden_heuristic_is_not_label_free(self):
        hidden = lf.POLICIES["HIDDEN_GLOBAL_DEADBAND (reference only)"]
        assert hidden(_obs(10, 10, False)) != hidden(_obs(10, 0, False))

    def test_inventory_marks_every_adaptive_controller_hidden(self):
        inv = lf.candidate_inventory()
        free = [c["name"] for c in inv if c["label_free"]]
        assert all("DEADBAND" not in n or "SYMMETRIC" in n for n in free)
        assert all(not c["hidden_dependencies"] for c in inv if c["label_free"])


class TestRolloutAndTargets:
    def test_rollout_with_hidden_matches_spec_rollout(self):
        for sc, seed, tick in (("S2", 10000, 0), ("S3", 10001, 5), ("S1", 10002, 0)):
            s = prior.replay_state(sc, seed, tick)
            for asg in list(spec._restrict(s["sim"], spec.FIXED_FAMILIES["BASE2_298"]).values())[:10]:
                a = lf.rollout_with(s["sim"], asg, s["has_hazard"], s["t_h"], lambda o: int(lf.HIDDEN(o)))
                b = spec.rollout(s["sim"], asg, s["has_hazard"], s["t_h"])
                assert lf.key4(a) == spec.key4(b)

    def test_d1_hazard_zero_without_hazard_and_onset_gated(self):
        s1 = prior.replay_state("S1", 10000, 0)
        o = lf.rollout_with(s1["sim"], {z: 0 for z in spec.nonempty(s1["sim"])}, s1["has_hazard"], s1["t_h"],
                            lf.POLICIES[M0])
        assert o["future_hazard_unmet"] == 0
        s2 = prior.replay_state("S2", 10000, 0)
        ref = spec.rollout(s2["sim"], {z: 0 for z in spec.nonempty(s2["sim"])}, True, 7)
        assert ref["check_hazard_onset_equals_tick_log"]

    def test_same_continuation_for_every_candidate(self):
        """The continuation is a function of the observation only, so two
        candidates reaching the same state get the same later actions."""
        s = prior.replay_state("S3", 10000, 0)
        asg = {z: 1 for z in spec.nonempty(s["sim"])}
        a = lf.rollout_with(s["sim"], asg, True, 5, lf.POLICIES[M0])
        b = lf.rollout_with(s["sim"], dict(asg), True, 5, lf.POLICIES[M0])
        assert a["continuation_actions"] == b["continuation_actions"] and lf.key4(a) == lf.key4(b)

    def test_margin0_matches_hidden_on_sample_episodes(self):
        configs = [c for c in lf.tick0_configs() if c["family"] in ("S2", "S3") or c["family"].startswith("timing")]
        configs = configs[::10]
        ep = lf.episode_comparison(configs)
        assert ep["per_policy"][M0]["vs_hidden_heuristic"].get("worse", 0) == 0
        assert ep["per_policy"]["CONSTANT_PARITY (baseline.py)"]["vs_hidden_heuristic"].get("worse", 0) > 0

    def test_selection_rule(self):
        per = {n: dict(vs_hidden_heuristic={"worse": (0 if n == M0 else 5)}, vs_no_intervention={},
                       max_T100_delta_vs_hidden=0.0) for n in lf.SIMPLICITY_ORDER}
        c = lf.choose_default(dict(per_policy=per))
        assert c["chosen"] == M0 and c["classification"].startswith("B.")
        per["CONSTANT_PARITY (baseline.py)"]["vs_hidden_heuristic"]["worse"] = 0
        c = lf.choose_default(dict(per_policy=per))
        assert c["chosen"] == "CONSTANT_PARITY (baseline.py)" and c["classification"].startswith("A.")


class TestDecisionRecord:
    def test_d1_d2(self):
        r = lf.decision_record(M0, "B. YES -- minimal new policy definition required")
        assert r["D1"]["status"] == "APPROVED" and "t >= t_h" in r["D1"]["decision"]
        assert r["D2"]["status"] == "DECIDED" and r["D2"]["resolved_by"] == M0
        assert any("Q_A_at_H" in x for x in r["hidden_information_removed"])


outputs_present = pytest.mark.skipif(
    not all(os.path.isfile(os.path.join(lf.OUT_DIR, n)) for n in lf.OUTPUT_FILES), reason="audit not run yet")


@outputs_present
class TestDeliverables:
    def _j(self, n):
        with open(os.path.join(lf.OUT_DIR, n), encoding="utf-8") as f:
            return json.load(f)

    def test_marked_and_parse(self):
        for n in lf.OUTPUT_FILES:
            with open(os.path.join(lf.OUT_DIR, n), encoding="utf-8") as f:
                text = f.read()
            assert lf.MARKER in text
            if n.endswith(".json"):
                json.loads(text)

    def test_recorded_choice(self):
        c = self._j("label_free_policy_candidates.json")
        assert c["selection"]["chosen"] == M0
        f = self._j("fallback_validity.json")
        assert f["label_free_verified"]["invariant_to_hidden_hub_split_on_all_existing_states"] is True
        assert f["additional_mechanism_required"] is False

    def test_frozen_spec_files_untouched(self):
        """This audit left the v1 specification untouched. Files later superseded by specification v2 are checked
        through their byte-for-byte v1 archive."""
        rec = self._j("d1_d2_decision_record.json")["integrity"]["before"]
        for n in spec.OUTPUT_FILES:
            p = os.path.join(lf.OUT_DIR, n)
            archived = os.path.join(lf.OUT_DIR, "spec_versions", "v1", n)
            with open(p, encoding="utf-8") as f:
                text = f.read()
            if "experiment_specification_v2_d1_d2" in text and os.path.isfile(archived):
                p = archived
            assert rec[f"OUT_DIR/{n}"] == prior.sha256_file(p)
            with open(p, encoding="utf-8") as f:
                assert spec.MARKER in f.read()


def test_protected_files_unchanged_after_tests(hashes_before):
    assert lf.protected_hashes() == hashes_before
