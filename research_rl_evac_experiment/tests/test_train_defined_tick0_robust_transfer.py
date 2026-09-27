"""Tests for the read-only TRAIN-defined tick-0 robust-transfer audit (synthetic data only)."""
from __future__ import annotations

import csv
import gzip
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import audit_train_defined_tick0_robust_transfer as A  # noqa: E402


def _Y(rows):
    """rows: per member, per candidate (T90, T100, U, H)."""
    return np.array(rows, dtype=float)


class TestRobustSet:
    def test_lexicographic_not_delta(self):
        # candidate 1 is 1 tick better on T90 but much worse on hazard: lexicographically better
        Y = _Y([[[10, 10, 5, 0], [9, 10, 5, 50]]])
        rs = A.robust_set(Y, 0)
        assert rs["strict"] == [1]

    def test_single_harmed_member_excludes(self):
        Y = _Y([[[10, 10, 5, 5], [9, 10, 5, 5]],
                [[10, 10, 5, 5], [11, 10, 5, 0]]])
        assert A.robust_set(Y, 0)["strict"] == []

    def test_ties_everywhere_are_weak_not_strict(self):
        Y = _Y([[[10, 10, 5, 5], [10, 10, 5, 5]]])
        rs = A.robust_set(Y, 0)
        assert rs["weak"] == [1] and rs["strict"] == [] and rs["outcome_identical_to_default"] == 1

    def test_default_never_in_sets(self):
        Y = _Y([[[10, 10, 5, 5], [9, 10, 5, 5], [10, 10, 5, 4]]])
        rs = A.robust_set(Y, 0)
        assert 0 not in rs["weak"] and rs["strict"] == [1, 2]

    def test_patterns_group_identical_outcomes(self):
        Y = _Y([[[10, 10, 5, 5], [9, 10, 5, 5], [9, 10, 5, 5], [10, 10, 5, 4]]])
        assert A.robust_set(Y, 0)["strict_patterns"] == [[1, 2], [3]]


class TestOutcomeBlock:
    def test_counts_and_worst(self):
        Yv = _Y([[[10, 10, 5, 5], [9, 10, 5, 5]],
                 [[10, 10, 5, 5], [10, 10, 5, 5]],
                 [[10, 10, 5, 5], [11, 9, 0, 0]]])
        b = A.outcome_block(Yv, [0, 0, 0], 1)
        assert (b["improved"], b["harmed"], b["tied"]) == (1, 1, 1)
        assert b["worst_observed_key"] == [11.0, 9.0, 0.0, 0.0] and not b["zero_harm"]
        assert b["mean_delta_selected_minus_default"]["future_T90"] == pytest.approx(0.0)


class TestClopperPearson:
    def test_zero_events_closed_form(self):
        for n in (10, 90, 540):
            assert A.cp_upper(0, n) == pytest.approx(1 - 0.05 ** (1 / n), abs=1e-9)

    def test_monotone_and_bounds(self):
        assert A.cp_upper(0, 0) is None
        assert A.cp_upper(5, 5) == 1.0
        assert A.cp_upper(1, 100) > A.cp_upper(0, 100)
        assert A.cp_upper(1, 200) < A.cp_upper(1, 100)

    def test_known_value(self):
        # one-sided 95% upper bound for 1/100 is about 0.0466
        assert A.cp_upper(1, 100) == pytest.approx(0.0466, abs=5e-4)

    def test_rule_of_three_scale(self):
        assert A.cp_upper(0, 299) < 0.01 < A.cp_upper(0, 298)


class TestRoleRestrictedLoader:
    def _write(self, path, rows):
        header = (["trajectory_id", "split_role", "decision_tick", "candidate_id"] + A.HIDDEN + A.FEATS + A.T)
        with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            for r in rows:
                w.writerow([r.get(c, "0") for c in header])

    def test_reads_only_requested_role_and_tick0(self, tmp_path):
        p = str(tmp_path / "rows.csv.gz")
        rows = []
        for role, tid in (("TRAIN", "t1"), ("VALIDATION", "v1"), ("TEST_PRIMARY_JOINT", "p1")):
            for tick in ("0", "5"):
                for cid in range(3):
                    rows.append({"trajectory_id": tid, "split_role": role, "decision_tick": tick, "candidate_id": str(cid),
                                 "future_T90": str(cid), "t_h": "3", "p": "0.5"})
        self._write(p, rows)
        st, other = A.load_tick0(p, "TRAIN", n_cand=3)
        assert [s["traj"] for s in st] == ["t1"] and st[0]["Y"][:, 0].tolist() == [0.0, 1.0, 2.0]
        assert other == {"VALIDATION": 6, "TEST_PRIMARY_JOINT": 6}

    def test_primary_role_values_never_parsed(self, tmp_path):
        p = str(tmp_path / "rows.csv.gz")
        rows = [{"trajectory_id": "p1", "split_role": "TEST_PRIMARY_JOINT", "decision_tick": "0", "candidate_id": "0",
                 "future_T90": "not-a-number"}]
        rows += [{"trajectory_id": "t1", "split_role": "TRAIN", "decision_tick": "0", "candidate_id": str(c)} for c in range(2)]
        self._write(p, rows)
        st, other = A.load_tick0(p, "TRAIN", n_cand=2)   # would raise ValueError if the PRIMARY row were parsed
        assert len(st) == 1 and other == {"TEST_PRIMARY_JOINT": 1}

    def test_missing_candidate_raises(self, tmp_path):
        p = str(tmp_path / "rows.csv.gz")
        rows = [{"trajectory_id": "t1", "split_role": "TRAIN", "decision_tick": "0", "candidate_id": "0"}]
        self._write(p, rows)
        with pytest.raises(A.IntegrityError):
            A.load_tick0(p, "TRAIN", n_cand=2)


def test_math_import_used():
    assert math.isfinite(A.binom_cdf(3, 10, 0.2))
