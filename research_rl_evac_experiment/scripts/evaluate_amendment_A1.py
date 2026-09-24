"""Evaluate Amendment A1 (G1-A1, X5-A1, X7-A1) exactly as written, once.

Reads every parameter from the committed amendment record
(amendments/amendment_A1_acceptance_criteria.json) and asserts it; changes
nothing in A1, the dataset or Spec v2. Read-only apart from writing the
evaluation record (amendment_A1_evaluation.json / .md). Trains nothing.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)
import ai_learning_label_free_default as lfd  # noqa: E402
import audit_pre_training_ai_dataset as prior  # noqa: E402

ART = prior.OUT_DIR
AMD = os.path.join(ART, "amendments")
A1_PATH = os.path.join(AMD, "amendment_A1_acceptance_criteria.json")
ROWS = os.path.join(ART, "full_dataset", "full_ai_learning_rows.csv.gz")
FEATS = list(lfd.ADMISSIBLE_FEATURES)
LAYOUTS = ["D0_sym", "D1_east_heavy", "D2_west_heavy", "D3_first_room_heavy", "D4_second_room_heavy",
           "D5_R1_concentrated", "Z1_R4_concentrated", "Z2_mixed"]
ZERO_SHOT = {"Z1_R4_concentrated", "Z2_mixed"}
T_H = ["none", "3", "5", "7", "9"]
P = [0.5, 0.7, 0.9]
ROLES = ["TRAIN", "VALIDATION", "TEST_PRIMARY_JOINT", "TEST_HELDOUT_TIMING", "TEST_HELDOUT_COMPLIANCE",
         "DIAG_EXTRAPOLATION", "DIAG_TEMPLATE_ZERO_SHOT"]
MARKER = "amendment_A1_evaluation_v1"
FLOAT_TOL = 1e-9


def load_a1() -> dict:
    with open(A1_PATH, encoding="utf-8") as f:
        a1 = json.load(f)
    g1, x5, x7 = (a1["amended_criteria"][k] for k in ("G1-A1", "X5-A1", "X7-A1"))
    assert g1["resampling"]["n_resamples"] == 2000 and "default_rng(5544240)" in g1["resampling"]["rng"]
    assert g1["reference_value"]["value"] == 0.5158 and x5["reference_value"]["value"] == 0.2299
    assert "B = 10000" in x7["null_distribution"] and "default_rng(5544007)" in x7["null_distribution"]
    assert a1["status"].startswith("PROPOSED")
    return a1


def protected() -> dict:
    files = ["spec_version_record.json", "diversity_requirements.json", "dataset_design.json", "split_specification.json",
             "target_definitions.json", "candidate_action_set.json", "final_feature_manifest.json",
             "full_dataset/full_ai_learning_rows.csv.gz", "full_dataset/full_generation_record.json",
             "full_dataset/full_go_no_go.json", "pilot/pilot_rows.csv.gz",
             "amendments/amendment_A1_acceptance_criteria.json", "amendments/AMENDMENT_A1_ACCEPTANCE_CRITERIA.md"]
    return {f: prior.sha256_file(os.path.join(ART, f)) for f in files}


def load_dataset():
    """Per trajectory: metadata + distinct-input ids of its decision states (all ticks and ticks >= 10)."""
    vec_id = {}
    traj = {}
    censored = 0
    last_key = None
    with gzip.open(ROWS, "rt", encoding="utf-8") as f:
        r = csv.reader(f)
        h = next(r)
        ix = {c: i for i, c in enumerate(h)}
        for row in r:
            if row[ix["T100_censored"]] == "1":
                censored += 1
            key = (row[ix["trajectory_id"]], row[ix["decision_tick"]])
            if key == last_key:
                continue
            last_key = key
            tid, tick = key[0], int(key[1])
            x = tuple(float(row[ix[c]]) for c in FEATS)
            vid = vec_id.setdefault(x, len(vec_id))
            t = traj.get(tid)
            if t is None:
                t = traj[tid] = dict(role=row[ix["split_role"]], layout=row[ix["template"]], t_h=row[ix["t_h"]],
                                     p=float(row[ix["p"]]), seed=int(row[ix["seed"]]), all=[], late=[], last_tick=-1,
                                     last_perturbed=[])
            t["all"].append(vid)
            if tick >= 10:
                t["late"].append(vid)
            if tick > t["last_tick"]:
                t["last_tick"] = tick
                t["last_perturbed"] = [v == "1" for v in row[ix["pre_decision_perturbed"]].split(",") if v != ""]
    return traj, vec_id, censored


def g1_x5(traj: dict, vec_id: dict) -> dict:
    pools = {}
    for layout in LAYOUTS:
        role = "DIAG_TEMPLATE_ZERO_SHOT" if layout in ZERO_SHOT else "TRAIN"
        for th in T_H:
            for p in P:
                pools[(layout, th, p)] = sorted(tid for tid, t in traj.items()
                                                if t["role"] == role and t["layout"] == layout and t["t_h"] == th and t["p"] == p)
    sizes = Counter(len(v) for v in pools.values())
    assert len(pools) == 120 and all(len(v) >= 2 for v in pools.values())
    rng = np.random.default_rng(5544240)
    s_late, s_all = [], []
    for _ in range(2000):
        sel = []
        for layout in LAYOUTS:
            for th in T_H:
                for p in P:
                    pool = pools[(layout, th, p)]
                    idx = rng.choice(len(pool), size=2, replace=False)
                    sel.extend(pool[i] for i in idx)
        assert len(sel) == 240 and len(set(sel)) == 240
        late = [v for tid in sel for v in traj[tid]["late"]]
        allv = [v for tid in sel for v in traj[tid]["all"]]
        s_late.append(len(set(late)) / len(late))
        s_all.append(len(set(allv)) / len(allv))

    def summary(vals, ref):
        v = np.asarray(vals)
        p = (1 + int(np.sum(v >= ref))) / (len(v) + 1)
        return dict(n=len(v), reference=ref, p_value=round(p, 4), mc_se_p=round(math.sqrt(p * (1 - p) / len(v)), 4),
                    mean=round(float(v.mean()), 4), median=round(float(np.median(v)), 4),
                    percentiles={q: round(float(np.percentile(v, q)), 4) for q in (2.5, 5, 95, 97.5)},
                    share_of_resamples_at_or_above_reference=round(float(np.mean(v >= ref)), 4))
    # retained absolute clauses (full dataset)
    per_layout_late = {l: len({v for t in traj.values() if t["layout"] == l for v in t["late"]}) for l in LAYOUTS}
    distinct_inputs = len(vec_id)
    return dict(pool_sizes=dict(sizes), g1=summary(s_late, 0.5158), x5=summary(s_all, 0.2299),
                per_layout_distinct_tick_ge10=per_layout_late, distinct_inputs=distinct_inputs,
                distinct_input_candidate_pairs=distinct_inputs * 298)


def chi2(layout_idx: np.ndarray, level_idx: np.ndarray, nl: int, nv: int) -> float:
    if nl < 2 or nv < 2:
        return 0.0
    tab = np.bincount(layout_idx * nv + level_idx, minlength=nl * nv).reshape(nl, nv).astype(float)
    n = tab.sum()
    e = np.outer(tab.sum(1), tab.sum(0)) / n
    m = e > 0
    return float((((tab - e) ** 2)[m] / e[m]).sum())


def x7(traj: dict) -> dict:
    strata = {}
    for role in ROLES:
        tids = sorted(t for t, v in traj.items() if v["role"] == role)
        layouts = sorted({traj[t]["layout"] for t in tids})
        ths = sorted({traj[t]["t_h"] for t in tids}, key=str)
        ps = sorted({traj[t]["p"] for t in tids})
        lab = np.array([layouts.index(traj[t]["layout"]) for t in tids])
        strata[role] = dict(n=len(tids), layouts=layouts, lab=lab,
                            th=np.array([ths.index(traj[t]["t_h"]) for t in tids]), n_th=len(ths),
                            p=np.array([ps.index(traj[t]["p"]) for t in tids]), n_p=len(ps))

    def T(labels_by_role, key, nkey):
        per = {r: chi2(labels_by_role[r], strata[r][key], len(strata[r]["layouts"]), strata[r][nkey]) for r in ROLES}
        return sum(per.values()), per
    obs_th, per_th = T({r: strata[r]["lab"] for r in ROLES}, "th", "n_th")
    obs_p, per_p = T({r: strata[r]["lab"] for r in ROLES}, "p", "n_p")
    rng = np.random.default_rng(5544007)
    ge_th = ge_p = 0
    B = 10000
    for _ in range(B):
        perm = {}
        for r in ROLES:
            perm[r] = rng.permutation(strata[r]["lab"])
        tb_th, _ = T(perm, "th", "n_th")
        tb_p, _ = T(perm, "p", "n_p")
        ge_th += int(tb_th >= obs_th - FLOAT_TOL)
        ge_p += int(tb_p >= obs_p - FLOAT_TOL)
    p_th = (1 + ge_th) / (B + 1)
    p_p = (1 + ge_p) / (B + 1)

    def v(stat, s, nkey):
        k = min(len(s["layouts"]), s[nkey]) - 1
        return round(math.sqrt(stat / (s["n"] * k)), 4) if k > 0 else None
    return dict(
        permutations=B, T_obs_timing=round(obs_th, 6), T_obs_compliance=round(obs_p, 6),
        p_timing=round(p_th, 4), p_compliance=round(p_p, 4),
        per_stratum={r: dict(n=strata[r]["n"], layouts=len(strata[r]["layouts"]), chi2_timing=round(per_th[r], 6),
                             chi2_compliance=round(per_p[r], 6), cramers_v_timing=v(per_th[r], strata[r], "n_th"),
                             cramers_v_compliance=v(per_p[r], strata[r], "n_p")) for r in ROLES},
        pooled_main_strata=dict(timing=round(sum(per_th[r] for r in ROLES[:6]), 6),
                                compliance=round(sum(per_p[r] for r in ROLES[:6]), 6)),
        pooled_zero_shot_stratum=dict(timing=round(per_th[ROLES[6]], 6), compliance=round(per_p[ROLES[6]], 6)),
        float_tolerance=FLOAT_TOL)


def main():
    t0 = time.perf_counter()
    a1 = load_a1()
    before = protected()
    traj, vec_id, censored = load_dataset()
    assert len(traj) == 5544
    gx = g1_x5(traj, vec_id)
    x = x7(traj)
    seeds = [t["seed"] for t in traj.values()]
    pert = [b for t in traj.values() for b in t["last_perturbed"]]
    rate = sum(pert) / len(pert)
    after = protected()
    changed = [k for k in before if before[k] != after[k]]
    if changed:
        raise RuntimeError(f"PROTECTED FILE CHANGED: {changed}")

    g1_pass_1 = gx["g1"]["p_value"] >= 0.05
    g1_pass_2 = min(gx["per_layout_distinct_tick_ge10"].values()) >= 10
    x5_pass = (gx["x5"]["p_value"] >= 0.05, gx["distinct_inputs"] >= 110, gx["distinct_input_candidate_pairs"] >= 1400)
    x7_c = dict(timing=x["p_timing"] >= 0.05, compliance=x["p_compliance"] >= 0.05, seeds_unique=len(set(seeds)) == len(seeds),
                perturbation=abs(rate - 0.5) <= 0.1, no_censored=censored == 0)
    res = {
        "G1-A1": dict(passed=bool(g1_pass_1 and g1_pass_2), clause_1=dict(passed=g1_pass_1, **gx["g1"]),
                      clause_2=dict(passed=g1_pass_2, per_layout=gx["per_layout_distinct_tick_ge10"])),
        "X5-A1": dict(passed=bool(all(x5_pass)), clause_1=dict(passed=x5_pass[0], **gx["x5"]),
                      clause_2=dict(passed=x5_pass[1], distinct_inputs=gx["distinct_inputs"]),
                      clause_3=dict(passed=x5_pass[2], pairs=gx["distinct_input_candidate_pairs"])),
        "X7-A1": dict(passed=bool(all(x7_c.values())), clauses=x7_c, stratified_test=x,
                      seeds=dict(n=len(seeds), unique=len(set(seeds))), perturbation_rate=round(rate, 4),
                      censored_rows=censored),
    }
    with open(os.path.join(ART, "full_dataset", "full_go_no_go.json"), encoding="utf-8") as f:
        orig = json.load(f)
    unchanged = {k: orig["results"][k]["passed"] for k in a1["unchanged_criteria"]}
    all_a1 = all(r["passed"] for r in res.values())
    status = ("DATASET ACCEPTED UNDER AMENDMENT A1" if all_a1 and all(unchanged.values())
              else "DATASET NOT ACCEPTED UNDER AMENDMENT A1")
    record = dict(
        audit=MARKER, amendment="A1", evaluated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        results=res, unchanged_criteria_original_results=unchanged, a1_status=status,
        provenance=dict(
            original_spec_v2_status="FULL DATASET NO-GO (unchanged, permanent)",
            original_G1_X5_X7="FAIL (unchanged, permanent)",
            nature_of_A1="post-hoc acceptance-criteria amendment, created after inspecting the frozen dataset and "
                         "before any model training; it does not retroactively change the original result",
            dataset_byte_identical=True, spec_v2_hashes_unchanged=True, a1_file_unchanged=True,
            model_results_informed_execution=False, hashes=before,
            x7_multiplicity="the two X7-A1 endpoints (layout x timing, layout x compliance) are treated as separately "
                            "prespecified acceptance tests, each at alpha = 0.05, with no family-wise-error correction",
            implementation_notes=[
                f"X7-A1 permutation comparisons T_b >= T_obs use a floating-point tolerance of {FLOAT_TOL} (a balanced "
                "table can evaluate to rounding noise instead of exactly 0)",
                "X7-A1 perturbation rate recomputed as in the original full audit: pre-decision perturbation flags of "
                "each trajectory's last recorded decision state",
                "all parameters read from and asserted against the committed A1 record; A1 not modified"]),
        counts=dict(g1_x5_resamples=gx["g1"]["n"], x7_permutations=x["permutations"],
                    seeds=dict(g1_x5=5544240, x7=5544007)),
        seconds=round(time.perf_counter() - t0, 1))
    with open(os.path.join(AMD, "amendment_A1_evaluation.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, default=str)
    print(json.dumps(dict(status=status, **{k: v["passed"] for k, v in res.items()},
                          g1=gx["g1"], x5=gx["x5"], x7=dict(T_th=x["T_obs_timing"], p_th=x["p_timing"],
                                                            T_p=x["T_obs_compliance"], p_p=x["p_compliance"]),
                          seconds=record["seconds"]), indent=1, default=str))


if __name__ == "__main__":
    main()
