"""VALIDATION-only diagnostic: does the existing C3 predictor contain a usable
safety margin under R1?

Offline simulation on the same hash-verified C3 predictions as
diagnose_selector_r1_r6.py. The R1 selector (variant B there) is the base:
step 2 = q9_a,k <= q9_d,k + delta_k; all other frozen steps unchanged. Nothing
here is a selector, protocol or threshold choice. No file other than this
script's output is written; nothing is retrained or recalibrated; PRIMARY
TEST and other held-out roles stay sealed (only split_role is read).

Fixed before results were inspected:
  * margin grid g in {0, 0.5, 1, 1.5, 2, 3, 4} x delta_k (for T90/T100 this is
    exactly 0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0 ticks);
  * integer rule: predicted means rounded half up, q0.9 upper bounds read as
    ceil(q9 - 1e-6), for T90/T100 only.
Frozen bootstrap (cluster_boot, 2,000 resamples): 5544102 capture, 5544103
harm, 5544105 descriptive deltas.
Writes validation_eval/selector_margin_diagnostic.json.
"""
from __future__ import annotations

import json
import os
import pickle
import sys

import numpy as np

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)
import ai_learning_label_free_default as lfd  # noqa: E402
import audit_pre_training_ai_dataset as prior  # noqa: E402
import evaluate_c3_validation as EV  # noqa: E402
import run_issue2_baseline_selection as I2  # noqa: E402
from diagnose_selector_r1_r6 import auc, pct  # noqa: E402

ART = prior.OUT_DIR
MDIR = os.path.join(ART, "models", "c3_hgb")
EVAL = os.path.join(ART, "validation_eval", "c3_validation_evaluation.json")
OUT = os.path.join(ART, "validation_eval", "selector_margin_diagnostic.json")
T = ["future_T90", "future_T100", "future_unmet", "future_hazard_unmet"]
FEATS = list(lfd.ADMISSIBLE_FEATURES)
SEEDS = dict(recommendation=5544102, safety=5544103, descriptive_delta=5544105)
GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]          # multiples of delta_k (fixed in advance)
TICK_BUCKETS = [(-np.inf, 0.0), (0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, np.inf)]   # (lo, hi]
EPS_INT = 1e-6


def main():
    ev = json.load(open(EVAL, encoding="utf-8"))
    before = {k: prior.sha256_file(os.path.join(ART, k)) for k in ev["integrity"]["protected_hashes"]}
    assert before == ev["integrity"]["protected_hashes"], "protected files differ from the evaluation record"
    man = json.load(open(os.path.join(MDIR, "training_manifest.json"), encoding="utf-8"))
    model_hashes = {}
    for name, mm in man["models"].items():
        h = prior.sha256_file(os.path.join(MDIR, mm["model_file"]))
        assert h == mm["model_sha256"]
        model_hashes[name] = h
    states, rows, skipped = I2.load()
    tr, va = states["TRAIN"], states["VALIDATION"]
    nv = len(va)
    traj = np.array([s["traj"] for s in va])
    Y = np.stack([s["Y"] for s in va])
    cas = json.load(open(os.path.join(ART, "candidate_action_set.json"), encoding="utf-8"))
    zones, values = ["R1", "R2", "C1", "R3", "R4", "C2", "H"], ["UNTOUCHED", "FAVOR_A", "PARITY", "FAVOR_B"]
    desc = np.zeros((298, 28))
    for c in cas["candidates"]:
        for zi, z in enumerate(zones):
            desc[c["id"], zi * 4 + values.index(c["encoding"][z])] = 1.0
    Xs = np.array([s["x"] for s in va], dtype=np.float64)
    X = np.concatenate([np.repeat(Xs, 298, axis=0), np.tile(desc, (nv, 1))], axis=1)
    m = np.zeros((nv, 298, 4))
    qr = np.zeros((3, nv, 298, 4))
    for k, t in enumerate(T):
        for name, slot in (("mean", None), ("q0.1", 0), ("q0.5", 1), ("q0.9", 2)):
            est = pickle.loads(open(os.path.join(MDIR, f"{t}__{name}.pkl"), "rb").read())
            p = est.predict(X).reshape(nv, 298)
            if slot is None:
                m[:, :, k] = p
            else:
                qr[slot, :, :, k] = p
    qs = np.sort(qr, axis=0)
    q1, q9 = qs[0], qs[2]
    sel = ev["part_C_selectors"]
    delta = np.array([sel["learned"]["deltas"][k] for k in ("T90", "T100", "U", "H")])
    W = sel["learned"]["width_threshold_T100"]
    tau = sel["ood"]["tau"]
    name_to_id = {c["name"]: c["id"] for c in cas["candidates"]}
    d_id = np.array([name_to_id[{0: "BASE=A", 1: "BASE=P", 2: "BASE=B"}[lfd.label_free_default_action(np.array(s["x"]))]]
                     for s in va])
    iRem = FEATS.index("normalized_remaining")
    Kmask = np.ones((nv, 4), dtype=bool)
    Kmask[:, 0] = np.array([round(x * 40) > 4 for x in Xs[:, iRem]])
    rec = json.load(open(os.path.join(ART, "amendments", "issue2_validation_baseline_selection.json"), encoding="utf-8"))
    mu = np.array([rec["one_nn"]["train_mean"][f] for f in I2.DIST_FEATS])
    sd = np.array([rec["one_nn"]["train_sd"][f] for f in I2.DIST_FEATS])
    trvecs = sorted({s["x"] for s in tr})
    U = (np.array([[v[FEATS.index(f)] for f in I2.DIST_FEATS] for v in trvecs]) - mu) / sd
    Zv = (Xs[:, [FEATS.index(f) for f in I2.DIST_FEATS]] - mu) / sd
    ood = np.array([np.sqrt(((U - z) ** 2).sum(axis=1)).min() > tau for z in Zv])
    ar = np.arange(nv)
    key = [list(map(tuple, Y[i].tolist())) for i in range(nv)]
    dkey = [key[i][d_id[i]] for i in range(nv)]
    best = [min(key[i]) for i in range(nv)]
    improvable = np.array([best[i] < dkey[i] for i in range(nv)])

    def run(mm, qq9, width_src_m, width_src_q9):
        """R1 selector on (possibly transformed) mean/q9. Returns a*, first k, raw switch, final rec."""
        md, q9d = mm[ar, d_id], qq9[ar, d_id]
        allowed = ~(((qq9 > q9d[:, None, :] + delta) & Kmask[:, None, :]).any(axis=2))
        allowed[ar, d_id] = True
        astar = np.empty(nv, int)
        firstk = np.full(nv, -1)
        sw = np.zeros(nv, bool)
        recm = d_id.copy()
        for i in range(nv):
            d = d_id[i]
            K = [k for k in range(4) if Kmask[i, k]]
            cand = np.nonzero(allowed[i])[0]
            for k in K[:-1]:
                mn = mm[i, cand, k].min()
                cand = cand[mm[i, cand, k] <= mn + delta[k]]
            cand = cand[mm[i, cand, 3] == mm[i, cand, 3].min()]
            a = d if d in cand else int(cand.min())
            astar[i] = a
            r = d
            for k in K:
                diff = mm[i, a, k] - mm[i, d, k]
                if abs(diff) > delta[k]:
                    firstk[i] = k
                    r = a if diff < -delta[k] else d
                    break
            sw[i] = r != d
            if r != d and (ood[i] or (width_src_q9[i, a, 1] - width_src_m[i, a, 1]) > W):
                r = d
            recm[i] = r
        return astar, firstk, sw, recm

    astar, firstk, sw_raw, rec_r1 = run(m, q9, m, q9)
    dev = rec_r1 != d_id
    assert int(dev.sum()) == 540, "R1 base must reproduce the 540 switches of the previous diagnostic"

    def decisions(r):
        dv = r != d_id
        cap = np.array([key[i][r[i]] == best[i] for i in range(nv)])
        harm = np.array([key[i][r[i]] > dkey[i] for i in range(nv)])
        dy = Y[ar, r] - Y[ar, d_id]
        out = dict(
            switches=int(dv.sum()),
            capture_improvable=dict(count=int(cap[improvable].sum()), rate=float(cap[improvable].mean()),
                                    ci95=EV.cluster_boot(traj, (cap & improvable).astype(float), improvable.astype(float),
                                                         SEEDS["recommendation"])),
            harmful=int(harm.sum()), harm_rate_all_states=float(harm.mean()),
            harm_rate_all_states_ci95=EV.cluster_boot(traj, harm.astype(float), np.ones(nv), SEEDS["safety"]),
            harm_rate_among_switches=float(harm[dv].mean()) if dv.any() else None,
            meets_frozen_1pct_upper_ci=None,
            paired_delta_selected_minus_default_all_states={
                T[k]: dict(mean=float(dy[:, k].mean()),
                           ci95=EV.cluster_boot(traj, dy[:, k], np.ones(nv), SEEDS["descriptive_delta"])) for k in range(4)})
        out["meets_frozen_1pct_upper_ci"] = bool(out["harm_rate_all_states_ci95"][1] <= 0.01)
        return out, harm, cap

    base, harm, cap = decisions(rec_r1)
    a = astar
    md = m[ar, d_id]
    I = md - m[ar, a]                                  # predicted improvement (positive = candidate better)
    Iq9 = q9[ar, d_id] - q9[ar, a]                      # q0.9 improvement
    width = q9[ar, a] - q1[ar, a]                       # candidate 80% interval width
    uwidth = q9[ar, a] - m[ar, a]                       # candidate upper width
    LB = md - q9[ar, a]                                 # improvement if candidate lands at its q0.9
    ratio = I / np.maximum(width, 1e-9)
    R = Y[ar, d_id] - Y[ar, a]                          # realized improvement of a* (positive = better)
    valid = Kmask.copy()                                # T90 not a criterion in late states
    harmful_sw = dev & harm
    harmless_sw = dev & ~harm
    first_worse = np.full(nv, -1)
    for i in np.nonzero(harmful_sw)[0]:
        first_worse[i] = next(k for k in range(4) if key[i][rec_r1[i]][k] != dkey[i][k])
    t90_harm = harmful_sw & (first_worse == 0)
    assert int(t90_harm.sum()) == 126

    # ---- 1. margin distributions ----
    groups = dict(all_a_star=np.ones(nv, bool), switches=dev, harmful_switches=harmful_sw, harmless_switches=harmless_sw)
    step1 = {}
    for g, gm in groups.items():
        step1[g] = dict(n=int(gm.sum()), per_target={T[k]: dict(
            predicted_improvement=pct(I[gm & valid[:, k], k]), predicted_q9_improvement=pct(Iq9[gm & valid[:, k], k]),
            interval_width_q9_minus_q1=pct(width[gm & valid[:, k], k]), upper_width_q9_minus_mean=pct(uwidth[gm & valid[:, k], k]),
            realized_improvement=pct(R[gm & valid[:, k], k]),
            realized_improvement_mean=float(R[gm & valid[:, k], k].mean()) if (gm & valid[:, k]).any() else None)
            for k in range(4)})
    # prediction granularity among the R1 switches
    sw_idx = np.nonzero(dev)[0]
    sig = [tuple(np.round(np.concatenate([I[i], Iq9[i]]), 6)) for i in sw_idx]
    groups_ = {}
    for i, s_ in zip(sw_idx, sig):
        groups_.setdefault(s_, []).append(i)
    step1["prediction_granularity_among_switches"] = dict(
        distinct_predicted_margin_vectors=len(groups_),
        per_group=[dict(n=len(v), harmful=int(harm[v].sum()), captured=int(cap[v].sum()),
                        a_star_ids=sorted({int(astar[i]) for i in v}), default_ids=sorted({int(d_id[i]) for i in v}),
                        ticks=sorted({int(va[i]["tick"]) for i in v}))
                   for v in sorted(groups_.values(), key=len, reverse=True)],
        distinct_trajectories=int(len(np.unique(traj[dev]))),
        switch_ticks=dict(zip(*[x.tolist() for x in np.unique([va[i]["tick"] for i in sw_idx], return_counts=True)])))
    step1["first_switch_criterion_among_switches"] = {T[k]: int((dev & (firstk == k)).sum()) for k in range(4)}
    step1["first_worse_criterion_among_harmful"] = {T[k]: int((first_worse == k).sum()) for k in range(4)}

    # ---- 2. 0.5-tick buckets ----
    def bucket_table(mask, k):
        rows_ = []
        for lo, hi in TICK_BUCKETS:
            b = mask & valid[:, k] & (I[:, k] > lo) & (I[:, k] <= hi)
            dy = -R                                          # selected - default (positive = worse)
            row = dict(bucket=f"({lo}, {hi}]", n=int(b.sum()), harmful=int((b & harm).sum()),
                       harm_rate=float(harm[b].mean()) if b.any() else None,
                       harm_rate_ci95=EV.cluster_boot(traj, (b & harm).astype(float), b.astype(float), SEEDS["safety"]) if b.any() else None)
            for kk, nm in ((0, "T90"), (1, "T100"), (2, "unmet"), (3, "hazard")):
                row[f"actual_{nm}_diff_mean"] = float(dy[b, kk].mean()) if b.any() else None
                row[f"actual_{nm}_diff_ci95"] = EV.cluster_boot(traj, dy[:, kk] * b, b.astype(float), SEEDS["descriptive_delta"]) if b.any() else None
            rows_.append(row)
        rows_.append(dict(t90_not_a_criterion=int((mask & ~valid[:, k]).sum())))
        return rows_

    def int_hist(vals):
        u, c = np.unique(vals, return_counts=True)
        return {str(int(x)) if float(x).is_integer() else str(x): int(n) for x, n in zip(u, c)}
    step2 = dict(
        switches_by_predicted_T90_improvement=bucket_table(dev, 0),
        switches_by_predicted_T100_improvement=bucket_table(dev, 1),
        t90_harmful_by_predicted_T90_improvement=bucket_table(t90_harm, 0),
        t90_harmful_by_predicted_T100_improvement=bucket_table(t90_harm, 1),
        t90_harmful_actual_T90_diff_histogram=int_hist(-R[t90_harm, 0]),
        t90_harmful_predicted_T90_diff_candidate_minus_default=pct(-I[t90_harm, 0]),
        switches_actual_T90_diff_histogram=int_hist(-R[dev & valid[:, 0], 0]),
        switches_actual_T100_diff_histogram=int_hist(-R[dev, 1]),
        switches_predicted_abs_T90_diff=pct(np.abs(I[dev & valid[:, 0], 0])),
        share_switches_predicted_abs_T90_diff_below_0_5=float((np.abs(I[dev & valid[:, 0], 0]) <= 0.5).mean()),
        share_switches_actual_T90_diff_nonzero=float((R[dev & valid[:, 0], 0] != 0).mean()))

    # ---- 3. margin vs harm / capture separation ----
    def cohen(a_, b_):
        if len(a_) < 2 or len(b_) < 2:
            return None
        s = np.sqrt(((len(a_) - 1) * a_.var(ddof=1) + (len(b_) - 1) * b_.var(ddof=1)) / (len(a_) + len(b_) - 2))
        return float((a_.mean() - b_.mean()) / s) if s > 0 else None
    margins = dict(predicted_improvement=I, q9_candidate_minus_q9_default=-Iq9, interval_width=width,
                   improvement_over_width=ratio, lower_margin_default_mean_minus_candidate_q9=LB)
    uncaptured_improvable = improvable & ~cap
    captured = cap & improvable
    step3 = {}
    for mn, arr in margins.items():
        step3[mn] = {}
        for k in range(4):
            v = valid[:, k]
            h, hl = arr[harmful_sw & v, k], arr[harmless_sw & v, k]
            c_, uc = arr[captured & v, k], arr[uncaptured_improvable & v, k]
            lab = harm[dev & v]
            step3[mn][T[k]] = dict(
                harmful_median=float(np.median(h)) if len(h) else None, harmless_median=float(np.median(hl)) if len(hl) else None,
                captured_median=float(np.median(c_)) if len(c_) else None,
                uncaptured_improvable_median=float(np.median(uc)) if len(uc) else None,
                cohen_d_harmful_minus_harmless=cohen(h, hl),
                auc_margin_predicts_harm_among_switches=auc(arr[dev & v, k], lab),
                cohen_d_captured_minus_uncaptured_improvable=cohen(c_, uc))
    # best single-margin separation (for the key question; no threshold chosen)
    best_sep = max(((abs(x["auc_margin_predicts_harm_among_switches"] - 0.5) + 0.5, mn, t)
                    for mn, d_ in step3.items() for t, x in d_.items() if x["auc_margin_predicts_harm_among_switches"] is not None))
    step3["strongest_single_margin_auc_oriented"] = dict(auc=best_sep[0], margin=best_sep[1], target=best_sep[2])

    # ---- 4. fixed grid on the first switch criterion; 5. per-target non-inferiority; 4b all non-switch criteria ----
    def rule_rec(keep):
        r = rec_r1.copy()
        r[dev & ~keep] = d_id[dev & ~keep]
        return r
    fk = np.where(firstk >= 0, firstk, 0)
    step4, step5, step4b = {}, {}, {}
    for g in GRID:
        mu_k = g * delta
        keep = I[ar, fk] > delta[fk] + mu_k[fk]
        step4[f"g={g}"] = dict(margins_native=mu_k.tolist(), **decisions(rule_rec(keep))[0])
        keep_all = np.ones(nv, bool)
        for k in range(4):
            keep_all &= ~(valid[:, k] & (k != firstk)) | (I[:, k] >= mu_k[k])
        step4b[f"g={g}"] = dict(margins_native=mu_k.tolist(), **decisions(rule_rec(keep_all))[0])
    for k in range(4):
        step5[T[k]] = {}
        for g in GRID:
            mu_ = g * delta[k]
            keep = ~valid[:, k] | (I[:, k] >= mu_)
            d_, h_, _ = decisions(rule_rec(keep))
            removed = dev & ~keep
            step5[T[k]][f"g={g}"] = dict(margin_native=mu_, **d_,
                                         removed_switches=int(removed.sum()), removed_harmful=int((removed & harm).sum()),
                                         removed_t90_harmful=int((removed & t90_harm).sum()),
                                         removed_captured=int((removed & cap).sum()))

    # ---- 6. integer-aware diagnostic ----
    mI, q9I = m.copy(), q9.copy()
    mI[:, :, :2] = np.floor(m[:, :, :2] + 0.5)
    q9I[:, :, :2] = np.ceil(q9[:, :, :2] - EPS_INT)
    edge = {}
    for k, nm in ((0, "T90"), (1, "T100")):
        frac = q9[:, :, k] - np.floor(q9[:, :, k])
        near = frac > 1 - 1e-4
        ya = Y[:, :, k]
        edge[nm] = dict(q9_within_1em4_below_integer_all_rows=int(near.sum()),
                        of_rows=int(near.size),
                        rows_where_actual_equals_that_integer=int((near & (ya == np.ceil(q9[:, :, k]))).sum()),
                        coverage_P_y_le_q9_raw=float((ya <= q9[:, :, k]).mean()),
                        coverage_P_y_le_ceil_q9_minus_eps=float((ya <= np.ceil(q9[:, :, k] - EPS_INT)).mean()),
                        r1_switch_candidates_q9_near_integer=int(near[ar, a][dev].sum()))
    step6 = dict(rule="T90/T100 means rounded half up; T90/T100 q0.9 read as ceil(q9 - 1e-6); unmet/hazard unchanged; "
                      "width abstention still uses the continuous T100 width",
                 edge_cases=edge)
    for vname, (mm, qq) in dict(integer_steps_2_to_4=(mI, q9I), integer_step4_only=(None, None)).items():
        if vname == "integer_step4_only":
            # steps 2-3 continuous (R1), step-4 comparison on rounded T90/T100 means
            r = d_id.copy()
            for i in range(nv):
                aa, d = astar[i], d_id[i]
                K = [k for k in range(4) if Kmask[i, k]]
                rr = d
                for k in K:
                    diff = (mI[i, aa, k] - mI[i, d, k]) if k < 2 else (m[i, aa, k] - m[i, d, k])
                    if abs(diff) > delta[k]:
                        rr = aa if diff < -delta[k] else d
                        break
                if rr != d and (ood[i] or (q9[i, aa, 1] - m[i, aa, 1]) > W):
                    rr = d
                r[i] = rr
        else:
            _, _, _, r = run(mm, qq, m, q9)
        d_, h_, c_ = decisions(r)
        dv = r != d_id
        step6[vname] = dict(**d_,
                            t90_harm_cases_remaining_harmful=int((t90_harm & h_).sum()),
                            t90_harm_cases_no_longer_switching=int((t90_harm & ~dv).sum()),
                            t90_harm_cases_switch_but_not_harmful=int((t90_harm & dv & ~h_).sum()),
                            r1_switches_retained_same_candidate=int((dev & (r == rec_r1)).sum()),
                            r1_captures_retained=int((dev & improvable & cap & c_ & (r == rec_r1)).sum()),
                            new_switch_states_not_in_r1=int((dv & ~dev).sum()))

    # ---- 7. is <= 1% reachable within the fixed diagnostics? ----
    cands = []
    for fam, dct in (("first_criterion_margin", step4), ("all_non_switch_criteria_noninferiority", step4b)):
        for gname, x in dct.items():
            cands.append((fam, gname, x))
    for t, dct in step5.items():
        for gname, x in dct.items():
            cands.append((f"per_target_{t}", gname, x))
    for vname in ("integer_steps_2_to_4", "integer_step4_only"):
        cands.append((vname, "-", step6[vname]))
    summary = [dict(rule=f, setting=g, switches=x["switches"], captured=x["capture_improvable"]["count"], harmful=x["harmful"],
                    harm_rate=x["harm_rate_all_states"], harm_upper_ci=x["harm_rate_all_states_ci95"][1],
                    harm_among_switches=x["harm_rate_among_switches"], meets_1pct=x["meets_frozen_1pct_upper_ci"])
               for f, g, x in cands]
    nonzero = [s for s in summary if s["switches"] > 0]
    step7 = dict(
        all_fixed_rules=summary,
        any_rule_meets_1pct_with_nonzero_switches=any(s["meets_1pct"] for s in nonzero),
        lowest_harm_upper_ci_with_nonzero_switches=min(nonzero, key=lambda s: s["harm_upper_ci"]) if nonzero else None,
        lowest_harm_among_switches_with_nonzero_switches=min(nonzero, key=lambda s: s["harm_among_switches"]) if nonzero else None,
        note="max harmful states compatible with upper CI <= 1% depends on clustering; at n = 1,845 states a point rate of "
             "1% is 18 states, and the upper bound needs materially fewer")

    after = {k: prior.sha256_file(os.path.join(ART, k)) for k in before}
    assert after == before
    out = dict(
        record="VALIDATION-only margin diagnostic on the R1 base (NOT a selector, NOT a threshold choice, NOT an amendment)",
        fixed_in_advance=dict(grid_multiples_of_delta=GRID, tick_buckets="(lo, hi] on predicted improvement m_d - m_a",
                              integer_rule="round half up for means; ceil(q9 - 1e-6) for q0.9; T90/T100 only"),
        inputs=dict(deltas=dict(zip(("T90", "T100", "U", "H"), delta.tolist())), width_threshold=W, tau=tau,
                    base="R1 selector (variant B of selector_r1_r6_diagnostic.json): 540 switches reproduced",
                    bootstrap="frozen cluster_boot, 2,000 resamples", seeds=SEEDS, rows_parsed=rows,
                    other_roles_seen_only_as_split_role=skipped),
        r1_base=base,
        step1_margin_distributions=step1, step2_tick_buckets=step2, step3_margin_separation=step3,
        step4_first_criterion_margin_grid=step4, step4b_all_non_switch_criteria_noninferiority_grid=step4b,
        step5_per_target_noninferiority_grid=step5, step6_integer_diagnostic=step6, step7_reachability=step7,
        integrity=dict(protected_files_unchanged=True, protected_hashes=before, model_hashes_verified=model_hashes,
                       selector_code_unchanged=True, no_retraining=True, no_recalibration=True, primary_test_accessed=False))
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=float)
    print("written", OUT)


if __name__ == "__main__":
    main()
