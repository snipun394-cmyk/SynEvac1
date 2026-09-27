"""VALIDATION-only offline diagnostic of two selector repairs (R1, R6).

R1: step-2 floor q9_a,k <= q9_d,k + delta_k (compare like with like) instead
    of the frozen q9_a,k <= m_d,k + delta_k.
R6: step 3 keeps candidates within delta_H of the best H mean (default
    preferred; otherwise minimal H mean, ties lowest id) instead of the exact
    minimum H mean.

Variants A (frozen, reference), B (R1), C (R6), D (R1 + R6) are simulated
offline on the same hash-verified C3 predictions. Nothing here is a selector
or a protocol: no file other than this script's own output is written, the
frozen selector code is not touched, nothing is retrained or recalibrated, and
PRIMARY TEST and every other held-out role stay sealed (only split_role is
read for their rows). Frozen harm/capture definitions and the frozen
trajectory-clustered bootstrap (2,000 resamples; seeds 5544102 capture,
5544103 harm, 5544105 descriptive deltas) are reused.
Writes validation_eval/selector_r1_r6_diagnostic.json.
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

ART = prior.OUT_DIR
MDIR = os.path.join(ART, "models", "c3_hgb")
EVAL = os.path.join(ART, "validation_eval", "c3_validation_evaluation.json")
OUT = os.path.join(ART, "validation_eval", "selector_r1_r6_diagnostic.json")
T = ["future_T90", "future_T100", "future_unmet", "future_hazard_unmet"]
FEATS = list(lfd.ADMISSIBLE_FEATURES)
SEEDS = dict(recommendation=5544102, safety=5544103, descriptive_delta=5544105)


def pct(a, ps=(5, 25, 50, 75, 95)):
    a = np.asarray(a, dtype=float)
    return {str(p): float(np.percentile(a, p)) for p in ps} if a.size else {}


def auc(score, label):
    """P(score_harmful > score_not_harmful), ties counted 1/2 (Mann-Whitney)."""
    score, label = np.asarray(score, float), np.asarray(label, bool)
    npos, nneg = int(label.sum()), int((~label).sum())
    if not npos or not nneg:
        return None
    _, inv, cnt = np.unique(score, return_inverse=True, return_counts=True)
    start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    avg_rank = start + (cnt + 1) / 2.0
    r = avg_rank[inv]
    return float((r[label].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def main():
    ev = json.load(open(EVAL, encoding="utf-8"))
    before = {k: prior.sha256_file(os.path.join(ART, k)) for k in ev["integrity"]["protected_hashes"]}
    assert before == ev["integrity"]["protected_hashes"], "protected files differ from the evaluation record"
    man = json.load(open(os.path.join(MDIR, "training_manifest.json"), encoding="utf-8"))
    for mm in man["models"].values():
        assert prior.sha256_file(os.path.join(MDIR, mm["model_file"])) == mm["model_sha256"]
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
    q9 = np.sort(qr, axis=0)[2]
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
    assert int(ood.sum()) == ev["part_C_selectors"]["ood"].get("validation_ood_states", int(ood.sum()))

    ar = np.arange(nv)
    md, q9d = m[ar, d_id], q9[ar, d_id]
    nd = np.ones((nv, 298), dtype=bool)
    nd[ar, d_id] = False
    floors = {
        "old": ~(((q9 > md[:, None, :] + delta) & Kmask[:, None, :]).any(axis=2)),
        "R1": ~(((q9 > q9d[:, None, :] + delta) & Kmask[:, None, :]).any(axis=2)),
    }
    for f in floors.values():
        f[ar, d_id] = True
    key = [list(map(tuple, Y[i].tolist())) for i in range(nv)]
    dkey = [key[i][d_id[i]] for i in range(nv)]
    best = [min(key[i]) for i in range(nv)]
    improvable = np.array([best[i] < dkey[i] for i in range(nv)])
    pair_harm = np.array([[key[i][a] > dkey[i] for a in range(298)] for i in range(nv)])
    pair_better = np.array([[key[i][a] < dkey[i] for a in range(298)] for i in range(nv)])

    def select(allowed, band, tiebreak="minH"):
        out = dict(fin=np.zeros((nv, 298), bool), astar=np.empty(nv, int), sw=np.zeros(nv, bool),
                   rec=d_id.copy(), ab_ood=np.zeros(nv, bool), ab_w=np.zeros(nv, bool), firstk=np.full(nv, -1))
        for i in range(nv):
            d = d_id[i]
            K = [k for k in range(4) if Kmask[i, k]]
            cand = np.nonzero(allowed[i])[0]
            for k in K[:-1]:
                mn = m[i, cand, k].min()
                cand = cand[m[i, cand, k] <= mn + delta[k]]
            mnH = m[i, cand, 3].min()
            cand = cand[m[i, cand, 3] <= mnH + delta[3]] if band else cand[m[i, cand, 3] == mnH]
            out["fin"][i, cand] = True
            if d in cand:
                a = d
            elif band and tiebreak == "minH":
                hh = m[i, cand, 3]
                a = int(cand[hh == hh.min()].min())
            else:
                a = int(cand.min())
            out["astar"][i] = a
            r = d
            for k in K:
                diff = m[i, a, k] - m[i, d, k]
                if abs(diff) > delta[k]:
                    out["firstk"][i] = k
                    r = a if diff < -delta[k] else d
                    break
            out["sw"][i] = r != d
            if r != d and ood[i]:
                out["ab_ood"][i] = True
                r = d
            elif r != d and (q9[i, a, 1] - m[i, a, 1]) > W:
                out["ab_w"][i] = True
                r = d
            out["rec"][i] = r
        return out

    def metrics(o, allowed):
        r = o["rec"]
        dev = r != d_id
        cap = np.array([key[i][r[i]] == best[i] for i in range(nv)])
        harm = np.array([key[i][r[i]] > dkey[i] for i in range(nv)])
        sel_y, dft_y = Y[ar, r], Y[ar, d_id]
        delta_y = sel_y - dft_y                      # paired outcome delta (selected - default); negative = better
        funnel = dict(
            non_default_pairs=int(nd.sum()),
            pass_floor=dict(pairs=int((allowed & nd).sum()), states=int((allowed & nd).any(axis=1).sum())),
            survive_step3=dict(pairs=int((o["fin"] & nd).sum()), states=int((o["fin"] & nd).any(axis=1).sum())),
            a_star_non_default_states=int((o["astar"] != d_id).sum()),
            pass_switch_rule_states=int(o["sw"].sum()),
            abstain_ood_states=int(o["ab_ood"].sum()), abstain_width_states=int(o["ab_w"].sum()),
            final_switch_states=int(dev.sum()))
        why = dict(a_star_non_default_but_no_criterion_beyond_delta=int(((o["astar"] != d_id) & (o["firstk"] == -1)).sum()),
                   a_star_non_default_first_differing_criterion_worse=int(((o["astar"] != d_id) & (o["firstk"] >= 0) & ~o["sw"]).sum()),
                   switch_decided_on={T[k]: int((o["sw"] & (o["firstk"] == k)).sum()) for k in range(4)})
        dec = dict(
            non_default_recommendation_states=int(dev.sum()), exact_switches=int(dev.sum()),
            agreement_with_default=float(1 - dev.mean()),
            capture_improvable=dict(count=int(cap[improvable].sum()), of=int(improvable.sum()),
                                    rate=float(cap[improvable].mean()),
                                    ci95=EV.cluster_boot(traj, (cap & improvable).astype(float), improvable.astype(float),
                                                         SEEDS["recommendation"])),
            harm_states=int(harm.sum()), harm_rate_all_states=float(harm.mean()),
            harm_rate_all_states_ci95=EV.cluster_boot(traj, harm.astype(float), np.ones(nv), SEEDS["safety"]),
            harm_rate_among_switches=float(harm[dev].mean()) if dev.any() else None,
            harm_rate_among_switches_ci95=EV.cluster_boot(traj, (harm & dev).astype(float), dev.astype(float), SEEDS["safety"])
            if dev.any() else None,
            frozen_safety_endpoint_on_validation_descriptive=dict(
                rule="harm-rate upper 95% CI <= 1% of decisions AND no recommendation raises hazard-edge unmet by more than delta_H",
                harm_upper_ci_le_1pct=bool(EV.cluster_boot(traj, harm.astype(float), np.ones(nv), SEEDS["safety"])[1] <= 0.01),
                max_hazard_increase_vs_default=float((Y[ar, r, 3] - Y[ar, d_id, 3]).max()),
                hazard_increase_gt_delta_H=int(((Y[ar, r, 3] - Y[ar, d_id, 3]) > delta[3]).sum())),
            abstention_states=int((o["ab_ood"] | o["ab_w"]).sum()),
            abstention_rate=float((o["ab_ood"] | o["ab_w"]).mean()),
            ood_states=int(ood.sum()), ood_rate=float(ood.mean()),
            paired_outcome_delta_selected_minus_default_all_states={
                T[k]: dict(mean=float(delta_y[:, k].mean()),
                           ci95=EV.cluster_boot(traj, delta_y[:, k], np.ones(nv), SEEDS["descriptive_delta"])) for k in range(4)},
            paired_outcome_delta_among_switches={
                T[k]: dict(mean=float(delta_y[dev, k].mean()), better=int((delta_y[dev, k] < 0).sum()),
                           equal=int((delta_y[dev, k] == 0).sum()), worse=int((delta_y[dev, k] > 0).sum()),
                           ci95=EV.cluster_boot(traj, delta_y[:, k] * dev, dev.astype(float), SEEDS["descriptive_delta"]))
                for k in range(4)} if dev.any() else None)
        sw_detail = None
        if dev.any():
            idx = np.nonzero(dev)[0]
            a = r[idx]
            driver = []
            for j, i in enumerate(idx):
                if harm[i]:
                    ka, kd = key[i][a[j]], dkey[i]
                    driver.append(next(k for k in range(4) if ka[k] != kd[k]))
            sw_detail = dict(
                predicted_vs_actual={T[k]: dict(pred_mean_candidate=pct(m[idx, a, k]), pred_q9_candidate=pct(q9[idx, a, k]),
                                                pred_mean_default=pct(md[idx, k]), actual_candidate=pct(Y[idx, a, k]),
                                                actual_default=pct(Y[idx, d_id[idx], k]),
                                                q9_candidate_covers_actual=float((Y[idx, a, k] <= q9[idx, a, k]).mean()),
                                                predicted_improvement_mean=float((md[idx, k] - m[idx, a, k]).mean()),
                                                realized_improvement_mean=float((Y[idx, d_id[idx], k] - Y[idx, a, k]).mean()))
                                     for k in range(4)},
                harm_driver_first_worse_criterion={T[k]: int(sum(1 for x in driver if x == k)) for k in range(4)},
                captured_among_switches=int(cap[idx].sum()))
        return dict(funnel=funnel, why_no_switch=why, decisions=dec, switches=sw_detail), dev, harm

    variants = {"A_frozen_reference": ("old", False), "B_R1_only": ("R1", False),
                "C_R6_only": ("old", True), "D_R1_plus_R6": ("R1", True)}
    res, outs, devs = {}, {}, {}
    for name, (fl, band) in variants.items():
        o = select(floors[fl], band)
        res[name], devs[name], _ = metrics(o, floors[fl])
        outs[name] = o
    # R6 tie-break sensitivity (lowest id inside the band instead of minimal H)
    sens = {}
    for name, fl in (("C_R6_only_lowest_id", "old"), ("D_R1_plus_R6_lowest_id", "R1")):
        o = select(floors[fl], True, tiebreak="lowest_id")
        mt, _, _ = metrics(o, floors[fl])
        sens[name] = dict(final_switch_states=mt["funnel"]["final_switch_states"], harm_states=mt["decisions"]["harm_states"],
                          capture=mt["decisions"]["capture_improvable"]["count"])
    assert res["A_frozen_reference"]["decisions"]["exact_switches"] == 0, "variant A must reproduce the frozen result"

    # ---- Step 1 empirical: R1 feasibility quantities ----
    u = q9 - m
    du = u - u[ar, d_id][:, None, :]
    feas = {}
    for k in range(4):
        mask = nd & Kmask[:, None, k]
        feas[T[k]] = dict(width_difference_ua_minus_ud=pct(du[:, :, k][mask]),
                          share_pairs_width_difference_le_2delta=float((du[:, :, k][mask] <= 2 * delta[k]).mean()),
                          share_pairs_R1_demands_predicted_improvement=float((du[:, :, k][mask] - delta[k] > 0).mean()),
                          share_pairs_old_floor_demands_predicted_improvement=float((u[:, :, k][mask] - delta[k] > 0).mean()))
    default_passes_R1 = bool(floors["R1"][ar, d_id].all()) and bool(((q9d <= q9d + delta) | ~Kmask).all())

    # ---- Step 6: is the R1 floor informative? ----
    marg = {"old": np.where(Kmask[:, None, :], q9 - md[:, None, :] - delta, -np.inf).max(axis=2),
            "R1": np.where(Kmask[:, None, :], q9 - q9d[:, None, :] - delta, -np.inf).max(axis=2)}
    info = {}
    for uname, band in (("no_floor_exact_H", False), ("no_floor_band_H", True)):
        o = select(np.ones((nv, 298), bool), band)
        sw = np.nonzero(o["rec"] != d_id)[0]
        a = o["rec"][sw]
        h = pair_harm[sw, a]
        blk = {}
        for fl in ("old", "R1"):
            rej = ~floors[fl][sw, a]
            blk[fl] = dict(harmful_rejected=float(rej[h].mean()) if h.any() else None,
                           non_harmful_rejected=float(rej[~h].mean()) if (~h).any() else None,
                           n_harmful=int(h.sum()), n_non_harmful=int((~h).sum()),
                           harmful_passing=int((~rej & h).sum()), non_harmful_passing=int((~rej & ~h).sum()),
                           auc_floor_margin_vs_harm=auc(marg[fl][sw, a], h))
        info[uname] = dict(n_switches=int(len(sw)), **blk)
    pairs = {}
    for fl in ("old", "R1"):
        pm = nd
        h = pair_harm[pm]
        rej = ~floors[fl][pm]
        pairs[fl] = dict(harmful_pairs=int(h.sum()), non_harmful_pairs=int((~h).sum()),
                         harmful_rejected=float(rej[h].mean()), non_harmful_rejected=float(rej[~h].mean()),
                         pass_rate_harmful=float((~rej[h]).mean()), pass_rate_non_harmful=float((~rej[~h]).mean()),
                         auc_floor_margin_vs_harm=auc(marg[fl][pm], h),
                         passing_pairs_by_realized_outcome=dict(
                             better_than_default=int((~rej & pair_better[pm]).sum()),
                             equal_to_default=int((~rej & ~h & ~pair_better[pm]).sum()),
                             worse_than_default=int((~rej & h).sum())),
                         all_pairs_by_realized_outcome=dict(better=int(pair_better[pm].sum()),
                                                            equal=int((~h & ~pair_better[pm]).sum()), worse=int(h.sum())))
    info["all_non_default_pairs"] = pairs

    after = {k: prior.sha256_file(os.path.join(ART, k)) for k in before}
    assert after == before
    out = dict(
        record="VALIDATION-only offline diagnostic of selector repairs R1 and R6 (NOT a selector, NOT an amendment)",
        rules=dict(
            frozen_floor="keep a iff q9_a,k <= m_d,k + delta_k for all k in K; d always kept",
            R1_floor="keep a iff q9_a,k <= q9_d,k + delta_k for all k in K; d passes by construction",
            frozen_step3_H="keep exact minimum H mean",
            R6_step3_H="keep m_a,H <= min H mean + delta_H; a* = d if d survives, else minimal H mean (ties lowest id)",
            unchanged="step 1 T90 skip, step-3 T90/T100/U tolerances, step-4 switch rule, OOD tau, width threshold, deltas"),
        inputs=dict(deltas=dict(zip(("T90", "T100", "U", "H"), delta.tolist())), width_threshold=W, tau=tau,
                    predictions="regenerated from hash-verified C3 models on the same VALIDATION rows (deterministic)",
                    bootstrap="frozen cluster_boot (evaluate_c3_validation.py): trajectory-clustered, 2,000 resamples",
                    seeds=SEEDS, rows_parsed=rows, other_roles_seen_only_as_split_role=skipped),
        step1_R1_feasibility=dict(default_passes_R1_floor_in_every_state=default_passes_R1, per_criterion=feas),
        variants=res, r6_tiebreak_sensitivity=sens,
        step6_floor_informativeness=info,
        integrity=dict(protected_files_unchanged=True, n_protected=len(before), selector_code_unchanged=True,
                       no_retraining=True, no_recalibration=True, primary_test_accessed=False))
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=float)
    print("written", OUT)


if __name__ == "__main__":
    main()
