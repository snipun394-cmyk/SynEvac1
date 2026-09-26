"""VALIDATION-only evaluation of the 16 fitted C3 models and of the learned vs
exact-input-baseline selectors (A2 + C1 + C2 + C3 + Issue 2 baseline record).

Reads only TRAIN (baseline groups, OOD reference) and VALIDATION rows; every
other role is seen only through its split_role field. Models are loaded,
hash-verified against the training manifest, and used for prediction only.
Metrics use raw predictions; the selector consumes row-sorted quantiles (C3).
Writes validation_eval/c3_validation_evaluation.json and a per-state CSV.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import pickle
import sys
import time

import numpy as np

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)
import ai_learning_label_free_default as lfd  # noqa: E402
import audit_pre_training_ai_dataset as prior  # noqa: E402
import run_issue2_baseline_selection as I2  # noqa: E402

ART = prior.OUT_DIR
AMD = os.path.join(ART, "amendments")
MDIR = os.path.join(ART, "models", "c3_hgb")
OUT_DIR = os.path.join(ART, "validation_eval")
TARGETS = ["future_T90", "future_T100", "future_unmet", "future_hazard_unmet"]
QS = [0.1, 0.5, 0.9]
FEATS = list(lfd.ADMISSIBLE_FEATURES)
SEEDS = dict(recommendation=5544102, safety=5544103, prediction=5544104)
N_BOOT = 2000


def sha_file(p):
    return prior.sha256_file(p)


def protected() -> dict:
    files = ["full_dataset/full_ai_learning_rows.csv.gz", "spec_version_record.json", "candidate_action_set.json",
             "final_feature_manifest.json", "selector_design.json", "preregistered_success_criteria.json",
             "models/c3_hgb/training_manifest.json"]
    files += [f"amendments/{n}" for n in (
        "amendment_A1_acceptance_criteria.json", "amendment_A1_evaluation.json", "amendment_A2_evaluation_protocol.json",
        "AMENDMENT_A2_EVALUATION_PROTOCOL.md", "amendment_A2_approval.json", "amendment_A2_reference_run.json",
        "amendment_A2_clarification_C1.json", "amendment_A2_clarification_C2.json",
        "amendment_A2_clarification_C3_model_specification.json", "issue2_validation_baseline_selection.json")]
    out = {f: sha_file(os.path.join(ART, f)) for f in files}
    for n in sorted(os.listdir(MDIR)):
        if n.endswith(".pkl"):
            out[f"models/c3_hgb/{n}"] = sha_file(os.path.join(MDIR, n))
    return out


def pinball(y, yhat, q):
    d = y - yhat
    return np.maximum(q * d, (q - 1) * d)


def cluster_boot(traj_ids, per_unit_num, per_unit_den, seed):
    """Ratio statistic sum(num)/sum(den) with whole trajectories resampled."""
    uniq, inv = np.unique(traj_ids, return_inverse=True)
    num = np.bincount(inv, weights=per_unit_num, minlength=len(uniq))
    den = np.bincount(inv, weights=per_unit_den, minlength=len(uniq))
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(uniq), size=len(uniq))
        d = den[idx].sum()
        stats.append(num[idx].sum() / d if d > 0 else np.nan)
    stats = np.array(stats)
    return [float(np.quantile(stats[~np.isnan(stats)], 0.025, method="linear")),
            float(np.quantile(stats[~np.isnan(stats)], 0.975, method="linear"))]


def main():
    t0 = time.perf_counter()
    before = protected()
    man = json.load(open(os.path.join(MDIR, "training_manifest.json"), encoding="utf-8"))
    for name, m in man["models"].items():
        if before[f"models/c3_hgb/{m['model_file']}"] != m["model_sha256"]:
            raise RuntimeError(f"model file {name} differs from the training manifest")
    order = man["training_data"]["input_columns"]
    rec_i2 = json.load(open(os.path.join(AMD, "issue2_validation_baseline_selection.json"), encoding="utf-8"))
    tau = rec_i2["ood"]["threshold_tau"]
    base_tol = rec_i2["tolerances"]
    mu = np.array([rec_i2["one_nn"]["train_mean"][f] for f in I2.DIST_FEATS])
    sd = np.array([rec_i2["one_nn"]["train_sd"][f] for f in I2.DIST_FEATS])

    states, role_rows, skipped = I2.load()
    tr, va = states["TRAIN"], states["VALIDATION"]
    nv = len(va)
    Yva = np.stack([s["Y"] for s in va])                        # (nv, 298, 4)
    traj = np.array([s["traj"] for s in va])

    # ---- 54-column input, VALIDATION ----
    cas = json.load(open(os.path.join(ART, "candidate_action_set.json"), encoding="utf-8"))
    zones, values = ["R1", "R2", "C1", "R3", "R4", "C2", "H"], ["UNTOUCHED", "FAVOR_A", "PARITY", "FAVOR_B"]
    assert order[26:] == [f"cand_{z}_{v}" for z in zones for v in values] and order[:26] == FEATS
    desc = np.zeros((298, 28))
    for c in cas["candidates"]:
        for zi, z in enumerate(zones):
            desc[c["id"], zi * 4 + values.index(c["encoding"][z])] = 1.0
    Xs = np.array([s["x"] for s in va], dtype=np.float64)          # (nv, 26)
    X = np.concatenate([np.repeat(Xs, 298, axis=0), np.tile(desc, (nv, 1))], axis=1)
    assert X.shape == (nv * 298, 54) and np.isfinite(X).all()

    # ---- learned predictions (raw) ----
    mean_l = np.zeros((nv, 298, 4))
    q_raw = np.zeros((3, nv, 298, 4))
    for k, t in enumerate(TARGETS):
        for name, slot in (("mean", None), ("q0.1", 0), ("q0.5", 1), ("q0.9", 2)):
            with open(os.path.join(MDIR, f"{t}__{name}.pkl"), "rb") as fh:
                est = pickle.loads(fh.read())
            p = est.predict(X).reshape(nv, 298)
            if slot is None:
                mean_l[:, :, k] = p
            else:
                q_raw[slot, :, :, k] = p
    q_sorted = np.sort(q_raw, axis=0)
    finite_ok = bool(np.isfinite(mean_l).all() and np.isfinite(q_raw).all())

    # ---- exact-input baseline (TRAIN groups; NumPy linear quantiles) ----
    Ytr = np.stack([s["Y"] for s in tr])
    by_vec, by_tick = {}, {}
    for i, s in enumerate(tr):
        by_vec.setdefault(s["x"], []).append(i)
        by_tick.setdefault(s["tick"], []).append(i)
    gstat = {}

    def stats(key, idx):
        if key not in gstat:
            Ys = Ytr[idx]
            gstat[key] = (Ys.mean(axis=0), np.quantile(Ys, QS, axis=0, method="linear"))
        return gstat[key]
    mean_b = np.zeros((nv, 298, 4))
    q_b = np.zeros((3, nv, 298, 4))
    b_route = {"exact": 0, "fallback": 0}
    for i, s in enumerate(va):
        if s["x"] in by_vec:
            m, q = stats(("v", s["x"]), by_vec[s["x"]])
            b_route["exact"] += 1
        else:
            m, q = stats(("t", s["tick"]), by_tick[s["tick"]])
            b_route["fallback"] += 1
        mean_b[i] = m
        q_b[:, i] = q

    # ---- Part A/B/E: prediction metrics ----
    def pred_metrics(mean, qr, qs):
        out = {}
        for k, t in enumerate(TARGETS):
            y = Yva[:, :, k]
            pl = {str(q): float(pinball(y, qr[qi, :, :, k], q).mean()) for qi, q in enumerate(QS)}
            m = mean[:, :, k]
            err = m - y
            neg = {}
            for lab, arr in (("mean", m), ("q0.1", qr[0, :, :, k]), ("q0.5", qr[1, :, :, k]), ("q0.9", qr[2, :, :, k])):
                vals, counts = np.unique(np.round(arr, 12), return_counts=True)
                neg[lab] = dict(share_negative=float((arr < 0).mean()), min=float(arr.min()), max=float(arr.max()),
                                n_distinct_values=int(len(vals)), share_modal_value=float(counts.max() / arr.size))
            cross = ((qr[0, :, :, k] > qr[1, :, :, k]) | (qr[1, :, :, k] > qr[2, :, :, k]))
            changed_state = cross.any(axis=1)
            width_raw = qr[2, :, :, k] - qr[0, :, :, k]
            width_sorted = qs[2, :, :, k] - qs[0, :, :, k]
            out[t] = dict(
                pinball=pl, pinball_mean=float(np.mean(list(pl.values()))),
                mae_conditional_mean=float(np.abs(err).mean()), rmse_conditional_mean=float(np.sqrt((err ** 2).mean())),
                bias_mean_minus_target=float(err.mean()), predictions=neg,
                quantile_crossing=dict(row_rate_raw=float(cross.mean()), state_rate_any_row=float(changed_state.mean())),
                coverage_q01_q09=dict(raw=float(((y >= qr[0, :, :, k]) & (y <= qr[2, :, :, k])).mean()),
                                      sorted=float(((y >= qs[0, :, :, k]) & (y <= qs[2, :, :, k])).mean()), nominal=0.8),
                upper_coverage_q09=dict(raw=float((y <= qr[2, :, :, k]).mean()), sorted=float((y <= qs[2, :, :, k]).mean()),
                                        nominal=0.9),
                interval_width=dict(raw={p: float(np.percentile(width_raw, p)) for p in (0, 5, 25, 50, 75, 95, 100)},
                                    sorted_mean=float(width_sorted.mean()), share_zero_width_sorted=float((width_sorted == 0).mean())))
        return out
    pm_learned = pred_metrics(mean_l, q_raw, q_sorted)
    pm_base = pred_metrics(mean_b, q_b, q_b)

    # paired prediction-loss differences (row-weighted, trajectory-clustered; seed 5544104)
    paired_pred = {}
    for k, t in enumerate(("future_T100", "future_unmet")):
        kk = TARGETS.index(t)
        y = Yva[:, :, kk]
        ll = np.mean([pinball(y, q_raw[qi, :, :, kk], q) for qi, q in enumerate(QS)], axis=0).sum(axis=1)
        lb = np.mean([pinball(y, q_b[qi, :, :, kk], q) for qi, q in enumerate(QS)], axis=0).sum(axis=1)
        diff = ll - lb
        paired_pred[t] = dict(learned=float(ll.sum() / (nv * 298)), baseline=float(lb.sum() / (nv * 298)),
                              difference_learned_minus_baseline=float(diff.sum() / (nv * 298)),
                              ci95=cluster_boot(traj, diff, np.full(nv, 298.0), SEEDS["prediction"]))

    # ---- Part C: selectors ----
    iRem = FEATS.index("normalized_remaining")
    default_act = np.array([lfd.label_free_default_action(np.array(s["x"])) for s in va])
    name_to_id = {c["name"]: c["id"] for c in cas["candidates"]}
    default_id = np.array([name_to_id[{0: "BASE=A", 1: "BASE=P", 2: "BASE=B"}[a]] for a in default_act])
    Zv = (Xs[:, [FEATS.index(f) for f in I2.DIST_FEATS]] - mu) / sd
    U = (np.array([[v[FEATS.index(f)] for f in I2.DIST_FEATS] for v in by_vec]) - mu) / sd
    dmin = np.array([np.sqrt(((U - z) ** 2).sum(axis=1)).min() for z in Zv])
    ood = dmin > tau

    def run_selector(mean, qsel, deltas, width_thr):
        rec = np.empty(nv, dtype=int)
        info = []
        for i in range(nv):
            d = default_id[i]
            skip90 = round(Xs[i, iRem] * 40) <= 4
            K = ([0] if not skip90 else []) + [1, 2, 3]
            dl = [deltas[k] for k in K]
            m, qh = mean[i], qsel[2, i]
            ok = np.ones(298, dtype=bool)
            for k, dk in zip(K, dl):
                ok &= qh[:, k] <= m[d, k] + dk
            ok[d] = True
            cand = np.nonzero(ok)[0]
            for k, dk in zip(K[:-1], dl[:-1]):
                mn = m[cand, k].min()
                cand = cand[m[cand, k] <= mn + dk]
            mnH = m[cand, 3].min()
            cand = cand[m[cand, 3] == mnH]
            a_star = d if d in cand else int(cand.min())
            r = d
            for k, dk in zip(K, dl):
                diff = m[a_star, k] - m[d, k]
                if abs(diff) > dk:
                    r = a_star if diff < -dk else d
                    break
            width = qsel[2, i, a_star, 1] - m[a_star, 1]
            abst_ood = bool(ood[i] and r != d)
            abst_width = bool((not ood[i]) and r != d and width > width_thr)
            if abst_ood or abst_width:
                r = d
            rec[i] = r
            info.append(dict(a_star=int(a_star), safe_set=int(ok.sum()), skip_t90=bool(skip90),
                             abstain_ood=abst_ood, abstain_width=abst_width, width=float(width)))
        return rec, info

    mae_l = {t: pm_learned[t]["mae_conditional_mean"] for t in TARGETS}
    d_learned = [0.5, 0.5, 0.5 * mae_l["future_unmet"], 0.5 * mae_l["future_hazard_unmet"]]
    d_base = [base_tol["delta_T90"], base_tol["delta_T100"], base_tol["delta_U"], base_tol["delta_H"]]
    wthr_l = float(np.quantile((q_sorted[2, :, :, 1] - mean_l[:, :, 1]).ravel(), 0.95, method="linear"))
    wthr_b = float(np.quantile((q_b[2, :, :, 1] - mean_b[:, :, 1]).ravel(), 0.95, method="linear"))
    rec_l, info_l = run_selector(mean_l, q_sorted, d_learned, wthr_l)
    rec_b, info_b = run_selector(mean_b, q_b, d_base, wthr_b)

    # ---- Part D: decision metrics (evaluation-side, exact key equality per A2) ----
    keys = Yva                                                   # realized (T90, T100, U, H)
    def key_of(i, a):
        return tuple(keys[i, a].tolist())
    best = [min(map(tuple, keys[i].tolist())) for i in range(nv)]
    dkey = [key_of(i, default_id[i]) for i in range(nv)]
    improvable = np.array([best[i] < dkey[i] for i in range(nv)])
    def dec_metrics(rec, info):
        cap = np.array([key_of(i, rec[i]) == best[i] for i in range(nv)])
        harm = np.array([key_of(i, rec[i]) > dkey[i] for i in range(nv)])
        dev = rec != default_id
        sel = keys[np.arange(nv), rec]
        dft = keys[np.arange(nv), default_id]
        return dict(
            n_states=nv, n_improvable=int(improvable.sum()),
            capture_rate_improvable=float(cap[improvable].mean()),
            capture_rate_all_states=float(cap.mean()),
            harm_rate=float(harm.mean()), harm_count=int(harm.sum()),
            harm_ci95=cluster_boot(traj, harm.astype(float), np.ones(nv), SEEDS["safety"]),
            deviation_rate=float(dev.mean()), harm_rate_among_deviations=float(harm[dev].mean()) if dev.any() else None,
            abstention_rate=float(np.mean([x["abstain_ood"] or x["abstain_width"] for x in info])),
            abstain_ood=int(sum(x["abstain_ood"] for x in info)), abstain_width=int(sum(x["abstain_width"] for x in info)),
            mean_outcome_selected={t: float(sel[:, k].mean()) for k, t in enumerate(TARGETS)},
            mean_outcome_default={t: float(dft[:, k].mean()) for k, t in enumerate(TARGETS)},
            mean_gain_vs_default={t: float((dft[:, k] - sel[:, k]).mean()) for k, t in enumerate(TARGETS)},
        ), cap, harm
    dm_l, cap_l, harm_l = dec_metrics(rec_l, info_l)
    dm_b, cap_b, harm_b = dec_metrics(rec_b, info_b)
    rand_ref = float(np.mean([np.mean([tuple(r) == best[i] for r in keys[i].tolist()]) for i in np.nonzero(improvable)[0]]))
    cap_diff = (cap_l.astype(float) - cap_b.astype(float)) * improvable
    comparative = dict(
        learned_capture=dm_l["capture_rate_improvable"], baseline_capture=dm_b["capture_rate_improvable"],
        difference=float(cap_diff.sum() / improvable.sum()),
        ci95=cluster_boot(traj, cap_diff, improvable.astype(float), SEEDS["recommendation"]),
        note="A2 comparative criterion computed on VALIDATION descriptively; the confirmatory test is on PRIMARY TEST")
    agree = float((rec_l == rec_b).mean())

    after = protected()
    changed = [k for k in before if before[k] != after[k]]
    if changed:
        raise RuntimeError(f"PROTECTED FILE CHANGED: {changed}")

    os.makedirs(OUT_DIR, exist_ok=True)
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["trajectory_id", "decision_tick", "default_id", "learned_rec", "baseline_rec", "learned_a_star", "baseline_a_star",
                "ood", "ood_min_distance", "learned_abstain_ood", "learned_abstain_width", "baseline_abstain_ood",
                "baseline_abstain_width", "agree", "improvable", "oracle_best_key", "learned_actual_key", "baseline_actual_key",
                "learned_pred_mean_selected", "learned_captured", "baseline_captured", "learned_harm", "baseline_harm"])
    for i, s in enumerate(va):
        w.writerow([s["traj"], s["tick"], default_id[i], rec_l[i], rec_b[i], info_l[i]["a_star"], info_b[i]["a_star"],
                    int(ood[i]), round(float(dmin[i]), 6), int(info_l[i]["abstain_ood"]), int(info_l[i]["abstain_width"]),
                    int(info_b[i]["abstain_ood"]), int(info_b[i]["abstain_width"]), int(rec_l[i] == rec_b[i]),
                    int(improvable[i]), "|".join(map(str, best[i])), "|".join(map(str, key_of(i, rec_l[i]))),
                    "|".join(map(str, key_of(i, rec_b[i]))), "|".join(f"{v:.4f}" for v in mean_l[i, rec_l[i]]),
                    int(cap_l[i]), int(cap_b[i]), int(harm_l[i]), int(harm_b[i])])
    csv_bytes = buf.getvalue().encode()
    with gzip.GzipFile(os.path.join(OUT_DIR, "c3_validation_per_state.csv.gz"), "wb", mtime=0) as fh:
        fh.write(csv_bytes)
    record = dict(
        record="VALIDATION-only evaluation of the fitted C3 models and the learned vs exact-input-baseline selectors",
        run_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        scope=dict(rows_fully_parsed=role_rows, other_roles_seen_only_as_split_role=skipped,
                   validation_states=nv, validation_rows=nv * 298),
        part_A_prediction_learned=pm_learned, part_B_prediction_baseline=pm_base,
        part_B_paired_prediction_loss=paired_pred,
        finite_predictions=finite_ok, baseline_routes=b_route,
        part_C_selectors=dict(
            learned=dict(deltas=dict(T90=d_learned[0], T100=d_learned[1], U=d_learned[2], H=d_learned[3]),
                         width_threshold_T100=wthr_l, quantiles="row-sorted (C3) for selector consumption"),
            baseline=dict(deltas=dict(T90=d_base[0], T100=d_base[1], U=d_base[2], H=d_base[3]), width_threshold_T100=wthr_b),
            ood=dict(tau=tau, share_states_ood=float(ood.mean()), n_ood=int(ood.sum()),
                     min_distance_percentiles={p: float(np.percentile(dmin, p)) for p in (50, 90, 99, 100)}),
            implementation_notes=[
                "step-5 interval width uses the row-sorted q0.9 (C3: sorting is for selector consumption)",
                "abstention is counted when an OOD or width condition overrides a non-default recommendation",
                "T90 skipped when round(normalized_remaining x 40) <= 4 (A2 Issue 5)",
                "the final hazard-edge step keeps the exact minimum; remaining ties -> d(s) if present else lowest id"]),
        part_D_decisions=dict(learned=dm_l, baseline=dm_b, learned_vs_baseline_agreement=agree,
                              comparative_capture=comparative, default_capture_improvable=0.0,
                              random_candidate_capture_improvable=rand_ref,
                              oracle_equivalence="exact equality of the realized 4-target key (A2 evaluation-side definition)"),
        integrity=dict(model_hashes_match_manifest=True, protected_files_unchanged=True, protected_hashes=before,
                       seeds=SEEDS, primary_test_and_other_roles_sealed=True, no_refit=True, raw_predictions_unmodified=True),
        per_state_csv_sha256=hashlib.sha256(csv_bytes).hexdigest(),
        seconds=round(time.perf_counter() - t0, 1))
    with open(os.path.join(OUT_DIR, "c3_validation_evaluation.json"), "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, default=float)
    print(json.dumps(dict(finite=finite_ok, learned_deltas=record["part_C_selectors"]["learned"], ood=record["part_C_selectors"]["ood"],
                          decisions=record["part_D_decisions"], paired_pred=paired_pred), indent=1, default=float))


if __name__ == "__main__":
    main()
