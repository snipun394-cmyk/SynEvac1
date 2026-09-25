"""Issue 2 validation-baseline fitting and selection (frozen protocol A2 + C1 + C2).

Fits the four frozen non-learning baselines on TRAIN, scores them on
VALIDATION with the frozen C2 selection score, selects one, freezes its
conditional-mean MAE tolerances, and computes the frozen C2 OOD threshold.

Reads only TRAIN and VALIDATION rows: for every other row only the
split_role field is inspected (the file is one gzip stream, so those bytes
are decompressed, but no other field of them is accessed). No learned model
is trained; no PRIMARY TEST statistic is computed; nothing is written except
amendments/issue2_validation_baseline_selection.json.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
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
ROWS = os.path.join(ART, "full_dataset", "full_ai_learning_rows.csv.gz")
OUT = os.path.join(AMD, "issue2_validation_baseline_selection.json")
FEATS = list(lfd.ADMISSIBLE_FEATURES)
EXCLUDED = ["in_transit_R2_C1", "in_transit_R3_C2", "in_transit_R4_C2"]
DIST_FEATS = [f for f in FEATS if f not in EXCLUDED]
TARGETS = ["future_T90", "future_T100", "future_unmet", "future_hazard_unmet"]
QS = [0.1, 0.5, 0.9]
BASELINE_ORDER = ["mean_by_candidate", "mean_by_candidate_x_tick", "exact_input_lookup", "one_nn"]
ALLOWED_ROLES = {"TRAIN", "VALIDATION"}


class AmbiguityStop(RuntimeError):
    pass


def protected() -> dict:
    with open(os.path.join(ART, "spec_version_record.json"), encoding="utf-8") as f:
        files = list(json.load(f)["files"])
    files += ["spec_version_record.json", "full_dataset/full_ai_learning_rows.csv.gz",
              "full_dataset/full_generation_record.json", "full_dataset/full_go_no_go.json", "pilot/pilot_rows.csv.gz"]
    files += [f"amendments/{n}" for n in (
        "amendment_A1_acceptance_criteria.json", "amendment_A1_evaluation.json", "amendment_A2_evaluation_protocol.json",
        "AMENDMENT_A2_EVALUATION_PROTOCOL.md", "amendment_A2_approval.json", "amendment_A2_reference_run.json",
        "amendment_A2_clarification_C1.json", "amendment_A2_clarification_C2.json")]
    return {f: prior.sha256_file(os.path.join(ART, f)) for f in files}


def load():
    """Per state (TRAIN/VALIDATION only): 26 features, tick, Y (298 x 4)."""
    states = {"TRAIN": [], "VALIDATION": []}
    role_rows = Counter()
    skipped = Counter()
    cur, buf, meta = None, [], None
    with gzip.open(ROWS, "rt", encoding="utf-8") as f:
        r = csv.reader(f)
        h = next(r)
        ix = {c: i for i, c in enumerate(h)}
        irole = ix["split_role"]

        def flush():
            if not buf:
                return
            role, tid, tick, x = meta
            if len(buf) != 298 or [int(b[ix["candidate_id"]]) for b in buf] != list(range(298)):
                raise AmbiguityStop(f"state {tid}/{tick} does not have exactly candidates 0..297")
            Y = np.array([[float(b[ix[t]]) for t in TARGETS] for b in buf])
            states[role].append(dict(traj=tid, tick=tick, x=x, Y=Y))

        for row in r:
            role = row[irole]
            if role not in ALLOWED_ROLES:
                skipped[role] += 1
                continue
            role_rows[role] += 1
            key = (row[ix["trajectory_id"]], row[ix["decision_tick"]])
            if key != cur:
                flush()
                buf = []
                cur = key
                meta = (role, key[0], int(key[1]), tuple(float(row[ix[c]]) for c in FEATS))
            buf.append(row)
        flush()
    return states, dict(role_rows), dict(skipped)


def pinball(y, yhat, q):
    d = y - yhat
    return np.maximum(q * d, (q - 1) * d)


def group_stats(Ystack: np.ndarray) -> tuple:
    """Ystack: (n_states, 298, 4) -> conditional mean (298, 4) and quantiles (3, 298, 4), NumPy linear."""
    return Ystack.mean(axis=0), np.quantile(Ystack, QS, axis=0, method="linear")


def main():
    t0 = time.perf_counter()
    before = protected()
    states, role_rows, skipped = load()
    tr, va = states["TRAIN"], states["VALIDATION"]
    Ytr = np.stack([s["Y"] for s in tr])
    train_ticks = sorted({s["tick"] for s in tr})
    val_ticks = sorted({s["tick"] for s in va})
    missing_ticks = sorted(set(val_ticks) - set(train_ticks))
    if missing_ticks:
        raise AmbiguityStop(f"VALIDATION ticks {missing_ticks} never occur in TRAIN: candidate x tick is undefined there")

    # ---- fit baselines (TRAIN only) ----
    b_cand = group_stats(Ytr)
    by_tick = defaultdict(list)
    for i, s in enumerate(tr):
        by_tick[s["tick"]].append(i)
    b_tick = {t: group_stats(Ytr[idx]) for t, idx in by_tick.items()}
    by_vec = defaultdict(list)
    for i, s in enumerate(tr):
        by_vec[s["x"]].append(i)
    b_exact = {v: group_stats(Ytr[idx]) for v, idx in by_vec.items()}
    dcols = [FEATS.index(f) for f in DIST_FEATS]
    Xtr = np.array([[s["x"][j] for j in dcols] for s in tr])
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0, ddof=0)
    if np.any(sd == 0):
        raise AmbiguityStop("a retained 1-NN feature has zero TRAIN variance")
    const_check = {f: float(np.array([s["x"][FEATS.index(f)] for s in tr]).std(ddof=0)) for f in EXCLUDED}
    uvecs = list(by_vec)
    U = (np.array([[v[j] for j in dcols] for v in uvecs]) - mu) / sd

    def predict(name, s):
        if name == "mean_by_candidate":
            return b_cand, "direct"
        if name == "mean_by_candidate_x_tick":
            return b_tick[s["tick"]], "direct"
        if name == "exact_input_lookup":
            return (b_exact[s["x"]], "exact") if s["x"] in b_exact else (b_tick[s["tick"]], "fallback")
        z = (np.array([s["x"][j] for j in dcols]) - mu) / sd
        d = np.sqrt(((U - z) ** 2).sum(axis=1))
        dmin = d.min()
        tied = [uvecs[k] for k in np.nonzero(d == dmin)[0]]
        idx = [i for v in tied for i in by_vec[v]]
        return group_stats(Ytr[idx]), f"nn_{len(tied)}"

    # ---- score on VALIDATION ----
    Yva = np.stack([s["Y"] for s in va])                     # (n_val, 298, 4)
    scale = {t: float(Yva[:, :, TARGETS.index(t)].std(ddof=0)) for t in ("future_T100", "future_unmet")}
    results, mae_cache, usage = {}, {}, {}
    for name in BASELINE_ORDER:
        means, quants, how = [], [], Counter()
        for s in va:
            (m, q), h = predict(name, s)
            means.append(m)
            quants.append(q)
            how[h if not h.startswith("nn_") else "nn_ties_" + ("1" if h == "nn_1" else ">1")] += 1
        M = np.stack(means)                                   # (n_val, 298, 4)
        Qp = np.stack(quants)                                 # (n_val, 3, 298, 4)
        loss = {}
        for t in ("future_T100", "future_unmet"):
            k = TARGETS.index(t)
            y = Yva[:, :, k]
            loss[t] = float(np.mean([pinball(y, Qp[:, qi, :, k], q).mean() for qi, q in enumerate(QS)]))
        norm = {t: loss[t] / scale[t] for t in loss}
        score = (norm["future_T100"] + norm["future_unmet"]) / 2
        results[name] = dict(raw_pinball_loss=loss, normalized_loss=norm, selection_score=score)
        mae_cache[name] = {t: float(np.abs(Yva[:, :, TARGETS.index(t)] - M[:, :, TARGETS.index(t)]).mean()) for t in TARGETS}
        usage[name] = dict(how)
    ranked = sorted(BASELINE_ORDER, key=lambda n: (round(results[n]["selection_score"], 12), BASELINE_ORDER.index(n)))
    selected = ranked[0]
    mae = mae_cache[selected]
    deltas = dict(delta_U=0.5 * mae["future_unmet"], delta_H=0.5 * mae["future_hazard_unmet"], delta_T90=0.5, delta_T100=0.5)

    # ---- C2 OOD threshold (TRAIN only; leave-one-out over TRAIN decision states) ----
    counts = np.array([len(by_vec[v]) for v in uvecs])
    loo_unique = np.empty(len(uvecs))
    for k in range(len(uvecs)):
        if counts[k] >= 2:
            loo_unique[k] = 0.0
        else:
            d = np.sqrt(((U - U[k]) ** 2).sum(axis=1))
            d[k] = np.inf
            loo_unique[k] = d.min()
    loo_states = np.repeat(loo_unique, counts)
    tau = float(np.quantile(loo_states, 0.99, method="linear"))

    after = protected()
    changed = [k for k in before if before[k] != after[k]]
    if changed:
        raise RuntimeError(f"PROTECTED FILE CHANGED: {changed}")
    wl = json.load(open(os.path.join(AMD, "amendment_A2_evaluation_protocol.json"), encoding="utf-8"))
    whitelist = wl["issues"]["6_model_input_whitelist"]["whitelist_state_features"]
    record = dict(
        record="Issue 2 validation-baseline fitting and selection (A2 + C1 + C2)",
        run_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        protocol=dict(A2="ab9aeb7a6e5c9349b4017a1fa51cbebc9809afda9f55c52e49a7a768be0b299a",
                      C1="795f55560ef7f18e61f91759c910eac7ab04641b7d92cda5e4385f9ca647405c",
                      C2="49a42a6db8a70ea1f6830e1f0f8135bc8ee43549afaf3e58a02e634543afb2e5"),
        data=dict(dataset_sha256=before["full_dataset/full_ai_learning_rows.csv.gz"],
                  train_states=len(tr), validation_states=len(va), train_rows=role_rows.get("TRAIN", 0),
                  validation_rows=role_rows.get("VALIDATION", 0), train_ticks=train_ticks, validation_ticks=val_ticks,
                  other_roles_seen_only_as_split_role=skipped),
        validation_target_scales=dict(scale, note="population SD (ddof=0) over all VALIDATION rows"),
        baselines={n: dict(results[n], prediction_route_counts=usage[n]) for n in BASELINE_ORDER},
        ranking=ranked, selected_baseline=selected,
        selected_baseline_validation_MAE_conditional_mean=mae, tolerances=deltas,
        one_nn=dict(distance_features=DIST_FEATS, n_distance_features=len(DIST_FEATS), excluded_zero_variance=EXCLUDED,
                    excluded_TRAIN_sd=const_check, train_mean={f: float(m) for f, m in zip(DIST_FEATS, mu)},
                    train_sd={f: float(v) for f, v in zip(DIST_FEATS, sd)}, n_unique_train_inputs=len(uvecs),
                    scaling="population mean/SD over TRAIN decision states (one point per state)"),
        ood=dict(threshold_tau=tau, rule="abstain when min distance to any TRAIN state > tau",
                 loo_zero_share=float(np.mean(loo_states == 0)), n_train_states=int(len(loo_states)),
                 quantile="numpy.quantile(q=0.99, method='linear')"),
        integrity=dict(
            primary_test_or_other_rows_parsed_beyond_split_role=0,
            roles_parsed=sorted(role_rows), statistics_sources="fitting and scaling: TRAIN; scoring, target scales and MAE: VALIDATION",
            learned_model_trained=False, candidates_per_state=298,
            model_input_whitelist_unchanged=bool(whitelist == FEATS),
            protected_files_unchanged=True, protected_hashes=before),
        seconds=round(time.perf_counter() - t0, 1))
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, default=float)
    print(json.dumps({k: v for k, v in record.items() if k not in ("one_nn",)} | dict(
        one_nn_summary=dict(n=len(DIST_FEATS), excluded=EXCLUDED, unique_train_inputs=len(uvecs))),
        indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
