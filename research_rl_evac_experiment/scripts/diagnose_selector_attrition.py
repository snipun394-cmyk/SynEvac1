"""DIAGNOSTIC ONLY (VALIDATION): where do the frozen selector steps remove
candidates? Explains a zero-deviation result without changing the selector.

Recomputes the same learned and exact-input-baseline predictions as
evaluate_c3_validation.py (same verified models, same TRAIN/VALIDATION rows),
then, for each selector, measures step-by-step attrition. The 'without step 2'
numbers are a counterfactual description of the mechanism, not a protocol and
not a candidate selector. Writes validation_eval/selector_attrition_diagnostic.json.
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
import run_issue2_baseline_selection as I2  # noqa: E402

ART = prior.OUT_DIR
MDIR = os.path.join(ART, "models", "c3_hgb")
EVAL = os.path.join(ART, "validation_eval", "c3_validation_evaluation.json")
TARGETS = ["future_T90", "future_T100", "future_unmet", "future_hazard_unmet"]
QS = [0.1, 0.5, 0.9]
FEATS = list(lfd.ADMISSIBLE_FEATURES)


def main():
    ev = json.load(open(EVAL, encoding="utf-8"))
    man = json.load(open(os.path.join(MDIR, "training_manifest.json"), encoding="utf-8"))
    for m in man["models"].values():
        assert prior.sha256_file(os.path.join(MDIR, m["model_file"])) == m["model_sha256"]
    states, _, _ = I2.load()
    tr, va = states["TRAIN"], states["VALIDATION"]
    nv = len(va)
    Yva = np.stack([s["Y"] for s in va])
    cas = json.load(open(os.path.join(ART, "candidate_action_set.json"), encoding="utf-8"))
    zones, values = ["R1", "R2", "C1", "R3", "R4", "C2", "H"], ["UNTOUCHED", "FAVOR_A", "PARITY", "FAVOR_B"]
    desc = np.zeros((298, 28))
    for c in cas["candidates"]:
        for zi, z in enumerate(zones):
            desc[c["id"], zi * 4 + values.index(c["encoding"][z])] = 1.0
    Xs = np.array([s["x"] for s in va], dtype=np.float64)
    X = np.concatenate([np.repeat(Xs, 298, axis=0), np.tile(desc, (nv, 1))], axis=1)
    mean_l, q_raw = np.zeros((nv, 298, 4)), np.zeros((3, nv, 298, 4))
    for k, t in enumerate(TARGETS):
        for name, slot in (("mean", None), ("q0.1", 0), ("q0.5", 1), ("q0.9", 2)):
            est = pickle.loads(open(os.path.join(MDIR, f"{t}__{name}.pkl"), "rb").read())
            p = est.predict(X).reshape(nv, 298)
            if slot is None:
                mean_l[:, :, k] = p
            else:
                q_raw[slot, :, :, k] = p
    q_l = np.sort(q_raw, axis=0)
    Ytr = np.stack([s["Y"] for s in tr])
    by_vec, by_tick, cache = {}, {}, {}
    for i, s in enumerate(tr):
        by_vec.setdefault(s["x"], []).append(i)
        by_tick.setdefault(s["tick"], []).append(i)
    mean_b, q_b = np.zeros((nv, 298, 4)), np.zeros((3, nv, 298, 4))
    for i, s in enumerate(va):
        key, idx = (("v", s["x"]), by_vec[s["x"]]) if s["x"] in by_vec else (("t", s["tick"]), by_tick[s["tick"]])
        if key not in cache:
            cache[key] = (Ytr[idx].mean(axis=0), np.quantile(Ytr[idx], QS, axis=0, method="linear"))
        mean_b[i], q_b[:, i] = cache[key]
    name_to_id = {c["name"]: c["id"] for c in cas["candidates"]}
    default_id = np.array([name_to_id[{0: "BASE=A", 1: "BASE=P", 2: "BASE=B"}[lfd.label_free_default_action(np.array(s["x"]))]]
                           for s in va])
    iRem = FEATS.index("normalized_remaining")
    best = [min(map(tuple, Yva[i].tolist())) for i in range(nv)]
    improvable = np.array([best[i] < tuple(Yva[i, default_id[i]].tolist()) for i in range(nv)])
    sel = ev["part_C_selectors"]

    def attrition(mean, qs, deltas):
        c = dict(states=nv, safe_nondefault_any=0, safe_nondefault_count=[], astar_nondefault=0, switch_passed=0,
                 block_by_criterion=[0, 0, 0, 0], nondefault_rows=0,
                 no_floor_astar_nondefault=0, no_floor_switch=0, no_floor_capture_improvable=0, no_floor_harm=0)
        for i in range(nv):
            d = default_id[i]
            K = ([0] if round(Xs[i, iRem] * 40) > 4 else []) + [1, 2, 3]
            m, qh = mean[i], qs[2, i]
            ok = np.ones(298, dtype=bool)
            nd = np.arange(298) != d
            for k in K:
                fail = qh[:, k] > m[d, k] + deltas[k]
                c["block_by_criterion"][k] += int((fail & nd).sum())
                ok &= ~fail
            c["nondefault_rows"] += int(nd.sum())
            ok[d] = True
            c["safe_nondefault_count"].append(int(ok.sum() - 1))
            c["safe_nondefault_any"] += int(ok.sum() > 1)

            def steps(cand):
                cand = np.nonzero(cand)[0]
                for k in K[:-1]:
                    mn = m[cand, k].min()
                    cand = cand[m[cand, k] <= mn + deltas[k]]
                cand = cand[m[cand, 3] == m[cand, 3].min()]
                a = d if d in cand else int(cand.min())
                r = d
                for k in K:
                    diff = m[a, k] - m[d, k]
                    if abs(diff) > deltas[k]:
                        r = a if diff < -deltas[k] else d
                        break
                return a, r
            a, r = steps(ok)
            c["astar_nondefault"] += int(a != d)
            c["switch_passed"] += int(r != d)
            a2, r2 = steps(np.ones(298, dtype=bool))
            c["no_floor_astar_nondefault"] += int(a2 != d)
            c["no_floor_switch"] += int(r2 != d)
            realized = tuple(Yva[i, r2].tolist())
            c["no_floor_capture_improvable"] += int(improvable[i] and realized == best[i])
            c["no_floor_harm"] += int(realized > tuple(Yva[i, d].tolist()))
        cnt = np.array(c.pop("safe_nondefault_count"))
        c["safe_nondefault_count_percentiles"] = {p: float(np.percentile(cnt, p)) for p in (50, 90, 99, 100)}
        c["block_share_by_criterion"] = {TARGETS[k]: c["block_by_criterion"][k] / c["nondefault_rows"] for k in range(4)}
        c["share_states_safe_nondefault_any"] = c["safe_nondefault_any"] / nv
        c["counterfactual_note"] = "no_floor_* = the same selector with step 2 skipped; diagnostic of the mechanism only"
        c["no_floor_capture_rate_improvable"] = c["no_floor_capture_improvable"] / int(improvable.sum())
        c["no_floor_harm_rate"] = c["no_floor_harm"] / nv
        # typical gap: best non-default q0.9 minus default mean, per criterion
        return c
    dl = [sel["learned"]["deltas"][k] for k in ("T90", "T100", "U", "H")]
    db = [sel["baseline"]["deltas"][k] for k in ("T90", "T100", "U", "H")]
    out = dict(record="DIAGNOSTIC ONLY: selector step attrition on VALIDATION (no protocol change)",
               learned=attrition(mean_l, q_l, dl), baseline=attrition(mean_b, q_b, db),
               improvable_states=int(improvable.sum()))
    with open(os.path.join(ART, "validation_eval", "selector_attrition_diagnostic.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=float)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
