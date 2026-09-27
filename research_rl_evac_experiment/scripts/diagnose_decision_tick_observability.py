"""VALIDATION-only diagnostic: is the selector failure a tick-0 observability
limit, or a failure of outcome prediction itself?

Per decision tick: observable diversity, hidden-regime ambiguity, C3
prediction diversity/error/discrimination, and frozen + R1 selector behaviour.
Hidden variables (t_h, p, compliance_key, template, seed, hazard_started) are
read for VALIDATION rows as DIAGNOSTIC LABELS ONLY; the models receive exactly
the frozen 54 columns. No retraining, no new feature, no selector/tolerance/
threshold change, no new data. PRIMARY TEST and other held-out roles sealed
(only split_role is read for their rows).
Writes validation_eval/decision_tick_observability_diagnostic.json.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import pickle
import sys
from collections import defaultdict

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
ROWS = os.path.join(ART, "full_dataset", "full_ai_learning_rows.csv.gz")
EVAL = os.path.join(ART, "validation_eval", "c3_validation_evaluation.json")
OUT = os.path.join(ART, "validation_eval", "decision_tick_observability_diagnostic.json")
T = ["future_T90", "future_T100", "future_unmet", "future_hazard_unmet"]
FEATS = list(lfd.ADMISSIBLE_FEATURES)
SEEDS = dict(recommendation=5544102, safety=5544103)
HIDDEN = ["template", "t_h", "p", "compliance_key", "seed", "hazard_started"]


def load_hidden():
    """(trajectory_id, decision_tick) -> hidden labels, VALIDATION rows only (candidate 0)."""
    out, other = {}, defaultdict(int)
    with gzip.open(ROWS, "rt", encoding="utf-8") as f:
        r = csv.reader(f)
        h = next(r)
        ix = {c: i for i, c in enumerate(h)}
        for row in r:
            role = row[ix["split_role"]]
            if role != "VALIDATION":
                other[role] += 1
                continue
            if row[ix["candidate_id"]] != "0":
                continue
            out[(row[ix["trajectory_id"]], int(row[ix["decision_tick"]]))] = {c: row[ix[c]] for c in HIDDEN}
    return out, dict(other)


def main():
    ev = json.load(open(EVAL, encoding="utf-8"))
    before = {k: prior.sha256_file(os.path.join(ART, k)) for k in ev["integrity"]["protected_hashes"]}
    assert before == ev["integrity"]["protected_hashes"], "protected files differ from the evaluation record"
    man = json.load(open(os.path.join(MDIR, "training_manifest.json"), encoding="utf-8"))
    for mm in man["models"].values():
        assert prior.sha256_file(os.path.join(MDIR, mm["model_file"])) == mm["model_sha256"]
    states, rows, skipped = I2.load()
    tr, va = states["TRAIN"], states["VALIDATION"]
    hidden, other_roles = load_hidden()
    nv = len(va)
    hid = [hidden[(s["traj"], s["tick"])] for s in va]
    traj = np.array([s["traj"] for s in va])
    ticks = np.array([s["tick"] for s in va])
    Y = np.stack([s["Y"] for s in va])
    cas = json.load(open(os.path.join(ART, "candidate_action_set.json"), encoding="utf-8"))
    zones, values = ["R1", "R2", "C1", "R3", "R4", "C2", "H"], ["UNTOUCHED", "FAVOR_A", "PARITY", "FAVOR_B"]
    desc = np.zeros((298, 28))
    for c in cas["candidates"]:
        for zi, z in enumerate(zones):
            desc[c["id"], zi * 4 + values.index(c["encoding"][z])] = 1.0
    n_desc_unique = len({tuple(r_) for r_ in desc.tolist()})
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
    ar = np.arange(nv)
    key = [list(map(tuple, Y[i].tolist())) for i in range(nv)]
    dkey = [key[i][d_id[i]] for i in range(nv)]
    best = [min(key[i]) for i in range(nv)]
    best_set = [frozenset(a for a in range(298) if key[i][a] == best[i]) for i in range(nv)]
    improvable = np.array([best[i] < dkey[i] for i in range(nv)])
    pair_harm = np.array([[key[i][a] > dkey[i] for a in range(298)] for i in range(nv)])
    pair_better = np.array([[key[i][a] < dkey[i] for a in range(298)] for i in range(nv)])
    nd = np.ones((nv, 298), bool)
    nd[ar, d_id] = False
    md, q9d = m[ar, d_id], q9[ar, d_id]

    def selector(floor):
        ref = md if floor == "frozen" else q9d
        allowed = ~(((q9 > ref[:, None, :] + delta) & Kmask[:, None, :]).any(axis=2))
        allowed[ar, d_id] = True
        astar, recm = np.empty(nv, int), d_id.copy()
        ab_ood, ab_w, sw = np.zeros(nv, bool), np.zeros(nv, bool), np.zeros(nv, bool)
        for i in range(nv):
            d = d_id[i]
            K = [k for k in range(4) if Kmask[i, k]]
            cand = np.nonzero(allowed[i])[0]
            for k in K[:-1]:
                mn = m[i, cand, k].min()
                cand = cand[m[i, cand, k] <= mn + delta[k]]
            cand = cand[m[i, cand, 3] == m[i, cand, 3].min()]
            a = d if d in cand else int(cand.min())
            astar[i] = a
            r = d
            for k in K:
                diff = m[i, a, k] - m[i, d, k]
                if abs(diff) > delta[k]:
                    r = a if diff < -delta[k] else d
                    break
            sw[i] = r != d
            if r != d and ood[i]:
                ab_ood[i], r = True, d
            elif r != d and (q9[i, a, 1] - m[i, a, 1]) > W:
                ab_w[i], r = True, d
            recm[i] = r
        return dict(allowed=allowed, astar=astar, rec=recm, ab_ood=ab_ood, ab_w=ab_w, sw=sw)
    SEL = {"frozen": selector("frozen"), "R1": selector("R1")}
    assert int((SEL["frozen"]["rec"] != d_id).sum()) == 0 and int((SEL["R1"]["rec"] != d_id).sum()) == 540

    def regime(i):
        return (hid[i]["t_h"], hid[i]["p"])

    def regime_full(i):
        return (hid[i]["t_h"], hid[i]["p"], hid[i]["compliance_key"])

    def entropy(labels):
        _, c = np.unique(np.array([str(x) for x in labels]), return_counts=True)
        p_ = c / c.sum()
        return float(-(p_ * np.log2(p_)).sum())

    # next-tick lookup for "does the next observation separate them?"
    by_traj_tick = {(traj[i], ticks[i]): i for i in range(nv)}
    tick_list = sorted(set(ticks.tolist()))
    out_ticks = {}
    for t in tick_list:
        S = np.nonzero(ticks == t)[0]
        # ---- 1. diversity ----
        pred_sig = {hashlib.sha256(np.round(np.concatenate([m[i].ravel(), qr[:, i].ravel()]), 9).tobytes()).hexdigest() for i in S}
        classes = defaultdict(list)
        for i in S:
            classes[va[i]["x"]].append(i)
        div = dict(states=int(len(S)), trajectories=int(len(set(traj[S].tolist()))),
                   unique_observable_x26=len(classes), unique_54_inputs=len(classes) * n_desc_unique,
                   distinct_prediction_matrices=len(pred_sig),
                   distinct_hidden_regimes_t_h_p=len({regime(i) for i in S}),
                   distinct_hidden_regimes_t_h_p_compliance=len({regime_full(i) for i in S}),
                   templates=len({hid[i]["template"] for i in S}),
                   hazard_started_states=int(sum(hid[i]["hazard_started"] in ("1", "True", "true") for i in S)))
        # ---- 2. ambiguity ----
        amb_cls = [v for v in classes.values() if len({regime(i) for i in v}) > 1]
        amb_full = [v for v in classes.values() if len({regime_full(i) for i in v}) > 1]
        multi = [v for v in classes.values() if len(v) > 1]
        y_het = [v for v in multi if any(not np.array_equal(Y[v[0]], Y[j]) for j in v[1:])]
        best_div = [v for v in multi if not frozenset.intersection(*[best_set[j] for j in v])]
        pure_states = sum(len(v) for v in classes.values() if len({regime(i) for i in v}) == 1)
        H_reg = entropy([regime(i) for i in S])
        H_cond = sum(len(v) / len(S) * entropy([regime(i) for i in v]) for v in classes.values())
        amb = dict(
            classes=len(classes), classes_with_multiple_members=len(multi),
            classes_with_multiple_t_h_p_regimes=len(amb_cls), share_classes_multi_regime=len(amb_cls) / len(classes),
            states_in_multi_regime_classes=int(sum(len(v) for v in amb_cls)),
            share_states_in_multi_regime_classes=sum(len(v) for v in amb_cls) / len(S),
            classes_with_multiple_regime_or_compliance=len(amb_full),
            share_states_in_multi_regime_or_compliance_classes=sum(len(v) for v in amb_full) / len(S),
            classes_where_true_outcomes_differ_across_members=len(y_het),
            share_states_in_outcome_heterogeneous_classes=sum(len(v) for v in y_het) / len(S),
            classes_with_no_common_best_action=len(best_div),
            share_states_in_classes_with_no_common_best_action=sum(len(v) for v in best_div) / len(S),
            share_states_regime_identified_by_observation=pure_states / len(S),
            regime_entropy_bits=H_reg, conditional_regime_entropy_given_observation_bits=H_cond,
            share_of_regime_uncertainty_resolved=(1 - H_cond / H_reg) if H_reg > 0 else None)
        # ---- 3. prediction quality / separability ----
        pq = {}
        for k in range(4):
            y, yh = Y[S, :, k], m[S, :, k]
            sst = float(((y - y.mean()) ** 2).sum())
            within = y - y.mean(axis=1, keepdims=True)
            within_h = yh - yh.mean(axis=1, keepdims=True)
            Rd = (Y[S, d_id[S], k][:, None] - y)            # realized improvement of each candidate over default
            Id = (md[S, k][:, None] - yh)                   # predicted improvement
            mask = nd[S] & (Rd != 0) & Kmask[S, k][:, None]
            pq[T[k]] = dict(
                mae=float(np.abs(y - yh).mean()), r2=float(1 - ((y - yh) ** 2).sum() / sst) if sst > 0 else None,
                outcome_sd=float(y.std()),
                within_state_candidate_sd_actual=float(within.std()), within_state_candidate_sd_predicted=float(within_h.std()),
                within_state_corr_pred_vs_actual=float(np.corrcoef(within.ravel(), within_h.ravel())[0, 1])
                if within.std() > 0 and within_h.std() > 0 else None,
                q9_coverage_raw=float((y <= q9[S, :, k]).mean()),
                q9_coverage_integer_read=float((y <= np.ceil(q9[S, :, k] - 1e-6)).mean()) if k < 2 else None,
                direction_auc_pred_improvement_vs_realized_better=auc(Id[mask], (Rd[mask] > 0)) if mask.any() else None,
                nondefault_pairs_realized_better=int((nd[S] & (Rd > 0)).sum()),
                nondefault_pairs_realized_worse=int((nd[S] & (Rd < 0)).sum()))
        mr1 = np.where(Kmask[S][:, None, :], q9[S] - q9d[S][:, None, :] - delta, -np.inf).max(axis=2)
        mold = np.where(Kmask[S][:, None, :], q9[S] - md[S][:, None, :] - delta, -np.inf).max(axis=2)
        hs = pair_harm[S][nd[S]]
        sep = dict(auc_R1_floor_margin_vs_pair_harm=auc(mr1[nd[S]], hs), auc_frozen_floor_margin_vs_pair_harm=auc(mold[nd[S]], hs),
                   nondefault_pairs=int(nd[S].sum()), harmful_pairs=int(hs.sum()), better_pairs=int(pair_better[S][nd[S]].sum()))
        # ---- 4. selectors ----
        sel_out = {}
        for sn, o in SEL.items():
            r = o["rec"][S]
            dv = r != d_id[S]
            cap = np.array([key[i][o["rec"][i]] == best[i] for i in S])
            harm = np.array([key[i][o["rec"][i]] > dkey[i] for i in S])
            imp = improvable[S]
            sel_out[sn] = dict(
                switches=int(dv.sum()), agreement_with_default=float(1 - dv.mean()),
                capture=dict(count=int((cap & imp).sum()), of_improvable=int(imp.sum()),
                             rate=float((cap & imp).sum() / imp.sum()) if imp.any() else None,
                             ci95=EV.cluster_boot(traj[S], (cap & imp).astype(float), imp.astype(float), SEEDS["recommendation"]) if imp.any() else None),
                harm=dict(count=int(harm.sum()), rate=float(harm.mean()),
                          ci95=EV.cluster_boot(traj[S], harm.astype(float), np.ones(len(S)), SEEDS["safety"])),
                abstain_ood=int(o["ab_ood"][S].sum()), abstain_width=int(o["ab_w"][S].sum()),
                ood_states=int(ood[S].sum()),
                safe_set_nondefault_states=int((o["allowed"][S] & nd[S]).any(axis=1).sum()),
                a_star_non_default_states=int((o["astar"][S] != d_id[S]).sum()),
                pass_switch_rule_states=int(o["sw"][S].sum()),
                a_star_predicted_improvement={T[k]: pct((md[S, k] - m[S, o["astar"][S], k])) for k in range(4)})
        oracle_gain = {T[k]: pct(Y[S, d_id[S], k] - Y[S, :, k].min(axis=1)) for k in range(4)}
        best_gain_best_action = {T[k]: pct([Y[i, d_id[i], k] - Y[i, next(iter(best_set[i])), k] for i in S if improvable[i]])
                                 for k in range(4)}
        # ---- 5. counterfactual class test ----
        cf = dict(ambiguous_classes=len(amb_cls))
        if amb_cls:
            common_best = safe_fixed = safe_fixed_improving = r1_safe = 0
            sep_pairs = tot_pairs = 0
            for v in amb_cls:
                common_best += bool(frozenset.intersection(*[best_set[j] for j in v]))
                d = d_id[v[0]]
                assert all(d_id[j] == d for j in v)
                never_worse = ~pair_harm[v].any(axis=0) & nd[v[0]]
                strictly = pair_better[v].any(axis=0)
                safe_fixed += bool(never_worse.any())
                safe_fixed_improving += bool((never_worse & strictly).any())
                a = SEL["R1"]["astar"][v[0]]
                r1_safe += bool(a == d or not pair_harm[v, a].any())
                for x_ in range(len(v)):
                    for y_ in range(x_ + 1, len(v)):
                        i, j = v[x_], v[y_]
                        if regime(i) == regime(j):
                            continue
                        ni, nj = by_traj_tick.get((traj[i], t + 5)), by_traj_tick.get((traj[j], t + 5))
                        if ni is None or nj is None:
                            continue
                        tot_pairs += 1
                        sep_pairs += va[ni]["x"] != va[nj]["x"]
            n = len(amb_cls)
            cf.update(share_with_common_best_action=common_best / n,
                      share_with_fixed_non_default_action_never_worse_than_default=safe_fixed / n,
                      share_with_fixed_action_never_worse_and_strictly_better_somewhere=safe_fixed_improving / n,
                      share_where_R1_winner_is_safe_for_every_member=r1_safe / n,
                      cross_regime_member_pairs_with_next_tick_state=tot_pairs,
                      share_of_those_separated_at_next_tick=(sep_pairs / tot_pairs) if tot_pairs else None,
                      note="members of a class have identical 54-column inputs, hence identical predictions")
        # ---- zero-harm class-oracle bound (in-sample, label-based UPPER BOUND, not a rule) ----
        zi = zc = cls_with = r1_admits = pred_h_imp = 0
        pred_h_list = []
        for v in classes.values():
            d = d_id[v[0]]
            never_worse = ~pair_harm[v].any(axis=0) & nd[v[0]]
            n_better = pair_better[v].sum(axis=0) * never_worse
            if n_better.max() <= 0:
                continue
            a = int(np.argmax(n_better))
            cls_with += 1
            zi += int(n_better[a])
            zc += int(sum(key[j][a] == best[j] and improvable[j] for j in v))
            r1_admits += bool(SEL["R1"]["allowed"][v[0], a])
            pred_h_list.append(float(md[v[0], 3] - m[v[0], a, 3]))
        zero_harm_bound = dict(
            definition="per observable class, the single non-default action never worse than default for ANY member "
                       "and strictly better for the most members; label-based, in-sample upper bound for any "
                       "deterministic observation-based policy with zero harm",
            classes_with_such_action=cls_with, of_classes=len(classes),
            states_strictly_improved=zi, states_captured=zc, improvable_states_at_tick=int(improvable[S].sum()),
            classes_where_R1_floor_admits_that_action=r1_admits,
            predicted_hazard_improvement_of_that_action=pct(pred_h_list) if pred_h_list else None,
            delta_H=float(delta[3]))
        out_ticks[str(t)] = dict(zero_harm_class_oracle_bound=zero_harm_bound,diversity=div, ambiguity=amb, prediction=pq, harm_separability=sep, selectors=sel_out,
                                 oracle_best_improvement_over_default=oracle_gain,
                                 improvement_of_a_best_action_in_improvable_states=best_gain_best_action,
                                 counterfactual_classes=cf)

    after = {k: prior.sha256_file(os.path.join(ART, k)) for k in before}
    assert after == before
    out = dict(
        record="VALIDATION-only decision-tick observability diagnostic (NOT a selector, NOT a new experiment)",
        hidden_labels_use="t_h, p, compliance_key, template, seed, hazard_started read for VALIDATION rows as diagnostic labels only; never model inputs",
        regime_definition="hidden regime = (t_h, p); finer = (t_h, p, compliance_key)",
        inputs=dict(deltas=dict(zip(("T90", "T100", "U", "H"), delta.tolist())), width_threshold=W, tau=tau,
                    unique_candidate_descriptors=n_desc_unique, ticks=tick_list, rows_parsed=rows,
                    other_roles_seen_only_as_split_role=skipped, hidden_pass_other_roles_seen_only_as_split_role=other_roles,
                    bootstrap="frozen cluster_boot, 2,000 resamples", seeds=SEEDS),
        per_tick=out_ticks,
        integrity=dict(protected_files_unchanged=True, protected_hashes=before, model_hashes_verified=True,
                       selector_code_unchanged=True, no_retraining=True, no_new_features=True, no_new_data=True,
                       primary_test_accessed=False))
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=float)
    print("written", OUT)


if __name__ == "__main__":
    main()
