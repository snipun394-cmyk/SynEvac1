"""READ-ONLY in-sample robust-action feasibility audit at tick 0 (VALIDATION only).

For each of the six tick-0 observable classes: every candidate that is never
worse than the default across all represented hidden realizations (weak) and
additionally strictly better for at least one (strict); outcome vectors,
predictions, frozen/R1 comparison, and whether robustness is observable.

"Worse" = the frozen A2 harm comparison: key(a) > key(d) under the exact
lexicographic order (T90, T100, unmet, hazard); equal keys are not harmful.
Selector tolerances (delta) are not part of that definition and are not used.

Hidden labels (t_h, p, compliance_key, template, seed) are read for VALIDATION
rows as diagnostic labels only; the models get exactly the frozen 54 columns.
No classifier, lookup policy, selector change, amendment or retraining.
PRIMARY TEST and other held-out roles sealed (only split_role is read).
Result label: IN-SAMPLE ROBUST-ACTION FEASIBILITY (not generalization).
Writes validation_eval/tick0_robust_action_audit.json.
"""
from __future__ import annotations

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
import run_issue2_baseline_selection as I2  # noqa: E402
from diagnose_decision_tick_observability import load_hidden  # noqa: E402
from diagnose_selector_r1_r6 import auc  # noqa: E402

ART = prior.OUT_DIR
MDIR = os.path.join(ART, "models", "c3_hgb")
EVAL = os.path.join(ART, "validation_eval", "c3_validation_evaluation.json")
OUT = os.path.join(ART, "validation_eval", "tick0_robust_action_audit.json")
T = ["future_T90", "future_T100", "future_unmet", "future_hazard_unmet"]
FEATS = list(lfd.ADMISSIBLE_FEATURES)
LABEL = "IN-SAMPLE ROBUST-ACTION FEASIBILITY (VALIDATION labels; not generalization)"


def keyv(y):
    return tuple(float(v) for v in y)


def main():
    ev = json.load(open(EVAL, encoding="utf-8"))
    before = {k: prior.sha256_file(os.path.join(ART, k)) for k in ev["integrity"]["protected_hashes"]}
    assert before == ev["integrity"]["protected_hashes"], "protected files differ from the evaluation record"
    man = json.load(open(os.path.join(MDIR, "training_manifest.json"), encoding="utf-8"))
    for mm in man["models"].values():
        assert prior.sha256_file(os.path.join(MDIR, mm["model_file"])) == mm["model_sha256"]
    states, rows, skipped = I2.load()
    hidden, other_roles = load_hidden()
    va = [s for s in states["VALIDATION"] if s["tick"] == 0]
    n = len(va)
    hid = [hidden[(s["traj"], 0)] for s in va]
    Y = np.stack([s["Y"] for s in va])
    cas = json.load(open(os.path.join(ART, "candidate_action_set.json"), encoding="utf-8"))
    names = {c["id"]: c["name"] for c in cas["candidates"]}
    zones, values = ["R1", "R2", "C1", "R3", "R4", "C2", "H"], ["UNTOUCHED", "FAVOR_A", "PARITY", "FAVOR_B"]
    desc = np.zeros((298, 28))
    for c in cas["candidates"]:
        for zi, z in enumerate(zones):
            desc[c["id"], zi * 4 + values.index(c["encoding"][z])] = 1.0
    classes = defaultdict(list)
    for i, s in enumerate(va):
        classes[s["x"]].append(i)
    assert len(classes) == 6
    # predictions: one per class (identical inputs -> identical predictions; verified below on a second member)
    sel = ev["part_C_selectors"]
    delta = np.array([sel["learned"]["deltas"][k] for k in ("T90", "T100", "U", "H")])
    W = sel["learned"]["width_threshold_T100"]
    name_to_id = {c["name"]: c["id"] for c in cas["candidates"]}
    iRem = FEATS.index("normalized_remaining")
    models = {}
    for t in T:
        for nm in ("mean", "q0.1", "q0.5", "q0.9"):
            models[(t, nm)] = pickle.loads(open(os.path.join(MDIR, f"{t}__{nm}.pkl"), "rb").read())

    def predict(x):
        X = np.concatenate([np.repeat(np.array([x], dtype=np.float64), 298, axis=0), desc], axis=1)
        m = np.stack([models[(t, "mean")].predict(X) for t in T], axis=1)
        q = np.sort(np.stack([np.stack([models[(t, nm)].predict(X) for t in T], axis=1)
                              for nm in ("q0.1", "q0.5", "q0.9")]), axis=0)
        return m, q

    out_classes = []
    for ci, (x, mem) in enumerate(sorted(classes.items(), key=lambda kv: -len(kv[1]))):
        m, q = predict(x)
        m2, _ = predict(va[mem[-1]]["x"])
        assert np.array_equal(m, m2)
        q1, q9 = q[0], q[2]
        d = name_to_id[{0: "BASE=A", 1: "BASE=P", 2: "BASE=B"}[lfd.label_free_default_action(np.array(x))]]
        K = [k for k in range(4) if not (k == 0 and round(x[iRem] * 40) <= 4)]
        Ym = Y[mem]                                                   # (members, 298, 4)
        keys = [[keyv(Ym[j, a]) for a in range(298)] for j in range(len(mem))]
        dk = [keys[j][d] for j in range(len(mem))]
        worse = np.array([[keys[j][a] > dk[j] for a in range(298)] for j in range(len(mem))])
        better = np.array([[keys[j][a] < dk[j] for a in range(298)] for j in range(len(mem))])
        bestk = [min(keys[j]) for j in range(len(mem))]
        captured = np.array([[keys[j][a] == bestk[j] and bestk[j] < dk[j] for a in range(298)] for j in range(len(mem))])
        identical_to_default = np.array([all(keys[j][a] == dk[j] for j in range(len(mem))) for a in range(298)])
        weak = ~worse.any(axis=0)
        weak_nd = weak.copy()
        weak_nd[d] = False
        strict = weak_nd & better.any(axis=0)
        regimes = [(hid[j]["t_h"], hid[j]["p"]) for j in mem]
        reg_set = sorted(set(regimes))
        # best fixed action under full VALIDATION knowledge
        order = sorted((a for a in range(298) if a != d),
                       key=lambda a: (int(worse[:, a].sum()), -int(better[:, a].sum()), -int(captured[:, a].sum()), a))
        bfa = order[0]

        def outcome_summary(a):
            ys = Ym[:, a]
            ks = [keys[j][a] for j in range(len(mem))]
            return dict(action_id=int(a), action=names[a],
                        worst_case_key=list(max(ks)), best_case_key=list(min(ks)),
                        per_target_max=dict(zip(T, ys.max(axis=0).tolist())), per_target_min=dict(zip(T, ys.min(axis=0).tolist())),
                        per_target_mean=dict(zip(T, ys.mean(axis=0).tolist())),
                        members_harmed=int(worse[:, a].sum()), members_strictly_improved=int(better[:, a].sum()),
                        members_equal=int(len(mem) - worse[:, a].sum() - better[:, a].sum()),
                        members_captured=int(captured[:, a].sum()),
                        mean_improvement_over_default=dict(zip(T, (Ym[:, d] - ys).mean(axis=0).tolist())))

        def prediction_summary(a):
            return dict(pred_mean=dict(zip(T, m[a].tolist())), pred_q9=dict(zip(T, q9[a].tolist())),
                        interval_width_q9_minus_q1=dict(zip(T, (q9[a] - q1[a]).tolist())),
                        pred_improvement_over_default=dict(zip(T, (m[d] - m[a]).tolist())),
                        pred_q9_improvement_over_default=dict(zip(T, (q9[d] - q9[a]).tolist())))
        # frozen / R1 on this class (identical for every member; tick 0 has no OOD state)
        res = {}
        for fl, ref in (("frozen", m[d]), ("R1", q9[d])):
            allowed = ~(((q9 > ref + delta) & np.isin(np.arange(4), K)[None, :]).any(axis=1))
            allowed[d] = True
            cand = np.nonzero(allowed)[0]
            for k in K[:-1]:
                cand = cand[m[cand, k] <= m[cand, k].min() + delta[k]]
            step3 = cand.copy()
            cand = cand[m[cand, 3] == m[cand, 3].min()]
            a = d if d in cand else int(cand.min())
            r = d
            for k in K:
                diff = m[a, k] - m[d, k]
                if abs(diff) > delta[k]:
                    r = a if diff < -delta[k] else d
                    break
            if r != d and (q9[a, 1] - m[a, 1]) > W:
                r = d
            res[fl] = dict(allowed=allowed, step3=set(step3.tolist()), a_star=a, rec=r)
        # can the model recognize strict-robust actions?  (AUC over the 297 non-default actions)
        nd = np.arange(298) != d
        lab = strict[nd]
        r1_margin = (q9 - q9[d] - delta)[:, K].max(axis=1)
        frozen_margin = (q9 - m[d] - delta)[:, K].max(axis=1)
        recog = dict(
            n_strict_robust=int(strict.sum()), n_non_default=int(nd.sum()),
            auc_minus_R1_floor_margin=auc(-r1_margin[nd], lab),
            auc_minus_frozen_floor_margin=auc(-frozen_margin[nd], lab),
            auc_pred_improvement={T[k]: auc((m[d, k] - m[nd, k]), lab) for k in range(4)},
            strict_robust_passing_R1_floor=int((strict & res["R1"]["allowed"]).sum()),
            strict_robust_passing_frozen_floor=int((strict & res["frozen"]["allowed"]).sum()),
            strict_robust_in_R1_step3_T90_T100_U_set=int(sum(a in res["R1"]["step3"] for a in np.nonzero(strict)[0])),
            R1_winner_is_strict_robust=bool(strict[res["R1"]["a_star"]]),
            rank_of_best_strict_robust_by_pred_hazard_among_R1_admitted=(
                int(1 + (m[res["R1"]["allowed"] & nd, 3] < m[bfa, 3]).sum()) if strict[bfa] else None),
            R1_admitted_non_default=int((res["R1"]["allowed"] & nd).sum()))
        # stability across hidden regimes
        per_regime = {}
        for rg in reg_set:
            idx = [j for j in range(len(mem)) if regimes[j] == rg]
            w_r = ~worse[idx].any(axis=0)
            w_r[d] = False
            s_r = w_r & better[idx].any(axis=0)
            per_regime[f"t_h={rg[0]},p={rg[1]}"] = dict(members=len(idx), weak_robust_non_default=int(w_r.sum()),
                                                        strict_robust=int(s_r.sum()),
                                                        class_strict_robust_actions_strictly_better_here=int((strict & better[idx].any(axis=0)).sum()))
        loro = []
        for rg in reg_set:
            tr_idx = [j for j in range(len(mem)) if regimes[j] != rg]
            te_idx = [j for j in range(len(mem)) if regimes[j] == rg]
            s_tr = ~worse[tr_idx].any(axis=0) & better[tr_idx].any(axis=0) & nd
            fails = s_tr & worse[te_idx].any(axis=0)
            loro.append(dict(held_out_regime=f"t_h={rg[0]},p={rg[1]}", strict_robust_on_other_regimes=int(s_tr.sum()),
                             of_which_harmful_in_held_out_regime=int(fails.sum()),
                             held_out_members_harmed_by_any_such_action=int(worse[np.ix_(te_idx, np.nonzero(s_tr)[0])].any(axis=1).sum())
                             if s_tr.any() else 0))
        # 6. observability of robustness: do identical observations need different actions?
        better_sets = [frozenset(np.nonzero(better[j])[0].tolist()) for j in range(len(mem))]
        safe_sets = [frozenset(np.nonzero(~worse[j] & nd)[0].tolist()) for j in range(len(mem))]
        n_pairs = disjoint_better = 0
        for j in range(len(mem)):
            for k2 in range(j + 1, len(mem)):
                n_pairs += 1
                disjoint_better += not (better_sets[j] & better_sets[k2]) and bool(better_sets[j]) and bool(better_sets[k2])
        universal_improver = [int(a) for a in np.nonzero(better.all(axis=0))[0]]
        profiles = defaultdict(list)
        for a in np.nonzero(strict)[0]:
            profiles[Ym[:, a].tobytes()].append(int(a))
        strict_profiles = [dict(actions=v, names=[names[a] for a in v], members_improved=int(better[:, v[0]].sum()),
                                members_captured=int(captured[:, v[0]].sum()),
                                mean_improvement_over_default=dict(zip(T, (Ym[:, d] - Ym[:, v[0]]).mean(axis=0).tolist())))
                           for v in sorted(profiles.values(), key=lambda v: -int(better[:, v[0]].sum()))]
        # does one robust profile dominate the others member by member?
        dominant = []
        for p_ in strict_profiles:
            a0 = p_["actions"][0]
            if all(all(keys[j][a0] <= keys[j][q_["actions"][0]] for j in range(len(mem))) for q_ in strict_profiles):
                dominant.append(p_["actions"])
        verdict = ("one common strict-robust action" if strict.sum() == 1 else
                   "multiple common strict-robust actions (action-selection ambiguity)" if strict.sum() > 1 else
                   "no common strict-robust action (early information insufficient for guaranteed-safe improvement)")
        out_classes.append(dict(
            class_index=ci,
            observable_signature=dict(zip(FEATS, x)),
            observable_signature_nonzero={f: v for f, v in zip(FEATS, x) if v != 0},
            validation_states=len(mem),
            hidden_templates=sorted({hid[j]["template"] for j in mem}),
            hidden_t_h=sorted({hid[j]["t_h"] for j in mem}), hidden_p=sorted({hid[j]["p"] for j in mem}),
            hidden_regimes_t_h_p=[f"t_h={a},p={b}" for a, b in reg_set],
            hidden_compliance_keys=len({hid[j]["compliance_key"] for j in mem}),
            candidates_evaluated=298, default=dict(id=int(d), name=names[d]),
            members_improvable=int(sum(bestk[j] < dk[j] for j in range(len(mem)))),
            actions_outcome_identical_to_default_everywhere=int(identical_to_default.sum() - 1),
            weak_robust_non_default=int(weak_nd.sum()),
            weak_robust_non_default_excluding_outcome_identical=int((weak_nd & ~identical_to_default).sum()),
            strict_robust=int(strict.sum()),
            best_fixed_action_full_validation_knowledge=dict(**outcome_summary(bfa),
                                                            non_worse_for_every_realization=bool(weak[bfa])),
            strict_robust_actions=[dict(**outcome_summary(a), prediction=prediction_summary(a))
                                   for a in np.nonzero(strict)[0]],
            comparison=dict(default=outcome_summary(d) | dict(prediction=prediction_summary(d)),
                            frozen_selector=dict(selected=int(res["frozen"]["rec"]), **outcome_summary(res["frozen"]["rec"])),
                            R1_selector=dict(selected=int(res["R1"]["rec"]), **outcome_summary(res["R1"]["rec"]),
                                             prediction=prediction_summary(res["R1"]["rec"]))),
            model_recognition=recog,
            stability_per_hidden_regime=per_regime,
            leave_one_regime_out_in_validation=loro,
            strict_robust_distinct_outcome_profiles=strict_profiles,
            strict_robust_profile_dominating_all_others_member_by_member=dominant,
            observability=dict(verdict=verdict, member_pairs=n_pairs,
                               member_pairs_both_improvable_with_disjoint_improving_sets=disjoint_better,
                               actions_strictly_better_for_every_member=universal_improver,
                               members_with_no_safe_non_default_action=int(sum(not s_ for s_ in safe_sets)))))

    # descriptive envelope: do classes WITHOUT a robust action contain actions whose predictions look like the robust ones?
    env_lo = np.full(4, np.inf)
    env_hi = np.full(4, -np.inf)
    for c in out_classes:
        for s in c["strict_robust_actions"]:
            v = np.array([s["prediction"]["pred_improvement_over_default"][t] for t in T])
            env_lo, env_hi = np.minimum(env_lo, v), np.maximum(env_hi, v)
    envelope = dict(definition="per-target [min, max] of predicted improvement over default across all strict-robust actions "
                               "of all classes; descriptive only, not a rule or threshold",
                    lower=dict(zip(T, env_lo.tolist())), upper=dict(zip(T, env_hi.tolist())), per_class={})
    for ci, (x, mem) in enumerate(sorted(classes.items(), key=lambda kv: -len(kv[1]))):
        m, _ = predict(x)
        d = out_classes[ci]["default"]["id"]
        pi = m[d] - m
        inside = np.all((pi >= env_lo) & (pi <= env_hi), axis=1)
        inside[d] = False
        Ym = Y[mem]
        dk = [keyv(Ym[j, d]) for j in range(len(mem))]
        harmed = np.array([sum(keyv(Ym[j, a]) > dk[j] for j in range(len(mem))) for a in range(298)])
        strict_ids = {s["action_id"] for s in out_classes[ci]["strict_robust_actions"]}
        ids = np.nonzero(inside)[0]
        envelope["per_class"][out_classes[ci]["hidden_templates"][0]] = dict(
            actions_inside_envelope=int(len(ids)), of_which_strict_robust=int(sum(a in strict_ids for a in ids)),
            members_harmed_by_inside_actions=dict(min=int(harmed[ids].min()), median=float(np.median(harmed[ids])),
                                                  max=int(harmed[ids].max())) if len(ids) else None,
            inside_actions_with_zero_harm=int((harmed[ids] == 0).sum()) if len(ids) else 0)
    after = {k: prior.sha256_file(os.path.join(ART, k)) for k in before}
    assert after == before
    out = dict(record="READ-ONLY tick-0 robust-action feasibility audit", label=LABEL,
               definitions=dict(worse="key(a) > key(d), exact lexicographic (T90, T100, unmet, hazard); A2 harm definition",
                                weak_robust="non-default a with key(a) <= key(d) for every VALIDATION member of the class",
                                strict_robust="weak robust and key(a) < key(d) for at least one member",
                                best_fixed_action="min members harmed, then max strictly improved, then max captured, then lowest id",
                                hidden_labels="diagnostic only; never model inputs"),
               tick0_states=n, classes=out_classes, prediction_envelope_check=envelope,
               inputs=dict(rows_parsed=rows, other_roles_seen_only_as_split_role=skipped,
                           hidden_pass_other_roles_seen_only_as_split_role=other_roles),
               not_done=["no classifier", "no lookup policy", "no hard-coded classes", "no selector change",
                         "no tolerance change", "no amendment", "no PRIMARY TEST"],
               integrity=dict(protected_files_unchanged=True, protected_hashes=before, model_hashes_verified=True,
                              primary_test_accessed=False))
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=float)
    print("written", OUT)


if __name__ == "__main__":
    main()
