"""READ-ONLY audit of the frozen selector specification on VALIDATION.

Answers whether the zero-switch result is an intended consequence of the
frozen safety standard or an incompatibility between the quantities the
safety floor and the switch rule compare. Regenerates the same VALIDATION
predictions from the hash-verified C3 models (deterministic; per-row
predictions were not stored earlier). Changes nothing: no selector,
threshold, model or protocol change; PRIMARY TEST and every other held-out
role sealed (only split_role is read for their rows).
Writes validation_eval/selector_specification_audit.json.
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
OUT = os.path.join(ART, "validation_eval", "selector_specification_audit.json")
T = ["future_T90", "future_T100", "future_unmet", "future_hazard_unmet"]
QS = [0.1, 0.5, 0.9]
FEATS = list(lfd.ADMISSIBLE_FEATURES)


def pct(a, ps=(5, 25, 50, 75, 95)):
    a = np.asarray(a, dtype=float)
    return {str(p): float(np.percentile(a, p)) for p in ps} if a.size else {}


def main():
    ev = json.load(open(EVAL, encoding="utf-8"))
    before = {k: prior.sha256_file(os.path.join(ART, k)) for k in ev["integrity"]["protected_hashes"]}
    assert before == ev["integrity"]["protected_hashes"], "protected files differ from the evaluation record"
    states, rows, skipped = I2.load()
    tr, va = states["TRAIN"], states["VALIDATION"]
    nv = len(va)
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
    q9 = qs[2]
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
    md = m[ar, d_id]                                   # (nv, 4) default means
    q9d = q9[ar, d_id]
    ua = q9 - m                                        # candidate upper width (nv, 298, 4)
    nd = np.ones((nv, 298), dtype=bool)
    nd[ar, d_id] = False
    # ---- floor per criterion (only criteria in K) ----
    floor_fail = (q9 > md[:, None, :] + delta) & Kmask[:, None, :]
    floor_pass = ~floor_fail.any(axis=2)
    # ---- pairwise switch condition vs default ----
    diff = m - md[:, None, :]                          # candidate minus default
    switch_ok = np.zeros((nv, 298), dtype=bool)
    first_k = np.full((nv, 298), -1)
    decided = np.zeros((nv, 298), dtype=bool)
    for k in range(4):
        big = (np.abs(diff[:, :, k]) > delta[k]) & Kmask[:, None, k] & ~decided
        switch_ok |= big & (diff[:, :, k] < -delta[k])
        first_k[big] = k
        decided |= big
    width_ok = (q9[:, :, 1] - m[:, :, 1]) <= W

    # ---- step-3 membership with the floor ----
    def step3(allowed):
        in_final = np.zeros((nv, 298), dtype=bool)
        astar = np.empty(nv, dtype=int)
        for i in range(nv):
            K = [k for k in range(4) if Kmask[i, k]]
            cand = np.nonzero(allowed[i])[0]
            for k in K[:-1]:
                mn = m[i, cand, k].min()
                cand = cand[m[i, cand, k] <= mn + delta[k]]
            cand = cand[m[i, cand, 3] == m[i, cand, 3].min()]
            in_final[i, cand] = True
            astar[i] = d_id[i] if d_id[i] in cand else int(cand.min())
        return in_final, astar
    allowed = floor_pass.copy()
    allowed[ar, d_id] = True
    fin, astar = step3(allowed)
    is_astar = np.zeros((nv, 298), dtype=bool)
    is_astar[ar, astar] = True
    N = int(nd.sum())
    stage = {}
    s0 = nd
    s1 = s0 & floor_pass
    s2 = s1 & fin
    s3 = s2 & is_astar
    s4 = s3 & switch_ok
    s5 = s4 & ~ood[:, None]
    s6 = s5 & width_ok
    funnel = [("0_non_default_pairs", s0), ("1_pass_safety_floor_all_criteria", s1), ("2_in_step3_final_set", s2),
              ("3_is_step3_winner_a_star", s3), ("4_pass_switch_rule", s4), ("5_not_OOD", s5), ("6_pass_width_abstention", s6)]
    for name, s in funnel:
        stage[name] = dict(pairs=int(s.sum()), states=int(s.any(axis=1).sum()))
    marginal = dict(
        mean_not_sufficiently_better_switch_rule=int((s0 & ~switch_ok).sum()),
        floor_fail_T90_or_T100=int((s0 & (floor_fail[:, :, 0] | floor_fail[:, :, 1])).sum()),
        floor_fail_unmet=int((s0 & floor_fail[:, :, 2]).sum()),
        floor_fail_hazard=int((s0 & floor_fail[:, :, 3]).sum()),
        floor_fail_any=int((s0 & ~floor_pass).sum()),
        step3_tolerance_removed_after_floor=int((s1 & ~fin).sum()),
        ood_pairs=int((s0 & ood[:, None]).sum()),
        width_fail_pairs=int((s0 & ~width_ok).sum()),
        switch_fail_among_a_star=int((s3 & ~switch_ok).sum()),
        pairs_passing_switch_rule_alone=int((s0 & switch_ok).sum()),
        pairs_passing_switch_and_floor=int((s0 & switch_ok & floor_pass).sum()),
        pairs_passing_switch_floor_width_not_ood=int((s0 & switch_ok & floor_pass & width_ok & ~ood[:, None]).sum()))
    # why do the 203 a* != d fail the switch rule?
    a_nd = astar != d_id
    fk = first_k[ar, astar]
    why = dict(states_astar_non_default=int(a_nd.sum()),
               no_criterion_differs_by_more_than_delta=int((a_nd & (fk == -1)).sum()),
               first_differing_criterion_is_worse=int((a_nd & (fk >= 0) & ~switch_ok[ar, astar]).sum()),
               first_differing_criterion_by_target={T[k]: int((a_nd & (fk == k)).sum()) for k in range(4)},
               improvement_of_a_star_over_default_on_mean={T[k]: pct(-diff[ar, astar, k][a_nd]) for k in range(4)})

    # ---- self-consistency: does the default pass its own floor? ----
    self_fail = (q9d > md + delta) & Kmask
    default_self = {T[k]: dict(share_states_default_fails_own_floor=float(self_fail[Kmask[:, k], k].mean()),
                               default_upper_width_minus_delta=pct((q9d - md)[Kmask[:, k], k] - delta[k]))
                    for k in range(4)}
    copy_test = "a candidate with exactly the default's mean and quantiles passes the floor only if q9_d,k - m_d,k <= delta_k for every k in K"
    # required predicted improvement vs realized oracle improvement
    req = {}
    for k in range(4):
        mask = s0 & Kmask[:, None, k]
        req_imp = (ua[:, :, k] - delta[k])[mask]
        realized = (Y[ar, d_id, k][:, None] - Y[:, :, k])[mask]
        req[T[k]] = dict(required_predicted_improvement_ua_minus_delta=pct(req_imp),
                         share_pairs_floor_demands_improvement=float((req_imp > 0).mean()),
                         realized_improvement_vs_default=pct(realized),
                         share_pairs_realized_improvement_ge_required=float((realized >= req_imp).mean()),
                         best_realized_improvement_per_state=pct((Y[ar, d_id, k] - Y[:, :, k].min(axis=1))[Kmask[:, k]]))

    # ---- Audit 3: the no-floor diagnostic switches ----
    allowed_all = np.ones((nv, 298), dtype=bool)
    fin0, astar0 = step3(allowed_all)
    sw0 = (astar0 != d_id) & switch_ok[ar, astar0]
    idx = np.nonzero(sw0)[0]
    a = astar0[idx]
    Ya, Yd = Y[idx, a], Y[idx, d_id[idx]]
    harm = np.array([tuple(Ya[j]) > tuple(Yd[j]) for j in range(len(idx))])
    fpass = floor_pass[idx, a]
    margin = (q9[idx, a] - md[idx] - delta)                         # >0 means floor violated on that criterion
    margin_max = np.where(Kmask[idx], margin, -np.inf).max(axis=1)

    def auc(score, label):
        pos, neg = score[label], score[~label]
        if not len(pos) or not len(neg):
            return None
        allv = np.concatenate([pos, neg])
        ranks = allv.argsort().argsort() + 1.0
        # average ranks for ties
        from collections import defaultdict
        grp = defaultdict(list)
        for r_, v in zip(ranks, allv):
            grp[v].append(r_)
        avg = {v: np.mean(rs) for v, rs in grp.items()}
        rp = sum(avg[v] for v in pos)
        return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
    audit3 = dict(
        n_switches=int(len(idx)), n_harmful=int(harm.sum()),
        floor_pass_among_switches=int(fpass.sum()),
        floor_fail_rate=dict(harmful=float((~fpass[harm]).mean()) if harm.any() else None,
                             not_harmful=float((~fpass[~harm]).mean()) if (~harm).any() else None),
        floor_fail_by_criterion={T[k]: dict(harmful=float(((margin[:, k] > 0) & Kmask[idx, k])[harm].mean()),
                                            not_harmful=float(((margin[:, k] > 0) & Kmask[idx, k])[~harm].mean()))
                                 for k in range(4)},
        auc_floor_margin_predicts_harm=auc(margin_max, harm),
        predicted_vs_actual={T[k]: dict(pred_mean_a=pct(m[idx, a, k]), pred_q9_a=pct(q9[idx, a, k]), actual_a=pct(Ya[:, k]),
                                        actual_default=pct(Yd[:, k]),
                                        q9_covers_actual=float((Ya[:, k] <= q9[idx, a, k]).mean()),
                                        predicted_improvement_mean=float((md[idx, k] - m[idx, a, k]).mean()),
                                        realized_improvement_mean=float((Yd[:, k] - Ya[:, k]).mean())) for k in range(4)},
        harm_by_first_switch_criterion={T[k]: dict(n=int((first_k[idx, a] == k).sum()),
                                                   harmful=int(harm[first_k[idx, a] == k].sum())) for k in range(4)})

    # ---- Audit 4: calibration ----
    cal = {}
    for k, t in enumerate(T):
        y = Y[:, :, k]
        def cov(qq):
            return {f"P(y<=q{q})": float((y <= qq[i, :, :, k]).mean()) for i, q in enumerate(QS)}
        cal[t] = dict(raw=cov(qr), sorted=cov(qs), nominal={f"P(y<=q{q})": q for q in QS},
                      central_80_sorted=float(((y >= qs[0, :, :, k]) & (y <= qs[2, :, :, k])).mean()),
                      default_rows_P_y_le_q9=float((Y[ar, d_id, k] <= q9d[:, k]).mean()),
                      nondefault_rows_P_y_le_q9=float((y <= q9[:, :, k])[nd].mean()),
                      mean_residual_actual_minus_q9_when_exceeded=float((y - q9[:, :, k])[y > q9[:, :, k]].mean())
                      if (y > q9[:, :, k]).any() else None)

    after = {k: prior.sha256_file(os.path.join(ART, k)) for k in before}
    assert after == before
    out = dict(
        record="READ-ONLY selector-specification audit on VALIDATION (no selector/threshold/model change)",
        inputs=dict(learned_deltas=dict(zip(("T90", "T100", "U", "H"), delta.tolist())), width_threshold=W, tau=tau,
                    predictions="regenerated from hash-verified C3 models on the same VALIDATION rows",
                    rows_parsed=rows, other_roles_seen_only_as_split_role=skipped),
        audit2_funnel=stage, audit2_marginal=marginal, audit2_why_a_star_does_not_switch=why,
        audit1_empirical=dict(default_self_floor=default_self, copy_test=copy_test, required_vs_realized_improvement=req),
        audit3_no_floor_switches=audit3, audit4_calibration=cal,
        integrity=dict(protected_files_unchanged=True, n_protected=len(before)))
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=float)
    print(json.dumps(out, indent=1, default=float)[:20000])


if __name__ == "__main__":
    main()
