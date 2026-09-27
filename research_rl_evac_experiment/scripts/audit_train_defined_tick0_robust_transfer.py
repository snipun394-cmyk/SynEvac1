"""READ-ONLY audit: do TRAIN-defined tick-0 robust actions transfer to VALIDATION?

Phase 1 (TRAIN only): stream the dataset, keep TRAIN tick-0 rows, and for each
tick-0 observable class build the TRAIN-robust set: non-default actions whose
realized key is never worse than the label-free default for ANY TRAIN member of
the class (exact lexicographic key T90 -> T100 -> unmet -> hazard; no delta) and
strictly better for at least one. The sets are frozen and hashed BEFORE any
VALIDATION row is read.

Phase 2 (VALIDATION): stream again, keep VALIDATION tick-0 rows, and evaluate
every frozen action unchanged. VALIDATION outcomes never define, rank, select or
alter an action.

No model is loaded or trained; no selector, tolerance, dataset or protocol
change; no policy. PRIMARY TEST and every other held-out role: only the
split_role field of their rows is read.
Writes validation_eval/train_defined_tick0_robust_transfer.json (+ .md written by hand).
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict

import numpy as np

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)
import ai_learning_label_free_default as lfd  # noqa: E402
import audit_pre_training_ai_dataset as prior  # noqa: E402

ART = prior.OUT_DIR
ROWS = os.path.join(ART, "full_dataset", "full_ai_learning_rows.csv.gz")
MDIR = os.path.join(ART, "models", "c3_hgb")
VE = os.path.join(ART, "validation_eval")
EVAL = os.path.join(VE, "c3_validation_evaluation.json")
TICK0_AUDIT = os.path.join(VE, "tick0_robust_action_audit.json")
OUT = os.path.join(VE, "train_defined_tick0_robust_transfer.json")
OWN_OUTPUTS = {os.path.normpath(OUT), os.path.normpath(os.path.join(VE, "TRAIN_DEFINED_TICK0_ROBUST_TRANSFER.md"))}
T = ["future_T90", "future_T100", "future_unmet", "future_hazard_unmet"]
FEATS = list(lfd.ADMISSIBLE_FEATURES)
HIDDEN = ["template", "t_h", "p", "compliance_key", "seed"]
DATASET_SHA = "36aa757f6f2414713d7f5625f562298df2e72baf97a61969a69c0e3ecb967179"
UNSEEN_T_H, UNSEEN_P = "6", "0.8"
ALPHA = 0.05


class IntegrityError(RuntimeError):
    pass


# ---------------------------------------------------------------- pure helpers
def key(y) -> tuple:
    return tuple(float(v) for v in y)


def default_id_for(x, name_to_id) -> int:
    return name_to_id[{0: "BASE=A", 1: "BASE=P", 2: "BASE=B"}[lfd.label_free_default_action(np.array(x))]]


def robust_set(Ymem: np.ndarray, d: int) -> dict:
    """Ymem: (members, n_cand, 4) realized outcomes of one class. Returns weak/strict robust ids."""
    n_m, n_c, _ = Ymem.shape
    worse = np.zeros((n_m, n_c), bool)
    better = np.zeros((n_m, n_c), bool)
    for j in range(n_m):
        dk = key(Ymem[j, d])
        for a in range(n_c):
            ka = key(Ymem[j, a])
            worse[j, a] = ka > dk
            better[j, a] = ka < dk
    weak = ~worse.any(axis=0)
    weak[d] = False
    strict = weak & better.any(axis=0)
    identical = np.array([all(key(Ymem[j, a]) == key(Ymem[j, d]) for j in range(n_m)) for a in range(n_c)])
    identical[d] = False
    patterns = defaultdict(list)
    for a in np.nonzero(strict)[0]:
        patterns[Ymem[:, a].tobytes()].append(int(a))
    return dict(weak=[int(a) for a in np.nonzero(weak)[0]], strict=[int(a) for a in np.nonzero(strict)[0]],
                outcome_identical_to_default=int(identical.sum()),
                strict_patterns=sorted(patterns.values(), key=lambda v: v[0]),
                members_improved={int(a): int(better[:, a].sum()) for a in np.nonzero(strict)[0]})


def binom_cdf(k: int, n: int, p: float) -> float:
    if p <= 0:
        return 1.0
    if p >= 1:
        return 0.0 if k < n else 1.0
    lp, lq = math.log(p), math.log1p(-p)
    s = 0.0
    for i in range(k + 1):
        s += math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq)
    return min(1.0, s)


def cp_upper(k: int, n: int, alpha: float = ALPHA) -> float | None:
    """Exact one-sided (1 - alpha) Clopper-Pearson upper bound for a binomial proportion."""
    if n == 0:
        return None
    if k >= n:
        return 1.0
    lo, hi = k / n, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if binom_cdf(k, n, mid) > alpha:
            lo = mid
        else:
            hi = mid
    return hi


def load_tick0(path: str, role: str, n_cand: int = 298):
    """Tick-0 states of ONE role. For rows of any other role only split_role is inspected."""
    states, other = {}, Counter()
    with gzip.open(path, "rt", encoding="utf-8") as f:
        r = csv.reader(f)
        h = next(r)
        ix = {c: i for i, c in enumerate(h)}
        irole, itick = ix["split_role"], ix["decision_tick"]
        for row in r:
            if row[irole] != role:
                other[row[irole]] += 1
                continue
            if row[itick] != "0":
                continue
            tid = row[ix["trajectory_id"]]
            s = states.get(tid)
            if s is None:
                s = states[tid] = dict(traj=tid, x=tuple(float(row[ix[c]]) for c in FEATS),
                                       hidden={c: row[ix[c]] for c in HIDDEN}, Y=np.full((n_cand, 4), np.nan), seen=set())
            cid = int(row[ix["candidate_id"]])
            s["Y"][cid] = [float(row[ix[t]]) for t in T]
            s["seen"].add(cid)
    for s in states.values():
        if s["seen"] != set(range(n_cand)):
            raise IntegrityError(f"state {s['traj']} does not have candidates 0..{n_cand - 1}")
        del s["seen"]
    return list(states.values()), dict(other)


def outcome_block(Yv: np.ndarray, dv: list, a: int) -> dict:
    """Evaluate one fixed action a on VALIDATION members (Yv: (n, n_cand, 4); dv: default id per member)."""
    n = len(dv)
    if n == 0:
        return dict(states=0)
    imp = harm = 0
    worst = None
    deltas = np.zeros((n, 4))
    for j in range(n):
        ka, kd = key(Yv[j, a]), key(Yv[j, dv[j]])
        imp += ka < kd
        harm += ka > kd
        worst = ka if worst is None or ka > worst else worst
        deltas[j] = Yv[j, a] - Yv[j, dv[j]]
    return dict(states=n, improved=int(imp), harmed=int(harm), tied=int(n - imp - harm),
                mean_delta_selected_minus_default=dict(zip(T, deltas.mean(axis=0).tolist())),
                worst_observed_key=list(worst), zero_harm=bool(harm == 0),
                harm_rate=harm / n, harm_upper_bound_one_sided_95_clopper_pearson=cp_upper(int(harm), n))


# ---------------------------------------------------------------- integrity
def hash_tree(root: str, exclude: set) -> dict:
    out = {}
    for dp, _, fns in os.walk(root):
        for fn in fns:
            p = os.path.normpath(os.path.join(dp, fn))
            if p in exclude:
                continue
            out[os.path.relpath(p, root).replace("\\", "/")] = prior.sha256_file(p)
    return out


def main():
    ev = json.load(open(EVAL, encoding="utf-8"))
    tree_before = hash_tree(ART, OWN_OUTPUTS)
    prot = ev["integrity"]["protected_hashes"]
    for k_, h_ in prot.items():
        if tree_before.get(k_) != h_:
            raise IntegrityError(f"protected file changed: {k_}")
    if tree_before["full_dataset/full_ai_learning_rows.csv.gz"] != DATASET_SHA:
        raise IntegrityError("dataset hash mismatch")
    man = json.load(open(os.path.join(MDIR, "training_manifest.json"), encoding="utf-8"))
    for mm in man["models"].values():
        if tree_before.get(f"models/c3_hgb/{mm['model_file']}") != mm["model_sha256"]:
            raise IntegrityError(f"model hash mismatch: {mm['model_file']}")
    tick0_prev = json.load(open(TICK0_AUDIT, encoding="utf-8"))
    cas = json.load(open(os.path.join(ART, "candidate_action_set.json"), encoding="utf-8"))
    names = {c["id"]: c["name"] for c in cas["candidates"]}
    name_to_id = {c["name"]: c["id"] for c in cas["candidates"]}
    sigs_prev = {tuple(c["observable_signature"][f] for f in FEATS): c["hidden_templates"][0] for c in tick0_prev["classes"]}

    # ------------------------------------------------ phase 1: TRAIN only
    train, other_train_pass = load_tick0(ROWS, "TRAIN")
    tr_classes = defaultdict(list)
    for s in train:
        tr_classes[s["x"]].append(s)
    if set(tr_classes) != set(sigs_prev):
        raise IntegrityError("TRAIN tick-0 classes differ from the six classes of the tick-0 audit")
    frozen, train_report = {}, {}
    for x, mem in tr_classes.items():
        tmpl = sigs_prev[x]
        ds = {default_id_for(s["x"], name_to_id) for s in mem}
        if len(ds) != 1:
            raise IntegrityError("default differs within a class")
        d = ds.pop()
        rs = robust_set(np.stack([s["Y"] for s in mem]), d)
        frozen[tmpl] = dict(default=d, actions=rs["strict"])
        regimes = Counter((s["hidden"]["t_h"], s["hidden"]["p"]) for s in mem)
        train_report[tmpl] = dict(
            train_states=len(mem), train_regimes={f"t_h={a},p={b}": n for (a, b), n in sorted(regimes.items())},
            train_templates=sorted({s["hidden"]["template"] for s in mem}),
            default=dict(id=d, name=names[d]),
            weak_robust_non_default=len(rs["weak"]), outcome_identical_to_default=rs["outcome_identical_to_default"],
            train_robust_improving=len(rs["strict"]),
            train_robust_actions=[dict(id=a, name=names[a], train_members_improved=rs["members_improved"][a]) for a in rs["strict"]],
            distinct_train_outcome_patterns=len(rs["strict_patterns"]), train_patterns=rs["strict_patterns"],
            common_robust_action_exists=bool(rs["strict"]),
            prespecified_tie_break_required=len(rs["strict"]) > 1,
            tie_break_outcome_relevant_on_train=len(rs["strict_patterns"]) > 1)
    frozen_json = json.dumps(dict(sorted(frozen.items())), sort_keys=True)
    frozen_sha = hashlib.sha256(frozen_json.encode()).hexdigest()
    train_ids = {s["traj"] for s in train}
    train_seeds = {s["hidden"]["seed"] for s in train}
    train_regime_set = {(s["hidden"]["t_h"], s["hidden"]["p"]) for s in train}
    del train, tr_classes                                   # nothing from TRAIN outcomes is used below except `frozen`

    # ------------------------------------------------ phase 2: VALIDATION evaluation of the frozen sets
    val, other_val_pass = load_tick0(ROWS, "VALIDATION")
    if hashlib.sha256(json.dumps(dict(sorted(frozen.items())), sort_keys=True).encode()).hexdigest() != frozen_sha:
        raise IntegrityError("frozen TRAIN sets changed")
    if train_ids & {s["traj"] for s in val} or train_seeds & {s["hidden"]["seed"] for s in val}:
        raise IntegrityError("TRAIN/VALIDATION trajectory or seed overlap")
    va_classes = defaultdict(list)
    for s in val:
        va_classes[s["x"]].append(s)
    if set(va_classes) != set(sigs_prev):
        raise IntegrityError("VALIDATION tick-0 classes differ from the tick-0 audit")
    val_regimes = sorted({(s["hidden"]["t_h"], s["hidden"]["p"]) for s in val})

    def subgroup(s, g):
        th, p = s["hidden"]["t_h"], s["hidden"]["p"]
        return dict(all=True,
                    t_h_6=th == UNSEEN_T_H, p_0_8=p == UNSEEN_P,
                    t_h_6_and_p_0_8=th == UNSEEN_T_H and p == UNSEEN_P,
                    t_h_6_only_seen_p=th == UNSEEN_T_H and p != UNSEEN_P,
                    p_0_8_only_seen_t_h=p == UNSEEN_P and th != UNSEEN_T_H,
                    other_regimes_neither_unseen=th != UNSEEN_T_H and p != UNSEEN_P)[g]
    GROUPS = ["all", "t_h_6", "p_0_8", "t_h_6_and_p_0_8", "t_h_6_only_seen_p", "p_0_8_only_seen_t_h", "other_regimes_neither_unseen"]
    per_class, overall_union = {}, {g: dict(states=0, states_under_a_train_robust_action=0, harmed_by_some_action=0,
                                            improved_by_every_action=0) for g in GROUPS}
    for x, mem in va_classes.items():
        tmpl = sigs_prev[x]
        fz = frozen[tmpl]
        d = fz["default"]
        if any(default_id_for(s["x"], name_to_id) != d for s in mem):
            raise IntegrityError("VALIDATION default differs from TRAIN default for a class")
        Yv = np.stack([s["Y"] for s in mem])
        dv = [d] * len(mem)
        cls = dict(validation_states=len(mem), frozen_train_actions=fz["actions"], per_group={}, per_regime={})
        for g in GROUPS:
            idx = [j for j, s in enumerate(mem) if subgroup(s, g)]
            blk = dict(states=len(idx))
            if fz["actions"] and idx:
                acts = {int(a): outcome_block(Yv[idx], [d] * len(idx), a) for a in fz["actions"]}
                harmed_any = sum(any(key(Yv[j, a]) > key(Yv[j, d]) for a in fz["actions"]) for j in idx)
                improved_all = sum(all(key(Yv[j, a]) < key(Yv[j, d]) for a in fz["actions"]) for j in idx)
                blk.update(per_action=acts,
                           actions_with_zero_harm=sum(v["zero_harm"] for v in acts.values()), actions=len(acts),
                           zero_harm_survives_for_every_action=all(v["zero_harm"] for v in acts.values()),
                           states_harmed_by_at_least_one_action=int(harmed_any),
                           harm_upper_bound_any_tie_break=cp_upper(int(harmed_any), len(idx)),
                           harmed_range_over_actions=[min(v["harmed"] for v in acts.values()), max(v["harmed"] for v in acts.values())],
                           improved_range_over_actions=[min(v["improved"] for v in acts.values()), max(v["improved"] for v in acts.values())],
                           default_worst_observed_key=list(max(key(Yv[j, d]) for j in idx)))
                overall_union[g]["states"] += len(idx)
                overall_union[g]["states_under_a_train_robust_action"] += len(idx)
                overall_union[g]["harmed_by_some_action"] += int(harmed_any)
                overall_union[g]["improved_by_every_action"] += int(improved_all)
            elif idx:
                blk.update(note="no TRAIN-robust improving action: default retained")
                overall_union[g]["states"] += len(idx)
            cls["per_group"][g] = blk
        for rg in val_regimes:
            idx = [j for j, s in enumerate(mem) if (s["hidden"]["t_h"], s["hidden"]["p"]) == rg]
            key_ = f"t_h={rg[0]},p={rg[1]}"
            if fz["actions"] and idx:
                acts = {int(a): outcome_block(Yv[idx], [d] * len(idx), a) for a in fz["actions"]}
                cls["per_regime"][key_] = dict(states=len(idx), seen_in_train=rg in train_regime_set,
                                               harmed_range_over_actions=[min(v["harmed"] for v in acts.values()), max(v["harmed"] for v in acts.values())],
                                               improved_range_over_actions=[min(v["improved"] for v in acts.values()), max(v["improved"] for v in acts.values())],
                                               zero_harm_survives_for_every_action=all(v["zero_harm"] for v in acts.values()),
                                               harm_upper_bound_if_zero=cp_upper(0, len(idx)))
            else:
                cls["per_regime"][key_] = dict(states=len(idx), seen_in_train=rg in train_regime_set, actions=0)
        # descriptive comparison with the VALIDATION in-sample sets of the previous audit (not used for anything)
        prev = next(c for c in tick0_prev["classes"] if c["hidden_templates"][0] == tmpl)
        prev_set = {s_["action_id"] for s_ in prev["strict_robust_actions"]}
        cls["descriptive_overlap_with_validation_in_sample_set_0ada5ec"] = dict(
            train_set=len(fz["actions"]), validation_in_sample_set=len(prev_set),
            intersection=len(set(fz["actions"]) & prev_set))
        per_class[tmpl] = cls
    for g, v in overall_union.items():
        n_act = v["states_under_a_train_robust_action"]
        # bound only over states where an action is applied (default-retained states cannot be harmed)
        v["harm_upper_bound_any_tie_break_over_acted_states"] = cp_upper(v["harmed_by_some_action"], n_act) if n_act else None
    n_val = len(val)
    cov_states = sum(len(mem) for x, mem in va_classes.items() if frozen[sigs_prev[x]]["actions"])
    multi_states = sum(len(mem) for x, mem in va_classes.items() if len(frozen[sigs_prev[x]]["actions"]) > 1)
    multi_pattern_states = sum(len(mem) for x, mem in va_classes.items() if train_report[sigs_prev[x]]["distinct_train_outcome_patterns"] > 1)
    coverage = dict(validation_tick0_states=n_val,
                    share_with_at_least_one_train_robust_improving_action=cov_states / n_val,
                    share_with_no_such_action=1 - cov_states / n_val,
                    share_with_multiple_train_robust_actions=multi_states / n_val,
                    share_with_multiple_distinct_train_outcome_patterns=multi_pattern_states / n_val)
    classes_with_actions = [t for t in per_class if frozen[t]["actions"]]
    surv = {t: per_class[t]["per_group"]["all"]["zero_harm_survives_for_every_action"] for t in classes_with_actions}
    surv_some = {t: per_class[t]["per_group"]["all"]["actions_with_zero_harm"] > 0 for t in classes_with_actions}
    if classes_with_actions and all(surv.values()):
        cls_ = "A"
    elif any(surv_some.values()):
        cls_ = "B"
    else:
        cls_ = "C"

    tree_after = hash_tree(ART, OWN_OUTPUTS)
    changed = sorted(k_ for k_ in set(tree_before) | set(tree_after) if tree_before.get(k_) != tree_after.get(k_))
    if changed:
        raise IntegrityError(f"files changed during the audit: {changed}")
    out = dict(
        record="READ-ONLY TRAIN-defined tick-0 robust-set transfer audit (no model, no policy)",
        definition=dict(
            comparison="exact frozen lexicographic key (T90, T100, future_unmet, future_hazard_unmet); no delta",
            train_robust="non-default action never worse than the label-free default for ANY TRAIN tick-0 member of the "
                         "observable class, and strictly better for at least one",
            delta_sign="mean_delta = selected - default (negative = better)",
            bound="exact one-sided 95% Clopper-Pearson upper bound on the harm proportion, treating states as "
                  "independent draws; descriptive only, NOT a population safety claim"),
        train_phase=dict(regimes_in_train=sorted(f"t_h={a},p={b}" for a, b in train_regime_set), per_class=train_report,
                         frozen_sets={t: dict(default=v["default"], actions=v["actions"], names=[names[a] for a in v["actions"]])
                                      for t, v in sorted(frozen.items())},
                         frozen_sets_sha256=frozen_sha,
                         other_roles_seen_only_as_split_role=other_train_pass),
        validation_phase=dict(regimes_in_validation=[f"t_h={a},p={b}" for a, b in val_regimes],
                              regimes_seen_in_train=[f"t_h={a},p={b}" for a, b in val_regimes if (a, b) in train_regime_set],
                              per_class=per_class, overall_worst_case_over_tie_breaks=overall_union, coverage=coverage,
                              other_roles_seen_only_as_split_role=other_val_pass),
        classification=dict(result=cls_, per_class_every_action_zero_harm=surv, per_class_some_action_zero_harm=surv_some,
                            rule="A: every class with a TRAIN-robust set keeps zero harm for every frozen action on all "
                                 "VALIDATION (incl. t_h=6 / p=0.8); B: some classes/actions keep zero harm; C: none do"),
        integrity=dict(
            dataset_sha256=tree_before["full_dataset/full_ai_learning_rows.csv.gz"], dataset_hash_verified=True,
            protected_hashes_verified=len(prot), model_hashes_verified=len(man["models"]),
            train_validation_trajectories_disjoint=True, train_validation_seeds_disjoint=True,
            robust_sets_built_from_train_only=True, validation_rows_read_after_freeze=True,
            primary_test_labels_read=False, primary_rows_seen_only_as_split_role=other_val_pass.get("TEST_PRIMARY_JOINT"),
            files_hashed=len(tree_before), files_changed_during_audit=changed,
            artifact_tree_hashes_before=tree_before))
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=float)
    print(json.dumps(dict(classification=cls_, frozen_sha=frozen_sha, coverage=coverage,
                          overall=overall_union), indent=1, default=float))


if __name__ == "__main__":
    main()
