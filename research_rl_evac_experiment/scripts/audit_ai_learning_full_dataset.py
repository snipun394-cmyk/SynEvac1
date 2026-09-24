"""READ-ONLY audit of the Phase-3 full AI-learning dataset.

Equivalent to the pilot audit, streamed state by state (the dataset has
~5.5M rows). Applies the pre-registered G1-G7 / X1-X10 with the
definitions and thresholds from diversity_requirements.json (spec v2),
exactly as declared in full_generation_record.json BEFORE generation
(the only mapping: G7's pilot constant "2 per cell" -> the frozen
allocated count per cell; X7 evaluated literally over all trajectories).

Trains nothing, builds no selector, runs no closed-loop evaluation, writes
only the full-dataset audit deliverables, never modifies the dataset.
"""
from __future__ import annotations

import copy
import csv
import gzip
import hashlib
import io
import itertools
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)
import run_ai_learning_full_dataset as F  # noqa: E402
import run_ai_learning_pilot as P  # noqa: E402
import audit_ai_learning_pilot as A  # noqa: E402
import ai_learning_label_free_default as lfd  # noqa: E402
import audit_pre_training_ai_dataset as prior  # noqa: E402
import update_experiment_specification_v2 as v2  # noqa: E402

from actions import apply_action  # noqa: E402
from differential_targeting import apply_differential_action  # noqa: E402
from topology import MAX_STEPS  # noqa: E402

FULL_DIR = F.FULL_DIR
MARKER = "ai_learning_full_dataset_audit_v1"
OUTPUT_FILES = ["FULL_DATASET_AUDIT.md", "full_dataset_manifest.json", "full_diversity.json", "full_target_audit.json",
                "full_joint_grid.json", "full_leakage_audit.json", "full_continuation_audit.json", "full_go_no_go.json"]
FEATS = list(lfd.ADMISSIBLE_FEATURES)
TGT = ("future_T90", "future_T100", "future_unmet", "future_hazard_unmet")
IRES, IHT = FEATS.index("residual_cap_H_CE1_normalized"), FEATS.index("waiting_H_total")


def protected_hashes() -> dict:
    out = dict(A.protected_hashes())
    for n in sorted(os.listdir(P.PILOT_DIR)):
        out[f"PILOT/{n}"] = prior.sha256_file(os.path.join(P.PILOT_DIR, n))
    for n in (F.ROWS_FILE, F.GENERATION_RECORD):
        out[f"FULL/{n}"] = prior.sha256_file(os.path.join(FULL_DIR, n))
    for rel in ("scripts/run_ai_learning_full_dataset.py", "scripts/audit_ai_learning_pilot.py"):
        out[f"OUT_ROOT/{rel}"] = prior.sha256_file(os.path.join(prior.OUT_ROOT, rel))
    return out


def _safe_write(name, content, created):
    if name not in OUTPUT_FILES:
        raise PermissionError(name)
    path = os.path.join(FULL_DIR, name)
    if os.path.exists(path) and name not in created:
        with open(path, encoding="utf-8") as f:
            if MARKER not in f.read():
                raise PermissionError(path)
    text = content if isinstance(content, str) else json.dumps(content, indent=2, default=prior._json_default)
    assert MARKER in text
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    created.add(name)


def _num(v):
    if v in ("inf", "-inf", "nan", "Infinity"):
        return float(v)
    try:
        return int(v)
    except ValueError:
        return float(v)


# ===========================================================================
# Stream pass: schema, targets, aggregates, compact per-state summaries.
# ===========================================================================

def stream_pass(cand_names: list) -> dict:
    ix = None
    violations = Counter()
    states = []
    seen_keys = set()
    acc = {}          # x -> (298, 4, 3) [n, sum, sumsq]
    acc_post = {}     # x -> (298, 3) for T100, post-onset only
    post_trajs = defaultdict(set)
    feat_values = [Counter() for _ in FEATS]
    seqs = defaultdict(list)
    hashes = Counter()
    tc = Counter()
    cur, buf = None, []
    t0 = time.perf_counter()

    def process(rows):
        r0 = rows[0]
        key = (r0[ix["trajectory_id"]], int(r0[ix["decision_tick"]]))
        if key in seen_keys:
            violations["non_contiguous_state"] += 1
        seen_keys.add(key)
        th = r0[ix["t_h"]]
        t_h_int = MAX_STEPS + 1 if th == "none" else int(th)
        tick = key[1]
        xs = {tuple(float(r[ix[f]]) for f in FEATS) for r in rows}
        x = next(iter(xs))
        if len(xs) != 1:
            violations["features_differ_across_candidates"] += 1
        if len(x) != 26 or any(not math.isfinite(v) for v in x):
            violations["invalid_feature"] += 1
        if len(rows) != 298:
            violations["candidate_count_not_298"] += 1
        ids = [int(r[ix["candidate_id"]]) for r in rows]
        if ids != list(range(298)) or [r[ix["candidate_name"]] for r in rows] != cand_names:
            violations["candidate_set_mismatch"] += 1
        d = int(r0[ix["default_action"]])
        defaults = [r for r in rows if r[ix["is_default_candidate"]] == "1"]
        if len(defaults) != 1 or defaults[0][ix["candidate_name"]] != f"BASE={'APB'[d]}":
            violations["default_flag_invalid"] += 1
        cls_y = {}
        ys = []
        n_evac_rows = 0
        for r in rows:
            hashes[(r[ix["continuation_policy"]], r[ix["continuation_hash"]], r[ix["spec_version"]],
                    r[ix["seed_rule"]])] += 1
            y = tuple(_num(r[ix[k]]) for k in TGT)
            ys.append(y)
            tc["rows"] += 1
            if any(not math.isfinite(v) for v in y) or any(v < 0 for v in y):
                violations["invalid_target"] += 1
            if r[ix["T100_censored"]] == "1":
                violations["censored"] += 1
            t100a, t90a = _num(r[ix["T100_abs"]]), _num(r[ix["T90_abs"]])
            if y[1] != t100a - tick:
                violations["T100_semantics"] += 1
            reached = r[ix["T90_reached_before_decision"]] == "1"
            tc["T90_reached_rows"] += int(reached)
            if (y[0] != 0) if reached else (y[0] != t90a - tick):
                violations["T90_semantics"] += 1
            if t_h_int > MAX_STEPS:
                tc["no_hazard_rows"] += 1
                if y[3] != 0:
                    violations["hazard_nonzero_without_hazard"] += 1
            elif t100a < t_h_int:
                tc["ends_before_onset_rows"] += 1
                if y[3] != 0:
                    violations["hazard_nonzero_before_onset"] += 1
            c = r[ix["equivalence_class"]]
            if c in cls_y and cls_y[c] != y:
                violations["class_outcome_inconsistent"] += 1
            cls_y.setdefault(c, y)
        # aggregates
        a = acc.get(x)
        if a is None:
            a = acc[x] = np.zeros((298, 4, 3))
        Y = np.asarray(ys, dtype=float)
        a[:, :, 0] += 1
        a[:, :, 1] += Y
        a[:, :, 2] += Y ** 2
        traj = key[0]
        post = t_h_int <= MAX_STEPS and tick >= t_h_int
        if post:
            p_ = acc_post.get(x)
            if p_ is None:
                p_ = acc_post[x] = np.zeros((298, 3))
            p_[:, 0] += 1
            p_[:, 1] += Y[:, 1]
            p_[:, 2] += Y[:, 1] ** 2
            post_trajs[x].add(traj)
        for i, v in enumerate(x):
            feat_values[i][v] += 1
        seqs[traj].append(x)
        dy = ys[[i for i, r in enumerate(rows) if r[ix["is_default_candidate"]] == "1"][0]] if defaults else ys[0]
        rep_ids, seen = [], set()
        for r in rows:
            c = r[ix["equivalence_class"]]
            if c not in seen:
                seen.add(c)
                rep_ids.append(int(r[ix["candidate_id"]]))
        states.append(dict(
            traj=traj, role=r0[ix["split_role"]], template=r0[ix["template"]], cell=r0[ix["cell"]], tick=tick,
            t_h=th, t_h_int=t_h_int, p=float(r0[ix["p"]]), seed=int(r0[ix["seed"]]),
            history=[int(v) for v in r0[ix["pre_decision_action_history"]].split(",") if v != ""],
            perturbed=[v == "1" for v in r0[ix["pre_decision_perturbed"]].split(",") if v != ""],
            pre_unmet=int(r0[ix["pre_decision_unmet"]]), fp=r0[ix["state_fingerprint"]],
            n_nonempty=int(r0[ix["n_nonempty_eligible_zones"]]), default_action=d, x=x, best_y=min(ys),
            default_y=dy, rep_ids=rep_ids, n_distinct_y=len(set(ys)),
            tied_default=sum(1 for y in ys if y == dy) / len(ys), n_classes=len(cls_y), post=post))

    with gzip.open(os.path.join(FULL_DIR, F.ROWS_FILE), "rt", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        if header != F.HEADER:
            violations["header_mismatch"] += 1
        ix = {c: i for i, c in enumerate(header)}
        for r in reader:
            k = (r[ix["trajectory_id"]], r[ix["decision_tick"]])
            if k != cur and buf:
                process(buf)
                buf = []
            cur = k
            buf.append(r)
        if buf:
            process(buf)
    return dict(header=header, violations=dict(violations), states=states, acc=acc, acc_post=acc_post,
                post_trajs=post_trajs, feat_values=feat_values, seqs=seqs, hashes=dict(hashes), counts=dict(tc),
                seconds=round(time.perf_counter() - t0, 1))


def _shares(groups_n_sum_sq: list) -> float | None:
    n = sum(g[0] for g in groups_n_sum_sq)
    s = sum(g[1] for g in groups_n_sum_sq)
    sq = sum(g[2] for g in groups_n_sum_sq)
    if n < 2:
        return None
    sst = sq - s * s / n
    ssw = sum(g[2] - (g[1] ** 2) / g[0] for g in groups_n_sum_sq if g[0] > 0)
    return round(ssw / sst, 4) if sst > 1e-12 else 0.0


# ===========================================================================
# Simulation pass: replay, relabel invariance, pre-onset counterfactual,
# decomposition, candidate semantics.
# ===========================================================================

def simulation_pass(states: list, trajs: dict, fs: dict) -> dict:
    c = Counter()
    by_id = {k["id"]: k for k in fs["candidates"]}
    t0 = time.perf_counter()
    for i, s in enumerate(states):
        tr = trajs[s["traj"]]
        sim, pre = A._replay_to(tr, s["history"], s["tick"])
        c["states"] += 1
        c["replay_features_match"] += int(tuple(P.features(sim).tolist()) == s["x"])
        c["replay_fingerprint_match"] += int(P.fingerprint(sim)[:16] == s["fp"])
        c["pre_decision_unmet_match"] += int(pre == s["pre_unmet"])
        c["residual_ok"] += int(abs(s["x"][IRES] - (1.0 if s["t_h_int"] > s["tick"] else 2.0 / 6.0)) < 1e-12)
        c["default_equals_rule"] += int(lfd.label_free_default_action(np.asarray(s["x"])) == s["default_action"])
        c["default_equals_closed_form"] += int(s["default_action"] == (2 if (s["x"][IRES] < 1 and s["x"][IHT] >= 1) else 1))
        for cid in s["rep_ids"]:
            asg = P.restrict(sim, by_id[cid]["asg"])
            fork = copy.deepcopy(sim)
            pairs = [(z, v) for z, v in asg.items() if v is not None]
            if pairs:
                apply_differential_action(fork, pairs)
            c["class_relabels_checked"] += 1
            c["features_unchanged_by_relabel"] += int(tuple(P.features(fork).tolist()) == s["x"])
            if i % 4 == 0:
                c["label_map_checked"] += 1
                c["fast_label_map_equals_apply"] += int(P.label_map(sim, asg) == tuple(
                    o["target_exit"] for _, o in sorted(fork.occupants.items())))
        if s["t_h_int"] <= MAX_STEPS and s["tick"] < s["t_h_int"]:
            cf, _ = A._replay_to(tr, s["history"], s["tick"], t_h_override=MAX_STEPS + 1)
            c["pre_onset_states"] += 1
            c["pre_onset_features_identical_without_hazard"] += int(tuple(P.features(cf).tolist()) == s["x"])
        if i % 4 == 0:
            g = copy.deepcopy(sim)
            apply_action(g, s["default_action"])
            dasg = P.restrict(sim, {z: s["default_action"] for z in P.ELIGIBLE})
            c["default_states_checked"] += 1
            c["default_candidate_equals_global_apply"] += int(P.label_map(sim, dasg) == tuple(
                o["target_exit"] for _, o in sorted(g.occupants.items())))
        if s["tick"] > 0:
            st = dict(traj=tr, tick=s["tick"], sim=sim, fp=P.fingerprint(sim), n_evacuated=len(sim.evacuated))
            o = P.rollout(st, P.restrict(sim, {z: s["default_action"] for z in P.ELIGIBLE}))
            c["decomposition_states"] += 1
            c["decomposition_exact"] += int(pre == s["pre_unmet"] and o["future_unmet"] == s["default_y"][2])
    c_ = dict(c)
    c_["seconds"] = round(time.perf_counter() - t0, 1)
    return c_


def regeneration_with_trap(gen_record: dict) -> dict:
    import deadband_heuristic as dh
    import capacity_aware_heuristic as cah

    class Trap(RuntimeError):
        pass

    def boom(*a, **k):
        raise Trap("hidden-label heuristic called")
    saved = (prior.asgc.HEURISTIC, dh.DeadbandCapacityAwareHeuristic.__call__, dh.DeadbandCapacityAwareHeuristic.predict,
             cah.heuristic_action)
    prior.asgc.HEURISTIC = boom
    dh.DeadbandCapacityAwareHeuristic.__call__ = boom
    dh.DeadbandCapacityAwareHeuristic.predict = boom
    cah.heuristic_action = boom
    sha = hashlib.sha256()
    b = io.StringIO()
    csv.writer(b, lineterminator="\n").writerow(F.HEADER)
    sha.update(b.getvalue().encode("utf-8"))

    def sink(rows):
        bb = io.StringIO()
        w = csv.writer(bb, lineterminator="\n")
        for r in rows:
            w.writerow(r)
        sha.update(bb.getvalue().encode("utf-8"))
    t0 = time.perf_counter()
    try:
        res = F.generate(F.APPROVED_SEED_RULE, sink=sink)
        ok = True
    except Trap:
        res, ok = None, False
    finally:
        prior.asgc.HEURISTIC, dh.DeadbandCapacityAwareHeuristic.__call__, dh.DeadbandCapacityAwareHeuristic.predict, \
            cah.heuristic_action = saved
    return dict(completed_with_hidden_heuristic_disabled=ok, regenerated_sha256=sha.hexdigest() if ok else None,
                recorded_sha256=gen_record["rows_uncompressed_sha256"],
                identical_rows=bool(ok and sha.hexdigest() == gen_record["rows_uncompressed_sha256"]),
                checks=res["checks"] if res else None, continuation_calls=res["continuation_calls"] if res else None,
                seconds=round(time.perf_counter() - t0, 1))


def g3_recheck(states: list, trajs: dict) -> dict:
    order = sorted(states, key=lambda s: (-s["n_nonempty"], s["traj"], s["tick"]))[:40]
    matches, regrets, rows = 0, [], []
    for s in order:
        tr = trajs[s["traj"]]
        sim, _ = A._replay_to(tr, s["history"], s["tick"])
        st = dict(traj=tr, tick=s["tick"], sim=sim, fp=P.fingerprint(sim), n_evacuated=len(sim.evacuated))
        zs = [z for z in P.ELIGIBLE if len(sim.zone_waiting[z]) > 0]
        cache = {}
        for vals in itertools.product([None, 0, 1, 2], repeat=len(zs)):
            asg = dict(zip(zs, vals))
            lm = P.label_map(sim, asg)
            if lm not in cache:
                o = P.rollout(st, asg)
                cache[lm] = (o["future_T90"], o["future_T100"], o["future_unmet"], o["future_hazard_unmet"])
        best = min(cache.values())
        matches += int(s["best_y"] == best)
        regrets.append(s["best_y"][1] - best[1])
        rows.append(dict(state=f"{s['traj']}|t{s['tick']}", n_nonempty=s["n_nonempty"], classes=len(cache),
                         base2_best=list(s["best_y"]), unrestricted_best=list(best)))
    return dict(n_states=len(order), matches=matches, share=round(matches / len(order), 4),
                mean_T100_regret=round(float(np.mean(regrets)), 4), max_T100_regret=float(np.max(regrets)),
                nonempty_zone_counts=dict(Counter(s["n_nonempty"] for s in order)), states=rows)


# ===========================================================================
# Assemble.
# ===========================================================================

def run() -> dict:
    before = protected_hashes()
    t0 = time.perf_counter()
    with open(os.path.join(FULL_DIR, F.GENERATION_RECORD), encoding="utf-8") as f:
        gen = json.load(f)
    if gen.get("status") != "COMPLETE":
        raise RuntimeError(f"generation status is {gen.get('status')}")
    fs = P.load_frozen_spec()
    trajs_list = F.full_trajectories(F.APPROVED_SEED_RULE)
    trajs = {t["trajectory_id"]: t for t in trajs_list}
    cand_names = [c["name"] for c in fs["candidates"]]
    sp = stream_pass(cand_names)
    states = sp["states"]
    print(f"stream pass: {len(states)} states, violations {sp['violations']} ({sp['seconds']}s)")
    sim = simulation_pass(states, trajs, fs)
    print(f"simulation pass done ({sim['seconds']}s)")
    regen = regeneration_with_trap(gen)
    print(f"regeneration identical {regen['identical_rows']} ({regen['seconds']}s)")
    g3 = g3_recheck(states, trajs)
    print(f"G3 {g3['matches']}/{g3['n_states']}")
    old = A.old_dataset_reference()

    # ---------------- diversity ----------------
    xs = [s["x"] for s in states]
    distinct = set(xs)
    by = lambda k: {kk: len({s["x"] for s in states if s[k] == kk}) for kk in sorted({s[k] for s in states})}  # noqa: E731
    late = [s for s in states if s["tick"] >= 10]
    late_tpl = defaultdict(set)
    for s in late:
        late_tpl[s["template"]].add(s["x"])
    seq_counts = Counter(tuple(v) for v in sp["seqs"].values())
    ent = {f: round(A._entropy_bits(list(sp["feat_values"][i].elements())), 4) for i, f in enumerate(FEATS)}
    cond = {}
    for ti, t in enumerate(TGT):
        g_state = [(a[:, ti, 0].sum(), a[:, ti, 1].sum(), a[:, ti, 2].sum()) for a in sp["acc"].values()]
        g_sa = [(a[k, ti, 0], a[k, ti, 1], a[k, ti, 2]) for a in sp["acc"].values() for k in range(298)]
        cond[t] = dict(given_state=_shares(g_state), given_state_and_candidate=_shares(g_sa))
    n_states = len(states)
    div = dict(
        audit=MARKER, n_trajectories=len(sp["seqs"]), n_decision_states=n_states, n_candidate_rows=sp["counts"]["rows"],
        n_unique_feature_vectors=len(distinct), distinct_share=round(len(distinct) / n_states, 4),
        n_unique_state_candidate_pairs=len(distinct) * 298,
        unique_pairs_note="every state carries all 298 candidates, so unique (input, candidate) pairs = unique inputs x 298",
        unique_states_by_tick={str(k): dict(states=sum(1 for s in states if s["tick"] == k), unique=v)
                               for k, v in by("tick").items()},
        unique_states_by_template=by("template"), unique_states_by_role=by("role"),
        tick_ge_10=dict(states=len(late), unique=len({s["x"] for s in late}),
                        share=round(len({s["x"] for s in late}) / max(1, len(late)), 4),
                        unique_by_template={k: len(v) for k, v in sorted(late_tpl.items())}),
        repeated_state_fraction=round(1 - len(distinct) / n_states, 4),
        repeated_trajectory_fraction=round(sum(c for c in seq_counts.values() if c > 1) / len(sp["seqs"]), 4),
        feature_entropy_bits=ent, mean_feature_entropy_bits=round(float(np.mean(list(ent.values()))), 4),
        conditional_outcome_variance=cond,
        target_variation=dict(
            states_with_ge2_outcome_vectors=sum(1 for s in states if s["n_distinct_y"] >= 2),
            states_with_ge3_outcome_vectors=sum(1 for s in states if s["n_distinct_y"] >= 3),
            share_ge2=round(sum(1 for s in states if s["n_distinct_y"] >= 2) / n_states, 4),
            share_ge3=round(sum(1 for s in states if s["n_distinct_y"] >= 3) / n_states, 4),
            tied_with_default_mean=round(float(np.mean([s["tied_default"] for s in states])), 4),
            classes_per_state_mean=round(float(np.mean([s["n_classes"] for s in states])), 2)),
        old_dataset=old, pilot_reference=dict(unique_inputs=183, distinct_share=0.2299, pairs=54534))

    # ---------------- joint grid ----------------
    dd = A.prior_json("dataset_design.json")
    sps = A.prior_json("split_specification.json")
    present = Counter(s["traj"] for s in states)
    expected_cells = Counter((t["split_role"], t["template"], t["cell"]) for t in trajs_list)
    present_cells = Counter((trajs[t]["split_role"], trajs[t]["template"], trajs[t]["cell"]) for t in present)
    missing_trajs = [t for t in trajs if t not in present]
    seeds = [t["seed"] for t in trajs_list]
    seed_by_role = defaultdict(list)
    for t in trajs_list:
        seed_by_role[t["split_role"]].append(t["seed"])
    rng = {r: tuple(int(v) for v in dd["seed_ranges_by_role"][r].split(" ")[0].split("-")) for r in dd["seed_ranges_by_role"]}
    rng_eff = dict(rng, TRAIN=(100000, 101799))
    rng_eff["TEST_A_RESERVED"] = (125000, 125999)
    overlaps = [(a, b) for a, b in itertools.combinations(rng_eff, 2)
                if not (rng_eff[a][1] < rng_eff[b][0] or rng_eff[b][1] < rng_eff[a][0])]
    inside = {r: all(rng_eff[r][0] <= s <= rng_eff[r][1] for s in v) for r, v in seed_by_role.items()}
    matrix = defaultdict(Counter)
    for t in trajs_list:
        matrix[t["split_role"]][f"t_h={t['t_h']}|p={t['p']}"] += 1
    train_th = {t["t_h"] for t in trajs_list if t["split_role"] == "TRAIN"}
    train_p = {t["p"] for t in trajs_list if t["split_role"] == "TRAIN"}
    prim = [t for t in trajs_list if t["split_role"] == "TEST_PRIMARY_JOINT"]
    zs_tpl = set(dd["factors"]["templates_zero_shot_only"])
    x4_design = A.joint_grid({t["trajectory_id"]: t for t in P.trajectories(fs["pilot"])}, fs)["primary_test_constructible"]
    grid = dict(
        audit=MARKER,
        n_trajectories_expected=len(trajs_list), n_trajectories_present=len(present), missing_trajectories=missing_trajs[:20],
        allocation_by_role=dict(Counter(t["split_role"] for t in trajs_list)),
        all_cells_complete=bool(not missing_trajs and present_cells == expected_cells),
        n_cells=len(expected_cells),
        timing_x_compliance_by_role={r: dict(sorted(m.items())) for r, m in matrix.items()},
        seeds=dict(n=len(seeds), unique=len(set(seeds)), all_unique=len(set(seeds)) == len(seeds),
                   by_role={r: dict(n=len(v), min=min(v), max=max(v), inside_effective_range=inside[r])
                            for r, v in seed_by_role.items()},
                   effective_ranges={k: f"{a}-{b}" for k, (a, b) in rng_eff.items()},
                   documented_TRAIN_range=dd["seed_ranges_by_role"]["TRAIN"],
                   range_overlaps=overlaps, pilot_seeds_reused=bool(set(seeds) & set(range(190000, 191000))),
                   test_A_range_untouched=not (set(seeds) & set(range(125000, 126000)))),
        held_out=dict(
            primary_trajectories=len(prim),
            primary_t_h_values=sorted({t["t_h"] for t in prim}), primary_p_values=sorted({t["p"] for t in prim}),
            primary_t_h_absent_from_train=not ({t["t_h"] for t in prim} & train_th),
            primary_p_absent_from_train=not ({t["p"] for t in prim} & train_p),
            zero_shot_templates_only_in_zero_shot_role=all((t["template"] in zs_tpl) == (t["split_role"] == F.ZERO_SHOT_ROLE)
                                                           for t in trajs_list),
            compliance_keys_role_specific=all(t["compliance_key"].startswith(f"AILX_v1|{t['split_role']}|") for t in trajs_list),
            trajectory_in_exactly_one_role=len({t["trajectory_id"] for t in trajs_list}) == len(trajs_list)),
        association=dict(
            all_trajectories=dict(template_vs_t_h=A._cramers_v([(t["template"], t["t_h"]) for t in trajs_list]),
                                  template_vs_p=A._cramers_v([(t["template"], t["p"]) for t in trajs_list])),
            six_main_templates=dict(
                template_vs_t_h=A._cramers_v([(t["template"], t["t_h"]) for t in trajs_list if t["template"] not in zs_tpl]),
                template_vs_p=A._cramers_v([(t["template"], t["p"]) for t in trajs_list if t["template"] not in zs_tpl])),
            every_template_has_every_t_h_and_p_level=all(
                len({t["t_h"] for t in trajs_list if t["template"] == k}) == 9
                and len({t["p"] for t in trajs_list if t["template"] == k}) == 7
                for k in {t["template"] for t in trajs_list})),
        primary_test_constructible_design=x4_design)
    x4_ok = bool(x4_design["passed"] and grid["held_out"]["primary_t_h_absent_from_train"]
                 and grid["held_out"]["primary_p_absent_from_train"] and len(prim) == 960
                 and grid["held_out"]["zero_shot_templates_only_in_zero_shot_role"]
                 and grid["held_out"]["compliance_keys_role_specific"] and not overlaps)

    # ---------------- leakage / targets / continuation ----------------
    v = sp["violations"]
    fcols = F.HEADER[18:18 + 26]
    forbidden = {"Q_A_at_H", "Q_B_at_H", "compliant", "compliance", "hazard_phase", "time_since_hazard_onset_normalized",
                 "target_exit"}
    leak_ok = bool(fcols == FEATS and not (set(fcols) & forbidden) and v.get("features_differ_across_candidates", 0) == 0
                   and sim["replay_features_match"] == sim["states"] and sim["replay_fingerprint_match"] == sim["states"]
                   and sim["features_unchanged_by_relabel"] == sim["class_relabels_checked"]
                   and sim["pre_onset_features_identical_without_hazard"] == sim["pre_onset_states"]
                   and sim["residual_ok"] == sim["states"])
    tick0 = defaultdict(set)
    for s in states:
        if s["tick"] == 0:
            tick0[s["template"]].add(s["x"])
    leak = dict(audit=MARKER, passed=leak_ok, feature_columns_exact=fcols == FEATS,
                forbidden_feature_columns=sorted(set(fcols) & forbidden), stream_violations=v, simulation_checks=sim,
                tick0_input_independent_of_t_h_p_and_seed=all(len(x) == 1 for x in tick0.values()),
                metadata_columns_not_features=["split_role", "template", "distribution", "cell", "t_h", "p", "seed",
                                               "replicate", "compliance_key", "pre_decision_*", "hazard_started"])
    tgt_ok = bool(not any(v.get(k, 0) for k in ("invalid_target", "censored", "T100_semantics", "T90_semantics",
                                                  "hazard_nonzero_without_hazard", "hazard_nonzero_before_onset",
                                                  "class_outcome_inconsistent"))
                  and sim["decomposition_exact"] == sim["decomposition_states"]
                  and sim["pre_decision_unmet_match"] == sim["states"])
    tgt = dict(audit=MARKER, passed=tgt_ok, counts=sp["counts"], violations=v,
               pre_decision_unmet_excluded=dict(states_checked=sim["decomposition_states"],
                                                exact=sim["decomposition_exact"]),
               hazard_tick_log=f"{gen['generation_checks']['hazard_tick_log']}/{gen['generation_checks']['n_classes']}",
               identical_start=f"{gen['generation_checks']['same_start']}/{gen['generation_checks']['n_classes']}",
               target_variation=div["target_variation"], g3_unrestricted_recheck=g3)
    src = {}
    for rel in ("run_ai_learning_full_dataset.py", "run_ai_learning_pilot.py", "ai_learning_label_free_default.py"):
        with open(os.path.join(SCRIPTS_DIR, rel), encoding="utf-8") as fh:
            code = "\n".join(l for l in fh.read().splitlines() if not l.strip().startswith(("#", '"', "'")))
        src[rel] = [tok for tok in ("HEURISTIC(", "DeadbandCapacityAwareHeuristic", "run_intervention", "Q_A_IDX",
                                    "Q_B_IDX", "asgc.") if tok in code]
    expected_hash = (lfd.POLICY_ID, lfd.policy_hash(), "v2", F.APPROVED_SEED_RULE)
    cc = gen["continuation_calls"]
    cont_ok = bool(set(sp["hashes"]) == {expected_hash} and not any(src.values())
                   and regen["completed_with_hidden_heuristic_disabled"] and cc["calls"] == cc["input_len_26"])
    sem_ok = bool(sim["default_equals_rule"] == sim["states"] and sim["default_equals_closed_form"] == sim["states"]
                  and sim["fast_label_map_equals_apply"] == sim["label_map_checked"]
                  and sim["default_candidate_equals_global_apply"] == sim["default_states_checked"]
                  and not v.get("candidate_set_mismatch") and not v.get("default_flag_invalid"))
    cont = dict(audit=MARKER, label_free_passed=cont_ok, semantics_passed=sem_ok,
                row_provenance_values=[list(k) + [n] for k, n in sp["hashes"].items()],
                expected=list(expected_hash), forbidden_tokens_in_generation_source=src,
                continuation_calls_generation=cc, regeneration=regen, hidden_heuristic_reference_arm_in_dataset=False)

    # ---------------- G6 ----------------
    multi = [x for x, t in sp["post_trajs"].items() if len(t) >= 2]
    post_rows = sum(a[:, 0].sum() for a in sp["acc_post"].values())
    cov_rows = sum(sp["acc_post"][x][:, 0].sum() for x in multi)
    g6_share = _shares([(sp["acc_post"][x][k, 0], sp["acc_post"][x][k, 1], sp["acc_post"][x][k, 2])
                        for x in multi for k in range(298)]) if multi else None
    cov = cov_rows / max(1, post_rows)
    g6 = dict(post_onset_rows=int(post_rows), coverage=round(float(cov), 4), T100_irreducible_share=g6_share,
              evaluable=bool(cov >= 0.10 and g6_share is not None), passed=bool(cov >= 0.10 and g6_share is not None
                                                                                   and g6_share < 0.6))

    # ---------------- decision ----------------
    thr = v2.OLD
    pert = [b for s in A._last_states(states) for b in s["perturbed"]]
    rate = sum(pert) / max(1, len(pert))
    assoc = grid["association"]["all_trajectories"]
    R = {
        "G1": (div["tick_ge_10"]["share"] >= 0.25 and min(div["tick_ge_10"]["unique_by_template"].values()) >= 10,
               dict(share=div["tick_ge_10"]["share"], min_unique_per_template=min(div["tick_ge_10"]["unique_by_template"].values()))),
        "G2": (div["target_variation"]["share_ge3"] >= 0.5, div["target_variation"]["share_ge3"]),
        "G3": (g3["share"] >= 0.95 and g3["mean_T100_regret"] <= 0.1,
               dict(share=g3["share"], mean_T100_regret=g3["mean_T100_regret"], nonempty=g3["nonempty_zone_counts"])),
        "G4": (regen["identical_rows"] and gen["generation_checks"]["hazard_tick_log"] == gen["generation_checks"]["n_classes"],
               dict(identical_rows=regen["identical_rows"], hazard_tick_log=tgt["hazard_tick_log"])),
        "G5": (leak_ok, "see full_leakage_audit.json"),
        "G6": (g6["passed"], dict(share=g6["T100_irreducible_share"], coverage=g6["coverage"])),
        "G7": (grid["all_cells_complete"], dict(cells=grid["n_cells"], trajectories=grid["n_trajectories_present"])),
        "X1": (tgt_ok, tgt["pre_decision_unmet_excluded"]),
        "X2": (leak_ok, "= G5"),
        "X3": (gen["generation_checks"]["same_start"] == gen["generation_checks"]["n_classes"], tgt["identical_start"]),
        "X4": (x4_ok, dict(primary_trajectories=len(prim), held_out=grid["held_out"], overlaps=overlaps)),
        "X5": (div["n_unique_feature_vectors"] >= 10 * thr["distinct_inputs"]
               and div["distinct_share"] >= round(5 * thr["distinct_inputs"] / thr["states"], 4)
               and div["n_unique_state_candidate_pairs"] >= 10 * thr["distinct_input_action_pairs"],
               dict(unique_inputs=div["n_unique_feature_vectors"], distinct_share=div["distinct_share"],
                    pairs=div["n_unique_state_candidate_pairs"])),
        "X6": (div["target_variation"]["share_ge3"] >= 0.5, "= G2"),
        "X7": (assoc["template_vs_t_h"] == 0 and assoc["template_vs_p"] == 0 and grid["seeds"]["all_unique"]
               and abs(rate - 0.5) <= 0.1 and not v.get("censored"),
               dict(association_all_trajectories=assoc, association_six_main_templates=grid["association"]["six_main_templates"],
                    seeds_unique=grid["seeds"]["all_unique"], perturbation_rate=round(rate, 4),
                    censored_rows=v.get("censored", 0))),
        "X8": (tgt["pre_decision_unmet_excluded"]["exact"] == tgt["pre_decision_unmet_excluded"]["states_checked"],
               "= X1 decomposition"),
        "X9": (sem_ok, {k: sim[k] for k in ("default_equals_rule", "default_equals_closed_form", "label_map_checked",
                                            "fast_label_map_equals_apply", "default_states_checked",
                                            "default_candidate_equals_global_apply")}),
        "X10": (cont_ok, dict(booby_trap=regen["completed_with_hidden_heuristic_disabled"], tokens=src)),
    }
    results = {k: dict(passed=bool(p), value=val) for k, (p, val) in R.items()}
    failed = [k for k, r in results.items() if not r["passed"]]
    dec = dict(audit=MARKER, criteria_source="diversity_requirements.json (spec v2), unchanged",
               application_declared_before_generation=gen["criteria_application_declared_before_generation"],
               results=results, failed=failed,
               decision="FULL DATASET GO" if not failed else "FULL DATASET NO-GO",
               deviations_from_spec_v2=[gen["user_decisions"]["seed_rule"]["documented_deviation"]],
               not_done=["no model trained", "no selector", "no closed-loop evaluation", "no Test A batch"])
    after = protected_hashes()
    changed = [k for k in before if before[k] != after[k]]
    if changed:
        raise RuntimeError(f"PROTECTED FILES CHANGED: {changed}")
    integrity = dict(n_files=len(before), all_unchanged=True, controller=prior.design.verify_controller_hash(),
                     checkpoints=prior.design.verify_checkpoint_hashes(), hashes=before)
    manifest = dict(audit=MARKER, **{k: gen[k] for k in (
        "rows_file", "rows_file_sha256", "rows_uncompressed_sha256", "n_rows", "n_columns", "header", "n_trajectories",
        "n_decision_states", "generation_checks", "continuation_calls", "provenance", "user_decisions",
        "criteria_application_declared_before_generation", "spec_v2_hashes_verified", "seed_report")},
        generation_record_sha256=prior.sha256_file(os.path.join(FULL_DIR, F.GENERATION_RECORD)),
        integrity=integrity)
    ctx = dict(div=div, grid=grid, leak=leak, tgt=tgt, cont=cont, g6=g6, dec=dec, manifest=manifest,
               seconds=round(time.perf_counter() - t0, 1))
    ctx["outputs"] = {"full_dataset_manifest.json": manifest, "full_diversity.json": div,
                      "full_target_audit.json": dict(tgt, g6=g6), "full_joint_grid.json": grid,
                      "full_leakage_audit.json": leak, "full_continuation_audit.json": cont, "full_go_no_go.json": dec}
    ctx["outputs"]["FULL_DATASET_AUDIT.md"] = render(ctx)
    return ctx


def render(ctx) -> str:
    div, grid, leak, tgt, cont, g6, dec, man = (ctx[k] for k in ("div", "grid", "leak", "tgt", "cont", "g6", "dec", "manifest"))
    L = [f"<!-- {MARKER} -->", "# Full AI-Learning Dataset (Phase 3) Audit\n",
         "No model trained, no selector, no closed-loop evaluation. G1-G7 / X1-X10 applied with the spec v2 "
         "definitions and thresholds, as declared in full_generation_record.json before generation.\n",
         f"## Decision: **{dec['decision']}**" + (f" (failed: {', '.join(dec['failed'])})" if dec["failed"] else "") + "\n",
         "| criterion | result | value |", "|---|---|---|"]
    for k, r in dec["results"].items():
        L.append(f"| {k} | {'PASS' if r['passed'] else 'FAIL'} | {json.dumps(r['value'], default=str)[:240]} |")
    L += ["\nDeviation from spec v2: " + "; ".join(dec["deviations_from_spec_v2"]) + ".\n",
          "## Scale\n",
          f"{man['n_trajectories']} trajectories, {man['n_decision_states']:,} decision states, {man['n_rows']:,} candidate "
          f"rows, {man['generation_checks']['n_classes']:,} unique rollouts, {div['n_unique_feature_vectors']:,} unique "
          f"observable inputs, {div['n_unique_state_candidate_pairs']:,} unique (input, candidate) pairs.\n",
          "## Diversity\n",
          f"Distinct share {div['distinct_share']:.1%}; tick>=10 {div['tick_ge_10']}; by tick {div['unique_states_by_tick']}; "
          f"by template {div['unique_states_by_template']}; by role {div['unique_states_by_role']}; repeated-state "
          f"{div['repeated_state_fraction']:.1%}, repeated-trajectory {div['repeated_trajectory_fraction']:.1%}; mean "
          f"feature entropy {div['mean_feature_entropy_bits']} bits (old {div['old_dataset']['mean_feature_entropy_bits']}).\n",
          f"Target variation: {div['target_variation']}. Conditional variance: {div['conditional_outcome_variance']}. "
          f"G6: {g6}.\n",
          "## Joint grid and seeds\n",
          f"Allocation {grid['allocation_by_role']}; all {grid['n_cells']} cells complete {grid['all_cells_complete']}; "
          f"seeds {grid['seeds']['unique']}/{grid['seeds']['n']} unique; effective ranges {grid['seeds']['effective_ranges']}; "
          f"overlaps {grid['seeds']['range_overlaps']}; pilot seeds reused {grid['seeds']['pilot_seeds_reused']}; Test A "
          f"range untouched {grid['seeds']['test_A_range_untouched']}; held-out {grid['held_out']}; association "
          f"{grid['association']}.\n",
          "## Leakage\n", f"passed {leak['passed']}; {leak['simulation_checks']}; stream violations {leak['stream_violations']}.\n",
          "## Targets\n", f"passed {tgt['passed']}; decomposition {tgt['pre_decision_unmet_excluded']}; tick-log "
          f"{tgt['hazard_tick_log']}; identical starts {tgt['identical_start']}; G3 {tgt['g3_unrestricted_recheck']['matches']}/40.\n",
          "## Continuation\n", f"label-free {cont['label_free_passed']}; semantics {cont['semantics_passed']}; provenance "
          f"{cont['row_provenance_values']}; regeneration identical {cont['regeneration']['identical_rows']} with hidden "
          f"heuristic disabled.\n",
          "## Integrity\n", f"{man['integrity']['n_files']} protected files unchanged: {man['integrity']['all_unchanged']}; "
          f"controller {man['integrity']['controller']['match']}; checkpoints {man['integrity']['checkpoints']['all_match']}."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ctx = run()
    created = set()
    for n in OUTPUT_FILES:
        _safe_write(n, ctx["outputs"][n], created)
    print(f"\n{ctx['dec']['decision']} failed={ctx['dec']['failed']} ({ctx['seconds']}s)")
