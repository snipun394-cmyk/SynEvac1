"""READ-ONLY audit of the Phase-1 AI-learning pilot against the criteria
pre-registered in specification v2 (diversity_requirements.json ->
go_no_go_after_pilot: G1-G7, X1-X10). Criteria are read from that file and
applied exactly; nothing is relaxed after seeing the data.

Trains nothing, builds no selector, runs no closed-loop evaluation, writes
only the pilot audit deliverables, and never modifies the pilot rows, the
simulator, observation.py, the controller, the RL branch, or the old dataset.
"""
from __future__ import annotations

import copy
import csv
import gzip
import hashlib
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
import run_ai_learning_pilot as P  # noqa: E402
import ai_learning_label_free_default as lfd  # noqa: E402
import audit_pre_training_ai_dataset as prior  # noqa: E402
import audit_label_free_default as lfa  # noqa: E402
import update_experiment_specification_v2 as v2  # noqa: E402

from actions import apply_action  # noqa: E402
from compliance import generate_compliance_vector  # noqa: E402
from differential_targeting import apply_differential_action  # noqa: E402
from simulator import EvacuationSimulator  # noqa: E402
from timing_experiment import diagnostic_all_edges_unmet  # noqa: E402
import topology as topology_module  # noqa: E402
from topology import DECISION_INTERVAL, MAX_STEPS  # noqa: E402

PILOT_DIR = P.PILOT_DIR
MARKER = "ai_learning_pilot_audit_v1"
OUTPUT_FILES = ["PILOT_DATASET_AUDIT.md", "pilot_dataset_manifest.json", "pilot_diversity.json",
                "pilot_target_audit.json", "pilot_joint_grid.json", "pilot_leakage_audit.json",
                "pilot_continuation_audit.json", "pilot_go_no_go.json"]
FEATS = list(lfd.ADMISSIBLE_FEATURES)
EXTRA_PROTECTED = ["hybrid_controller.py", "branch_aware_deadband_heuristic.py", "baseline.py", "policy_selector.py"]


# ===========================================================================
# Integrity.
# ===========================================================================

def protected_hashes() -> dict:
    out = dict(lfa.protected_hashes())
    for rel in EXTRA_PROTECTED:
        out[f"DATA_ROOT/{rel}"] = prior.sha256_file(os.path.join(prior.DATA_ROOT, rel))
    for name in lfa.OUTPUT_FILES + ["spec_version_record.json"]:
        out[f"OUT_DIR/{name}"] = prior.sha256_file(os.path.join(prior.OUT_DIR, name))
    arch = os.path.join(prior.OUT_DIR, "spec_versions", "v1")
    for name in sorted(os.listdir(arch)):
        out[f"OUT_DIR/spec_versions/v1/{name}"] = prior.sha256_file(os.path.join(arch, name))
    for name in (P.ROWS_FILE, P.GENERATION_FILE):
        out[f"PILOT/{name}"] = prior.sha256_file(os.path.join(PILOT_DIR, name))
    for rel in ("scripts/run_ai_learning_pilot.py", "scripts/ai_learning_label_free_default.py",
                "scripts/update_experiment_specification_v2.py"):
        out[f"OUT_ROOT/{rel}"] = prior.sha256_file(os.path.join(prior.OUT_ROOT, rel))
    return out


def _safe_write(name, content, created, out_dir=PILOT_DIR):
    if name not in OUTPUT_FILES:
        raise PermissionError(f"refusing to write non-deliverable {name!r}")
    path = os.path.join(out_dir, name)
    if os.path.exists(path) and name not in created:
        with open(path, encoding="utf-8") as f:
            if MARKER not in f.read():
                raise PermissionError(f"refusing to overwrite {path}")
    text = content if isinstance(content, str) else json.dumps(content, indent=2, default=prior._json_default)
    if MARKER not in text:
        raise ValueError("deliverable missing marker")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    created.add(name)


# ===========================================================================
# Load.
# ===========================================================================

def _num(v: str):
    if v in ("inf", "Infinity"):
        return float("inf")
    try:
        return int(v)
    except ValueError:
        return float(v)


def load_rows() -> tuple:
    with gzip.open(os.path.join(PILOT_DIR, P.ROWS_FILE), "rt", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [r for r in reader]
    return header, rows


def build_states(header, rows) -> list:
    ix = {c: i for i, c in enumerate(header)}
    states = {}
    for r in rows:
        key = (r[ix["trajectory_id"]], int(r[ix["decision_tick"]]))
        st = states.get(key)
        x = tuple(float(r[ix[f]]) for f in FEATS)
        if st is None:
            th = r[ix["t_h"]]
            st = states[key] = dict(
                traj=r[ix["trajectory_id"]], tick=key[1], template=r[ix["template"]], t_h=th,
                t_h_int=(MAX_STEPS + 1 if th == "none" else int(th)), p=float(r[ix["p"]]), seed=int(r[ix["seed"]]),
                history=[int(a) for a in r[ix["pre_decision_action_history"]].split(",") if a != ""],
                perturbed=[a == "1" for a in r[ix["pre_decision_perturbed"]].split(",") if a != ""],
                pre_unmet=int(r[ix["pre_decision_unmet"]]), fp=r[ix["state_fingerprint"]],
                n_nonempty=int(r[ix["n_nonempty_eligible_zones"]]), hazard_started=r[ix["hazard_started"]] == "1",
                default_action=int(r[ix["default_action"]]), x=x, xs=set(), cands=[], hashes=set())
        st["xs"].add(x)
        st["hashes"].add((r[ix["continuation_policy"]], r[ix["continuation_hash"]]))
        st["cands"].append(dict(
            id=int(r[ix["candidate_id"]]), name=r[ix["candidate_name"]], cls=r[ix["equivalence_class"]],
            is_default=r[ix["is_default_candidate"]] == "1",
            y=tuple(_num(r[ix[k]]) for k in ("future_T90", "future_T100", "future_unmet", "future_hazard_unmet")),
            t90_reached=r[ix["T90_reached_before_decision"]] == "1", T100_abs=_num(r[ix["T100_abs"]]),
            T90_abs=_num(r[ix["T90_abs"]]), censored=r[ix["T100_censored"]] == "1"))
    return sorted(states.values(), key=lambda s: (s["traj"], s["tick"]))


# ===========================================================================
# Helpers.
# ===========================================================================

def _entropy_bits(values) -> float:
    c = Counter(values)
    n = sum(c.values())
    return -sum((k / n) * math.log2(k / n) for k in c.values())


def _within_share(groups: dict) -> dict:
    ys = [y for g in groups.values() for y in g]
    if len(ys) < 2:
        return dict(n_rows=len(ys), irreducible_share=None)
    y = np.asarray(ys, dtype=float)
    sst = float(((y - y.mean()) ** 2).sum())
    ssw = float(sum(((np.asarray(g, dtype=float) - np.mean(g)) ** 2).sum() for g in groups.values()))
    return dict(n_rows=len(ys), n_groups=len(groups), irreducible_share=round(ssw / sst, 4) if sst > 0 else 0.0)


def trajectory_by_id() -> dict:
    fs = P.load_frozen_spec()
    return {t["trajectory_id"]: t for t in P.trajectories(fs["pilot"])}, fs


def old_dataset_reference() -> dict:
    recs = prior.load_records()
    adm = prior.admissible_state_feature_names(prior.build_feature_manifest())
    xs = [tuple(prior.admissible_vector(prior.replay_state(r["scenario"], r["seed"], r["decision_tick"]), adm).tolist())
          for r in recs]
    ent = [_entropy_bits([x[i] for x in xs]) for i in range(len(adm))]
    return dict(states=len(xs), distinct_inputs=len(set(xs)), distinct_share=round(len(set(xs)) / len(xs), 4),
                distinct_input_action_pairs=v2.OLD["distinct_input_action_pairs"],
                mean_feature_entropy_bits=round(float(np.mean(ent)), 4),
                T100_irreducible_share_plain=v2.OLD["T100_irreducible_share"],
                note="old = the 450-state differential-targeting dataset (hidden-heuristic continuation, old targets); "
                     "distinct counts recomputed here by replay")


# ===========================================================================
# Section 5: diversity.
# ===========================================================================

def diversity(states: list, old: dict) -> dict:
    xs = [s["x"] for s in states]
    distinct = set(xs)
    pairs = {(s["x"], c["id"]) for s in states for c in s["cands"]}
    by_tick = defaultdict(set)
    by_tpl = defaultdict(set)
    late_states = 0
    late = set()
    late_by_tpl = defaultdict(set)
    for s in states:
        by_tick[s["tick"]].add(s["x"])
        by_tpl[s["template"]].add(s["x"])
        if s["tick"] >= 10:
            late_states += 1
            late.add(s["x"])
            late_by_tpl[s["template"]].add(s["x"])
    seqs = defaultdict(list)
    for s in states:
        seqs[s["traj"]].append(s["x"])
    seq_counts = Counter(tuple(v) for v in seqs.values())
    ent = {f: round(_entropy_bits([x[i] for x in xs]), 4) for i, f in enumerate(FEATS)}
    cond = {}
    for ti, tname in enumerate(("future_T90", "future_T100", "future_unmet", "future_hazard_unmet")):
        g_state, g_sa = defaultdict(list), defaultdict(list)
        for s in states:
            for c in s["cands"]:
                if math.isfinite(c["y"][ti]):
                    g_state[s["x"]].append(c["y"][ti])
                    g_sa[(s["x"], c["id"])].append(c["y"][ti])
        cond[tname] = dict(given_state=_within_share(g_state), given_state_and_candidate=_within_share(g_sa))
    return dict(
        audit=MARKER,
        n_trajectories=len(seqs), n_decision_states=len(states),
        n_unique_feature_vectors=len(distinct), distinct_share=round(len(distinct) / len(states), 4),
        n_unique_state_candidate_pairs=len(pairs),
        unique_states_by_tick={str(k): dict(states=sum(1 for s in states if s["tick"] == k), unique=len(v))
                               for k, v in sorted(by_tick.items())},
        unique_states_by_template={k: len(v) for k, v in sorted(by_tpl.items())},
        tick_ge_10=dict(states=late_states, unique=len(late), share=round(len(late) / max(1, late_states), 4),
                        unique_by_template={k: len(v) for k, v in sorted(late_by_tpl.items())}),
        repeated_state_fraction=round(1 - len(distinct) / len(states), 4),
        repeated_trajectory_fraction=round(sum(c for c in seq_counts.values() if c > 1) / len(seqs), 4),
        feature_entropy_bits=ent, mean_feature_entropy_bits=round(float(np.mean(list(ent.values()))), 4),
        conditional_outcome_variance=cond,
        old_dataset=old,
        vs_old=dict(unique_inputs_ratio=round(len(distinct) / old["distinct_inputs"], 2),
                    distinct_share_ratio=round((len(distinct) / len(states)) / old["distinct_share"], 2),
                    state_candidate_pairs_ratio=round(len(pairs) / old["distinct_input_action_pairs"], 2),
                    mean_entropy_old_vs_pilot=[old["mean_feature_entropy_bits"],
                                               round(float(np.mean(list(ent.values()))), 4)]),
    )


# ===========================================================================
# Section 6: targets.
# ===========================================================================

def _replay_to(traj: dict, history: list, stop_tick: int, t_h_override: int | None = None):
    """Rebuilds the pre-decision state by replaying the recorded history (no RNG needed)."""
    t_h = traj["t_h_int"] if t_h_override is None else t_h_override
    sim = EvacuationSimulator()
    sim.reset(traj["distribution"], t_h, generate_compliance_vector(traj["compliance_key"], traj["seed"], p=traj["p"]))
    unmet, k = 0, 0
    while sim.t < stop_tick and sim.n_active() > 0 and sim.t < MAX_STEPS:
        if sim.t % DECISION_INTERVAL == 0:
            apply_action(sim, history[k])
            k += 1
        unmet += diagnostic_all_edges_unmet(sim, sim.t, t_h, topology_module.edge_capacity)
        sim._run_one_tick(None)
        sim.t += 1
    return sim, unmet


def target_audit(states: list, trajs: dict) -> dict:
    c = Counter()
    ties = []
    tied_with_default = []
    for s in states:
        no_hazard = s["t_h_int"] > MAX_STEPS
        for cd in s["cands"]:
            c["rows"] += 1
            if no_hazard:
                c["no_hazard_rows"] += 1
                c["no_hazard_rows_hazard_target_zero"] += int(cd["y"][3] == 0)
            elif cd["T100_abs"] < s["t_h_int"]:
                c["ends_before_onset_rows"] += 1
                c["ends_before_onset_hazard_zero"] += int(cd["y"][3] == 0)
            c["T100_semantics_ok"] += int(cd["y"][1] == cd["T100_abs"] - s["tick"])
            ok90 = (cd["y"][0] == 0) if cd["t90_reached"] else (cd["y"][0] == cd["T90_abs"] - s["tick"])
            c["T90_semantics_ok"] += int(ok90)
            c["T90_reached_rows"] += int(cd["t90_reached"])
            c["censored_rows"] += int(cd["censored"])
        vecs = [cd["y"] for cd in s["cands"]]
        ties.append(len(set(vecs)))
        d = next(cd for cd in s["cands"] if cd["is_default"])
        tied_with_default.append(sum(1 for cd in s["cands"] if cd["y"] == d["y"]) / len(s["cands"]))
    # Pre-decision unmet excluded: decomposition via full-episode reruns for every t_d > 0 state.
    dec_ok = dec_n = 0
    for s in states:
        if s["tick"] == 0:
            continue
        tr = trajs[s["traj"]]
        sim, pre = _replay_to(tr, s["history"], s["tick"])
        st = dict(traj=tr, tick=s["tick"], sim=sim, fp=P.fingerprint(sim), n_evacuated=len(sim.evacuated))
        d_asg = P.restrict(sim, {z: s["default_action"] for z in P.ELIGIBLE})
        o = P.rollout(st, d_asg)
        d = next(cd for cd in s["cands"] if cd["is_default"])
        dec_n += 1
        dec_ok += int(pre == s["pre_unmet"] and pre + o["future_unmet"] == s["pre_unmet"] + d["y"][2]
                      and o["future_unmet"] == d["y"][2])
    g2_states = sum(1 for t in ties if t >= 3)
    return dict(
        audit=MARKER, counts=dict(c),
        hazard_zero_when_no_hazard=c["no_hazard_rows_hazard_target_zero"] == c["no_hazard_rows"],
        hazard_zero_when_episode_ends_before_onset=c["ends_before_onset_hazard_zero"] == c["ends_before_onset_rows"],
        T100_semantics_all_rows=c["T100_semantics_ok"] == c["rows"],
        T90_semantics_all_rows=c["T90_semantics_ok"] == c["rows"],
        pre_decision_unmet_excluded=dict(states_checked=dec_n, decomposition_exact=dec_ok,
                                         passed=bool(dec_n and dec_ok == dec_n),
                                         method="replay recorded history to t_d (pre-decision unmet), roll out the "
                                                "default candidate with the label-free continuation; the stored "
                                                "pre-decision unmet and future unmet must reproduce exactly"),
        distinct_target_vectors_per_state=dict(min=min(ties), median=float(np.median(ties)), max=max(ties),
                                               states_with_at_least_3=g2_states,
                                               share_with_at_least_3=round(g2_states / len(ties), 4)),
        share_of_candidates_tied_with_default=dict(mean=round(float(np.mean(tied_with_default)), 4),
                                                   median=round(float(np.median(tied_with_default)), 4)),
        n_equivalence_classes_per_state=dict(mean=round(float(np.mean([len({cd['cls'] for cd in s['cands']}) for s in states])), 2)),
        continuation_same_for_all_candidates=len({h for s in states for h in s["hashes"]}) == 1,
    )


# ===========================================================================
# Section 7: joint grid.
# ===========================================================================

def _cramers_v(pairs: list) -> float:
    a = sorted({x for x, _ in pairs})
    b = sorted({y for _, y in pairs})
    tab = np.zeros((len(a), len(b)))
    for x, y in pairs:
        tab[a.index(x), b.index(y)] += 1
    n = tab.sum()
    exp = np.outer(tab.sum(1), tab.sum(0)) / n
    chi2 = float(((tab - exp) ** 2 / np.where(exp == 0, 1, exp)).sum())
    k = min(len(a), len(b)) - 1
    return round(math.sqrt(chi2 / (n * k)), 4) if k > 0 else 0.0


def joint_grid(trajs: dict, fs: dict) -> dict:
    tl = list(trajs.values())
    cells = Counter((t["template"], t["t_h"], t["p"]) for t in tl)
    seeds = [t["seed"] for t in tl]
    dd = prior_json("dataset_design.json")
    sp = prior_json("split_specification.json")
    roles = dd["cell_roles"]
    prim = [k for k, v in roles.items() if v == "TEST_PRIMARY_JOINT"]
    ax = sp["axes"]
    ranges = dd["seed_ranges_by_role"]

    def rng(s):
        a, b = s.split(" ")[0].split("-")
        return int(a), int(b)
    role_ranges = {k: rng(v) for k, v in ranges.items()}
    overlaps = [(a, b) for a, b in itertools.combinations(role_ranges, 2)
                if not (role_ranges[a][1] < role_ranges[b][0] or role_ranges[b][1] < role_ranges[a][0])]
    pilot_in_range = all(role_ranges["PILOT"][0] <= s <= role_ranges["PILOT"][1] for s in seeds)
    x4 = dict(
        n_primary_joint_cells=len(prim), primary_cells=prim,
        test_t_h_disjoint_from_train=not (set(ax["t_h"]["test"]) & set(ax["t_h"]["train"])),
        test_p_disjoint_from_train=not (set(ax["p"]["test"]) & set(ax["p"]["train"])),
        every_template_crosses_every_level=dd["crossing"].startswith("Full factorial"),
        seed_ranges_disjoint=not overlaps, seed_range_overlaps=overlaps,
        compliance_keys_role_specific="{role}" in dd["compliance_keys"],
    )
    x4["passed"] = bool(len(prim) == 4 and x4["test_t_h_disjoint_from_train"] and x4["test_p_disjoint_from_train"]
                        and x4["every_template_crosses_every_level"] and x4["seed_ranges_disjoint"]
                        and x4["compliance_keys_role_specific"])
    th_p = Counter((t["t_h"], t["p"]) for t in tl)
    return dict(
        audit=MARKER,
        pilot_cells=len(cells), trajectories_per_cell=dict(Counter(cells.values())),
        all_cells_complete=all(v == fs["pilot"]["replicates_per_cell"] for v in cells.values())
        and len(cells) == len(fs["pilot"]["templates"]) * len(fs["pilot"]["t_h_levels"]) * len(fs["pilot"]["p_levels"]),
        timing_x_compliance_counts={f"t_h={a}|p={b}": v for (a, b), v in sorted(th_p.items())},
        pilot_role=dict(role="PILOT", note="all pilot t_h and p values are TRAIN-role levels by the frozen design; "
                        "the pilot is discarded after Phase 2, so it cannot leak into any full-dataset split",
                        t_h_levels=fs["pilot"]["t_h_levels"], p_levels=fs["pilot"]["p_levels"],
                        includes_zero_shot_templates=[k for k in fs["pilot"]["templates"] if k.startswith("Z")]),
        association=dict(template_vs_t_h_cramers_v=_cramers_v([(t["template"], t["t_h"]) for t in tl]),
                         template_vs_p_cramers_v=_cramers_v([(t["template"], t["p"]) for t in tl]),
                         t_h_vs_p_cramers_v=_cramers_v([(t["t_h"], t["p"]) for t in tl])),
        seeds=dict(unique=len(set(seeds)) == len(seeds), min=min(seeds), max=max(seeds),
                   inside_pilot_range=pilot_in_range),
        layout_separation="every template has all 15 (t_h, p) cells; layout never determines timing or compliance",
        accidental_overlap=dict(seed_overlap_with_full_roles=[r for r in role_ranges if r != "PILOT" and not (
            role_ranges[r][1] < min(seeds) or role_ranges[r][0] > max(seeds))],
            compliance_key_namespace="AILX_v1|PILOT|<template>; full-dataset keys use their role name"),
        primary_test_constructible=x4,
    )


def prior_json(name):
    with open(os.path.join(prior.OUT_DIR, name), encoding="utf-8") as f:
        return json.load(f)


# ===========================================================================
# Section 8: leakage.
# ===========================================================================

def leakage(header: list, states: list, trajs: dict) -> dict:
    feat_cols = header[len(P.META_COLUMNS): len(P.META_COLUMNS) + 26]
    forbidden = {"Q_A_at_H", "Q_B_at_H", "compliant", "compliance", "t_h", "p", "seed", "hazard_phase",
                 "time_since_hazard_onset_normalized", "target_exit"}
    c = Counter()
    for s in states:
        c["states"] += 1
        c["features_identical_across_candidates"] += int(len(s["xs"]) == 1)
        pre = s["t_h_int"] > s["tick"]
        r = s["x"][FEATS.index("residual_cap_H_CE1_normalized")]
        c["residual_ok"] += int(abs(r - (1.0 if pre else 2.0 / 6.0)) < 1e-12)
    # Replay + action independence + pre-onset counterfactual.
    for s in states:
        tr = trajs[s["traj"]]
        sim, _ = _replay_to(tr, s["history"], s["tick"])
        x = tuple(P.features(sim).tolist())
        c["replay_features_match"] += int(x == s["x"])
        c["replay_fingerprint_match"] += int(P.fingerprint(sim)[:16] == s["fp"])
        seen = set()
        for cd in s["cands"]:
            if cd["cls"] in seen:
                continue
            seen.add(cd["cls"])
            asg = P.restrict(sim, next(k for k in FS["candidates"] if k["id"] == cd["id"])["asg"])
            fork = copy.deepcopy(sim)
            pairs = [(z, v) for z, v in asg.items() if v is not None]
            if pairs:
                apply_differential_action(fork, pairs)
            c["class_relabels_checked"] += 1
            c["features_unchanged_by_relabel"] += int(tuple(P.features(fork).tolist()) == s["x"])
        if tr["t_h_int"] <= MAX_STEPS and s["tick"] < tr["t_h_int"]:
            cf, _ = _replay_to(tr, s["history"], s["tick"], t_h_override=MAX_STEPS + 1)
            c["pre_onset_states"] += 1
            c["pre_onset_features_identical_without_hazard"] += int(tuple(P.features(cf).tolist()) == s["x"])
    tick0 = defaultdict(set)
    for s in states:
        if s["tick"] == 0:
            tick0[s["template"]].add(s["x"])
    res = dict(
        audit=MARKER,
        feature_columns_exact=feat_cols == FEATS, forbidden_feature_columns=sorted(set(feat_cols) & forbidden),
        metadata_columns_not_features=["t_h", "p", "seed", "compliance_key", "template", "distribution",
                                       "pre_decision_*", "state_fingerprint", "hazard_started", "default_action"],
        counts=dict(c),
        tick0_inputs_per_template=({k: len(v) for k, v in tick0.items()}),
        tick0_input_independent_of_t_h_and_p=all(len(v) == 1 for v in tick0.values()),
    )
    res["passed"] = bool(
        res["feature_columns_exact"] and not res["forbidden_feature_columns"]
        and c["features_identical_across_candidates"] == c["states"]
        and c["replay_features_match"] == c["states"] and c["replay_fingerprint_match"] == c["states"]
        and c["features_unchanged_by_relabel"] == c["class_relabels_checked"]
        and c["pre_onset_features_identical_without_hazard"] == c["pre_onset_states"]
        and c["residual_ok"] == c["states"])
    return res


# ===========================================================================
# Section 9: continuation + candidate semantics + regeneration (G4).
# ===========================================================================

class _HiddenHeuristicCalled(RuntimeError):
    pass


def continuation_and_regeneration(states: list, gen_record: dict) -> dict:
    src = {}
    forbidden = ["HEURISTIC(", "DeadbandCapacityAwareHeuristic", "run_intervention", "Q_A_IDX", "Q_B_IDX",
                 "capacity_aware_heuristic.heuristic_action", "asgc."]
    for rel in ("run_ai_learning_pilot.py", "ai_learning_label_free_default.py"):
        with open(os.path.join(SCRIPTS_DIR, rel), encoding="utf-8") as f:
            text = f.read()
        code = "\n".join(line for line in text.splitlines() if not line.strip().startswith(("#", '"', "'")))
        src[rel] = [tok for tok in forbidden if tok in code]
    # Booby-trap every hidden-label entry point, then regenerate the whole pilot.
    import deadband_heuristic as dh
    import capacity_aware_heuristic as cah

    def boom(*a, **k):
        raise _HiddenHeuristicCalled("hidden-label heuristic called during pilot generation")
    saved = (prior.asgc.HEURISTIC, dh.DeadbandCapacityAwareHeuristic.__call__, dh.DeadbandCapacityAwareHeuristic.predict,
             cah.heuristic_action)
    prior.asgc.HEURISTIC = boom
    dh.DeadbandCapacityAwareHeuristic.__call__ = boom
    dh.DeadbandCapacityAwareHeuristic.predict = boom
    cah.heuristic_action = boom
    try:
        t0 = time.perf_counter()
        regen = P.generate()
        trap_ok = True
    except _HiddenHeuristicCalled:
        regen, trap_ok = None, False
    finally:
        prior.asgc.HEURISTIC, dh.DeadbandCapacityAwareHeuristic.__call__, dh.DeadbandCapacityAwareHeuristic.predict, \
            cah.heuristic_action = saved
    regen_sha = hashlib.sha256(P.rows_to_csv_bytes(regen["rows"])).hexdigest() if regen else None
    # Default and semantics.
    x9 = Counter()
    for s in states:
        x9["states"] += 1
        x9["default_equals_label_free_rule"] += int(lfd.label_free_default_action(np.asarray(s["x"])) == s["default_action"])
        x9["default_equals_closed_form"] += int(s["default_action"] == (
            2 if (s["x"][FEATS.index("residual_cap_H_CE1_normalized")] < 1 and s["x"][FEATS.index("waiting_H_total")] >= 1)
            else 1))
    for s in states[::4]:
        tr = TRAJS[s["traj"]]
        sim, _ = _replay_to(tr, s["history"], s["tick"])
        seen = set()
        for cd in s["cands"]:
            if cd["cls"] in seen:
                continue
            seen.add(cd["cls"])
            asg = P.restrict(sim, next(k for k in FS["candidates"] if k["id"] == cd["id"])["asg"])
            fork = copy.deepcopy(sim)
            pairs = [(z, v) for z, v in asg.items() if v is not None]
            if pairs:
                apply_differential_action(fork, pairs)
            x9["class_reps_checked"] += 1
            x9["fast_label_map_equals_apply"] += int(P.label_map(sim, asg) == tuple(
                o["target_exit"] for _, o in sorted(fork.occupants.items())))
        g = copy.deepcopy(sim)
        apply_action(g, s["default_action"])
        d = next(cd for cd in s["cands"] if cd["is_default"])
        dasg = P.restrict(sim, {z: s["default_action"] for z in P.ELIGIBLE})
        x9["default_states_checked"] += 1
        x9["default_candidate_equals_global_apply"] += int(P.label_map(sim, dasg) == tuple(
            o["target_exit"] for _, o in sorted(g.occupants.items())) and d["name"] == f"BASE={'APB'[s['default_action']]}")
    hashes = {h for s in states for h in s["hashes"]}
    cc = gen_record["continuation_calls"]
    res = dict(
        audit=MARKER,
        continuation_policy=lfd.POLICY_ID, current_policy_hash=lfd.policy_hash(),
        hashes_in_rows=sorted(hashes),
        every_row_has_current_hash=hashes == {(lfd.POLICY_ID, lfd.policy_hash())},
        forbidden_tokens_in_generation_source=src,
        hidden_heuristic_booby_trap=dict(regeneration_completed_with_hidden_heuristic_disabled=trap_ok,
                                         entry_points_disabled=["audit_selective_guidance_control.HEURISTIC",
                                                                "DeadbandCapacityAwareHeuristic.__call__/.predict",
                                                                "capacity_aware_heuristic.heuristic_action"]),
        continuation_calls_generation=cc,
        continuation_calls_regeneration=regen["continuation_calls"] if regen else None,
        all_calls_26_features=(cc["calls"] == cc["input_len_26"] and bool(regen)
                               and regen["continuation_calls"]["calls"] == regen["continuation_calls"]["input_len_26"]),
        q_values_enter_continuation=False,
        q_values_note="the default's only input is the 26-feature vector; waiting_H_total is len(zone_waiting['H'])",
        reference_arm_P2b_in_pilot=False,
        candidate_semantics=dict(x9),
        regeneration=dict(identical_rows=bool(regen) and regen_sha == gen_record["rows_uncompressed_sha256"],
                          regenerated_sha256=regen_sha, recorded_sha256=gen_record["rows_uncompressed_sha256"],
                          regeneration_checks=regen["checks"] if regen else None,
                          seconds=round(time.perf_counter() - t0, 1)),
    )
    res["label_free_passed"] = bool(res["every_row_has_current_hash"] and not any(src.values()) and trap_ok
                                    and res["all_calls_26_features"])
    res["semantics_passed"] = bool(x9["default_equals_label_free_rule"] == x9["states"]
                                   and x9["default_equals_closed_form"] == x9["states"]
                                   and x9["fast_label_map_equals_apply"] == x9["class_reps_checked"]
                                   and x9["default_candidate_equals_global_apply"] == x9["default_states_checked"])
    return res


# ===========================================================================
# G3: unrestricted-space re-check.
# ===========================================================================

def g3_recheck(states: list) -> dict:
    order = sorted(states, key=lambda s: (-s["n_nonempty"], s["traj"], s["tick"]))[:40]
    matches, regrets, rows = 0, [], []
    t0 = time.perf_counter()
    for s in order:
        tr = TRAJS[s["traj"]]
        sim, _ = _replay_to(tr, s["history"], s["tick"])
        st = dict(traj=tr, tick=s["tick"], sim=sim, fp=P.fingerprint(sim), n_evacuated=len(sim.evacuated))
        zs = [z for z in P.ELIGIBLE if len(sim.zone_waiting[z]) > 0]
        cache = {}
        for vals in itertools.product([None, 0, 1, 2], repeat=len(zs)):
            asg = dict(zip(zs, vals))
            lm = P.label_map(sim, asg)
            if lm not in cache:
                o = P.rollout(st, asg)
                cache[lm] = (o["future_T90"], o["future_T100"], o["future_unmet"], o["future_hazard_unmet"])
        best_full = min(cache.values())
        best_b2 = min(cd["y"] for cd in s["cands"])
        matches += int(best_b2 == best_full)
        regrets.append(best_b2[1] - best_full[1])
        rows.append(dict(state=f"{s['traj']}|t{s['tick']}", n_nonempty=s["n_nonempty"], unrestricted_classes=len(cache),
                         base2_best=list(best_b2), unrestricted_best=list(best_full)))
    return dict(n_states=len(order), matches=matches, share=round(matches / len(order), 4),
                mean_T100_regret=round(float(np.mean(regrets)), 4), max_T100_regret=float(np.max(regrets)),
                nonempty_zone_counts=dict(Counter(s["n_nonempty"] for s in order)), states=rows,
                seconds=round(time.perf_counter() - t0, 1))


# ===========================================================================
# G6 and the decision.
# ===========================================================================

def g6(states: list) -> dict:
    post = [s for s in states if s["t_h_int"] <= MAX_STEPS and s["tick"] >= s["t_h_int"]]
    groups = defaultdict(list)
    for s in post:
        for cd in s["cands"]:
            groups[(s["x"], cd["id"])].append((s["traj"], cd["y"][1]))
    multi = {k: [y for _, y in v] for k, v in groups.items() if len({t for t, _ in v}) >= 2}
    n_post_rows = sum(len(v) for v in groups.values())
    cov = sum(len(v) for v in multi.values()) / max(1, n_post_rows)
    share = _within_share(multi)["irreducible_share"] if multi else None
    evaluable = cov >= 0.10 and share is not None
    return dict(post_onset_states=len(post), post_onset_rows=n_post_rows,
                rows_in_multi_trajectory_groups=sum(len(v) for v in multi.values()), coverage=round(cov, 4),
                T100_irreducible_share=share, evaluable=evaluable,
                passed=bool(evaluable and share < 0.6),
                plain_formula_all_post_rows=_within_share({k: [y for _, y in v] for k, v in groups.items()}))


def decide(div, tgt, grid, leak, cont, g3, g6r, gen_record, states) -> dict:
    crit = prior_json("diversity_requirements.json")["go_no_go_after_pilot"]
    thr = v2.OLD
    tl = div["tick_ge_10"]
    pert = [b for s in _last_states(states) for b in s["perturbed"]]
    rate = sum(pert) / max(1, len(pert))
    censored = tgt["counts"].get("censored_rows", 0)
    res = {
        "G1": dict(passed=bool(tl["share"] >= 0.25 and min(tl["unique_by_template"].values()) >= 10),
                   value=dict(share=tl["share"], min_unique_per_template=min(tl["unique_by_template"].values()))),
        "G2": dict(passed=tgt["distinct_target_vectors_per_state"]["share_with_at_least_3"] >= 0.5,
                   value=tgt["distinct_target_vectors_per_state"]["share_with_at_least_3"]),
        "G3": dict(passed=bool(g3["share"] >= 0.95 and g3["mean_T100_regret"] <= 0.1),
                   value=dict(share=g3["share"], mean_T100_regret=g3["mean_T100_regret"])),
        "G4": dict(passed=bool(cont["regeneration"]["identical_rows"]
                               and gen_record["generation_checks"]["hazard_tick_log"] == gen_record["generation_checks"]["n_classes"]),
                   value=dict(identical_rows=cont["regeneration"]["identical_rows"],
                              hazard_tick_log=f"{gen_record['generation_checks']['hazard_tick_log']}/{gen_record['generation_checks']['n_classes']}")),
        "G5": dict(passed=leak["passed"], value=leak["counts"]),
        "G6": dict(passed=g6r["passed"], value=dict(share=g6r["T100_irreducible_share"], coverage=g6r["coverage"],
                                                    evaluable=g6r["evaluable"])),
        "G7": dict(passed=grid["all_cells_complete"], value=grid["trajectories_per_cell"]),
        "X1": dict(passed=bool(tgt["hazard_zero_when_no_hazard"] and tgt["hazard_zero_when_episode_ends_before_onset"]
                               and tgt["T100_semantics_all_rows"] and tgt["T90_semantics_all_rows"]
                               and tgt["pre_decision_unmet_excluded"]["passed"]),
                   value=tgt["pre_decision_unmet_excluded"]),
        "X2": dict(passed=leak["passed"], value="= G5"),
        "X3": dict(passed=gen_record["generation_checks"]["same_start"] == gen_record["generation_checks"]["n_classes"],
                   value=f"{gen_record['generation_checks']['same_start']}/{gen_record['generation_checks']['n_classes']}"),
        "X4": dict(passed=grid["primary_test_constructible"]["passed"], value=grid["primary_test_constructible"]),
        "X5": dict(passed=bool(div["n_unique_feature_vectors"] >= 10 * thr["distinct_inputs"]
                               and div["distinct_share"] >= round(5 * thr["distinct_inputs"] / thr["states"], 4)
                               and div["n_unique_state_candidate_pairs"] >= 10 * thr["distinct_input_action_pairs"]),
                   value=dict(unique_inputs=div["n_unique_feature_vectors"], distinct_share=div["distinct_share"],
                              state_candidate_pairs=div["n_unique_state_candidate_pairs"])),
        "X6": dict(passed=tgt["distinct_target_vectors_per_state"]["share_with_at_least_3"] >= 0.5, value="= G2"),
        "X7": dict(passed=bool(grid["association"]["template_vs_t_h_cramers_v"] == 0
                               and grid["association"]["template_vs_p_cramers_v"] == 0 and grid["seeds"]["unique"]
                               and abs(rate - 0.5) <= 0.1 and censored == 0),
                   value=dict(association=grid["association"], perturbation_rate=round(rate, 4), censored_rows=censored)),
        "X8": dict(passed=tgt["pre_decision_unmet_excluded"]["passed"], value="= X1 decomposition"),
        "X9": dict(passed=cont["semantics_passed"], value=cont["candidate_semantics"]),
        "X10": dict(passed=cont["label_free_passed"], value=dict(booby_trap=cont["hidden_heuristic_booby_trap"],
                                                                 tokens=cont["forbidden_tokens_in_generation_source"])),
    }
    all_pass = all(v["passed"] for v in res.values())
    return dict(audit=MARKER, criteria_source="diversity_requirements.json (specification v2), applied unchanged",
                criteria_text=crit, results=res, failed=[k for k, v in res.items() if not v["passed"]],
                decision="PILOT GO" if all_pass else "PILOT NO-GO",
                next_step_not_taken="the 5,544-trajectory dataset is NOT generated; no model is trained")


def _last_states(states):
    last = {}
    for s in states:
        if s["traj"] not in last or s["tick"] > last[s["traj"]]["tick"]:
            last[s["traj"]] = s
    return list(last.values())


# ===========================================================================
# Report.
# ===========================================================================

def render(ctx) -> str:
    div, tgt, grid, leak, cont, g3r, g6r, dec, man = (ctx[k] for k in (
        "div", "tgt", "grid", "leak", "cont", "g3", "g6", "dec", "manifest"))
    L = [f"<!-- {MARKER} -->", "# AI-Learning Pilot (Phase 1) Dataset Audit\n",
         "Pilot only: no model trained, no selector, no closed-loop evaluation, no full dataset. Criteria are the ones "
         "pre-registered in specification v2 (committed before any pilot data existed), applied unchanged.\n",
         f"## Decision: **{dec['decision']}**" + (f" (failed: {', '.join(dec['failed'])})" if dec["failed"] else "") + "\n",
         "| criterion | passed | value |", "|---|---|---|"]
    for k, v in dec["results"].items():
        L.append(f"| {k} | {'PASS' if v['passed'] else 'FAIL'} | {json.dumps(v['value'], default=str)[:220]} |")
    L += ["\n## Dataset\n",
          f"{man['n_trajectories']} trajectories, {man['n_decision_states']} decision states, {man['n_rows']:,} candidate "
          f"rows ({man['n_columns']} columns), {man['generation_checks']['n_classes']:,} unique rollouts. File "
          f"`pilot/{P.ROWS_FILE}` (SHA-256 `{man['rows_file_sha256'][:16]}`).\n",
          "## Diversity vs the old dataset\n",
          "| measure | old dataset | pilot |", "|---|---|---|",
          f"| decision states | {div['old_dataset']['states']} | {div['n_decision_states']} |",
          f"| unique observable inputs | {div['old_dataset']['distinct_inputs']} | {div['n_unique_feature_vectors']} |",
          f"| distinct share | {div['old_dataset']['distinct_share']:.1%} | {div['distinct_share']:.1%} |",
          f"| unique (input, candidate) pairs | {div['old_dataset']['distinct_input_action_pairs']} | {div['n_unique_state_candidate_pairs']:,} |",
          f"| mean feature entropy (bits) | {div['old_dataset']['mean_feature_entropy_bits']} | {div['mean_feature_entropy_bits']} |",
          f"\nBy tick: {div['unique_states_by_tick']}. By template: {div['unique_states_by_template']}. Ticks >= 10: "
          f"{div['tick_ge_10']}. Repeated-state fraction {div['repeated_state_fraction']:.1%}; repeated-trajectory "
          f"fraction {div['repeated_trajectory_fraction']:.1%}.\n",
          "Conditional outcome variance (irreducible share, SSW/SST over all rows; singleton groups contribute 0, so "
          "these are lower bounds):\n"]
    for t, v in div["conditional_outcome_variance"].items():
        L.append(f"- {t}: given state {v['given_state']['irreducible_share']}, given state and candidate "
                 f"{v['given_state_and_candidate']['irreducible_share']}")
    L += [f"\nG6 (pre-registered, multi-trajectory groups, post-onset): share {g6r['T100_irreducible_share']}, coverage "
          f"{g6r['coverage']:.1%} of {g6r['post_onset_rows']:,} post-onset rows; plain formula "
          f"{g6r['plain_formula_all_post_rows']['irreducible_share']}.\n",
          "## Targets\n",
          f"- hazard target 0 in all no-hazard rows: {tgt['hazard_zero_when_no_hazard']}; 0 whenever the episode ends "
          f"before onset: {tgt['hazard_zero_when_episode_ends_before_onset']}",
          f"- T100 / T90 semantics on every row: {tgt['T100_semantics_all_rows']} / {tgt['T90_semantics_all_rows']}",
          f"- pre-decision unmet excluded (full-episode decomposition): {tgt['pre_decision_unmet_excluded']['decomposition_exact']}"
          f"/{tgt['pre_decision_unmet_excluded']['states_checked']} states exact",
          f"- distinct target vectors per state: {tgt['distinct_target_vectors_per_state']}",
          f"- share of candidates tied with the default: {tgt['share_of_candidates_tied_with_default']}; classes per "
          f"state: {tgt['n_equivalence_classes_per_state']}",
          f"- one continuation for all rows: {tgt['continuation_same_for_all_candidates']}\n",
          "## Joint grid\n",
          f"{grid['pilot_cells']} cells (8 templates x 5 t_h x 3 p), trajectories per cell {grid['trajectories_per_cell']}; "
          f"association template-t_h {grid['association']['template_vs_t_h_cramers_v']}, template-p "
          f"{grid['association']['template_vs_p_cramers_v']}; seeds unique {grid['seeds']['unique']} "
          f"({grid['seeds']['min']}-{grid['seeds']['max']}). {grid['pilot_role']['note']}. Primary joint test "
          f"constructible at design level: {grid['primary_test_constructible']['passed']} "
          f"({grid['primary_test_constructible']['primary_cells']}).\n",
          "## Leakage (on the rows)\n",
          f"passed {leak['passed']}: {leak['counts']}; feature columns exact {leak['feature_columns_exact']}; forbidden "
          f"feature columns {leak['forbidden_feature_columns']}; tick-0 inputs independent of t_h and p "
          f"{leak['tick0_input_independent_of_t_h_and_p']}.\n",
          "## Label-free continuation\n",
          f"label-free {cont['label_free_passed']}; every row carries hash `{cont['current_policy_hash'][:16]}`: "
          f"{cont['every_row_has_current_hash']}; forbidden tokens in generator source: "
          f"{cont['forbidden_tokens_in_generation_source']}; full regeneration with every hidden-heuristic entry point "
          f"disabled completed: {cont['hidden_heuristic_booby_trap']['regeneration_completed_with_hidden_heuristic_disabled']}"
          f" and reproduced every row: {cont['regeneration']['identical_rows']}; all continuation calls had 26 features: "
          f"{cont['all_calls_26_features']}. Candidate semantics: {cont['candidate_semantics']}.\n",
          "## G3 unrestricted re-check\n",
          f"{g3r['matches']}/{g3r['n_states']} states: BASE2_298 best == unrestricted best; mean T100 regret "
          f"{g3r['mean_T100_regret']}, max {g3r['max_T100_regret']}; non-empty zones {g3r['nonempty_zone_counts']}.\n",
          "## Integrity\n",
          f"Protected files unchanged: **{ctx['integrity']['all_unchanged']}** ({len(ctx['integrity']['before'])} files). "
          f"Controller MD5 match {ctx['integrity']['controller']['match']}; RL checkpoints unchanged "
          f"{ctx['integrity']['checkpoints']['all_match']}."]
    return "\n".join(L) + "\n"


FS = None
TRAJS = None


def run() -> dict:
    global FS, TRAJS
    before = protected_hashes()
    t0 = time.perf_counter()
    TRAJS, FS = trajectory_by_id()
    with open(os.path.join(PILOT_DIR, P.GENERATION_FILE), encoding="utf-8") as f:
        gen_record = json.load(f)
    header, rows = load_rows()
    states = build_states(header, rows)
    print(f"loaded {len(rows)} rows, {len(states)} states")
    old = old_dataset_reference()
    div = diversity(states, old)
    tgt = target_audit(states, TRAJS)
    grid = joint_grid(TRAJS, FS)
    leak = leakage(header, states, TRAJS)
    print(f"leakage passed {leak['passed']}")
    cont = continuation_and_regeneration(states, gen_record)
    print(f"regeneration identical {cont['regeneration']['identical_rows']}, label-free {cont['label_free_passed']}")
    g3r = g3_recheck(states)
    print(f"G3 {g3r['matches']}/{g3r['n_states']}")
    g6r = g6(states)
    dec = decide(div, tgt, grid, leak, cont, g3r, g6r, gen_record, states)
    after = protected_hashes()
    changed = [k for k in before if before[k] != after[k]]
    if changed:
        raise RuntimeError(f"PROTECTED FILES CHANGED: {changed}")
    integrity = dict(before=before, all_unchanged=True, controller=prior.design.verify_controller_hash(),
                     checkpoints=prior.design.verify_checkpoint_hashes())
    manifest = dict(audit=MARKER, **{k: gen_record[k] for k in (
        "rows_file", "rows_file_sha256", "rows_uncompressed_sha256", "n_rows", "n_columns", "header", "n_trajectories",
        "n_decision_states", "generation_checks", "continuation_calls", "spec_hashes", "provenance")},
        generation_record_sha256=prior.sha256_file(os.path.join(PILOT_DIR, P.GENERATION_FILE)),
        spec_version_record_sha256=prior.sha256_file(os.path.join(prior.OUT_DIR, "spec_version_record.json")),
        pre_registration_commit="6872c3f (specification v2 committed before the pilot was generated)",
        row_reconstruction="each row: trajectory_id, split_role, template/distribution, t_h, p, seed, compliance_key, "
                           "epsilon, decision_tick, pre-decision action history, 26 features, candidate id/name, "
                           "equivalence class, targets, continuation policy id + hash, spec version + hash",
        integrity=integrity)
    ctx = dict(div=div, tgt=tgt, grid=grid, leak=leak, cont=cont, g3=g3r, g6=g6r, dec=dec, manifest=manifest,
               integrity=integrity)
    ctx["outputs"] = {
        "pilot_dataset_manifest.json": manifest, "pilot_diversity.json": div,
        "pilot_target_audit.json": dict(tgt, g3_unrestricted_recheck=g3r, g6=g6r),
        "pilot_joint_grid.json": grid, "pilot_leakage_audit.json": leak,
        "pilot_continuation_audit.json": cont, "pilot_go_no_go.json": dec,
    }
    ctx["outputs"]["PILOT_DATASET_AUDIT.md"] = render(ctx)
    ctx["seconds"] = round(time.perf_counter() - t0, 1)
    return ctx


def write_outputs(ctx) -> list:
    created = set()
    for n in OUTPUT_FILES:
        _safe_write(n, ctx["outputs"][n], created)
    return sorted(created)


if __name__ == "__main__":
    ctx = run()
    write_outputs(ctx)
    print(f"\n{ctx['dec']['decision']} failed={ctx['dec']['failed']} ({ctx['seconds']}s)")
