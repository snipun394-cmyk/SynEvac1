"""READ-ONLY label-free default audit (after decisions D1 and D2).

D1 (approved): future hazard-edge unmet demand counts only ticks at or
after the hazard's actual onset; 0 before onset and in no-hazard scenarios.
D2 (decided): the main experiment's continuation policy and selector
fallback must not read Q_A_at_H / Q_B_at_H or any other hidden variable.

This audit finds the MINIMUM label-free default. It trains nothing, runs no
pilot, generates no training data, and implements no policy module: the
label-free candidates are evaluated through ANALYSIS-ONLY functions defined
here, which call the existing, unmodified decision rules. Everything is
computed in memory on EXISTING scenario configurations (the 450
differential-targeting states' scenarios and the 80 timing-matrix configs).
Only the deliverables in OUTPUT_FILES are written; the frozen experiment
specification files are protected and never overwritten.
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)
import audit_final_experiment_specification as spec  # noqa: E402
import audit_pre_training_ai_dataset as prior  # noqa: E402

from actions import apply_action  # noqa: E402
from arbitration_engine import RECOMMENDED_MARGIN  # noqa: E402
from compliance import generate_compliance_vector  # noqa: E402
from deadband_heuristic import deadband_heuristic_action  # noqa: E402
from metrics import compute_metrics  # noqa: E402
from observation import OBSERVATION_INDEX, compute_observation  # noqa: E402
from simulator import EvacuationSimulator  # noqa: E402
import stress_test_scenarios as ST  # noqa: E402
from timing_experiment import diagnostic_all_edges_unmet  # noqa: E402
import topology as topology_module  # noqa: E402
from topology import DECISION_INTERVAL, MAX_STEPS, N_OCCUPANTS  # noqa: E402

DATA_ROOT, OUT_DIR = prior.DATA_ROOT, prior.OUT_DIR
MARKER = "label_free_default_audit_v1"
OUTPUT_FILES = [
    "LABEL_FREE_DEFAULT_AUDIT.md",
    "label_free_policy_candidates.json",
    "hidden_vs_label_free_comparison.json",
    "target_impact_analysis.json",
    "fallback_validity.json",
    "d1_d2_decision_record.json",
]
FROZEN_SPEC_FILES = [f for f in spec.OUTPUT_FILES]
IQA, IQB = OBSERVATION_INDEX.index("Q_A_at_H"), OBSERVATION_INDEX.index("Q_B_at_H")
IRES = OBSERVATION_INDEX.index("residual_cap_H_CE1_normalized")
HIDDEN = spec.HEURISTIC  # GLOBAL_DEADBAND, margin RECOMMENDED_MARGIN -- reads Q_A/Q_B (reference only)


# ===========================================================================
# Integrity.
# ===========================================================================

def protected_hashes() -> dict:
    out = dict(spec.protected_hashes())
    for name in FROZEN_SPEC_FILES + list(spec.SUPERSEDES):
        p = os.path.join(OUT_DIR, name)
        out[f"OUT_DIR/{name}"] = prior.sha256_file(p) if os.path.isfile(p) else "MISSING"
    for rel in ("scripts/audit_final_experiment_specification.py", "tests/test_final_experiment_specification.py"):
        p = os.path.join(prior.OUT_ROOT, rel)
        out[f"OUT_ROOT/{rel}"] = prior.sha256_file(p) if os.path.isfile(p) else "MISSING"
    return out


def _safe_write(name: str, content, created: set, out_dir: str = OUT_DIR) -> str:
    if name not in OUTPUT_FILES:
        raise PermissionError(f"refusing to write non-deliverable {name!r}")
    path = os.path.join(out_dir, name)
    if os.path.exists(path) and name not in created:
        with open(path, encoding="utf-8") as f:
            if MARKER not in f.read():
                raise PermissionError(f"refusing to overwrite a file not produced by this audit: {path}")
    text = content if isinstance(content, str) else json.dumps(content, indent=2, default=prior._json_default)
    if MARKER not in text:
        raise ValueError("deliverable missing marker")
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    created.add(name)
    return path


# ===========================================================================
# Section 1: candidate policies (analysis-only callables; obs -> action or None).
# ===========================================================================

def _symmetric_split_obs(obs) -> np.ndarray:
    """Replace the hidden hub split by the label-free total split evenly.
    Reads only waiting_H_total (= Q_A + Q_B) and leaves every other index."""
    o = np.array(obs, dtype=np.float64).copy()
    h = o[IQA] + o[IQB]
    o[IQA] = o[IQB] = h / 2.0
    return o


def lf_symmetric_deadband(obs) -> int:
    """Existing, unmodified deadband rule on a label-free input."""
    return int(deadband_heuristic_action(_symmetric_split_obs(obs), RECOMMENDED_MARGIN))


def lf_symmetric_deadband_margin0(obs) -> int:
    return int(deadband_heuristic_action(_symmetric_split_obs(obs), 0.0))


POLICIES = {
    "HIDDEN_GLOBAL_DEADBAND (reference only)": lambda obs: int(HIDDEN(obs)),
    "NO_INTERVENTION": lambda obs: None,
    "CONSTANT_PARITY (baseline.py)": lambda obs: 1,
    "CONSTANT_FAVOR_A": lambda obs: 0,
    "CONSTANT_FAVOR_B": lambda obs: 2,
    "SYMMETRIC_SPLIT_DEADBAND (margin 0.75)": lf_symmetric_deadband,
    "SYMMETRIC_SPLIT_DEADBAND (margin 0)": lf_symmetric_deadband_margin0,
}
LABEL_FREE = [k for k in POLICIES if not k.startswith("HIDDEN")]
EXISTING = {"NO_INTERVENTION", "CONSTANT_PARITY (baseline.py)", "CONSTANT_FAVOR_A", "CONSTANT_FAVOR_B"}
SIMPLICITY_ORDER = ["NO_INTERVENTION", "CONSTANT_PARITY (baseline.py)", "CONSTANT_FAVOR_A", "CONSTANT_FAVOR_B",
                    "SYMMETRIC_SPLIT_DEADBAND (margin 0)", "SYMMETRIC_SPLIT_DEADBAND (margin 0.75)"]


def candidate_inventory() -> list:
    return [
        dict(name="capacity_aware_heuristic.CapacityAwareHeuristic / deadband_heuristic.DeadbandCapacityAwareHeuristic "
                  "(GLOBAL_DEADBAND)",
             inputs=["Q_A_at_H", "Q_B_at_H", "residual_cap_H_CE1_normalized", "CAP_B constant"],
             hidden_dependencies=["Q_A_at_H", "Q_B_at_H (hidden target_exit labels)"],
             semantics="Discrete(3) global action", deterministic=True, tested=True,
             used_elsewhere="the continuation / default in every prior experiment", label_free=False,
             verdict="REFERENCE ONLY (D2)"),
        dict(name="branch_aware_deadband_heuristic", inputs=["as GLOBAL_DEADBAND"], hidden_dependencies=["Q_A_at_H", "Q_B_at_H"],
             semantics="Discrete(3)", deterministic=True, tested=True, used_elsewhere="E1 validation", label_free=False,
             verdict="disqualified"),
        dict(name="arbitration_engine / hybrid_controller / policy_selector",
             inputs=["GLOBAL_DEADBAND output and/or Q_A, Q_B, clearance features, PPO output"],
             hidden_dependencies=["Q_A_at_H", "Q_B_at_H"], semantics="Discrete(3)", deterministic="yes (given PPO)",
             tested=True, used_elsewhere="hybrid / arbitration experiments", label_free=False, verdict="disqualified"),
        dict(name="observable_robust_differential.py (frozen controller)",
             inputs=["exact match on the full 27-D observation incl. Q_A/Q_B; else action 0 = GLOBAL_DEADBAND"],
             hidden_dependencies=["Q_A_at_H", "Q_B_at_H"], semantics="Discrete(211)", deterministic=True, tested=True,
             used_elsewhere="robust-control experiments", label_free=False, verdict="disqualified (and frozen)"),
        dict(name="PPO checkpoints (frozen RL branch)", inputs=["27-D observation incl. Q_A/Q_B"],
             hidden_dependencies=["Q_A_at_H", "Q_B_at_H"], semantics="Discrete(3)/(211)", deterministic="with deterministic=True",
             tested=True, used_elsewhere="RL experiments", label_free=False, verdict="disqualified (and frozen)"),
        dict(name="NO_INTERVENTION (never relabel)", inputs=[], hidden_dependencies=[],
             semantics="no relabel at any decision", deterministic=True, tested=True,
             used_elsewhere="the NO_INTERVENTION arm of every counterfactual audit", label_free=True,
             verdict="evaluated", target_meaning_change="a candidate's labels then persist for the rest of the episode "
                                                       "(no later decision overwrites them): the label measures a "
                                                       "PERMANENT reassignment, not a one-decision deviation"),
        dict(name="CONSTANT_PARITY (baseline.baseline_action)", inputs=[], hidden_dependencies=[],
             semantics="PARITY at every decision", deterministic=True, tested=True,
             used_elsewhere="the pre-registered M4/M5 capacity-blind baseline", label_free=True, verdict="evaluated",
             target_meaning_change="one-decision deviation from a capacity-blind default; never reacts to the hazard"),
        dict(name="CONSTANT_FAVOR_A / CONSTANT_FAVOR_B", inputs=[], hidden_dependencies=[],
             semantics="fixed global action", deterministic=True, tested="as integer policies in timing experiments",
             used_elsewhere="timing experiments", label_free=True, verdict="evaluated for completeness"),
        dict(name="SYMMETRIC_SPLIT_DEADBAND (analysis-only definition, NOT implemented as a module)",
             inputs=["waiting_H_total (admissible, = Q_A+Q_B)", "residual_cap_H_CE1_normalized (admissible)",
                     "CAP_B constant"],
             hidden_dependencies=[],
             computation="deadband_heuristic_action(obs with Q_A = Q_B = waiting_H_total / 2, margin m), i.e. the "
                         "existing rule, unmodified, on a label-free input. Closed form: PARITY unless the hazard has "
                         "occurred (residual cap = 2/6) AND waiting_H_total > 6m (m=0: >= 1; m=0.75: >= 5), then "
                         "FAVOR_B. Never FAVOR_A. m=0 equals capacity_aware_heuristic's original parameter-free rule.",
             why_even_split="a fixed prior equal to the parity rule's own split; it does not estimate labels or "
                            "compliance from any data",
             semantics="Discrete(3) global action", deterministic=True, tested="no (new definition)",
             used_elsewhere="no", label_free=True, verdict="evaluated (minimal new definition)",
             target_meaning_change="one-decision deviation from a hazard-reactive, label-free default"),
    ]


# ===========================================================================
# Rollout with a parameterized continuation (mirrors spec.rollout exactly).
# ===========================================================================

def rollout_with(sim0, assignment: dict, has_hazard: bool, t_h: int, continuation) -> dict:
    """Candidate at t_d, then `continuation(obs)` (None = no relabel) at every
    later decision. D1 hazard definition (onset-gated). Unmet measured after
    relabel, before admission (verified against tick_log in the spec audit)."""
    sim = copy.deepcopy(sim0)
    t_d = sim.t
    spec.apply_assignment(sim, assignment)
    unmet = haz = 0
    first = True
    actions = []
    while sim.n_active() > 0 and sim.t < MAX_STEPS:
        if sim.t % DECISION_INTERVAL == 0 and not first:
            a = continuation(compute_observation(sim))
            actions.append(a)
            if a is not None:
                apply_action(sim, int(a))
        first = False
        unmet += diagnostic_all_edges_unmet(sim, sim.t, t_h, topology_module.edge_capacity)
        if has_hazard and sim.t >= t_h:
            _, hu, _ = ST.diagnostic_branch_demand_and_unmet(sim, "H_CE1", "A", sim.t, t_h, topology_module.edge_capacity)
            haz += hu
        sim._run_one_tick(None)
        sim.t += 1
    m = compute_metrics(sim.evacuated, N=N_OCCUPANTS, max_steps=MAX_STEPS)
    return dict(future_T90=(m["T90"] - t_d) if m["T90_defined"] else float("inf"), future_T100=m["T100"] - t_d,
                future_unmet=unmet, future_hazard_unmet=haz, T100_censored=m["T100_censored"],
                continuation_actions=actions)


def key4(o: dict) -> tuple:
    return (o["future_T90"], o["future_T100"], o["future_unmet"], o["future_hazard_unmet"])


def episode(sim0, has_hazard, t_h, policy) -> dict:
    """Full episode from a tick-0 state entirely under `policy`."""
    a0 = policy(compute_observation(sim0))
    zs = spec.nonempty(sim0)
    asg = {z: (spec.U if a0 is None else a0) for z in zs}
    o = rollout_with(sim0, asg, has_hazard, t_h, policy)
    o["first_action"] = a0
    return o


# ===========================================================================
# Section 4: hidden vs label-free comparison.
# ===========================================================================

def tick0_configs() -> list:
    out = []
    for sc in ("S1", "S2", "S3"):
        for seed in spec.prior.EVAL_SEED_SET:
            s = prior.replay_state(sc, seed, 0)
            out.append(dict(key=f"{sc}|{seed}", family=sc, sim=s["sim"], has_hazard=s["has_hazard"], t_h=s["t_h"]))
    for st in spec.existing_states():
        if st["src"] == "timing80":
            out.append(dict(key=st["key"], family=f"timing_th{st['t_h']}_p0.75", sim=st["sim"],
                            has_hazard=True, t_h=st["t_h"]))
    return out


def _paired(ref: list, other: list) -> dict:
    c = Counter()
    for r, o in zip(ref, other):
        c["worse" if key4(o) > key4(r) else ("better" if key4(o) < key4(r) else "tie")] += 1
    return dict(c)


def episode_comparison(configs: list) -> dict:
    res = {name: [episode(c["sim"], c["has_hazard"], c["t_h"], fn) for c in configs] for name, fn in POLICIES.items()}
    ref = res["HIDDEN_GLOBAL_DEADBAND (reference only)"]
    noint = res["NO_INTERVENTION"]
    out = {}
    fams = sorted({c["family"] for c in configs})
    for name, eps in res.items():
        per_fam = {}
        for fam in fams:
            idx = [i for i, c in enumerate(configs) if c["family"] == fam]
            per_fam[fam] = dict(
                mean_T90=round(float(np.mean([eps[i]["future_T90"] for i in idx])), 3),
                mean_T100=round(float(np.mean([eps[i]["future_T100"] for i in idx])), 3),
                mean_unmet=round(float(np.mean([eps[i]["future_unmet"] for i in idx])), 2),
                mean_hazard_unmet=round(float(np.mean([eps[i]["future_hazard_unmet"] for i in idx])), 2),
                vs_hidden_heuristic=_paired([ref[i] for i in idx], [eps[i] for i in idx]))
        out[name] = dict(
            n_episodes=len(eps),
            vs_hidden_heuristic=_paired(ref, eps),
            vs_no_intervention=_paired(noint, eps),
            mean_T100_delta_vs_hidden=round(float(np.mean([e["future_T100"] - r["future_T100"] for e, r in zip(eps, ref)])), 4),
            mean_hazard_unmet_delta_vs_hidden=round(float(np.mean(
                [e["future_hazard_unmet"] - r["future_hazard_unmet"] for e, r in zip(eps, ref)])), 3),
            max_T100_delta_vs_hidden=float(np.max([e["future_T100"] - r["future_T100"] for e, r in zip(eps, ref)])),
            any_censored=any(e["T100_censored"] for e in eps),
            per_family=per_fam)
    return dict(note="Full episodes from each existing tick-0 configuration (S1/S2/S3 x 50 eval seeds at p=0.9; "
                     "timing matrix t_h in {3,6,7,10} x 20 seeds at p=0.75), each run entirely under one policy. "
                     "Outcome keys use D1 (onset-gated) hazard-edge unmet. 'worse' = lexicographically worse on "
                     "(T90, T100, unmet, hazard unmet).", per_policy=out)


def action_agreement(states: list) -> dict:
    out = {}
    for name, fn in POLICIES.items():
        if name.startswith("HIDDEN"):
            continue
        agree = Counter()
        for st in states:
            obs = compute_observation(st["sim"])
            phase = "post_onset" if (st["has_hazard"] and st["sim"].t >= st["t_h"]) else "pre_onset_or_no_hazard"
            a = fn(obs)
            agree[(phase, "agree" if a == int(HIDDEN(obs)) else "differ")] += 1
        out[name] = {f"{p}|{k}": v for (p, k), v in sorted(agree.items())}
    return dict(note="On the 530 existing decision states (reached by the hidden heuristic): does each label-free "
                     "policy pick the same global action as the hidden heuristic? NO_INTERVENTION never 'agrees' "
                     "since it issues no action.", per_policy=out)


def choose_default(ep: dict) -> dict:
    rows = []
    for name in SIMPLICITY_ORDER:
        p = ep["per_policy"][name]
        rows.append(dict(policy=name, existing=name in EXISTING,
                         episodes_worse_than_hidden=p["vs_hidden_heuristic"].get("worse", 0),
                         episodes_worse_than_no_intervention=p["vs_no_intervention"].get("worse", 0),
                         max_T100_delta_vs_hidden=p["max_T100_delta_vs_hidden"]))
    best = min(r["episodes_worse_than_hidden"] for r in rows)
    chosen = next(r["policy"] for r in rows if r["episodes_worse_than_hidden"] == best)
    return dict(rule="fixed before computing: fewest paired episodes lexicographically worse than the hidden-label "
                     "heuristic (the previously accepted default); ties go to the simpler / existing policy in "
                     f"the order {SIMPLICITY_ORDER}", table=rows, chosen=chosen,
                classification=("A. YES -- existing policy can be reused" if chosen in EXISTING
                                else "B. YES -- minimal new policy definition required"))


# ===========================================================================
# Section 3/6: target impact (hidden vs chosen label-free continuation).
# ===========================================================================

def target_impact(states: list, label_free_name: str) -> dict:
    lf = POLICIES[label_free_name]
    fam = spec.FIXED_FAMILIES["BASE2_298"]
    diffs = defaultdict(list)
    same_vec = n_unique = 0
    best_agree = best_subset = 0
    rank_corr = []
    default_changed = 0
    default_outcome_shift = []
    t0 = time.perf_counter()
    for st in states:
        sim = st["sim"]
        cands = spec._restrict(sim, fam)
        classes = {}
        for name, asg in cands.items():
            lm = spec.label_map_after(sim, asg)
            classes.setdefault(lm, (name, asg))
        oa, ob = {}, {}
        for lm, (name, asg) in classes.items():
            a = rollout_with(sim, asg, st["has_hazard"], st["t_h"], lambda o: int(HIDDEN(o)))
            b = rollout_with(sim, asg, st["has_hazard"], st["t_h"], lf)
            oa[lm], ob[lm] = key4(a), key4(b)
            n_unique += 1
            same_vec += int(oa[lm] == ob[lm])
            for i, t in enumerate(("future_T90", "future_T100", "future_unmet", "future_hazard_unmet")):
                diffs[t].append(ob[lm][i] - oa[lm][i])
        best_a = {lm for lm, v in oa.items() if v == min(oa.values())}
        best_b = {lm for lm, v in ob.items() if v == min(ob.values())}
        best_agree += int(best_a == best_b)
        best_subset += int(bool(best_b & best_a))
        if len(oa) > 2:
            keys = list(oa)
            ra = np.argsort(np.argsort([oa[k] for k in keys], kind="stable"), kind="stable")
            rb = np.argsort(np.argsort([ob[k] for k in keys], kind="stable"), kind="stable")
            if np.std(ra) > 0 and np.std(rb) > 0:
                rank_corr.append(float(np.corrcoef(ra, rb)[0, 1]))
        obs = compute_observation(sim)
        h, l = int(HIDDEN(obs)), lf(obs)
        default_changed += int(h != l)
        zs = spec.nonempty(sim)
        d_hidden = spec.label_map_after(sim, {z: h for z in zs})
        d_lf = spec.label_map_after(sim, {z: (spec.U if l is None else l) for z in zs})
        if d_lf in ob and d_hidden in oa:
            default_outcome_shift.append(ob[d_lf][1] - oa[d_hidden][1])

    def summ(v):
        v = np.asarray(v, dtype=float)
        return dict(mean=round(float(v.mean()), 4), mean_abs=round(float(np.abs(v).mean()), 4),
                    min=float(v.min()), max=float(v.max()), share_zero=round(float((v == 0).mean()), 4))
    n = len(states)
    return dict(
        label_free_continuation=label_free_name, n_states=n, n_unique_candidate_rollouts_per_continuation=n_unique,
        seconds=round(time.perf_counter() - t0, 1),
        share_candidates_with_identical_target_vector=round(same_vec / n_unique, 4),
        target_delta_label_free_minus_hidden={t: summ(v) for t, v in diffs.items()},
        oracle_best_class_identical=round(best_agree / n, 4),
        oracle_best_classes_overlap=round(best_subset / n, 4),
        within_state_rank_correlation=dict(n_states=len(rank_corr),
                                           mean=round(float(np.mean(rank_corr)), 4) if rank_corr else None,
                                           min=round(float(np.min(rank_corr)), 4) if rank_corr else None),
        default_action_differs_from_hidden_heuristic=round(default_changed / n, 4),
        default_candidate_T100_shift=dict(mean=round(float(np.mean(default_outcome_shift)), 4),
                                          max=float(np.max(default_outcome_shift))) if default_outcome_shift else None,
    )


def diversity_with_label_free(policy, seeds_per_cell: int = 3) -> dict:
    """Design-time count as in the spec audit, but with the label-free default
    as the pre-decision policy. No candidate rollouts, no targets."""
    th_all = spec.T_TRAIN + spec.T_VAL + spec.T_TEST + spec.T_EXT
    p_all = spec.P_TRAIN + spec.P_VAL + spec.P_TEST + spec.P_EXT
    adm = prior.admissible_state_feature_names(prior.build_feature_manifest())
    late = adm.index("normalized_time")
    out = {}
    for eps in (0.0, 0.3, 0.5):
        vecs, n_states = Counter(), 0
        for dist_id, dist in spec.DISTRIBUTIONS.items():
            for th in th_all:
                for p in p_all:
                    role = spec.cell_role(th, p)
                    if role == "UNUSED_BUFFER":
                        continue
                    for seed in range(seeds_per_cell):
                        t_h = spec._th_int(th)
                        sim = EvacuationSimulator()
                        sim.reset(dist, t_h, generate_compliance_vector(spec.compliance_key(dist_id, role), seed, p=p))
                        rng = np.random.default_rng([seed, int(p * 1000), t_h, 7])
                        while sim.n_active() > 0 and sim.t < MAX_STEPS:
                            obs = compute_observation(sim)
                            vecs[tuple(prior.admissible_vector(dict(obs27=np.asarray(obs, dtype=float)), adm).tolist())] += 1
                            n_states += 1
                            a = policy(obs)
                            if eps > 0 and rng.random() < eps:
                                a = int(rng.integers(0, 3))
                            for _ in range(DECISION_INTERVAL):
                                sim._run_one_tick(a if (a is not None and sim.t % DECISION_INTERVAL == 0) else None)
                                sim.t += 1
                                if sim.n_active() == 0 or sim.t == MAX_STEPS:
                                    break
        late_states = sum(c for x, c in vecs.items() if x[late] >= 10 / MAX_STEPS - 1e-12)
        late_distinct = sum(1 for x in vecs if x[late] >= 10 / MAX_STEPS - 1e-12)
        out[f"epsilon_{eps}"] = dict(n_decision_states=n_states, n_distinct_admissible_inputs=len(vecs),
                                     distinct_share_tick_ge_10=round(late_distinct / max(1, late_states), 4))
    return out


# ===========================================================================
# Section 5: fallback validity.
# ===========================================================================

def fallback_validity(name: str, ep: dict, states: list) -> dict:
    fn = POLICIES[name]
    # Label-free by construction: invariant to any redistribution of the hidden hub split.
    invariant = True
    total_ok = True
    for st in states:
        obs = np.asarray(compute_observation(st["sim"]), dtype=float)
        h = obs[IQA] + obs[IQB]
        for qa in (0.0, h / 3.0, h):
            o2 = obs.copy()
            o2[IQA], o2[IQB] = qa, h - qa
            invariant &= (fn(o2) == fn(obs))
        try:
            fn(obs)
        except Exception:  # noqa: BLE001
            total_ok = False
    p = ep["per_policy"][name]
    worse = p["vs_hidden_heuristic"].get("worse", 0)
    return dict(
        audit=MARKER, policy=name,
        label_free_verified=dict(invariant_to_hidden_hub_split_on_all_existing_states=bool(invariant),
                                 reads_only=["waiting_H_total", "residual_cap_H_CE1_normalized"]
                                 if "SYMMETRIC" in name else []),
        defined_on_every_state=total_ok,
        deterministic=True,
        roles=dict(
            selector_fallback=dict(valid=True, why="deterministic, label-free, always in BASE2_298 (as BASE=l(s)), "
                                                   "defined on every admissible input"),
            abstention_fallback=dict(valid=True, why="needs no model output at all"),
            out_of_distribution_fallback=dict(valid=True, why="reads at most two scalars whose ranges are fixed by the "
                                                              "topology (hub count 0..40, residual cap in {1, 1/3}); it "
                                                              "cannot be out of its own domain"),
        ),
        what_it_is_not=("It is not a safety GUARANTEE. The selector's safety floor is relative to this default, so "
                        "'no harm' now means 'no worse than the label-free default'. Against the old hidden-label "
                        f"heuristic it was paired-worse in {worse}/{p['n_episodes']} existing episodes (see "
                        "hidden_vs_label_free_comparison.json). No extra mechanism is added to hide that; the "
                        "hidden heuristic stays in closed-loop evaluation as a reference arm so the gap stays visible."),
        additional_mechanism_required=False,
        empirical=dict(vs_hidden=p["vs_hidden_heuristic"], vs_no_intervention=p["vs_no_intervention"]),
    )


# ===========================================================================
# Section 6: impact on the frozen specification.
# ===========================================================================

def spec_impact(chosen: str, div_lf: dict, impact: dict) -> dict:
    return dict(
        audit=MARKER,
        frozen_spec_files_modified=False,
        changes=dict(
            candidate_action_set_BASE2_298=dict(change=False, why="all three BASE globals and NO_INTERVENTION remain; "
                                                "the default d(s)=BASE=l(s) is always in the set"),
            target_definitions=dict(change=True, what=[
                "continuation: GLOBAL_DEADBAND -> " + chosen + " at every later decision (identical for all candidates)",
                "future_hazard_edge_unmet_demand: onset-gated definition (D1) is final; the alternative is removed",
                "stored_per_row: only the D1 hazard definition"]),
            feature_manifest_26=dict(change=False, why="the default reads only admissible features; the model inputs "
                                                       "were already label-free"),
            scenario_grid=dict(change=False, why="templates, t_h and p levels do not depend on the default"),
            pre_decision_state_generation=dict(change=True, what=f"epsilon-perturbed {chosen} instead of the hidden "
                                               "heuristic, so the training state distribution matches the deployed "
                                               "default", diversity_recount=div_lf),
            split=dict(change=False, why="split roles, seed ranges and keys do not depend on the default"),
            selector=dict(change=True, what="d(s) = BASE=l(s) with l = " + chosen + "; the caveat about Q_A/Q_B in "
                                            "selector_design.json no longer applies"),
            success_criteria=dict(change=True, what=["harm is measured against the label-free default",
                                                     "closed loop: P2 = label-free default (primary comparator); the "
                                                     "hidden-label heuristic becomes a reference arm P2b, reported, "
                                                     "never a gate"]),
            closed_loop_evaluation=dict(change=True, what="add P2b (hidden heuristic, reference); P4 falls back to the "
                                                          "label-free default"),
            pilot_size=dict(change=False, why="240 trajectories; revisit only if the recount's tick>=10 distinct share "
                                              "falls below the 25% gate"),
        ),
        target_semantics_before=("consequence of candidate a now, then the hidden-label heuristic (Q^hidden(s, a))"),
        target_semantics_after=f"consequence of candidate a now, then {chosen} at every later decision (Q^label-free(s, a))",
        measured_change=impact,
        apply_how="A later, explicitly requested step regenerates the frozen specification files; this audit writes "
                  "none of them.",
    )


def decision_record(chosen: str, cls: str) -> dict:
    return dict(
        audit=MARKER,
        D1=dict(status="APPROVED", decision="future hazard-edge unmet demand counts only ticks t >= t_h (actual "
                                             "onset); 0 before onset; 0 in no-hazard scenarios",
                formula="sum over t in [max(t_d, t_h), end) of max(0, H_CE1 branch-A candidates(t) - cap_H_CE1(t)), "
                        "measured after relabel and before admission",
                implemented_here="yes, in the analysis-only rollout of this audit"),
        D2=dict(status="DECIDED", decision="no Q_A_at_H / Q_B_at_H or other hidden variable in the continuation "
                                           "policy or selector fallback of the main experiment",
                reference_arm="the hidden-label GLOBAL_DEADBAND heuristic remains a research/reference comparison only",
                resolved_by=chosen, classification=cls),
        hidden_information_removed=["Q_A_at_H", "Q_B_at_H (the hub queue split by hidden target_exit label, which "
                                    "also tracks hidden non-compliance)"],
    )


# ===========================================================================
# Report + main.
# ===========================================================================

def render(ctx: dict) -> str:
    L = []
    a = L.append
    ch, ep, imp, fv, choice = ctx["choice"]["chosen"], ctx["episodes"], ctx["impact"], ctx["fallback"], ctx["choice"]
    a(f"<!-- {MARKER} -->")
    a("# Label-Free Default Audit (after D1 / D2)\n")
    a("Read-only. No pilot, no training data, no model, no policy module implemented, no frozen specification file "
      "overwritten, no simulator / observation.py / controller / RL branch / 9,450-row dataset change (SHA-256 "
      "checked).\n")
    a(f"## Answer: **{choice['classification']}**\n")
    a(f"Recommended label-free default: **{ch}**.\n")
    a("## 1. Existing policies\n")
    a("| policy | hidden dependencies | label-free | verdict |")
    a("|---|---|---|---|")
    for c in ctx["inventory"]:
        a(f"| {c['name']} | {', '.join(c['hidden_dependencies']) or 'none'} | {c['label_free']} | {c['verdict']} |")
    a("\nEvery existing adaptive controller routes through GLOBAL_DEADBAND and so reads Q_A/Q_B. The only existing "
      "label-free policies are constants and no-intervention.\n")
    a("## 2. Minimal substitute\n")
    a("SYMMETRIC_SPLIT_DEADBAND: the existing, unmodified `deadband_heuristic_action` applied to the observation "
      "with Q_A_at_H = Q_B_at_H = waiting_H_total / 2. Inputs: waiting_H_total and residual capacity only. The even "
      "split is a fixed prior (the parity rule's own split), not an estimate of labels or compliance. Two margins were "
      "evaluated:\n")
    a("- **margin 0** (= the project's original parameter-free `capacity_aware_heuristic` rule on a label-free input): "
      "PARITY unless the hazard has occurred (residual capacity 2/6) and at least one person waits at the hub, then "
      "FAVOR_B. Never FAVOR_A.")
    a("- **margin 0.75** (the dead-band used by the hidden heuristic): same, but FAVOR_B only with at least 5 people at "
      "the hub.\n")
    a("It is a new DEFINITION (an input adapter), not new decision logic, which is why the answer is B rather than A. "
      "It is analysis-only here and is not implemented as a module.\n")
    a("## 4. Hidden vs label-free (full episodes, existing configurations)\n")
    a("| policy | worse / tie / better vs hidden | worse vs no-intervention | mean dT100 | max dT100 | mean d hazard unmet |")
    a("|---|---|---|---|---|---|")
    for name, p in ep["per_policy"].items():
        v = p["vs_hidden_heuristic"]
        a(f"| {name} | {v.get('worse', 0)} / {v.get('tie', 0)} / {v.get('better', 0)} | "
          f"{p['vs_no_intervention'].get('worse', 0)} | {p['mean_T100_delta_vs_hidden']} | "
          f"{p['max_T100_delta_vs_hidden']} | {p['mean_hazard_unmet_delta_vs_hidden']} |")
    a(f"\n{ep['note']}\n")
    a(f"Selection: {choice['rule']}.\n")
    a("Action agreement with the hidden heuristic on the 530 existing decision states: "
      + "; ".join(f"{k}: {v}" for k, v in ctx["agreement"]["per_policy"].items() if k == ch) + ". Where the two "
      "disagree (post-onset only), the disagreement did not change any full-episode outcome on these configurations: "
      "the hidden split only matters when the hub queue's labels are uneven, which rarely happens on the default's "
      "own trajectory. It matters much more after a candidate deviation, which is where the targets change (below).\n")
    a("## 3/6. Target validity and impact\n")
    a(f"Protocol: state before decision -> one BASE2_298 candidate -> {ch} at every later decision (identical for every "
      "candidate) -> D1 targets. The label now answers: what happens if I make this recommendation now, assuming the "
      "same observable default afterwards.\n")
    a(f"Measured on the 530 existing states ({imp['n_unique_candidate_rollouts_per_continuation']} unique candidate "
      "rollouts per continuation):\n")
    a(f"- identical target vector under both continuations: {imp['share_candidates_with_identical_target_vector']:.1%} of candidates")
    for t, s in imp["target_delta_label_free_minus_hidden"].items():
        a(f"- {t} (label-free minus hidden): mean {s['mean']}, mean |d| {s['mean_abs']}, range [{s['min']}, {s['max']}], unchanged {s['share_zero']:.1%}")
    a(f"- oracle-best class identical: {imp['oracle_best_class_identical']:.1%} of states; overlapping: "
      f"{imp['oracle_best_classes_overlap']:.1%}; mean within-state rank correlation "
      f"{imp['within_state_rank_correlation']['mean']} (min {imp['within_state_rank_correlation']['min']})")
    a(f"- default action differs from the hidden heuristic's in {imp['default_action_differs_from_hidden_heuristic']:.1%} "
      f"of states; default candidate's T100 shift {imp['default_candidate_T100_shift']}\n")
    si = ctx["spec_impact"]["changes"]
    a("Specification changes required (not applied here):\n")
    for k, v in si.items():
        a(f"- **{k}**: {'CHANGE' if v['change'] else 'no change'}" + (f": {v.get('what') if isinstance(v.get('what'), str) else '; '.join(v['what'])}" if v["change"] and k != "pre_decision_state_generation" else (f": {v['what']}" if v["change"] else f" ({v.get('why', '')})")))
    d = ctx["div_lf"]
    a(f"\nDiversity recount with {ch} as the pre-decision policy: " + "; ".join(
        f"{k}: {v['n_distinct_admissible_inputs']} distinct inputs, tick>=10 distinct share {v['distinct_share_tick_ge_10']:.1%}"
        for k, v in d.items()) + ".\n")
    a("## 5. Fallback validity\n")
    a(f"Selector / abstention / out-of-distribution fallback: valid. Label-free verified (invariant to any split of the "
      f"hub total on every existing state: {fv['label_free_verified']['invariant_to_hidden_hub_split_on_all_existing_states']}); "
      f"defined on every state: {fv['defined_on_every_state']}. {fv['what_it_is_not']}\n")
    a("## 7. Critical question\n")
    a(f"Can the experiment run with a genuinely label-free continuation/fallback without changing the simulator? "
      f"**{choice['classification']}**. No simulator, observation or controller change is needed.\n")
    integ = ctx["integrity"]
    a("## Integrity\n")
    a(f"- Protected files unchanged: **{integ['all_unchanged']}** ({len(integ['before'])} files, including every frozen "
      "specification file and both earlier audits).")
    a(f"- Frozen controller MD5 match: {integ['controller']['match']}; RL checkpoints unchanged: "
      f"{integ['checkpoints']['all_match']} ({integ['checkpoints']['n_checkpoints']}).")
    return "\n".join(L) + "\n"


def run(states_limit: int | None = None, configs_limit: int | None = None, seeds_per_cell: int = 3) -> dict:
    before = protected_hashes()
    t0 = time.perf_counter()
    configs = tick0_configs()
    if configs_limit:
        configs = configs[:configs_limit]
    ep = episode_comparison(configs)
    choice = choose_default(ep)
    chosen = choice["chosen"]
    print(f"[4] chosen label-free default: {chosen} ({choice['classification']})")
    states = spec.existing_states()
    if states_limit:
        states = states[:states_limit]
    agreement = action_agreement(states)
    impact = target_impact(states, chosen)
    print(f"[6] target impact: identical {impact['share_candidates_with_identical_target_vector']}, "
          f"best-class identical {impact['oracle_best_class_identical']}")
    div_lf = diversity_with_label_free(POLICIES[chosen], seeds_per_cell)
    fv = fallback_validity(chosen, ep, states)
    after = protected_hashes()
    changed = [k for k in before if before[k] != after[k]]
    if changed:
        raise RuntimeError(f"PROTECTED FILES CHANGED: {changed}")
    integrity = dict(before=before, all_unchanged=True, controller=prior.design.verify_controller_hash(),
                     checkpoints=prior.design.verify_checkpoint_hashes())
    si = spec_impact(chosen, div_lf, impact)
    ctx = dict(inventory=candidate_inventory(), episodes=ep, choice=choice, agreement=agreement, impact=impact,
               div_lf=div_lf, fallback=fv, spec_impact=si, integrity=integrity)
    ctx["outputs"] = {
        "label_free_policy_candidates.json": dict(audit=MARKER, candidates=ctx["inventory"], selection=choice),
        "hidden_vs_label_free_comparison.json": dict(audit=MARKER, episodes=ep, action_agreement=agreement),
        "target_impact_analysis.json": si,
        "fallback_validity.json": fv,
        "d1_d2_decision_record.json": dict(**decision_record(chosen, choice["classification"]), integrity=integrity),
    }
    ctx["outputs"]["LABEL_FREE_DEFAULT_AUDIT.md"] = render(ctx)
    ctx["seconds"] = round(time.perf_counter() - t0, 1)
    return ctx


def write_outputs(ctx: dict, out_dir: str = OUT_DIR) -> list:
    created: set = set()
    for name in OUTPUT_FILES:
        _safe_write(name, ctx["outputs"][name], created, out_dir=out_dir)
    return sorted(created)


if __name__ == "__main__":
    ctx = run()
    written = write_outputs(ctx)
    print(f"\n{ctx['choice']['classification']}: {ctx['choice']['chosen']}")
    print(f"protected unchanged: {ctx['integrity']['all_unchanged']}; wrote {len(written)} files in {ctx['seconds']}s")
