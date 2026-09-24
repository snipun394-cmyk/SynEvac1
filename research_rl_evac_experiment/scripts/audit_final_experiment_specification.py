"""READ-ONLY final experiment-specification / data-generation design audit.

Freezes (on paper) the consequence-prediction experiment that the
pre-training audit (PRE_TRAINING_DATA_INTEGRITY_AUDIT.md, decision C)
said needs new simulation data. It trains nothing, generates no dataset,
runs no pilot, implements no selector, and never writes to the simulator,
observation.py, the controller, the frozen RL branch, or the existing
9,450-row dataset.

What it COMPUTES (all in memory, nothing persisted except summary counts):

  1. Source checks of the three candidate semantics that coexist in the code.
  2. A candidate-set sufficiency analysis on EXISTING decision states only
     (the 450 differential-targeting states and the 80 timing-matrix
     states): every absolute per-zone assignment is rolled out from the
     replayed state with the proposed target definitions, and each
     candidate-set proposal is scored by how often its best member matches
     the unrestricted best.
  3. A verification that the proposed "realized" unmet-demand measurement
     matches the simulator's own tick_log exactly.
  4. A check of how compliance vectors relate across p / t_h for one key.
  5. A DESIGN-TIME DIVERSITY COUNT for the proposed factorial grid:
     pre-decision reference trajectories only (no candidate rollouts, no
     targets, no rows kept), counting distinct admissible input vectors.
     This is not the pilot: it produces no training data.

Writes only the deliverables in OUTPUT_FILES (see `_safe_write`).
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)
import audit_pre_training_ai_dataset as prior  # noqa: E402  (sets up DATA_ROOT import paths)

from actions import apply_action  # noqa: E402
from compliance import generate_compliance_vector  # noqa: E402
from differential_rl_action_space import decode_action  # noqa: E402
from differential_targeting import apply_differential_action  # noqa: E402
from metrics import compute_metrics  # noqa: E402
from observation import compute_observation  # noqa: E402
from simulator import EvacuationSimulator  # noqa: E402
import stress_test_scenarios as ST  # noqa: E402
from timing_experiment import diagnostic_all_edges_unmet  # noqa: E402
import topology as topology_module  # noqa: E402
from topology import DECISION_INTERVAL, MAX_STEPS, N_OCCUPANTS  # noqa: E402

DATA_ROOT = prior.DATA_ROOT
OUT_DIR = prior.OUT_DIR
HEURISTIC = prior.asgc.HEURISTIC
ELIGIBLE_ZONES = list(prior.asgc.ELIGIBLE_ZONES)
MARKER = "final_experiment_specification_audit_v1"

OUTPUT_FILES = [
    "FINAL_EXPERIMENT_SPECIFICATION.md",
    "candidate_action_set.json",
    "target_definitions.json",
    "final_feature_manifest.json",
    "scenario_space.json",
    "dataset_design.json",
    "diversity_requirements.json",
    "split_specification.json",
    "selector_design.json",
    "closed_loop_evaluation.json",
    "preregistered_success_criteria.json",
]
SUPERSEDES = {"preregistered_success_criteria.json"}  # written by the prior audit; replaced, prior content embedded

PRIOR_DELIVERABLES = [f for f in prior.OUTPUT_FILES if f not in SUPERSEDES]
PROTECTED_DATA_PATHS = list(prior.PROTECTED_RELATIVE_PATHS) + [
    "differential_rl_env.py", "arbitration_engine.py", "scripts/audit_observable_robust_generalization.py",
    "artifacts/hidden_compliance_robust_control/equivalence_classes.json",
    "artifacts/hidden_compliance_robust_control/observable_only_upper_bound.json",
]

U, FA, PA, FB = "U", 0, 1, 2
VALUE_NAMES = {U: "UNTOUCHED", FA: "FAVOR_A", PA: "PARITY", FB: "FAVOR_B"}
GROUPS = {"WEST": ["R1", "R2", "C1"], "EAST": ["R3", "R4", "C2"], "HUB": ["H"]}


# ===========================================================================
# Integrity.
# ===========================================================================

def protected_hashes() -> dict:
    out = {}
    for rel in dict.fromkeys(PROTECTED_DATA_PATHS):
        p = os.path.join(DATA_ROOT, rel)
        out[f"DATA_ROOT/{rel}"] = prior.sha256_file(p) if os.path.isfile(p) else "MISSING"
    for name in PRIOR_DELIVERABLES:
        p = os.path.join(OUT_DIR, name)
        out[f"OUT_DIR/{name}"] = prior.sha256_file(p) if os.path.isfile(p) else "MISSING"
    for rel in ("scripts/audit_pre_training_ai_dataset.py", "tests/test_pre_training_ai_dataset.py"):
        p = os.path.join(prior.OUT_ROOT, rel)
        out[f"OUT_ROOT/{rel}"] = prior.sha256_file(p) if os.path.isfile(p) else "MISSING"
    return out


def _safe_write(name: str, content, created: set, out_dir: str = OUT_DIR) -> str:
    if name not in OUTPUT_FILES:
        raise PermissionError(f"refusing to write non-deliverable {name!r}")
    path = os.path.join(out_dir, name)
    if os.path.exists(path) and name not in created:
        with open(path, encoding="utf-8") as f:
            existing = f.read()
        ok = MARKER in existing or (name in SUPERSEDES and prior.AUDIT_MARKER in existing)
        if not ok:
            raise PermissionError(f"refusing to overwrite a file not produced by these audits: {path}")
    text = content if isinstance(content, str) else json.dumps(content, indent=2, default=prior._json_default)
    if MARKER not in text:
        raise ValueError("deliverable missing marker")
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    created.add(name)
    return path


def load_prior_audit() -> dict:
    def j(n):
        with open(os.path.join(OUT_DIR, n), encoding="utf-8") as f:
            return json.load(f)
    return dict(split=j("split_design.json"), manifest=j("exact_feature_manifest.json"),
                target=j("target_generation_audit.json"), ident=j("target_identifiability.json"),
                traj=j("trajectory_dependence.json"))


def previous_preregistration() -> dict | None:
    p = os.path.join(OUT_DIR, "preregistered_success_criteria.json")
    if not os.path.isfile(p):
        return None
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    if d.get("audit") == MARKER:
        return d.get("previous_draft_superseded")
    return dict(sha256=prior.sha256_file(p), content=d)


# ===========================================================================
# Section 3: candidate semantics + sufficiency.
# ===========================================================================

def semantics_inventory() -> dict:
    def src(rel):
        with open(os.path.join(DATA_ROOT, rel), encoding="utf-8") as f:
            return f.read()
    env, edr, asp = src("differential_rl_env.py"), src("scripts/evaluate_differential_rl.py"), src("differential_rl_action_space.py")
    return dict(
        S1_leave_untouched_pairwise=dict(
            where="scripts/audit_differential_targeting.py::run_differential_intervention (the 9,450-row dataset)",
            rule="relabel only the targeted zones; every other zone keeps its current labels"),
        S2_apply_high_level_action=dict(
            where="differential_rl_action_space.apply_high_level_action",
            rule="action 0 = heuristic globally; overrides relabel only their zones, others untouched",
            code_confirms="apply_differential_action(sim, assignments)" in asp and "if not assignments:" in asp),
        S3_global_then_override=dict(
            where="differential_rl_env.DifferentialRLGymEnv.step and evaluate_differential_rl._oracle_candidate_continuation "
                  "(also the frozen robust controller's audited actions)",
            rule="heuristic globally first, then overrides overwrite their zones",
            code_confirms=("apply_action(sim, heuristic_action, target_group=None)" in env
                           and "apply_action(sim, heuristic_action, target_group=None)" in edr)),
        finding="Three different meanings exist for 'the same' differential action id. All three are special cases "
                "of an ABSOLUTE per-zone assignment over the currently non-empty eligible zones with values "
                "{UNTOUCHED, FAVOR_A, PARITY, FAVOR_B}; the frozen spec uses that absolute representation so a "
                "candidate's meaning never depends on the heuristic's output.",
    )


def nonempty(sim) -> list:
    return [z for z in ELIGIBLE_ZONES if len(sim.zone_waiting[z]) > 0]


def apply_assignment(sim, assignment: dict) -> None:
    """assignment: zone -> value in {U, 0, 1, 2}. Groups are built from the
    pre-decision state before any relabel (apply_differential_action)."""
    pairs = [(z, v) for z, v in assignment.items() if v != U]
    if pairs:
        apply_differential_action(sim, pairs)


def label_map_after(sim, assignment: dict) -> tuple:
    fork = copy.deepcopy(sim)
    apply_assignment(fork, assignment)
    return tuple(o["target_exit"] for _, o in sorted(fork.occupants.items()))


def rollout(sim0, assignment: dict, has_hazard: bool, t_h: int) -> dict:
    """Proposed protocol: apply the candidate at decision tick t_d, then the
    frozen heuristic at every later decision tick. Unmet demand is measured
    AFTER the tick's relabel and BEFORE admission (what admission sees)."""
    sim = copy.deepcopy(sim0)
    t_d = sim.t
    evac_before = dict(sim.evacuated)
    apply_assignment(sim, assignment)
    unmet, haz, haz_all_ticks = 0, 0, 0
    first = True
    while sim.n_active() > 0 and sim.t < MAX_STEPS:
        if sim.t % DECISION_INTERVAL == 0 and not first:
            apply_action(sim, int(HEURISTIC(compute_observation(sim))))
        first = False
        unmet += diagnostic_all_edges_unmet(sim, sim.t, t_h, topology_module.edge_capacity)
        _, hu, _ = ST.diagnostic_branch_demand_and_unmet(sim, "H_CE1", "A", sim.t, t_h, topology_module.edge_capacity)
        haz_all_ticks += hu
        if has_hazard and sim.t >= t_h:
            haz += hu
        sim._run_one_tick(None)
        sim.t += 1
    m = compute_metrics(sim.evacuated, N=N_OCCUPANTS, max_steps=MAX_STEPS)
    realized_h = sum(e["h_ce1_unmet"] for e in sim.tick_log if e["t"] >= t_d)
    realized_h_onset = sum(e["h_ce1_unmet"] for e in sim.tick_log if e["t"] >= t_d and has_hazard and e["t"] >= t_h)
    n_before = len(evac_before)
    t90_pre = n_before >= int(np.ceil(0.9 * N_OCCUPANTS))
    return dict(
        T90_abs=m["T90"] if m["T90_defined"] else float("inf"), T100_abs=m["T100"], T100_censored=m["T100_censored"],
        future_T90=(0.0 if t90_pre else ((m["T90"] - t_d) if m["T90_defined"] else float("inf"))),
        T90_reached_before_decision=t90_pre,
        future_T100=m["T100"] - t_d,
        future_unmet=unmet, future_hazard_unmet=haz,
        check_hazard_all_ticks_equals_tick_log=(haz_all_ticks == realized_h),
        check_hazard_onset_equals_tick_log=(haz == realized_h_onset),
    )


def key4(o: dict) -> tuple:
    return (o["future_T90"], o["future_T100"], o["future_unmet"], o["future_hazard_unmet"])


def full_space(sim) -> list:
    zs = nonempty(sim)
    return [dict(zip(zs, vals)) for vals in itertools.product([U, FA, PA, FB], repeat=len(zs))]


SHORT = {U: "U", FA: "A", PA: "P", FB: "B"}


def _grouped_family(groups: dict, values: list, add_no_intervention: bool) -> dict:
    """Fixed, state-independent family: every group gets one value; returns
    name -> full 7-zone absolute assignment."""
    fam = {}
    if add_no_intervention:
        fam["NO_INTERVENTION"] = {z: U for z in ELIGIBLE_ZONES}
    for combo in itertools.product(values, repeat=len(groups)):
        a = {}
        for (g, zs), v in zip(groups.items(), combo):
            for z in zs:
                a[z] = v
        fam["|".join(f"{g}={SHORT[v]}" for g, v in zip(groups, combo))] = a
    return fam


def _base_override_family(max_overrides: int) -> dict:
    fam = {"NO_INTERVENTION": {z: U for z in ELIGIBLE_ZONES}}
    for base in (FA, PA, FB):
        for k in range(0, max_overrides + 1):
            for zs in itertools.combinations(ELIGIBLE_ZONES, k):
                for vals in itertools.product([v for v in (FA, PA, FB)], repeat=k):
                    if any(v == base for v in vals):
                        continue
                    a = {z: base for z in ELIGIBLE_ZONES}
                    a.update(dict(zip(zs, vals)))
                    fam[f"BASE={SHORT[base]}" + "".join(f"|{z}={SHORT[v]}" for z, v in zip(zs, vals))] = a
    return fam


FIXED_FAMILIES = {
    "GLOBAL4": {"NO_INTERVENTION": {z: U for z in ELIGIBLE_ZONES},
                **{f"GLOBAL_{SHORT[v]}": {z: v for z in ELIGIBLE_ZONES} for v in (FA, PA, FB)}},
    "SIDE28": _grouped_family(GROUPS, [FA, PA, FB], True),
    "SIDE64": _grouped_family(GROUPS, [U, FA, PA, FB], False),
    "PAIRED82": _grouped_family({"FIRST": ["R1", "R3"], "SECOND": ["R2", "R4"], "CORR": ["C1", "C2"], "HUB": ["H"]},
                                [FA, PA, FB], True),
    "BASE1_46": _base_override_family(1),
    "BASE2_298": _base_override_family(2),
}


def _restrict(sim, fam: dict) -> dict:
    zs = nonempty(sim)
    return {n: {z: a[z] for z in zs} for n, a in fam.items()}


def cand_e211(sim, h) -> dict:
    zs = set(nonempty(sim))
    out = {}
    for a in range(211):
        asg = decode_action(a)
        if not {z for z, _ in asg} <= zs:
            continue
        d = {z: h for z in zs}
        d.update(dict(asg))
        out[f"E211_{a}"] = d
    return out


def cand_dt(sim, h) -> dict:
    zs = nonempty(sim)
    out = {"GLOBAL_H": {z: h for z in zs}, "NO_INTERVENTION": {z: U for z in zs}}
    for zi, zj in itertools.combinations(zs, 2):
        for a, b in itertools.product((FA, PA, FB), repeat=2):
            d = {z: U for z in zs}
            d[zi], d[zj] = a, b
            out[f"{zi}={a}|{zj}={b}"] = d
    return out


CANDIDATE_SETS = {name: (lambda sim, h, fam=fam: _restrict(sim, fam)) for name, fam in FIXED_FAMILIES.items()}
CANDIDATE_SETS["E211_global_then_override"] = cand_e211          # state-dependent, comparison only
CANDIDATE_SETS["DT_leave_untouched_pairwise"] = cand_dt            # state-dependent, comparison only


def existing_states() -> list:
    """The 450 differential-targeting states (replayed) and the 80 timing-matrix
    tick-0 states (rebuilt with the same helper that produced them)."""
    out = []
    for r in prior.load_records():
        s = prior.replay_state(r["scenario"], r["seed"], r["decision_tick"])
        out.append(dict(src="dt450", key=f"{r['scenario']}|{r['seed']}|{r['decision_tick']}", sim=s["sim"],
                        has_hazard=s["has_hazard"], t_h=s["t_h"]))
    sys.path.insert(0, os.path.join(DATA_ROOT, "scripts"))
    import audit_observable_robust_generalization as aorg
    with open(prior.TIMING_MATRIX_PATH, encoding="utf-8") as f:
        tm = json.load(f)
    for r in tm:
        sim = aorg.build_custom_sim_at_tick0(dict(config_id="sweep", distribution=aorg.DIST_A1_SYMMETRIC,
                                                  t_h=r["t_h"], p=0.75), seed=r["seed"])
        out.append(dict(src="timing80", key=f"th{r['t_h']}|{r['seed']}|0", sim=sim, has_hazard=True, t_h=r["t_h"]))
    return out


def candidate_sufficiency(states: list) -> dict:
    """For every existing state: roll out every behaviorally distinct absolute
    assignment once (cached by resulting label map), find the unrestricted best
    under the established lexicographic order, and score each candidate set."""
    per_set = {k: dict(n_states=0, best_matches_full=0, t90_regret=[], t100_regret=[], unmet_regret=[],
                       n_candidates=[], n_distinct=[], full_best_in_set=0) for k in CANDIDATE_SETS}
    per_candidate = {f: defaultdict(Counter) for f in FIXED_FAMILIES}
    checks = Counter()
    n_rollouts = 0
    t90_pre = 0
    t0 = time.perf_counter()
    for st in states:
        sim = st["sim"]
        h = int(HEURISTIC(compute_observation(sim)))
        cache = {}

        def outcome(asg):
            nonlocal n_rollouts
            lm = label_map_after(sim, asg)
            if lm not in cache:
                cache[lm] = rollout(sim, asg, st["has_hazard"], st["t_h"])
                n_rollouts += 1
                o = cache[lm]
                checks["hazard_all_ticks_matches_tick_log"] += int(o["check_hazard_all_ticks_equals_tick_log"])
                checks["hazard_onset_matches_tick_log"] += int(o["check_hazard_onset_equals_tick_log"])
                checks["n"] += 1
            return cache[lm], lm

        full = [outcome(a) for a in full_space(sim)]
        full_best = min(key4(o) for o, _ in full)
        t90_pre += int(full[0][0]["T90_reached_before_decision"])
        for name, fn in CANDIDATE_SETS.items():
            cands = fn(sim, h)
            res = [outcome(a) for a in cands.values()]
            best = min(key4(o) for o, _ in res)
            if name in FIXED_FAMILIES:
                glob = {v: outcome({z: v for z in nonempty(sim)})[1] for v in (FA, PA, FB)}
                for (cname, asg), (o, lm) in zip(cands.items(), res):
                    pc = per_candidate[name][cname]
                    pc["in_unrestricted_best_class"] += int(key4(o) == full_best)
                    base = next((v for v in (FA, PA, FB) if cname.startswith(f"BASE={SHORT[v]}")), None)
                    if base is not None:
                        pc["behaviorally_differs_from_its_base_global"] += int(lm != glob[base])
                    pc["behaviorally_differs_from_every_global"] += int(lm not in glob.values())
            ps = per_set[name]
            ps["n_states"] += 1
            ps["best_matches_full"] += int(best == full_best)
            ps["t90_regret"].append(best[0] - full_best[0])
            ps["t100_regret"].append(best[1] - full_best[1])
            ps["unmet_regret"].append(best[2] - full_best[2])
            ps["n_candidates"].append(len(cands))
            ps["n_distinct"].append(len({lm for _, lm in res}))
    summary = {}
    for name, ps in per_set.items():
        summary[name] = dict(
            n_states=ps["n_states"],
            share_states_where_set_best_equals_unrestricted_best=round(ps["best_matches_full"] / ps["n_states"], 4),
            mean_T100_regret_ticks=round(float(np.mean(ps["t100_regret"])), 4),
            max_T100_regret_ticks=float(np.max(ps["t100_regret"])),
            mean_T90_regret_ticks=round(float(np.mean(ps["t90_regret"])), 4),
            mean_future_unmet_regret=round(float(np.mean(ps["unmet_regret"])), 4),
            candidates_per_state=dict(min=min(ps["n_candidates"]), max=max(ps["n_candidates"])),
            behaviorally_distinct_per_state=dict(min=min(ps["n_distinct"]), max=max(ps["n_distinct"]),
                                                 mean=round(float(np.mean(ps["n_distinct"])), 2)),
        )
    return dict(
        states_used=dict(Counter(s["src"] for s in states)),
        note="EXISTING states only (no new configuration). The unrestricted reference is every absolute assignment "
             "{UNTOUCHED, FAVOR_A, PARITY, FAVOR_B} over the non-empty eligible zones. Outcomes use the proposed "
             "future-target definitions and heuristic continuation. Rollouts are cached by resulting label map, "
             "since identical label maps give identical deterministic outcomes.",
        n_unique_rollouts=n_rollouts, n_states_T90_reached_before_decision=t90_pre,
        seconds=round(time.perf_counter() - t0, 1),
        measurement_checks=dict(checks),
        per_candidate_set=summary,
        per_candidate_evidence={f: {c: dict(v) for c, v in d.items()} for f, d in per_candidate.items()},
    )


def known_interventions_expressible() -> dict:
    """Are the previously useful interventions expressible in each fixed
    family at the states where they were found?"""
    out = {}
    cases = [("S2", 10000, 0, "robust_controller_S2_action_102", ("E211", 102)),
             ("S3", 10000, 0, "robust_controller_S3_action_6", ("E211", 6)),
             ("S2", 10000, 0, "dt_winner_R1=1|R2=2", ("DT", [("R1", 1), ("R2", 2)])),
             ("S2", 10001, 0, "dt_winner_R1=2|R2=2", ("DT", [("R1", 2), ("R2", 2)]))]
    for sc, seed, tick, name, (kind, spec) in cases:
        s = prior.replay_state(sc, seed, tick)
        sim = s["sim"]
        h = int(HEURISTIC(compute_observation(sim)))
        zs = nonempty(sim)
        if kind == "E211":
            asg = {z: h for z in zs}
            asg.update(dict(decode_action(spec)))
            desc = decode_action(spec)
        else:
            asg = {z: U for z in zs}
            asg.update(dict(spec))
            desc = spec
        target_lm = label_map_after(sim, asg)
        target_out = key4(rollout(sim, asg, s["has_hazard"], s["t_h"]))
        fams = {}
        for fname, fam in FIXED_FAMILIES.items():
            cands = _restrict(sim, fam)
            lms = {k: label_map_after(sim, a) for k, a in cands.items()}
            cache = {}
            for k, a in cands.items():
                if lms[k] not in cache:
                    cache[lms[k]] = key4(rollout(sim, a, s["has_hazard"], s["t_h"]))
            best = min(cache.values())
            fams[fname] = dict(exactly_expressible=[k for k, lm in lms.items() if lm == target_lm][:4],
                               best_outcome=list(best), best_at_least_as_good=bool(best <= target_out))
        out[name] = dict(state=f"{sc}|{seed}|tick{tick}", assignment=[list(x) for x in desc], heuristic_action=h,
                         outcome=list(target_out), by_family=fams)
    return out


def choose_candidate_set(suff: dict, known: dict) -> dict:
    """Smallest FIXED family that (a) matches the unrestricted best in >= 99%
    of existing states with zero max T100 regret and (b) reaches an outcome at
    least as good as every known useful intervention."""
    rows = []
    for fname, fam in sorted(FIXED_FAMILIES.items(), key=lambda kv: len(kv[1])):
        s = suff["per_candidate_set"][fname]
        k_ok = all(v["by_family"][fname]["best_at_least_as_good"] for v in known.values())
        ok = (s["share_states_where_set_best_equals_unrestricted_best"] >= 0.99 and s["max_T100_regret_ticks"] == 0
              and k_ok)
        rows.append(dict(family=fname, size=len(fam), match=s["share_states_where_set_best_equals_unrestricted_best"],
                         max_T100_regret=s["max_T100_regret_ticks"], known_interventions_reached=k_ok, qualifies=ok))
    chosen = next((r["family"] for r in rows if r["qualifies"]), None)
    return dict(rule="smallest fixed family with >= 99% best-match, 0 max T100 regret, and every known intervention "
                     "reached (outcome at least as good)", table=rows, chosen=chosen)


# ===========================================================================
# Section 7-9: scenario space + design-time diversity count.
# ===========================================================================

DISTRIBUTIONS = {
    "D0_sym": dict(R1=10, R2=10, R3=10, R4=10),
    "D1_east_heavy": dict(R1=5, R2=5, R3=15, R4=15),
    "D2_west_heavy": dict(R1=15, R2=15, R3=5, R4=5),
    "D3_first_room_heavy": dict(R1=15, R2=5, R3=15, R4=5),
    "D4_second_room_heavy": dict(R1=5, R2=15, R3=5, R4=15),
    "D5_R1_concentrated": dict(R1=25, R2=5, R3=5, R4=5),
    "Z1_R4_concentrated": dict(R1=5, R2=5, R3=5, R4=25),
    "Z2_mixed": dict(R1=7, R2=13, R3=13, R4=7),
}
TRAIN_TEMPLATES = ["D0_sym", "D1_east_heavy", "D2_west_heavy", "D3_first_room_heavy", "D4_second_room_heavy",
                   "D5_R1_concentrated"]
ZERO_SHOT_TEMPLATES = ["Z1_R4_concentrated", "Z2_mixed"]
T_TRAIN, T_VAL, T_TEST, T_EXT = ["none", "3", "5", "7", "9"], ["6"], ["4", "8"], ["10"]
P_TRAIN, P_VAL, P_TEST, P_EXT = [0.5, 0.7, 0.9], [0.8], [0.6, 0.85], [0.95]
EPSILON = 0.3
EPSILONS_COUNTED = (0.0, 0.3, 0.5)
# Filled by run() from the sufficiency analysis: name, size, mean distinct rollouts per state.
CHOSEN = dict(name=None, size=None, distinct_mean=None)


def cell_role(th: str, p: float) -> str:
    if th in T_TRAIN and p in P_TRAIN:
        return "TRAIN"
    if (th in T_VAL and p in P_TRAIN + P_VAL) or (th in T_TRAIN and p in P_VAL):
        return "VALIDATION"
    if th in T_TEST and p in P_TEST:
        return "TEST_PRIMARY_JOINT"
    if th in T_TEST and p in P_TRAIN:
        return "TEST_HELDOUT_TIMING"
    if th in T_TRAIN and p in P_TEST:
        return "TEST_HELDOUT_COMPLIANCE"
    if th in T_EXT or p in P_EXT:
        return "DIAG_EXTRAPOLATION"
    return "UNUSED_BUFFER"


def _th_int(th: str) -> int:
    return MAX_STEPS + 1 if th == "none" else int(th)


def compliance_key(dist_id: str, split_role: str) -> str:
    return f"AILX_v1|{split_role}|{dist_id}"


def reference_decision_states(dist: dict, th: str, p: float, seed: int, key: str, epsilon: float):
    """Pre-decision generation only: heuristic, optionally epsilon-perturbed
    with a uniformly random GLOBAL action (seeded RNG). Yields the admissible
    input vector at every decision tick. No candidate rollout, no target."""
    t_h = _th_int(th)
    cv = generate_compliance_vector(key, seed, p=p)
    sim = EvacuationSimulator()
    sim.reset(dist, t_h, cv)
    rng = np.random.default_rng([seed, int(p * 1000), t_h, 7])
    adm = prior.admissible_state_feature_names(prior.build_feature_manifest())
    while sim.n_active() > 0 and sim.t < MAX_STEPS:
        obs = compute_observation(sim)
        yield dict(tick=sim.t, x=tuple(prior.admissible_vector(dict(obs27=np.asarray(obs, dtype=float)), adm).tolist()),
                   t90_reached=len(sim.evacuated) >= 36)
        a = int(HEURISTIC(obs))
        if epsilon > 0 and rng.random() < epsilon:
            a = int(rng.integers(0, 3))
        for _ in range(DECISION_INTERVAL):
            sim._run_one_tick(a if sim.t % DECISION_INTERVAL == 0 else None)
            sim.t += 1
            if sim.n_active() == 0 or sim.t == MAX_STEPS:
                break


def diversity_count(seeds_per_cell: int = 3) -> dict:
    th_all = T_TRAIN + T_VAL + T_TEST + T_EXT
    p_all = P_TRAIN + P_VAL + P_TEST + P_EXT
    res = {}
    t0 = time.perf_counter()
    late_idx = prior.admissible_state_feature_names(prior.build_feature_manifest()).index("normalized_time")
    for eps in EPSILONS_COUNTED:
        vecs_by_role = defaultdict(set)
        all_vecs = Counter()
        per_seed_growth = {k: set() for k in range(1, seeds_per_cell + 1)}
        by_tick = defaultdict(set)
        n_states = n_traj = t90_pre = 0
        ticks_per_traj = []
        vec_by_template = defaultdict(set)
        for dist_id, dist in DISTRIBUTIONS.items():
            for th in th_all:
                for p in p_all:
                    role = cell_role(th, p)
                    if role == "UNUSED_BUFFER":
                        continue
                    if dist_id in ZERO_SHOT_TEMPLATES:
                        role = "DIAG_TEMPLATE_ZERO_SHOT"
                    for k, seed in enumerate(range(seeds_per_cell)):
                        n_traj += 1
                        nt = 0
                        for d in reference_decision_states(dist, th, p, seed, compliance_key(dist_id, role), eps):
                            n_states += 1
                            nt += 1
                            t90_pre += int(d["t90_reached"])
                            all_vecs[d["x"]] += 1
                            vecs_by_role[role].add(d["x"])
                            vec_by_template[dist_id].add(d["x"])
                            by_tick[d["tick"]].add(d["x"])
                            for kk in range(k + 1, seeds_per_cell + 1):
                                per_seed_growth[kk].add(d["x"])
                        ticks_per_traj.append(nt)
        mult = Counter(all_vecs.values())
        train_vecs = vecs_by_role["TRAIN"]
        late_states = sum(c for x, c in all_vecs.items() if x[late_idx] >= 10 / MAX_STEPS - 1e-12)
        late_distinct = sum(1 for x in all_vecs if x[late_idx] >= 10 / MAX_STEPS - 1e-12)
        res[f"epsilon_{eps}"] = dict(
            n_trajectories=n_traj, n_decision_states=n_states,
            n_distinct_admissible_inputs=len(all_vecs),
            distinct_share=round(len(all_vecs) / n_states, 4),
            n_states_tick_ge_10=late_states, n_distinct_inputs_tick_ge_10=late_distinct,
            distinct_share_tick_ge_10=round(late_distinct / max(1, late_states), 4),
            distinct_inputs_vs_seeds_per_cell={str(k): len(v) for k, v in per_seed_growth.items()},
            distinct_inputs_by_role={r: len(v) for r, v in sorted(vecs_by_role.items())},
            distinct_inputs_by_template={d: len(v) for d, v in vec_by_template.items()},
            distinct_inputs_by_tick={str(t): len(v) for t, v in sorted(by_tick.items())},
            share_primary_test_inputs_also_seen_in_train=round(
                len(vecs_by_role["TEST_PRIMARY_JOINT"] & train_vecs) / max(1, len(vecs_by_role["TEST_PRIMARY_JOINT"])), 4),
            decision_states_per_trajectory=dict(min=min(ticks_per_traj), max=max(ticks_per_traj),
                                                mean=round(float(np.mean(ticks_per_traj)), 2)),
            multiplicity_histogram_top={str(k): v for k, v in sorted(mult.items())[:12]},
            n_states_with_T90_already_reached=t90_pre,
        )
    res["seconds"] = round(time.perf_counter() - t0, 1)
    res["note"] = ("DESIGN-TIME COUNT, not the pilot: pre-decision reference trajectories only, 3 seeds per non-buffer "
                   "cell, no candidate rollouts, no targets, nothing persisted but these counts.")
    return res


def compliance_nesting_check() -> dict:
    key = "AILX_probe"
    nested_p = all((~generate_compliance_vector(key, s, p=0.5) | generate_compliance_vector(key, s, p=0.9)).all()
                   for s in range(50))
    same_across_th = True  # t_h is not an input to generate_compliance_vector at all
    diff_key = np.mean([(generate_compliance_vector("A", s, p=0.7) != generate_compliance_vector("B", s, p=0.7)).mean()
                        for s in range(50)])
    return dict(
        vectors_nested_across_p_for_same_key_and_seed=bool(nested_p),
        vector_independent_of_t_h=same_across_th,
        mean_disagreement_between_different_keys=round(float(diff_key), 4),
        consequence="With one key and one seed, the p=0.5 compliant set is a subset of the p=0.9 set and the vector "
                    "is identical across every t_h. Reusing a (key, seed) across a training cell and a test cell would "
                    "give nearly the same hidden compliance on both sides of the split. Keys must include the split "
                    "role and seed ranges must be disjoint by role.",
    )


# ===========================================================================
# Static specification blocks.
# ===========================================================================

def candidate_action_set_spec(suff: dict, known: dict, choice: dict) -> dict:
    fname = choice["chosen"]
    if fname is None:
        return dict(audit=MARKER, chosen=None, selection=choice, status="NO FIXED FAMILY QUALIFIED -- unresolved",
                    sufficiency_analysis={k: v for k, v in suff.items() if k != "per_candidate_evidence"})
    fam = FIXED_FAMILIES[fname]
    ev = suff["per_candidate_evidence"][fname]
    n_states = suff["per_candidate_set"][fname]["n_states"]
    cands = []
    for i, (cname, asg) in enumerate(fam.items()):
        vals = set(asg.values())
        kind = "NONE" if vals == {U} else ("GLOBAL" if len(vals) == 1 else "DIFFERENTIAL")
        e = ev.get(cname, {})
        cands.append(dict(
            id=i, name=cname, kind=kind, encoding={z: VALUE_NAMES[asg[z]] for z in ELIGIBLE_ZONES},
            meaning=("No relabel at this decision." if kind == "NONE" else
                     "Relabel every eligible (compliant, uncommitted, waiting) occupant of each zone with that zone's "
                     "value (actions.apply_action rules); all groups built from the pre-decision state; zones with "
                     "no eligible occupant are unaffected."),
            evidence_on_existing_states=dict(
                n_states=n_states,
                n_states_in_unrestricted_best_outcome_class=e.get("in_unrestricted_best_class", 0),
                n_states_behaviorally_distinct_from_its_base_global=e.get("behaviorally_differs_from_its_base_global", 0),
                n_states_behaviorally_distinct_from_every_global=e.get("behaviorally_differs_from_every_global", 0)),
            can_be_identical_to_another_candidate=True,
        ))
    never_best = sum(1 for c in cands if c["evidence_on_existing_states"]["n_states_in_unrestricted_best_outcome_class"] == 0)
    return dict(
        audit=MARKER, name=fname, size=len(cands), selection=choice,
        representation=("ABSOLUTE per-zone assignment over the 7 eligible zones (R1, R2, C1, R3, R4, C2, H): "
                        "NO_INTERVENTION, or a BASE global action (FAVOR_A / PARITY / FAVOR_B) on every eligible zone "
                        "with up to two zones overridden to a different action. A candidate's meaning never depends on "
                        "the heuristic's output. CE1/CE2 are never eligible (always committed)."),
        encoding_rule="id 0 = NO_INTERVENTION; then for base in (A, P, B): the base alone, then single-zone overrides "
                      "(zone order R1,R2,C1,R3,R4,C2,H; values in A,P,B order skipping the base), then two-zone "
                      "overrides over zone pairs in that order.",
        applied_via="differential_targeting.apply_differential_action(sim, [(zone, value) for non-empty zones]), which "
                    "constructs every TargetGroup from the pre-decision state before relabeling.",
        includes_heuristic_default="The heuristic's global action h is always the candidate 'BASE=h' (no overrides), so "
                                   "the default d(s) is always in the set.",
        relation_to_existing_code="Contains, as absolute assignments, every Discrete(211) action under the env/oracle "
                                  "'global-then-override' semantics (base = h), the same overrides on the two other "
                                  "bases, and NO_INTERVENTION.",
        candidates=cands,
        n_candidates_never_in_best_class_on_existing_states=never_best,
        duplicates=dict(
            can_two_candidates_be_identical=True,
            when=["an override on a zone with no eligible waiting occupant equals its base candidate",
                  "an override value equal to every eligible occupant's current label behaves like the base there",
                  "at t=0 NO_INTERVENTION equals BASE=P (the reset labels are the parity labels)"],
            policy="Keep all slots (fixed encoding and size). Per state, store each candidate's behavioral-equivalence "
                   "class id (hash of the resulting label map), roll out one member per class, copy its outcome to the "
                   "rest, and evaluate equivalent candidates as ties. Candidates are NOT removed: removal would make "
                   "the set state-dependent.",
            distinct_rollouts_per_state=suff["per_candidate_set"][fname]["behaviorally_distinct_per_state"]),
        known_interventions=known,
        sufficiency_analysis={k: v for k, v in suff.items() if k != "per_candidate_evidence"},
        limitation="Sufficiency is established on the existing 530 states, where at most 4 eligible zones are "
                   "non-empty. States with 5-7 non-empty zones might need 3+ overrides; the pilot re-checks this (G3).",
        semantics_inventory=semantics_inventory(),
    )


def target_definitions_spec(suff: dict) -> dict:
    return dict(
        audit=MARKER,
        decision_time="t_d: the decision tick (multiple of 5). The observation is computed at the end of tick t_d-1 "
                      "(sim.t == t_d) before any relabel; the candidate is applied at the start of tick t_d.",
        horizon="From tick t_d until the episode ends (all occupants evacuated, or MAX_STEPS=180).",
        continuation="Every later decision tick: the frozen GLOBAL_DEADBAND heuristic (margin 0.75) applied globally.",
        targets=dict(
            future_T90=dict(
                definition="T90_abs - t_d, where T90_abs = compute_metrics(sim.evacuated).T90 (exit tick of the "
                           "ceil(0.9*40)=36th evacuee). If >= 36 occupants had exited before t_d: 0, with flag "
                           "T90_reached_before_decision=True, and the criterion is constant across candidates, so the "
                           "selector skips it and the loss masks it. If T90 is undefined: +inf (never observed).",
                unit="ticks", source="metrics.compute_metrics (unchanged)"),
            future_T100=dict(
                definition="T100_abs - t_d. T100_abs = last exit tick, or MAX_STEPS with T100_censored=True. A "
                           "decision state requires n_active > 0, so T100 is never reached before t_d.",
                unit="ticks"),
            future_unmet_demand=dict(
                definition="sum over ticks t in [t_d, end) of sum over all 10 edges of max(0, candidates_e(t) - "
                           "capacity_e(t)), measured AFTER tick t's relabel (if t is a decision tick) and BEFORE its "
                           "admission, i.e. on exactly the queue that admission sees. Pre-decision history is excluded.",
                unit="occupant-ticks",
                implementation_without_simulator_change="apply the relabel with actions.apply_action / "
                    "apply_differential_action, then call timing_experiment.diagnostic_all_edges_unmet, then "
                    "sim._run_one_tick(None). Equivalent to _run_one_tick(action) because relabeling is its first step "
                    "after the pure-function hazard update."),
            future_hazard_edge_unmet_demand=dict(
                definition_recommended="sum over t in [max(t_d, t_h), end) of max(0, H_CE1 branch-A candidates(t) - "
                                       "cap_H_CE1(t)), measured the same way; 0 when no hazard is scheduled or it "
                                       "never starts before the end.",
                alternative="the same sum over ALL ticks t >= t_d for EVERY scenario (onset-agnostic).",
                status="REQUIRES USER DECISION (D1): both definitions remove the prior label's future-information "
                       "problem; they change the meaning of the 4th ranking criterion differently.",
                unit="occupant-ticks"),
        ),
        verification=dict(
            realized_measurement_matches_simulator_tick_log=suff["measurement_checks"],
            meaning="For every unique rollout of the sufficiency analysis, the post-relabel H_CE1 overflow summed by the "
                    "proposed measurement equals the simulator's own tick_log h_ce1_unmet exactly (all ticks, and "
                    "onset-gated)."),
        stored_per_row=["future_T90", "T90_reached_before_decision", "future_T100", "T100_censored",
                        "future_unmet_demand", "future_hazard_edge_unmet_demand (both definitions until D1 is decided)",
                        "T90_abs", "T100_abs (for comparability with prior experiments)"],
        changes_vs_existing_labels=[
            "pre-decision unmet removed (fixes T1)",
            "hazard-edge unmet no longer keyed on a scheduled-but-future hazard (fixes T2)",
            "unmet measured post-relabel, matching the simulator (fixes T3)",
            "one absolute candidate semantics (fixes T4, T5)",
        ],
    )


def final_feature_manifest_spec() -> dict:
    manifest = prior.build_feature_manifest()
    adm = prior.admissible_state_feature_names(manifest)
    by = {m["name"]: m for m in manifest}
    interp = {
        "waiting_": "people queued in that zone", "in_transit_": "people walking on that corridor segment",
        "residual_cap_H_CE1_normalized": "whether the hub->CE1 corridor is currently degraded (1.0 nominal, 0.333 degraded)",
        "normalized_time": "time since alarm / 180", "normalized_remaining": "fraction of occupants not yet out",
        "waiting_H_total": "people queued at the hub (label-free)",
    }

    def phys(n):
        for k, v in interp.items():
            if n.startswith(k) or n == k:
                return v
        return ""
    feats = []
    for i, n in enumerate(adm):
        m = by[n]
        feats.append(dict(index=i, name=n, type="float64 (integer count)" if n.startswith(("waiting", "in_transit"))
                          else "float64", source=m["source_variable"], computation=m["computation"],
                          physical_interpretation=phys(n), availability=m["available_when"],
                          leakage_class=m["classification"], leakage_label=m["classification_label"],
                          why_allowed="Computed before the candidate relabel from current physical state only; no "
                                      "future state, no hidden compliance or intent label, no oracle output "
                                      "(verified in the pre-training audit: temporal/action-independence checks)."))
    return dict(
        audit=MARKER, n_features=len(feats), features=feats,
        excluded=dict(
            time_since_hazard_onset_normalized="F: reveals the future onset tick before it happens",
            hazard_phase="F: 0.5 before onset reveals a scheduled hazard",
            Q_A_at_H="D: counts the hidden target_exit label; tracks hidden non-compliance (r=0.93)",
            Q_B_at_H="D: same",
            normalized_sim_time_temporal="duplicate of normalized_time",
            normalized_decision_index="duplicate of normalized_time",
            trend_features="NOT included in v1: undefined at t=0 and no retained-history spec is frozen. If added later: "
                           "keep exactly the admissible vector from the previous decision tick (t_d - 5); at t_d=0 the "
                           "delta is defined as 0 with a has_history=0 flag; re-run the leakage audit before use.",
            metadata_never_features=["template id", "t_h", "p", "seed", "trajectory id", "split role",
                                     "compliance vector", "pre-decision action history"],
        ),
        candidate_descriptor=f"Separate conditioning input (not a state feature): the candidate's absolute 7-zone "
                             f"encoding as 7 x {{UNTOUCHED, FAVOR_A, PARITY, FAVOR_B}} one-hot (28 values), which lets "
                             f"the model share structure across the {CHOSEN['size']} candidates of {CHOSEN['name']}.",
        source_disproves_prior_audit=False,
    )


def scenario_space_spec(nest: dict) -> dict:
    return dict(
        audit=MARKER,
        dimensions=dict(
            A_occupant_distribution=dict(
                parameter="distribution over R1..R4 summing to 40 (simulator.reset; _initial_zone_assignment)",
                current_range="3 templates used (10/10/10/10, 5/5/15/15); the robust-generalization audit also used "
                              "12/8/12/8, 16/4/16/4, 4/16/4/16",
                physically_meaningful="any non-negative split of 40 across the four rooms",
                changes_observable_state=True, changes_outcomes=True, useful_diversity="HIGH: the only dimension that "
                "changes tick-0 inputs", confounding="occupant ids are assigned R1-first and baseline exits are "
                "id-parity based, so mirror-image templates are not exactly symmetric; disclosed, not a leak",
                chosen=DISTRIBUTIONS),
            B_compliance_probability=dict(
                parameter="compliance.generate_compliance_vector(key, seed, p=...)",
                current_range="0.9 only (0.75 in the timing matrix)",
                physically_meaningful="0.5 to 0.95 (prior validated range)",
                changes_observable_state="only indirectly and only after hub admissions (corridor counts); never at t=0",
                changes_outcomes=True, useful_diversity="HIGH for outcomes, LOW for inputs",
                confounding=nest["consequence"], chosen=P_TRAIN + P_VAL + P_TEST + P_EXT),
            C_hazard_onset=dict(
                parameter="t_h passed to simulator.reset",
                current_range="none, 5, 7 (3, 6, 7, 10 in the timing matrix)",
                physically_meaningful="3..10 or none; onsets after about tick 13 fall after evacuation ends and are "
                                      "equivalent to none",
                changes_observable_state="only after onset (residual capacity), never before",
                changes_outcomes=True, useful_diversity="HIGH for outcomes; post-onset inputs only",
                confounding="must be crossed with distribution and p (full factorial), never tied to a template",
                chosen=T_TRAIN + T_VAL + T_TEST + T_EXT),
            D_hazard_location=dict(supported=False, reason="topology.cap_h_ce1/edge_capacity hard-code H_CE1; varying "
                                   "it requires a simulator change (forbidden). Fixed and disclosed."),
            E_initial_congestion=dict(supported_directly=False,
                                      reason="reset places occupants only in R1..R4",
                                      indirect_sources=["later decision ticks (5, 10, 15, ...)",
                                                        f"epsilon-perturbed pre-decision history (epsilon={EPSILON}, a "
                                                        "uniformly random GLOBAL action at each pre-decision tick)"]),
            F_behavioral_parameters=dict(supported=["compliance probability (B)"],
                                         fixed=["baseline exit by id parity", "commitment on hub admission"],
                                         reason="no other behavior parameter exists without a simulator change"),
            G_other=dict(fixed=["N_OCCUPANTS=40 (asserted)", "capacities/traversal times (topology constants)",
                                "DECISION_INTERVAL=5", "MAX_STEPS=180"],
                         reason="module constants; changing them changes the simulator model"),
        ),
        compliance_vector_behavior=nest,
    )


def dataset_design_spec(div: dict) -> dict:
    th_all = T_TRAIN + T_VAL + T_TEST + T_EXT
    p_all = P_TRAIN + P_VAL + P_TEST + P_EXT
    roles = {f"t_h={th}|p={p}": cell_role(th, p) for th in th_all for p in p_all}
    seeds_full = dict(TRAIN=20, VALIDATION=10, TEST_PRIMARY_JOINT=40, TEST_HELDOUT_TIMING=15,
                      TEST_HELDOUT_COMPLIANCE=15, DIAG_EXTRAPOLATION=5, UNUSED_BUFFER=0)
    n_cells = Counter(roles.values())
    traj = {r: n_cells[r] * seeds_full[r] * len(TRAIN_TEMPLATES) for r in seeds_full}
    traj["DIAG_TEMPLATE_ZERO_SHOT"] = len(ZERO_SHOT_TEMPLATES) * sum(
        n_cells[r] for r in n_cells if r != "UNUSED_BUFFER") * 3
    states_per_traj = div[f"epsilon_{EPSILON}"]["decision_states_per_trajectory"]["mean"]
    total_traj = sum(traj.values())
    return dict(
        audit=MARKER,
        factors=dict(templates_train_and_test=TRAIN_TEMPLATES, templates_zero_shot_only=ZERO_SHOT_TEMPLATES,
                     hazard_timing_levels=th_all, compliance_levels=p_all,
                     pre_decision_policy=f"frozen heuristic, epsilon-perturbed (epsilon={EPSILON}: a uniformly random "
                                         "GLOBAL action at each pre-decision tick) with a seeded RNG keyed on (seed, p, "
                                         "t_h); the history is stored per row. Caveat: off the deployed state "
                                         "distribution by design; closed-loop evaluation measures the deployed one.",
                     decision_ticks="every decision tick (multiple of 5) while n_active > 0"),
        crossing="Full factorial: every template x every t_h x every p (minus UNUSED_BUFFER cells). t_h and p are "
                 "never tied to a template.",
        cell_roles=roles, n_cells_by_role=dict(n_cells),
        seeds_per_cell_per_template_FULL=seeds_full,
        trajectories_FULL_by_role=traj, trajectories_FULL_total=total_traj,
        expected_decision_states_per_trajectory=states_per_traj,
        candidate_set=CHOSEN["name"],
        expected_candidate_rows_FULL=int(total_traj * states_per_traj * CHOSEN["size"]),
        expected_unique_rollouts_FULL=int(total_traj * states_per_traj * CHOSEN["distinct_mean"]),
        estimated_single_core_hours_FULL=round(
            total_traj * states_per_traj * CHOSEN["distinct_mean"] * CHOSEN["sec_per_unique"] / 3600, 2),
        cost_basis=f"{CHOSEN['sec_per_unique']} s per unique candidate (label-map dedup + rollout), measured in the "
                   "sufficiency analysis",
        compliance_keys="generate_compliance_vector(f'AILX_v1|{role}|{template}', seed, p)",
        seed_ranges_by_role=dict(TRAIN="100000-100999", VALIDATION="110000-110999", TEST_PRIMARY_JOINT="120000-120999",
                                 TEST_HELDOUT_TIMING="130000-130999", TEST_HELDOUT_COMPLIANCE="140000-140999",
                                 DIAG_EXTRAPOLATION="150000-150999", DIAG_TEMPLATE_ZERO_SHOT="160000-160999",
                                 PILOT="190000-190999 (discarded after the pilot audit, never reused)"),
        stored_per_row=["trajectory_id", "template", "t_h", "p", "seed", "split_role", "decision_tick",
                        "pre_decision_action_history", "26 admissible features",
                        f"candidate id (0-{CHOSEN['size'] - 1 if CHOSEN['size'] else '?'})",
                        "behavioral equivalence class id", "all targets of target_definitions.json"],
        phases=dict(
            PHASE_1_PILOT=dict(
                size="all 8 templates x t_h {none, 3, 5, 7, 9} x p {0.5, 0.7, 0.9} x 2 seeds = 240 trajectories "
                     f"(~{int(240 * states_per_traj)} decision states, ~{int(240 * states_per_traj * CHOSEN['size'])} "
                     f"candidate rows, ~{int(240 * states_per_traj * CHOSEN['distinct_mean'])} unique rollouts)",
                also="for 40 pilot states, roll out the UNRESTRICTED absolute space too (sufficiency re-check)"),
            PHASE_2_AUDIT="re-run the pre-training audit checks and the go/no-go criteria in diversity_requirements.json",
            PHASE_3_FULL=f"{total_traj} trajectories as above, only if Phase 2 passes"),
    )


def diversity_requirements_spec(div: dict, prior_ctx: dict) -> dict:
    e = div[f"epsilon_{EPSILON}"]
    h = div["epsilon_0.0"]
    return dict(
        audit=MARKER,
        existing_dataset_baseline=dict(states=450, distinct_admissible_inputs=prior_ctx["traj"]["repeated_state_frequency"][
            "n_distinct_admissible_feature_vectors"]),
        design_time_count=div,
        expected=dict(
            distinct_inputs_heuristic_only=h["n_distinct_admissible_inputs"],
            distinct_inputs_with_epsilon=e["n_distinct_admissible_inputs"],
            distinct_share_with_epsilon=e["distinct_share"],
            state_action_combinations_per_state=CHOSEN["size"],
            unique_rollouts_per_state_mean=CHOSEN["distinct_mean"],
            target_variance="NOT estimable without candidate rollouts; measured in the pilot. Prior data: T100 std "
                            "about 1 tick, post-decision total unmet std about 22 occupant-ticks.",
            repeated_states="see multiplicity_histogram_top; tick-0 states are identical across t_h, p and seed "
                            "within a template by construction (8 distinct tick-0 inputs in total).",
        ),
        irreducible_structure=("At t=0 (and before onset / before any hub admission) the input is identical across t_h, "
                               "p and seed within a template. For those states the learnable target is the conditional "
                               "outcome distribution under the TRAINING mixture of t_h and p; the joint hold-out tests "
                               "robustness to a shift in that mixture, not input generalization. Every result must be "
                               "reported separately for pre-onset and post-onset states."),
        go_no_go_after_pilot=dict(
            G1_distinct_inputs="for decision ticks >= 10: distinct admissible inputs >= 25% of those pilot states and >= "
                               "10 per template. Ticks 0 and 5 are template-determined by construction (one or two "
                               "inputs per template) and are exempt; they are handled by the distributional target.",
            G2_target_spread="in >= 50% of pilot states, candidates produce >= 3 distinct outcome vectors "
                             "(otherwise the ranking task is near-trivial)",
            G3_candidate_sufficiency=f"{CHOSEN['name']} best equals the unrestricted-absolute best in >= 95% of the 40 "
                                     "re-checked pilot states (chosen to include states with >= 5 non-empty zones), and "
                                     "mean T100 regret <= 0.1 tick",
            G4_label_integrity="replay reproduces every pilot row exactly; measurement check matches tick_log 100%",
            G5_leakage="action-independence and temporal-leak checks pass on pilot data (admissible features unchanged "
                       "by any candidate relabel; no feature recovers t_h before onset)",
            G6_identifiability="irreducible share (admissible + candidate) for T100 below 0.6 on post-onset states; "
                               "if above, the model target must be distributional only",
            G7_coverage="every non-buffer cell of the pilot grid has >= 2 complete trajectories",
            rule="All of G1-G7 must pass to proceed to Phase 3; any failure returns to design, not to training.",
        ),
    )


def split_specification_spec() -> dict:
    th_all = T_TRAIN + T_VAL + T_TEST + T_EXT
    p_all = P_TRAIN + P_VAL + P_TEST + P_EXT
    return dict(
        audit=MARKER,
        unit="trajectory. Every decision state and every candidate row of a trajectory share its split; no seed, "
             "compliance key or (key, seed) pair is shared between splits.",
        axes=dict(t_h=dict(train=T_TRAIN, validation=T_VAL, test=T_TEST, extrapolation=T_EXT),
                  p=dict(train=P_TRAIN, validation=P_VAL, test=P_TEST, extrapolation=P_EXT),
                  templates=dict(train_and_test=TRAIN_TEMPLATES, zero_shot=ZERO_SHOT_TEMPLATES)),
        matrix={f"t_h={th}": {f"p={p}": cell_role(th, p) for p in p_all} for th in th_all},
        tests=dict(
            A_heldout_seed=dict(role="diagnostic", shared="templates, t_h, p", unseen="seeds (TRAIN cells, separate "
                                "seed range 125000-125999)"),
            B_heldout_timing=dict(role="diagnostic", shared="templates, p", unseen="t_h in {4, 8}"),
            C_heldout_compliance=dict(role="diagnostic", shared="templates, t_h", unseen="p in {0.6, 0.85}"),
            D_joint_timing_x_compliance=dict(role="PRIMARY", shared="templates", unseen="t_h in {4, 8} AND p in "
                                             "{0.6, 0.85} AND seeds"),
            E_template_zero_shot=dict(role="diagnostic, expected hard, never a gate", shared="t_h and p levels",
                                      unseen=f"templates {ZERO_SHOT_TEMPLATES}"),
            F_extrapolation=dict(role="diagnostic", unseen="t_h=10 or p=0.95 (outside the training range)"),
        ),
        report_strata=["pre-onset vs post-onset", "t=0 vs t>0", "template"],
        validation_use="hyperparameters, quantile calibration, selector tolerances; never touched by test data",
    )


def selector_design_spec() -> dict:
    return dict(
        audit=MARKER,
        status="DESIGN ONLY",
        inputs=f"for each of the {CHOSEN['size']} candidates a of {CHOSEN['name']}: predicted conditional mean m_a and "
               "upper quantile q_a (level alpha=0.9) for each target; the default d(s) = the candidate BASE=h, h = the "
               "heuristic's global action",
        ordering="T90 -> T100 -> future unmet -> future hazard-edge unmet (established; unchanged)",
        operates_on=dict(
            objective="conditional means (expected outcome), because the question is which action is best on average "
                      "for states that look like this one",
            safety_floor="upper quantiles (conservative bounds), because harm must be controlled under the hidden "
                         "compliance / timing spread the model cannot see",
            abstention="interval width and support distance"),
        steps=[
            "1. Masking: skip T90 when T90_reached_before_decision; drop behaviorally duplicate candidates (keep lowest id).",
            "2. Safety floor: A(s) = {a : q_a,k <= m_d,k + delta_k for every criterion k}; d(s) always in A(s).",
            "3. Objective: tolerance-lexicographic over A(s) on means: keep a with m_a,T90 <= min + delta_T90, then T100 "
            "within delta_T100, then unmet within delta_U, then minimal hazard unmet.",
            "4. Switch rule: recommend a* != d(s) only if a* beats d(s) by more than delta on the first criterion where "
            "they differ by more than delta; otherwise d(s).",
            "5. Abstain -> d(s) if: state outside training support (nearest training input farther than the 99th "
            "percentile of training nearest-neighbour distances); any feature invalid; model failure; or "
            "(q_a*,T100 - m_a*,T100) > the 95th percentile of that width on validation.",
        ],
        tolerances=dict(delta_T90=0.5, delta_T100=0.5,
                        delta_U="0.5 x validation MAE of the model on future_unmet (fixed after validation)",
                        delta_H="0.5 x validation MAE on future_hazard_edge_unmet",
                        justification="T90/T100 are integer ticks, so a half tick separates distinct values; unmet "
                                      "tolerances cannot be justified before validation error is known (disclosed)."),
        never_uses=["future information", "t_h", "p", "compliance", "Q_A/Q_B_at_H", "template id"],
        caveat="d(s) is the GLOBAL_DEADBAND heuristic, which itself reads Q_A_at_H/Q_B_at_H (class D). See decision D2.",
    )


def closed_loop_spec() -> dict:
    return dict(
        audit=MARKER,
        status="DESIGN ONLY; run after training, on TEST_PRIMARY_JOINT (primary) and each diagnostic family",
        policies=dict(
            P1_no_intervention="never relabel",
            P2_capacity_aware_heuristic="GLOBAL_DEADBAND (margin 0.75) at every decision",
            P3_frozen_deterministic_safety_controller="observable_robust_differential.py (MD5 98b570ba7d7d451c82acf919113db2e5), "
                                                      "executed under DifferentialRLGymEnv's global-then-override semantics "
                                                      "(the semantics its actions 102/6 were audited under). It matches "
                                                      "only the exact S2/S3 tick-0 observations, so on the new templates "
                                                      "it reduces to the heuristic; reported for continuity.",
            P4_ai_assisted="selector_design over the trained model at every decision; d(s)=P2",
            P5_full_information_oracle=f"at every decision, the {CHOSEN['name']} candidate with the best TRUE one-step "
                                       "outcome (heuristic continuation); an upper reference, not achievable",
        ),
        protocol="Common random numbers: identical (template, t_h, p, compliance key, seed) for every policy; full "
                 "episodes; paired comparisons against P2.",
        metrics=["T90", "T100", "total unmet demand (episode)", "hazard-edge unmet demand (episode, D1 definition)",
                 "harmful recommendation rate (decisions where P4 deviated from d(s) and the one-step counterfactual was "
                 "worse than d(s))", "recommendation coverage (share of decisions where P4 deviates from d(s))",
                 "abstention / fallback rate", "switch count"],
        primary_endpoint="Episode-level paired lexicographic comparison P4 vs P2: share of episodes strictly worse "
                         "(safety) with share strictly better reported alongside.",
        secondary=["paired T100 difference", "paired T90 difference", "paired unmet differences", "gap to P5"],
        caveat="Training labels assume heuristic continuation (one-step counterfactual, Q^heuristic). Applying the "
               "selector at every decision is a policy-improvement step whose lexicographic, non-additive objective "
               "is not covered by the policy-improvement theorem, so closed-loop gains must be measured, not assumed.",
    )


def preregistration_spec(prev) -> dict:
    return dict(
        audit=MARKER,
        status="FROZEN DRAFT -- final once decisions D1 and D2 are made; record this file's SHA-256 before Phase 3",
        primary_test="TEST_PRIMARY_JOINT (t_h in {4,8} x p in {0.6,0.85}, unseen seeds), all training templates",
        statistics="95% cluster-bootstrap CIs over trajectories, 2000 resamples; per test family, never pooled; "
                   "pre-onset and post-onset reported separately.",
        outcome_prediction=dict(
            baselines=["mean by candidate", "mean by candidate x decision tick", "exact-input lookup", "1-NN",
                       "heuristic-outcome-as-prediction (d(s)'s realized outcome for every candidate)"],
            target_form="conditional mean + conditional quantiles (0.1, 0.5, 0.9)",
            criterion="On the primary test, pinball loss (averaged over the three quantiles) on future_T100 and "
                      "future_unmet is lower than the best baseline, with the paired-difference CI excluding 0; "
                      "upper-quantile coverage within 0.9 +/- 0.05.",
            irreducible_floor="report the within-identical-input spread (Bayes floor proxy) next to every error; the "
                              "model is judged against baselines, not against zero error",
        ),
        recommendation=dict(
            metrics=[f"outcome-equivalence with the {CHOSEN['name']} oracle-best class",
                     "T100 and unmet regret vs oracle-best"],
            criterion="Outcome-equivalence rate greater than d(s)'s own, CI excluding 0, on states where some candidate "
                      "strictly beats d(s). Exact oracle-action match: diagnostic only.",
        ),
        safety=dict(
            criterion="Harmful-recommendation rate: upper 95% CI bound <= 1% of decisions in every test family.",
            derivation="1% is a safety requirement, not fitted to data. With zero observed harms the rule-of-three "
                       "bound needs >= 300 independent deviating decisions; trajectory clustering raises this, which "
                       "sets the TEST_PRIMARY_JOINT size (40 seeds x 4 cells x 6 templates = 960 trajectories).",
        ),
        generalization=dict(primary="D_joint_timing_x_compliance", diagnostics=["A", "B", "C", "E", "F"],
                            criterion="Prediction + recommendation + safety all pass on D."),
        closed_loop=dict(
            criterion="Safety: P4 strictly worse than P2 in <= 1% of paired episodes (upper CI). Usefulness: NO "
                      "threshold is justified yet -- the minimum meaningful gain will be set from the Phase-1 pilot "
                      "(spread between P2 and P5) BEFORE Phase 3 and appended here with its hash.",
        ),
        not_required=["beating the heuristic", "reproducing a unique oracle action", "using any D/E/F/G feature"],
        previous_draft_superseded=prev,
    )


# ===========================================================================
# Decision + report.
# ===========================================================================

def go_no_go(suff: dict, div: dict, choice: dict) -> dict:
    cs = suff["per_candidate_set"][choice["chosen"]] if choice["chosen"] else None
    open_decisions = [] if choice["chosen"] else [dict(
        id="D0", area="action semantics", question="No fixed candidate family met the sufficiency rule; choose one.",
        recommendation="see candidate_action_set.json selection table", why_not_decided_silently="no family qualified")]
    open_decisions += [
        dict(id="D1", area="target semantics",
             question="future_hazard_edge_unmet_demand: onset-gated (t >= t_h, 0 without hazard) or all ticks for every "
                      "scenario?",
             recommendation="onset-gated: matches the metric's stated purpose (congestion caused by the degraded edge); "
                            "pre-onset H_CE1 overflow is already counted in future_unmet_demand",
             why_not_decided_silently="it changes the meaning of the 4th ranking criterion relative to every prior "
                                      "experiment"),
        dict(id="D2", area="action semantics / default policy",
             question="The continuation policy and the fallback d(s) are the GLOBAL_DEADBAND heuristic, which reads "
                      "Q_A_at_H/Q_B_at_H (hidden labels, class D). Accept it as the simulated building's existing "
                      "controller (inside label generation and as fallback), or require a label-free default first?",
             recommendation="accept for label generation and as fallback, disclosed: the model's own inputs stay "
                            "admissible, and P2 carries the same privilege, so comparisons stay paired. A label-free "
                            "default would be a new controller, which is out of scope",
             why_not_decided_silently="it determines whether the AI-assisted system as evaluated is fully "
                                      "perception-honest"),
    ]
    resolved = [
        f"candidate set: absolute {choice['chosen']} ({CHOSEN['size']} fixed slots), selected by an explicit rule; it "
        "removes the three-way semantic conflict in the existing code",
        "continuation policy: frozen heuristic, identical across candidates",
        "future_T90 / future_T100 / future_unmet definitions (post-decision, post-relabel measurement verified against "
        "the simulator's tick_log)",
        "features: the 26 admissible features, no trend features",
        f"pre-decision state generation: heuristic with epsilon={EPSILON} random global actions (diversity count)",
        "split unit and roles: trajectory; joint timing x compliance primary test; role-keyed compliance vectors",
        "target form: conditional mean + quantiles",
    ]
    if cs:
        resolved.append(f"candidate sufficiency on existing states: {choice['chosen']} matches the unrestricted best in "
                        f"{cs['share_states_where_set_best_equals_unrestricted_best']:.1%} (mean T100 regret "
                        f"{cs['mean_T100_regret_ticks']} ticks)")
    return dict(classification="REQUIRES ONE MORE DESIGN DECISION", open_decisions=open_decisions,
                resolved=resolved,
                after_decisions="With D1 and D2 answered, the specification is complete and the next step is Phase 1 "
                                "(pilot, 240 trajectories), not the full dataset.")


def render(ctx: dict) -> str:
    L = []
    a = L.append
    suff, div, dec = ctx["suff"], ctx["div"], ctx["decision"]
    a(f"<!-- {MARKER} -->")
    a("# Final Experiment Specification / Data-Generation Design Audit\n")
    a("Read-only. No model trained, no dataset or pilot generated, no selector implemented, no simulator / "
      "observation.py / controller / frozen-RL / 9,450-row dataset file modified (SHA-256 checked; Section J).\n")
    a(f"## Status: **{dec['classification']}**\n")
    for d in dec["open_decisions"]:
        a(f"- **{d['id']} ({d['area']})**: {d['question']} Recommendation: {d['recommendation']}. Not decided "
          f"silently because {d['why_not_decided_silently']}.")
    a("\nResolved in this audit: " + "; ".join(dec["resolved"]) + ".\n")

    a("## A. Experimental question\n")
    a("> Can a supervised model, given only the 26 admissible pre-decision features and a candidate from a fixed "
      f"{CHOSEN['size']}-candidate set, predict the conditional distribution of that candidate's future T90, T100, unmet demand and "
      "hazard-edge unmet demand (heuristic continuation), better than non-learning baselines on jointly held-out "
      "hazard timings and compliance levels; and does a deterministic, safety-floored selector over those predictions "
      "recommend actions whose outcomes approach the oracle-best candidate without harming relative to the heuristic "
      "default, in one-step and closed-loop evaluation?\n")
    a("\"Learn\" means, separately: **A** outcome prediction beats baselines (pinball loss, calibrated quantiles); "
      "**B** recommendations reach oracle-best outcome classes more often than the default; **C** harm rate <= 1% "
      "(upper CI); **D** A-C hold on the joint timing x compliance hold-out; **E** closed-loop episodes are no worse "
      "than the heuristic in >= 99% of paired episodes. Beating the heuristic and matching a unique oracle action are "
      "not required.\n")

    ch = CHOSEN["name"]
    a(f"## B/D. Candidate action set ({ch}, {CHOSEN['size']} fixed slots)\n")
    a("Absolute per-zone assignment over R1, R2, C1, R3, R4, C2, H: NO_INTERVENTION, or a base global action "
      "(FAVOR_A / PARITY / FAVOR_B) on every eligible zone with up to two zones overridden to a different action "
      "(1 + 3 x (1 + 14 + 84) = 298). The heuristic's action h is always the candidate BASE=h. Duplicates are kept "
      f"as fixed slots and marked by behavioral-equivalence class; on existing states a state has on average "
      f"{CHOSEN['distinct_mean']} behaviorally distinct candidates, so that is the real rollout count.\n")
    a("The code today has three incompatible meanings for a differential action (leave-untouched sweep; "
      "`apply_high_level_action`; env/oracle global-then-override). All are special cases of the absolute "
      "representation, which is why it is frozen here.\n")
    a(f"Selection rule: {ctx['choice']['rule']}. Sufficiency on existing states ({suff['states_used']}; "
      f"{suff['n_unique_rollouts']} unique rollouts; the reference is every absolute assignment over the non-empty "
      "zones):\n")
    a("| set | size | best == unrestricted best | mean T100 regret | max T100 regret | distinct candidates/state | known interventions reached |")
    a("|---|---|---|---|---|---|---|")
    table = {r["family"]: r for r in ctx["choice"]["table"]}
    for n, s in suff["per_candidate_set"].items():
        r = table.get(n)
        a(f"| {n} | {r['size'] if r else 'state-dependent'} | {s['share_states_where_set_best_equals_unrestricted_best']:.1%} "
          f"| {s['mean_T100_regret_ticks']} | {s['max_T100_regret_ticks']} | {s['behaviorally_distinct_per_state']} | "
          f"{r['known_interventions_reached'] if r else 'n/a'} |")
    a("\nThe wing-level SIDE28 was the first design tried; it cannot express the robust controller's S2 action "
      "(R2 and R4 -> FAVOR_B) and loses up to 2 ticks of T100. Useful splits are room-level.\n")
    a(f"Known interventions (outcome = [future T90, future T100, future unmet, future hazard unmet]) vs {ch}:\n")
    for k, v in ctx["known"].items():
        f = v["by_family"].get(ch, {})
        a(f"- {k} at {v['state']}: {v['outcome']}; exactly expressible: {bool(f.get('exactly_expressible'))}; "
          f"{ch} best {f.get('best_outcome')} (at least as good: {f.get('best_at_least_as_good')}).")

    a("\n## E. Target definitions\n")
    for k, v in ctx["targets"]["targets"].items():
        a(f"- **{k}**: {v.get('definition', v.get('definition_recommended'))}" + (f" Status: {v['status']}" if v.get("status") else ""))
    a(f"\nMeasurement check (post-relabel H_CE1 overflow vs simulator tick_log): {suff['measurement_checks']}. "
      f"States where T90 was already reached before the decision (existing states): {suff['n_states_T90_reached_before_decision']}.\n")

    a("## Rollout protocol\n")
    a("1. Compute the 26 features at the decision tick, before any relabel. 2. deepcopy the simulator (the prior audit "
      f"confirmed identical starting fingerprints). 3. Apply exactly one {CHOSEN['name']} candidate via "
      "apply_differential_action. "
      "4. Continue with the frozen GLOBAL_DEADBAND heuristic at every later decision, identical for all candidates. "
      "5. Measure the targets above. The heuristic continuation is chosen on causal grounds: the label is then "
      "Q^heuristic(s, a), i.e. the consequence of deviating once from the default the selector falls back to; "
      "no-further-intervention would freeze a candidate's labels for the rest of the episode, which no deployed "
      "policy does, and would make the default's own label inconsistent with the default's behavior.\n")

    a("## C. Feature set\n")
    a(f"{ctx['features']['n_features']} admissible features, unchanged from the pre-training audit (source inspection "
      "did not disprove it): " + ", ".join(f"`{f['name']}`" for f in ctx["features"]["features"]) + ". Excluded: "
      "time_since_hazard_onset_normalized, hazard_phase (F); Q_A_at_H, Q_B_at_H (D); duplicate clocks; trend features.\n")

    a("## Scenario space\n")
    a("Varied without simulator changes: occupant distribution (the only lever that changes t=0 inputs), compliance "
      "p, hazard onset t_h, and (indirectly) decision tick and pre-decision history. Not variable: hazard location "
      "(hard-coded H_CE1), capacities, N=40, behavior beyond compliance.\n")
    n = ctx["nest"]
    a(f"Compliance vectors: nested across p for one (key, seed): {n['vectors_nested_across_p_for_same_key_and_seed']}; "
      f"independent of t_h: {n['vector_independent_of_t_h']}. {n['consequence']}\n")

    a("## B. Dataset design\n")
    dd = ctx["design"]
    a(f"Full factorial over {len(TRAIN_TEMPLATES)} train/test templates + {len(ZERO_SHOT_TEMPLATES)} zero-shot templates x "
      f"{len(T_TRAIN + T_VAL + T_TEST + T_EXT)} onset levels x {len(P_TRAIN + P_VAL + P_TEST + P_EXT)} compliance levels. "
      f"Cells by role: {dd['n_cells_by_role']}. Full-phase trajectories: {dd['trajectories_FULL_by_role']} "
      f"(total {dd['trajectories_FULL_total']}, ~{dd['expected_candidate_rows_FULL']:,} candidate rows, "
      f"~{dd['expected_unique_rollouts_FULL']:,} unique rollouts, ~{dd['estimated_single_core_hours_FULL']} single-core "
      "hours). The size is driven by the primary test (safety criterion), not by training: see H.\n")
    a(f"Pre-decision states are reached by the heuristic with epsilon={EPSILON} random GLOBAL actions. This buys input "
      "diversity but means training states are not distributed like states under the heuristic or the AI-assisted "
      "policy; the closed-loop evaluation (G) is what measures performance on the deployed state distribution, and "
      "results are also reported on the epsilon=0 subset of test decisions (their first decision, t=0, is always "
      "on-policy).\n")

    a("## Diversity before simulation (design-time count, not the pilot)\n")
    a("Pre-decision reference trajectories only (3 seeds per non-buffer cell, all 8 templates), no candidate rollouts "
      "or targets.\n")
    a("| pre-decision policy | trajectories | decision states | distinct inputs | distinct share | tick>=10 distinct share | distinct by seeds/cell (1,2,3) | by tick | primary-test inputs seen in train |")
    a("|---|---|---|---|---|---|---|---|---|")
    for e in EPSILONS_COUNTED:
        d = div[f"epsilon_{e}"]
        a(f"| epsilon={e} | {d['n_trajectories']} | {d['n_decision_states']} | **{d['n_distinct_admissible_inputs']}** | "
          f"{d['distinct_share']:.1%} | {d['distinct_share_tick_ge_10']:.1%} | "
          f"{list(d['distinct_inputs_vs_seeds_per_cell'].values())} | {d['distinct_inputs_by_tick']} | "
          f"{d['share_primary_test_inputs_also_seen_in_train']:.1%} |")
    a(f"\nChosen: epsilon = {EPSILON} ({div['epsilon_selection']['rule']}).")
    a(f"\nExisting dataset for comparison: 450 states -> 11 distinct inputs. {ctx['divreq']['irreducible_structure']}\n")

    a("## F. Split\n")
    a("Trajectory-level. Primary = t_h {4,8} x p {0.6,0.85} with unseen seeds; diagnostics: held-out seed, timing, "
      "compliance, template zero-shot, extrapolation. Keys include the split role; seed ranges are disjoint by role.\n")

    a("## Target identifiability\n")
    a("Chosen target form: **D, conditional expectation + conditional quantiles** (0.1/0.5/0.9). Point prediction "
      "alone would penalize the model for hidden compliance/timing it cannot see; pure distributions complicate the "
      "selector; a conservative bound alone discards the objective. Means drive the objective, upper quantiles drive "
      "the safety floor, and errors are reported next to the within-identical-input spread.\n")

    a("## Selector (design only)\n")
    a("Means for the tolerance-lexicographic objective (T90 -> T100 -> unmet -> hazard-edge unmet, delta 0.5 tick on "
      "T90/T100), upper quantiles for a non-regression safety floor vs d(s), abstention on out-of-support states or wide "
      "intervals, deterministic fallback to d(s). See selector_design.json.\n")

    a("## G. Future evaluation\n")
    a("Closed loop on the primary cells with common random numbers: no intervention, heuristic, frozen robust "
      "controller, AI-assisted, full-information one-step oracle. Primary endpoint: share of paired episodes where "
      "AI-assisted is strictly worse than the heuristic under the established order. See closed_loop_evaluation.json.\n")

    a("## H/I/J. Minimum data, pilot, go/no-go\n")
    a("No defensible analytic minimum exists for training size; the test size is set by the safety criterion (>= 300 "
      "independent deviating decisions -> 960 primary-test trajectories). Staged: Phase 1 pilot (240 trajectories, "
      "seeds 190000+, discarded afterwards) -> Phase 2 audit against G1-G7 in diversity_requirements.json -> Phase 3 "
      "full data only if all pass.\n")
    integ = ctx["integrity"]
    a("## Integrity\n")
    a(f"- Protected files unchanged: **{integ['all_unchanged']}** ({len(integ['before'])} files incl. the prior audit's "
      "deliverables, script and tests; SHA-256 in dataset_design.json -> integrity).")
    a(f"- Frozen controller MD5 match: {integ['controller']['match']}; RL checkpoints unchanged: "
      f"{integ['checkpoints']['all_match']} ({integ['checkpoints']['n_checkpoints']}).")
    a("- preregistered_success_criteria.json from the prior audit is superseded; its full content and SHA-256 are "
      "embedded under `previous_draft_superseded`.")
    return "\n".join(L) + "\n"


def run(seeds_per_cell: int = 3, states_limit: int | None = None) -> dict:
    before = protected_hashes()
    t0 = time.perf_counter()
    prior_ctx = load_prior_audit()
    assert prior_ctx["split"]["final_decision"]["classification"] == "C"
    sem = semantics_inventory()
    states = existing_states()
    if states_limit:
        states = states[:states_limit]
    suff = candidate_sufficiency(states)
    print(f"[3] sufficiency: {json.dumps({k: v['share_states_where_set_best_equals_unrestricted_best'] for k, v in suff['per_candidate_set'].items()})}")
    known = known_interventions_expressible()
    choice = choose_candidate_set(suff, known)
    global EPSILON
    if choice["chosen"]:
        CHOSEN.update(name=choice["chosen"], size=len(FIXED_FAMILIES[choice["chosen"]]),
                      distinct_mean=suff["per_candidate_set"][choice["chosen"]]["behaviorally_distinct_per_state"]["mean"],
                      sec_per_unique=round(suff["seconds"] / max(1, suff["n_unique_rollouts"]), 4))
    print(f"[3] chosen candidate set: {choice['chosen']}")
    nest = compliance_nesting_check()
    div = diversity_count(seeds_per_cell)
    eps_ok = [e for e in EPSILONS_COUNTED[1:] if div[f"epsilon_{e}"]["distinct_share_tick_ge_10"] >= 0.25]
    EPSILON = eps_ok[0] if eps_ok else EPSILONS_COUNTED[-1]
    div["epsilon_selection"] = dict(rule="smallest epsilon > 0 whose decision states at tick >= 10 are >= 25% distinct "
                                         "admissible inputs; else the largest counted", chosen=EPSILON)
    print(f"[9] diversity: " + ", ".join(f"eps={e}: {div[f'epsilon_{e}']['n_distinct_admissible_inputs']} "
                                         f"(late share {div[f'epsilon_{e}']['distinct_share_tick_ge_10']})"
                                         for e in EPSILONS_COUNTED) + f"; chosen eps={EPSILON}")
    after = protected_hashes()
    changed = [k for k in before if before[k] != after[k]]
    if changed:
        raise RuntimeError(f"PROTECTED FILES CHANGED: {changed}")
    integrity = dict(before=before, all_unchanged=True, controller=prior.design.verify_controller_hash(),
                     checkpoints=prior.design.verify_checkpoint_hashes())
    cas = candidate_action_set_spec(suff, known, choice)
    cas["semantics_inventory"] = sem
    targets = target_definitions_spec(suff)
    feats = final_feature_manifest_spec()
    design = dataset_design_spec(div)
    design["integrity"] = integrity
    divreq = diversity_requirements_spec(div, prior_ctx)
    decision = go_no_go(suff, div, choice)
    ctx = dict(suff=suff, known=known, nest=nest, div=div, targets=targets, features=feats, design=design,
               divreq=divreq, decision=decision, integrity=integrity, choice=choice)
    split = split_specification_spec()
    split["final_go_no_go"] = decision
    ctx["outputs"] = {
        "candidate_action_set.json": cas,
        "target_definitions.json": targets,
        "final_feature_manifest.json": feats,
        "scenario_space.json": scenario_space_spec(nest),
        "dataset_design.json": design,
        "diversity_requirements.json": divreq,
        "split_specification.json": split,
        "selector_design.json": selector_design_spec(),
        "closed_loop_evaluation.json": closed_loop_spec(),
        "preregistered_success_criteria.json": preregistration_spec(previous_preregistration()),
    }
    ctx["outputs"]["FINAL_EXPERIMENT_SPECIFICATION.md"] = render(ctx)
    ctx["seconds"] = round(time.perf_counter() - t0, 1)
    return ctx


SUPERSEDING_MARKER = "experiment_specification_v2_d1_d2"


def write_outputs(ctx: dict, out_dir: str = OUT_DIR) -> list:
    """Skips any file already superseded by specification v2 (D1/D2), so a re-run of this v1 audit never
    overwrites the frozen v2 specification. v1 content stays archived in spec_versions/v1/."""
    created: set = set()
    for name in OUTPUT_FILES:
        path = os.path.join(out_dir, name)
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                if SUPERSEDING_MARKER in f.read():
                    print(f"  skipping {name}: superseded by specification v2")
                    continue
        _safe_write(name, ctx["outputs"][name], created, out_dir=out_dir)
    return sorted(created)


if __name__ == "__main__":
    ctx = run()
    written = write_outputs(ctx)
    print(f"\nSTATUS: {ctx['decision']['classification']}")
    for d in ctx["decision"]["open_decisions"]:
        print(f"  {d['id']}: {d['question']}")
    print(f"protected unchanged: {ctx['integrity']['all_unchanged']}; wrote {len(written)} files in {ctx['seconds']}s")
