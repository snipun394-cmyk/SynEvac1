"""READ-ONLY pre-training data / feature / target integrity audit for the
proposed consequence-prediction experiment:

    "Train an AI model to predict the consequences of each candidate
    evacuation action from observable state, then use those predictions
    through a deterministic selector to generate an adaptive recommendation."

This script trains NOTHING, implements no selector, and never writes to the
simulator, observation.py, actions.py, the frozen RL branch, the frozen
controller (observable_robust_differential.py), or the existing candidate
dataset. It:

  1. loads the existing differential-targeting candidate dataset (the
     claimed "9,450 (state, action, outcome) tuples") and tabulates its
     real structure;
  2. deterministically REPLAYS each of the 450 decision states in memory
     (the same `action_space_audit._advance_to_tick` path that generated
     them) to regenerate the observation vector, which the dataset does
     NOT store;
  3. RE-RUNS the stored candidate rollouts in memory and checks the stored
     labels reproduce exactly;
  4. audits every candidate input feature for leakage, action dependence,
     identifiability, and trajectory dependence;
  5. writes its findings ONLY to the deliverable files listed in
     OUTPUT_FILES (it refuses to overwrite any file it did not itself
     produce -- see `_safe_write`).

Every protected source/data file is SHA-256 hashed before and after the run
and the run aborts if anything changed.

ROOTS: the research sandbox (`research_rl_evac_experiment/`) is untracked by
git, so this script may run from a git worktree that holds only the new
deliverables. DATA_ROOT (modules + existing data, read-only) resolves to
this script's own research directory when it contains simulator.py,
otherwise to $SYNEVAC_RESEARCH_ROOT, otherwise to the main checkout above a
`.claude/worktrees/<name>/` path. OUT_ROOT is always this script's own
research directory.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = os.path.dirname(SCRIPTS_DIR)


def _resolve_data_root() -> str:
    if os.path.isfile(os.path.join(OUT_ROOT, "simulator.py")):
        return OUT_ROOT
    env = os.environ.get("SYNEVAC_RESEARCH_ROOT")
    if env and os.path.isfile(os.path.join(env, "simulator.py")):
        return env
    norm = OUT_ROOT.replace("\\", "/")
    marker = "/.claude/worktrees/"
    if marker in norm:
        main_root = norm.split(marker)[0]
        cand = os.path.normpath(os.path.join(main_root, os.path.basename(OUT_ROOT)))
        if os.path.isfile(os.path.join(cand, "simulator.py")):
            return cand
    raise FileNotFoundError("cannot locate research_rl_evac_experiment (set SYNEVAC_RESEARCH_ROOT)")


DATA_ROOT = _resolve_data_root()
sys.path.insert(0, DATA_ROOT)
sys.path.insert(0, os.path.join(DATA_ROOT, "scripts"))

import action_space_audit as asa  # noqa: E402
import audit_selective_guidance_control as asgc  # noqa: E402
import audit_differential_targeting as adt  # noqa: E402
import audit_ai_learning_experiment_design as design  # noqa: E402
from actions import apply_action  # noqa: E402
from compliance import COMPLIANCE_PROBABILITY, generate_compliance_vector  # noqa: E402
from differential_targeting import apply_differential_action  # noqa: E402
from observation import (  # noqa: E402
    OBSERVATION_INDEX, OBSERVATION_INDEX_TEMPORAL, OBSERVATION_TEMPORAL,
    NO_HAZARD_TIME_SINCE_ONSET_SENTINEL, compute_observation,
)
from scenarios import EVAL_SEED_POOL, SCENARIOS  # noqa: E402
from topology import MAX_STEPS  # noqa: E402

OUT_DIR = os.path.join(OUT_ROOT, "artifacts", "ai_learning_experiment_design")
DATASET_PATH = os.path.join(DATA_ROOT, "artifacts", "differential_targeting",
                            "differential_targeting_raw_records.json")
TIMING_MATRIX_PATH = os.path.join(DATA_ROOT, "artifacts", "timing_conditioned_differential",
                                  "timing_action_matrix.json")

AUDIT_MARKER = "pre_training_ai_dataset_audit_v1"

OUTPUT_FILES = [
    "PRE_TRAINING_DATA_INTEGRITY_AUDIT.md",
    "exact_feature_manifest.json",
    "feature_leakage_audit.json",
    "dataset_structure.json",
    "target_generation_audit.json",
    "trajectory_dependence.json",
    "split_design.json",
    "target_identifiability.json",
    "selector_definition.json",
    "preregistered_success_criteria.json",
]

# Files (relative to DATA_ROOT) that must be byte-identical before and after this audit.
PROTECTED_RELATIVE_PATHS = [
    "simulator.py", "observation.py", "actions.py", "compliance.py", "scenarios.py", "topology.py",
    "metrics.py", "target_group.py", "differential_targeting.py", "deadband_heuristic.py",
    "capacity_aware_heuristic.py", "observable_robust_differential.py", "action_space_audit.py",
    "timing_experiment.py", "stress_test_scenarios.py", "gym_env.py", "differential_rl_action_space.py",
    "differential_rl_env.py",
    "scripts/audit_differential_targeting.py", "scripts/audit_selective_guidance_control.py",
    "scripts/evaluate_differential_rl.py", "scripts/audit_ai_learning_experiment_design.py",
    "scripts/audit_timing_conditioned_differential_policy.py",
    "artifacts/differential_targeting/differential_targeting_raw_records.json",
    "artifacts/differential_targeting/differential_targeting_results.json",
    "artifacts/differential_targeting/DIFFERENTIAL_TARGETING_AUDIT.md",
    "artifacts/timing_conditioned_differential/timing_action_matrix.json",
    "artifacts/ai_learning_experiment_design/AI_LEARNING_EXPERIMENT_DESIGN.md",
    "artifacts/ai_learning_experiment_design/candidate_paradigms.json",
    "artifacts/ai_learning_experiment_design/integrity_verification.json",
    "artifacts/ai_learning_experiment_design/leakage_audit.json",
    "artifacts/ai_learning_experiment_design/objective_definitions.json",
    "artifacts/ai_learning_experiment_design/observable_feature_analysis.json",
    "artifacts/ai_learning_experiment_design/proposed_experiment.json",
    "artifacts/ai_learning_experiment_design/simulator_gap_analysis.json",
    "artifacts/ai_learning_experiment_design/success_criteria.json",
]

TARGETS = ["T90", "T100", "total_unmet_post", "hazard_unmet_post"]
EVAL_SEED_SET = list(EVAL_SEED_POOL)
N_FOLDS = 5


# ===========================================================================
# Integrity helpers.
# ===========================================================================

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def protected_hashes() -> dict:
    out = {}
    for rel in PROTECTED_RELATIVE_PATHS:
        p = os.path.join(DATA_ROOT, rel)
        out[rel] = sha256_file(p) if os.path.isfile(p) else "MISSING"
    return out


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    return str(o)


def _safe_write(name: str, content, created_this_run: set, out_dir: str = OUT_DIR) -> str:
    """Only ever writes one of OUTPUT_FILES, only inside `out_dir`. Refuses to
    overwrite an existing file unless it carries this audit's marker (i.e.
    this script produced it), so no pre-existing artifact can be clobbered."""
    if name not in OUTPUT_FILES:
        raise PermissionError(f"refusing to write non-deliverable file {name!r}")
    path = os.path.join(out_dir, name)
    if os.path.exists(path) and name not in created_this_run:
        with open(path, encoding="utf-8") as f:
            if AUDIT_MARKER not in f.read():
                raise PermissionError(f"refusing to overwrite a file not produced by this audit: {path}")
    text = content if isinstance(content, str) else json.dumps(content, indent=2, default=_json_default)
    if AUDIT_MARKER not in text:
        raise ValueError("deliverable is missing the audit marker")
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    created_this_run.add(name)
    return path


# ===========================================================================
# Section 1: load + structure.
# ===========================================================================

def load_records(path: str = DATASET_PATH) -> list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def outcome_vector(res: dict) -> tuple:
    """The established 4-criterion ordering key (sort_key_full_ranking)."""
    m = res["metrics"]
    t90 = m["T90"] if m["T90_defined"] else float("inf")
    return (t90, m["T100"], res["total_unmet_demand"], res["hazard_edge_unmet_demand"])


def state_key(r: dict) -> tuple:
    return (r["scenario"], r["seed"], r["decision_tick"])


def trajectory_key(r: dict) -> tuple:
    return (r["scenario"], r["seed"])


def flatten_tuples(records: list) -> list:
    """One row per (state, candidate action) with a stored simulator outcome.
    Action keys: 'GLOBAL:<a>' (the heuristic's own action applied to every
    eligible occupant), 'NO_INTERVENTION', or the pairwise key 'Zi=a|Zj=b'."""
    rows = []
    for r in records:
        sk = state_key(r)
        for name, res in r["candidates"].items():
            akey = f"GLOBAL:{r['heuristic_action']}" if name == "GLOBAL" else name
            rows.append(dict(state=sk, scenario=r["scenario"], seed=r["seed"], tick=r["decision_tick"],
                             family=name, action_key=akey, res=res, outcome=outcome_vector(res)))
        for key, res in r["pairwise_results"].items():
            rows.append(dict(state=sk, scenario=r["scenario"], seed=r["seed"], tick=r["decision_tick"],
                             family="PAIRWISE", action_key=key, res=res, outcome=outcome_vector(res)))
    return rows


def scenario_t_h(scenario: str):
    return SCENARIOS[scenario]["t_h"]


def dataset_structure(records: list, rows: list) -> dict:
    keys_present = sorted({k for r in records for k in r.keys()})
    n_candidate = sum(len(r["candidates"]) for r in records)
    n_pairwise = sum(len(r["pairwise_results"]) for r in records)
    per_st = defaultdict(lambda: dict(n_states=0, n_tuples=0, nonempty_zones=Counter(), n_pairwise=Counter()))
    for r in records:
        c = per_st[f"{r['scenario']}_tick{r['decision_tick']}"]
        c["n_states"] += 1
        c["n_tuples"] += len(r["candidates"]) + len(r["pairwise_results"])
        c["nonempty_zones"]["+".join(r["nonempty_zones"]) or "(none)"] += 1
        c["n_pairwise"][len(r["pairwise_results"])] += 1
    per_st = {k: dict(n_states=v["n_states"], n_tuples=v["n_tuples"],
                      nonempty_zones=dict(v["nonempty_zones"]), pairwise_per_state=dict(v["n_pairwise"]))
              for k, v in sorted(per_st.items())}
    action_keys = Counter(r["action_key"] for r in rows)
    pair_zone_pairs = Counter("|".join(z.split("=")[0] for z in r["action_key"].split("|"))
                              for r in rows if r["family"] == "PAIRWISE")
    n_with_pairs = sum(1 for r in records if r["pairwise_results"])
    return dict(
        file=os.path.relpath(DATASET_PATH, DATA_ROOT).replace("\\", "/"),
        file_sha256=sha256_file(DATASET_PATH),
        n_records_decision_states=len(records),
        record_fields=keys_present,
        stores_observation_or_feature_vector=any(k in keys_present for k in ("observation", "obs", "features")),
        n_candidate_tuples_GLOBAL_and_NO_INTERVENTION=n_candidate,
        n_pairwise_tuples=n_pairwise,
        n_total_state_action_outcome_tuples=len(rows),
        exact_row_count_matches_claim_9450=bool(len(rows) == 9450),
        n_unique_states=len({r["state"] for r in rows}),
        n_unique_action_keys=len(action_keys),
        action_family_counts=dict(Counter(r["family"] for r in rows)),
        global_action_variants=dict(Counter(r["action_key"] for r in rows if r["family"] == "GLOBAL")),
        pairwise_zone_pairs=dict(pair_zone_pairs),
        scenarios=dict(Counter(r["scenario"] for r in records)),
        seeds=dict(n_unique=len({r["seed"] for r in records}), min=min(r["seed"] for r in records),
                   max=max(r["seed"] for r in records), pool="EVAL_SEED_POOL (10000-10049)"),
        decision_ticks={str(k): v for k, v in sorted(Counter(r["decision_tick"] for r in records).items())},
        hazard_timings_represented={s: scenario_t_h(s) for s in sorted({r["scenario"] for r in records})},
        compliance_probabilities_represented=[COMPLIANCE_PROBABILITY],
        compliance_probability_source=(
            "NOT stored in the dataset. compliance.COMPLIANCE_PROBABILITY=0.9 is the module default used by "
            "gym_env.reset() -> generate_compliance_vector(scenario, seed) with no p override; the generator "
            "script never varies it."),
        every_tuple_has_simulator_outcome=bool(all(r["res"].get("metrics") is not None for r in rows)),
        n_exact_duplicate_state_action_rows=len(rows) - len({(r["state"], r["action_key"]) for r in rows}),
        n_rows_T90_undefined=sum(1 for r in rows if not r["res"]["metrics"]["T90_defined"]),
        n_rows_T100_censored=sum(1 for r in rows if r["res"]["metrics"]["T100_censored"]),
        per_scenario_tick=per_st,
        n_states_with_zero_pairwise_rows=len(records) - n_with_pairs,
        composition_of_9450=(
            f"{len(records)} states x 2 'candidates' (GLOBAL = the heuristic's own action, NO_INTERVENTION) = "
            f"{n_candidate}, plus {n_pairwise} pairwise zone-differential rollouts concentrated on {n_with_pairs} "
            f"states. It is NOT 9,450 independent states; it is {len(records)} states with a highly unequal "
            "number of evaluated actions per state."),
    )


# ===========================================================================
# Section 2: deterministic state replay (regenerates the missing features).
# ===========================================================================

def _state_fingerprint(sim) -> str:
    payload = json.dumps(dict(
        t=sim.t, t_h=sim.t_h,
        occ={str(k): v for k, v in sorted(sim.occupants.items())},
        zw={k: list(v) for k, v in sim.zone_waiting.items()},
        et={k: list(v) for k, v in sim.edge_transit.items()},
        ev={str(k): v for k, v in sorted(sim.evacuated.items())},
    ), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def _noncompliant_by_room(scenario: str, cv) -> tuple:
    dist = SCENARIOS[scenario]["distribution"]
    out, oid = [], 0
    for z in ["R1", "R2", "R3", "R4"]:
        n = dist.get(z, 0)
        out.append(int(sum(1 for i in range(oid, oid + n) if not cv[i])))
        oid += n
    return tuple(out)


def replay_state(scenario: str, seed: int, tick: int) -> dict:
    st = asa._advance_to_tick(scenario, seed, tick)
    sim = st["sim"]
    obs27 = np.asarray(compute_observation(sim), dtype=float)
    obs31 = np.asarray(compute_observation(sim, obs_version=OBSERVATION_TEMPORAL), dtype=float)
    cv = generate_compliance_vector(scenario, seed)
    active = [oid for oid, o in sim.occupants.items() if o["location"] != "evacuated"]
    h_ids = list(sim.zone_waiting["H"])
    occ = sim.occupants
    hidden = dict(
        compliance_vector_sha=hashlib.sha256(np.asarray(cv, dtype=np.uint8).tobytes()).hexdigest()[:16],
        n_compliant_total=int(cv.sum()),
        n_noncompliant_active=sum(1 for oid in active if not occ[oid]["compliant"]),
        n_noncompliant_at_H=sum(1 for oid in h_ids if not occ[oid]["compliant"]),
        n_noncompliant_at_H_baseline_A=sum(1 for oid in h_ids if not occ[oid]["compliant"]
                                           and occ[oid]["baseline_target"] == "A"),
        n_committed_active=sum(1 for oid in active if occ[oid]["committed"]),
        noncompliant_by_initial_room=_noncompliant_by_room(scenario, cv),
    )
    return dict(
        scenario=scenario, seed=seed, tick=tick, reached=st["reached"], sim=sim,
        obs27=obs27, obs31=obs31, t_h=st["t_h"], has_hazard=st["has_hazard"],
        prior_actions=list(st["prior_actions"]), prior_total_unmet=st["prior_total_unmet"],
        prior_hazard_unmet=st["prior_hazard_unmet"], hidden=hidden, fingerprint=_state_fingerprint(sim),
        heuristic_action=int(asgc.HEURISTIC(obs27)),
        nonempty_zones=[z for z in asgc.ELIGIBLE_ZONES if len(sim.zone_waiting[z]) > 0],
    )


def replay_all_states(records: list) -> dict:
    return {state_key(r): replay_state(*state_key(r)) for r in records}


def replay_consistency(records: list, states: dict) -> dict:
    mismatches = []
    for r in records:
        s = states[state_key(r)]
        if not s["reached"] or s["heuristic_action"] != r["heuristic_action"] \
                or s["nonempty_zones"] != r["nonempty_zones"]:
            mismatches.append(list(state_key(r)))
    return dict(n_states_replayed=len(records), n_replay_mismatches=len(mismatches),
                mismatches=mismatches[:20], all_reached=all(s["reached"] for s in states.values()),
                passed=bool(not mismatches))


# ===========================================================================
# Feature manifest.
# ===========================================================================

WAITING_IDX = list(range(0, 8))
TREND_NAMES = [f"delta_{OBSERVATION_INDEX[i]}" for i in WAITING_IDX] + ["delta_waiting_H_total",
                                                                        "delta_residual_cap_H_CE1"]

CLASS_LABELS = {
    "A": "DIRECTLY OBSERVABLE BEFORE DECISION",
    "B": "REALISTICALLY ESTIMABLE BEFORE DECISION",
    "C": "SIMULATOR-DERIVED BUT PHYSICALLY OBSERVABLE",
    "D": "HIDDEN GROUND TRUTH",
    "E": "POST-DECISION LEAKAGE",
    "F": "FUTURE-INFORMATION LEAKAGE",
    "G": "UNCERTAIN - REQUIRES INVESTIGATION",
}

END_OF_TICK = ("compute_observation(sim) after tick t-1 completes (sim.t already incremented to the decision "
               "tick t), BEFORE the candidate's relabeling in tick t's step 2/3")


def _feat(name, source, computation, available, future, compliance, oracle, action_dep, cls, note=""):
    assert cls in CLASS_LABELS
    return dict(name=name, source_variable=source, computation=computation, available_when=available,
                depends_on_future_state=future, depends_on_hidden_compliance=compliance,
                depends_on_oracle_information=oracle, depends_on_candidate_action=action_dep,
                classification=cls, classification_label=CLASS_LABELS[cls], note=note)


def build_feature_manifest() -> list:
    f = []
    for i, z in enumerate(["R1", "R2", "C1", "R3", "R4", "C2", "CE1", "CE2"]):
        f.append(_feat(OBSERVATION_INDEX[i], f"len(sim.zone_waiting['{z}'])", f"count of occupants waiting in {z}",
                       END_OF_TICK, False, False, False, False, "A",
                       "Zone headcount; a real system would estimate it from perception (noise not modeled)."))
    q_note = ("Reads the hidden per-occupant intent label target_exit. Non-compliant occupants keep their parity "
              "baseline label, so after a non-parity global action the minority-label count at the hub is a "
              "read-out of hidden non-compliance (quantified in feature_leakage_audit.json). It is also the only "
              "feature that changes when a candidate relabels (action-independence audit).")
    f.append(_feat("Q_A_at_H", "sim.occupants[oid]['target_exit'] for oid in zone_waiting['H']",
                   "count of hub-waiting occupants whose HIDDEN target_exit == 'A'", END_OF_TICK,
                   False, True, False, False, "D", q_note))
    f.append(_feat("Q_B_at_H", "as Q_A_at_H with label 'B'", "len(zone_waiting['H']) - Q_A_at_H", END_OF_TICK,
                   False, True, False, False, "D", q_note))
    for i in range(10, 24):
        name = OBSERVATION_INDEX[i]
        split = name.endswith("_1tick") or name.endswith("_2tick")
        f.append(_feat(name, "sim.edge_transit[...]" + (" + occupants[oid]['ticks_remaining']" if split else ""),
                       "count of occupants in transit on the edge" + (" with that many ticks remaining" if split else ""),
                       END_OF_TICK, False, False, False, False, "B" if split else "A",
                       "Remaining-ticks split needs per-person tracking of corridor entry time: estimable, not a "
                       "raw count." if split else "Corridor headcount."))
    f.append(_feat("residual_cap_H_CE1_normalized", "topology.edge_capacity('H_CE1', sim.t, sim.t_h) / 6",
                   "1.0 if sim.t < t_h else 2/6",
                   END_OF_TICK + ". The hazard update is step 1 of tick t, executed BEFORE the decision (step 2/3), "
                   "so the degraded value at t == t_h is already physically true at decision time",
                   False, False, False, False, "C",
                   "Binary 'hazard has occurred' indicator. Never reveals t_h before onset (verified on all replayed states)."))
    f.append(_feat("normalized_time", "sim.t", "t / MAX_STEPS", END_OF_TICK, False, False, False, False, "A",
                   "Time since alarm; known to an operator."))
    f.append(_feat("normalized_remaining", "sim.n_active()", "(N - evacuated) / N", END_OF_TICK,
                   False, False, False, False, "B", "Needs the initial headcount N and exit counters."))
    f.append(_feat("normalized_sim_time_temporal", "sim.t", "t / MAX_STEPS (== normalized_time)",
                   END_OF_TICK, False, False, False, False, "A", "Exact duplicate of normalized_time; excluded as redundant."))
    f.append(_feat("normalized_decision_index", "sim.t", "(t/5)/(180/5) (== normalized_time at decision ticks)",
                   END_OF_TICK, False, False, False, False, "A", "Exact duplicate of normalized_time; excluded as redundant."))
    f.append(_feat("time_since_hazard_onset_normalized", "sim.t_h (the scheduled FUTURE onset tick)",
                   "(t - t_h)/MAX_STEPS whenever t_h <= MAX_STEPS, else sentinel -2.0; computed even when t < t_h",
                   "every observation, including before the hazard exists", True, False, True, False, "F",
                   "Before onset, t - value*MAX_STEPS recovers the exact future onset tick; the sentinel reveals "
                   "whether a hazard will ever happen. Source: observation.py `has_hazard = sim.t_h <= MAX_STEPS`. "
                   "Confirmed empirically on the replayed states."))
    f.append(_feat("hazard_phase", "sim.t_h", "0.0 no-hazard scenario / 0.5 if t < t_h / 1.0 if t >= t_h",
                   "every observation", True, False, True, False, "F",
                   "0.5 before onset reveals that a hazard WILL happen (vs 0.0 in S1). Only the post-onset 1.0 is "
                   "observable, and it duplicates residual_cap_H_CE1_normalized."))
    for name in TREND_NAMES:
        is_cap = name == "delta_residual_cap_H_CE1"
        f.append(_feat(name, "observation at t and at t - DECISION_INTERVAL (NOT IMPLEMENTED in any module)",
                       "x(t) - x(t-5)" + ("; H total uses Q_A+Q_B" if name == "delta_waiting_H_total" else ""),
                       "decision tick t, from past observations only; UNDEFINED at t = 0", False, False, False, False,
                       "C" if is_cap else "A",
                       "Proposed trend family. Not implemented anywhere; computed by this audit in memory only. "
                       "Non-zero only when onset fell in (t-5, t]." if is_cap else
                       "Proposed 'delta-waiting per zone' trend family. Not implemented anywhere; computed by this "
                       "audit in memory only."))
    f.append(_feat("waiting_H_total", "len(sim.zone_waiting['H'])", "Q_A_at_H + Q_B_at_H (label-free)",
                   END_OF_TICK, False, False, False, False, "A",
                   "Label-free replacement for the hidden Q_A/Q_B split; derivable from the existing observation."))
    for name, src, cls, comp, note in [
        ("scenario_id", "record['scenario']", "D", False, "Template identity; lets a model memorize the scenario."),
        ("seed", "record['seed']", "D", True, "Keys the hidden compliance vector."),
        ("t_h", "SCENARIOS[s]['t_h']", "F", False, "Future onset tick; unknown to a real operator before onset."),
        ("compliance_probability_p", "compliance.COMPLIANCE_PROBABILITY", "D", True, "Population parameter; constant 0.9 here."),
        ("compliance_vector_or_n_compliant", "sim.occupants[oid]['compliant']", "D", True, "Hidden per-occupant compliance."),
        ("committed_flags", "sim.occupants[oid]['committed']", "D", False, "Hidden."),
        ("prior_total_unmet_or_prior_hazard_unmet", "_advance_to_tick bookkeeping", "G", False,
         "Cumulative pre-decision overflow. Not in the observation; arguably estimable from queue history; ALSO an "
         "action-independent additive part of the stored unmet targets (target audit T1). Needs a decision before use."),
        ("heuristic_action", "HEURISTIC(obs)", "G", True,
         "Deterministic function of obs INCLUDING hidden Q_A/Q_B; usable as a candidate identifier, not as a feature."),
        ("candidate_outcomes_or_oracle_best", "stored rollouts", "E", False, "Target only; never an input."),
    ]:
        f.append(_feat(name, src, "excluded", "n/a", cls == "F", comp, cls in ("E", "F"), False, cls, note))
    return f


def admissible_state_feature_names(manifest: list) -> list:
    """v1 admissible = classes A/B/C only, computable from the EXISTING
    27-D observation today (no new code), redundant duplicates excluded,
    proposed-but-unimplemented trend features excluded."""
    out = []
    for m in manifest:
        n = m["name"]
        if m["classification"] not in ("A", "B", "C"):
            continue
        if n in ("normalized_sim_time_temporal", "normalized_decision_index") or n.startswith("delta_"):
            continue
        out.append(n)
    return out


_OBS_IDX = {n: i for i, n in enumerate(OBSERVATION_INDEX)}


def admissible_vector(state: dict, names: list) -> np.ndarray:
    obs = state["obs27"]
    return np.asarray([obs[8] + obs[9] if n == "waiting_H_total" else obs[_OBS_IDX[n]] for n in names], dtype=float)


def trend_vector(state: dict, prev_state) -> np.ndarray:
    if prev_state is None:
        return np.full(len(TREND_NAMES), np.nan)
    cur, prev = state["obs27"], prev_state["obs27"]
    d = [cur[i] - prev[i] for i in WAITING_IDX]
    d.append((cur[8] + cur[9]) - (prev[8] + prev[9]))
    d.append(cur[24] - prev[24])
    return np.asarray(d, dtype=float)


# ===========================================================================
# Sections 2/3: empirical leakage + action-independence checks.
# ===========================================================================

def temporal_leakage_check(states: dict) -> dict:
    i29 = OBSERVATION_INDEX_TEMPORAL.index("time_since_hazard_onset_normalized")
    i30 = OBSERVATION_INDEX_TEMPORAL.index("hazard_phase")
    pre = [s for s in states.values() if s["has_hazard"] and s["tick"] < s["t_h"]]
    recovered = sum(1 for s in pre if round(s["tick"] - s["obs31"][i29] * MAX_STEPS) == s["t_h"])
    phase_half = sum(1 for s in pre if s["obs31"][i30] == 0.5)
    no_haz = [s for s in states.values() if not s["has_hazard"]]
    sentinel = sum(1 for s in no_haz if s["obs31"][i29] == NO_HAZARD_TIME_SINCE_ONSET_SENTINEL)
    return dict(
        n_pre_onset_hazard_states=len(pre),
        n_pre_onset_states_where_feature29_recovers_exact_future_t_h=recovered,
        n_pre_onset_states_with_hazard_phase_0_5=phase_half,
        n_no_hazard_states=len(no_haz), n_no_hazard_states_with_sentinel=sentinel,
        baseline_residual_cap_values_pre_onset={str(k): v for k, v in Counter(float(s["obs27"][24]) for s in pre).items()},
        leak_confirmed=bool(pre and recovered == len(pre) and phase_half == len(pre)),
        source_evidence="observation.py: `has_hazard = sim.t_h <= MAX_STEPS`, then `time_since_onset = "
                        "(sim.t - sim.t_h) / MAX_STEPS` and `hazard_phase = 1.0 if sim.t >= sim.t_h else 0.5` -- "
                        "no check that the onset has happened.",
    )


def residual_capacity_timing_check(states: dict) -> dict:
    bad = []
    for s in states.values():
        expected = 1.0 if (not s["has_hazard"] or s["tick"] < s["t_h"]) else 2.0 / 6.0
        if abs(s["obs27"][24] - expected) > 1e-12:
            bad.append([s["scenario"], s["seed"], s["tick"]])
    with open(os.path.join(DATA_ROOT, "simulator.py"), encoding="utf-8") as fh:
        src = fh.read()
    order_ok = src.index("Step 1: hazard update") < src.index("Step 2/3: decision + target-label update")
    return dict(n_states=len(states), n_value_mismatches=len(bad), mismatches=bad[:10],
                hazard_update_precedes_decision_in_tick_order=order_ok,
                reveals_future_onset=False, passed=bool(not bad and order_ok))


def q_label_compliance_leak_check(states: dict) -> dict:
    """After a FAVOR_B prior action, eligible compliant occupants carry 'B';
    'A' labels at the hub belong to non-compliant baseline-A occupants (or to
    occupants who were in transit at the last decision). Measures how often
    Q_A_at_H equals the hidden non-compliant baseline-A hub count exactly."""
    xs, ys, hs = [], [], []
    for s in states.values():
        if not s["prior_actions"] or s["prior_actions"][-1] != 2 or s["obs27"][8] + s["obs27"][9] == 0:
            continue
        xs.append(int(s["obs27"][8]))
        ys.append(s["hidden"]["n_noncompliant_at_H_baseline_A"])
        hs.append(float(s["obs27"][8] + s["obs27"][9]))
    n = len(xs)
    x, y, h = (np.asarray(v, dtype=float) for v in (xs, ys, hs))

    def _r(a, b):
        return round(float(np.corrcoef(a, b)[0, 1]), 4) if n > 2 and np.std(a) > 0 and np.std(b) > 0 else None

    def _resid(v):
        if np.std(h) == 0:
            return v - v.mean()
        beta = np.polyfit(h, v, 1)
        return v - np.polyval(beta, h)

    corr = _r(x, y)
    partial = _r(_resid(x), _resid(y))
    return dict(n_states_prior_action_favor_B_with_nonempty_hub=n,
                n_states_Q_A_equals_hidden_noncompliant_baseline_A_count=sum(1 for a, b in zip(xs, ys) if a == b),
                mean_Q_A_at_H=round(float(x.mean()), 3) if n else None,
                mean_hidden_noncompliant_baseline_A_at_H=round(float(y.mean()), 3) if n else None,
                pearson_r=corr, partial_r_controlling_for_hub_size=partial,
                leaks_hidden_compliance=bool(corr is not None and corr >= 0.5),
                why_not_exact="Occupants in transit at the last decision tick were not eligible for relabel and keep "
                              "an older label, so the count is a strong proxy rather than an exact read-out.")


def _parse_pair(key: str) -> list:
    return [(p.split("=")[0], int(p.split("=")[1])) for p in key.split("|")]


def action_independence_check(records: list, states: dict) -> dict:
    """Recomputes the observation AFTER applying each candidate's first-decision
    relabeling on a deepcopy. Any index that changes is a feature that would
    become action-dependent if it were (wrongly) computed after the action."""
    changed27, changed31 = Counter(), Counter()
    n_checked, pre_ok = 0, True
    for r in records:
        s = states[state_key(r)]
        base27, base31 = s["obs27"], s["obs31"]
        if not np.array_equal(np.asarray(compute_observation(s["sim"]), dtype=float), base27):
            pre_ok = False
        cands = [lambda sim, a=r["heuristic_action"]: apply_action(sim, a), lambda sim: None]
        cands += [lambda sim, asg=_parse_pair(k): apply_differential_action(sim, asg) for k in r["pairwise_results"]]
        for fn in cands:
            fork = copy.deepcopy(s["sim"])
            fn(fork)
            o27 = np.asarray(compute_observation(fork), dtype=float)
            o31 = np.asarray(compute_observation(fork, obs_version=OBSERVATION_TEMPORAL), dtype=float)
            for i in np.nonzero(o27 != base27)[0]:
                changed27[OBSERVATION_INDEX[i]] += 1
            for i in np.nonzero(o31 != base31)[0]:
                changed31[OBSERVATION_INDEX_TEMPORAL[i]] += 1
            n_checked += 1
    return dict(
        n_state_candidate_pairs_checked=n_checked,
        observation_recomputed_on_unmodified_state_is_identical=pre_ok,
        features_that_change_if_computed_after_candidate_relabel=dict(changed27),
        temporal_only_features_that_change={k: v for k, v in changed31.items() if k not in changed27},
        action_sensitive_features=sorted(changed27),
        as_generated_any_feature_depends_on_candidate=False,
        interpretation="Every state feature is computed at end of tick t-1, before the candidate's relabel (step "
                       "2/3 of tick t), so as generated none depends on the candidate. Only the listed features "
                       "would become action-dependent under a wrong computation order; both are hidden-label "
                       "features already excluded from the admissible set.",
    )


# ===========================================================================
# Section 4: target / label audit.
# ===========================================================================

def _rerun(state: dict, record: dict, family: str, key):
    common = dict(continuation_policy=asgc.HEURISTIC, has_hazard=state["has_hazard"], t_h=state["t_h"],
                  prior_actions=list(state["prior_actions"]), prior_total_unmet=state["prior_total_unmet"],
                  prior_hazard_unmet=state["prior_hazard_unmet"])
    fork = copy.deepcopy(state["sim"])
    start_fp = _state_fingerprint(fork)
    if family == "GLOBAL":
        res = asgc.run_intervention(fork, record["heuristic_action"], None, **common)
    elif family == "NO_INTERVENTION":
        res = asgc.run_intervention(fork, None, None, **common)
    else:
        res = adt.run_differential_intervention(fork, _parse_pair(key), **common)
    realized = sum(e["h_ce1_unmet"] for e in fork.tick_log if e["t"] >= state["tick"])
    return res, start_fp, realized


def reproduce_labels(records: list, states: dict, pairwise_stride: int = 1) -> dict:
    """Re-runs every GLOBAL / NO_INTERVENTION rollout and every
    `pairwise_stride`-th pairwise rollout from the replayed state."""
    n, mism, start_mismatch, s1_overflow = 0, [], 0, 0
    realized_cmp = Counter()
    t0 = time.perf_counter()
    for r in records:
        s = states[state_key(r)]
        jobs = [("GLOBAL", None, r["candidates"]["GLOBAL"]), ("NO_INTERVENTION", None, r["candidates"]["NO_INTERVENTION"])]
        jobs += [("PAIRWISE", k, v) for j, (k, v) in enumerate(r["pairwise_results"].items()) if j % pairwise_stride == 0]
        for fam, key, stored in jobs:
            res, fp, realized = _rerun(s, r, fam, key)
            n += 1
            start_mismatch += int(fp != s["fingerprint"])
            if not (res["metrics"] == stored["metrics"]
                    and res["total_unmet_demand"] == stored["total_unmet_demand"]
                    and res["hazard_edge_unmet_demand"] == stored["hazard_edge_unmet_demand"]
                    and res["switch_count"] == stored["switch_count"]):
                mism.append([r["scenario"], r["seed"], r["decision_tick"], fam, key])
            if s["has_hazard"]:
                diag_post = stored["hazard_edge_unmet_demand"] - s["prior_hazard_unmet"]
                realized_cmp["diagnostic_equals_realized" if diag_post == realized else "diagnostic_differs"] += 1
            elif realized > 0:
                s1_overflow += 1
    dt = time.perf_counter() - t0
    return dict(
        n_rollouts_rerun=n, pairwise_stride=pairwise_stride, n_label_mismatches=len(mism), mismatches=mism[:20],
        n_rollouts_not_starting_from_identical_state=start_mismatch,
        hazard_unmet_post_diagnostic_vs_realized_tick_log=dict(realized_cmp),
        n_no_hazard_rollouts_with_realized_H_CE1_overflow_but_label_0=s1_overflow,
        seconds=round(dt, 2), seconds_per_rollout=round(dt / max(n, 1), 5),
        passed=bool(not mism and start_mismatch == 0),
    )


def target_structure_checks(records: list, states: dict) -> dict:
    by_traj = defaultdict(dict)
    for r in records:
        by_traj[trajectory_key(r)][r["decision_tick"]] = outcome_vector(r["candidates"]["GLOBAL"])
    same = sum(1 for d in by_traj.values() if len(set(d.values())) == 1)
    eq = [r for r in records if outcome_vector(r["candidates"]["NO_INTERVENTION"]) == outcome_vector(r["candidates"]["GLOBAL"])]
    shares = [states[state_key(r)]["prior_total_unmet"] / r["candidates"]["GLOBAL"]["total_unmet_demand"]
              for r in records if r["candidates"]["GLOBAL"]["total_unmet_demand"] > 0]
    pre_onset_haz = [states[state_key(r)]["prior_hazard_unmet"] for r in records
                     if states[state_key(r)]["has_hazard"] and r["decision_tick"] > 0]
    return dict(
        n_trajectories=len(by_traj),
        n_trajectories_where_GLOBAL_outcome_identical_at_all_3_ticks=same,
        n_states_NO_INTERVENTION_outcome_equals_GLOBAL=len(eq),
        n_tick0_states_NO_INTERVENTION_equals_GLOBAL=sum(1 for r in eq if r["decision_tick"] == 0),
        n_states_with_nonzero_prior_total_unmet=sum(1 for s in states.values() if s["prior_total_unmet"] > 0),
        prior_unmet_share_of_GLOBAL_total_unmet=dict(mean=round(float(np.mean(shares)), 4) if shares else None,
                                                     max=round(float(np.max(shares)), 4) if shares else None),
        prior_hazard_unmet_at_later_ticks_values=dict(Counter(pre_onset_haz)),
        S1_hazard_edge_unmet_label_values=dict(Counter(r["candidates"]["GLOBAL"]["hazard_edge_unmet_demand"]
                                                       for r in records if r["scenario"] == "S1")),
    )


def target_generation_audit(label_repro: dict, structure: dict) -> dict:
    return dict(
        audit=AUDIT_MARKER,
        generator="scripts/audit_differential_targeting.py::run_full_differential_sweep -> audit_one_state_differential",
        state_construction="action_space_audit._advance_to_tick: resets ResearchEvacGymEnv(scenario) with the eval "
                           "seed and replays the GLOBAL_DEADBAND heuristic (margin=RECOMMENDED_MARGIN) to the decision tick.",
        rollouts=dict(
            GLOBAL="audit_selective_guidance_control.run_intervention(deepcopy(sim), heuristic_action, None): the "
                   "heuristic's own action applied to every eligible occupant at the decision tick.",
            NO_INTERVENTION="run_intervention(deepcopy(sim), None, None): no relabel at the decision tick.",
            PAIRWISE="audit_differential_targeting.run_differential_intervention(deepcopy(sim), [(Zi,a),(Zj,b)]): "
                     "relabel ONLY eligible occupants of two currently non-empty zones; every OTHER zone keeps its "
                     "existing labels (not the heuristic's).",
            continuation="Every later decision: the same deterministic GLOBAL_DEADBAND heuristic, applied globally.",
        ),
        targets=dict(
            T90="compute_metrics(sim.evacuated).T90 (tick at which ceil(0.9*40)=36 occupants have exited; inf if undefined)",
            T100="compute_metrics(sim.evacuated).T100 (last exit tick; MAX_STEPS if censored)",
            total_unmet_demand="prior_total_unmet + sum over post-decision ticks of timing_experiment."
                               "diagnostic_all_edges_unmet (sum over all edges of max(0, candidates - capacity), "
                               "measured BEFORE each tick's relabel/admission)",
            hazard_edge_unmet_demand="if the scenario has a scheduled hazard: prior_hazard_unmet + per-tick H_CE1 "
                                     "overflow for branch-A candidates (from t=0, including pre-onset ticks); else 0",
        ),
        checks=dict(
            real_counterfactual_simulator_rollout=True,
            rollout_starts_from_exactly_the_same_state=label_repro["n_rollouts_not_starting_from_identical_state"] == 0,
            stored_labels_reproduce_exactly=label_repro["passed"],
            prior_unmet_preserved="YES: prior_total_unmet / prior_hazard_unmet are seeded into the accumulators, so "
                                  "stored totals = pre-decision history + post-decision.",
            candidate_applied_only_at_decision_point="YES: target_group / apply_differential only on the first macro-step.",
            future_decisions_controlled_consistently="YES: deterministic heuristic continuation in every arm.",
            candidate_effect_isolated="YES for the one-step effect: arms differ only in the first decision.",
            input_contains_candidate_information="NO: features are computed before the relabel (action-independence audit).",
        ),
        label_reproduction=label_repro,
        structure=structure,
        contamination_and_definition_issues=[
            dict(id="T1", severity="must-fix before regression",
                 issue="Stored total_unmet_demand and hazard_edge_unmet_demand INCLUDE the pre-decision accumulated "
                       "unmet. That part is identical for every candidate at a state and is not in the observation: "
                       "an unobserved additive offset for regression (harmless for within-state ranking).",
                 fix="Target = stored - prior (post-decision unmet)."),
            dict(id="T2", severity="must-fix",
                 issue="hazard_edge_unmet_demand counts H_CE1 overflow from t=0 whenever a hazard is SCHEDULED "
                       "(including pre-onset ticks at nominal capacity) and is forced to 0 in S1 even when H_CE1 "
                       "overflows. The label therefore encodes whether a hazard is scheduled (future information "
                       "in the TARGET, not an input leak): identical pre-onset S1/S2 observations get "
                       "systematically different hazard-edge labels.",
                 fix="Define it uniformly: H_CE1 overflow at ticks t >= t_h only (0 if no hazard), or H_CE1 overflow "
                     "at all ticks for every scenario."),
            dict(id="T3", severity="disclose",
                 issue="Per-tick unmet is measured BEFORE that tick's relabel and admission, so on the decision tick "
                       "it uses PRE-action labels (all arms alike). The count may therefore differ from the "
                       "simulator's own realized tick_log overflow.",
                 evidence=label_repro["hazard_unmet_post_diagnostic_vs_realized_tick_log"]),
            dict(id="T4", severity="must-fix for a common action space",
                 issue="Candidate semantics differ across existing data. PAIRWISE rows here leave non-targeted zones "
                       "untouched; evaluate_differential_rl._oracle_candidate_continuation (the machinery the prior "
                       "design proposed for new data, and the semantics of the frozen robust controller's "
                       "Discrete(211) space) applies the heuristic globally THEN overrides, with prior unmet reset "
                       "to 0. Rows from the two machineries are neither the same action nor the same target.",
                 fix="Adopt ONE candidate semantics and ONE target definition for all training data."),
            dict(id="T5", severity="must-fix for a comparable candidate set",
                 issue="'GLOBAL' is only the heuristic's chosen action (PARITY or FAVOR_B here); the unchosen global "
                       "actions were never rolled out. 250/450 states have ONLY 2 candidates; pairwise rows exist "
                       "only where >= 2 zones are non-empty.",
                 fix="Evaluate one fixed candidate set at every state (e.g. the 3 global actions, NO_INTERVENTION, "
                     "and all feasible zone overrides)."),
            dict(id="T6", severity="disclose",
                 issue="Labels are ONE-STEP counterfactuals with heuristic continuation: they describe 'take a now, "
                       "then the heuristic', not an AI-controlled closed loop.",
                 fix="Closed-loop evaluation must be a separate, explicitly-run experiment."),
        ],
    )


# ===========================================================================
# Section 5: trajectory dependence.
# ===========================================================================

def trajectory_dependence(records: list, rows: list, states: dict, adm: list) -> dict:
    per_traj = Counter((r["scenario"], r["seed"]) for r in rows)
    per_seed = Counter(r["seed"] for r in rows)
    adm_vec = {k: tuple(admissible_vector(s, adm).tolist()) for k, s in states.items()}
    obs_mult = Counter(tuple(s["obs27"].tolist()) for s in states.values())
    adm_mult = Counter(adm_vec.values())
    cv_by_traj = {(s["scenario"], s["seed"]): s["hidden"]["compliance_vector_sha"] for s in states.values()}
    cv_mult = Counter(cv_by_traj.values())
    room_profiles = Counter((s["scenario"], s["hidden"]["noncompliant_by_initial_room"])
                            for s in states.values() if s["tick"] == 0)
    by_group = defaultdict(set)
    scen_by_vec = defaultdict(set)
    groups_by_vec = defaultdict(set)
    for k, s in states.items():
        by_group[f"{s['scenario']}_tick{s['tick']}"].add(adm_vec[k])
        scen_by_vec[adm_vec[k]].add(s["scenario"])
        groups_by_vec[adm_vec[k]].add(f"{s['scenario']}_tick{s['tick']}")
    shared_groups = sorted(" = ".join(sorted(g)) for v, g in groups_by_vec.items() if len(scen_by_vec[v]) > 1)

    def hist(c):
        return {str(k): v for k, v in sorted(Counter(c.values()).items())}

    return dict(
        audit=AUDIT_MARKER,
        rows_per_trajectory=dict(n_trajectories=len(per_traj), histogram_rows_to_n_trajectories=hist(per_traj),
                                 min=min(per_traj.values()), max=max(per_traj.values())),
        states_per_trajectory=hist(Counter(trajectory_key(r) for r in records)),
        rows_per_seed=dict(n_seeds=len(per_seed), min=min(per_seed.values()), max=max(per_seed.values())),
        rows_per_scenario=dict(Counter(r["scenario"] for r in rows)),
        rows_per_decision_tick={str(k): v for k, v in sorted(Counter(r["tick"] for r in rows).items())},
        repeated_state_frequency=dict(
            n_states=len(states),
            n_distinct_27D_observations=len(obs_mult),
            n_distinct_admissible_feature_vectors=len(adm_mult),
            admissible_vector_multiplicity_histogram=hist(adm_mult),
            largest_identical_admissible_group=max(adm_mult.values()),
            distinct_admissible_vectors_per_scenario_tick={k: len(v) for k, v in sorted(by_group.items())},
            n_admissible_vectors_shared_by_more_than_one_scenario=sum(1 for v in scen_by_vec.values() if len(v) > 1),
            scenario_tick_groups_with_identical_admissible_input=shared_groups,
        ),
        repeated_compliance_vector_frequency=dict(
            n_trajectories=len(cv_by_traj), n_distinct_compliance_vectors=len(cv_mult), max_repeat=max(cv_mult.values()),
            n_distinct_noncompliance_by_initial_room_profiles=len(room_profiles),
            note="generate_compliance_vector keys on (scenario NAME, seed), so the same seed gives DIFFERENT "
                 "vectors in S1/S2/S3; 'seed' is not a latent shared across scenarios.",
        ),
        shared_rollouts="The GLOBAL row at ticks 0/5/10 of one trajectory is the same heuristic reference trajectory "
                        "re-simulated (see target_generation_audit.structure for identical-outcome counts).",
        grouping_hierarchy=[
            "scenario template (occupant distribution)",
            "hazard-timing x compliance-probability cell",
            "trajectory = (scenario, seed): one hidden compliance vector + one heuristic reference trajectory",
            "decision state = (scenario, seed, decision_tick)",
            "candidate-action row",
        ],
        minimum_split_unit="trajectory (scenario, seed): every tick and every candidate row of a trajectory in "
                           "the same split. A random row split places the same state (2 to 56 candidate rows) and "
                           "the shared GLOBAL rollout in both train and test.",
        recommended_split_unit="seed index assigned identically in every scenario (seed k in the same split in S1, "
                               "S2, S3), plus explicit cell / template hold-outs for generalization tests.",
    )


# ===========================================================================
# Sections 6/7: split design.
# ===========================================================================

def split_design(records: list, sec_per_rollout: float) -> dict:
    cells = defaultdict(lambda: dict(n_states=0, n_tuples=0, scenarios=set(), distributions=set()))
    for r in records:
        th = scenario_t_h(r["scenario"])
        c = cells[f"t_h={th if th is not None else 'none'}|p={COMPLIANCE_PROBABILITY}"]
        c["n_states"] += 1
        c["n_tuples"] += len(r["candidates"]) + len(r["pairwise_results"])
        c["scenarios"].add(r["scenario"])
        c["distributions"].add(json.dumps(SCENARIOS[r["scenario"]]["distribution"], sort_keys=True))
    existing = {k: dict(n_states=v["n_states"], n_tuples=v["n_tuples"], scenarios=sorted(v["scenarios"]),
                        distributions=sorted(v["distributions"])) for k, v in sorted(cells.items())}
    timing_values = ["none", "3", "4", "5", "6", "7", "8", "9", "10"]
    p_values = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
    grid = {f"t_h={th}|p={p}": existing.get(f"t_h={th}|p={p}", {}).get("n_states", 0)
            for th in timing_values for p in p_values}

    T_train, T_val, T_test, T_ext = ["none", "3", "5", "7", "9"], ["6"], ["4", "8"], ["10"]
    P_train, P_val, P_test, P_ext = [0.5, 0.7, 0.9], [0.8], [0.6, 0.85], [0.95]
    role = {}
    for th in T_train + T_val + T_test + T_ext:
        for p in P_train + P_val + P_test + P_ext:
            if th in T_train and p in P_train:
                r = "TRAIN"
            elif (th in T_val and p in P_train + P_val) or (th in T_train and p in P_val):
                r = "VALIDATION"
            elif th in T_test and p in P_test:
                r = "TEST_PRIMARY_JOINT"
            elif th in T_test and p in P_train:
                r = "TEST_DIAG_HELDOUT_TIMING"
            elif th in T_train and p in P_test:
                r = "TEST_DIAG_HELDOUT_COMPLIANCE"
            elif th in T_ext or p in P_ext:
                r = "TEST_DIAG_EXTRAPOLATION"
            else:
                r = "UNUSED_BUFFER"
            role[f"t_h={th}|p={p}"] = r
    seeds = dict(TRAIN=30, VALIDATION=15, TEST_PRIMARY_JOINT=30, TEST_DIAG_HELDOUT_TIMING=20,
                 TEST_DIAG_HELDOUT_COMPLIANCE=20, TEST_DIAG_EXTRAPOLATION=15, UNUSED_BUFFER=0)
    n_dist, states_per_traj, cands_per_state = 2, 3, 40
    n_traj = Counter()
    n_cells = Counter()
    for r in role.values():
        n_traj[r] += seeds[r] * n_dist
        n_cells[r] += 1
    total_traj = sum(n_traj.values())
    total_rollouts = total_traj * states_per_traj * cands_per_state
    return dict(
        audit=AUDIT_MARKER,
        existing_data_cells=existing,
        existing_grid_state_counts=grid,
        existing_hazard_timings=["none (S1)", "7 (S2)", "5 (S3)"],
        existing_compliance_probabilities=[COMPLIANCE_PROBABILITY],
        n_existing_nonempty_cells=sum(1 for v in grid.values() if v > 0),
        n_grid_cells_absent=sum(1 for v in grid.values() if v == 0),
        confounding=dict(
            timing_is_confounded_one_to_one_with_scenario=True,
            S3_timing_is_also_confounded_with_occupant_distribution=True,
            S1_and_S2_share_occupant_distribution=True,
            compliance_axis_has_a_single_value=True,
        ),
        joint_timing_x_compliance_split_possible_with_existing_data=False,
        heldout_timing_split_possible_with_existing_data=False,
        heldout_compliance_split_possible_with_existing_data=False,
        strongest_valid_split_with_existing_data=dict(
            description="In-distribution only: trajectory-grouped split by seed index, identical in S1/S2/S3 -- "
                        "seeds 10000-10029 TRAIN, 10030-10039 VALIDATION, 10040-10049 TEST. Leave-one-scenario-out "
                        "exists only as a diagnostic and cannot be read as a timing or compliance hold-out (it moves "
                        "timing and/or distribution at once, with one value per axis).",
            matrix={"t_h=none|p=0.9 (S1)": "TRAIN 30 / VAL 10 / TEST 10 seeds",
                    "t_h=7|p=0.9 (S2)": "TRAIN 30 / VAL 10 / TEST 10 seeds",
                    "t_h=5|p=0.9 (S3)": "TRAIN 30 / VAL 10 / TEST 10 seeds",
                    "every other timing x compliance cell": "ABSENT"},
            n_test_trajectories=30, n_test_states=90,
            cannot_test=["held-out hazard timing", "held-out compliance probability", "joint timing x compliance",
                         "clean template zero-shot (only 2 distributions, each tied to specific timings)"],
        ),
        proposed_new_data_design=dict(
            status="PROPOSAL ONLY -- nothing generated by this audit",
            axes=dict(hazard_timing=dict(train=T_train, validation=T_val, test=T_test, extrapolation_diagnostic=T_ext),
                      compliance_p=dict(train=P_train, validation=P_val, test=P_test, extrapolation_diagnostic=P_ext),
                      occupant_distributions=["symmetric 10/10/10/10 (S1/S2)", "asymmetric 5/5/15/15 (S3)"],
                      template_zero_shot="train on one distribution, test on the other: diagnostic only"),
            cell_roles=role, n_cells_by_role=dict(n_cells),
            seeds_per_cell_per_distribution=seeds,
            assumptions=dict(occupant_distributions=n_dist, decision_states_per_trajectory=states_per_traj,
                             candidates_per_state=cands_per_state),
            n_trajectories_by_role=dict(n_traj), n_trajectories_total=total_traj,
            n_candidate_rollouts_total=total_rollouts,
            measured_seconds_per_rollout=sec_per_rollout,
            estimated_single_core_hours=round(total_rollouts * sec_per_rollout / 3600.0, 2),
            requirements=[
                "Seeds disjoint across TRAIN / VALIDATION / TEST as well as cells.",
                "The observable feature vector computed and STORED at generation time (the existing dataset stores none).",
                "One fixed candidate set, one candidate semantics, one target definition (target audit T1-T5).",
                "t_h, p, distribution, seed, trajectory id stored per row as SPLIT METADATA only, never as features.",
                "A new compliance-vector key namespace so no vector collides with the existing evaluation set.",
                "INPUT DIVERSITY, not just more seeds: in the existing data 50 seeds of one scenario/tick give the same "
                "observable state, so seeds alone add almost no distinct inputs. Add occupant distributions beyond "
                "the two templates and reach decision states through varied prior action histories (e.g. random or "
                "epsilon-perturbed pre-decision actions), not only the heuristic reference trajectory.",
            ],
        ),
    )


# ===========================================================================
# Section 8: identifiability.
# ===========================================================================

def _target_values(row: dict, state: dict) -> dict:
    m = row["res"]["metrics"]
    return dict(
        T90=float(m["T90"]) if m["T90_defined"] else float(MAX_STEPS),
        T100=float(m["T100"]),
        total_unmet_post=float(row["res"]["total_unmet_demand"] - state["prior_total_unmet"]),
        hazard_unmet_post=float(row["res"]["hazard_edge_unmet_demand"] - state["prior_hazard_unmet"]),
    )


def build_modeling_table(rows: list, states: dict, adm: list) -> list:
    table = []
    for r in rows:
        s = states[r["state"]]
        table.append(dict(
            state=r["state"], scenario=r["scenario"], seed=r["seed"], tick=r["tick"],
            action_key=r["action_key"], family=r["family"], x=admissible_vector(s, adm),
            y=_target_values(r, s), t_h=s["t_h"], cv=s["hidden"]["compliance_vector_sha"],
            n_nc=s["hidden"]["n_noncompliant_active"], nc_room=s["hidden"]["noncompliant_by_initial_room"],
        ))
    return table


def _within_group_share(table: list, keyfn, target: str) -> dict:
    groups = defaultdict(list)
    for t in table:
        groups[keyfn(t)].append(t["y"][target])
    y = np.array([t["y"][target] for t in table])
    sst = float(((y - y.mean()) ** 2).sum())
    ssw = float(sum(((np.array(v) - np.mean(v)) ** 2).sum() for v in groups.values()))
    multi = [v for v in groups.values() if len(v) > 1]
    return dict(n_groups=len(groups), n_groups_with_repeats=len(multi),
                n_groups_with_conflicting_targets=sum(1 for v in multi if len(set(v)) > 1),
                total_sum_sq=round(sst, 3), within_group_sum_sq=round(ssw, 3),
                irreducible_share=round(ssw / sst, 4) if sst > 0 else 0.0)


def target_identifiability(table: list, records: list, states: dict, adm: list) -> dict:
    # The full compliance vector is unique per trajectory (it keys on scenario name + seed), so it also
    # identifies the scenario and t_h. Compliance is therefore probed with NON-identifying summaries first.
    keys = dict(
        admissible_features_plus_action=lambda t: (tuple(t["x"].tolist()), t["action_key"]),
        plus_n_noncompliant_active_only=lambda t: (tuple(t["x"].tolist()), t["action_key"], t["n_nc"]),
        plus_true_t_h=lambda t: (tuple(t["x"].tolist()), t["action_key"], t["t_h"]),
        plus_t_h_and_n_noncompliant_active=lambda t: (tuple(t["x"].tolist()), t["action_key"], t["t_h"], t["n_nc"]),
        plus_t_h_and_noncompliance_by_room=lambda t: (tuple(t["x"].tolist()), t["action_key"], t["t_h"], t["nc_room"]),
        plus_full_compliance_vector_eq_trajectory_id=lambda t: (tuple(t["x"].tolist()), t["action_key"], t["t_h"],
                                                                t["scenario"], t["cv"]),
    )
    decomposition = {tgt: {name: _within_group_share(table, fn, tgt) for name, fn in keys.items()} for tgt in TARGETS}

    s1s2 = defaultdict(lambda: defaultdict(list))
    for t in table:
        if t["tick"] == 0 and t["scenario"] in ("S1", "S2") and t["family"] != "PAIRWISE":
            s1s2[t["action_key"]][t["scenario"]].append(t["y"]["T100"])
    s1s2_summary = {a: {sc: dict(n=len(v), mean=round(float(np.mean(v)), 3), min=min(v), max=max(v))
                        for sc, v in d.items()} for a, d in s1s2.items()}
    x_s1 = {tuple(admissible_vector(states[("S1", sd, 0)], adm).tolist()) for sd in EVAL_SEED_SET}
    x_s2 = {tuple(admissible_vector(states[("S2", sd, 0)], adm).tolist()) for sd in EVAL_SEED_SET}

    ratios, n_pair, n_pair_eq, cmp = [], 0, 0, Counter()
    for r in records:
        outs = [outcome_vector(v) for v in r["candidates"].values()] + [outcome_vector(v) for v in r["pairwise_results"].values()]
        ratios.append(len(set(outs)) / len(outs))
        g = outcome_vector(r["candidates"]["GLOBAL"])
        for v in r["pairwise_results"].values():
            o = outcome_vector(v)
            n_pair += 1
            n_pair_eq += int(o == g)
            cmp["PAIRWISE_" + ("worse_than_GLOBAL" if o > g else ("tie" if o == g else "better_than_GLOBAL"))] += 1
        ni = outcome_vector(r["candidates"]["NO_INTERVENTION"])
        cmp["NO_INTERVENTION_" + ("worse_than_GLOBAL" if ni > g else ("tie" if ni == g else "better_than_GLOBAL"))] += 1

    corr = {}
    grp = defaultdict(list)
    for t in table:
        if t["family"] == "GLOBAL" and t["tick"] == 0:
            grp[(t["scenario"], t["action_key"])].append(t)
    for (sc, ak), v in sorted(grp.items()):
        nc = np.array([t["n_nc"] for t in v], dtype=float)
        for tgt in ("T100", "total_unmet_post", "hazard_unmet_post"):
            yy = np.array([t["y"][tgt] for t in v])
            if len(v) >= 10 and np.std(nc) > 0 and np.std(yy) > 0:
                corr[f"{sc}|{ak}|{tgt}"] = round(float(np.corrcoef(nc, yy)[0, 1]), 3)

    return dict(
        audit=AUDIT_MARKER,
        simulator_is_deterministic_given="(occupant distribution, t_h, compliance vector, action sequence); no other RNG",
        sources_of_apparent_stochasticity=["hidden per-occupant compliance vector",
                                           "hidden future hazard onset t_h (before onset)"],
        variance_decomposition=decomposition,
        interpretation=("irreducible_share = within-group sum of squares / total sum of squares when rows are grouped "
                        "by the listed key: the share of target variance that NO function of that key can explain "
                        "on this dataset. 'admissible_features_plus_action' is the proposed model's information set; "
                        "adding the hidden t_h and compliance vector should drive it to 0 because the simulator is "
                        "deterministic."),
        S1_vs_S2_tick0=dict(
            identical_admissible_observation_across_S1_and_S2=bool(x_s1 == x_s2 and len(x_s1) == 1),
            T100_by_action_and_scenario=s1s2_summary,
            meaning="At t=0 every S1 and every S2 state is the same admissible input; only the unobservable future "
                    "hazard (and hidden compliance) separates their outcomes.",
        ),
        multiple_actions_identical_outcomes=dict(
            mean_distinct_outcome_share_per_state=round(float(np.mean(ratios)), 4),
            n_pairwise_rows=n_pair, n_pairwise_rows_identical_to_GLOBAL_outcome=n_pair_eq,
            outcome_vs_GLOBAL_counts=dict(cmp),
        ),
        hidden_noncompliance_vs_outcome_correlation_tick0_GLOBAL=corr,
        candidate_target_forms=dict(
            exact_per_row_outcome="NOT identifiable wherever identical observations map to different outcomes.",
            expected_outcome="Identifiable as a conditional mean over the TRAINING distribution of hidden compliance "
                             "and hazard timing, and only as transferable as that distribution. With one compliance "
                             "level and one timing per template, the expectation here is over seeds only.",
            prediction_interval_or_distribution="Estimable: identical observations repeat many times, so conditional "
                                                "spread is measurable.",
            probability_of_harmful_outcome="Estimable as a conditional probability; base rates in outcome_vs_GLOBAL_counts.",
            decision="Deferred by instruction.",
        ),
    )


# ===========================================================================
# Section 9: non-learning baselines.
# ===========================================================================

def _metrics(pred: np.ndarray, y: np.ndarray) -> dict:
    e = pred - y
    return dict(MAE=round(float(np.mean(np.abs(e))), 4), RMSE=round(float(np.sqrt(np.mean(e ** 2))), 4))


BASELINES = ["global_mean", "mean_by_action_family", "mean_by_action_key",
             "mean_by_scenario_tick_action_DIAGNOSTIC_uses_scenario_id", "exact_state_lookup",
             "nearest_state_analogue_1NN"]


def baseline_predictability(table: list) -> dict:
    X = np.vstack([t["x"] for t in table])
    sd = X.std(0)
    sd[sd == 0] = 1.0
    Xs = (X - X.mean(0)) / sd
    results = {}
    splits = [("trajectory_grouped_5fold", lambda t: EVAL_SEED_SET.index(t["seed"]) % N_FOLDS, N_FOLDS),
              ("leave_one_scenario_out_DIAGNOSTIC", lambda t: ["S1", "S2", "S3"].index(t["scenario"]), 3)]
    for split_name, fold_fn, n_f in splits:
        folds = np.array([fold_fn(t) for t in table])
        out = {}
        for tgt in TARGETS:
            y = np.array([t["y"][tgt] for t in table])
            preds = {k: np.zeros(len(table)) for k in BASELINES}
            hits = 0
            for f in range(n_f):
                tr, te = np.nonzero(folds != f)[0], np.nonzero(folds == f)[0]
                gmean = y[tr].mean()
                fam, akey, sta, exact, aidx = (defaultdict(list) for _ in range(5))
                for i in tr:
                    t = table[i]
                    fam[t["family"]].append(y[i])
                    akey[t["action_key"]].append(y[i])
                    sta[(t["scenario"], t["tick"], t["action_key"])].append(y[i])
                    exact[(tuple(t["x"].tolist()), t["action_key"])].append(y[i])
                    aidx[t["action_key"]].append(i)
                aidx = {k: np.array(v) for k, v in aidx.items()}
                for i in te:
                    t = table[i]
                    fmean = np.mean(fam[t["family"]]) if t["family"] in fam else gmean
                    amean = np.mean(akey[t["action_key"]]) if t["action_key"] in akey else fmean
                    preds["global_mean"][i] = gmean
                    preds["mean_by_action_family"][i] = fmean
                    preds["mean_by_action_key"][i] = amean
                    k3 = (t["scenario"], t["tick"], t["action_key"])
                    preds["mean_by_scenario_tick_action_DIAGNOSTIC_uses_scenario_id"][i] = np.mean(sta[k3]) if k3 in sta else amean
                    ke = (tuple(t["x"].tolist()), t["action_key"])
                    if ke in exact:
                        preds["exact_state_lookup"][i] = np.mean(exact[ke])
                        hits += 1
                    else:
                        preds["exact_state_lookup"][i] = amean
                    cand = aidx.get(t["action_key"])
                    if cand is None:
                        preds["nearest_state_analogue_1NN"][i] = amean
                    else:
                        d = np.linalg.norm(Xs[cand] - Xs[i], axis=1)
                        preds["nearest_state_analogue_1NN"][i] = y[cand[d <= d.min() + 1e-12]].mean()
            out[tgt] = dict(target_std=round(float(y.std()), 4), n_rows=len(y),
                            exact_lookup_hit_rate=round(hits / len(y), 4),
                            **{k: _metrics(v, y) for k, v in preds.items()})
        results[split_name] = out
    return dict(
        note="NOT the model experiment. Folds group whole trajectories (seed index mod 5, identical in every "
             "scenario). Inputs = the admissible state features + action key; 1-NN ties averaged; lookups fall back "
             "to the action-key mean. Row-weighted: states with 56 candidate rows dominate.",
        results=results,
    )


# ===========================================================================
# Sections 10/11: selector definition + preregistered criteria (definitions only).
# ===========================================================================

def selector_definition() -> dict:
    return dict(
        audit=AUDIT_MARKER,
        status="DEFINITION ONLY -- not implemented",
        notation=dict(
            s="decision state; x(s) = admissible feature vector, computed before any candidate is applied",
            C_s="fixed candidate set at s; always contains the default d(s)",
            d_s="default action: the frozen deterministic controller. GLOBAL_DEADBAND applied globally, or "
                "observable_robust_differential.py (MD5 98b570ba7d7d451c82acf919113db2e5) if the Discrete(211) "
                "'heuristic globally + override' semantics is adopted for C_s",
            y_hat="predicted vector y_hat(s,a) = [T90, T100, U, H]; U = post-decision total unmet, H = post-decision "
                  "hazard-edge unmet (after target fixes T1/T2)",
            q_hi="per-component upper predictive quantile (or conformal upper bound) at a preregistered level",
            delta="practical-equivalence tolerances (d90, d100, dU, dH), fixed before training; T90/T100 are integer "
                  "ticks, so d90 = d100 = 0.5 tick is the natural choice",
        ),
        established_ordering=dict(order=["T90 (undefined -> +inf)", "T100", "total_unmet_demand",
                                         "hazard_edge_unmet_demand"],
                                  source="audit_differential_targeting.sort_key_full_ranking", preserved=True),
        why_literal_lexicographic_argmin_is_inappropriate_for_regression=(
            "Continuous predictions essentially never tie exactly on T90, so a literal lexicographic argmin reduces to "
            "argmin predicted T90 and silently ignores T100, U and H -- an effective CHANGE of the established "
            "ordering. The order is therefore kept as a TOLERANCE-lexicographic order: criterion k+1 is consulted "
            "only among candidates within delta_k of the best on criterion k."),
        steps=[
            dict(step=1, name="safety constraint (pessimistic non-regression vs the default)",
                 rule="A(s) = { a in C_s : q_hi_k(s,a) <= y_hat_k(s,d) + delta_k for every component k }. d(s) is "
                      "always in A(s). This is a constraint, not a reordering: the objective in steps 2-3 keeps the "
                      "established order."),
            dict(step=2, name="primary objective", rule="Within A(s): keep a with y_hat_T90(s,a) <= min_A y_hat_T90 + d90."),
            dict(step=3, name="secondary objectives (tolerance-lexicographic, established order)",
                 rule="Then T100 within d100 of the best, then U within dU, then minimal H. Remaining ties: prefer "
                      "d(s), then the lowest candidate id."),
            dict(step=4, name="strict-improvement requirement",
                 rule="Output a* != d(s) only if a* beats d(s) by more than delta_k on the FIRST criterion where they "
                      "differ by more than the tolerance; otherwise output d(s)."),
            dict(step=5, name="fallback",
                 rule="Output d(s) if: (i) x(s) is outside training support (nearest-training-state distance above the "
                      "99th percentile of within-training nearest-neighbour distances); (ii) any feature is missing or "
                      "invalid (e.g. trend features at t=0); (iii) the model or quantile head fails; (iv) A(s) = {d(s)}."),
            dict(step=6, name="uncertainty handling",
                 rule="Upper bounds in step 1 only; point predictions in steps 2-4. Quantile level and conformal "
                      "calibration are fit on VALIDATION cells only and frozen in the preregistration."),
        ],
        properties=["Deterministic given model outputs.",
                    "Never predicted-worse than d(s) at the pessimistic bound; realized safety is measured separately.",
                    "Reduces to the frozen default when the model is uninformative."],
        open_decisions_for_the_user=[
            "Default d(s): GLOBAL heuristic vs the frozen robust controller (follows from the candidate semantics chosen).",
            "Quantile level for the safety bound.",
            "delta values for U and H (unit: occupant-ticks).",
        ],
    )


def preregistered_success_criteria() -> dict:
    return dict(
        audit=AUDIT_MARKER,
        status="PREREGISTRATION DRAFT -- freeze (record the file hash) before any training",
        primary_test_family="D_joint_timing_x_compliance on the new-data design (held-out t_h AND held-out p, "
                            "disjoint seeds). With existing data only family A is possible; any such result must be "
                            "labeled in-distribution.",
        statistics="95% cluster-bootstrap CIs resampling TRAJECTORIES (not rows), 2000 resamples; every criterion "
                   "reported per test family, never pooled.",
        prediction_success=dict(
            question="Does the model predict action outcomes better than simple baselines?",
            baselines=["global mean", "mean by action key", "exact-state lookup", "1-NN state analogue"],
            criterion="MAE on T100 and on post-decision total unmet each below the BEST non-learning baseline with "
                      "the paired-difference CI excluding 0; T90 and hazard-edge unmet reported, not gated; "
                      "upper-quantile coverage within +/-5 points of nominal.",
            not_success="Beating only the global mean.",
        ),
        recommendation_success=dict(
            question="Does the selector choose actions whose realized outcomes are close to the oracle's?",
            metrics=["outcome-equivalence rate (realized vector == oracle-best vector over the SAME candidate set)",
                     "regret in T100 and post-decision unmet vs the oracle-best candidate",
                     "oracle-advantage capture on states where the oracle strictly beats d(s)"],
            criterion="Outcome-equivalence rate above d(s)'s own rate with CI excluding 0 on states where a strict "
                      "improvement exists. Exact oracle-action match is diagnostic only.",
        ),
        safety_success=dict(
            question="Does the AI-assisted policy avoid harmful recommendations?",
            harmful_definition="Realized outcome strictly worse than d(s) under the established 4-criterion order "
                               "(one-step counterfactual from the same state).",
            criterion="Harmful-recommendation rate upper 95% CI bound <= 1% in every test family, and no "
                      "recommendation raises post-decision hazard-edge unmet by more than dH.",
        ),
        generalization_success=dict(
            families={"A_in_distribution_heldout_seeds": "diagnostic",
                      "B_heldout_hazard_timing": "diagnostic",
                      "C_heldout_compliance_probability": "diagnostic",
                      "D_joint_timing_x_compliance": "PRIMARY",
                      "E_template_zero_shot": "diagnostic, expected hard, never a gate"},
            criterion="Prediction, recommendation and safety criteria all met on D; A-C and E reported with the same metrics.",
        ),
        closed_loop_success=dict(
            question="Executed at every decision, is evacuation at least as good and as safe as the deterministic baseline?",
            protocol="Full-episode rollouts on the primary test cells with common random numbers (same seeds / "
                     "compliance vectors) for the AI-assisted and d(s)-only policies. A SEPARATE run: training labels "
                     "are one-step counterfactuals with heuristic continuation (target audit T6).",
            criterion="Non-inferiority: paired T100 difference upper 95% CI bound <= 0.5 tick; paired hazard-edge "
                      "unmet difference upper bound <= dH; no episode with lower completion rate than d(s).",
        ),
        explicitly_not_success_criteria=[
            "Beating the heuristic is NOT required: recommendation quality is measured against the oracle over the "
            "same candidate set, safety against d(s).",
            "Exact oracle-action match is NOT required (diagnostic only).",
            "Any result obtained with a feature classified D, E, F or G does not count.",
        ],
    )


# ===========================================================================
# Section 12: decision.
# ===========================================================================

def final_decision(structure: dict, split: dict, temporal_leak: dict, traj: dict | None = None) -> dict:
    blockers = []
    if traj is not None:
        rs = traj["repeated_state_frequency"]
        blockers.append(f"C: the {rs['n_states']} decision states collapse to {rs['n_distinct_admissible_feature_vectors']} "
                        f"distinct admissible inputs ({rs['n_distinct_27D_observations']} distinct raw 27-D "
                        "observations): per scenario and tick, every seed yields essentially the same observable "
                        "state. The effective sample size for learning a state -> outcome map is about a dozen "
                        "inputs, not 9,450 rows.")
    if not split["joint_timing_x_compliance_split_possible_with_existing_data"]:
        blockers.append(f"C: the jointly held-out hazard-timing x compliance evaluation cannot be built. The dataset "
                        f"occupies {split['n_existing_nonempty_cells']} cells, has one compliance level (0.9), and "
                        "hazard timing is confounded one-to-one with scenario.")
    if not structure["stores_observation_or_feature_vector"]:
        blockers.append("C: no feature vector is stored. Features can be regenerated by deterministic replay (verified "
                        "here), but new data must store them at generation time.")
    blockers.append("B: the candidate set is not comparable across states (only 2 candidates on 250/450 states; only "
                    "the heuristic's own global action), and its semantics differ from the proposed expansion "
                    "machinery and the fallback controller (T4, T5).")
    blockers.append("B: stored unmet targets include action-independent pre-decision history (T1); the hazard-edge "
                    "label encodes whether a hazard is scheduled (T2).")
    if temporal_leak["leak_confirmed"]:
        blockers.append("B: observation indices 29-30 leak the future onset (excluded); Q_A/Q_B_at_H read hidden labels "
                        "and track hidden compliance (excluded from the admissible set).")
    return dict(
        classification="C",
        label="REQUIRES NEW SIMULATION DATA",
        also_requires="B-type cleanup (target definitions, candidate semantics, feature list), applied to the NEW data.",
        not_classified_as=dict(
            A="Blocked by the items listed.",
            B="Cleanup alone cannot create the missing timing x compliance cells.",
            D="No new observable is REQUIRED to run the preregistered experiment; better observables would raise "
              "the ceiling, not unblock the experiment.",
            E="Expected-outcome / interval / harm-probability targets are identifiable from the admissible features; "
              "only exact per-row prediction is not, which is a target choice, not a fatal flaw.",
        ),
        blockers=blockers,
        must_happen_first=[
            "1. Freeze ONE candidate semantics and ONE fixed candidate set per state (T4, T5).",
            "2. Freeze target definitions: post-decision unmet (T1) and a uniform hazard-edge definition (T2).",
            "3. Freeze the admissible feature list from exact_feature_manifest.json (no D/E/F/G features, no "
            "Q_A/Q_B_at_H, no indices 27-30); decide whether trend features are added and how t=0 is handled.",
            "4. Generate NEW data over the timing x compliance x distribution grid in split_design.json, storing "
            "features, t_h, p, seed and trajectory id per row, with disjoint seeds per split.",
            "5. Re-run this audit's checks on the new data (replay determinism, action independence, label "
            "reproduction, identifiability) before any training.",
            "6. Freeze selector_definition.json and preregistered_success_criteria.json and record their hashes.",
        ],
    )


# ===========================================================================
# Markdown report.
# ===========================================================================

def render_report(ctx: dict) -> str:
    st, fl, tg, td, sp, ti, bl, dec = (ctx[k] for k in ("structure", "leakage", "target", "traj", "split",
                                                         "ident", "baselines", "decision"))
    L = []
    a = L.append
    a(f"<!-- {AUDIT_MARKER} -->")
    a("# Pre-Training Data / Feature / Target Integrity Audit\n")
    a("Read-only audit. No model trained, no selector implemented, no simulator / observation.py / controller / "
      "frozen-RL / dataset file modified (SHA-256 checked before and after; Section 13).\n")
    a(f"## Final decision: **{dec['classification']}. {dec['label']}**\n")
    for b in dec["blockers"]:
        a(f"- {b}")
    a(f"\nAlso required: {dec['also_requires']}\n")

    a("## 1. The \"9,450 tuples\" dataset\n")
    a(f"- File: `{st['file']}` (SHA-256 `{st['file_sha256']}`).")
    a(f"- Exact row count: **{st['n_total_state_action_outcome_tuples']}** (state, action, outcome) tuples = "
      f"{st['n_candidate_tuples_GLOBAL_and_NO_INTERVENTION']} GLOBAL/NO_INTERVENTION + {st['n_pairwise_tuples']} pairwise.")
    a(f"- Unique decision states: **{st['n_unique_states']}**. Unique action keys: {st['n_unique_action_keys']} "
      f"(GLOBAL variants {st['global_action_variants']}).")
    a(f"- Record fields: `{', '.join(st['record_fields'])}`. **No observation or feature vector is stored.**")
    a(f"- Scenarios {st['scenarios']}; seeds {st['seeds']['n_unique']} ({st['seeds']['min']}-{st['seeds']['max']}); "
      f"decision ticks {st['decision_ticks']}.")
    a(f"- Hazard timings {st['hazard_timings_represented']} (one per scenario). Compliance p "
      f"{st['compliance_probabilities_represented']} only (not stored; module default).")
    a(f"- Every tuple has a real simulator outcome: {st['every_tuple_has_simulator_outcome']}. Exact duplicate "
      f"(state, action) rows: {st['n_exact_duplicate_state_action_rows']}. T90 undefined rows: "
      f"{st['n_rows_T90_undefined']}. T100 censored rows: {st['n_rows_T100_censored']}.")
    a(f"- {st['composition_of_9450']}\n")
    a("| scenario/tick | states | tuples | non-empty zones | pairwise rows per state |")
    a("|---|---|---|---|---|")
    for k, v in st["per_scenario_tick"].items():
        a(f"| {k} | {v['n_states']} | {v['n_tuples']} | {v['nonempty_zones']} | {v['pairwise_per_state']} |")
    lr = tg["label_reproduction"]
    a(f"\nReplay: {ctx['replay']['n_states_replayed']} states regenerated deterministically "
      f"({ctx['replay']['n_replay_mismatches']} mismatches in heuristic action / non-empty zones). Label "
      f"reproduction: {lr['n_rollouts_rerun']} rollouts re-run, {lr['n_label_mismatches']} label mismatches, "
      f"{lr['n_rollouts_not_starting_from_identical_state']} starting-state mismatches. Every row is a real rollout "
      "of the simulator; rows from one trajectory share the hidden compliance vector and (for GLOBAL) the same "
      "reference rollout (Section 5).\n")

    a("## 2. Feature-by-feature leakage audit\n")
    a("Classes: " + "; ".join(f"**{k}** {v}" for k, v in CLASS_LABELS.items()) + ".\n")
    a("| feature | class | future state | hidden compliance | oracle | candidate action |")
    a("|---|---|---|---|---|---|")
    for m in ctx["manifest"]:
        a(f"| `{m['name']}` | {m['classification']} | {m['depends_on_future_state']} | "
          f"{m['depends_on_hidden_compliance']} | {m['depends_on_oracle_information']} | {m['depends_on_candidate_action']} |")
    a("\nSource variable, computation and availability for every row are in `exact_feature_manifest.json`.\n")
    tl = fl["temporal_leakage"]
    a(f"**`time_since_hazard_onset_normalized` and `hazard_phase`: F (future-information leakage), confirmed.** "
      f"On all {tl['n_pre_onset_hazard_states']} pre-onset hazard states, index 29 recovers the exact future t_h "
      f"({tl['n_pre_onset_states_where_feature29_recovers_exact_future_t_h']}/{tl['n_pre_onset_hazard_states']}) and "
      f"hazard_phase is 0.5 ({tl['n_pre_onset_states_with_hazard_phase_0_5']}/{tl['n_pre_onset_hazard_states']}); all "
      f"{tl['n_no_hazard_states_with_sentinel']} no-hazard states carry the -2.0 sentinel. By contrast the 27-D "
      f"residual capacity is {tl['baseline_residual_cap_values_pre_onset']} on every pre-onset state, so it does not leak.\n")
    ql = fl["q_label_compliance_leak"]
    a(f"**`Q_A_at_H` / `Q_B_at_H`: D (hidden ground truth).** They count the hidden `target_exit` label. On the "
      f"{ql['n_states_prior_action_favor_B_with_nonempty_hub']} states whose previous action was FAVOR_B, Q_A_at_H "
      f"tracks the hidden number of non-compliant baseline-A hub occupants with r = {ql['pearson_r']} (partial r "
      f"controlling for hub size = {ql['partial_r_controlling_for_hub_size']}; exact equality in "
      f"{ql['n_states_Q_A_equals_hidden_noncompliant_baseline_A_count']} states because occupants in transit at the "
      "last decision keep older labels). The prior design kept them with a disclosure; this audit excludes them and "
      "uses the label-free `waiting_H_total` instead.\n")
    a(f"**`residual_cap_H_CE1_normalized`: C.** Value/timing check passed: {fl['residual_capacity']['passed']} (hazard "
      "update is step 1 of the tick, before the decision).\n")
    a(f"**Exact admissible v1 state feature list ({len(ctx['adm_names'])}):** " + ", ".join(f"`{n}`" for n in ctx["adm_names"]) + ".\n")
    a("Plus a candidate-action descriptor (not a state feature): per eligible zone {untouched, FAVOR_A, PARITY, "
      "FAVOR_B} + a mode flag.\n")
    a("**The prior \"17 + 2 = 21\" count is wrong.** The prior design's own index list (0-7, 10-26, plus 8-9, plus "
      "2 trend features) is 29 features, and \"delta-waiting per zone\" is a family of 9 scalars, not 1. The trend "
      "features exist in no code. They are listed in the manifest (A / C) but are NOT in the v1 set, and they are "
      f"undefined for the {ctx['outputs']['exact_feature_manifest.json']['proposed_trend_extension']['undefined_at_tick0_states']} t=0 states.\n")

    a("## 3. Action-independence\n")
    ai = fl["action_independence"]
    a(f"{ai['n_state_candidate_pairs_checked']} (state, candidate) pairs checked by recomputing the observation after "
      f"each candidate's relabel. As generated, no feature depends on the candidate. Computed after the relabel, only "
      f"these would change: {ai['features_that_change_if_computed_after_candidate_relabel']}. Both are already "
      "excluded. Nothing is removed from the admissible set on action-dependence grounds.\n")

    a("## 4. Target / label audit\n")
    for k, v in tg["checks"].items():
        a(f"- {k}: {v}")
    a("")
    for it in tg["contamination_and_definition_issues"]:
        extra = f" Evidence: {it['evidence']}." if it.get("evidence") else ""
        a(f"- **{it['id']} ({it['severity']})**: {it['issue']}{extra} Fix: {it['fix'] if it.get('fix') else 'disclose'}")
    s = tg["structure"]
    a(f"\nGLOBAL outcome identical at all 3 ticks in {s['n_trajectories_where_GLOBAL_outcome_identical_at_all_3_ticks']}/"
      f"{s['n_trajectories']} trajectories. NO_INTERVENTION == GLOBAL in {s['n_states_NO_INTERVENTION_outcome_equals_GLOBAL']}"
      f"/450 states ({s['n_tick0_states_NO_INTERVENTION_equals_GLOBAL']} at t=0). Pre-decision unmet averages "
      f"{s['prior_unmet_share_of_GLOBAL_total_unmet']['mean']:.1%} of the stored GLOBAL total unmet (max "
      f"{s['prior_unmet_share_of_GLOBAL_total_unmet']['max']:.1%}). S1 hazard-edge labels: {s['S1_hazard_edge_unmet_label_values']}; "
      f"S1 rollouts with real H_CE1 overflow but label 0: {lr['n_no_hazard_rollouts_with_realized_H_CE1_overflow_but_label_0']}.\n")

    a("## 5. Temporal / trajectory dependence\n")
    rs = td["repeated_state_frequency"]
    a(f"- Rows per trajectory (rows -> number of trajectories): {td['rows_per_trajectory']['histogram_rows_to_n_trajectories']}; "
      f"states per trajectory: {td['states_per_trajectory']}.")
    a(f"- Rows per seed: {td['rows_per_seed']}. Rows per scenario: {td['rows_per_scenario']}. Rows per tick: "
      f"{td['rows_per_decision_tick']}.")
    a(f"- Distinct 27-D observations: {rs['n_distinct_27D_observations']}/450. Distinct admissible vectors: "
      f"{rs['n_distinct_admissible_feature_vectors']}/450. Largest identical group: {rs['largest_identical_admissible_group']} "
      f"states. Per scenario/tick: {rs['distinct_admissible_vectors_per_scenario_tick']}. Vectors shared by more than one "
      f"scenario: {rs['n_admissible_vectors_shared_by_more_than_one_scenario']} "
      f"({'; '.join(rs['scenario_tick_groups_with_identical_admissible_input'])}).")
    cv = td["repeated_compliance_vector_frequency"]
    a(f"- Compliance vectors: {cv['n_distinct_compliance_vectors']} distinct over {cv['n_trajectories']} trajectories "
      f"(max repeat {cv['max_repeat']}); the same seed gives a different vector in each scenario.")
    a(f"- Grouping hierarchy: {' > '.join(td['grouping_hierarchy'])}.")
    a(f"- Minimum split unit: {td['minimum_split_unit']}")
    a(f"- Recommended: {td['recommended_split_unit']}\n")

    a("## 6. Hazard-timing x compliance grid\n")
    a("| cell | states | tuples | scenarios |")
    a("|---|---|---|---|")
    for k, v in sp["existing_data_cells"].items():
        a(f"| {k} | {v['n_states']} | {v['n_tuples']} | {v['scenarios']} |")
    a(f"\nOccupied: {sp['n_existing_nonempty_cells']} of {sp['n_existing_nonempty_cells'] + sp['n_grid_cells_absent']} "
      "cells of a 9-timing x 7-p grid. Held-out timing / held-out compliance / joint split possible: **no / no / no**.\n")
    a(f"Strongest valid split with existing data: {sp['strongest_valid_split_with_existing_data']['description']}\n")
    a("| cell | TRAIN | VALIDATION | TEST |")
    a("|---|---|---|---|")
    for k in sp["existing_data_cells"]:
        a(f"| {k} | seeds 10000-10029 | seeds 10030-10039 | seeds 10040-10049 |")
    a("| all other cells | absent | absent | absent |")
    nd = sp["proposed_new_data_design"]
    a(f"\nNew data needed (proposal, not generated): {nd['n_trajectories_total']} trajectories, about "
      f"{nd['n_candidate_rollouts_total']:,} candidate rollouts (~{nd['estimated_single_core_hours']} single-core hours at "
      f"the measured {nd['measured_seconds_per_rollout']} s/rollout). Cell roles:\n")
    roles = defaultdict(list)
    for k, v in nd["cell_roles"].items():
        roles[v].append(k)
    for r, ks in roles.items():
        a(f"- **{r}** ({len(ks)} cells): {', '.join(ks)}")

    a("\n## 7. Scenario / template leakage\n")
    a("S1 and S2 share topology and occupant distribution; S2 and S3 share the hazard location (H_CE1); every "
      "scenario shares the single topology. Tick-0 observations are fully determined by the distribution, so template "
      "identity is readable from waiting_R1..R4 at t=0, and a model CAN memorize it. Compliance vectors are unique per "
      "trajectory. Proposed tests: **A** in-distribution held-out seeds (diagnostic; the only one constructible now), "
      "**B** held-out hazard timing (diagnostic), **C** held-out compliance probability (diagnostic), **D** joint "
      "timing x compliance (**PRIMARY**), **E** template zero-shot (diagnostic, expected hard, never a gate).\n")

    a("## 8. Target identifiability\n")
    a("Irreducible share of target variance (within-group SS / total SS) for each information set:\n")
    dk = list(next(iter(ti["variance_decomposition"].values())).keys())
    a("| target | " + " | ".join(dk) + " |")
    a("|---|" + "---|" * len(dk))
    for t, d in ti["variance_decomposition"].items():
        a(f"| {t} | " + " | ".join(str(d[k]["irreducible_share"]) for k in dk) + " |")
    first = next(iter(ti["variance_decomposition"].values()))["admissible_features_plus_action"]
    a(f"\nThe 9,450 rows contain only **{first['n_groups']} distinct (admissible input, action) pairs**; "
      f"{first['n_groups_with_conflicting_targets']} of the repeated pairs carry conflicting targets for at least one "
      "target. The full compliance vector is unique per trajectory, so its column is equivalent to trajectory "
      "identity and reaching 0 there only confirms determinism.")
    a(f"\nS1 and S2 at t=0 are the same admissible input: {ti['S1_vs_S2_tick0']['identical_admissible_observation_across_S1_and_S2']}. "
      f"T100 by action and scenario: {ti['S1_vs_S2_tick0']['T100_by_action_and_scenario']}.")
    ma = ti["multiple_actions_identical_outcomes"]
    a(f"\nPairwise rows with exactly GLOBAL's outcome: {ma['n_pairwise_rows_identical_to_GLOBAL_outcome']}/{ma['n_pairwise_rows']}. "
      f"Outcome vs GLOBAL: {ma['outcome_vs_GLOBAL_counts']}. Correlation of hidden non-compliance with outcome inside "
      f"tick-0 GLOBAL groups: {ti['hidden_noncompliance_vs_outcome_correlation_tick0_GLOBAL']}.")
    a("\nConclusion: exact per-row outcomes are not identifiable from observable input. Conditional expectations, "
      "intervals and harm probabilities are, but only relative to the training distribution of hidden compliance and "
      "hazard timing, which here has a single compliance level and one timing per template. Target form not chosen.\n")

    a("## 9. Non-learning baselines (grouped by trajectory; not the experiment)\n")
    cols = ["global_mean", "mean_by_action_key", "mean_by_scenario_tick_action_DIAGNOSTIC_uses_scenario_id",
            "exact_state_lookup", "nearest_state_analogue_1NN"]
    for split_name, res in bl["results"].items():
        a(f"**{split_name}**, MAE:\n")
        a("| target | std | global mean | by action | by scenario+tick+action (diag.) | exact lookup | 1-NN |")
        a("|---|---|---|---|---|---|---|")
        for t, d in res.items():
            a(f"| {t} | {d['target_std']} | " + " | ".join(str(d[c]["MAE"]) for c in cols) + " |")
        a("")
    a("Do not over-interpret: rows are dominated by the 150 tick-0 states with 56 rollouts each. With only ~11 "
      "distinct inputs, exact lookup and 1-NN coincide. The gap between exact lookup on admissible inputs and the "
      "scenario-id lookup (which a real system does not have) is roughly what the hidden hazard timing is worth "
      "here: it is the S1-vs-S2 ambiguity at ticks 0 and 5. Under leave-one-scenario-out, no baseline beats the "
      "target's own standard deviation by much, i.e. nothing in the existing data supports cross-scenario transfer.\n")

    a("## 10. Selector (definition only)\n")
    a("See `selector_definition.json`. The established order (T90, T100, total unmet, hazard-edge unmet) is kept, "
      "applied tolerance-lexicographically, because a literal lexicographic argmin over continuous predictions "
      "collapses to T90 alone. Safety is a pessimistic-bound non-regression constraint against the default d(s); "
      "fallback to d(s) on out-of-support states, missing features, model failure, or no admissible improvement.\n")
    a("## 11. Preregistered success criteria\n")
    a("See `preregistered_success_criteria.json`: prediction, recommendation, safety, generalization (primary = joint "
      "timing x compliance) and closed-loop (separate run, non-inferiority). Beating the heuristic is not a criterion.\n")
    a("## 12. What must happen first\n")
    for m in dec["must_happen_first"]:
        a(f"- {m}")
    integ = ctx["integrity"]
    a("\n## 13. Integrity\n")
    a(f"- Protected files unchanged: **{integ['all_protected_unchanged']}** ({len(integ['before'])} files, SHA-256; "
      "list and hashes in `dataset_structure.json` -> `integrity`).")
    a(f"- Frozen controller MD5 matches: {integ['controller']['match']} (`{integ['controller']['actual']}`).")
    a(f"- RL checkpoints unchanged: {integ['checkpoints']['all_match']} ({integ['checkpoints']['n_checkpoints']} checked).")
    a(f"- Dataset SHA-256: `{st['file_sha256']}`.")
    a(f"- Data root read: `{DATA_ROOT}`.")
    return "\n".join(L) + "\n"


# ===========================================================================
# Main.
# ===========================================================================

def run(pairwise_stride: int = 1) -> dict:
    before = protected_hashes()
    t0 = time.perf_counter()
    records = load_records()
    rows = flatten_tuples(records)
    structure = dataset_structure(records, rows)
    print(f"[1] {structure['n_records_decision_states']} states, {structure['n_total_state_action_outcome_tuples']} tuples")

    states = replay_all_states(records)
    replay = replay_consistency(records, states)
    print(f"[2] replay mismatches: {replay['n_replay_mismatches']}")

    manifest = build_feature_manifest()
    adm = admissible_state_feature_names(manifest)
    leakage = dict(
        temporal_leakage=temporal_leakage_check(states),
        residual_capacity=residual_capacity_timing_check(states),
        q_label_compliance_leak=q_label_compliance_leak_check(states),
        action_independence=action_independence_check(records, states),
    )
    print(f"[3] temporal leak confirmed: {leakage['temporal_leakage']['leak_confirmed']}")

    label_repro = reproduce_labels(records, states, pairwise_stride=pairwise_stride)
    print(f"[4] label reproduction: {label_repro['n_rollouts_rerun']} rerun, {label_repro['n_label_mismatches']} mismatches")
    target = target_generation_audit(label_repro, target_structure_checks(records, states))

    traj = trajectory_dependence(records, rows, states, adm)
    split = split_design(records, label_repro["seconds_per_rollout"])
    table = build_modeling_table(rows, states, adm)
    ident = target_identifiability(table, records, states, adm)
    baselines = baseline_predictability(table)
    print("[5-9] dependence / split / identifiability / baselines done")

    after = protected_hashes()
    changed = [k for k in before if before[k] != after[k]]
    if changed:
        raise RuntimeError(f"PROTECTED FILES CHANGED DURING AUDIT: {changed}")
    integrity = dict(before=before, all_protected_unchanged=not changed,
                     controller=design.verify_controller_hash(), checkpoints=design.verify_checkpoint_hashes())
    decision = final_decision(structure, split, leakage["temporal_leakage"], traj)

    trend_examples = {}
    for sc in ("S2", "S3"):
        s5, s0 = states[(sc, EVAL_SEED_SET[0], 5)], states[(sc, EVAL_SEED_SET[0], 0)]
        trend_examples[f"{sc}_seed{EVAL_SEED_SET[0]}_tick5"] = dict(zip(TREND_NAMES, trend_vector(s5, s0).tolist()))

    ctx = dict(structure=structure, replay=replay, manifest=manifest, adm_names=adm, leakage=leakage,
               target=target, traj=traj, split=split, ident=ident, baselines=baselines, decision=decision,
               integrity=integrity)
    ctx["outputs"] = {
        "exact_feature_manifest.json": dict(
            audit=AUDIT_MARKER, classes=CLASS_LABELS, features=manifest,
            n_features_audited=len(manifest),
            admissible_v1_state_features=adm, n_admissible_v1_state_features=len(adm),
            candidate_action_descriptor=dict(
                description="NOT a state feature: the conditioning input naming the candidate. Per eligible zone (R1, "
                            "R2, C1, R3, R4, C2, H) one of {untouched, FAVOR_A, PARITY, FAVOR_B}, plus a mode flag "
                            "(global vs zone override). Depends on the candidate by design; contains no outcome information.",
                n_scalars_one_hot=7 * 4 + 1),
            proposed_trend_extension=dict(names=TREND_NAMES, n=len(TREND_NAMES), implemented=False,
                                          undefined_at_tick0_states=sum(1 for s in states.values() if s["tick"] == 0),
                                          example_values_computed_in_audit_only=trend_examples),
            prior_design_feature_count_claim=dict(
                claimed="17 leakage-clean + 2 trend = 21",
                actual_from_its_own_index_list="indices 0-7 (8) + 10-26 (17) + 8-9 (2) + 2 trend = 29, or 38 with "
                                               "9 per-zone deltas + 1 capacity delta",
                consistent=False),
            excluded=[m["name"] for m in manifest if m["classification"] in ("D", "E", "F", "G")],
        ),
        "feature_leakage_audit.json": dict(audit=AUDIT_MARKER, **leakage),
        "dataset_structure.json": dict(
            audit=AUDIT_MARKER, **structure, replay=replay, integrity=integrity,
            other_candidate_data_NOT_part_of_9450=dict(
                file=os.path.relpath(TIMING_MATRIX_PATH, DATA_ROOT).replace("\\", "/"),
                n_records=80, t_h=[3, 6, 7, 10], compliance_p=0.75, decision_tick=0, seeds="90000-90019",
                machinery="evaluate_differential_rl._oracle_candidate_continuation (heuristic globally, then "
                          "override; prior unmet reset to 0)",
                compatible_with_9450=False,
                note="All 80 are tick-0 states of one symmetric distribution, so the 27-D observation is the same "
                     "vector in every record; timing and compliance are invisible in its inputs.")),
        "target_generation_audit.json": target,
        "trajectory_dependence.json": traj,
        "split_design.json": dict(**split, final_decision=decision),
        "target_identifiability.json": dict(**ident, baseline_target_predictability=baselines),
        "selector_definition.json": selector_definition(),
        "preregistered_success_criteria.json": preregistered_success_criteria(),
    }
    ctx["outputs"]["PRE_TRAINING_DATA_INTEGRITY_AUDIT.md"] = render_report(ctx)
    ctx["seconds"] = round(time.perf_counter() - t0, 1)
    return ctx


def write_outputs(ctx: dict, out_dir: str = OUT_DIR) -> list:
    created: set = set()
    for name in OUTPUT_FILES:
        _safe_write(name, ctx["outputs"][name], created, out_dir=out_dir)
    return sorted(created)


if __name__ == "__main__":
    ctx = run(pairwise_stride=int(os.environ.get("PAIRWISE_STRIDE", "1")))
    written = write_outputs(ctx)
    d = ctx["decision"]
    print(f"\nDECISION: {d['classification']}. {d['label']}")
    print(f"admissible v1 features ({len(ctx['adm_names'])}): {ctx['adm_names']}")
    print(f"protected files unchanged: {ctx['integrity']['all_protected_unchanged']}")
    print(f"wrote {len(written)} files to {OUT_DIR} in {ctx['seconds']}s")
