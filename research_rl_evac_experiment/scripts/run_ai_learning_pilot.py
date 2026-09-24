"""PHASE 1 ONLY: generate the 240-trajectory AI-learning pilot.

Pilot-only code, deliberately separate from any future full-dataset
generator. Everything it does is read from the frozen specification v2:
  - pilot grid, seeds, compliance keys, RNG draw order, epsilon:
      dataset_design.json -> phases.PHASE_1_PILOT.exact_specification
  - candidate set: candidate_action_set.json (BASE2_298, unchanged)
  - targets: target_definitions.json (v2, D1)
  - continuation / pre-decision default: scripts/ai_learning_label_free_default.py (D2)

Per decision state:  pre-decision state (captured before the tick's action)
  -> exactly one BASE2_298 candidate (deepcopy of that identical state)
  -> label-free default at every later decision -> future targets.

The hidden-label heuristic is never imported or called here. The 26 input
features are built directly from simulator counts; waiting_H_total is the
hub queue LENGTH, never Q_A + Q_B. Trains nothing; writes only the pilot
directory. Does not modify the simulator, observation.py, the RL branch or
the old 9,450-row dataset.
"""
from __future__ import annotations

import copy
import csv
import gzip
import hashlib
import io
import json
import os
import sys
import time
from collections import Counter

import numpy as np

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)
import ai_learning_label_free_default as lfd  # noqa: E402  (also sets DATA_ROOT import paths)
import audit_pre_training_ai_dataset as paths  # noqa: E402

from actions import apply_action  # noqa: E402
from compliance import generate_compliance_vector  # noqa: E402
from differential_targeting import apply_differential_action  # noqa: E402
from metrics import compute_metrics  # noqa: E402
from observation import OBSERVATION_INDEX, compute_observation  # noqa: E402
from simulator import EvacuationSimulator  # noqa: E402
import stress_test_scenarios as ST  # noqa: E402
from timing_experiment import diagnostic_all_edges_unmet  # noqa: E402
import topology as topology_module  # noqa: E402
from topology import DECISION_INTERVAL, MAX_STEPS, N_OCCUPANTS  # noqa: E402

SPEC_DIR = paths.OUT_DIR
PILOT_DIR = os.path.join(SPEC_DIR, "pilot")
ROWS_FILE = "pilot_rows.csv.gz"
GENERATION_FILE = "pilot_generation_record.json"
ELIGIBLE = ["R1", "R2", "C1", "R3", "R4", "C2", "H"]
VALUE = {"UNTOUCHED": None, "FAVOR_A": 0, "PARITY": 1, "FAVOR_B": 2}
T90_COUNT = int(np.ceil(0.9 * N_OCCUPANTS))
_OBS_POS = {n: i for i, n in enumerate(OBSERVATION_INDEX)}

TARGET_COLUMNS = ["future_T90", "T90_reached_before_decision", "future_T100", "T100_censored", "future_unmet",
                  "future_hazard_unmet", "T90_abs", "T100_abs"]
META_COLUMNS = ["trajectory_id", "split_role", "template", "distribution", "t_h", "p", "seed", "compliance_key",
                "epsilon", "decision_tick", "pre_decision_action_history", "pre_decision_perturbed",
                "pre_decision_unmet", "state_fingerprint", "n_nonempty_eligible_zones", "hazard_started"]
CAND_COLUMNS = ["candidate_id", "candidate_name", "equivalence_class", "is_default_candidate", "default_action"]
PROV_COLUMNS = ["continuation_policy", "continuation_hash", "spec_version", "spec_dataset_design_sha256"]


# ===========================================================================
# Frozen specification.
# ===========================================================================

def _read_json(name):
    with open(os.path.join(SPEC_DIR, name), encoding="utf-8") as f:
        return json.load(f)


def load_frozen_spec() -> dict:
    dd = _read_json("dataset_design.json")
    td = _read_json("target_definitions.json")
    cas = _read_json("candidate_action_set.json")
    if dd.get("spec_version") != "v2" or td.get("spec_version") != "v2":
        raise RuntimeError("specification v2 (D1/D2) is required before generating the pilot")
    pilot = dd["phases"]["PHASE_1_PILOT"]["exact_specification"]
    if td["continuation"]["policy_id"] != lfd.POLICY_ID or td["continuation"]["policy_hash"] != lfd.policy_hash():
        raise RuntimeError("continuation policy does not match the frozen specification")
    if cas["name"] != "BASE2_298" or cas["size"] != 298:
        raise RuntimeError("candidate set is not BASE2_298")
    cands = [dict(id=c["id"], name=c["name"], asg={z: VALUE[c["encoding"][z]] for z in ELIGIBLE})
             for c in cas["candidates"]]
    return dict(pilot=pilot, candidates=cands,
                hashes={n: paths.sha256_file(os.path.join(SPEC_DIR, n))
                        for n in ("dataset_design.json", "target_definitions.json", "candidate_action_set.json",
                                  "final_feature_manifest.json")})


# ===========================================================================
# Label-free inputs and default.
# ===========================================================================

def features(sim) -> np.ndarray:
    """The 26 admissible features. waiting_H_total = hub queue length (label-free)."""
    obs = compute_observation(sim)
    vals = {}
    for name in lfd.ADMISSIBLE_FEATURES:
        vals[name] = float(len(sim.zone_waiting["H"])) if name == "waiting_H_total" else float(obs[_OBS_POS[name]])
    return np.asarray([vals[n] for n in lfd.ADMISSIBLE_FEATURES], dtype=np.float64)


CONTINUATION_CALLS = Counter()


def default_action(x26) -> int:
    CONTINUATION_CALLS["calls"] += 1
    CONTINUATION_CALLS["input_len_26"] += int(len(x26) == 26)
    return lfd.label_free_default_action(x26)


# ===========================================================================
# Trajectories and pre-decision states.
# ===========================================================================

def trajectories(pilot: dict) -> list:
    out = []
    for template, dist in pilot["templates"].items():
        for th in pilot["t_h_levels"]:
            for p in pilot["p_levels"]:
                for rep in range(pilot["replicates_per_cell"]):
                    idx = len(out)
                    seed = 190000 + idx
                    out.append(dict(trajectory_id=f"PILOT-{idx:03d}", index=idx, template=template, distribution=dist,
                                    t_h=th, t_h_int=(pilot["t_h_none_value"] if th == "none" else int(th)), p=p,
                                    seed=seed, replicate=rep, compliance_key=f"AILX_v1|PILOT|{template}",
                                    epsilon=pilot["pre_decision_policy"]["epsilon"]))
    assert len(out) == pilot["n_trajectories"]
    return out


def fingerprint(sim) -> str:
    return paths._state_fingerprint(sim)


def pre_decision_states(traj: dict) -> list:
    sim = EvacuationSimulator()
    sim.reset(traj["distribution"], traj["t_h_int"],
              generate_compliance_vector(traj["compliance_key"], traj["seed"], p=traj["p"]))
    rng = np.random.default_rng([traj["seed"], int(round(traj["p"] * 1000)), traj["t_h_int"], 7])
    history, perturbed, states, pre_unmet = [], [], [], 0
    while sim.n_active() > 0 and sim.t < MAX_STEPS:
        if sim.t % DECISION_INTERVAL == 0:
            x = features(sim)
            states.append(dict(traj=traj, tick=sim.t, sim=copy.deepcopy(sim), x=x, history=list(history),
                               perturbed=list(perturbed), pre_unmet=pre_unmet, fp=fingerprint(sim),
                               n_evacuated=len(sim.evacuated)))
            u = rng.random()
            if u < traj["epsilon"]:
                a, pert = int(rng.integers(0, 3)), True
            else:
                a, pert = default_action(x), False
            history.append(a)
            perturbed.append(pert)
            apply_action(sim, a)
        pre_unmet += diagnostic_all_edges_unmet(sim, sim.t, traj["t_h_int"], topology_module.edge_capacity)
        sim._run_one_tick(None)
        sim.t += 1
    return states


# ===========================================================================
# Candidates, equivalence classes and rollouts.
# ===========================================================================

def label_map(sim, asg: dict) -> tuple:
    """Resulting target_exit labels (actions.apply_action rules), without mutating sim."""
    labs = []
    for oid in sorted(sim.occupants):
        o = sim.occupants[oid]
        lab = o["target_exit"]
        if o["location"] == "zone" and o["compliant"] and not o["committed"]:
            v = asg.get(o["place"])
            if v is not None:
                lab = "A" if v == 0 else ("B" if v == 2 else ("A" if oid % 2 == 0 else "B"))
        labs.append(lab)
    return tuple(labs)


def restrict(sim, asg: dict) -> dict:
    return {z: v for z, v in asg.items() if len(sim.zone_waiting[z]) > 0}


def rollout(state: dict, asg: dict) -> dict:
    sim = copy.deepcopy(state["sim"])
    same_start = fingerprint(sim) == state["fp"]
    pairs = [(z, v) for z, v in asg.items() if v is not None]
    if pairs:
        apply_differential_action(sim, pairs)
    t_d, t_h = sim.t, state["traj"]["t_h_int"]
    has_h = t_h <= MAX_STEPS
    unmet = haz = 0
    first = True
    while sim.n_active() > 0 and sim.t < MAX_STEPS:
        if sim.t % DECISION_INTERVAL == 0 and not first:
            apply_action(sim, default_action(features(sim)))
        first = False
        unmet += diagnostic_all_edges_unmet(sim, sim.t, t_h, topology_module.edge_capacity)
        if has_h and sim.t >= t_h:
            _, hu, _ = ST.diagnostic_branch_demand_and_unmet(sim, "H_CE1", "A", sim.t, t_h, topology_module.edge_capacity)
            haz += hu
        sim._run_one_tick(None)
        sim.t += 1
    m = compute_metrics(sim.evacuated, N=N_OCCUPANTS, max_steps=MAX_STEPS)
    realized = sum(e["h_ce1_unmet"] for e in sim.tick_log if e["t"] >= t_d and has_h and e["t"] >= t_h)
    t90_pre = state["n_evacuated"] >= T90_COUNT
    return dict(
        future_T90=0 if t90_pre else ((m["T90"] - t_d) if m["T90_defined"] else float("inf")),
        T90_reached_before_decision=t90_pre, future_T100=m["T100"] - t_d, T100_censored=m["T100_censored"],
        future_unmet=unmet, future_hazard_unmet=haz,
        T90_abs=m["T90"] if m["T90_defined"] else float("inf"), T100_abs=m["T100"],
        _same_start=same_start, _hazard_matches_tick_log=(realized == haz))


def evaluate_state(state: dict, candidates: list, prov: dict) -> tuple:
    sim = state["sim"]
    d = default_action(state["x"])
    classes, cand_class = {}, []
    for c in candidates:
        asg = restrict(sim, c["asg"])
        lm = label_map(sim, asg)
        cls = hashlib.sha1("".join(lm).encode()).hexdigest()[:12]
        if cls not in classes:
            classes[cls] = rollout(state, asg)
        cand_class.append(cls)
    default_name = {0: "BASE=A", 1: "BASE=P", 2: "BASE=B"}[d]
    tr = state["traj"]
    hazard_started = tr["t_h_int"] <= MAX_STEPS and state["tick"] >= tr["t_h_int"]
    meta = [tr["trajectory_id"], "PILOT", tr["template"], json.dumps(tr["distribution"], sort_keys=True), tr["t_h"],
            tr["p"], tr["seed"], tr["compliance_key"], tr["epsilon"], state["tick"],
            ",".join(map(str, state["history"])), ",".join("1" if b else "0" for b in state["perturbed"]),
            state["pre_unmet"], state["fp"][:16], sum(1 for z in ELIGIBLE if len(sim.zone_waiting[z]) > 0),
            int(hazard_started)]
    feats = [repr(float(v)) for v in state["x"]]
    rows = []
    for c, cls in zip(candidates, cand_class):
        o = classes[cls]
        rows.append(meta + feats + [c["id"], c["name"], cls, int(c["name"] == default_name), d]
                    + [o[k] if not isinstance(o[k], bool) else int(o[k]) for k in TARGET_COLUMNS]
                    + [prov["policy_id"], prov["policy_hash"], "v2", prov["dataset_design_sha256"]])
    checks = dict(n_classes=len(classes),
                  same_start=sum(int(o["_same_start"]) for o in classes.values()),
                  hazard_tick_log=sum(int(o["_hazard_matches_tick_log"]) for o in classes.values()))
    return rows, checks


HEADER = META_COLUMNS + list(lfd.ADMISSIBLE_FEATURES) + CAND_COLUMNS + TARGET_COLUMNS + PROV_COLUMNS


def generate(limit_trajectories: int | None = None) -> dict:
    fs = load_frozen_spec()
    prov = dict(policy_id=lfd.POLICY_ID, policy_hash=lfd.policy_hash(),
                dataset_design_sha256=fs["hashes"]["dataset_design.json"])
    trajs = trajectories(fs["pilot"])
    if limit_trajectories:
        trajs = trajs[:limit_trajectories]
    CONTINUATION_CALLS.clear()
    all_rows, checks, n_states, pert = [], Counter(), 0, Counter()
    t0 = time.perf_counter()
    for tr in trajs:
        for st in pre_decision_states(tr):
            rows, ck = evaluate_state(st, fs["candidates"], prov)
            all_rows.extend(rows)
            checks.update(ck)
            n_states += 1
        pert["trajectories"] += 1
    return dict(rows=all_rows, n_trajectories=len(trajs), n_states=n_states, checks=dict(checks),
                continuation_calls=dict(CONTINUATION_CALLS), seconds=round(time.perf_counter() - t0, 1),
                spec_hashes=fs["hashes"], provenance=prov)


def rows_to_csv_bytes(rows: list) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(HEADER)
    for r in rows:
        w.writerow(r)
    return buf.getvalue().encode("utf-8")


def write(result: dict) -> dict:
    os.makedirs(PILOT_DIR, exist_ok=True)
    rows_path = os.path.join(PILOT_DIR, ROWS_FILE)
    if os.path.exists(rows_path):
        raise FileExistsError(f"{rows_path} exists; the pilot is generated once (delete it deliberately to regenerate)")
    raw = rows_to_csv_bytes(result["rows"])
    with gzip.GzipFile(rows_path, "wb", mtime=0) as f:
        f.write(raw)
    rec = dict(
        marker="ai_learning_pilot_generation_v1", phase="PHASE 1 PILOT ONLY", rows_file=ROWS_FILE,
        rows_uncompressed_sha256=hashlib.sha256(raw).hexdigest(), rows_file_sha256=paths.sha256_file(rows_path),
        n_rows=len(result["rows"]), n_columns=len(HEADER), header=HEADER,
        n_trajectories=result["n_trajectories"], n_decision_states=result["n_states"],
        generation_checks=result["checks"], continuation_calls=result["continuation_calls"],
        spec_hashes=result["spec_hashes"], provenance=result["provenance"], seconds=result["seconds"],
        not_done=["no model trained", "no selector", "no closed-loop evaluation", "no full 5,544-trajectory dataset"],
    )
    with open(os.path.join(PILOT_DIR, GENERATION_FILE), "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2, default=paths._json_default)
    return rec


if __name__ == "__main__":
    res = generate()
    rec = write(res)
    print(f"trajectories {rec['n_trajectories']}, decision states {rec['n_decision_states']}, rows {rec['n_rows']}, "
          f"checks {rec['generation_checks']}, continuation {rec['continuation_calls']}, {rec['seconds']}s")
