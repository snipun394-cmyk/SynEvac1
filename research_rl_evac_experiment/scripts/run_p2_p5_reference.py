"""Model-free closed-loop P2/P5 reference run (Amendment A2, Issue 1), exactly as frozen.

240 discarded pilot configurations (seeds 190000-190239), full episodes, no
epsilon, paired (identical configuration and compliance vector for both).
  P2: the label-free default issued globally at every decision tick.
  P5: at every decision tick, every BASE2_298 candidate is rolled out with the
      frozen dataset rollout (candidate now, label-free continuation after)
      and the lexicographically best key is applied; ties -> the default
      candidate BASE=l(s) if tied-optimal, else the lowest candidate id.
No learned model, no hidden heuristic (every hidden-heuristic entry point is
disabled for the whole run and would raise). Delta = mean(T100_P2 - T100_P5);
95% percentile bootstrap over paired episodes, 2,000 resamples, seed 5544105.
Writes only amendments/amendment_A2_reference_run.json. Trains nothing.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import numpy as np

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)
import run_ai_learning_pilot as P  # noqa: E402
import ai_learning_label_free_default as lfd  # noqa: E402
import audit_pre_training_ai_dataset as prior  # noqa: E402

from actions import apply_action  # noqa: E402
from compliance import generate_compliance_vector  # noqa: E402
from differential_targeting import apply_differential_action  # noqa: E402
from metrics import compute_metrics  # noqa: E402
from simulator import EvacuationSimulator  # noqa: E402
import stress_test_scenarios as ST  # noqa: E402
from timing_experiment import diagnostic_all_edges_unmet  # noqa: E402
import topology as topology_module  # noqa: E402
from topology import DECISION_INTERVAL, MAX_STEPS, N_OCCUPANTS  # noqa: E402

ART = prior.OUT_DIR
AMD = os.path.join(ART, "amendments")
OUT = os.path.join(AMD, "amendment_A2_reference_run.json")
BOOT_SEED, N_BOOT = 5544105, 2000


def protected() -> dict:
    with open(os.path.join(ART, "spec_version_record.json"), encoding="utf-8") as f:
        spec_files = list(json.load(f)["files"])
    files = spec_files + ["spec_version_record.json", "full_dataset/full_ai_learning_rows.csv.gz",
                          "full_dataset/full_generation_record.json", "full_dataset/full_go_no_go.json",
                          "pilot/pilot_rows.csv.gz", "amendments/amendment_A1_acceptance_criteria.json",
                          "amendments/amendment_A1_evaluation.json", "amendments/amendment_A2_evaluation_protocol.json",
                          "amendments/AMENDMENT_A2_EVALUATION_PROTOCOL.md", "amendments/amendment_A2_approval.json"]
    return {f: prior.sha256_file(os.path.join(ART, f)) for f in files}


def _reset(cfg):
    sim = EvacuationSimulator()
    sim.reset(cfg["distribution"], cfg["t_h_int"], generate_compliance_vector(cfg["compliance_key"], cfg["seed"], p=cfg["p"]))
    return sim


def p5_choice(sim, cfg, candidates, checks) -> tuple:
    x = P.features(sim)
    d = P.default_action(x)
    default_name = {0: "BASE=A", 1: "BASE=P", 2: "BASE=B"}[d]
    state = dict(traj=cfg, tick=sim.t, sim=sim, fp=P.fingerprint(sim), n_evacuated=len(sim.evacuated))
    cache, keys = {}, []
    for c in candidates:
        asg = P.restrict(sim, c["asg"])
        lm = P.label_map(sim, asg)
        if lm not in cache:
            o = P.rollout(state, asg)
            checks["one_step_rollouts"] += 1
            checks["same_start"] += int(o["_same_start"])
            checks["hazard_tick_log"] += int(o["_hazard_matches_tick_log"])
            cache[lm] = (o["future_T90"], o["future_T100"], o["future_unmet"], o["future_hazard_unmet"])
        keys.append(cache[lm])
    best = min(keys)
    tied = [i for i, k in enumerate(keys) if k == best]
    default_id = next(c["id"] for c in candidates if c["name"] == default_name)
    chosen = default_id if default_id in tied else min(tied)
    return chosen, P.restrict(sim, candidates[chosen]["asg"]), chosen == default_id


def episode(cfg, policy, candidates, checks) -> dict:
    sim = _reset(cfg)
    fp0 = P.fingerprint(sim)
    unmet = haz = decisions = deviations = 0
    has_h = cfg["t_h_int"] <= MAX_STEPS
    while sim.n_active() > 0 and sim.t < MAX_STEPS:
        if sim.t % DECISION_INTERVAL == 0:
            decisions += 1
            if policy == "P2":
                apply_action(sim, P.default_action(P.features(sim)))
            else:
                _, asg, is_default = p5_choice(sim, cfg, candidates, checks)
                deviations += int(not is_default)
                pairs = [(z, v) for z, v in asg.items() if v is not None]
                if pairs:
                    apply_differential_action(sim, pairs)
        unmet += diagnostic_all_edges_unmet(sim, sim.t, cfg["t_h_int"], topology_module.edge_capacity)
        if has_h and sim.t >= cfg["t_h_int"]:
            _, hu, _ = ST.diagnostic_branch_demand_and_unmet(sim, "H_CE1", "A", sim.t, cfg["t_h_int"],
                                                             topology_module.edge_capacity)
            haz += hu
        sim._run_one_tick(None)
        sim.t += 1
    m = compute_metrics(sim.evacuated, N=N_OCCUPANTS, max_steps=MAX_STEPS)
    return dict(start_fingerprint=fp0, T100=m["T100"], T100_censored=m["T100_censored"],
                T90=m["T90"] if m["T90_defined"] else None, unmet=unmet, hazard_unmet=haz,
                decisions=decisions, deviations_from_default=deviations)


def main():
    t0 = time.perf_counter()
    before = protected()
    fs = P.load_frozen_spec()
    cfgs = P.trajectories(fs["pilot"])
    assert len(cfgs) == 240 and [c["seed"] for c in cfgs] == list(range(190000, 190240))
    import deadband_heuristic as dh
    import capacity_aware_heuristic as cah

    class HiddenHeuristicCalled(RuntimeError):
        pass

    def boom(*a, **k):
        raise HiddenHeuristicCalled("hidden-label heuristic called")
    saved = (prior.asgc.HEURISTIC, dh.DeadbandCapacityAwareHeuristic.__call__, dh.DeadbandCapacityAwareHeuristic.predict,
             cah.heuristic_action)
    prior.asgc.HEURISTIC = boom
    dh.DeadbandCapacityAwareHeuristic.__call__ = boom
    dh.DeadbandCapacityAwareHeuristic.predict = boom
    cah.heuristic_action = boom
    P.CONTINUATION_CALLS.clear()
    from collections import Counter
    checks = Counter()
    try:
        rows = []
        for cfg in cfgs:
            e2 = episode(cfg, "P2", fs["candidates"], checks)
            e5 = episode(cfg, "P5", fs["candidates"], checks)
            rows.append(dict(trajectory_id=cfg["trajectory_id"], seed=cfg["seed"], layout=cfg["template"], t_h=cfg["t_h"],
                             p=cfg["p"], P2=e2, P5=e5, paired_start_equal=e2["start_fingerprint"] == e5["start_fingerprint"]))
    finally:
        prior.asgc.HEURISTIC, dh.DeadbandCapacityAwareHeuristic.__call__, dh.DeadbandCapacityAwareHeuristic.predict, \
            cah.heuristic_action = saved
    diff = np.array([r["P2"]["T100"] - r["P5"]["T100"] for r in rows], dtype=float)
    delta = float(diff.mean())
    rng = np.random.default_rng(BOOT_SEED)
    boots = np.array([diff[rng.integers(0, len(diff), size=len(diff))].mean() for _ in range(N_BOOT)])
    ci = [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)]
    cc = dict(P.CONTINUATION_CALLS)
    after = protected()
    changed = [k for k in before if before[k] != after[k]]
    if changed:
        raise RuntimeError(f"PROTECTED FILE CHANGED: {changed}")

    def mean(key, pol):
        v = [r[pol][key] for r in rows if r[pol][key] is not None]
        return round(float(np.mean(v)), 4)
    lex = Counter()
    for r in rows:
        k2 = (r["P2"]["T90"], r["P2"]["T100"], r["P2"]["unmet"], r["P2"]["hazard_unmet"])
        k5 = (r["P5"]["T90"], r["P5"]["T100"], r["P5"]["unmet"], r["P5"]["hazard_unmet"])
        lex["P5_better" if k5 < k2 else ("tie" if k5 == k2 else "P5_worse")] += 1
    assessable = delta >= 0.5
    with open(os.path.abspath(__file__), "rb") as f:
        script_sha = hashlib.sha256(f.read()).hexdigest()
    record = dict(
        record="Amendment A2, Issue 1: model-free P2/P5 closed-loop reference run",
        run_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        a2_approval_commit="5eebb24",
        configurations=dict(n_episodes_per_policy=len(rows), seeds="190000-190239", layouts=8,
                            t_h=["none", "3", "5", "7", "9"], p=[0.5, 0.7, 0.9], epsilon=0,
                            outside_every_split=True, primary_test_used=False),
        results=dict(
            mean_T100_P2=round(float(np.mean([r["P2"]["T100"] for r in rows])), 4),
            mean_T100_P5=round(float(np.mean([r["P5"]["T100"] for r in rows])), 4),
            delta=round(delta, 4), delta_bootstrap_95ci=ci, bootstrap=dict(resamples=N_BOOT, seed=BOOT_SEED,
                                                                           unit="paired episode (one per configuration)"),
            delta_ge_0_5=assessable,
            usefulness_assessable="YES" if assessable else "NO",
            P4_criterion_if_assessable="lower bound of the 95% CI of mean(T100_P2 - T100_P4) on TEST_PRIMARY_JOINT >= 0.5 tick (not run)",
            per_episode_T100_difference=dict(P5_better=int((diff > 0).sum()), tie=int((diff == 0).sum()),
                                             P5_worse=int((diff < 0).sum()), min=float(diff.min()), max=float(diff.max())),
            lexicographic_episode_comparison=dict(lex),
            descriptive=dict(mean_T90_P2=mean("T90", "P2"), mean_T90_P5=mean("T90", "P5"),
                             mean_unmet_P2=mean("unmet", "P2"), mean_unmet_P5=mean("unmet", "P5"),
                             mean_hazard_unmet_P2=mean("hazard_unmet", "P2"), mean_hazard_unmet_P5=mean("hazard_unmet", "P5"),
                             P5_decisions=int(sum(r["P5"]["decisions"] for r in rows)),
                             P5_deviations_from_default=int(sum(r["P5"]["deviations_from_default"] for r in rows))),
            censored_episodes=dict(P2=sum(r["P2"]["T100_censored"] for r in rows), P5=sum(r["P5"]["T100_censored"] for r in rows))),
        integrity=dict(
            paired_start_equal=f"{sum(r['paired_start_equal'] for r in rows)}/{len(rows)}",
            one_step_rollouts=checks["one_step_rollouts"], one_step_same_start=checks["same_start"],
            one_step_hazard_tick_log=checks["hazard_tick_log"],
            continuation=dict(policy=lfd.POLICY_ID, policy_hash=lfd.policy_hash(), calls=cc.get("calls", 0),
                              calls_with_26_features=cc.get("input_len_26", 0)),
            hidden_heuristic_disabled_for_whole_run=True, hidden_heuristic_called=False,
            protected_files_unchanged=True, protected_hashes=before, script_sha256=script_sha),
        episodes=rows, seconds=round(time.perf_counter() - t0, 1))
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, default=str)
    print(json.dumps(dict(results=record["results"], integrity={k: v for k, v in record["integrity"].items()
                                                              if k != "protected_hashes"}, seconds=record["seconds"]),
                     indent=1, default=str))


if __name__ == "__main__":
    main()
