"""PHASE 3: full AI-learning dataset generator (frozen specification v2).

The frozen v2 specification did not fix a per-trajectory seed rule for the
full dataset; the user decided it after the pilot: unique seed per
trajectory (USER_DECISIONS). The rule is still passed explicitly
(`--seed-rule`), and only the approved rule may write the dataset.

Everything else is read from the frozen v2 files, whose hashes are verified
against spec_version_record.json before anything is generated (mismatch =
STOP). Rollouts reuse the validated pilot functions unchanged
(run_ai_learning_pilot: pre_decision_states, features, label_map, restrict,
rollout), so the label-free continuation, D1 targets and candidate
semantics are byte-for-byte those the pilot audit verified. Every invariant
is checked per state during generation; the first failure stops the run.
Trains nothing.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import sys
import time
from collections import Counter

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)
import run_ai_learning_pilot as P  # noqa: E402
import ai_learning_label_free_default as lfd  # noqa: E402
import audit_pre_training_ai_dataset as paths  # noqa: E402

from topology import MAX_STEPS  # noqa: E402

SPEC_DIR = paths.OUT_DIR
FULL_DIR = os.path.join(SPEC_DIR, "full_dataset")
ROWS_FILE = "full_ai_learning_rows.csv.gz"
ROLE_ORDER = ["TRAIN", "VALIDATION", "TEST_PRIMARY_JOINT", "TEST_HELDOUT_TIMING", "TEST_HELDOUT_COMPLIANCE",
              "DIAG_EXTRAPOLATION"]
ZERO_SHOT_ROLE = "DIAG_TEMPLATE_ZERO_SHOT"
ZERO_SHOT_SEEDS_PER_CELL = 3  # dataset_design.json: zero-shot trajectories = 2 templates x 59 cells x 3

SEED_RULES = {
    "unique_per_trajectory": "seed = role range start + running index within the role (as in the pilot; satisfies X7) "
                             "-- TRAIN would use 100000-101799, beyond its documented 100000-100999 range",
    "replicate_index_per_cell": "seed = role range start + replicate index (fits every documented range) -- the same "
                                "(key, seed) then repeats across all cells of a template within a role, and X7 "
                                "('seeds unique per trajectory') fails as written",
}


class InvariantError(RuntimeError):
    pass


def verify_spec_v2() -> dict:
    with open(os.path.join(SPEC_DIR, "spec_version_record.json"), encoding="utf-8") as f:
        rec = json.load(f)
    diffs = {n: v["v2_sha256"] for n, v in rec["files"].items()
             if paths.sha256_file(os.path.join(SPEC_DIR, n)) != v["v2_sha256"]}
    if diffs:
        raise InvariantError(f"specification differs from frozen v2: {sorted(diffs)} -- STOP")
    return {n: v["v2_sha256"] for n, v in rec["files"].items()}


def _range_start(ranges: dict, role: str) -> int:
    return int(ranges[role].split(" ")[0].split("-")[0])


def full_trajectories(seed_rule: str) -> list:
    if seed_rule not in SEED_RULES:
        raise InvariantError(f"--seed-rule must be one of {sorted(SEED_RULES)} (no default: unresolved design decision)")
    dd = P._read_json("dataset_design.json")
    tpl_train = dd["factors"]["templates_train_and_test"]
    tpl_zs = dd["factors"]["templates_zero_shot_only"]
    dist = P._read_json("scenario_space.json")["dimensions"]["A_occupant_distribution"]["chosen"]
    roles = dd["cell_roles"]
    seeds_per = dd["seeds_per_cell_per_template_FULL"]
    ranges = dd["seed_ranges_by_role"]
    eps = dd["phases"]["PHASE_1_PILOT"]["exact_specification"]["pre_decision_policy"]["epsilon"]
    cells = list(roles)  # t_h-major, p-minor, frozen order

    def cell_parts(cell):
        th, p = cell.split("|")
        th = th.split("=")[1]
        return th, float(p.split("=")[1])

    out, per_role_counter = [], Counter()

    def add(role, template, cell, rep):
        th, p = cell_parts(cell)
        k = per_role_counter[role]
        per_role_counter[role] += 1
        start = _range_start(ranges, role)
        seed = start + (k if seed_rule == "unique_per_trajectory" else rep)
        out.append(dict(trajectory_id=f"{role}-{template}-{cell}-r{rep}", index=len(out), split_role=role,
                        template=template, distribution=dist[template], t_h=th,
                        t_h_int=(MAX_STEPS + 1 if th == "none" else int(th)), p=p, seed=seed, replicate=rep,
                        cell=cell, compliance_key=f"AILX_v1|{role}|{template}", epsilon=eps))

    for role in ROLE_ORDER:
        for template in tpl_train:
            for cell in cells:
                if roles[cell] == role:
                    for rep in range(seeds_per[role]):
                        add(role, template, cell, rep)
    for template in tpl_zs:
        for cell in cells:
            if roles[cell] != "UNUSED_BUFFER":
                for rep in range(ZERO_SHOT_SEEDS_PER_CELL):
                    add(ZERO_SHOT_ROLE, template, cell, rep)
    expected = dd["trajectories_FULL_total"]
    if len(out) != expected:
        raise InvariantError(f"trajectory count {len(out)} != frozen {expected}")
    by_role = Counter(t["split_role"] for t in out)
    for role, n in dd["trajectories_FULL_by_role"].items():
        if n and by_role[role] != n:
            raise InvariantError(f"{role}: {by_role[role]} trajectories != frozen {n}")
    return out


def seed_rule_report(trajs: list) -> dict:
    dd = P._read_json("dataset_design.json")
    ranges = dd["seed_ranges_by_role"]
    rep = {}
    for role in ROLE_ORDER + [ZERO_SHOT_ROLE]:
        seeds = [t["seed"] for t in trajs if t["split_role"] == role]
        lo, hi = [int(x) for x in ranges[role].split(" ")[0].split("-")]
        keys = Counter((t["compliance_key"], t["seed"]) for t in trajs if t["split_role"] == role)
        rep[role] = dict(n=len(seeds), min=min(seeds), max=max(seeds), inside_documented_range=lo <= min(seeds)
                         and max(seeds) <= hi, unique_seeds=len(set(seeds)) == len(seeds),
                         max_trajectories_sharing_key_and_seed=max(keys.values()))
    all_seeds = [(t["split_role"], t["seed"]) for t in trajs]
    cross = {}
    for a in rep:
        for b in rep:
            if a < b:
                sa = {s for r, s in all_seeds if r == a}
                sb = {s for r, s in all_seeds if r == b}
                if sa & sb:
                    cross[f"{a}&{b}"] = len(sa & sb)
    return dict(per_role=rep, cross_role_seed_overlap=cross)


# ===========================================================================
# Generation with per-state invariants (pilot functions reused unchanged).
# ===========================================================================

HEADER = (["trajectory_id", "split_role", "template", "distribution", "cell", "t_h", "p", "seed", "replicate",
           "compliance_key", "epsilon", "decision_tick", "pre_decision_action_history", "pre_decision_perturbed",
           "pre_decision_unmet", "state_fingerprint", "n_nonempty_eligible_zones", "hazard_started"]
          + list(lfd.ADMISSIBLE_FEATURES) + P.CAND_COLUMNS + P.TARGET_COLUMNS + P.PROV_COLUMNS + ["seed_rule"])


def _check_state(rows_for_state, n_cands, classes, prov):
    if len(rows_for_state) != n_cands:
        raise InvariantError("candidate count != 298")
    for o in classes.values():
        if not o["_same_start"]:
            raise InvariantError("candidate did not start from the identical pre-decision state")
        if not o["_hazard_matches_tick_log"]:
            raise InvariantError("hazard target != simulator tick_log")
        if o["T100_censored"]:
            raise InvariantError("censored T100")
        for k in ("future_T90", "future_T100", "future_unmet", "future_hazard_unmet"):
            v = o[k]
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                raise InvariantError(f"non-finite target {k}")
            if v < 0:
                raise InvariantError(f"negative target {k}")


def evaluate_state(state: dict, candidates: list, prov: dict, seed_rule: str) -> tuple:
    import hashlib as _h
    sim = state["sim"]
    x = state["x"]
    if len(x) != 26 or any(math.isnan(v) or math.isinf(v) for v in x):
        raise InvariantError("feature vector invalid")
    d = P.default_action(x)
    classes, cand_class = {}, []
    for c in candidates:
        asg = P.restrict(sim, c["asg"])
        lm = P.label_map(sim, asg)
        cls = _h.sha1("".join(lm).encode()).hexdigest()[:12]
        if cls not in classes:
            classes[cls] = P.rollout(state, asg)
        cand_class.append(cls)
    tr = state["traj"]
    if tr["t_h_int"] > MAX_STEPS and any(o["future_hazard_unmet"] for o in classes.values()):
        raise InvariantError("hazard target non-zero in a no-hazard scenario")
    default_name = {0: "BASE=A", 1: "BASE=P", 2: "BASE=B"}[d]
    hazard_started = tr["t_h_int"] <= MAX_STEPS and state["tick"] >= tr["t_h_int"]
    meta = [tr["trajectory_id"], tr["split_role"], tr["template"], json.dumps(tr["distribution"], sort_keys=True),
            tr["cell"], tr["t_h"], tr["p"], tr["seed"], tr["replicate"], tr["compliance_key"], tr["epsilon"],
            state["tick"], ",".join(map(str, state["history"])), ",".join("1" if b else "0" for b in state["perturbed"]),
            state["pre_unmet"], state["fp"][:16], sum(1 for z in P.ELIGIBLE if len(sim.zone_waiting[z]) > 0),
            int(hazard_started)]
    feats = [repr(float(v)) for v in x]
    rows = []
    for c, cls in zip(candidates, cand_class):
        o = classes[cls]
        rows.append(meta + feats + [c["id"], c["name"], cls, int(c["name"] == default_name), d]
                    + [o[k] if not isinstance(o[k], bool) else int(o[k]) for k in P.TARGET_COLUMNS]
                    + [prov["policy_id"], prov["policy_hash"], "v2", prov["dataset_design_sha256"], seed_rule])
    _check_state(rows, len(candidates), classes, prov)
    return rows, dict(n_classes=len(classes), same_start=sum(int(o["_same_start"]) for o in classes.values()),
                      hazard_tick_log=sum(int(o["_hazard_matches_tick_log"]) for o in classes.values()))


def generate(seed_rule: str, limit_per_role: int | None = None, sink=None) -> dict:
    spec_hashes = verify_spec_v2()
    fs = P.load_frozen_spec()
    if [c["id"] for c in fs["candidates"]] != list(range(298)):
        raise InvariantError("candidate set is not BASE2_298")
    prov = dict(policy_id=lfd.POLICY_ID, policy_hash=lfd.policy_hash(),
                dataset_design_sha256=fs["hashes"]["dataset_design.json"])
    trajs = full_trajectories(seed_rule)
    if limit_per_role:
        taken, keep = Counter(), []
        for t in trajs:
            if taken[t["split_role"]] < limit_per_role:
                keep.append(t)
                taken[t["split_role"]] += 1
        trajs = keep
    P.CONTINUATION_CALLS.clear()
    checks, n_states, n_rows = Counter(), 0, 0
    t0 = time.perf_counter()
    for tr in trajs:
        for st in P.pre_decision_states(tr):
            rows, ck = evaluate_state(st, fs["candidates"], prov, seed_rule)
            checks.update(ck)
            n_states += 1
            n_rows += len(rows)
            if sink is not None:
                sink(rows)
    cc = dict(P.CONTINUATION_CALLS)
    if cc.get("calls") != cc.get("input_len_26"):
        raise InvariantError("a continuation call received something other than 26 features")
    return dict(n_trajectories=len(trajs), n_states=n_states, n_rows=n_rows, checks=dict(checks),
                continuation_calls=cc, seconds=round(time.perf_counter() - t0, 1), spec_hashes=spec_hashes,
                provenance=prov, seed_rule=seed_rule)


GENERATION_RECORD = "full_generation_record.json"
APPROVED_SEED_RULE = "unique_per_trajectory"

USER_DECISIONS = dict(
    seed_rule=dict(
        decision="unique seed per trajectory (user decision after the pilot)",
        rule=SEED_RULES["unique_per_trajectory"],
        documented_deviation="TRAIN seeds extend from the documented 100000-100999 to 100000-101799 to give 1,800 "
                             "unique TRAIN seeds; non-overlap with every other role range, the PILOT range and the "
                             "Test A range 125000-125999 is preserved and verified",
        no_seed_reused=True),
    test_A=dict(decision="separate later batch: the 5,544-trajectory main dataset only; no Test A trajectories here",
                reserved_range="125000-125999 (untouched by this run)"),
    unchanged=["Spec v2 files", "BASE2_298", "26 admissible features", "target definitions (D1)",
               "label-free continuation", "split structure", "G1-G7 / X1-X10 thresholds and definitions"],
)

# Declared BEFORE generation (no full-dataset data exists when this is written).
CRITERIA_APPLICATION = dict(
    G1_G6_X1_X10="applied with identical definitions and thresholds to all 5,544 trajectories",
    G7="the only pilot-specific constant: 'every pilot cell has 2 complete trajectories' is evaluated as 'every "
       "allocated (role, template, t_h, p) cell has its frozen number of complete trajectories' (dataset_design.json)",
    X7_association="evaluated LITERALLY over all 5,544 trajectories (template vs t_h and template vs p, Cramer's V "
                   "must be 0). Known from the frozen allocation alone, before generation: V = 0.0399 (t_h) and "
                   "0.0477 (p) because the two zero-shot templates get 3 trajectories per non-buffer cell while the "
                   "six main templates are role-weighted; over the six main templates V = 0. Reported, not redefined.",
    quantiles="Spec v2 stores exact per-rollout targets; conditional quantiles are model outputs defined for "
              "training, not stored targets, so no quantile columns are generated",
)


def write_full(seed_rule: str) -> dict:
    if seed_rule != APPROVED_SEED_RULE:
        raise InvariantError(f"only the approved seed rule {APPROVED_SEED_RULE!r} may write the full dataset")
    os.makedirs(FULL_DIR, exist_ok=True)
    rows_path = os.path.join(FULL_DIR, ROWS_FILE)
    rec_path = os.path.join(FULL_DIR, GENERATION_RECORD)
    if os.path.exists(rows_path):
        raise FileExistsError(f"{rows_path} exists; the full dataset is generated once")
    spec_hashes = verify_spec_v2()
    trajs = full_trajectories(seed_rule)
    pre = dict(marker="ai_learning_full_generation_v1", status="STARTED", phase="PHASE 3 FULL DATASET",
               user_decisions=USER_DECISIONS, criteria_application_declared_before_generation=CRITERIA_APPLICATION,
               spec_v2_hashes_verified=spec_hashes, allocation=dict(Counter(t["split_role"] for t in trajs)),
               n_trajectories_planned=len(trajs), seed_report=seed_rule_report(trajs),
               started=time.strftime("%Y-%m-%dT%H:%M:%S"))
    with open(rec_path, "w", encoding="utf-8") as f:
        json.dump(pre, f, indent=2, default=paths._json_default)
    sha = hashlib.sha256()
    gz = gzip.GzipFile(rows_path, "wb", mtime=0)
    try:
        def emit(text: str):
            b = text.encode("utf-8")
            sha.update(b)
            gz.write(b)
        buf = io.StringIO()
        csv.writer(buf, lineterminator="\n").writerow(HEADER)
        emit(buf.getvalue())

        def sink(rows):
            b = io.StringIO()
            w = csv.writer(b, lineterminator="\n")
            for r in rows:
                w.writerow(r)
            emit(b.getvalue())
        try:
            res = generate(seed_rule, sink=sink)
        except Exception as e:
            pre.update(status="STOPPED ON INVARIANT FAILURE", error=f"{type(e).__name__}: {e}")
            with open(rec_path, "w", encoding="utf-8") as f:
                json.dump(pre, f, indent=2, default=paths._json_default)
            raise
    finally:
        gz.close()
    pre.update(status="COMPLETE", rows_file=ROWS_FILE, rows_uncompressed_sha256=sha.hexdigest(),
               rows_file_sha256=paths.sha256_file(rows_path), header=HEADER, n_columns=len(HEADER),
               n_trajectories=res["n_trajectories"], n_decision_states=res["n_states"], n_rows=res["n_rows"],
               generation_checks=res["checks"], continuation_calls=res["continuation_calls"],
               provenance=res["provenance"], seconds=res["seconds"], finished=time.strftime("%Y-%m-%dT%H:%M:%S"))
    with open(rec_path, "w", encoding="utf-8") as f:
        json.dump(pre, f, indent=2, default=paths._json_default)
    return pre


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-rule", required=True, choices=sorted(SEED_RULES))
    ap.add_argument("--smoke-per-role", type=int, default=None,
                    help="in-memory smoke run: this many trajectories per role, nothing written")
    a = ap.parse_args()
    if a.smoke_per_role:
        r = generate(a.seed_rule, limit_per_role=a.smoke_per_role)
        print(json.dumps({k: v for k, v in r.items() if k != "spec_hashes"}, indent=1, default=str))
        sys.exit(0)
    rec = write_full(a.seed_rule)
    print(f"trajectories {rec['n_trajectories']}, states {rec['n_decision_states']}, rows {rec['n_rows']}, "
          f"checks {rec['generation_checks']}, {rec['seconds']}s")
