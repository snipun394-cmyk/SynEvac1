"""Specification v2: apply the approved decisions D1 and D2 to the frozen
experiment specification, and pre-register the exact Phase-1 pilot.

D1: future hazard-edge unmet demand counts only ticks at or after the actual
    onset (t >= t_h); 0 before onset; 0 in no-hazard scenarios.
D2: continuation after every candidate, selector fallback, abstention
    fallback, OOD fallback, closed-loop default and pre-decision state
    generation all use the label-free SYMMETRIC_SPLIT_DEADBAND (margin 0)
    from scripts/ai_learning_label_free_default.py. The hidden-label
    GLOBAL_DEADBAND heuristic is reference arm P2b only.

v1 files are archived byte-for-byte under spec_versions/v1/ and v2 is always
rebuilt from those archives (idempotent). Files untouched by D1/D2
(candidate_action_set.json, final_feature_manifest.json) are left exactly as
they are. spec_version_record.json lists every file's v1 and v2 SHA-256 and
the fields changed. Trains nothing; generates no pilot data.
"""
from __future__ import annotations

import copy
import json
import os
import shutil
import sys
import time

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)
import audit_final_experiment_specification as spec  # noqa: E402
import audit_label_free_default as lfa  # noqa: E402
import audit_pre_training_ai_dataset as prior  # noqa: E402
import ai_learning_label_free_default as lfd  # noqa: E402
from topology import MAX_STEPS  # noqa: E402

OUT_DIR = prior.OUT_DIR
ARCHIVE_DIR = os.path.join(OUT_DIR, "spec_versions", "v1")
V2_MARKER = "experiment_specification_v2_d1_d2"
CHANGED = ["target_definitions.json", "selector_design.json", "closed_loop_evaluation.json",
           "preregistered_success_criteria.json", "dataset_design.json", "diversity_requirements.json",
           "scenario_space.json", "split_specification.json", "FINAL_EXPERIMENT_SPECIFICATION.md"]
UNCHANGED = ["candidate_action_set.json", "final_feature_manifest.json"]
assert sorted(CHANGED + UNCHANGED) == sorted(spec.OUTPUT_FILES)

LABEL_FREE_DEFAULT = dict(
    policy_id=lfd.POLICY_ID,
    module="scripts/ai_learning_label_free_default.py::label_free_default_action",
    definition="deadband_heuristic_action(rule input with Q_A = Q_B = waiting_H_total / 2, margin 0); PARITY by "
               "default; FAVOR_B once the hazard has actually started (residual H->CE1 capacity 2/6) and at least one "
               "person waits at the hub; never FAVOR_A",
    inputs="exactly the 26 admissible features (the 27-D rule input is rebuilt from them)",
    never_reads=["Q_A_at_H", "Q_B_at_H", "hidden compliance", "target_exit", "future information", "t_h", "p"],
    evidence="LABEL_FREE_DEFAULT_AUDIT.md: ties the hidden-label heuristic in 230/230 existing episodes",
)

# Old-dataset reference values (pre-training audit), fixed here BEFORE any pilot data exists.
OLD = dict(states=450, distinct_inputs=11, distinct_input_action_pairs=140, T100_irreducible_share=0.4599)

PILOT_TEMPLATE_ORDER = list(spec.DISTRIBUTIONS)
PILOT = dict(
    role="PILOT",
    purpose="validate the data-generation process only; discarded afterwards, never used for training or testing",
    templates={k: spec.DISTRIBUTIONS[k] for k in PILOT_TEMPLATE_ORDER},
    t_h_levels=["none", "3", "5", "7", "9"],
    p_levels=[0.5, 0.7, 0.9],
    replicates_per_cell=2,
    n_trajectories=240,
    trajectory_order="index = enumerate over template (TEMPLATE_ORDER) -> t_h level -> p level -> replicate (0, 1)",
    seed_rule="seed = 190000 + trajectory index (0..239): unique per trajectory, inside the PILOT range, disjoint from "
              "every full-dataset role range",
    compliance="generate_compliance_vector('AILX_v1|PILOT|' + template, seed, p=p)",
    t_h_none_value=MAX_STEPS + 1,
    pre_decision_policy=dict(
        policy=lfd.POLICY_ID, epsilon=0.5,
        rng="numpy.random.default_rng([seed, int(round(p * 1000)), t_h_int, 7])",
        draw_order="at each decision tick: u = rng.random(); if u < 0.5: a = int(rng.integers(0, 3)) else a = "
                   "label_free_default(x); a is issued as a global action at that tick"),
    decision_states="every decision tick t (multiple of 5) with n_active > 0, captured BEFORE that tick's action",
    candidate_set="BASE2_298 (candidate_action_set.json, unchanged); all 298 rows stored per state; one rollout per "
                  "behavioral-equivalence class (hash of the resulting label map), copied to its members",
    continuation=lfd.POLICY_ID,
    unmet_measurement="after each tick's relabel, before its admission",
    storage="artifacts/ai_learning_experiment_design/pilot/pilot_rows.csv.gz",
    g3_recheck_states="the 40 pilot states with the most non-empty eligible zones; ties by trajectory index, then tick",
    g3_unrestricted_space="every absolute assignment {UNTOUCHED, FAVOR_A, PARITY, FAVOR_B} over the non-empty eligible "
                          "zones, same continuation and targets",
)
GO_CRITERIA = dict(
    G1="decision ticks >= 10: distinct admissible inputs >= 25% of those states AND >= 10 distinct per template",
    G2=">= 50% of states have >= 3 distinct target vectors among their 298 candidates",
    G3="BASE2_298 best == unrestricted best in >= 95% of the 40 re-check states AND mean T100 regret <= 0.1 tick",
    G4="independent regeneration reproduces every stored row exactly AND the hazard measurement equals the simulator "
       "tick_log in 100% of rollouts",
    G5="leakage re-audit on the rows passes: stored features are exactly the 26 admissible names; identical across a "
       "state's candidates; equal to the replayed pre-decision observation; unchanged by every candidate relabel; for "
       "every pre-onset hazard state, replaying the same history with no hazard gives identical features; residual "
       "capacity is 1 before onset and 2/6 after",
    G6="T100 irreducible share < 0.6 on post-onset rows, computed as SSW/SST over (features, candidate_id) groups "
       "that contain rows from >= 2 distinct trajectories; if such groups cover < 10% of post-onset rows the "
       "criterion is NOT EVALUABLE and counts as not passed",
    G7="every (template, t_h, p) pilot cell has 2 complete trajectories",
    X1="targets valid: no-hazard rows have hazard target 0; rows whose episode ends before onset have hazard target 0; "
       "future_T100 = T100_abs - t_d and future_T90 = T90_abs - t_d (or 0 with the reached flag) on every row; for "
       "every t_d > 0 state, pre-decision unmet + future unmet of the default candidate equals the unmet of a "
       "full-episode rerun from t=0",
    X2="zero leakage (= G5)",
    X3="every candidate rollout starts from a state whose fingerprint equals the recorded pre-decision state (100%)",
    X4="joint timing x compliance design valid: 4 TEST_PRIMARY_JOINT cells; test t_h and p levels disjoint from "
       "training levels; every template crosses every t_h and p; seed ranges disjoint across roles and from PILOT; "
       "compliance keys role-specific",
    X5=f"diversity materially exceeds the old dataset: distinct admissible inputs >= {10 * OLD['distinct_inputs']} "
       f"(10x old) AND distinct share >= {round(5 * OLD['distinct_inputs'] / OLD['states'], 4)} (5x old) AND distinct "
       f"(input, candidate) pairs >= {10 * OLD['distinct_input_action_pairs']} (10x old)",
    X6="sufficient target variation (= G2)",
    X7="no structural confound: every template has every (t_h, p) level (association 0); seeds unique per trajectory; "
       "realized perturbation rate within 0.5 +/- 0.1; no censored T100",
    X8="targets not dominated by unobservable history: pre-decision unmet is excluded (X1 decomposition check)",
    X9="candidate semantics consistent: fast label maps equal apply_differential_action label maps for every class "
       "representative; the default candidate BASE=l(s) equals applying l(s) globally with actions.apply_action",
    X10="continuation label-free: the generator and default module never reference the hidden heuristic; every "
        "continuation call receives a 26-feature vector; every row records the current continuation hash",
    rule="GO only if G1-G7 and X1-X10 all pass. Thresholds are fixed in this v2 specification before any pilot data "
         "exists and must not be relaxed afterwards.",
)


def _load(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def archive_v1() -> dict:
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    rec = {}
    for name in spec.OUTPUT_FILES:
        live = os.path.join(OUT_DIR, name)
        arch = os.path.join(ARCHIVE_DIR, name)
        if not os.path.exists(arch):
            text = _load(live)
            if V2_MARKER in text:
                raise RuntimeError(f"{name} is already v2 but no v1 archive exists")
            shutil.copyfile(live, arch)
        rec[name] = prior.sha256_file(arch)
    return rec


def _supersedes(name: str, sha: str) -> dict:
    return dict(version="v1", v1_marker=spec.MARKER, v1_sha256=sha,
                archived_copy=f"spec_versions/v1/{name}")


def build_v2(v1_hashes: dict, sufficiency_recheck: dict, div_lf: dict, pol_hash: str) -> dict:
    v1 = {n: json.loads(_load(os.path.join(ARCHIVE_DIR, n))) for n in CHANGED if n.endswith(".json")}
    out, changed_fields = {}, {}

    def stamp(name, d, fields):
        d = copy.deepcopy(d)
        d["audit"] = V2_MARKER
        d["spec_version"] = "v2"
        d["decisions_applied"] = ["D1", "D2"]
        d["supersedes"] = _supersedes(name, v1_hashes[name])
        out[name] = d
        changed_fields[name] = fields

    lfd_block = dict(LABEL_FREE_DEFAULT, policy_hash=pol_hash)

    t = copy.deepcopy(v1["target_definitions.json"])
    t["continuation"] = dict(lfd_block, rule="at every decision tick after t_d, identical for every candidate")
    hz = t["targets"]["future_hazard_edge_unmet_demand"]
    hz.pop("alternative", None)
    hz.pop("status", None)
    hz.pop("definition_recommended", None)
    hz["definition"] = ("sum over t in [max(t_d, t_h), end) of max(0, H_CE1 branch-A candidates(t) - cap_H_CE1(t)), "
                        "measured after tick t's relabel and before its admission; 0 before onset; 0 in no-hazard "
                        "scenarios and whenever the episode ends before onset")
    hz["decision"] = "D1 (approved)"
    t["stored_per_row"] = [x if not x.startswith("future_hazard_edge_unmet_demand") else
                           "future_hazard_edge_unmet_demand (D1 definition only)" for x in t["stored_per_row"]]
    t["stored_per_row"].append("continuation policy id + hash")
    t["changes_vs_existing_labels"].append("continuation: hidden-label heuristic -> label-free default (D2)")
    stamp("target_definitions.json", t, ["continuation", "targets.future_hazard_edge_unmet_demand",
                                         "stored_per_row", "changes_vs_existing_labels"])

    s = copy.deepcopy(v1["selector_design.json"])
    s["inputs"] = s["inputs"].replace("h = the heuristic's global action",
                                      f"h = the label-free default's action l(s) ({lfd.POLICY_ID})")
    s["default_and_fallbacks"] = dict(
        default="d(s) = BASE=l(s), l = " + lfd.POLICY_ID, selector_fallback="d(s)", abstention_fallback="d(s)",
        out_of_distribution_fallback="d(s)", safety_floor_reference="d(s)", label_free_default=lfd_block)
    s.pop("caveat", None)
    s["never_uses"] = sorted(set(s["never_uses"]) | {"the hidden-label GLOBAL_DEADBAND heuristic (reference arm P2b only)"})
    stamp("selector_design.json", s, ["inputs", "default_and_fallbacks", "caveat (removed)", "never_uses"])

    c = copy.deepcopy(v1["closed_loop_evaluation.json"])
    p = c["policies"]
    p["P2_label_free_default"] = f"{lfd.POLICY_ID} at every decision (primary comparator)"
    p["P2b_hidden_label_heuristic_REFERENCE"] = (p.pop("P2_capacity_aware_heuristic") + " -- REFERENCE ARM ONLY: reads "
                                                 "hidden Q_A/Q_B; reported, never a gate, never a fallback")
    p["P4_ai_assisted"] = "selector_design over the trained model at every decision; d(s) = BASE=l(s) (label-free)"
    p["P5_full_information_oracle"] = p["P5_full_information_oracle"].replace("heuristic continuation",
                                                                              "label-free continuation")
    c["policies"] = dict(sorted(p.items()))
    c["protocol"] = c["protocol"].replace("paired comparisons against P2.", "paired comparisons against P2 "
                                          "(label-free default); P2b reported alongside.")
    c["primary_endpoint"] = c["primary_endpoint"].replace("P4 vs P2", "P4 vs P2 (label-free default)")
    c["caveat"] = c["caveat"].replace("heuristic continuation", "label-free continuation").replace(
        "Q^heuristic", "Q^label-free")
    stamp("closed_loop_evaluation.json", c, ["policies", "protocol", "primary_endpoint", "caveat"])

    r = copy.deepcopy(v1["preregistered_success_criteria.json"])
    r["status"] = "FROZEN (v2, D1 and D2 applied); record this file's SHA-256 before Phase 3"
    r["outcome_prediction"]["baselines"] = [b.replace("heuristic-outcome-as-prediction", "default-outcome-as-"
                                                      "prediction (label-free d(s))") for b in r["outcome_prediction"]["baselines"]]
    r["safety"]["harm_reference"] = "d(s) = BASE=l(s), the label-free default"
    r["closed_loop"]["criterion"] = r["closed_loop"]["criterion"].replace("than P2", "than P2 (label-free default)")
    r["closed_loop"]["reference_arm"] = "P2b hidden-label heuristic: reported for context, never a gate"
    stamp("preregistered_success_criteria.json", r, ["status", "outcome_prediction.baselines", "safety.harm_reference",
                                                     "closed_loop.criterion", "closed_loop.reference_arm"])

    d = copy.deepcopy(v1["dataset_design.json"])
    d["factors"]["pre_decision_policy"] = (f"{lfd.POLICY_ID} (label-free default), epsilon-perturbed (epsilon=0.5: a "
                                           "uniformly random GLOBAL action at each pre-decision tick) with a seeded RNG "
                                           "keyed on (seed, p, t_h); history stored per row")
    d["continuation"] = lfd_block
    d["phases"]["PHASE_1_PILOT"] = dict(d["phases"]["PHASE_1_PILOT"], exact_specification=PILOT)
    stamp("dataset_design.json", d, ["factors.pre_decision_policy", "continuation", "phases.PHASE_1_PILOT"])

    q = copy.deepcopy(v1["diversity_requirements.json"])
    q["design_time_count_label_free_pre_decision"] = div_lf
    q["go_no_go_after_pilot"] = GO_CRITERIA
    q["old_dataset_reference_values"] = OLD
    stamp("diversity_requirements.json", q, ["design_time_count_label_free_pre_decision",
                                             "go_no_go_after_pilot (G6 computation made exact; X1-X10 added)",
                                             "old_dataset_reference_values"])

    sc = copy.deepcopy(v1["scenario_space.json"])
    sc["dimensions"]["E_initial_congestion"]["indirect_sources"] = [
        "later decision ticks (5, 10, 15, ...)",
        f"epsilon-perturbed pre-decision history (epsilon=0.5, a uniformly random GLOBAL action at each pre-decision "
        f"tick) around the label-free default {lfd.POLICY_ID}"]
    stamp("scenario_space.json", sc, ["dimensions.E_initial_congestion.indirect_sources"])

    sp = copy.deepcopy(v1["split_specification.json"])
    sp["final_go_no_go"] = dict(classification="READY FOR PHASE 1 PILOT", resolved=["D1 (approved)", "D2 (decided: "
                                f"{lfd.POLICY_ID})"], previous=v1["split_specification.json"]["final_go_no_go"]["classification"])
    sp["pilot_role"] = "PILOT: seeds 190000-190239, keys 'AILX_v1|PILOT|<template>'; discarded after Phase 2"
    stamp("split_specification.json", sp, ["final_go_no_go", "pilot_role"])

    out["FINAL_EXPERIMENT_SPECIFICATION.md"] = render_md(v1_hashes, sufficiency_recheck, div_lf, pol_hash)
    changed_fields["FINAL_EXPERIMENT_SPECIFICATION.md"] = ["rewritten as v2 (v1 archived)"]
    return out, changed_fields


def render_md(v1_hashes, suff, div_lf, pol_hash) -> str:
    L = [f"<!-- {V2_MARKER} -->", f"<!-- supersedes {spec.MARKER} (archived: spec_versions/v1/) -->",
         "# Final Experiment Specification, v2 (D1 and D2 applied)\n",
         "v1 is archived byte-for-byte in `spec_versions/v1/`; `spec_version_record.json` lists every file's v1 and "
         "v2 SHA-256 and the fields that changed. Nothing in v2 changes the simulator, observation.py, the RL branch "
         "or the old 9,450-row dataset.\n",
         "## Status: READY FOR PHASE 1 PILOT\n",
         "## Decisions applied\n",
         "- **D1 (approved)**: future hazard-edge unmet demand counts only ticks t >= t_h (actual onset); 0 before "
         "onset; 0 in no-hazard scenarios.",
         f"- **D2 (decided)**: continuation, selector / abstention / OOD fallback, closed-loop default and pre-decision "
         f"generation all use the label-free `{lfd.POLICY_ID}` (hash `{pol_hash[:16]}`): PARITY by default, FAVOR_B "
         "once the hazard has actually started and anyone waits at the hub. Inputs: the 26 admissible features only. "
         "The hidden-label heuristic is reference arm P2b only.\n",
         "## Preserved exactly\n",
         "- Candidate set BASE2_298 (`candidate_action_set.json` byte-identical to v1).",
         "- 26-feature admissible input (`final_feature_manifest.json` byte-identical to v1).",
         "- Scenario grid, joint timing x compliance primary test, trajectory-level split, 240-trajectory pilot.\n",
         "## Evidence re-checked under the label-free continuation\n",
         f"BASE2_298 vs the unrestricted absolute space on the existing 530 states with the label-free continuation: "
         f"{suff['per_candidate_set']['BASE2_298']['share_states_where_set_best_equals_unrestricted_best']:.1%} best-match, "
         f"mean T100 regret {suff['per_candidate_set']['BASE2_298']['mean_T100_regret_ticks']}, max "
         f"{suff['per_candidate_set']['BASE2_298']['max_T100_regret_ticks']} (v1 under the hidden heuristic: 100%, 0).\n",
         "Design-time diversity with the label-free pre-decision policy: " + "; ".join(
             f"{k}: {v['n_distinct_admissible_inputs']} distinct inputs, tick>=10 distinct share "
             f"{v['distinct_share_tick_ge_10']:.1%}" for k, v in div_lf.items()) + ".\n",
         "## Phase 1 pilot (pre-registered)\n",
         f"{PILOT['n_trajectories']} trajectories = 8 templates x t_h {PILOT['t_h_levels']} x p {PILOT['p_levels']} x 2 "
         f"replicates; {PILOT['seed_rule']}; compliance {PILOT['compliance']}; pre-decision: {lfd.POLICY_ID} with "
         f"epsilon 0.5 ({PILOT['pre_decision_policy']['draw_order']}). The pilot uses only TRAIN-role t_h and p levels "
         "by design; the joint test is verified at design level (X4), and the pilot is discarded after Phase 2.\n",
         "## Pilot go / no-go (fixed before any pilot data exists)\n"]
    L += [f"- **{k}**: {v}" for k, v in GO_CRITERIA.items()]
    return "\n".join(L) + "\n"


def recheck_sufficiency_label_free() -> dict:
    """Re-runs the v1 candidate-set sufficiency analysis with the label-free continuation (monkeypatched only inside
    this process; the spec module file is not modified)."""
    saved = spec.HEURISTIC
    spec.HEURISTIC = lambda obs: lfd.label_free_default_action(
        prior.admissible_vector(dict(obs27=__import__("numpy").asarray(obs, dtype=float)), lfd.ADMISSIBLE_FEATURES))
    try:
        states = spec.existing_states()
        return spec.candidate_sufficiency(states)
    finally:
        spec.HEURISTIC = saved


def run() -> dict:
    before = lfa.protected_hashes()
    v1_hashes = archive_v1()
    pol_hash = lfd.policy_hash()
    t0 = time.perf_counter()
    suff = recheck_sufficiency_label_free()
    print(f"BASE2 under label-free continuation: {suff['per_candidate_set']['BASE2_298']['share_states_where_set_best_equals_unrestricted_best']}")
    div_lf = lfa.diversity_with_label_free(lambda obs: lfd.label_free_default_action(
        prior.admissible_vector(dict(obs27=__import__("numpy").asarray(obs, dtype=float)), lfd.ADMISSIBLE_FEATURES)))
    out, fields = build_v2(v1_hashes, suff, div_lf, pol_hash)
    for name, content in out.items():
        text = content if isinstance(content, str) else json.dumps(content, indent=2, default=prior._json_default)
        with open(os.path.join(OUT_DIR, name), "w", encoding="utf-8") as f:
            f.write(text)
    record = dict(
        audit=V2_MARKER, spec_version="v2", decisions=dict(D1="approved", D2=f"decided: {lfd.POLICY_ID}"),
        label_free_default=dict(LABEL_FREE_DEFAULT, policy_hash=pol_hash),
        files={n: dict(changed=(n in CHANGED), v1_sha256=v1_hashes[n],
                       v2_sha256=prior.sha256_file(os.path.join(OUT_DIR, n)),
                       fields_changed=fields.get(n, [])) for n in spec.OUTPUT_FILES},
        unchanged_verified={n: v1_hashes[n] == prior.sha256_file(os.path.join(OUT_DIR, n)) for n in UNCHANGED},
        base2_recheck_label_free={k: v for k, v in suff.items() if k != "per_candidate_evidence"},
        pilot_exact_specification=PILOT, go_no_go=GO_CRITERIA, seconds=round(time.perf_counter() - t0, 1),
    )
    with open(os.path.join(OUT_DIR, "spec_version_record.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, default=prior._json_default)
    after = lfa.protected_hashes()
    changed = [k for k in before if before[k] != after[k] and not any(k.endswith("/" + n) for n in CHANGED)]
    if changed:
        raise RuntimeError(f"unexpected protected-file change: {changed}")
    return record


if __name__ == "__main__":
    rec = run()
    for n, v in rec["files"].items():
        print(f"{'CHANGED  ' if v['changed'] else 'unchanged'} {n} {v['v1_sha256'][:12]} -> {v['v2_sha256'][:12]}")
