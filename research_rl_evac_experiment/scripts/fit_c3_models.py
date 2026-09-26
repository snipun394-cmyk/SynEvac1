"""TRAIN-only fitting of the 16 frozen HistGradientBoostingRegressor models (A2 + C1 + C2 + C3).

Reads only TRAIN rows (for all other rows only split_role is inspected; the
file is one gzip stream, so their bytes are decompressed but no other field
is accessed). No VALIDATION or PRIMARY TEST value is loaded. Saves 16 pickled
models (protocol 5) and an auditable manifest under models/c3_hgb/. Computes
no validation metric, builds no selector.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import pickle
import platform
import sys
import time

import numpy as np

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)
import audit_pre_training_ai_dataset as prior  # noqa: E402

import sklearn  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

ART = prior.OUT_DIR
AMD = os.path.join(ART, "amendments")
ROWS = os.path.join(ART, "full_dataset", "full_ai_learning_rows.csv.gz")
C3_PATH = os.path.join(AMD, "amendment_A2_clarification_C3_model_specification.json")
OUT_DIR = os.path.join(ART, "models", "c3_hgb")
TARGETS = ["future_T90", "future_T100", "future_unmet", "future_hazard_unmet"]
OUTPUTS = {"mean": dict(loss="squared_error"), "q0.1": dict(loss="quantile", quantile=0.1),
           "q0.5": dict(loss="quantile", quantile=0.5), "q0.9": dict(loss="quantile", quantile=0.9)}
EXPECTED_TRAIN_ROWS = 1783530


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def protected() -> dict:
    files = ["full_dataset/full_ai_learning_rows.csv.gz", "spec_version_record.json", "candidate_action_set.json",
             "final_feature_manifest.json", "selector_design.json", "preregistered_success_criteria.json"]
    files += [f"amendments/{n}" for n in (
        "amendment_A1_acceptance_criteria.json", "amendment_A1_evaluation.json", "amendment_A2_evaluation_protocol.json",
        "AMENDMENT_A2_EVALUATION_PROTOCOL.md", "amendment_A2_approval.json", "amendment_A2_reference_run.json",
        "amendment_A2_clarification_C1.json", "amendment_A2_clarification_C2.json",
        "amendment_A2_clarification_C3_model_specification.json", "issue2_validation_baseline_selection.json")]
    return {f: prior.sha256_file(os.path.join(ART, f)) for f in files}


def main():
    t0 = time.perf_counter()
    before = protected()
    c3 = json.load(open(C3_PATH, encoding="utf-8"))
    if (sklearn.__version__, np.__version__, platform.python_version()) != (
            c3["environment"]["scikit_learn"], c3["environment"]["numpy"], c3["environment"]["python"]):
        raise RuntimeError("environment differs from C3")
    order = c3["inputs"]["order"]
    assert len(order) == 54 and len(set(order)) == 54
    state_cols = order[:26]
    cas = json.load(open(os.path.join(ART, "candidate_action_set.json"), encoding="utf-8"))
    zones, values = ["R1", "R2", "C1", "R3", "R4", "C2", "H"], ["UNTOUCHED", "FAVOR_A", "PARITY", "FAVOR_B"]
    assert order[26:] == [f"cand_{z}_{v}" for z in zones for v in values]
    desc = np.zeros((298, 28))
    for c in cas["candidates"]:
        for zi, z in enumerate(zones):
            desc[c["id"], zi * 4 + values.index(c["encoding"][z])] = 1.0
    assert (desc.sum(axis=1) == 7).all()

    X = np.empty((EXPECTED_TRAIN_ROWS, 54), dtype=np.float64)
    Y = np.empty((EXPECTED_TRAIN_ROWS, 4), dtype=np.float64)
    n, other = 0, {}
    with gzip.open(ROWS, "rt", encoding="utf-8") as f:
        r = csv.reader(f)
        h = next(r)
        for col in state_cols + TARGETS + ["candidate_id", "split_role"]:
            if h.count(col) != 1:
                raise RuntimeError(f"column {col} missing or duplicated in the dataset header")
        ix = {c: i for i, c in enumerate(h)}
        irole = ix["split_role"]
        sidx = [ix[c] for c in state_cols]
        tidx = [ix[t] for t in TARGETS]
        icand = ix["candidate_id"]
        for row in r:
            role = row[irole]
            if role != "TRAIN":
                other[role] = other.get(role, 0) + 1
                continue
            X[n, :26] = [float(row[i]) for i in sidx]
            X[n, 26:] = desc[int(row[icand])]
            Y[n] = [float(row[i]) for i in tidx]
            n += 1
    if n != EXPECTED_TRAIN_ROWS:
        raise RuntimeError(f"TRAIN rows {n} != {EXPECTED_TRAIN_ROWS}")
    if not np.isfinite(X).all() or not np.isfinite(Y).all():
        raise RuntimeError("non-finite value in TRAIN inputs or targets")
    x_hash = sha(np.ascontiguousarray(X).tobytes())
    y_hash = {t: sha(np.ascontiguousarray(Y[:, k]).tobytes()) for k, t in enumerate(TARGETS)}
    order_hash = sha(json.dumps(order).encode())
    print(f"loaded TRAIN rows {n}; other roles skipped by split_role only: {other}", flush=True)

    base = c3["configuration"]["user_frozen"]
    os.makedirs(OUT_DIR, exist_ok=True)
    models, diag = {}, {}
    sample = X[:2980]   # first 10 TRAIN states (TRAIN-only diagnostic sample)
    for t_i, t in enumerate(TARGETS):
        for out, kw in OUTPUTS.items():
            name = f"{t}__{out}"
            est = HistGradientBoostingRegressor(**base, **kw)
            params = est.get_params()
            ts = time.perf_counter()
            est.fit(X, Y[:, t_i])
            fit_s = round(time.perf_counter() - ts, 1)
            blob = pickle.dumps(est, protocol=5)
            path = os.path.join(OUT_DIR, f"{name}.pkl")
            with open(path, "wb") as fh:
                fh.write(blob)
            pred = est.predict(sample)
            models[name] = dict(
                target=t, output=out, effective_parameters={k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                                                           for k, v in params.items()},
                n_train_rows=n, n_iter=int(est.n_iter_), input_order_sha256=order_hash, input_matrix_sha256=x_hash,
                target_vector_sha256=y_hash[t], model_file=f"{name}.pkl", model_sha256=sha(blob), pickle_protocol=5,
                seed=base["random_state"], fit_seconds=fit_s)
            diag[name] = dict(finite=bool(np.isfinite(pred).all()), min=float(pred.min()), max=float(pred.max()))
            print(f"fitted {name} in {fit_s}s n_iter={est.n_iter_}", flush=True)

    after = protected()
    changed = [k for k in before if before[k] != after[k]]
    if changed:
        raise RuntimeError(f"PROTECTED FILE CHANGED: {changed}")
    manifest = dict(
        record="C3 TRAIN-only fitting of the 16 frozen HistGradientBoostingRegressor models",
        run_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        c3_sha256=before["amendments/amendment_A2_clarification_C3_model_specification.json"],
        environment=dict(python=platform.python_version(), scikit_learn=sklearn.__version__, numpy=np.__version__,
                         omp_num_threads=os.environ.get("OMP_NUM_THREADS", "unset (library default)")),
        training_data=dict(split="TRAIN", rows=n, equal_row_weight=True, sample_weight_passed=False,
                           internal_validation_split=False, early_stopping=False,
                           other_roles_seen_only_as_split_role=other, input_columns=order,
                           input_order_sha256=order_hash, input_matrix_sha256=x_hash, target_vector_sha256=y_hash),
        fit_status=f"{len(models)}/16 fitted", models=models,
        post_fit_diagnostic=dict(sample="first 2,980 TRAIN rows (10 TRAIN states); TRAIN only", predictions=diag,
                                 all_finite=all(d["finite"] for d in diag.values())),
        not_done=["no VALIDATION evaluation", "no validation losses", "no learned-selector tolerances",
                  "no selector", "no PRIMARY TEST", "no learned-vs-baseline comparison"],
        integrity=dict(protected_files_unchanged=True, protected_hashes=before),
        seconds=round(time.perf_counter() - t0, 1))
    with open(os.path.join(OUT_DIR, "training_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(json.dumps(dict(fit_status=manifest["fit_status"], rows=n, x=x_hash, y=y_hash, all_finite=manifest["post_fit_diagnostic"]["all_finite"],
                          seconds=manifest["seconds"]), indent=1), flush=True)


if __name__ == "__main__":
    main()
