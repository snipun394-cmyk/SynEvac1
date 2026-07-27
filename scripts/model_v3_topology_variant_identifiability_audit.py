"""Predictive Dataset V3 milestone, Phase 10 -- topology-family AND
structural-VARIANT identifiability check. Reuses the exact scenario-
level shuffled 80/20 split + XGBClassifier pattern scripts/
model_v3_1_topology_representation_audit.py already established for
family-only classification (V3.1 found 100% family accuracy from
features alone) -- this script runs BOTH a family classifier and a
finer-grained structural-VARIANT classifier over the SAME feature
matrix, to quantify whether structural diversity makes template
memorization harder (we do NOT require it to become unidentifiable,
Phase 10's own explicit instruction).

Usage: python scripts/model_v3_topology_variant_identifiability_audit.py <candidate_dataset_v3.csv>
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier

SEED = 20260727
N_JOBS = 2

FEATURE_COLUMNS = (
    "total_active_occupant_count", "candidate_capacity", "candidate_walking_distance",
    "candidate_adjacent_zone_occupancy", "candidate_queue_length", "candidate_approaching_count",
    "candidate_recent_flow_rate", "candidate_alternative_route_count",
)
CATEGORICAL_COLUMNS = ("candidate_type", "candidate_traversable", "candidate_congestion_level", "candidate_congestion_trend")


def _build_matrix(frame: pd.DataFrame) -> pd.DataFrame:

    X = frame[list(FEATURE_COLUMNS)].apply(pd.to_numeric, errors="coerce").fillna(-1.0)
    for col in CATEGORICAL_COLUMNS:
        dummies = pd.get_dummies(frame[col].astype(str), prefix=col)
        X = pd.concat([X.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
    return X


def _classify(frame: pd.DataFrame, label_col: str, rng: np.random.Generator) -> dict:

    unique_scenarios = frame["scenario_id"].unique()
    shuffled = rng.permutation(unique_scenarios)
    n_test = max(1, int(0.2 * len(shuffled)))
    test_scenarios = set(shuffled[:n_test])

    test_mask = frame["scenario_id"].isin(test_scenarios)
    train_frame, test_frame = frame[~test_mask], frame[test_mask]

    labels = sorted(frame[label_col].unique())
    label_to_int = {label: i for i, label in enumerate(labels)}

    X_train, X_test = _build_matrix(train_frame), _build_matrix(test_frame)
    X_train, X_test = X_train.align(X_test, join="outer", axis=1, fill_value=0)

    y_train = train_frame[label_col].map(label_to_int).to_numpy()
    y_test = test_frame[label_col].map(label_to_int).to_numpy()

    clf = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1, tree_method="hist",
        random_state=SEED, n_jobs=N_JOBS,
    )
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, pred)
    macro_f1 = f1_score(y_test, pred, average="macro")

    majority_label = train_frame[label_col].value_counts().idxmax()
    majority_baseline = (test_frame[label_col] == majority_label).mean()

    return {
        "label_column": label_col,
        "n_classes": len(labels),
        "train_scenarios": len(unique_scenarios) - len(test_scenarios),
        "test_scenarios": len(test_scenarios),
        "train_rows": len(train_frame),
        "test_rows": len(test_frame),
        "classifier_accuracy": accuracy,
        "macro_f1": macro_f1,
        "majority_class_baseline_accuracy": majority_baseline,
    }


def main() -> None:

    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if csv_path is None:
        raise SystemExit("Usage: model_v3_topology_variant_identifiability_audit.py <candidate_dataset_v3.csv>")

    frame = pd.read_csv(csv_path)
    rng = np.random.default_rng(SEED)

    family_result = _classify(frame, "topology_family", rng)
    rng = np.random.default_rng(SEED)  # same split seed for both -- comparable train/test scenario sets
    variant_result = _classify(frame, "structural_variant_id", rng)

    report = {
        "csv_path": str(csv_path),
        "row_count": len(frame),
        "phase10_family_classifier": family_result,
        "phase10_variant_classifier": variant_result,
    }

    out_path = csv_path.parent / "topology_variant_identifiability_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(json.dumps(report, indent=2, default=str))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
