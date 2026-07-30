from typing import Any, Dict, Tuple


# =====================================================
# Predictive Model Development & Benchmark Campaign milestone, Phase 2 --
# the model candidate roster. Every name here must already resolve via
# ai_training.models.base.build_classifier/build_regressor (this
# milestone extended that factory with decision_tree/logistic_regression/
# linear_regression/mlp/dummy -- exactly the same factory pattern
# random_forest/gradient_boosting/xgboost already used, never a second,
# parallel model-construction mechanism).
#
# LightGBM is deliberately EXCLUDED: it is not an installed dependency
# in this environment (confirmed: `import lightgbm` raises
# ModuleNotFoundError), and the milestone brief explicitly says "Do not
# introduce external dependencies solely for this milestone." XGBoost,
# by contrast, is already an existing project dependency (ai_training,
# predictive_model, both pre-existing) -- included.
# =====================================================

EXCLUDED_ALGORITHMS = {
    "lightgbm": "not an installed dependency in this environment; milestone brief forbids adding "
                 "new external dependencies solely for this benchmark.",
}

CLASSIFICATION_ALGORITHMS: Tuple[str, ...] = (
    "random_forest", "gradient_boosting", "xgboost", "decision_tree",
    "logistic_regression", "mlp", "dummy",
)

REGRESSION_ALGORITHMS: Tuple[str, ...] = (
    "random_forest", "gradient_boosting", "xgboost", "decision_tree",
    "linear_regression", "mlp", "dummy",
)

# Small, deterministic hyperparameter grids -- Phase 3's own "ensure all
# searches are deterministic" requirement is satisfied by (a) every
# estimator already defaulting random_state=0 in the factory, (b) every
# grid enumerated here being a fixed, explicit list (never a randomly
# sampled distribution), and (c) GroupKFold (model_benchmark/search.py)
# never shuffling. Kept intentionally small (2-4 combinations per
# algorithm) so the full 7-algorithm x 2-target search completes in
# minutes, not hours, on this environment's hardware -- still a genuine
# grid search, not a single fixed configuration dressed up as one.
CLASSIFICATION_PARAM_GRIDS: Dict[str, Dict[str, Any]] = {
    "random_forest": {"n_estimators": [100, 300], "max_depth": [None, 10]},
    "gradient_boosting": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1]},
    "xgboost": {"n_estimators": [100, 300], "max_depth": [3, 6]},
    "decision_tree": {"max_depth": [None, 5, 10]},
    "logistic_regression": {"C": [0.1, 1.0, 10.0]},
    "mlp": {"hidden_layer_sizes": [(32,), (64, 32)]},
    "dummy": {},  # control model -- no hyperparameters to search
}

REGRESSION_PARAM_GRIDS: Dict[str, Dict[str, Any]] = {
    "random_forest": {"n_estimators": [100, 300], "max_depth": [None, 10]},
    "gradient_boosting": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1]},
    "xgboost": {"n_estimators": [100, 300], "max_depth": [3, 6]},
    "decision_tree": {"max_depth": [None, 5, 10]},
    "linear_regression": {},  # no meaningful hyperparameters to search
    "mlp": {"hidden_layer_sizes": [(32,), (64, 32)]},
    "dummy": {},  # control model -- no hyperparameters to search
}


def display_name(algorithm: str) -> str:

    return {
        "random_forest": "Random Forest",
        "gradient_boosting": "Gradient Boosting",
        "xgboost": "XGBoost",
        "decision_tree": "Decision Tree",
        "logistic_regression": "Logistic Regression",
        "linear_regression": "Linear Regression",
        "mlp": "MLP Neural Network",
        "dummy": "Dummy (control)",
    }.get(algorithm, algorithm)
