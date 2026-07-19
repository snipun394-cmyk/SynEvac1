from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder

from ai_training.preprocessing import FeatureSchema, Preprocessor


# =====================================================
# Estimator factories -- "classical ML only" (Phase 2): every baseline
# model in this package picks its estimator from here by name, never
# by importing a specific sklearn/xgboost class directly. xgboost is
# genuinely optional: requesting it raises a clear, actionable
# ValueError if it isn't installed, rather than an ImportError at
# module import time (so importing ai_training.models never requires
# xgboost to be present).
# =====================================================


def build_regressor(algorithm: str, kwargs: Optional[Dict[str, Any]] = None):

    kwargs = dict(kwargs or {})

    if algorithm == "random_forest":

        from sklearn.ensemble import RandomForestRegressor

        kwargs.setdefault("n_estimators", 200)
        kwargs.setdefault("random_state", 0)

        return RandomForestRegressor(**kwargs)

    if algorithm == "gradient_boosting":

        from sklearn.ensemble import GradientBoostingRegressor

        kwargs.setdefault("random_state", 0)

        return GradientBoostingRegressor(**kwargs)

    if algorithm == "xgboost":

        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ValueError(
                "algorithm='xgboost' requires the xgboost package -- pip install xgboost, "
                "or choose 'random_forest'/'gradient_boosting' instead."
            ) from exc

        kwargs.setdefault("random_state", 0)

        return XGBRegressor(**kwargs)

    raise ValueError(f"Unknown regression algorithm: {algorithm!r}")


# =====================================================


def build_classifier(algorithm: str, kwargs: Optional[Dict[str, Any]] = None):

    kwargs = dict(kwargs or {})

    if algorithm == "random_forest":

        from sklearn.ensemble import RandomForestClassifier

        kwargs.setdefault("n_estimators", 200)
        kwargs.setdefault("random_state", 0)

        return RandomForestClassifier(**kwargs)

    if algorithm == "gradient_boosting":

        from sklearn.ensemble import GradientBoostingClassifier

        kwargs.setdefault("random_state", 0)

        return GradientBoostingClassifier(**kwargs)

    if algorithm == "xgboost":

        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ValueError(
                "algorithm='xgboost' requires the xgboost package -- pip install xgboost, "
                "or choose 'random_forest'/'gradient_boosting' instead."
            ) from exc

        kwargs.setdefault("random_state", 0)
        kwargs.setdefault("eval_metric", "logloss")

        return XGBClassifier(**kwargs)

    raise ValueError(f"Unknown classification algorithm: {algorithm!r}")


# =====================================================
# Phase 5 -- the one persisted unit. Bundling the estimator with the
# exact Preprocessor/FeatureSchema it was fit against means a saved
# model can never silently drift out of sync with its own
# preprocessing pipeline.
# =====================================================


@dataclass
class ModelBundle:

    estimator: Any
    preprocessor: Preprocessor
    feature_schema: FeatureSchema
    target_name: str
    label_classes: Optional[List[Any]] = None
    output_names: Optional[List[str]] = None
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


# =====================================================


class BaseModel:

    # Shared fit/predict/save/load scaffolding for every
    # ai_training/models/*.py baseline model. A subclass provides only:
    #   - target_name / is_classifier (class attributes)
    #   - default_estimator(self) -- which algorithm this model uses
    #   - build_table(dataset, ...) -- a @staticmethod deriving
    #     (X_rows, y, extra) from a TrainingDataset; this is model-
    #     specific label engineering and deliberately lives on the
    #     subclass, never here.

    target_name: str = ""
    is_classifier: bool = False

    def __init__(self, estimator: Any = None, config: Optional[Dict[str, Any]] = None):

        self.config: Dict[str, Any] = dict(config or {})
        self.estimator = estimator if estimator is not None else self.default_estimator()

        self.preprocessor: Optional[Preprocessor] = None
        self.feature_schema: Optional[FeatureSchema] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.output_names: Optional[List[str]] = None
        self.metadata: Dict[str, Any] = {}

    # =====================================================

    def default_estimator(self):

        raise NotImplementedError

    # =====================================================

    def fit(self, X_rows: Sequence[Dict[str, Any]], y: Sequence[Any]) -> "BaseModel":

        self.feature_schema = FeatureSchema.infer(X_rows)
        self.preprocessor = Preprocessor(self.feature_schema).fit(X_rows)

        X = self.preprocessor.transform(X_rows)
        y_fit = self._prepare_targets_for_fit(y)

        self.estimator.fit(X, y_fit)
        self.metadata["n_training_rows"] = len(X_rows)

        return self

    # =====================================================

    def predict(self, X_rows: Sequence[Dict[str, Any]]):

        if self.preprocessor is None:
            raise RuntimeError(f"{type(self).__name__}.fit() must be called before predict()")

        X = self.preprocessor.transform(X_rows)
        raw = self.estimator.predict(X)

        return self._decode_predictions(raw)

    # =====================================================

    def predict_proba(self, X_rows: Sequence[Dict[str, Any]]) -> np.ndarray:

        if not self.is_classifier:
            raise TypeError(f"{type(self).__name__} is not a classifier -- predict_proba() is unavailable")

        if self.preprocessor is None:
            raise RuntimeError(f"{type(self).__name__}.fit() must be called before predict_proba()")

        X = self.preprocessor.transform(X_rows)

        return self.estimator.predict_proba(X)

    # =====================================================

    def _prepare_targets_for_fit(self, y: Sequence[Any]):

        if not self.is_classifier:
            return np.asarray(y, dtype=float)

        self.label_encoder = LabelEncoder()

        # dtype left for numpy to infer (bool array for a list of pure
        # Python bools, string array for a list of strings) rather than
        # forced to dtype=object -- LabelEncoder.classes_ then keeps
        # that same natural dtype, so inverse_transform() in
        # _decode_predictions() below hands back predictions whose
        # dtype matches a caller's own y_true exactly (both "binary
        # bool", say), instead of an object-dtype array that sklearn's
        # metrics can no longer recognize as the same target type.
        return self.label_encoder.fit_transform(np.asarray(y))

    # =====================================================

    def _decode_predictions(self, raw: np.ndarray):

        if self.is_classifier and self.label_encoder is not None:
            return self.label_encoder.inverse_transform(np.asarray(raw).astype(int))

        return raw

    # =====================================================
    # Phase 5 -- persistence.
    # =====================================================

    def to_bundle(self) -> ModelBundle:

        if self.preprocessor is None:
            raise RuntimeError("Cannot persist an unfitted model -- call fit() first")

        return ModelBundle(
            estimator=self.estimator,
            preprocessor=self.preprocessor,
            feature_schema=self.feature_schema,
            target_name=self.target_name,
            label_classes=(
                list(self.label_encoder.classes_) if self.label_encoder is not None else None
            ),
            output_names=list(self.output_names) if self.output_names is not None else None,
            config=dict(self.config),
            metadata=dict(self.metadata),
        )

    # =====================================================

    def save(self, path: str) -> None:

        joblib.dump(self.to_bundle(), path)

    # =====================================================

    @classmethod
    def _construct_for_load(cls, bundle: ModelBundle) -> "BaseModel":

        # The only step subclasses with extra required constructor
        # arguments (e.g. BottleneckModel's target=) need to override --
        # everything else in from_bundle() below is common restoration
        # logic.

        return cls(estimator=bundle.estimator, config=bundle.config)

    @classmethod
    def from_bundle(cls, bundle: ModelBundle) -> "BaseModel":

        instance = cls._construct_for_load(bundle)

        instance.preprocessor = bundle.preprocessor
        instance.feature_schema = bundle.feature_schema
        instance.metadata = dict(bundle.metadata)
        instance.output_names = list(bundle.output_names) if bundle.output_names is not None else None

        if bundle.label_classes is not None:

            encoder = LabelEncoder()
            encoder.classes_ = np.asarray(bundle.label_classes, dtype=object)
            instance.label_encoder = encoder

        return instance

    @classmethod
    def load(cls, path: str) -> "BaseModel":

        bundle = joblib.load(path)

        return cls.from_bundle(bundle)
