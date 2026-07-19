import hashlib
import json
import os

from dataclasses import dataclass
from typing import Any, Dict, Optional

from ai_training.experiment import MODEL_REGISTRY


# =====================================================
# Phase 1 -- Model Loader. Loads exactly what
# ai_training.experiment.ExperimentRunner.save_result() wrote
# (model.joblib + manifest.json) -- never trains, never re-fits, never
# touches the simulation platform. Verifies the artifact is
# structurally complete and reconstructible BEFORE handing it back;
# rejects anything else with a clear IncompatibleModelError rather than
# letting a corrupted/foreign artifact fail unpredictably later, deep
# inside predictor.py/ensemble.py.
# =====================================================


class IncompatibleModelError(Exception):

    # Raised for any directory that is not a structurally valid,
    # reconstructible ai_training experiment artifact (missing files,
    # broken JSON, missing manifest keys, a model_name ai_inference
    # doesn't recognize, a corrupted joblib payload, or a successfully
    # deserialized model that is nonetheless missing something its own
    # type requires -- e.g. an ExitUsageModel without output_names).
    # Never raised merely because a model scores poorly -- that is a
    # modeling-quality concern, not a compatibility one.

    pass


# =====================================================


@dataclass(frozen=True)
class ModelProvenance:

    experiment_name: str
    model_name: str
    algorithm: str
    model_version: str
    generated_at: str
    train_size: int
    test_size: int
    metrics: Dict[str, Any]

    # Never fabricated -- None unless a caller of load_model() who
    # actually knows which training-dataset export produced this model
    # explicitly supplies one. ai_training's own experiment manifest
    # records no dataset identity of its own to read this from.
    dataset_version: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:

        return {
            "experiment_name": self.experiment_name,
            "model_name": self.model_name,
            "algorithm": self.algorithm,
            "model_version": self.model_version,
            "generated_at": self.generated_at,
            "train_size": self.train_size,
            "test_size": self.test_size,
            "metrics": self.metrics,
            "dataset_version": self.dataset_version,
        }


@dataclass(frozen=True)
class LoadedModel:

    model: Any  # an ai_training.models.base.BaseModel subclass instance, already fit
    provenance: ModelProvenance
    directory: str


# =====================================================


def load_model(directory: str, *, dataset_version: Optional[str] = None) -> LoadedModel:

    manifest_path = os.path.join(directory, "manifest.json")
    model_path = os.path.join(directory, "model.joblib")

    _verify_files_exist(directory, manifest_path, model_path)
    manifest = _read_manifest(manifest_path)
    _verify_manifest_shape(manifest, directory)

    model_name = manifest["config"]["model_name"]

    if model_name not in MODEL_REGISTRY:

        raise IncompatibleModelError(
            f"{directory}: manifest.json's config.model_name is {model_name!r}, which "
            f"ai_inference does not recognize -- known model names are {sorted(MODEL_REGISTRY)}"
        )

    model_cls, _task = MODEL_REGISTRY[model_name]

    try:
        model = model_cls.load(model_path)
    except IncompatibleModelError:
        raise
    except Exception as exc:

        # joblib/pickle can raise many different exception types for a
        # corrupted file or an artifact saved by an incompatible
        # version of ai_training (ModuleNotFoundError, AttributeError,
        # UnpicklingError, ...) -- all of them mean the same thing to a
        # caller: this directory is not a loadable model, wrapped into
        # one typed error rather than leaking an arbitrary pickle
        # traceback.
        raise IncompatibleModelError(
            f"{directory}: failed to deserialize model.joblib "
            f"({type(exc).__name__}: {exc})"
        ) from exc

    _verify_model_matches_manifest(model, model_name, directory)

    provenance = ModelProvenance(
        experiment_name=manifest["config"]["name"],
        model_name=model_name,
        algorithm=manifest["config"]["algorithm"],
        model_version=_compute_model_version(model_path),
        generated_at=manifest["generated_at"],
        train_size=manifest["train_size"],
        test_size=manifest["test_size_actual"],
        metrics=manifest["metrics"],
        dataset_version=dataset_version,
    )

    return LoadedModel(model=model, provenance=provenance, directory=directory)


# =====================================================


def _verify_files_exist(directory: str, manifest_path: str, model_path: str) -> None:

    if not os.path.isfile(manifest_path):
        raise IncompatibleModelError(
            f"{directory}: missing manifest.json -- not a valid ai_training experiment directory"
        )

    if not os.path.isfile(model_path):
        raise IncompatibleModelError(
            f"{directory}: missing model.joblib -- not a valid ai_training experiment directory"
        )


# =====================================================


def _read_manifest(manifest_path: str) -> Dict[str, Any]:

    try:

        with open(manifest_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    except json.JSONDecodeError as exc:

        raise IncompatibleModelError(f"{manifest_path}: not valid JSON ({exc})") from exc


# =====================================================

_REQUIRED_MANIFEST_KEYS = (
    "config", "metrics", "train_size", "test_size_actual", "generated_at", "model_metadata",
)

_REQUIRED_CONFIG_KEYS = (
    "name", "model_name", "algorithm", "algorithm_kwargs", "model_kwargs",
    "feature_set", "test_size", "val_size", "random_state",
)


def _verify_manifest_shape(manifest: Dict[str, Any], directory: str) -> None:

    missing = [key for key in _REQUIRED_MANIFEST_KEYS if key not in manifest]

    if missing:
        raise IncompatibleModelError(f"{directory}: manifest.json is missing required keys: {missing}")

    if not isinstance(manifest.get("config"), dict):
        raise IncompatibleModelError(f"{directory}: manifest.json's config is not an object")

    missing_config = [key for key in _REQUIRED_CONFIG_KEYS if key not in manifest["config"]]

    if missing_config:
        raise IncompatibleModelError(
            f"{directory}: manifest.json's config is missing required keys: {missing_config}"
        )


# =====================================================


def _verify_model_matches_manifest(model: Any, model_name: str, directory: str) -> None:

    if model.preprocessor is None or model.feature_schema is None:

        raise IncompatibleModelError(
            f"{directory}: loaded model has no fitted preprocessing pipeline -- "
            "the saved artifact is incomplete or corrupted"
        )

    if model_name == "exit_usage" and model.output_names is None:

        raise IncompatibleModelError(
            f"{directory}: exit_usage model is missing its output_names (exit-id vocabulary) "
            "-- incompatible artifact"
        )

    if model_name == "bottleneck" and not hasattr(model, "target"):

        raise IncompatibleModelError(
            f"{directory}: bottleneck model is missing its target "
            "('occurrence'/'location') -- incompatible artifact"
        )


# =====================================================


def _compute_model_version(model_path: str) -> str:

    # Content-addressed: identical model.joblib bytes always produce
    # the identical model_version, and any change to the saved
    # artifact (retraining, a different algorithm, ...) always
    # produces a different one. Same "sha256 of the artifact,
    # truncated" convention designer.campaign.campaign_controller.
    # derive_definition_id() already uses for ScenarioDefinition
    # identity -- restated independently here rather than imported,
    # since ai_inference must not depend on the Designer/Campaign
    # Studio layer.

    with open(model_path, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()

    return f"model-{digest[:16]}"
