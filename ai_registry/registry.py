from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from ai_inference.loader import LoadedModel, ModelProvenance

from ai_features.compatibility import CompatibilityIssue, CompatibilityReport, check_model_compatibility
from ai_features.feature_schema import CANONICAL_LIVE_FEATURE_NAMES, SCHEMA_VERSION

from ai_registry.metadata import Deployability, ModelMetadata, load_live_metadata, load_live_model_bundle, save_live_model


# =====================================================
# Phase 10's model registry. Discovers/holds trained model artifacts in
# memory (loaded once, cached -- see Phase 14's "models should not be
# reloaded from disk every cycle" requirement) and answers exactly the
# question this phase poses: "give me the latest LIVE_COMPATIBLE
# Evacuation Time model compatible with feature schema X" -- returning
# either a valid model or a clear "no compatible model", never a silent
# RESEARCH_ONLY fallback.
# =====================================================


class ResearchOnlyModelError(Exception):

    # Raised by get_model() when a caller asks for a specific model_id
    # that exists but is RESEARCH_ONLY -- get_model() never returns a
    # research-only model silently; a caller must explicitly know it
    # wants one and catch/expect this.

    pass


class ModelNotFoundError(Exception):

    pass


@dataclass(frozen=True)
class RegistryEntry:

    model: object  # a fitted ai_training.models.base.BaseModel subclass instance
    metadata: ModelMetadata
    source_directory: Optional[str] = None


class ModelRegistry:

    def __init__(self):

        self._entries: Dict[str, RegistryEntry] = {}

    # =====================================================
    # Registration
    # =====================================================

    def register_model(self, model, metadata: ModelMetadata, *, source_directory: Optional[str] = None) -> None:

        if metadata.model_id in self._entries:

            raise ValueError(
                f"A model is already registered under model_id {metadata.model_id!r} -- "
                f"model_id must be unique per trained artifact."
            )

        self._entries[metadata.model_id] = RegistryEntry(
            model=model, metadata=metadata, source_directory=source_directory,
        )

    # =====================================================

    def register_model_directory(self, directory: str) -> ModelMetadata:

        # Loads a directory saved via ai_registry.metadata.save_live_model()
        # -- the model bundle is loaded once, here, and cached in memory
        # for every subsequent lookup (Phase 14).

        metadata = load_live_metadata(directory)
        bundle = load_live_model_bundle(directory)

        model = _model_from_bundle(metadata.model_type, bundle)

        self.register_model(model, metadata, source_directory=directory)

        return metadata

    # =====================================================
    # Discovery
    # =====================================================

    def list_models(
        self,
        *,
        model_type: Optional[str] = None,
        deployability: Optional[Deployability] = None,
        feature_schema_version: Optional[str] = None,
        model_version: Optional[str] = None,
    ) -> Tuple[ModelMetadata, ...]:

        results = []

        for entry in self._entries.values():

            metadata = entry.metadata

            if model_type is not None and metadata.model_type != model_type:
                continue

            if deployability is not None and metadata.model_deployability != deployability:
                continue

            if feature_schema_version is not None and metadata.feature_schema_version != feature_schema_version:
                continue

            if model_version is not None and metadata.model_version != model_version:
                continue

            results.append(metadata)

        return tuple(sorted(results, key=lambda m: m.training_timestamp))

    # =====================================================

    def get_model(self, model_id: str, *, allow_research_only: bool = False):

        entry = self._entries.get(model_id)

        if entry is None:
            raise ModelNotFoundError(f"No model registered under model_id {model_id!r}.")

        if entry.metadata.model_deployability == Deployability.RESEARCH_ONLY and not allow_research_only:

            raise ResearchOnlyModelError(
                f"Model {model_id!r} is RESEARCH_ONLY -- refusing to hand it back through the "
                f"default (live-inference-facing) path. Pass allow_research_only=True if this "
                f"call site is genuinely a research workflow, never a live one."
            )

        return entry.model, entry.metadata

    # =====================================================

    def get_latest_compatible_model(
        self, model_type: str, *, feature_schema_version: str = SCHEMA_VERSION,
    ):

        # The one method a future live inference path should ever call.
        # Filters to LIVE_COMPATIBLE + the requested schema version, then
        # additionally re-validates via ai_features.compatibility (never
        # trusts the metadata's own self-declared deployability alone) --
        # returns the most-recently-trained match, or None. NEVER falls
        # back to a RESEARCH_ONLY or schema-incompatible model.

        candidates = self.list_models(
            model_type=model_type,
            deployability=Deployability.LIVE_COMPATIBLE,
            feature_schema_version=feature_schema_version,
        )

        for metadata in reversed(candidates):  # most recent (list_models sorts ascending) first

            report = self.validate_model_compatibility(metadata.model_id)

            if report.compatible:
                return self._entries[metadata.model_id].model, metadata

        return None

    # =====================================================

    def validate_model_compatibility(self, model_id: str) -> CompatibilityReport:

        entry = self._entries.get(model_id)

        if entry is None:
            raise ModelNotFoundError(f"No model registered under model_id {model_id!r}.")

        loaded = LoadedModel(
            model=entry.model,
            provenance=ModelProvenance(
                experiment_name=entry.metadata.model_id,
                model_name=entry.metadata.model_type,
                algorithm="unknown",
                model_version=entry.metadata.model_version,
                generated_at=entry.metadata.training_timestamp,
                train_size=0,
                test_size=0,
                metrics={"feature_schema_version": entry.metadata.feature_schema_version},
            ),
            directory=entry.source_directory or "in-memory",
        )

        report = check_model_compatibility(loaded)
        extra_issues = list(report.issues)

        # A model self-declaring RESEARCH_ONLY is never "compatible",
        # regardless of what the mechanical feature-column check finds --
        # deployability is a stronger, human-reviewed classification the
        # column-level check cannot see (e.g. BottleneckModel-location is
        # RESEARCH_ONLY for granularity reasons no column-presence check
        # would ever catch).
        if entry.metadata.model_deployability == Deployability.RESEARCH_ONLY:

            extra_issues.append(CompatibilityIssue(
                "research_only_deployability",
                f"{model_id!r} is explicitly marked RESEARCH_ONLY in its own metadata.",
            ))

        # A model's own recorded ordered_feature_names must still match
        # the CURRENTLY RUNNING canonical schema's order/membership
        # exactly when it claims the same feature_schema_version -- a
        # divergence here (e.g. schema fields silently reordered/renamed
        # without bumping SCHEMA_VERSION) is a real, distinct
        # incompatibility a plain column-set check would miss entirely.
        if (
            entry.metadata.feature_schema_version == SCHEMA_VERSION
            and entry.metadata.ordered_feature_names != CANONICAL_LIVE_FEATURE_NAMES
        ):

            extra_issues.append(CompatibilityIssue(
                "feature_ordering_mismatch",
                f"{model_id!r} was trained against ordered_feature_names "
                f"{entry.metadata.ordered_feature_names!r}, which no longer matches the "
                f"currently running CANONICAL_LIVE_FEATURE_NAMES {CANONICAL_LIVE_FEATURE_NAMES!r} "
                f"even though both claim feature_schema_version {SCHEMA_VERSION!r}.",
            ))

        return CompatibilityReport(
            model_name=report.model_name,
            compatible=not extra_issues,
            issues=tuple(extra_issues),
        )


# =====================================================


def _model_from_bundle(model_type: str, bundle):

    from ai_training.models.bottleneck_model import BottleneckModel
    from ai_training.models.evacuation_time_model import EvacuationTimeModel
    from ai_training.models.exit_usage_model import ExitUsageModel
    from ai_training.models.smoke_prediction_model import SmokePredictionModel

    model_cls = {
        "evacuation_time": EvacuationTimeModel,
        "bottleneck_occurrence": BottleneckModel,
        "bottleneck_location": BottleneckModel,
        "exit_usage": ExitUsageModel,
        "smoke_prediction": SmokePredictionModel,
    }.get(model_type)

    if model_cls is None:

        raise ValueError(f"Unknown model_type {model_type!r} -- cannot reconstruct a model instance.")

    return model_cls.from_bundle(bundle)
