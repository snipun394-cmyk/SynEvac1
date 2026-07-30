import time as _time

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Protocol, Tuple

from building_state.models import BuildingState

from ai_registry.inference_service import (
    BottleneckOccurrencePrediction,
    EvacuationTimePrediction,
    InferenceUnavailableError,
    LiveAIInferenceService,
)
from ai_registry.metadata import Deployability


# =====================================================
# Live AI Inference Runtime Integration milestone -- the seam
# LiveOrchestrator uses to reach the already-built, already-registry-
# served AI inference infrastructure (ai_registry/, ai_features/),
# mirroring live_system.building_state_gateway's own established
# "thin Protocol + real adapter, orchestrator never constructs the
# underlying package itself" shape exactly. LiveOrchestrator holds only
# a LiveAIInferenceGateway -- never a ModelRegistry, a trained model, a
# TrainingDataset, or a feature extractor directly; composing an
# ai_registry.LiveAIInferenceService (and the ModelRegistry it wraps) is
# a caller's job, the same way composing CameraManager/SensorManager is
# a building_state_gateway caller's job.
#
# This was originally a SEPARATE, newly-added slot on LiveOrchestrator,
# alongside the older live_system.integration's own Phase-7 inference
# seam (operating on LiveBuildingSnapshot via an injected
# feature_row_builder that was never implemented in production -- see
# docs/architecture/live_system_integration_audit.md §5). That older
# seam has since been RETIRED entirely by the Shadow-Mode Predictive AI
# Integration milestone (see live_system/integration.py's own docstring)
# -- this gateway is now the ONLY inference seam on LiveOrchestrator,
# operating on the canonical BuildingState specifically, via the proven
# ai_registry/ai_features pipeline.
#
# Only BottleneckOccurrenceModel_LiveCompatible (PRODUCTION_CANDIDATE)
# is treated as an operational signal. EvacuationTimeModel_
# LiveCompatible (EXPERIMENTAL) is surfaced only under the field name
# `evacuation_time_experimental` -- never "predicted RSET" anywhere in
# this module -- and this milestone stops before any code path could
# let it reach Decision Policy or Advisory System (neither package is
# touched here at all).
# =====================================================


class AISystemStatus(Enum):

    AVAILABLE = auto()      # the operational (bottleneck) prediction succeeded
    PARTIAL = auto()        # bottleneck succeeded, something else (e.g. the
                             # experimental evacuation-time model) did not
    UNAVAILABLE = auto()    # no compatible model is currently registered
    INCOMPATIBLE = auto()   # a registered model's feature contract does not
                             # match the canonical schema BuildingState produces
    ERROR = auto()          # an unexpected exception (e.g. a corrupted model
                             # artifact) -- never allowed to propagate out of
                             # this gateway and crash the live cycle


@dataclass(frozen=True)
class LiveAIPredictionSnapshot:

    # One live cycle's complete AI output -- Phase 2's own required
    # shape. Immutable, same "replace, never mutate" convention every
    # other snapshot type in this codebase already follows.
    #
    # bottleneck is the one PRODUCTION_CANDIDATE, operationally-usable
    # signal this milestone produces. evacuation_time_experimental is
    # named the way it is deliberately -- see this module's own
    # docstring -- so a future reader can never mistake it for an
    # authoritative RSET without reading past its own field name.
    #
    # `bottleneck.probability`/`evacuation_time_experimental.
    # uncertainty_seconds` remain exactly what ai_registry.
    # inference_service already defined them as -- MODEL probability/
    # uncertainty, never recommendation confidence. Nothing in this
    # module renames, rescales, or reinterprets either value.

    timestamp: float
    building_state_timestamp: Optional[float]
    feature_schema_version: Optional[str]

    system_status: AISystemStatus

    bottleneck: Optional[BottleneckOccurrencePrediction] = None
    evacuation_time_experimental: Optional[EvacuationTimePrediction] = None

    errors: Tuple[str, ...] = field(default_factory=tuple)
    warnings: Tuple[str, ...] = field(default_factory=tuple)

    # Shadow-Mode Predictive AI Integration milestone, Phase 6 -- pure
    # instrumentation (wall-clock time spent inside predict() below),
    # never a change to WHAT is predicted or HOW -- added as a new,
    # defaulted field only, so every existing construction of this
    # (frozen) dataclass keeps working unchanged. None means "not
    # measured" (e.g. a hand-built test snapshot, or the state=None
    # early-return path above, which has no inference work to time),
    # never a fabricated 0.0.
    inference_duration_seconds: Optional[float] = None


# =====================================================


class LiveAIInferenceGateway(Protocol):

    # Same "thin Protocol, real adapter composes an already-public API"
    # shape as live_system.building_state_gateway.BuildingStateGateway.
    # Returning None means "skip this cycle, leave the previously stored
    # LiveAIPredictionSnapshot as-is" -- the throttling seam Phase 9
    # uses (see ThrottledLiveAIInferenceGateway below); it never means
    # "silently show a stale prediction as fresh" (see that class's own
    # docstring for why those are different things).

    def predict(self, state: Optional[BuildingState], time: float) -> Optional[LiveAIPredictionSnapshot]: ...


# =====================================================


class RegistryLiveAIInferenceGateway:

    # The real adapter -- composes an already-constructed ai_registry.
    # LiveAIInferenceService (itself wrapping an already-populated
    # ModelRegistry). Every failure mode Phase 7 names is caught here
    # and turned into an honest LiveAIPredictionSnapshot field, never
    # allowed to raise out of predict() and crash the live cycle --
    # Camera/Sensor/FACP/BuildingState/Voice Evacuation/Building
    # Control/LiveOrchestrator itself must never be affected by an AI
    # failure (Phase 7's own explicit requirement).

    def __init__(
        self,
        inference_service: LiveAIInferenceService,
        *,
        include_evacuation_time: bool = True,
    ):

        self._service = inference_service
        self._include_evacuation_time = include_evacuation_time

    # =====================================================

    def predict(self, state: Optional[BuildingState], time: float) -> LiveAIPredictionSnapshot:

        if state is None:

            return LiveAIPredictionSnapshot(
                timestamp=time,
                building_state_timestamp=None,
                feature_schema_version=None,
                system_status=AISystemStatus.UNAVAILABLE,
                errors=("no BuildingState available this cycle -- AI inference has nothing to run against",),
            )

        errors = []
        warnings = []

        started = _time.perf_counter()

        bottleneck = self._predict_bottleneck(state, time, errors)
        evacuation_time = (
            self._predict_evacuation_time_experimental(state, time, warnings)
            if self._include_evacuation_time else None
        )

        inference_duration_seconds = _time.perf_counter() - started

        feature_schema_version = (
            bottleneck.feature_schema_version if bottleneck is not None
            else evacuation_time.feature_schema_version if evacuation_time is not None
            else None
        )

        return LiveAIPredictionSnapshot(
            timestamp=time,
            building_state_timestamp=state.timestamp,
            feature_schema_version=feature_schema_version,
            system_status=self._resolve_status(bottleneck, errors, warnings),
            bottleneck=bottleneck,
            evacuation_time_experimental=evacuation_time,
            errors=tuple(errors),
            warnings=tuple(warnings),
            inference_duration_seconds=inference_duration_seconds,
        )

    # =====================================================

    def _bottleneck_model_registered_but_incompatible(self) -> bool:

        # Distinguishes "no bottleneck_occurrence model exists at all"
        # (UNAVAILABLE) from "one exists but failed its own compatibility
        # check" (INCOMPATIBLE) -- ai_registry.LiveAIInferenceService's
        # own predict_*() methods intentionally can't tell these apart
        # from the outside (ModelRegistry.get_latest_compatible_model()
        # hides *why* a candidate was excluded), so this queries the
        # registry directly via its own public `.registry` accessor.

        candidates = self._service.registry.list_models(
            model_type="bottleneck_occurrence", deployability=Deployability.LIVE_COMPATIBLE,
        )

        return len(candidates) > 0

    # =====================================================

    def _predict_bottleneck(self, state, time, errors) -> Optional[BottleneckOccurrencePrediction]:

        try:

            return self._service.predict_bottleneck_occurrence(state, timestamp=time)

        except InferenceUnavailableError as exc:

            errors.append(f"bottleneck_occurrence: {exc}")
            return None

        except Exception as exc:  # noqa: BLE001 -- a corrupted/broken model must never crash the live cycle

            errors.append(f"bottleneck_occurrence: unexpected error: {exc}")
            return None

    # =====================================================

    def _predict_evacuation_time_experimental(self, state, time, warnings) -> Optional[EvacuationTimePrediction]:

        # Deliberately appended to `warnings`, never `errors` -- the
        # experimental model being unavailable/incompatible must never
        # downgrade system_status away from AVAILABLE on its own (Phase
        # 5: it may appear in diagnostics; its absence is not an
        # operational AI failure).

        try:

            return self._service.predict_evacuation_time(state, timestamp=time)

        except InferenceUnavailableError as exc:

            warnings.append(f"evacuation_time_experimental: {exc}")
            return None

        except Exception as exc:  # noqa: BLE001

            warnings.append(f"evacuation_time_experimental: unexpected error: {exc}")
            return None

    # =====================================================

    def _resolve_status(self, bottleneck, errors, warnings) -> AISystemStatus:

        if bottleneck is not None:
            return AISystemStatus.AVAILABLE if not warnings else AISystemStatus.PARTIAL

        if not errors:
            return AISystemStatus.UNAVAILABLE

        message = errors[0]

        if "requires feature" in message or "unexpected key" in message:
            return AISystemStatus.INCOMPATIBLE

        if "No LIVE_COMPATIBLE" in message:

            return (
                AISystemStatus.INCOMPATIBLE if self._bottleneck_model_registered_but_incompatible()
                else AISystemStatus.UNAVAILABLE
            )

        return AISystemStatus.ERROR


# =====================================================


class ThrottledLiveAIInferenceGateway:

    # Phase 9's configurable-interval seam -- deliberately the simplest
    # possible shape (no threads, no async, no scheduler): wraps another
    # LiveAIInferenceGateway and only actually calls it once at least
    # min_interval_seconds has elapsed since the last real inference,
    # returning None otherwise. None is this Protocol's own documented
    # "skip this cycle, leave the previous snapshot as-is" signal --
    # StateManager's own component_timestamps["ai_prediction"] then
    # honestly keeps reporting when that snapshot was ACTUALLY computed
    # (never bumped to look fresher than it is), which is what keeps
    # this different from silently presenting a stale prediction as
    # current (Phase 8's own distinction -- see docs/architecture/
    # live_ai_runtime_integration.md for why every-cycle inference,
    # not this class, is this milestone's own chosen default).

    def __init__(self, inner: LiveAIInferenceGateway, *, min_interval_seconds: float):

        if min_interval_seconds <= 0:

            raise ValueError(
                f"min_interval_seconds must be > 0, got {min_interval_seconds!r} -- "
                f"use the inner gateway directly for every-cycle inference."
            )

        self._inner = inner
        self._min_interval_seconds = min_interval_seconds
        self._last_run_time: Optional[float] = None

    # =====================================================

    def predict(self, state: Optional[BuildingState], time: float) -> Optional[LiveAIPredictionSnapshot]:

        if self._last_run_time is not None and (time - self._last_run_time) < self._min_interval_seconds:
            return None

        self._last_run_time = time

        return self._inner.predict(state, time)
