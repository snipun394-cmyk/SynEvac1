from typing import Any, Mapping, Optional, Tuple

from prediction_evaluation.models import PredictionRecord


# =====================================================
# Prediction vs Reality Evaluation Framework milestone, Phase 1 -- the
# ONE place a PredictionRecord is created. Mirrors live_occupants.
# manager.LiveOccupantManager's own "immutable value objects, a mutable
# OWNING registry" convention: PredictionRegistry itself is a plain,
# append-only accumulator (never rewrites/removes a stored record --
# Phase 1's own "predictions must remain immutable" requirement, applied
# to the STORE as well as the individual record). Never runs inference,
# never touches ai_registry/live_system -- a caller (a script, a test,
# or a future Command Center evaluation panel this milestone does not
# build) has ALREADY produced a real prediction object and simply hands
# it here to be recorded.
# =====================================================


class PredictionRegistry:

    def __init__(self):

        self._records = []
        self._by_id = {}

    # =====================================================

    def record(
        self,
        *,
        timestamp: float,
        prediction_horizon_seconds: float,
        payload: Any,
        model_id: Optional[str] = None,
        model_version: Optional[str] = None,
        feature_schema_version: Optional[str] = None,
        source: str = "unknown",
        scenario_id: Optional[str] = None,
        building_id: Optional[str] = None,
        context_tags: Optional[Mapping[str, str]] = None,
    ) -> PredictionRecord:

        record = PredictionRecord(
            timestamp=timestamp,
            model_id=model_id,
            model_version=model_version,
            feature_schema_version=feature_schema_version,
            prediction_horizon_seconds=prediction_horizon_seconds,
            payload=payload,
            source=source,
            scenario_id=scenario_id,
            building_id=building_id,
            context_tags=context_tags or {},
        )

        self._records.append(record)
        self._by_id[record.prediction_id] = record

        return record

    # =====================================================

    def record_from_snapshot(
        self,
        snapshot,
        *,
        prediction_horizon_seconds: float,
        source: str = "unknown",
        scenario_id: Optional[str] = None,
        building_id: Optional[str] = None,
        context_tags: Optional[Mapping[str, str]] = None,
    ) -> PredictionRecord:

        # Convenience for the common case: `snapshot` is already a
        # live_system.live_ai_gateway.LiveAIPredictionSnapshot (or
        # anything duck-typed the same way -- .timestamp, .model_id/
        # .feature_schema_version are read via getattr, never assumed).
        # This is the ONE bridge point a Shadow-Mode caller uses; it
        # never imports live_system itself (no hard dependency), and
        # never re-derives anything the snapshot doesn't already carry.

        model_id = None
        bottleneck = getattr(snapshot, "bottleneck", None)
        if bottleneck is not None:
            model_id = getattr(bottleneck, "model_id", None)

        return self.record(
            timestamp=getattr(snapshot, "timestamp", 0.0),
            prediction_horizon_seconds=prediction_horizon_seconds,
            payload=snapshot,
            model_id=model_id,
            model_version=getattr(bottleneck, "model_version", None) if bottleneck is not None else None,
            feature_schema_version=getattr(snapshot, "feature_schema_version", None),
            source=source,
            scenario_id=scenario_id,
            building_id=building_id,
            context_tags=context_tags,
        )

    # =====================================================
    # Queries -- total accessors, never raise.
    # =====================================================

    def all(self) -> Tuple[PredictionRecord, ...]:

        return tuple(self._records)

    def by_id(self, prediction_id: str) -> Optional[PredictionRecord]:

        return self._by_id.get(prediction_id)

    def by_source(self, source: str) -> Tuple[PredictionRecord, ...]:

        return tuple(r for r in self._records if r.source == source)

    def by_scenario(self, scenario_id: str) -> Tuple[PredictionRecord, ...]:

        return tuple(r for r in self._records if r.scenario_id == scenario_id)

    def by_model(self, model_id: str) -> Tuple[PredictionRecord, ...]:

        return tuple(r for r in self._records if r.model_id == model_id)

    def __len__(self) -> int:

        return len(self._records)
