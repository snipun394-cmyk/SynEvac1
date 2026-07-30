from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Optional, Tuple
from uuid import uuid4


# =====================================================
# Prediction vs Reality Evaluation Framework milestone -- Phase 1's
# prediction registry model. Every value object here is frozen (matches
# this codebase's own "immutable value + mutable owning registry"
# convention throughout -- LiveOccupant/CalibrationProfile/BuildingState/
# every snapshot type this session has touched). This package is purely
# EVALUATIVE: it never runs inference, never trains a model, never
# influences Recommendation/Guidance/Simulation/Live Runtime -- see
# docs/architecture/prediction_evaluation.md and tests/
# test_prediction_evaluation_architecture_guards.py for the mechanical
# proof.
# =====================================================


@dataclass(frozen=True)
class PredictionRecord:

    # One immutable prediction, as it was actually produced -- Phase 1's
    # own required shape. `payload` is deliberately typed Any and NEVER
    # unpacked/flattened into named fields here: it holds whatever the
    # real prediction object already is (e.g. live_system.live_ai_gateway.
    # LiveAIPredictionSnapshot, or its own nested BottleneckOccurrencePrediction/
    # EvacuationTimePrediction, or a future model's own differently-shaped
    # output) -- this package never hard-imports live_system/ai_registry
    # to avoid depending on their concrete types (mirrors evacuation_
    # recommendation.evidence.ai_bottleneck_probability()'s own "duck-
    # typed, no hard import" convention). Metric extraction (metrics.py)
    # reads whatever attributes it needs off `payload` via getattr(),
    # never assuming a fixed shape.

    prediction_id: str = field(default_factory=lambda: str(uuid4()))

    timestamp: float = 0.0

    model_id: Optional[str] = None
    model_version: Optional[str] = None
    feature_schema_version: Optional[str] = None

    prediction_horizon_seconds: float = 0.0

    payload: Any = None

    # "simulation" | "live" -- a plain string, not a closed enum, same
    # "no forced import for a two-value vocabulary" convention live_
    # camera_pipeline.rtsp_frame_source.RTSPFrameSource.status already
    # establishes for STATUS_ONLINE etc. Phase 0's own explicit "must
    # support both Simulation and Live Runtime using identical
    # evaluation logic" requirement is satisfied by every function in
    # this package treating `source` as opaque metadata, never branching
    # on it.
    source: str = "unknown"

    # Optional grouping keys -- Phase 6/8/9's own "per-building,
    # per-scenario, per-condition, per-model" breakdowns all key off
    # these, never off a hidden assumption about how many scenarios/
    # models a caller is evaluating.
    scenario_id: Optional[str] = None
    building_id: Optional[str] = None

    # Phase 6's own "building condition" breakdown -- free-form
    # key/value tags a caller attaches when recording a prediction (e.g.
    # {"occupancy_level": "high", "floor_mode": "multi_floor",
    # "exit_status": "blocked"}). This package defines no closed
    # vocabulary for tag values -- see condition_analysis.py's own
    # docstring for the well-known keys it groups by when present, never
    # enforced.
    context_tags: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self):

        object.__setattr__(self, "context_tags", MappingProxyType(dict(self.context_tags)))


@dataclass(frozen=True)
class GroundTruthSample:

    # One immutable, already-realized fact about the building at one
    # instant -- Phase 3's own comparison categories, each Optional and
    # NEVER fabricated: absence means "not resolvable from what this
    # cycle's already-computed snapshots contained," never a guessed
    # zero (the same "None means genuinely unavailable" discipline every
    # snapshot type in this codebase already follows).

    timestamp: float
    source: str = "unknown"

    total_occupant_count: Optional[int] = None

    # True iff crowd_intelligence.models.BuildingCrowdSummary reported
    # ANY door/exit/stair as congested this cycle (congestion_level >=
    # HIGH, the SAME threshold crowd_intelligence.engine already applies
    # in both simulation and live -- see extraction.py). This is the
    # ground-truth counterpart to LiveAIPredictionSnapshot.bottleneck.
    # predicted_occurrence -- see docs/architecture/prediction_evaluation.md
    # Sec "Ground truth definition, precisely" for why this is a
    # deliberately WINDOWED/point-in-time definition, genuinely
    # different in scope from bottleneck_occurrence's own WHOLE-SCENARIO
    # training target (ground_truth.bottleneck.compute_engineering_
    # findings' own doors_that_became_bottlenecks), and why that
    # difference is disclosed rather than hidden.
    congestion_detected: Optional[bool] = None
    congested_asset_ids: Tuple[str, ...] = field(default_factory=tuple)

    # asset_id -> queue_candidate_count (Door/Exit/Stair, whichever this
    # cycle's CrowdIntelligenceSnapshot reported).
    queue_lengths: Mapping[str, int] = field(default_factory=dict)

    # asset_id -> a recent flow-rate reading (Stair reuses stair_flow.
    # models.StairFlowMetrics.exit_rate_per_minute -- the Stair
    # Predictive-Feature Live Parity milestone's own proved live
    # counterpart to candidate_recent_flow_rate; Exit reuses evacuation_
    # progress.models.ExitFlow.recent_flow_per_minute).
    flow_rates: Mapping[str, float] = field(default_factory=dict)

    hazard_overall_severity: Optional[str] = None  # hazard.severity.HazardSeverity's own .name

    evacuation_complete: Optional[bool] = None
    evacuation_time_seconds: Optional[float] = None

    def __post_init__(self):

        object.__setattr__(self, "congested_asset_ids", tuple(self.congested_asset_ids))
        object.__setattr__(self, "queue_lengths", MappingProxyType(dict(self.queue_lengths)))
        object.__setattr__(self, "flow_rates", MappingProxyType(dict(self.flow_rates)))


@dataclass(frozen=True)
class MatchedEvaluation:

    # Phase 2's own output shape -- one prediction, successfully matched
    # to the ground truth sample closest to its own target timestamp
    # (prediction.timestamp + prediction.prediction_horizon_seconds),
    # within a caller-configured tolerance. `match_time_delta_seconds` is
    # always reported (never hidden) -- Phase 2's own "support
    # configurable tolerances" requirement means a consumer of this type
    # must be able to see exactly how close the match actually was, not
    # just that it passed some threshold.

    prediction: PredictionRecord
    ground_truth: GroundTruthSample
    target_timestamp: float
    match_time_delta_seconds: float
