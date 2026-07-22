from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

from models.building import Building
from scenario.scenario import Scenario

from perception.models.human_observation import HumanObservation

from advisory_system.ai_evidence import AIDecisionEvidence
from advisory_system.crowd_evidence import CrowdDecisionEvidence
from advisory_system.evacuation_progress_evidence import EvacuationProgressEvidence
from advisory_system.emergency_response_evidence import EmergencyResponseEvidence
from advisory_system.trajectory_evidence import TrajectoryDecisionEvidence
from advisory_system.evacuation_recommendation_evidence import EvacuationRecommendationEvidence
from advisory_system.evacuation_guidance_evidence import EvacuationGuidanceEvidence


# =====================================================
# Advisory System -- data model. Every dataclass below is a plain,
# frozen, to_dict()-able record, the same convention ground_truth.
# GroundTruth/decision_policy.DecisionPolicy already establish. This
# package produces recommendations *about* a scenario/incident; it
# never generates a prediction of its own (no model is fit or invoked
# here, no simulation is run, no RL policy is stepped) -- every field
# on every output record below is either copied straight from an
# already-computed platform artifact or is a disclosed, deterministic
# combination of such fields (see advisory_engine.py/confidence_engine.py
# for exactly which).
# =====================================================


@dataclass(frozen=True)
class AdvisoryInputs:

    # Everything the Advisory System reads for one scenario/incident
    # snapshot. Optional fields default to empty/None so an orchestrator
    # with only a subset of the platform wired up (e.g. Ground Truth
    # from a completed simulation but no live AI/RL/Perception deployed
    # yet) still produces an honest, partial Advisory Report rather than
    # failing or fabricating the missing piece.

    building: Building
    scenario: Scenario

    # Ground Truth (Simulation Mode) -- ground_truth.labels.GroundTruth.
    # Typed Any to avoid a hard import of ground_truth's internals
    # beyond the field names read in advisory_engine.py -- the same
    # "typed Any, not imported for its own sake" convention
    # decision_policy.policy.DecisionInputs.ground_truth and
    # command_center.incident_data.IncidentData.ground_truth already use.
    ground_truth: Any

    # Decision Policy -- decision_policy.policy.DecisionPolicy, already
    # computed by decision_policy.generate_policy() elsewhere. This
    # package never regenerates it -- every civilian/building
    # recommendation below is an annotation on top of an already-decided
    # zone/exit/stair action, never a replacement for one.
    decision_policy: Any

    # Live Perception -- perception.models.human_observation.
    # HumanObservation, keyed by person_id. Empty when no perception
    # source is configured for this run.
    human_observations: Mapping[str, HumanObservation] = field(default_factory=dict)

    # AI Inference -- ai_inference.predictor.Prediction /
    # ai_inference.confidence.ConfidenceReport (or any object exposing
    # the same .value/.confidence attributes), keyed by prediction_type.
    # Already computed by an ai_inference.Predictor elsewhere; this
    # package never loads or calls a model.
    ai_predictions: Mapping[str, Any] = field(default_factory=dict)
    ai_confidence: Mapping[str, Optional[float]] = field(default_factory=dict)

    # RL Outputs -- one already-computed predicted action (e.g. a
    # rl_training.action_space.ActionSpaceEntry, or any object exposing
    # .label/.action_type/.target_id), plus the policy's own reported
    # confidence for it if available. This package never steps a
    # SynEvacGymEnv or calls RLTrainer.predict() itself.
    rl_action: Optional[Any] = None
    rl_confidence: Optional[float] = None

    # Building System State -- current door/exit/stair/detector/camera
    # readings, the same id-keyed {object_id: {"state": "..."}} shape
    # command_center.incident_data.IncidentFrame.door_states/exit_states/
    # stair_states/detector_states/camera_states already use. Empty
    # means "not currently known" -- never guessed from Building's own
    # static baseline (Building Engineering State, read directly off
    # `building`/`ground_truth` instead, covers that case).
    building_system_state: Mapping[str, Mapping[str, str]] = field(default_factory=dict)

    # Optional Timeline Dataset rows for this run (dataset_builder.timeline.
    # extract_timeline_rows()'s own row shape) -- the only source of a
    # per-zone smoke reading anywhere in this platform (GroundTruth
    # itself carries no Zone_N_Smoke-equivalent field). Only the most
    # recent row is read (see advisory_engine._latest_zone_smoke_levels()),
    # the same "latest row" convention decision_policy.policy.
    # DecisionInputs.timeline_rows already establishes for its own
    # current_fire_zone_ids() use. Empty means smoke conditions are
    # simply omitted, never guessed from a risk_score.
    timeline_rows: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)

    # Simulation-mode elapsed time (seconds) this snapshot corresponds
    # to -- used only for RecommendationHistory timestamps.
    simulation_time: float = 0.0

    # AI-Augmented Decision Policy & Advisory Integration milestone --
    # additive. None means "no AI evidence this cycle" (the honest
    # default every existing caller keeps producing unchanged); see
    # advisory_system.ai_evidence's own module docstring for exactly
    # why this attaches here rather than to decision_policy.
    ai_decision_evidence: Optional[AIDecisionEvidence] = None

    # Live Crowd Intelligence -> Operational Advisory Integration
    # milestone -- additive, mirrors ai_decision_evidence exactly. None
    # means "no crowd evidence this cycle" (the honest default every
    # existing caller keeps producing unchanged); see
    # advisory_system.crowd_evidence's own module docstring for why
    # this attaches here rather than to decision_policy.
    crowd_decision_evidence: Optional[CrowdDecisionEvidence] = None

    # Live Evacuation Progress, Flow & Clearance Intelligence milestone --
    # additive, mirrors crowd_decision_evidence exactly. None means "no
    # evacuation-progress evidence this cycle."
    evacuation_progress_evidence: Optional[EvacuationProgressEvidence] = None

    # Live Emergency Response & Rescue Priority Intelligence milestone --
    # additive, mirrors evacuation_progress_evidence exactly.
    emergency_response_evidence: Optional[EmergencyResponseEvidence] = None

    # Live Occupant Trajectory, Movement Anomaly & Route-Deviation
    # Intelligence milestone -- additive, mirrors emergency_response_
    # evidence exactly. None means "no trajectory evidence this cycle."
    trajectory_decision_evidence: Optional[TrajectoryDecisionEvidence] = None

    # Live Dynamic Evacuation Recommendation Engine milestone --
    # additive, mirrors trajectory_decision_evidence exactly. None
    # means "no recommendation evidence this cycle."
    evacuation_recommendation_evidence: Optional[EvacuationRecommendationEvidence] = None

    # Live Evacuation Guidance & Zoned Message Planning milestone --
    # additive, mirrors evacuation_recommendation_evidence exactly. None
    # means "no guidance evidence this cycle."
    evacuation_guidance_evidence: Optional[EvacuationGuidanceEvidence] = None

    # =====================================================

    @property
    def scenario_id(self) -> str:

        return self.scenario.metadata.scenario_id


# =====================================================
# Phase 2 -- Civilian Advisory.
# =====================================================


@dataclass(frozen=True)
class CivilianAnnouncement:

    zone_id: str
    zone_name: str
    announcement: str
    reason: str
    confidence: Optional[float]
    predicted_rset_improvement_seconds: Optional[float]

    # Which real signals actually contributed to `confidence` for THIS
    # recommendation -- "ai" and/or "rl" appear only when advisory_
    # engine.py found a genuine, non-None ai_confidence/rl_confidence
    # for this exact recommendation; empty means confidence is entirely
    # DETERMINISTIC_RULE_BASE_CONFIDENCE (+ the deterministic risk-
    # score/agreement terms, neither of which is AI/RL either). Additive,
    # defaulted field -- any existing caller constructing this dataclass
    # without it keeps working unchanged. See command_center.
    # recommendation_center's own use of this field: it is the one place
    # allowed to say "AI confidence," and only when "ai" is present here.
    confidence_source: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:

        return {
            "zone_id": self.zone_id,
            "zone_name": self.zone_name,
            "announcement": self.announcement,
            "reason": self.reason,
            "confidence": self.confidence,
            "predicted_rset_improvement_seconds": self.predicted_rset_improvement_seconds,
            "confidence_source": list(self.confidence_source),
        }


# =====================================================
# Phase 3 -- Firefighter Intelligence. Every field here is information,
# never an instruction -- there is no "assigned_task"/"mission" field
# anywhere on this record, by design (see advisory_engine.build_
# firefighter_intelligence()'s own docstring).
# =====================================================


@dataclass(frozen=True)
class FirefighterIntelligenceReport:

    building_status: str
    occupants_remaining: int
    occupants_by_zone: Dict[str, int]

    children_count: int
    wheelchair_users_count: int
    possible_injury_count: int
    fallen_count: int
    helping_groups_count: int

    hazard_severity_by_zone: Dict[str, Optional[float]]
    smoke_conditions_by_zone: Dict[str, Optional[float]]

    available_routes: Tuple[str, ...]
    blocked_routes: Tuple[str, ...]

    building_system_status: Mapping[str, Mapping[str, str]]

    predicted_rset_seconds: Optional[float]
    confidence: Optional[float]

    rescue_priority_areas: Tuple[Dict[str, Any], ...]
    suggested_access_routes: Tuple[Dict[str, Any], ...]

    # AI-Augmented Decision Policy & Advisory Integration milestone --
    # additive. The raw, unblended AI Bottleneck Occurrence probability
    # (kept structurally separate from `confidence` above, which may
    # ALSO have this value blended into it -- see advisory_engine.
    # build_firefighter_intelligence()). None when no AI evidence was
    # available this cycle. Building-wide only -- never claims a
    # specific stair/door/exit/zone (see ai_evidence.py's own docstring
    # for why no such field exists anywhere in this package).
    ai_bottleneck_probability: Optional[float] = None
    ai_bottleneck_model_id: Optional[str] = None

    # Mirrors CivilianAnnouncement.confidence_source exactly -- "ai"
    # appears only when a genuine, non-None AI signal actually
    # contributed to `confidence` above.
    confidence_source: Tuple[str, ...] = ()

    # Live Emergency Response & Rescue Priority Intelligence milestone --
    # additive, deliberately "live_"-prefixed and kept structurally
    # SEPARATE from rescue_priority_areas above (that field remains
    # exclusively decision_policy.rescue_policy-sourced -- a completed
    # simulation's own GroundTruth-derived ranking -- never replaced or
    # merged; this pair is emergency_response/'s own, independently-
    # provenanced, genuinely live ranking).
    live_priority_zone_ids: Tuple[str, ...] = ()
    live_possible_assistance_zone_ids: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:

        return {
            "building_status": self.building_status,
            "occupants_remaining": self.occupants_remaining,
            "occupants_by_zone": dict(self.occupants_by_zone),
            "children_count": self.children_count,
            "wheelchair_users_count": self.wheelchair_users_count,
            "possible_injury_count": self.possible_injury_count,
            "fallen_count": self.fallen_count,
            "helping_groups_count": self.helping_groups_count,
            "hazard_severity_by_zone": dict(self.hazard_severity_by_zone),
            "smoke_conditions_by_zone": dict(self.smoke_conditions_by_zone),
            "available_routes": list(self.available_routes),
            "blocked_routes": list(self.blocked_routes),
            "building_system_status": {k: dict(v) for k, v in self.building_system_status.items()},
            "predicted_rset_seconds": self.predicted_rset_seconds,
            "confidence": self.confidence,
            "rescue_priority_areas": [dict(entry) for entry in self.rescue_priority_areas],
            "suggested_access_routes": [dict(entry) for entry in self.suggested_access_routes],
            "ai_bottleneck_probability": self.ai_bottleneck_probability,
            "ai_bottleneck_model_id": self.ai_bottleneck_model_id,
            "confidence_source": list(self.confidence_source),
            "live_priority_zone_ids": list(self.live_priority_zone_ids),
            "live_possible_assistance_zone_ids": list(self.live_possible_assistance_zone_ids),
        }


# =====================================================
# Phase 4 -- Building Advisory. Recommendation only -- there is no
# "execute"/"apply" method anywhere in this package (see
# advisory_engine.build_building_recommendations()'s own docstring).
# =====================================================


@dataclass(frozen=True)
class BuildingRecommendation:

    action: str
    target_type: str
    target_id: Optional[str]
    reason: str
    confidence: Optional[float]
    expected_engineering_benefit: str

    # AI-Augmented Decision Policy & Advisory Integration milestone --
    # additive, mirrors CivilianAnnouncement.confidence_source exactly.
    confidence_source: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:

        return {
            "action": self.action,
            "target_type": self.target_type,
            "target_id": self.target_id,
            "reason": self.reason,
            "confidence": self.confidence,
            "expected_engineering_benefit": self.expected_engineering_benefit,
            "confidence_source": list(self.confidence_source),
        }


# =====================================================
# Phase 5 -- Incident Commander Dashboard.
# =====================================================


@dataclass(frozen=True)
class IncidentCommanderDashboard:

    safe_zones: Tuple[str, ...]
    critical_zones: Tuple[str, ...]
    warning_zones: Tuple[str, ...]

    occupants_remaining: int
    occupants_by_zone: Dict[str, int]

    predicted_bottlenecks: Tuple[str, ...]
    blocked_routes: Tuple[str, ...]
    available_exits: Tuple[str, ...]

    building_system_status: Mapping[str, Mapping[str, str]]

    predicted_rset_seconds: Optional[float]
    highest_priority_rescue_areas: Tuple[str, ...]

    occupancy_confidence: Optional[float]
    recommendation_confidence: Optional[float]

    overall_incident_severity: str

    # AI-Augmented Decision Policy & Advisory Integration milestone --
    # additive. Raw building-wide AI Bottleneck Occurrence probability,
    # kept structurally separate from BOTH occupancy_confidence AND
    # recommendation_confidence above -- three genuinely different
    # quantities, never conflated (Phase 10's own explicit requirement).
    # `predicted_bottlenecks` above remains exclusively GroundTruth-
    # sourced (doors_that_became_bottlenecks/worst_exit/worst_stair) --
    # this milestone never appends a location string derived from this
    # building-wide-only probability to that tuple.
    ai_bottleneck_probability: Optional[float] = None
    ai_bottleneck_model_id: Optional[str] = None

    # Live Crowd Intelligence -> Operational Advisory Integration
    # milestone -- additive, mirrors ai_bottleneck_probability/
    # ai_bottleneck_model_id's own structurally-separate placement.
    # Unlike the AI fields (building-wide only), crowd intelligence CAN
    # honestly name a specific zone/asset (see advisory_system.
    # crowd_evidence's own docstring) -- these three fields are commander
    # AWARENESS only, never folded into predicted_bottlenecks/
    # available_exits/blocked_routes above, which remain exclusively
    # GroundTruth/DecisionPolicy-sourced (Phase 11's own "keep provenance
    # separate" requirement).
    crowd_highest_density_zone_id: Optional[str] = None
    crowd_most_congested_asset_id: Optional[str] = None
    crowd_most_congested_level: Optional[str] = None

    # Live Evacuation Progress, Flow & Clearance Intelligence milestone --
    # additive, mirrors the crowd_* fields immediately above exactly.
    # Phase 16's own "zones still containing known occupants, stalled
    # zones, uncertain-clearance zones" commander awareness.
    evacuation_progress_fraction: Optional[float] = None
    evacuation_stalled_zone_ids: Tuple[str, ...] = field(default_factory=tuple)
    evacuation_clearance_unknown_zone_ids: Tuple[str, ...] = field(default_factory=tuple)

    # Live Emergency Response & Rescue Priority Intelligence milestone --
    # additive, mirrors the evacuation_*/crowd_* fields immediately
    # above exactly. Phase 18's own "highest-priority zones, known
    # occupants remaining, possible assistance cases" commander summary.
    response_highest_priority_zone_id: Optional[str] = None
    response_critical_zone_ids: Tuple[str, ...] = field(default_factory=tuple)
    response_possible_assistance_zone_ids: Tuple[str, ...] = field(default_factory=tuple)

    # Live Human State & Assistance Perception Bridge milestone --
    # additive, mirrors the fields immediately above exactly (Phase 20).
    response_being_assisted_zone_ids: Tuple[str, ...] = field(default_factory=tuple)

    # Live Occupant Trajectory, Movement Anomaly & Route-Deviation
    # Intelligence milestone -- additive, mirrors the response_*/
    # evacuation_*/crowd_* fields immediately above exactly. Zones with
    # at least one occupant showing route-deviation/movement-stall/
    # hazardous-zone-movement evidence -- commander AWARENESS only,
    # never folded into critical_zones/warning_zones/blocked_routes
    # above (Phase 22's own "keep provenance separate" requirement,
    # same as crowd_*/evacuation_*/response_* before it).
    movement_route_deviation_zone_ids: Tuple[str, ...] = field(default_factory=tuple)
    movement_stalled_zone_ids: Tuple[str, ...] = field(default_factory=tuple)
    movement_hazardous_zone_ids: Tuple[str, ...] = field(default_factory=tuple)

    # Live Dynamic Evacuation Recommendation Engine milestone --
    # additive, mirrors the movement_*/response_*/evacuation_*/crowd_*
    # fields immediately above exactly. Commander AWARENESS only --
    # never folded into available_exits/blocked_routes above, which
    # remain exclusively GroundTruth/DecisionPolicy-sourced (Phase 17's
    # own "keep provenance separate" requirement, same as every prior
    # live-evidence field before it).
    zones_with_evacuation_recommendation_ids: Tuple[str, ...] = field(default_factory=tuple)
    zones_without_safe_exit_ids: Tuple[str, ...] = field(default_factory=tuple)

    # Live Evacuation Guidance & Zoned Message Planning milestone --
    # additive, mirrors the zones_with_evacuation_recommendation_ids/
    # zones_without_safe_exit_ids fields immediately above exactly.
    zones_with_valid_guidance_route_ids: Tuple[str, ...] = field(default_factory=tuple)
    zones_with_guidance_inconsistency_ids: Tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, Any]:

        return {
            "safe_zones": list(self.safe_zones),
            "critical_zones": list(self.critical_zones),
            "warning_zones": list(self.warning_zones),
            "occupants_remaining": self.occupants_remaining,
            "occupants_by_zone": dict(self.occupants_by_zone),
            "predicted_bottlenecks": list(self.predicted_bottlenecks),
            "blocked_routes": list(self.blocked_routes),
            "available_exits": list(self.available_exits),
            "building_system_status": {k: dict(v) for k, v in self.building_system_status.items()},
            "predicted_rset_seconds": self.predicted_rset_seconds,
            "highest_priority_rescue_areas": list(self.highest_priority_rescue_areas),
            "occupancy_confidence": self.occupancy_confidence,
            "recommendation_confidence": self.recommendation_confidence,
            "overall_incident_severity": self.overall_incident_severity,
            "ai_bottleneck_probability": self.ai_bottleneck_probability,
            "ai_bottleneck_model_id": self.ai_bottleneck_model_id,
            "crowd_highest_density_zone_id": self.crowd_highest_density_zone_id,
            "crowd_most_congested_asset_id": self.crowd_most_congested_asset_id,
            "crowd_most_congested_level": self.crowd_most_congested_level,
            "evacuation_progress_fraction": self.evacuation_progress_fraction,
            "evacuation_stalled_zone_ids": list(self.evacuation_stalled_zone_ids),
            "evacuation_clearance_unknown_zone_ids": list(self.evacuation_clearance_unknown_zone_ids),
            "response_highest_priority_zone_id": self.response_highest_priority_zone_id,
            "response_critical_zone_ids": list(self.response_critical_zone_ids),
            "response_possible_assistance_zone_ids": list(self.response_possible_assistance_zone_ids),
            "response_being_assisted_zone_ids": list(self.response_being_assisted_zone_ids),
            "movement_route_deviation_zone_ids": list(self.movement_route_deviation_zone_ids),
            "movement_stalled_zone_ids": list(self.movement_stalled_zone_ids),
            "movement_hazardous_zone_ids": list(self.movement_hazardous_zone_ids),
            "zones_with_evacuation_recommendation_ids": list(self.zones_with_evacuation_recommendation_ids),
            "zones_without_safe_exit_ids": list(self.zones_without_safe_exit_ids),
            "zones_with_valid_guidance_route_ids": list(self.zones_with_valid_guidance_route_ids),
            "zones_with_guidance_inconsistency_ids": list(self.zones_with_guidance_inconsistency_ids),
        }


# =====================================================
# Phase 6 -- Explainability.
# =====================================================


@dataclass(frozen=True)
class Explanation:

    summary: str
    reasons: Tuple[str, ...]
    confidence: Optional[float]

    def to_dict(self) -> Dict[str, Any]:

        return {
            "summary": self.summary,
            "reasons": list(self.reasons),
            "confidence": self.confidence,
        }


# =====================================================
# Phase 7 -- Recommendation History.
# =====================================================


@dataclass(frozen=True)
class RecommendationChange:

    timestamp: float
    zone_id: str
    previous_recommendation: Optional[str]
    new_recommendation: str
    reason_for_change: str
    confidence_before: Optional[float]
    confidence_after: Optional[float]

    def to_dict(self) -> Dict[str, Any]:

        return {
            "timestamp": self.timestamp,
            "zone_id": self.zone_id,
            "previous_recommendation": self.previous_recommendation,
            "new_recommendation": self.new_recommendation,
            "reason_for_change": self.reason_for_change,
            "confidence_before": self.confidence_before,
            "confidence_after": self.confidence_after,
        }


# =====================================================
# Phase 1 -- Advisory Core. The one combined report the orchestrator
# produces per snapshot.
# =====================================================


@dataclass(frozen=True)
class AdvisoryReport:

    scenario_id: str
    simulation_time: float

    civilian_announcements: Tuple[CivilianAnnouncement, ...]
    firefighter_intelligence: FirefighterIntelligenceReport
    building_recommendations: Tuple[BuildingRecommendation, ...]
    commander_dashboard: IncidentCommanderDashboard

    # Only the changes produced by *this* report's own civilian
    # announcements (see orchestrator.AdvisoryOrchestrator.generate_report()) --
    # the full running history lives on the orchestrator's own
    # RecommendationHistory instance, not duplicated onto every report.
    recommendation_history: Tuple[RecommendationChange, ...] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, Any]:

        return {
            "scenario_id": self.scenario_id,
            "simulation_time": self.simulation_time,
            "civilian_announcements": [entry.to_dict() for entry in self.civilian_announcements],
            "firefighter_intelligence": self.firefighter_intelligence.to_dict(),
            "building_recommendations": [entry.to_dict() for entry in self.building_recommendations],
            "commander_dashboard": self.commander_dashboard.to_dict(),
            "recommendation_history": [entry.to_dict() for entry in self.recommendation_history],
        }
