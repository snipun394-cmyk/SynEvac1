from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class GroundTruth:

    # One immutable record per completed simulation -- the correct
    # engineering answer for that run, computed deterministically by
    # analyzer.py from artifacts that already exist (Scenario/Building/
    # MultiAgentSimulationResult/TickResult). Every list-shaped field
    # below is a plain tuple of plain dicts (never a nested dataclass)
    # so to_dict()/from_dict() never need anything beyond tuple<->list
    # conversion -- there is no richer structure to lose or reconstruct.
    #
    # Optional[...] fields default to None, meaning "could not be
    # honestly computed from this run's artifacts" -- never a guessed
    # or interpolated value (see each ground_truth/*.py module for
    # exactly when that happens and why).

    scenario_id: str
    definition_id: str

    # ---- Evacuation ----
    total_evacuation_time: Optional[float] = None
    building_cleared: bool = False
    reachable_occupants: int = 0
    unreachable_occupants: int = 0
    people_trapped: int = 0
    people_evacuated: int = 0

    # ---- Congestion ----
    worst_exit: Optional[str] = None
    worst_stair: Optional[str] = None
    worst_door: Optional[str] = None
    peak_congestion_location_id: Optional[str] = None
    peak_congestion_location_type: Optional[str] = None
    peak_congestion_value: Optional[int] = None
    congestion_duration: Optional[float] = None

    # ---- Routes (one entry per zone) ----
    zone_route_stats: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)

    # ---- Fire ----
    first_hazardous_zone: Optional[str] = None
    hazard_spread_order: Tuple[str, ...] = field(default_factory=tuple)
    maximum_hazard_zone: Optional[str] = None
    highest_risk_floor: Optional[str] = None

    # ---- Engineering ----
    doors_that_became_bottlenecks: Tuple[str, ...] = field(default_factory=tuple)
    exits_underutilized: Tuple[str, ...] = field(default_factory=tuple)
    exits_exceeding_capacity: Tuple[str, ...] = field(default_factory=tuple)
    stairs_exceeding_capacity: Tuple[str, ...] = field(default_factory=tuple)

    # ---- Risk (normalized 0-1, one entry per object; risk_score may
    # itself be None within an entry when not honestly computable for
    # that particular object) ----
    zone_risk_scores: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)
    exit_risk_scores: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)
    stair_risk_scores: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)

    # ---- Recommendations ----
    recommendations: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)

    # ---- Human Behavior (Human Population & Assisted Evacuation
    # Modeling Phase 2/7) -- additive. occupant_behavior is one entry
    # per occupant in movement_result (occupant_id, dynamic_state,
    # ever_helping/assisted/fallen/possible_injury); the four *_count
    # fields are that same data already summed, restated as top-level
    # scalars for a consumer that only wants the aggregate (matching
    # Dataset Builder's own Helping_Group_Count/Assisted_Occupant_Count/
    # Fallen_Count/Possible_Injury_Count columns 1:1). Every existing
    # GroundTruth construction/serialized payload that predates this
    # phase is unaffected -- occupant_behavior defaults to (), every
    # count defaults to 0.
    occupant_behavior: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)
    helping_group_count: int = 0
    assisted_occupant_count: int = 0
    fallen_count: int = 0
    possible_injury_count: int = 0

    # ---- Profile-Aware Outcomes (Phase 7) -- additive. Each *_evacuated/
    # *_trapped pair is a straight count of that OccupantCategory's own
    # ARRIVED vs. not-ARRIVED occupants (behaviour_profile_resolver.
    # occupant_category()); rescued_occupant_count/firefighter_rescue_count/
    # average_rescue_time_seconds summarize Phase 4's own firefighter
    # rescue outcomes (0/0/None when no Scenario.firefighters were ever
    # deployed -- never fabricated).
    wheelchair_evacuated: int = 0
    wheelchair_trapped: int = 0
    children_evacuated: int = 0
    children_trapped: int = 0
    rescued_occupant_count: int = 0
    firefighter_rescue_count: int = 0
    average_rescue_time_seconds: Optional[float] = None

    # ---- Dynamic Decision Events (Human Decision Engine Phase 6) --
    # additive. Every count is 0 for any run that did not use
    # behaviour_profile_resolver.dynamic_registrar.
    # register_population_dynamic() -- see ground_truth.decision_events
    # for exactly how these are derived from SimulationArtifacts.
    # decision_events.
    help_decision_count: int = 0
    help_rejected_count: int = 0
    firefighter_task_count: int = 0
    group_formed_count: int = 0
    group_dissolved_count: int = 0
    rescue_initiated_count: int = 0
    rescue_completed_count: int = 0

    # ---- Occupant Attributes outcome summaries -- additive. Mean
    # values are None whenever the relevant occupant set (evacuated,
    # trapped, grouped, or ungrouped) is empty for this run -- never a
    # guessed/interpolated value (see ground_truth.
    # occupant_attribute_outcomes.compute_occupant_attribute_outcomes()
    # for exactly how each is derived). group_count/grouped_occupant_
    # count/mean_group_size describe Phase 3's own deterministic social
    # groups (behaviour_profile_resolver.occupant_grouping); every
    # other field here is 0/None for a run with no civilian occupants.
    mean_walking_speed_multiplier_evacuated: Optional[float] = None
    mean_walking_speed_multiplier_trapped: Optional[float] = None
    mean_risk_aversion_evacuated: Optional[float] = None
    mean_risk_aversion_trapped: Optional[float] = None
    mean_panic_susceptibility_evacuated: Optional[float] = None
    mean_panic_susceptibility_trapped: Optional[float] = None
    group_count: int = 0
    grouped_occupant_count: int = 0
    mean_group_size: Optional[float] = None
    grouped_evacuation_rate: Optional[float] = None
    ungrouped_evacuation_rate: Optional[float] = None

    # =====================================================

    def __post_init__(self):

        object.__setattr__(self, "zone_route_stats", tuple(self.zone_route_stats))
        object.__setattr__(self, "hazard_spread_order", tuple(self.hazard_spread_order))
        object.__setattr__(
            self, "doors_that_became_bottlenecks", tuple(self.doors_that_became_bottlenecks),
        )
        object.__setattr__(self, "exits_underutilized", tuple(self.exits_underutilized))
        object.__setattr__(
            self, "exits_exceeding_capacity", tuple(self.exits_exceeding_capacity),
        )
        object.__setattr__(
            self, "stairs_exceeding_capacity", tuple(self.stairs_exceeding_capacity),
        )
        object.__setattr__(self, "zone_risk_scores", tuple(self.zone_risk_scores))
        object.__setattr__(self, "exit_risk_scores", tuple(self.exit_risk_scores))
        object.__setattr__(self, "stair_risk_scores", tuple(self.stair_risk_scores))
        object.__setattr__(self, "recommendations", tuple(self.recommendations))
        object.__setattr__(self, "occupant_behavior", tuple(self.occupant_behavior))

    # =====================================================

    def to_dict(self) -> dict:

        return {
            "scenario_id": self.scenario_id,
            "definition_id": self.definition_id,
            "total_evacuation_time": self.total_evacuation_time,
            "building_cleared": self.building_cleared,
            "reachable_occupants": self.reachable_occupants,
            "unreachable_occupants": self.unreachable_occupants,
            "people_trapped": self.people_trapped,
            "people_evacuated": self.people_evacuated,
            "worst_exit": self.worst_exit,
            "worst_stair": self.worst_stair,
            "worst_door": self.worst_door,
            "peak_congestion_location_id": self.peak_congestion_location_id,
            "peak_congestion_location_type": self.peak_congestion_location_type,
            "peak_congestion_value": self.peak_congestion_value,
            "congestion_duration": self.congestion_duration,
            "zone_route_stats": [dict(entry) for entry in self.zone_route_stats],
            "first_hazardous_zone": self.first_hazardous_zone,
            "hazard_spread_order": list(self.hazard_spread_order),
            "maximum_hazard_zone": self.maximum_hazard_zone,
            "highest_risk_floor": self.highest_risk_floor,
            "doors_that_became_bottlenecks": list(self.doors_that_became_bottlenecks),
            "exits_underutilized": list(self.exits_underutilized),
            "exits_exceeding_capacity": list(self.exits_exceeding_capacity),
            "stairs_exceeding_capacity": list(self.stairs_exceeding_capacity),
            "zone_risk_scores": [dict(entry) for entry in self.zone_risk_scores],
            "exit_risk_scores": [dict(entry) for entry in self.exit_risk_scores],
            "stair_risk_scores": [dict(entry) for entry in self.stair_risk_scores],
            "recommendations": [dict(entry) for entry in self.recommendations],
            "occupant_behavior": [dict(entry) for entry in self.occupant_behavior],
            "helping_group_count": self.helping_group_count,
            "assisted_occupant_count": self.assisted_occupant_count,
            "fallen_count": self.fallen_count,
            "possible_injury_count": self.possible_injury_count,
            "wheelchair_evacuated": self.wheelchair_evacuated,
            "wheelchair_trapped": self.wheelchair_trapped,
            "children_evacuated": self.children_evacuated,
            "children_trapped": self.children_trapped,
            "rescued_occupant_count": self.rescued_occupant_count,
            "firefighter_rescue_count": self.firefighter_rescue_count,
            "average_rescue_time_seconds": self.average_rescue_time_seconds,
            "help_decision_count": self.help_decision_count,
            "help_rejected_count": self.help_rejected_count,
            "firefighter_task_count": self.firefighter_task_count,
            "group_formed_count": self.group_formed_count,
            "group_dissolved_count": self.group_dissolved_count,
            "rescue_initiated_count": self.rescue_initiated_count,
            "rescue_completed_count": self.rescue_completed_count,
            "mean_walking_speed_multiplier_evacuated": self.mean_walking_speed_multiplier_evacuated,
            "mean_walking_speed_multiplier_trapped": self.mean_walking_speed_multiplier_trapped,
            "mean_risk_aversion_evacuated": self.mean_risk_aversion_evacuated,
            "mean_risk_aversion_trapped": self.mean_risk_aversion_trapped,
            "mean_panic_susceptibility_evacuated": self.mean_panic_susceptibility_evacuated,
            "mean_panic_susceptibility_trapped": self.mean_panic_susceptibility_trapped,
            "group_count": self.group_count,
            "grouped_occupant_count": self.grouped_occupant_count,
            "mean_group_size": self.mean_group_size,
            "grouped_evacuation_rate": self.grouped_evacuation_rate,
            "ungrouped_evacuation_rate": self.ungrouped_evacuation_rate,
        }

    # =====================================================

    @classmethod
    def from_dict(cls, data: dict) -> "GroundTruth":

        return cls(
            scenario_id=data["scenario_id"],
            definition_id=data["definition_id"],
            total_evacuation_time=data.get("total_evacuation_time"),
            building_cleared=data.get("building_cleared", False),
            reachable_occupants=data.get("reachable_occupants", 0),
            unreachable_occupants=data.get("unreachable_occupants", 0),
            people_trapped=data.get("people_trapped", 0),
            people_evacuated=data.get("people_evacuated", 0),
            worst_exit=data.get("worst_exit"),
            worst_stair=data.get("worst_stair"),
            worst_door=data.get("worst_door"),
            peak_congestion_location_id=data.get("peak_congestion_location_id"),
            peak_congestion_location_type=data.get("peak_congestion_location_type"),
            peak_congestion_value=data.get("peak_congestion_value"),
            congestion_duration=data.get("congestion_duration"),
            zone_route_stats=data.get("zone_route_stats", []),
            first_hazardous_zone=data.get("first_hazardous_zone"),
            hazard_spread_order=data.get("hazard_spread_order", []),
            maximum_hazard_zone=data.get("maximum_hazard_zone"),
            highest_risk_floor=data.get("highest_risk_floor"),
            doors_that_became_bottlenecks=data.get("doors_that_became_bottlenecks", []),
            exits_underutilized=data.get("exits_underutilized", []),
            exits_exceeding_capacity=data.get("exits_exceeding_capacity", []),
            stairs_exceeding_capacity=data.get("stairs_exceeding_capacity", []),
            zone_risk_scores=data.get("zone_risk_scores", []),
            exit_risk_scores=data.get("exit_risk_scores", []),
            stair_risk_scores=data.get("stair_risk_scores", []),
            recommendations=data.get("recommendations", []),
            occupant_behavior=data.get("occupant_behavior", []),
            helping_group_count=data.get("helping_group_count", 0),
            assisted_occupant_count=data.get("assisted_occupant_count", 0),
            fallen_count=data.get("fallen_count", 0),
            possible_injury_count=data.get("possible_injury_count", 0),
            wheelchair_evacuated=data.get("wheelchair_evacuated", 0),
            wheelchair_trapped=data.get("wheelchair_trapped", 0),
            children_evacuated=data.get("children_evacuated", 0),
            children_trapped=data.get("children_trapped", 0),
            rescued_occupant_count=data.get("rescued_occupant_count", 0),
            firefighter_rescue_count=data.get("firefighter_rescue_count", 0),
            average_rescue_time_seconds=data.get("average_rescue_time_seconds"),
            help_decision_count=data.get("help_decision_count", 0),
            help_rejected_count=data.get("help_rejected_count", 0),
            firefighter_task_count=data.get("firefighter_task_count", 0),
            group_formed_count=data.get("group_formed_count", 0),
            group_dissolved_count=data.get("group_dissolved_count", 0),
            rescue_initiated_count=data.get("rescue_initiated_count", 0),
            rescue_completed_count=data.get("rescue_completed_count", 0),
            mean_walking_speed_multiplier_evacuated=data.get(
                "mean_walking_speed_multiplier_evacuated",
            ),
            mean_walking_speed_multiplier_trapped=data.get("mean_walking_speed_multiplier_trapped"),
            mean_risk_aversion_evacuated=data.get("mean_risk_aversion_evacuated"),
            mean_risk_aversion_trapped=data.get("mean_risk_aversion_trapped"),
            mean_panic_susceptibility_evacuated=data.get("mean_panic_susceptibility_evacuated"),
            mean_panic_susceptibility_trapped=data.get("mean_panic_susceptibility_trapped"),
            group_count=data.get("group_count", 0),
            grouped_occupant_count=data.get("grouped_occupant_count", 0),
            mean_group_size=data.get("mean_group_size"),
            grouped_evacuation_rate=data.get("grouped_evacuation_rate"),
            ungrouped_evacuation_rate=data.get("ungrouped_evacuation_rate"),
        )
