from dataclasses import dataclass

from models.building import Building
from models.door import Door
from models.exit import Exit
from models.floor import Floor
from models.staircase import Staircase
from models.zone import Zone

from scenario_definition import (
    EngineeringConstraints,
    FireDefinition,
    OccupantDefinition,
    ScenarioDefinition,
    UniformRange,
    WeightedOptions,
)
from scenario_definition.firefighter_definition import FirefighterDeploymentDefinition


# =====================================================
# Predictive Dataset V2 milestone, Phase 3 -- genuinely different
# Building/ScenarioDefinition topologies, reusing the exact same
# models.building/models.zone/.../scenario_definition abstractions
# ai_registry.training_scenario already established for V1 (no new
# graph representation, no new authoring machinery). V1's own
# make_training_building()/make_training_definition() are NEVER
# imported or modified here -- V1 stays byte-for-byte reproducible from
# its own untouched source (docs/architecture/predictive_dataset_campaign_v1.md).
#
# Phase 1's own investigation (docs/architecture/predictive_dataset_campaign_v2.md
# Section 1) traced V1's Stair failure to ONE root cause: V1's
# `Staircase(..., to_floor_id="floor-2")` never set `from_floor_id`.
# `Staircase.vertical_height()` requires BOTH floor ids to resolve real
# Floor objects (models/staircase.py); with from_floor_id defaulting to
# "" (building.get_floor("") -> None), vertical_height silently
# short-circuits to 0.0 -- collapsing candidate_walking_distance,
# capacity-model distance penalty, and simulated stair traversal
# duration all to zero. Every Staircase authored below sets BOTH
# from_floor_id and to_floor_id explicitly -- the single, root, and
# sufficient fix for that specific bug (verified: see
# tests/test_predictive_dataset_topologies_v2.py's
# StaircaseVerticalHeightRegressionTests).
# =====================================================


@dataclass(frozen=True)
class TopologySpec:

    name: str
    description: str
    building: Building
    definition: ScenarioDefinition
    scenario_count: int


# =====================================================
# Family 1 -- SINGLE_EXIT_LOWRISE (Phase 4). One floor, two zones, ONE
# exit -- no alternative route exists at all. Occupancy deliberately
# low/medium (Phase 2's own occupancy-stratum requirement). The sole
# exit is allowed to occasionally draw CLOSED (exit_state_distribution),
# which is exactly this family's genuine total-lockout case (Phase 9) --
# not fabricated, a real "only one exit exists and it is blocked" state.
# =====================================================


def build_single_exit_lowrise() -> TopologySpec:

    floor1 = Floor(
        id="sel-floor-1", name="Ground", display_order=0,
        zones=[
            Zone(id="sel-zone-lobby", name="Lobby", floor_id="sel-floor-1", x=0, y=0, width=10, height=8),
            Zone(id="sel-zone-hall", name="Hall", floor_id="sel-floor-1", x=14, y=0, width=14, height=8),
        ],
        doors=[
            Door(id="sel-door-1", normally_open=True, zone_a_id="sel-zone-lobby", zone_b_id="sel-zone-hall"),
        ],
        exits=[
            Exit(id="sel-exit-1", zone_id="sel-zone-lobby"),
        ],
    )

    building = Building(id="topology-single-exit-lowrise", name="Single-Exit Lowrise", floors=[floor1])

    definition = ScenarioDefinition(
        fire=FireDefinition(
            growth_parameter_distribution=UniformRange(80.0, 420.0),
            ignition_zone_preference=WeightedOptions({"sel-zone-lobby": 0.4, "sel-zone-hall": 0.6}),
            allowed_fire_profiles=frozenset({"Flaming", "Smoldering"}),
        ),
        occupant=OccupantDefinition(
            occupancy_distribution={
                "sel-zone-lobby": UniformRange(1, 8, discrete=True),
                "sel-zone-hall": UniformRange(2, 12, discrete=True),
            },
            behaviour_profile_distribution={
                "sel-zone-lobby": WeightedOptions({"Adult_Default": 0.6, "Elderly_Default": 0.2, "Wheelchair_Default": 0.1, "Visitor_Default": 0.1}),
                "sel-zone-hall": WeightedOptions({"Adult_Default": 0.7, "Staff_Default": 0.3}),
            },
            assistance_pairing_probability=0.2,
        ),
        engineering=EngineeringConstraints(
            door_state_distribution={
                "sel-door-1": WeightedOptions({"OPEN": 0.75, "CLOSED": 0.2, "LOCKED": 0.05}),
            },
            exit_state_distribution={
                # Deliberately NOT overwhelmingly open -- this is the one
                # family where the sole exit closing is the genuine,
                # meaningful total-lockout case Phase 9 investigates.
                "sel-exit-1": WeightedOptions({True: 0.85, False: 0.15}),
            },
            min_open_exits=1,
        ),
        firefighter=FirefighterDeploymentDefinition(
            team_count_distribution=WeightedOptions({0: 0.6, 1: 0.3, 2: 0.1}),
            team_size_distribution=UniformRange(2, 4, discrete=True),
            arrival_time_distribution=UniformRange(45.0, 180.0),
            entry_zone_ids=("sel-zone-lobby",),
            rescue_assignment_probability=0.6,
        ),
    )

    return TopologySpec(
        name="single_exit_lowrise",
        description="One floor, two zones, exactly one exit -- no alternative route exists. Low/medium occupancy.",
        building=building, definition=definition, scenario_count=500,
    )


# =====================================================
# Family 2 -- TWIN_STAIR_HIGHRISE (Phase 5/7/8). Three floors: ground
# (2 exits, 2 zones, 1 door) plus two independent upper floors, each
# with its OWN dedicated stair down to ground and its own door pair --
# genuinely load-bearing stairs (moderate/high upper occupancy against
# a narrow, capacity-limited stair), and Door+Stair simultaneous-
# bottleneck potential (Phase 7) since each upper floor's internal door
# and its stair can congest at the same time. Closing one stair (stair_
# state_distribution) strands that floor's occupants -- a genuine,
# disclosed stress case, not a bug (matches Obstacle/blocked-route
# precedent elsewhere in this codebase of "some occupants become
# unreachable" being a legitimate, already-tracked outcome, see
# movement_result.unreachable_occupant_ids).
# =====================================================


def build_twin_stair_highrise() -> TopologySpec:

    ground = Floor(
        id="tsh-floor-ground", name="Ground", display_order=0,
        zones=[
            Zone(id="tsh-zone-lobby-a", name="Lobby A", floor_id="tsh-floor-ground", x=0, y=0, width=14, height=10),
            Zone(id="tsh-zone-lobby-b", name="Lobby B", floor_id="tsh-floor-ground", x=20, y=0, width=14, height=10),
        ],
        doors=[
            Door(id="tsh-door-ground", normally_open=True, zone_a_id="tsh-zone-lobby-a", zone_b_id="tsh-zone-lobby-b"),
        ],
        exits=[
            Exit(id="tsh-exit-1", zone_id="tsh-zone-lobby-a"),
            Exit(id="tsh-exit-2", zone_id="tsh-zone-lobby-b"),
        ],
    )

    floor2 = Floor(
        id="tsh-floor-2", name="Level 2", display_order=1,
        zones=[
            Zone(id="tsh-zone-2a", name="Level 2 A", floor_id="tsh-floor-2", x=0, y=0, width=16, height=10),
            Zone(id="tsh-zone-2b", name="Level 2 B", floor_id="tsh-floor-2", x=20, y=0, width=16, height=10),
        ],
        doors=[
            Door(id="tsh-door-2", normally_open=True, zone_a_id="tsh-zone-2a", zone_b_id="tsh-zone-2b"),
        ],
        stairs=[
            # from_floor_id/to_floor_id BOTH set -- see module docstring.
            Staircase(id="tsh-stair-2", from_zone_id="tsh-zone-2a", to_zone_id="tsh-zone-lobby-a",
                      from_floor_id="tsh-floor-2", to_floor_id="tsh-floor-ground"),
        ],
    )

    floor3 = Floor(
        id="tsh-floor-3", name="Level 3", display_order=2,
        zones=[
            Zone(id="tsh-zone-3a", name="Level 3 A", floor_id="tsh-floor-3", x=0, y=0, width=16, height=10),
            Zone(id="tsh-zone-3b", name="Level 3 B", floor_id="tsh-floor-3", x=20, y=0, width=16, height=10),
        ],
        doors=[
            Door(id="tsh-door-3", normally_open=True, zone_a_id="tsh-zone-3a", zone_b_id="tsh-zone-3b"),
        ],
        stairs=[
            Staircase(id="tsh-stair-3", from_zone_id="tsh-zone-3b", to_zone_id="tsh-zone-lobby-b",
                      from_floor_id="tsh-floor-3", to_floor_id="tsh-floor-ground"),
        ],
    )

    building = Building(id="topology-twin-stair-highrise", name="Twin-Stair Highrise", floors=[ground, floor2, floor3])

    definition = ScenarioDefinition(
        fire=FireDefinition(
            growth_parameter_distribution=UniformRange(60.0, 400.0),
            ignition_zone_preference=WeightedOptions({
                "tsh-zone-lobby-a": 0.15, "tsh-zone-lobby-b": 0.15,
                "tsh-zone-2a": 0.2, "tsh-zone-2b": 0.2,
                "tsh-zone-3a": 0.15, "tsh-zone-3b": 0.15,
            }),
            allowed_fire_profiles=frozenset({"Flaming", "Smoldering"}),
        ),
        occupant=OccupantDefinition(
            occupancy_distribution={
                "tsh-zone-lobby-a": UniformRange(1, 8, discrete=True),
                "tsh-zone-lobby-b": UniformRange(1, 8, discrete=True),
                # Deliberately high -- genuine stair contention requires
                # more upper-floor demand than a capacity-limited stair
                # can instantly absorb (Phase 8's high-occupancy stress).
                "tsh-zone-2a": UniformRange(10, 35, discrete=True),
                "tsh-zone-2b": UniformRange(5, 20, discrete=True),
                "tsh-zone-3a": UniformRange(5, 20, discrete=True),
                "tsh-zone-3b": UniformRange(10, 35, discrete=True),
            },
            behaviour_profile_distribution={
                "tsh-zone-lobby-a": WeightedOptions({"Adult_Default": 0.7, "Visitor_Default": 0.3}),
                "tsh-zone-lobby-b": WeightedOptions({"Adult_Default": 0.7, "Visitor_Default": 0.3}),
                "tsh-zone-2a": WeightedOptions({"Adult_Default": 0.65, "Staff_Default": 0.25, "Elderly_Default": 0.1}),
                "tsh-zone-2b": WeightedOptions({"Adult_Default": 0.7, "Staff_Default": 0.3}),
                "tsh-zone-3a": WeightedOptions({"Adult_Default": 0.7, "Staff_Default": 0.3}),
                "tsh-zone-3b": WeightedOptions({"Adult_Default": 0.65, "Staff_Default": 0.25, "Wheelchair_Default": 0.1}),
            },
            assistance_pairing_probability=0.15,
        ),
        engineering=EngineeringConstraints(
            door_state_distribution={
                "tsh-door-ground": WeightedOptions({"OPEN": 0.8, "CLOSED": 0.15, "LOCKED": 0.05}),
                "tsh-door-2": WeightedOptions({"OPEN": 0.75, "CLOSED": 0.2, "LOCKED": 0.05}),
                "tsh-door-3": WeightedOptions({"OPEN": 0.75, "CLOSED": 0.2, "LOCKED": 0.05}),
            },
            exit_state_distribution={
                "tsh-exit-1": WeightedOptions({True: 0.9, False: 0.1}),
                "tsh-exit-2": WeightedOptions({True: 0.9, False: 0.1}),
            },
            min_open_exits=1,
            stair_state_distribution={
                "tsh-stair-2": WeightedOptions({"AVAILABLE": 0.8, "CLOSED": 0.2}),
                "tsh-stair-3": WeightedOptions({"AVAILABLE": 0.8, "CLOSED": 0.2}),
            },
        ),
        firefighter=FirefighterDeploymentDefinition(
            team_count_distribution=WeightedOptions({0: 0.4, 1: 0.35, 2: 0.25}),
            team_size_distribution=UniformRange(2, 4, discrete=True),
            arrival_time_distribution=UniformRange(45.0, 180.0),
            entry_zone_ids=("tsh-zone-lobby-a",),
            rescue_assignment_probability=0.6,
        ),
    )

    return TopologySpec(
        name="twin_stair_highrise",
        description="Three floors, two independent dedicated stairs each carrying high upper-floor demand, "
                    "plus per-floor doors -- Door+Stair simultaneous-bottleneck potential.",
        building=building, definition=definition, scenario_count=800,
    )


# =====================================================
# Family 3 -- MULTI_EXIT_WIDE (Phase 7). One floor, five zones in a
# star layout around a central hub, three exits, four doors -- enough
# independent chokepoints that Door+Door and Door+Exit simultaneous
# bottlenecks are structurally possible without any floor crossing.
# =====================================================


def build_multi_exit_wide() -> TopologySpec:

    floor1 = Floor(
        id="mew-floor-1", name="Ground", display_order=0,
        zones=[
            Zone(id="mew-zone-hub", name="Hub", floor_id="mew-floor-1", x=20, y=10, width=14, height=10),
            Zone(id="mew-zone-north", name="North Wing", floor_id="mew-floor-1", x=20, y=0, width=14, height=8),
            Zone(id="mew-zone-south", name="South Wing", floor_id="mew-floor-1", x=20, y=24, width=14, height=8),
            Zone(id="mew-zone-east", name="East Wing", floor_id="mew-floor-1", x=40, y=10, width=14, height=10),
            Zone(id="mew-zone-west", name="West Wing", floor_id="mew-floor-1", x=0, y=10, width=14, height=10),
        ],
        doors=[
            Door(id="mew-door-north", normally_open=True, zone_a_id="mew-zone-hub", zone_b_id="mew-zone-north"),
            Door(id="mew-door-south", normally_open=True, zone_a_id="mew-zone-hub", zone_b_id="mew-zone-south"),
            Door(id="mew-door-east", normally_open=True, zone_a_id="mew-zone-hub", zone_b_id="mew-zone-east"),
            Door(id="mew-door-west", normally_open=True, zone_a_id="mew-zone-hub", zone_b_id="mew-zone-west"),
        ],
        exits=[
            Exit(id="mew-exit-north", zone_id="mew-zone-north"),
            Exit(id="mew-exit-east", zone_id="mew-zone-east"),
            Exit(id="mew-exit-west", zone_id="mew-zone-west"),
        ],
    )

    building = Building(id="topology-multi-exit-wide", name="Multi-Exit Wide", floors=[floor1])

    definition = ScenarioDefinition(
        fire=FireDefinition(
            growth_parameter_distribution=UniformRange(60.0, 380.0),
            ignition_zone_preference=WeightedOptions({
                "mew-zone-hub": 0.3, "mew-zone-north": 0.175, "mew-zone-south": 0.175,
                "mew-zone-east": 0.175, "mew-zone-west": 0.175,
            }),
            allowed_fire_profiles=frozenset({"Flaming", "Smoldering"}),
        ),
        occupant=OccupantDefinition(
            occupancy_distribution={
                "mew-zone-hub": UniformRange(2, 12, discrete=True),
                "mew-zone-north": UniformRange(4, 22, discrete=True),
                "mew-zone-south": UniformRange(4, 22, discrete=True),
                "mew-zone-east": UniformRange(4, 22, discrete=True),
                "mew-zone-west": UniformRange(2, 14, discrete=True),
            },
            behaviour_profile_distribution={
                zone_id: WeightedOptions({"Adult_Default": 0.65, "Staff_Default": 0.2, "Elderly_Default": 0.15})
                for zone_id in ("mew-zone-hub", "mew-zone-north", "mew-zone-south", "mew-zone-east", "mew-zone-west")
            },
            assistance_pairing_probability=0.2,
        ),
        engineering=EngineeringConstraints(
            door_state_distribution={
                "mew-door-north": WeightedOptions({"OPEN": 0.75, "CLOSED": 0.2, "LOCKED": 0.05}),
                "mew-door-south": WeightedOptions({"OPEN": 0.75, "CLOSED": 0.2, "LOCKED": 0.05}),
                "mew-door-east": WeightedOptions({"OPEN": 0.75, "CLOSED": 0.2, "LOCKED": 0.05}),
                "mew-door-west": WeightedOptions({"OPEN": 0.75, "CLOSED": 0.2, "LOCKED": 0.05}),
            },
            exit_state_distribution={
                "mew-exit-north": WeightedOptions({True: 0.88, False: 0.12}),
                "mew-exit-east": WeightedOptions({True: 0.88, False: 0.12}),
                "mew-exit-west": WeightedOptions({True: 0.88, False: 0.12}),
            },
            min_open_exits=1,
        ),
        firefighter=FirefighterDeploymentDefinition(
            team_count_distribution=WeightedOptions({0: 0.45, 1: 0.35, 2: 0.2}),
            team_size_distribution=UniformRange(2, 4, discrete=True),
            arrival_time_distribution=UniformRange(45.0, 180.0),
            entry_zone_ids=("mew-zone-hub",),
            rescue_assignment_probability=0.6,
        ),
    )

    return TopologySpec(
        name="multi_exit_wide",
        description="One floor, five zones in a hub-and-spoke layout, three exits, four doors -- "
                    "structurally supports Door+Door and Door+Exit simultaneous bottlenecks.",
        building=building, definition=definition, scenario_count=700,
    )


# =====================================================
# Family 4 -- V1_TOPOLOGY_FIXED (Phase 14 direct comparison). The SAME
# shape as ai_registry.training_scenario.make_training_building() (2
# floors, 2 doors, 2 exits, 1 stair, similar occupancy/engineering
# ranges) but authored FRESH here with the stair's from_floor_id fixed
# -- isolates exactly how much of the V1-vs-V2 stair improvement comes
# from fixing the one authoring bug alone, independent of any of the
# other three families' added topology/occupancy diversity.
# =====================================================


def build_v1_topology_fixed() -> TopologySpec:

    floor1 = Floor(
        id="v1f-floor-1", name="Ground", display_order=0,
        zones=[
            Zone(id="v1f-zone-lobby", name="Lobby", floor_id="v1f-floor-1", x=0, y=0, width=12, height=8),
            Zone(id="v1f-zone-office-a", name="Office A", floor_id="v1f-floor-1", x=20, y=0, width=8, height=8),
            Zone(id="v1f-zone-office-b", name="Office B", floor_id="v1f-floor-1", x=32, y=0, width=8, height=8),
        ],
        doors=[
            Door(id="v1f-door-1", normally_open=True, zone_a_id="v1f-zone-lobby", zone_b_id="v1f-zone-office-a"),
            Door(id="v1f-door-2", normally_open=True, zone_a_id="v1f-zone-office-a", zone_b_id="v1f-zone-office-b"),
        ],
        exits=[
            Exit(id="v1f-exit-1", zone_id="v1f-zone-lobby"),
            Exit(id="v1f-exit-2", zone_id="v1f-zone-office-b"),
        ],
        stairs=[
            # Attached to floor1 (the FROM floor) -- matches
            # ai_registry.training_scenario.make_training_building()'s own
            # convention exactly. from_floor_id="v1f-floor-1" IS SET here
            # -- this is the exact V1 bug, fixed. Compare against V1's
            # Staircase(...), which omits from_floor_id entirely.
            Staircase(id="v1f-stair-1", from_zone_id="v1f-zone-office-a", to_zone_id="v1f-zone-upper",
                      from_floor_id="v1f-floor-1", to_floor_id="v1f-floor-2"),
        ],
    )

    floor2 = Floor(
        id="v1f-floor-2", name="Upper", display_order=1,
        zones=[
            Zone(id="v1f-zone-upper", name="Upper Hall", floor_id="v1f-floor-2", x=0, y=0, width=16, height=10),
        ],
    )

    building = Building(id="topology-v1-fixed", name="V1 Topology (Stair Bug Fixed)", floors=[floor1, floor2])

    definition = ScenarioDefinition(
        fire=FireDefinition(
            growth_parameter_distribution=UniformRange(80.0, 420.0),
            ignition_zone_preference=WeightedOptions({
                "v1f-zone-lobby": 0.3, "v1f-zone-office-a": 0.3, "v1f-zone-office-b": 0.25, "v1f-zone-upper": 0.15,
            }),
            allowed_fire_profiles=frozenset({"Flaming", "Smoldering"}),
        ),
        occupant=OccupantDefinition(
            occupancy_distribution={
                "v1f-zone-lobby": UniformRange(2, 10, discrete=True),
                "v1f-zone-office-a": UniformRange(1, 8, discrete=True),
                "v1f-zone-office-b": UniformRange(1, 8, discrete=True),
                "v1f-zone-upper": UniformRange(0, 6, discrete=True),
            },
            behaviour_profile_distribution={
                "v1f-zone-lobby": WeightedOptions({
                    "Adult_Default": 0.55, "Elderly_Default": 0.15, "Wheelchair_Default": 0.1,
                    "Child_Default": 0.1, "Visitor_Default": 0.1,
                }),
                "v1f-zone-office-a": WeightedOptions({"Adult_Default": 0.7, "Staff_Default": 0.3}),
                "v1f-zone-office-b": WeightedOptions({"Adult_Default": 0.7, "Staff_Default": 0.3}),
                "v1f-zone-upper": WeightedOptions({"Adult_Default": 0.6, "Visitor_Default": 0.4}),
            },
            assistance_pairing_probability=0.2,
        ),
        engineering=EngineeringConstraints(
            door_state_distribution={
                "v1f-door-1": WeightedOptions({"OPEN": 0.7, "CLOSED": 0.25, "LOCKED": 0.05}),
                "v1f-door-2": WeightedOptions({"OPEN": 0.7, "CLOSED": 0.25, "LOCKED": 0.05}),
            },
            exit_state_distribution={
                "v1f-exit-1": WeightedOptions({True: 0.9, False: 0.1}),
                "v1f-exit-2": WeightedOptions({True: 0.85, False: 0.15}),
            },
            min_open_exits=1,
            stair_state_distribution={
                "v1f-stair-1": WeightedOptions({"AVAILABLE": 0.85, "CLOSED": 0.15}),
            },
        ),
        firefighter=FirefighterDeploymentDefinition(
            team_count_distribution=WeightedOptions({0: 0.5, 1: 0.35, 2: 0.15}),
            team_size_distribution=UniformRange(2, 4, discrete=True),
            arrival_time_distribution=UniformRange(45.0, 180.0),
            entry_zone_ids=("v1f-zone-lobby",),
            rescue_assignment_probability=0.6,
        ),
    )

    return TopologySpec(
        name="v1_topology_fixed",
        description="Same shape as ai_registry.training_scenario.make_training_building() "
                    "(2 floors/2 doors/2 exits/1 stair), stair from_floor_id bug fixed -- isolates the "
                    "bug-fix effect from added topology diversity.",
        building=building, definition=definition, scenario_count=500,
    )


def all_topology_specs():

    return (
        build_single_exit_lowrise(),
        build_twin_stair_highrise(),
        build_multi_exit_wide(),
        build_v1_topology_fixed(),
    )
