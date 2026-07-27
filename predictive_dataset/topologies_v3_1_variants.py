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

from predictive_dataset.topologies_v2 import TopologySpec


# =====================================================
# Localized Predictive Model V3.1 milestone, Phase 10 -- controlled
# topology-diversity scaling experiment. NOT a new generator engine --
# uses the exact same Building/Zone/Door/Exit/Staircase/
# ScenarioDefinition authoring primitives predictive_dataset/
# topologies_v2.py already established, closely mirroring each parent
# family's own parameter-distribution style. Each variant is a
# genuinely different STRUCTURAL shape within the same design pattern
# as one of the two topology-holdout-failing families
# (multi_exit_wide, twin_stair_highrise) -- not an arbitrary new graph.
# topologies_v2.py itself is completely untouched.
# =====================================================


def build_multi_exit_wide_6exit() -> TopologySpec:
    """Same hub-and-spoke pattern as multi_exit_wide, but with TWO more
    spokes (northeast, southeast), each with its own door AND exit --
    6 doors / 5 exits total (vs the original's 4 doors / 3 exits).
    Tests whether more exits/doors within the same design pattern
    improves transfer to the ORIGINAL multi_exit_wide test scenarios."""

    floor1 = Floor(
        id="mew6-floor-1", name="Ground", display_order=0,
        zones=[
            Zone(id="mew6-zone-hub", name="Hub", floor_id="mew6-floor-1", x=20, y=10, width=14, height=10),
            Zone(id="mew6-zone-north", name="North Wing", floor_id="mew6-floor-1", x=20, y=0, width=14, height=8),
            Zone(id="mew6-zone-south", name="South Wing", floor_id="mew6-floor-1", x=20, y=24, width=14, height=8),
            Zone(id="mew6-zone-east", name="East Wing", floor_id="mew6-floor-1", x=40, y=10, width=14, height=10),
            Zone(id="mew6-zone-west", name="West Wing", floor_id="mew6-floor-1", x=0, y=10, width=14, height=10),
            Zone(id="mew6-zone-northeast", name="Northeast Wing", floor_id="mew6-floor-1", x=40, y=0, width=14, height=8),
            Zone(id="mew6-zone-southeast", name="Southeast Wing", floor_id="mew6-floor-1", x=40, y=24, width=14, height=8),
        ],
        doors=[
            Door(id="mew6-door-north", normally_open=True, zone_a_id="mew6-zone-hub", zone_b_id="mew6-zone-north"),
            Door(id="mew6-door-south", normally_open=True, zone_a_id="mew6-zone-hub", zone_b_id="mew6-zone-south"),
            Door(id="mew6-door-east", normally_open=True, zone_a_id="mew6-zone-hub", zone_b_id="mew6-zone-east"),
            Door(id="mew6-door-west", normally_open=True, zone_a_id="mew6-zone-hub", zone_b_id="mew6-zone-west"),
            Door(id="mew6-door-northeast", normally_open=True, zone_a_id="mew6-zone-hub", zone_b_id="mew6-zone-northeast"),
            Door(id="mew6-door-southeast", normally_open=True, zone_a_id="mew6-zone-hub", zone_b_id="mew6-zone-southeast"),
        ],
        exits=[
            Exit(id="mew6-exit-north", zone_id="mew6-zone-north"),
            Exit(id="mew6-exit-east", zone_id="mew6-zone-east"),
            Exit(id="mew6-exit-west", zone_id="mew6-zone-west"),
            Exit(id="mew6-exit-northeast", zone_id="mew6-zone-northeast"),
            Exit(id="mew6-exit-southeast", zone_id="mew6-zone-southeast"),
        ],
    )

    building = Building(id="topology-multi-exit-wide-6exit", name="Multi-Exit Wide (6-Spoke)", floors=[floor1])

    zone_ids = ("mew6-zone-hub", "mew6-zone-north", "mew6-zone-south", "mew6-zone-east",
                "mew6-zone-west", "mew6-zone-northeast", "mew6-zone-southeast")

    definition = ScenarioDefinition(
        fire=FireDefinition(
            growth_parameter_distribution=UniformRange(60.0, 380.0),
            ignition_zone_preference=WeightedOptions({zid: 1.0 / len(zone_ids) for zid in zone_ids}),
            allowed_fire_profiles=frozenset({"Flaming", "Smoldering"}),
        ),
        occupant=OccupantDefinition(
            occupancy_distribution={
                "mew6-zone-hub": UniformRange(2, 12, discrete=True),
                "mew6-zone-north": UniformRange(4, 22, discrete=True),
                "mew6-zone-south": UniformRange(4, 22, discrete=True),
                "mew6-zone-east": UniformRange(4, 22, discrete=True),
                "mew6-zone-west": UniformRange(2, 14, discrete=True),
                "mew6-zone-northeast": UniformRange(4, 22, discrete=True),
                "mew6-zone-southeast": UniformRange(2, 14, discrete=True),
            },
            behaviour_profile_distribution={
                zid: WeightedOptions({"Adult_Default": 0.65, "Staff_Default": 0.2, "Elderly_Default": 0.15})
                for zid in zone_ids
            },
            assistance_pairing_probability=0.2,
        ),
        engineering=EngineeringConstraints(
            door_state_distribution={
                door_id: WeightedOptions({"OPEN": 0.75, "CLOSED": 0.2, "LOCKED": 0.05})
                for door_id in ("mew6-door-north", "mew6-door-south", "mew6-door-east",
                                "mew6-door-west", "mew6-door-northeast", "mew6-door-southeast")
            },
            exit_state_distribution={
                exit_id: WeightedOptions({True: 0.88, False: 0.12})
                for exit_id in ("mew6-exit-north", "mew6-exit-east", "mew6-exit-west",
                                 "mew6-exit-northeast", "mew6-exit-southeast")
            },
            min_open_exits=1,
        ),
        firefighter=FirefighterDeploymentDefinition(
            team_count_distribution=WeightedOptions({0: 0.45, 1: 0.35, 2: 0.2}),
            team_size_distribution=UniformRange(2, 4, discrete=True),
            arrival_time_distribution=UniformRange(45.0, 180.0),
            entry_zone_ids=("mew6-zone-hub",),
            rescue_assignment_probability=0.6,
        ),
    )

    return TopologySpec(
        name="multi_exit_wide_6exit",
        description="Hub-and-spoke, SAME pattern as multi_exit_wide but 6 doors / 5 exits (vs 4/3) -- "
                    "structural-diversity variant for Phase 10's topology-diversity scaling experiment.",
        building=building, definition=definition, scenario_count=200,
    )


def build_twin_stair_highrise_3stair() -> TopologySpec:
    """Same per-floor-dedicated-stair pattern as twin_stair_highrise,
    but with a THIRD upper floor and its own stair (3 stairs total vs
    the original's 2). Tests whether more floors/stairs within the same
    design pattern improves transfer to the ORIGINAL twin_stair_highrise
    test scenarios."""

    ground = Floor(
        id="tsh3-floor-ground", name="Ground", display_order=0,
        zones=[
            Zone(id="tsh3-zone-lobby-a", name="Lobby A", floor_id="tsh3-floor-ground", x=0, y=0, width=14, height=10),
            Zone(id="tsh3-zone-lobby-b", name="Lobby B", floor_id="tsh3-floor-ground", x=20, y=0, width=14, height=10),
        ],
        doors=[
            Door(id="tsh3-door-ground", normally_open=True, zone_a_id="tsh3-zone-lobby-a", zone_b_id="tsh3-zone-lobby-b"),
        ],
        exits=[
            Exit(id="tsh3-exit-1", zone_id="tsh3-zone-lobby-a"),
            Exit(id="tsh3-exit-2", zone_id="tsh3-zone-lobby-b"),
        ],
    )

    floor2 = Floor(
        id="tsh3-floor-2", name="Level 2", display_order=1,
        zones=[
            Zone(id="tsh3-zone-2a", name="Level 2 A", floor_id="tsh3-floor-2", x=0, y=0, width=16, height=10),
            Zone(id="tsh3-zone-2b", name="Level 2 B", floor_id="tsh3-floor-2", x=20, y=0, width=16, height=10),
        ],
        doors=[
            Door(id="tsh3-door-2", normally_open=True, zone_a_id="tsh3-zone-2a", zone_b_id="tsh3-zone-2b"),
        ],
        stairs=[
            Staircase(id="tsh3-stair-2", from_zone_id="tsh3-zone-2a", to_zone_id="tsh3-zone-lobby-a",
                      from_floor_id="tsh3-floor-2", to_floor_id="tsh3-floor-ground"),
        ],
    )

    floor3 = Floor(
        id="tsh3-floor-3", name="Level 3", display_order=2,
        zones=[
            Zone(id="tsh3-zone-3a", name="Level 3 A", floor_id="tsh3-floor-3", x=0, y=0, width=16, height=10),
            Zone(id="tsh3-zone-3b", name="Level 3 B", floor_id="tsh3-floor-3", x=20, y=0, width=16, height=10),
        ],
        doors=[
            Door(id="tsh3-door-3", normally_open=True, zone_a_id="tsh3-zone-3a", zone_b_id="tsh3-zone-3b"),
        ],
        stairs=[
            Staircase(id="tsh3-stair-3", from_zone_id="tsh3-zone-3b", to_zone_id="tsh3-zone-lobby-b",
                      from_floor_id="tsh3-floor-3", to_floor_id="tsh3-floor-ground"),
        ],
    )

    floor4 = Floor(
        id="tsh3-floor-4", name="Level 4", display_order=3,
        zones=[
            Zone(id="tsh3-zone-4a", name="Level 4 A", floor_id="tsh3-floor-4", x=0, y=0, width=16, height=10),
            Zone(id="tsh3-zone-4b", name="Level 4 B", floor_id="tsh3-floor-4", x=20, y=0, width=16, height=10),
        ],
        doors=[
            Door(id="tsh3-door-4", normally_open=True, zone_a_id="tsh3-zone-4a", zone_b_id="tsh3-zone-4b"),
        ],
        stairs=[
            Staircase(id="tsh3-stair-4", from_zone_id="tsh3-zone-4a", to_zone_id="tsh3-zone-lobby-b",
                      from_floor_id="tsh3-floor-4", to_floor_id="tsh3-floor-ground"),
        ],
    )

    building = Building(id="topology-twin-stair-highrise-3stair", name="Triple-Stair Highrise",
                         floors=[ground, floor2, floor3, floor4])

    definition = ScenarioDefinition(
        fire=FireDefinition(
            growth_parameter_distribution=UniformRange(60.0, 400.0),
            ignition_zone_preference=WeightedOptions({
                "tsh3-zone-lobby-a": 0.12, "tsh3-zone-lobby-b": 0.12,
                "tsh3-zone-2a": 0.15, "tsh3-zone-2b": 0.15,
                "tsh3-zone-3a": 0.13, "tsh3-zone-3b": 0.13,
                "tsh3-zone-4a": 0.1, "tsh3-zone-4b": 0.1,
            }),
            allowed_fire_profiles=frozenset({"Flaming", "Smoldering"}),
        ),
        occupant=OccupantDefinition(
            occupancy_distribution={
                "tsh3-zone-lobby-a": UniformRange(1, 8, discrete=True),
                "tsh3-zone-lobby-b": UniformRange(1, 8, discrete=True),
                "tsh3-zone-2a": UniformRange(10, 35, discrete=True),
                "tsh3-zone-2b": UniformRange(5, 20, discrete=True),
                "tsh3-zone-3a": UniformRange(5, 20, discrete=True),
                "tsh3-zone-3b": UniformRange(10, 35, discrete=True),
                "tsh3-zone-4a": UniformRange(10, 35, discrete=True),
                "tsh3-zone-4b": UniformRange(5, 20, discrete=True),
            },
            behaviour_profile_distribution={
                "tsh3-zone-lobby-a": WeightedOptions({"Adult_Default": 0.7, "Visitor_Default": 0.3}),
                "tsh3-zone-lobby-b": WeightedOptions({"Adult_Default": 0.7, "Visitor_Default": 0.3}),
                "tsh3-zone-2a": WeightedOptions({"Adult_Default": 0.65, "Staff_Default": 0.25, "Elderly_Default": 0.1}),
                "tsh3-zone-2b": WeightedOptions({"Adult_Default": 0.7, "Staff_Default": 0.3}),
                "tsh3-zone-3a": WeightedOptions({"Adult_Default": 0.7, "Staff_Default": 0.3}),
                "tsh3-zone-3b": WeightedOptions({"Adult_Default": 0.65, "Staff_Default": 0.25, "Wheelchair_Default": 0.1}),
                "tsh3-zone-4a": WeightedOptions({"Adult_Default": 0.65, "Staff_Default": 0.25, "Elderly_Default": 0.1}),
                "tsh3-zone-4b": WeightedOptions({"Adult_Default": 0.7, "Staff_Default": 0.3}),
            },
            assistance_pairing_probability=0.15,
        ),
        engineering=EngineeringConstraints(
            door_state_distribution={
                door_id: WeightedOptions({"OPEN": 0.78, "CLOSED": 0.17, "LOCKED": 0.05})
                for door_id in ("tsh3-door-ground", "tsh3-door-2", "tsh3-door-3", "tsh3-door-4")
            },
            exit_state_distribution={
                "tsh3-exit-1": WeightedOptions({True: 0.9, False: 0.1}),
                "tsh3-exit-2": WeightedOptions({True: 0.9, False: 0.1}),
            },
            min_open_exits=1,
            stair_state_distribution={
                "tsh3-stair-2": WeightedOptions({"AVAILABLE": 0.8, "CLOSED": 0.2}),
                "tsh3-stair-3": WeightedOptions({"AVAILABLE": 0.8, "CLOSED": 0.2}),
                "tsh3-stair-4": WeightedOptions({"AVAILABLE": 0.8, "CLOSED": 0.2}),
            },
        ),
        firefighter=FirefighterDeploymentDefinition(
            team_count_distribution=WeightedOptions({0: 0.4, 1: 0.35, 2: 0.25}),
            team_size_distribution=UniformRange(2, 4, discrete=True),
            arrival_time_distribution=UniformRange(45.0, 180.0),
            entry_zone_ids=("tsh3-zone-lobby-a",),
            rescue_assignment_probability=0.6,
        ),
    )

    return TopologySpec(
        name="twin_stair_highrise_3stair",
        description="SAME per-floor-dedicated-stair pattern as twin_stair_highrise but 4 floors / 3 stairs "
                    "(vs 3/2) -- structural-diversity variant for Phase 10's topology-diversity scaling experiment.",
        building=building, definition=definition, scenario_count=200,
    )


def variant_specs():
    return (build_multi_exit_wide_6exit(), build_twin_stair_highrise_3stair())
