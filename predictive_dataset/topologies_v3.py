import dataclasses
from dataclasses import dataclass
from typing import Tuple

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

from predictive_dataset.topologies_v2 import (
    TopologySpec,
    build_multi_exit_wide,
    build_single_exit_lowrise,
    build_twin_stair_highrise,
    build_v1_topology_fixed,
)
from predictive_dataset.topologies_v3_1_variants import (
    build_multi_exit_wide_6exit,
    build_twin_stair_highrise_3stair,
)


# =====================================================
# Predictive Dataset V3 milestone, Phase 2/3 -- STRUCTURAL topology
# diversity, motivated directly by the V3.1 Robustness Investigation
# (docs/architecture/..._v3_1... / MEMORY localized_predictive_model_v3_1
# _robustness_milestone): controlled structural-diversity variants of
# multi_exit_wide/twin_stair_highrise improved unseen-topology PR-AUC by
# +23.5% and +126% respectively, while deduplication/reweighting/
# relational features did NOT improve generalization. This module is the
# generalization of that one-off experiment into a versioned, deliberate
# structural-variant architecture covering ALL FOUR topology families,
# not just the two V3.1 happened to test.
#
# topologies_v2.py and topologies_v3_1_variants.py are NEITHER modified
# NOR duplicated -- every existing builder is imported and reused
# verbatim as one variant of its family (Phase 2's own "prefer extending
# the existing topology-generation architecture rather than creating a
# parallel simulator" instruction). Every NEW builder below reuses the
# exact same Building/Zone/Door/Exit/Staircase/ScenarioDefinition
# authoring primitives, in the same per-family distribution style.
#
# A "structural variant" differs from its siblings in real graph
# properties (floor/zone/door/exit/stair counts, connectivity pattern --
# star/hub vs linear/chain vs branching, serial vs parallel stairs,
# symmetric vs asymmetric exit placement) -- never merely renamed ids or
# shifted (x, y) coordinates. See topology_signature.py (Phase 3) and
# topology_diversity_v3.py (Phase 4) for the mechanical verification
# that these are genuinely distinct, not just declared so.
# =====================================================


@dataclass(frozen=True)
class StructuralVariant:

    family: str
    variant_id: str
    variant_label: str
    topology: TopologySpec


def with_scenario_count(spec: TopologySpec, scenario_count: int) -> TopologySpec:
    """Returns a copy of `spec` with a different scenario_count -- used
    by the pilot/full-scale campaign runners to independently control
    campaign size without baking a single fixed count into each variant
    builder (Phase 8's ~25/variant pilot vs Phase 11's ~100-150/variant
    full-scale decision are NOT the same number)."""

    return dataclasses.replace(spec, scenario_count=scenario_count)


# =====================================================
# Family 1 -- SINGLE_EXIT_LOWRISE. Base (topologies_v2): 1 floor, 2
# zones, 1 door, 1 exit, no alternative route at all.
# =====================================================


def build_single_exit_deep_corridor() -> TopologySpec:
    """1 floor, FOUR zones chained in series (lobby -> corridor1 ->
    corridor2 -> hall), THREE doors, exit only at the lobby end --
    same single-exit/no-redundancy property as the base family, but a
    much longer, deeper serial path (more doors in series before the
    exit, longer candidate_walking_distance)."""

    floor1 = Floor(
        id="seldc-floor-1", name="Ground", display_order=0,
        zones=[
            Zone(id="seldc-zone-lobby", name="Lobby", floor_id="seldc-floor-1", x=0, y=0, width=10, height=8),
            Zone(id="seldc-zone-corridor1", name="Corridor 1", floor_id="seldc-floor-1", x=14, y=0, width=12, height=6),
            Zone(id="seldc-zone-corridor2", name="Corridor 2", floor_id="seldc-floor-1", x=30, y=0, width=12, height=6),
            Zone(id="seldc-zone-hall", name="Hall", floor_id="seldc-floor-1", x=46, y=0, width=14, height=8),
        ],
        doors=[
            Door(id="seldc-door-1", normally_open=True, zone_a_id="seldc-zone-lobby", zone_b_id="seldc-zone-corridor1"),
            Door(id="seldc-door-2", normally_open=True, zone_a_id="seldc-zone-corridor1", zone_b_id="seldc-zone-corridor2"),
            Door(id="seldc-door-3", normally_open=True, zone_a_id="seldc-zone-corridor2", zone_b_id="seldc-zone-hall"),
        ],
        exits=[
            Exit(id="seldc-exit-1", zone_id="seldc-zone-lobby"),
        ],
    )

    building = Building(id="topology-single-exit-deep-corridor", name="Single-Exit Deep Corridor", floors=[floor1])

    zone_ids = ("seldc-zone-lobby", "seldc-zone-corridor1", "seldc-zone-corridor2", "seldc-zone-hall")

    definition = ScenarioDefinition(
        fire=FireDefinition(
            growth_parameter_distribution=UniformRange(80.0, 420.0),
            ignition_zone_preference=WeightedOptions({zid: 1.0 / len(zone_ids) for zid in zone_ids}),
            allowed_fire_profiles=frozenset({"Flaming", "Smoldering"}),
        ),
        occupant=OccupantDefinition(
            occupancy_distribution={
                "seldc-zone-lobby": UniformRange(1, 6, discrete=True),
                "seldc-zone-corridor1": UniformRange(1, 6, discrete=True),
                "seldc-zone-corridor2": UniformRange(1, 6, discrete=True),
                "seldc-zone-hall": UniformRange(2, 12, discrete=True),
            },
            behaviour_profile_distribution={
                zid: WeightedOptions({"Adult_Default": 0.65, "Staff_Default": 0.2, "Elderly_Default": 0.15})
                for zid in zone_ids
            },
            assistance_pairing_probability=0.2,
        ),
        engineering=EngineeringConstraints(
            door_state_distribution={
                did: WeightedOptions({"OPEN": 0.75, "CLOSED": 0.2, "LOCKED": 0.05})
                for did in ("seldc-door-1", "seldc-door-2", "seldc-door-3")
            },
            exit_state_distribution={
                "seldc-exit-1": WeightedOptions({True: 0.85, False: 0.15}),
            },
            min_open_exits=1,
        ),
        firefighter=FirefighterDeploymentDefinition(
            team_count_distribution=WeightedOptions({0: 0.6, 1: 0.3, 2: 0.1}),
            team_size_distribution=UniformRange(2, 4, discrete=True),
            arrival_time_distribution=UniformRange(45.0, 180.0),
            entry_zone_ids=("seldc-zone-lobby",),
            rescue_assignment_probability=0.6,
        ),
    )

    return TopologySpec(
        name="single_exit_deep_corridor",
        description="Single-exit family, deep serial corridor variant: 1 floor, 4 zones chained, "
                    "3 doors in series, 1 exit -- long path length, no alternative route.",
        building=building, definition=definition, scenario_count=150,
    )


def build_single_exit_branching_deadends() -> TopologySpec:
    """1 floor, FOUR zones: lobby (the sole exit) + a hall that
    BRANCHES into two dead-end rooms. Still single-exit/no-redundancy,
    but branching (not purely linear) connectivity -- two doors converge
    on the same hall before the one path to the exit."""

    floor1 = Floor(
        id="selbd-floor-1", name="Ground", display_order=0,
        zones=[
            Zone(id="selbd-zone-lobby", name="Lobby", floor_id="selbd-floor-1", x=0, y=8, width=10, height=8),
            Zone(id="selbd-zone-hall", name="Hall", floor_id="selbd-floor-1", x=14, y=8, width=12, height=8),
            Zone(id="selbd-zone-roomA", name="Room A", floor_id="selbd-floor-1", x=30, y=0, width=12, height=8),
            Zone(id="selbd-zone-roomB", name="Room B", floor_id="selbd-floor-1", x=30, y=16, width=12, height=8),
        ],
        doors=[
            Door(id="selbd-door-lobby-hall", normally_open=True, zone_a_id="selbd-zone-lobby", zone_b_id="selbd-zone-hall"),
            Door(id="selbd-door-hall-roomA", normally_open=True, zone_a_id="selbd-zone-hall", zone_b_id="selbd-zone-roomA"),
            Door(id="selbd-door-hall-roomB", normally_open=True, zone_a_id="selbd-zone-hall", zone_b_id="selbd-zone-roomB"),
        ],
        exits=[
            Exit(id="selbd-exit-1", zone_id="selbd-zone-lobby"),
        ],
    )

    building = Building(id="topology-single-exit-branching-deadends", name="Single-Exit Branching Deadends", floors=[floor1])

    zone_ids = ("selbd-zone-lobby", "selbd-zone-hall", "selbd-zone-roomA", "selbd-zone-roomB")

    definition = ScenarioDefinition(
        fire=FireDefinition(
            growth_parameter_distribution=UniformRange(80.0, 420.0),
            ignition_zone_preference=WeightedOptions({zid: 1.0 / len(zone_ids) for zid in zone_ids}),
            allowed_fire_profiles=frozenset({"Flaming", "Smoldering"}),
        ),
        occupant=OccupantDefinition(
            occupancy_distribution={
                "selbd-zone-lobby": UniformRange(1, 6, discrete=True),
                "selbd-zone-hall": UniformRange(1, 8, discrete=True),
                "selbd-zone-roomA": UniformRange(2, 10, discrete=True),
                "selbd-zone-roomB": UniformRange(2, 10, discrete=True),
            },
            behaviour_profile_distribution={
                zid: WeightedOptions({"Adult_Default": 0.6, "Elderly_Default": 0.2, "Wheelchair_Default": 0.1, "Visitor_Default": 0.1})
                for zid in zone_ids
            },
            assistance_pairing_probability=0.2,
        ),
        engineering=EngineeringConstraints(
            door_state_distribution={
                did: WeightedOptions({"OPEN": 0.75, "CLOSED": 0.2, "LOCKED": 0.05})
                for did in ("selbd-door-lobby-hall", "selbd-door-hall-roomA", "selbd-door-hall-roomB")
            },
            exit_state_distribution={
                "selbd-exit-1": WeightedOptions({True: 0.85, False: 0.15}),
            },
            min_open_exits=1,
        ),
        firefighter=FirefighterDeploymentDefinition(
            team_count_distribution=WeightedOptions({0: 0.6, 1: 0.3, 2: 0.1}),
            team_size_distribution=UniformRange(2, 4, discrete=True),
            arrival_time_distribution=UniformRange(45.0, 180.0),
            entry_zone_ids=("selbd-zone-lobby",),
            rescue_assignment_probability=0.6,
        ),
    )

    return TopologySpec(
        name="single_exit_branching_deadends",
        description="Single-exit family, branching variant: 1 floor, hall branches into two dead-end "
                    "rooms, 3 doors, 1 exit -- branching (not linear) connectivity, no redundancy.",
        building=building, definition=definition, scenario_count=150,
    )


def build_single_exit_vertical() -> TopologySpec:
    """TWO floors -- an upper single-zone floor connected by ONE stair
    down to a ground floor with a single exit. Adds floor_count=2 and
    stair_count=1 to a family that otherwise never has either."""

    ground = Floor(
        id="selv-floor-1", name="Ground", display_order=0,
        zones=[
            Zone(id="selv-zone-lobby", name="Lobby", floor_id="selv-floor-1", x=0, y=0, width=10, height=8),
            Zone(id="selv-zone-hall", name="Hall", floor_id="selv-floor-1", x=14, y=0, width=14, height=8),
        ],
        doors=[
            Door(id="selv-door-1", normally_open=True, zone_a_id="selv-zone-lobby", zone_b_id="selv-zone-hall"),
        ],
        exits=[
            Exit(id="selv-exit-1", zone_id="selv-zone-lobby"),
        ],
    )

    upper = Floor(
        id="selv-floor-2", name="Upper", display_order=1,
        zones=[
            Zone(id="selv-zone-upper", name="Upper Hall", floor_id="selv-floor-2", x=0, y=0, width=16, height=10),
        ],
        stairs=[
            Staircase(id="selv-stair-1", from_zone_id="selv-zone-upper", to_zone_id="selv-zone-hall",
                      from_floor_id="selv-floor-2", to_floor_id="selv-floor-1"),
        ],
    )

    building = Building(id="topology-single-exit-vertical", name="Single-Exit Vertical", floors=[ground, upper])

    definition = ScenarioDefinition(
        fire=FireDefinition(
            growth_parameter_distribution=UniformRange(80.0, 420.0),
            ignition_zone_preference=WeightedOptions({
                "selv-zone-lobby": 0.25, "selv-zone-hall": 0.35, "selv-zone-upper": 0.4,
            }),
            allowed_fire_profiles=frozenset({"Flaming", "Smoldering"}),
        ),
        occupant=OccupantDefinition(
            occupancy_distribution={
                "selv-zone-lobby": UniformRange(1, 6, discrete=True),
                "selv-zone-hall": UniformRange(1, 8, discrete=True),
                "selv-zone-upper": UniformRange(4, 20, discrete=True),
            },
            behaviour_profile_distribution={
                "selv-zone-lobby": WeightedOptions({"Adult_Default": 0.6, "Elderly_Default": 0.2, "Wheelchair_Default": 0.1, "Visitor_Default": 0.1}),
                "selv-zone-hall": WeightedOptions({"Adult_Default": 0.7, "Staff_Default": 0.3}),
                "selv-zone-upper": WeightedOptions({"Adult_Default": 0.65, "Staff_Default": 0.25, "Elderly_Default": 0.1}),
            },
            assistance_pairing_probability=0.2,
        ),
        engineering=EngineeringConstraints(
            door_state_distribution={
                "selv-door-1": WeightedOptions({"OPEN": 0.75, "CLOSED": 0.2, "LOCKED": 0.05}),
            },
            exit_state_distribution={
                "selv-exit-1": WeightedOptions({True: 0.85, False: 0.15}),
            },
            min_open_exits=1,
            stair_state_distribution={
                "selv-stair-1": WeightedOptions({"AVAILABLE": 0.8, "CLOSED": 0.2}),
            },
        ),
        firefighter=FirefighterDeploymentDefinition(
            team_count_distribution=WeightedOptions({0: 0.55, 1: 0.3, 2: 0.15}),
            team_size_distribution=UniformRange(2, 4, discrete=True),
            arrival_time_distribution=UniformRange(45.0, 180.0),
            entry_zone_ids=("selv-zone-lobby",),
            rescue_assignment_probability=0.6,
        ),
    )

    return TopologySpec(
        name="single_exit_vertical",
        description="Single-exit family, vertical variant: 2 floors, 1 stair down to a single "
                    "ground-floor exit -- adds a floor/stair dimension the base family never has.",
        building=building, definition=definition, scenario_count=150,
    )


# =====================================================
# Family 2 -- TWIN_STAIR_HIGHRISE. Base (topologies_v2): 3 floors, 2
# independent dedicated stairs. v3.1 variant (topologies_v3_1_variants):
# 4 floors, 3 independent dedicated stairs.
# =====================================================


def build_twin_stair_low() -> TopologySpec:
    """Only TWO floors (ground + one upper), ONE stair -- a smaller/
    shorter version of the same per-floor-dedicated-stair pattern,
    testing floor_count=2/stair_count=1 within this family."""

    ground = Floor(
        id="tshl-floor-ground", name="Ground", display_order=0,
        zones=[
            Zone(id="tshl-zone-lobby-a", name="Lobby A", floor_id="tshl-floor-ground", x=0, y=0, width=14, height=10),
            Zone(id="tshl-zone-lobby-b", name="Lobby B", floor_id="tshl-floor-ground", x=20, y=0, width=14, height=10),
        ],
        doors=[
            Door(id="tshl-door-ground", normally_open=True, zone_a_id="tshl-zone-lobby-a", zone_b_id="tshl-zone-lobby-b"),
        ],
        exits=[
            Exit(id="tshl-exit-1", zone_id="tshl-zone-lobby-a"),
            Exit(id="tshl-exit-2", zone_id="tshl-zone-lobby-b"),
        ],
    )

    floor2 = Floor(
        id="tshl-floor-2", name="Level 2", display_order=1,
        zones=[
            Zone(id="tshl-zone-2a", name="Level 2 A", floor_id="tshl-floor-2", x=0, y=0, width=16, height=10),
            Zone(id="tshl-zone-2b", name="Level 2 B", floor_id="tshl-floor-2", x=20, y=0, width=16, height=10),
        ],
        doors=[
            Door(id="tshl-door-2", normally_open=True, zone_a_id="tshl-zone-2a", zone_b_id="tshl-zone-2b"),
        ],
        stairs=[
            Staircase(id="tshl-stair-2", from_zone_id="tshl-zone-2a", to_zone_id="tshl-zone-lobby-a",
                      from_floor_id="tshl-floor-2", to_floor_id="tshl-floor-ground"),
        ],
    )

    building = Building(id="topology-twin-stair-low", name="Twin-Stair Low", floors=[ground, floor2])

    definition = ScenarioDefinition(
        fire=FireDefinition(
            growth_parameter_distribution=UniformRange(60.0, 400.0),
            ignition_zone_preference=WeightedOptions({
                "tshl-zone-lobby-a": 0.2, "tshl-zone-lobby-b": 0.2, "tshl-zone-2a": 0.3, "tshl-zone-2b": 0.3,
            }),
            allowed_fire_profiles=frozenset({"Flaming", "Smoldering"}),
        ),
        occupant=OccupantDefinition(
            occupancy_distribution={
                "tshl-zone-lobby-a": UniformRange(1, 8, discrete=True),
                "tshl-zone-lobby-b": UniformRange(1, 8, discrete=True),
                "tshl-zone-2a": UniformRange(8, 30, discrete=True),
                "tshl-zone-2b": UniformRange(5, 20, discrete=True),
            },
            behaviour_profile_distribution={
                "tshl-zone-lobby-a": WeightedOptions({"Adult_Default": 0.7, "Visitor_Default": 0.3}),
                "tshl-zone-lobby-b": WeightedOptions({"Adult_Default": 0.7, "Visitor_Default": 0.3}),
                "tshl-zone-2a": WeightedOptions({"Adult_Default": 0.65, "Staff_Default": 0.25, "Elderly_Default": 0.1}),
                "tshl-zone-2b": WeightedOptions({"Adult_Default": 0.7, "Staff_Default": 0.3}),
            },
            assistance_pairing_probability=0.15,
        ),
        engineering=EngineeringConstraints(
            door_state_distribution={
                "tshl-door-ground": WeightedOptions({"OPEN": 0.8, "CLOSED": 0.15, "LOCKED": 0.05}),
                "tshl-door-2": WeightedOptions({"OPEN": 0.75, "CLOSED": 0.2, "LOCKED": 0.05}),
            },
            exit_state_distribution={
                "tshl-exit-1": WeightedOptions({True: 0.9, False: 0.1}),
                "tshl-exit-2": WeightedOptions({True: 0.9, False: 0.1}),
            },
            min_open_exits=1,
            stair_state_distribution={
                "tshl-stair-2": WeightedOptions({"AVAILABLE": 0.8, "CLOSED": 0.2}),
            },
        ),
        firefighter=FirefighterDeploymentDefinition(
            team_count_distribution=WeightedOptions({0: 0.45, 1: 0.35, 2: 0.2}),
            team_size_distribution=UniformRange(2, 4, discrete=True),
            arrival_time_distribution=UniformRange(45.0, 180.0),
            entry_zone_ids=("tshl-zone-lobby-a",),
            rescue_assignment_probability=0.6,
        ),
    )

    return TopologySpec(
        name="twin_stair_low",
        description="Twin-stair family, low-rise variant: 2 floors, 1 dedicated stair, 2 exits -- "
                    "smaller/shorter version of the per-floor-dedicated-stair pattern.",
        building=building, definition=definition, scenario_count=150,
    )


def build_twin_stair_chained_core() -> TopologySpec:
    """THREE floors but a SHARED, SERIAL stair core: floor 3's stair
    lands on floor 2 (not ground), and floor 2's own stair then carries
    that same demand down to ground -- genuinely different CONNECTIVITY
    (chained/serial stairs) vs the base family's two fully independent,
    parallel per-floor-to-ground stairs."""

    ground = Floor(
        id="tshc-floor-ground", name="Ground", display_order=0,
        zones=[
            Zone(id="tshc-zone-lobby-a", name="Lobby A", floor_id="tshc-floor-ground", x=0, y=0, width=14, height=10),
            Zone(id="tshc-zone-lobby-b", name="Lobby B", floor_id="tshc-floor-ground", x=20, y=0, width=14, height=10),
        ],
        doors=[
            Door(id="tshc-door-ground", normally_open=True, zone_a_id="tshc-zone-lobby-a", zone_b_id="tshc-zone-lobby-b"),
        ],
        exits=[
            Exit(id="tshc-exit-1", zone_id="tshc-zone-lobby-a"),
            Exit(id="tshc-exit-2", zone_id="tshc-zone-lobby-b"),
        ],
    )

    floor2 = Floor(
        id="tshc-floor-2", name="Level 2", display_order=1,
        zones=[
            Zone(id="tshc-zone-2core", name="Level 2 Stair Core", floor_id="tshc-floor-2", x=0, y=0, width=16, height=10),
            Zone(id="tshc-zone-2b", name="Level 2 B", floor_id="tshc-floor-2", x=20, y=0, width=16, height=10),
        ],
        doors=[
            Door(id="tshc-door-2", normally_open=True, zone_a_id="tshc-zone-2core", zone_b_id="tshc-zone-2b"),
        ],
        stairs=[
            # Carries BOTH this floor's own demand AND everything handed
            # down from floor 3's stair below -- the chained-core point.
            Staircase(id="tshc-stair-2", from_zone_id="tshc-zone-2core", to_zone_id="tshc-zone-lobby-a",
                      from_floor_id="tshc-floor-2", to_floor_id="tshc-floor-ground"),
        ],
    )

    floor3 = Floor(
        id="tshc-floor-3", name="Level 3", display_order=2,
        zones=[
            Zone(id="tshc-zone-3a", name="Level 3 A", floor_id="tshc-floor-3", x=0, y=0, width=16, height=10),
            Zone(id="tshc-zone-3b", name="Level 3 B", floor_id="tshc-floor-3", x=20, y=0, width=16, height=10),
        ],
        doors=[
            Door(id="tshc-door-3", normally_open=True, zone_a_id="tshc-zone-3a", zone_b_id="tshc-zone-3b"),
        ],
        stairs=[
            # Lands on floor 2's stair-core zone, NOT ground -- serial
            # chaining, not an independent parallel path to ground.
            Staircase(id="tshc-stair-3", from_zone_id="tshc-zone-3a", to_zone_id="tshc-zone-2core",
                      from_floor_id="tshc-floor-3", to_floor_id="tshc-floor-2"),
        ],
    )

    building = Building(id="topology-twin-stair-chained-core", name="Twin-Stair Chained Core", floors=[ground, floor2, floor3])

    definition = ScenarioDefinition(
        fire=FireDefinition(
            growth_parameter_distribution=UniformRange(60.0, 400.0),
            ignition_zone_preference=WeightedOptions({
                "tshc-zone-lobby-a": 0.12, "tshc-zone-lobby-b": 0.12,
                "tshc-zone-2core": 0.2, "tshc-zone-2b": 0.16,
                "tshc-zone-3a": 0.2, "tshc-zone-3b": 0.2,
            }),
            allowed_fire_profiles=frozenset({"Flaming", "Smoldering"}),
        ),
        occupant=OccupantDefinition(
            occupancy_distribution={
                "tshc-zone-lobby-a": UniformRange(1, 8, discrete=True),
                "tshc-zone-lobby-b": UniformRange(1, 8, discrete=True),
                "tshc-zone-2core": UniformRange(5, 18, discrete=True),
                "tshc-zone-2b": UniformRange(5, 18, discrete=True),
                "tshc-zone-3a": UniformRange(10, 30, discrete=True),
                "tshc-zone-3b": UniformRange(5, 18, discrete=True),
            },
            behaviour_profile_distribution={
                "tshc-zone-lobby-a": WeightedOptions({"Adult_Default": 0.7, "Visitor_Default": 0.3}),
                "tshc-zone-lobby-b": WeightedOptions({"Adult_Default": 0.7, "Visitor_Default": 0.3}),
                "tshc-zone-2core": WeightedOptions({"Adult_Default": 0.65, "Staff_Default": 0.25, "Elderly_Default": 0.1}),
                "tshc-zone-2b": WeightedOptions({"Adult_Default": 0.7, "Staff_Default": 0.3}),
                "tshc-zone-3a": WeightedOptions({"Adult_Default": 0.65, "Staff_Default": 0.25, "Wheelchair_Default": 0.1}),
                "tshc-zone-3b": WeightedOptions({"Adult_Default": 0.7, "Staff_Default": 0.3}),
            },
            assistance_pairing_probability=0.15,
        ),
        engineering=EngineeringConstraints(
            door_state_distribution={
                did: WeightedOptions({"OPEN": 0.78, "CLOSED": 0.17, "LOCKED": 0.05})
                for did in ("tshc-door-ground", "tshc-door-2", "tshc-door-3")
            },
            exit_state_distribution={
                "tshc-exit-1": WeightedOptions({True: 0.9, False: 0.1}),
                "tshc-exit-2": WeightedOptions({True: 0.9, False: 0.1}),
            },
            min_open_exits=1,
            stair_state_distribution={
                "tshc-stair-2": WeightedOptions({"AVAILABLE": 0.85, "CLOSED": 0.15}),
                "tshc-stair-3": WeightedOptions({"AVAILABLE": 0.8, "CLOSED": 0.2}),
            },
        ),
        firefighter=FirefighterDeploymentDefinition(
            team_count_distribution=WeightedOptions({0: 0.4, 1: 0.35, 2: 0.25}),
            team_size_distribution=UniformRange(2, 4, discrete=True),
            arrival_time_distribution=UniformRange(45.0, 180.0),
            entry_zone_ids=("tshc-zone-lobby-a",),
            rescue_assignment_probability=0.6,
        ),
    )

    return TopologySpec(
        name="twin_stair_chained_core",
        description="Twin-stair family, chained-core variant: 3 floors, floor 3's stair lands on "
                    "floor 2 (not ground) which then carries it down via its own stair -- serial, "
                    "not parallel, stair connectivity.",
        building=building, definition=definition, scenario_count=150,
    )


# =====================================================
# Family 3 -- MULTI_EXIT_WIDE. Base (topologies_v2): hub-and-spoke, 5
# zones, 4 doors, 3 exits. v3.1 variant: 7 zones, 6 doors, 5 exits.
# =====================================================


def build_multi_exit_linear_chain() -> TopologySpec:
    """FOUR zones in a straight LINEAR chain (A-B-C-D), THREE doors in
    series, exits at BOTH ends only -- same exit count order-of-
    magnitude as the base family but LINEAR (not star/hub) connectivity,
    with genuinely asymmetric path lengths for interior zones."""

    floor1 = Floor(
        id="mewl-floor-1", name="Ground", display_order=0,
        zones=[
            Zone(id="mewl-zone-a", name="Wing A", floor_id="mewl-floor-1", x=0, y=0, width=14, height=10),
            Zone(id="mewl-zone-b", name="Wing B", floor_id="mewl-floor-1", x=16, y=0, width=14, height=10),
            Zone(id="mewl-zone-c", name="Wing C", floor_id="mewl-floor-1", x=32, y=0, width=14, height=10),
            Zone(id="mewl-zone-d", name="Wing D", floor_id="mewl-floor-1", x=48, y=0, width=14, height=10),
        ],
        doors=[
            Door(id="mewl-door-ab", normally_open=True, zone_a_id="mewl-zone-a", zone_b_id="mewl-zone-b"),
            Door(id="mewl-door-bc", normally_open=True, zone_a_id="mewl-zone-b", zone_b_id="mewl-zone-c"),
            Door(id="mewl-door-cd", normally_open=True, zone_a_id="mewl-zone-c", zone_b_id="mewl-zone-d"),
        ],
        exits=[
            Exit(id="mewl-exit-a", zone_id="mewl-zone-a"),
            Exit(id="mewl-exit-d", zone_id="mewl-zone-d"),
        ],
    )

    building = Building(id="topology-multi-exit-linear-chain", name="Multi-Exit Linear Chain", floors=[floor1])

    zone_ids = ("mewl-zone-a", "mewl-zone-b", "mewl-zone-c", "mewl-zone-d")

    definition = ScenarioDefinition(
        fire=FireDefinition(
            growth_parameter_distribution=UniformRange(60.0, 380.0),
            ignition_zone_preference=WeightedOptions({zid: 1.0 / len(zone_ids) for zid in zone_ids}),
            allowed_fire_profiles=frozenset({"Flaming", "Smoldering"}),
        ),
        occupant=OccupantDefinition(
            occupancy_distribution={
                "mewl-zone-a": UniformRange(2, 12, discrete=True),
                # Interior zones (b, c) are farthest from BOTH exits --
                # deliberately higher demand to stress the interior doors.
                "mewl-zone-b": UniformRange(6, 24, discrete=True),
                "mewl-zone-c": UniformRange(6, 24, discrete=True),
                "mewl-zone-d": UniformRange(2, 12, discrete=True),
            },
            behaviour_profile_distribution={
                zid: WeightedOptions({"Adult_Default": 0.65, "Staff_Default": 0.2, "Elderly_Default": 0.15})
                for zid in zone_ids
            },
            assistance_pairing_probability=0.2,
        ),
        engineering=EngineeringConstraints(
            door_state_distribution={
                did: WeightedOptions({"OPEN": 0.75, "CLOSED": 0.2, "LOCKED": 0.05})
                for did in ("mewl-door-ab", "mewl-door-bc", "mewl-door-cd")
            },
            exit_state_distribution={
                "mewl-exit-a": WeightedOptions({True: 0.88, False: 0.12}),
                "mewl-exit-d": WeightedOptions({True: 0.88, False: 0.12}),
            },
            min_open_exits=1,
        ),
        firefighter=FirefighterDeploymentDefinition(
            team_count_distribution=WeightedOptions({0: 0.45, 1: 0.35, 2: 0.2}),
            team_size_distribution=UniformRange(2, 4, discrete=True),
            arrival_time_distribution=UniformRange(45.0, 180.0),
            entry_zone_ids=("mewl-zone-a",),
            rescue_assignment_probability=0.6,
        ),
    )

    return TopologySpec(
        name="multi_exit_linear_chain",
        description="Multi-exit family, linear-chain variant: 4 zones in a row, 3 doors in series, "
                    "exits at both ends only -- LINEAR (not hub/star) connectivity, asymmetric path lengths.",
        building=building, definition=definition, scenario_count=150,
    )


def build_multi_exit_reduced_redundancy() -> TopologySpec:
    """Same hub-and-spoke SHAPE as the base family (5 zones, 4 doors),
    but only TWO of the four spokes have their own exit -- the other two
    wings must route back through the hub to reach an exit. Reduced
    route redundancy / asymmetric exit placement within the identical
    connectivity pattern."""

    floor1 = Floor(
        id="mewr-floor-1", name="Ground", display_order=0,
        zones=[
            Zone(id="mewr-zone-hub", name="Hub", floor_id="mewr-floor-1", x=20, y=10, width=14, height=10),
            Zone(id="mewr-zone-north", name="North Wing", floor_id="mewr-floor-1", x=20, y=0, width=14, height=8),
            Zone(id="mewr-zone-south", name="South Wing", floor_id="mewr-floor-1", x=20, y=24, width=14, height=8),
            Zone(id="mewr-zone-east", name="East Wing", floor_id="mewr-floor-1", x=40, y=10, width=14, height=10),
            Zone(id="mewr-zone-west", name="West Wing", floor_id="mewr-floor-1", x=0, y=10, width=14, height=10),
        ],
        doors=[
            Door(id="mewr-door-north", normally_open=True, zone_a_id="mewr-zone-hub", zone_b_id="mewr-zone-north"),
            Door(id="mewr-door-south", normally_open=True, zone_a_id="mewr-zone-hub", zone_b_id="mewr-zone-south"),
            Door(id="mewr-door-east", normally_open=True, zone_a_id="mewr-zone-hub", zone_b_id="mewr-zone-east"),
            Door(id="mewr-door-west", normally_open=True, zone_a_id="mewr-zone-hub", zone_b_id="mewr-zone-west"),
        ],
        exits=[
            # Only north/east get their own exit -- south/west wings have
            # NONE, must route back through the hub (unlike the base
            # family, where every spoke except the hub itself has one).
            Exit(id="mewr-exit-north", zone_id="mewr-zone-north"),
            Exit(id="mewr-exit-east", zone_id="mewr-zone-east"),
        ],
    )

    building = Building(id="topology-multi-exit-reduced-redundancy", name="Multi-Exit Reduced Redundancy", floors=[floor1])

    zone_ids = ("mewr-zone-hub", "mewr-zone-north", "mewr-zone-south", "mewr-zone-east", "mewr-zone-west")

    definition = ScenarioDefinition(
        fire=FireDefinition(
            growth_parameter_distribution=UniformRange(60.0, 380.0),
            ignition_zone_preference=WeightedOptions({
                "mewr-zone-hub": 0.3, "mewr-zone-north": 0.175, "mewr-zone-south": 0.175,
                "mewr-zone-east": 0.175, "mewr-zone-west": 0.175,
            }),
            allowed_fire_profiles=frozenset({"Flaming", "Smoldering"}),
        ),
        occupant=OccupantDefinition(
            occupancy_distribution={
                "mewr-zone-hub": UniformRange(2, 12, discrete=True),
                "mewr-zone-north": UniformRange(4, 22, discrete=True),
                # South/west (no direct exit) deliberately carry demand
                # so the hub-routing consequence is a real stress case.
                "mewr-zone-south": UniformRange(6, 24, discrete=True),
                "mewr-zone-east": UniformRange(4, 22, discrete=True),
                "mewr-zone-west": UniformRange(6, 24, discrete=True),
            },
            behaviour_profile_distribution={
                zid: WeightedOptions({"Adult_Default": 0.65, "Staff_Default": 0.2, "Elderly_Default": 0.15})
                for zid in zone_ids
            },
            assistance_pairing_probability=0.2,
        ),
        engineering=EngineeringConstraints(
            door_state_distribution={
                did: WeightedOptions({"OPEN": 0.75, "CLOSED": 0.2, "LOCKED": 0.05})
                for did in ("mewr-door-north", "mewr-door-south", "mewr-door-east", "mewr-door-west")
            },
            exit_state_distribution={
                "mewr-exit-north": WeightedOptions({True: 0.88, False: 0.12}),
                "mewr-exit-east": WeightedOptions({True: 0.88, False: 0.12}),
            },
            min_open_exits=1,
        ),
        firefighter=FirefighterDeploymentDefinition(
            team_count_distribution=WeightedOptions({0: 0.45, 1: 0.35, 2: 0.2}),
            team_size_distribution=UniformRange(2, 4, discrete=True),
            arrival_time_distribution=UniformRange(45.0, 180.0),
            entry_zone_ids=("mewr-zone-hub",),
            rescue_assignment_probability=0.6,
        ),
    )

    return TopologySpec(
        name="multi_exit_reduced_redundancy",
        description="Multi-exit family, reduced-redundancy variant: same hub-and-spoke shape as the "
                    "base family but only 2 of 4 spokes have their own exit -- asymmetric exit "
                    "placement, reduced route redundancy.",
        building=building, definition=definition, scenario_count=150,
    )


# =====================================================
# Family 4 -- V1_TOPOLOGY_FIXED. Base (topologies_v2): 2 floors, 3
# ground zones in series, 2 doors, 2 exits, 1 stair.
# =====================================================


def build_v1_fixed_dual_stair() -> TopologySpec:
    """Same ground layout as the base family, but TWO stairs (from
    BOTH office-a and office-b) converge on the same upper zone --
    stair_count=2, a genuine parallel-route addition the base never has."""

    floor1 = Floor(
        id="v1fd-floor-1", name="Ground", display_order=0,
        zones=[
            Zone(id="v1fd-zone-lobby", name="Lobby", floor_id="v1fd-floor-1", x=0, y=0, width=12, height=8),
            Zone(id="v1fd-zone-office-a", name="Office A", floor_id="v1fd-floor-1", x=20, y=0, width=8, height=8),
            Zone(id="v1fd-zone-office-b", name="Office B", floor_id="v1fd-floor-1", x=32, y=0, width=8, height=8),
        ],
        doors=[
            Door(id="v1fd-door-1", normally_open=True, zone_a_id="v1fd-zone-lobby", zone_b_id="v1fd-zone-office-a"),
            Door(id="v1fd-door-2", normally_open=True, zone_a_id="v1fd-zone-office-a", zone_b_id="v1fd-zone-office-b"),
        ],
        exits=[
            Exit(id="v1fd-exit-1", zone_id="v1fd-zone-lobby"),
            Exit(id="v1fd-exit-2", zone_id="v1fd-zone-office-b"),
        ],
        stairs=[
            Staircase(id="v1fd-stair-1", from_zone_id="v1fd-zone-office-a", to_zone_id="v1fd-zone-upper",
                      from_floor_id="v1fd-floor-1", to_floor_id="v1fd-floor-2"),
            Staircase(id="v1fd-stair-2", from_zone_id="v1fd-zone-office-b", to_zone_id="v1fd-zone-upper",
                      from_floor_id="v1fd-floor-1", to_floor_id="v1fd-floor-2"),
        ],
    )

    floor2 = Floor(
        id="v1fd-floor-2", name="Upper", display_order=1,
        zones=[
            Zone(id="v1fd-zone-upper", name="Upper Hall", floor_id="v1fd-floor-2", x=0, y=0, width=16, height=10),
        ],
    )

    building = Building(id="topology-v1-fixed-dual-stair", name="V1 Topology Dual Stair", floors=[floor1, floor2])

    definition = ScenarioDefinition(
        fire=FireDefinition(
            growth_parameter_distribution=UniformRange(80.0, 420.0),
            ignition_zone_preference=WeightedOptions({
                "v1fd-zone-lobby": 0.25, "v1fd-zone-office-a": 0.25, "v1fd-zone-office-b": 0.25, "v1fd-zone-upper": 0.25,
            }),
            allowed_fire_profiles=frozenset({"Flaming", "Smoldering"}),
        ),
        occupant=OccupantDefinition(
            occupancy_distribution={
                "v1fd-zone-lobby": UniformRange(2, 10, discrete=True),
                "v1fd-zone-office-a": UniformRange(1, 8, discrete=True),
                "v1fd-zone-office-b": UniformRange(1, 8, discrete=True),
                "v1fd-zone-upper": UniformRange(4, 16, discrete=True),
            },
            behaviour_profile_distribution={
                "v1fd-zone-lobby": WeightedOptions({
                    "Adult_Default": 0.55, "Elderly_Default": 0.15, "Wheelchair_Default": 0.1,
                    "Child_Default": 0.1, "Visitor_Default": 0.1,
                }),
                "v1fd-zone-office-a": WeightedOptions({"Adult_Default": 0.7, "Staff_Default": 0.3}),
                "v1fd-zone-office-b": WeightedOptions({"Adult_Default": 0.7, "Staff_Default": 0.3}),
                "v1fd-zone-upper": WeightedOptions({"Adult_Default": 0.6, "Visitor_Default": 0.4}),
            },
            assistance_pairing_probability=0.2,
        ),
        engineering=EngineeringConstraints(
            door_state_distribution={
                "v1fd-door-1": WeightedOptions({"OPEN": 0.7, "CLOSED": 0.25, "LOCKED": 0.05}),
                "v1fd-door-2": WeightedOptions({"OPEN": 0.7, "CLOSED": 0.25, "LOCKED": 0.05}),
            },
            exit_state_distribution={
                "v1fd-exit-1": WeightedOptions({True: 0.9, False: 0.1}),
                "v1fd-exit-2": WeightedOptions({True: 0.85, False: 0.15}),
            },
            min_open_exits=1,
            stair_state_distribution={
                "v1fd-stair-1": WeightedOptions({"AVAILABLE": 0.85, "CLOSED": 0.15}),
                "v1fd-stair-2": WeightedOptions({"AVAILABLE": 0.85, "CLOSED": 0.15}),
            },
        ),
        firefighter=FirefighterDeploymentDefinition(
            team_count_distribution=WeightedOptions({0: 0.5, 1: 0.35, 2: 0.15}),
            team_size_distribution=UniformRange(2, 4, discrete=True),
            arrival_time_distribution=UniformRange(45.0, 180.0),
            entry_zone_ids=("v1fd-zone-lobby",),
            rescue_assignment_probability=0.6,
        ),
    )

    return TopologySpec(
        name="v1_fixed_dual_stair",
        description="V1-fixed family, dual-stair variant: same ground layout, but TWO stairs from "
                    "office-a AND office-b converge on the same upper zone -- stair_count=2.",
        building=building, definition=definition, scenario_count=150,
    )


def build_v1_fixed_three_floor() -> TopologySpec:
    """THREE floors, chained vertically (ground -> mid -> upper), each
    connected by its own stair -- deeper vertical structure than the
    base family's single ground->upper hop."""

    floor1 = Floor(
        id="v1ft-floor-1", name="Ground", display_order=0,
        zones=[
            Zone(id="v1ft-zone-lobby", name="Lobby", floor_id="v1ft-floor-1", x=0, y=0, width=12, height=8),
            Zone(id="v1ft-zone-office-a", name="Office A", floor_id="v1ft-floor-1", x=20, y=0, width=8, height=8),
        ],
        doors=[
            Door(id="v1ft-door-1", normally_open=True, zone_a_id="v1ft-zone-lobby", zone_b_id="v1ft-zone-office-a"),
        ],
        exits=[
            Exit(id="v1ft-exit-1", zone_id="v1ft-zone-lobby"),
        ],
        stairs=[
            Staircase(id="v1ft-stair-1", from_zone_id="v1ft-zone-office-a", to_zone_id="v1ft-zone-mid",
                      from_floor_id="v1ft-floor-1", to_floor_id="v1ft-floor-2"),
        ],
    )

    floor2 = Floor(
        id="v1ft-floor-2", name="Mid", display_order=1,
        zones=[
            Zone(id="v1ft-zone-mid", name="Mid Hall", floor_id="v1ft-floor-2", x=0, y=0, width=16, height=10),
        ],
        stairs=[
            Staircase(id="v1ft-stair-2", from_zone_id="v1ft-zone-mid", to_zone_id="v1ft-zone-upper",
                      from_floor_id="v1ft-floor-2", to_floor_id="v1ft-floor-3"),
        ],
    )

    floor3 = Floor(
        id="v1ft-floor-3", name="Upper", display_order=2,
        zones=[
            Zone(id="v1ft-zone-upper", name="Upper Hall", floor_id="v1ft-floor-3", x=0, y=0, width=16, height=10),
        ],
    )

    building = Building(id="topology-v1-fixed-three-floor", name="V1 Topology Three Floor", floors=[floor1, floor2, floor3])

    definition = ScenarioDefinition(
        fire=FireDefinition(
            growth_parameter_distribution=UniformRange(80.0, 420.0),
            ignition_zone_preference=WeightedOptions({
                "v1ft-zone-lobby": 0.2, "v1ft-zone-office-a": 0.2, "v1ft-zone-mid": 0.3, "v1ft-zone-upper": 0.3,
            }),
            allowed_fire_profiles=frozenset({"Flaming", "Smoldering"}),
        ),
        occupant=OccupantDefinition(
            occupancy_distribution={
                "v1ft-zone-lobby": UniformRange(2, 10, discrete=True),
                "v1ft-zone-office-a": UniformRange(1, 8, discrete=True),
                "v1ft-zone-mid": UniformRange(2, 12, discrete=True),
                "v1ft-zone-upper": UniformRange(4, 16, discrete=True),
            },
            behaviour_profile_distribution={
                "v1ft-zone-lobby": WeightedOptions({
                    "Adult_Default": 0.55, "Elderly_Default": 0.15, "Wheelchair_Default": 0.1,
                    "Child_Default": 0.1, "Visitor_Default": 0.1,
                }),
                "v1ft-zone-office-a": WeightedOptions({"Adult_Default": 0.7, "Staff_Default": 0.3}),
                "v1ft-zone-mid": WeightedOptions({"Adult_Default": 0.65, "Staff_Default": 0.35}),
                "v1ft-zone-upper": WeightedOptions({"Adult_Default": 0.6, "Visitor_Default": 0.4}),
            },
            assistance_pairing_probability=0.2,
        ),
        engineering=EngineeringConstraints(
            door_state_distribution={
                "v1ft-door-1": WeightedOptions({"OPEN": 0.7, "CLOSED": 0.25, "LOCKED": 0.05}),
            },
            exit_state_distribution={
                "v1ft-exit-1": WeightedOptions({True: 0.9, False: 0.1}),
            },
            min_open_exits=1,
            stair_state_distribution={
                "v1ft-stair-1": WeightedOptions({"AVAILABLE": 0.85, "CLOSED": 0.15}),
                "v1ft-stair-2": WeightedOptions({"AVAILABLE": 0.85, "CLOSED": 0.15}),
            },
        ),
        firefighter=FirefighterDeploymentDefinition(
            team_count_distribution=WeightedOptions({0: 0.5, 1: 0.35, 2: 0.15}),
            team_size_distribution=UniformRange(2, 4, discrete=True),
            arrival_time_distribution=UniformRange(45.0, 180.0),
            entry_zone_ids=("v1ft-zone-lobby",),
            rescue_assignment_probability=0.6,
        ),
    )

    return TopologySpec(
        name="v1_fixed_three_floor",
        description="V1-fixed family, three-floor variant: ground -> mid -> upper, chained vertically "
                    "with two stairs in series -- deeper vertical structure than the base 2-floor family.",
        building=building, definition=definition, scenario_count=150,
    )


def build_v1_fixed_long_corridor() -> TopologySpec:
    """Ground floor with FOUR offices chained in series (lobby ->
    office-a -> office-b -> office-c -> office-d), FOUR doors, exits at
    the lobby AND the far end (office-d), stair off office-b -- a much
    longer/deeper ground-floor corridor than the base family's 2-door
    layout, with the stair placed mid-corridor rather than near the exit."""

    floor1 = Floor(
        id="v1flc-floor-1", name="Ground", display_order=0,
        zones=[
            Zone(id="v1flc-zone-lobby", name="Lobby", floor_id="v1flc-floor-1", x=0, y=0, width=10, height=8),
            Zone(id="v1flc-zone-office-a", name="Office A", floor_id="v1flc-floor-1", x=14, y=0, width=10, height=8),
            Zone(id="v1flc-zone-office-b", name="Office B", floor_id="v1flc-floor-1", x=28, y=0, width=10, height=8),
            Zone(id="v1flc-zone-office-c", name="Office C", floor_id="v1flc-floor-1", x=42, y=0, width=10, height=8),
            Zone(id="v1flc-zone-office-d", name="Office D", floor_id="v1flc-floor-1", x=56, y=0, width=10, height=8),
        ],
        doors=[
            Door(id="v1flc-door-1", normally_open=True, zone_a_id="v1flc-zone-lobby", zone_b_id="v1flc-zone-office-a"),
            Door(id="v1flc-door-2", normally_open=True, zone_a_id="v1flc-zone-office-a", zone_b_id="v1flc-zone-office-b"),
            Door(id="v1flc-door-3", normally_open=True, zone_a_id="v1flc-zone-office-b", zone_b_id="v1flc-zone-office-c"),
            Door(id="v1flc-door-4", normally_open=True, zone_a_id="v1flc-zone-office-c", zone_b_id="v1flc-zone-office-d"),
        ],
        exits=[
            Exit(id="v1flc-exit-1", zone_id="v1flc-zone-lobby"),
            Exit(id="v1flc-exit-2", zone_id="v1flc-zone-office-d"),
        ],
        stairs=[
            Staircase(id="v1flc-stair-1", from_zone_id="v1flc-zone-office-b", to_zone_id="v1flc-zone-upper",
                      from_floor_id="v1flc-floor-1", to_floor_id="v1flc-floor-2"),
        ],
    )

    floor2 = Floor(
        id="v1flc-floor-2", name="Upper", display_order=1,
        zones=[
            Zone(id="v1flc-zone-upper", name="Upper Hall", floor_id="v1flc-floor-2", x=0, y=0, width=16, height=10),
        ],
    )

    building = Building(id="topology-v1-fixed-long-corridor", name="V1 Topology Long Corridor", floors=[floor1, floor2])

    zone_ids = ("v1flc-zone-lobby", "v1flc-zone-office-a", "v1flc-zone-office-b", "v1flc-zone-office-c", "v1flc-zone-office-d")

    definition = ScenarioDefinition(
        fire=FireDefinition(
            growth_parameter_distribution=UniformRange(80.0, 420.0),
            ignition_zone_preference=WeightedOptions({zid: 1.0 / len(zone_ids) for zid in zone_ids}),
            allowed_fire_profiles=frozenset({"Flaming", "Smoldering"}),
        ),
        occupant=OccupantDefinition(
            occupancy_distribution={
                "v1flc-zone-lobby": UniformRange(2, 10, discrete=True),
                "v1flc-zone-office-a": UniformRange(1, 8, discrete=True),
                "v1flc-zone-office-b": UniformRange(1, 8, discrete=True),
                "v1flc-zone-office-c": UniformRange(1, 8, discrete=True),
                "v1flc-zone-office-d": UniformRange(1, 8, discrete=True),
            },
            behaviour_profile_distribution={
                zid: WeightedOptions({"Adult_Default": 0.7, "Staff_Default": 0.3}) for zid in zone_ids
            },
            assistance_pairing_probability=0.2,
        ),
        engineering=EngineeringConstraints(
            door_state_distribution={
                did: WeightedOptions({"OPEN": 0.7, "CLOSED": 0.25, "LOCKED": 0.05})
                for did in ("v1flc-door-1", "v1flc-door-2", "v1flc-door-3", "v1flc-door-4")
            },
            exit_state_distribution={
                "v1flc-exit-1": WeightedOptions({True: 0.9, False: 0.1}),
                "v1flc-exit-2": WeightedOptions({True: 0.85, False: 0.15}),
            },
            min_open_exits=1,
            stair_state_distribution={
                "v1flc-stair-1": WeightedOptions({"AVAILABLE": 0.85, "CLOSED": 0.15}),
            },
        ),
        firefighter=FirefighterDeploymentDefinition(
            team_count_distribution=WeightedOptions({0: 0.5, 1: 0.35, 2: 0.15}),
            team_size_distribution=UniformRange(2, 4, discrete=True),
            arrival_time_distribution=UniformRange(45.0, 180.0),
            entry_zone_ids=("v1flc-zone-lobby",),
            rescue_assignment_probability=0.6,
        ),
    )

    return TopologySpec(
        name="v1_fixed_long_corridor",
        description="V1-fixed family, long-corridor variant: 5 zones/4 doors in series on the ground "
                    "floor, exits at both ends, stair placed mid-corridor -- longer/deeper serial "
                    "connectivity than the base 3-zone/2-door layout.",
        building=building, definition=definition, scenario_count=150,
    )


# =====================================================
# Registry -- 4 families x 4 structural variants = 16 templates.
# =====================================================

CANONICAL_FAMILIES: Tuple[str, ...] = (
    "single_exit_lowrise", "twin_stair_highrise", "multi_exit_wide", "v1_topology_fixed",
)


def all_structural_variants_v3() -> Tuple[StructuralVariant, ...]:

    return (
        # Family 1 -- single_exit_lowrise
        StructuralVariant("single_exit_lowrise", "single_exit_lowrise", "base", build_single_exit_lowrise()),
        StructuralVariant("single_exit_lowrise", "single_exit_deep_corridor", "deep_corridor", build_single_exit_deep_corridor()),
        StructuralVariant("single_exit_lowrise", "single_exit_branching_deadends", "branching_deadends", build_single_exit_branching_deadends()),
        StructuralVariant("single_exit_lowrise", "single_exit_vertical", "vertical", build_single_exit_vertical()),

        # Family 2 -- twin_stair_highrise
        StructuralVariant("twin_stair_highrise", "twin_stair_highrise", "base", build_twin_stair_highrise()),
        StructuralVariant("twin_stair_highrise", "twin_stair_highrise_3stair", "3stair", build_twin_stair_highrise_3stair()),
        StructuralVariant("twin_stair_highrise", "twin_stair_low", "low", build_twin_stair_low()),
        StructuralVariant("twin_stair_highrise", "twin_stair_chained_core", "chained_core", build_twin_stair_chained_core()),

        # Family 3 -- multi_exit_wide
        StructuralVariant("multi_exit_wide", "multi_exit_wide", "base", build_multi_exit_wide()),
        StructuralVariant("multi_exit_wide", "multi_exit_wide_6exit", "6exit", build_multi_exit_wide_6exit()),
        StructuralVariant("multi_exit_wide", "multi_exit_linear_chain", "linear_chain", build_multi_exit_linear_chain()),
        StructuralVariant("multi_exit_wide", "multi_exit_reduced_redundancy", "reduced_redundancy", build_multi_exit_reduced_redundancy()),

        # Family 4 -- v1_topology_fixed
        StructuralVariant("v1_topology_fixed", "v1_topology_fixed", "base", build_v1_topology_fixed()),
        StructuralVariant("v1_topology_fixed", "v1_fixed_dual_stair", "dual_stair", build_v1_fixed_dual_stair()),
        StructuralVariant("v1_topology_fixed", "v1_fixed_three_floor", "three_floor", build_v1_fixed_three_floor()),
        StructuralVariant("v1_topology_fixed", "v1_fixed_long_corridor", "long_corridor", build_v1_fixed_long_corridor()),
    )
