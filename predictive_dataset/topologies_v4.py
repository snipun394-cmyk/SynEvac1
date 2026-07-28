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

from predictive_dataset.topologies_v2 import TopologySpec
from predictive_dataset.topologies_v3 import StructuralVariant, all_structural_variants_v3


# =====================================================
# Predictive Dataset V4 milestone, Phase 5-8 -- TWO genuinely new
# topology families, chosen from a QUANTITATIVE structural-coverage-gap
# analysis (predictive_dataset/topology_signature_v4.py run against all
# 16 Dataset V3 structural variants -- see docs/architecture/
# predictive_dataset_campaign_v4.md Phase 5 for the full table), not
# preference:
#
#   - ZONE-ONLY CYCLES (genuine physical route redundancy between
#     zones, NOT mediated by routing out through the shared OUTSIDE
#     node and back in via a different exit): 15 of 16 V3 variants have
#     ZERO. Only `v1_fixed_dual_stair` has exactly one, as a side
#     effect of two stairs converging on the same zone, not a designed
#     ring. -> RING_CORRIDOR directly targets this gap.
#
#   - WING DEPTH + a SHARED FUNNELING CONNECTOR: V3's closest analog,
#     `multi_exit_wide`, is a hub with SHALLOW (single-zone) spokes,
#     each carrying its OWN direct exit -- symmetric, every candidate a
#     bridge, no catchment concentration. No V3 variant combines
#     multi-zone-deep wings with a SHARED connector multiple wings
#     funnel through. -> MULTI_WING directly targets this gap.
#
# A third family was investigated (Phase 8) and NOT added: every other
# structural dimension topology_signature_v4 measures (corridor depth
# 2-4, bridge fraction 0.00-1.00, exit asymmetry 0.00-0.67, zone degree
# up to 6) is already broadly covered by V3's 16 variants combined with
# what these two new families naturally add -- no comparably stark gap
# (e.g. 15/16 == 0) was found elsewhere. Per the milestone charter's own
# "do not add a family merely to increase the count" instruction.
#
# 4 genuinely distinct variants per family (8 new total), same
# authoring discipline topologies_v3.py established: every Stair sets
# BOTH from_floor_id/to_floor_id, every Staircase is registered on its
# OWN (from_floor) floor's stairs list (navigation/graph_builder.py's
# _add_stair_edges resolves from_zone against the floor iterating
# floor.stairs).
# =====================================================


CANONICAL_FAMILIES_V4: Tuple[str, ...] = ("multi_wing", "ring_corridor")


# =====================================================
# Family 5 -- MULTI_WING. Base: 1 core zone (shared circulation) + 3
# wings, each 2 zones DEEP (near/far), funneling through ONE shared
# exit on the core.
# =====================================================


def build_multi_wing() -> TopologySpec:
    """Core + 3 wings (each 2 zones deep: near/far), ONE shared exit on
    the core -- every wing's demand funnels through the SAME core-exit
    connector, unlike multi_exit_wide's shallow, symmetric, per-spoke
    exits."""

    floor1 = Floor(
        id="mw-floor-1", name="Ground", display_order=0,
        zones=[
            Zone(id="mw-core", name="Core", floor_id="mw-floor-1", x=0, y=20, width=14, height=12),
            Zone(id="mw-wingA-near", name="Wing A Near", floor_id="mw-floor-1", x=-30, y=0, width=12, height=10),
            Zone(id="mw-wingA-far", name="Wing A Far", floor_id="mw-floor-1", x=-50, y=0, width=12, height=10),
            Zone(id="mw-wingB-near", name="Wing B Near", floor_id="mw-floor-1", x=30, y=0, width=12, height=10),
            Zone(id="mw-wingB-far", name="Wing B Far", floor_id="mw-floor-1", x=50, y=0, width=12, height=10),
            Zone(id="mw-wingC-near", name="Wing C Near", floor_id="mw-floor-1", x=0, y=45, width=12, height=10),
            Zone(id="mw-wingC-far", name="Wing C Far", floor_id="mw-floor-1", x=0, y=65, width=12, height=10),
        ],
        doors=[
            Door(id="mw-door-core-a", normally_open=True, zone_a_id="mw-core", zone_b_id="mw-wingA-near"),
            Door(id="mw-door-a-far", normally_open=True, zone_a_id="mw-wingA-near", zone_b_id="mw-wingA-far"),
            Door(id="mw-door-core-b", normally_open=True, zone_a_id="mw-core", zone_b_id="mw-wingB-near"),
            Door(id="mw-door-b-far", normally_open=True, zone_a_id="mw-wingB-near", zone_b_id="mw-wingB-far"),
            Door(id="mw-door-core-c", normally_open=True, zone_a_id="mw-core", zone_b_id="mw-wingC-near"),
            Door(id="mw-door-c-far", normally_open=True, zone_a_id="mw-wingC-near", zone_b_id="mw-wingC-far"),
        ],
        exits=[Exit(id="mw-exit-1", zone_id="mw-core")],
    )

    building = Building(id="topology-multi-wing", name="Multi-Wing", floors=[floor1])

    zone_ids = ("mw-core", "mw-wingA-near", "mw-wingA-far", "mw-wingB-near", "mw-wingB-far", "mw-wingC-near", "mw-wingC-far")
    door_ids = ("mw-door-core-a", "mw-door-a-far", "mw-door-core-b", "mw-door-b-far", "mw-door-core-c", "mw-door-c-far")

    definition = ScenarioDefinition(
        fire=FireDefinition(
            growth_parameter_distribution=UniformRange(60.0, 380.0),
            ignition_zone_preference=WeightedOptions({zid: 1.0 / len(zone_ids) for zid in zone_ids}),
            allowed_fire_profiles=frozenset({"Flaming", "Smoldering"}),
        ),
        occupant=OccupantDefinition(
            occupancy_distribution={
                "mw-core": UniformRange(2, 10, discrete=True),
                "mw-wingA-near": UniformRange(3, 14, discrete=True),
                "mw-wingA-far": UniformRange(3, 14, discrete=True),
                "mw-wingB-near": UniformRange(3, 14, discrete=True),
                "mw-wingB-far": UniformRange(3, 14, discrete=True),
                "mw-wingC-near": UniformRange(3, 14, discrete=True),
                "mw-wingC-far": UniformRange(3, 14, discrete=True),
            },
            behaviour_profile_distribution={
                zid: WeightedOptions({"Adult_Default": 0.65, "Staff_Default": 0.2, "Elderly_Default": 0.15})
                for zid in zone_ids
            },
            assistance_pairing_probability=0.2,
        ),
        engineering=EngineeringConstraints(
            door_state_distribution={did: WeightedOptions({"OPEN": 0.75, "CLOSED": 0.2, "LOCKED": 0.05}) for did in door_ids},
            exit_state_distribution={"mw-exit-1": WeightedOptions({True: 0.9, False: 0.1})},
            min_open_exits=0,
        ),
        firefighter=FirefighterDeploymentDefinition(
            team_count_distribution=WeightedOptions({0: 0.45, 1: 0.35, 2: 0.2}),
            team_size_distribution=UniformRange(2, 4, discrete=True),
            arrival_time_distribution=UniformRange(45.0, 180.0),
            entry_zone_ids=("mw-core",),
            rescue_assignment_probability=0.6,
        ),
    )

    return TopologySpec(
        name="multi_wing",
        description="Multi-wing family base: core + 3 wings (2 zones deep each), ONE shared "
                    "core exit every wing funnels through.",
        building=building, definition=definition, scenario_count=150,
    )


def build_multi_wing_asymmetric_exits() -> TopologySpec:
    """Core itself is now 2 zones DEEP (core-a/core-b, in series), each
    with its OWN exit; wings attach ASYMMETRICALLY (1 wing on core-a, 2
    wings on core-b) -- real exit-catchment imbalance, unlike the base
    variant's single shared exit."""

    floor1 = Floor(
        id="mw2-floor-1", name="Ground", display_order=0,
        zones=[
            Zone(id="mw2-core-a", name="Core A", floor_id="mw2-floor-1", x=0, y=20, width=14, height=12),
            Zone(id="mw2-core-b", name="Core B", floor_id="mw2-floor-1", x=20, y=20, width=14, height=12),
            Zone(id="mw2-wingA-near", name="Wing A Near", floor_id="mw2-floor-1", x=-30, y=0, width=12, height=10),
            Zone(id="mw2-wingA-far", name="Wing A Far", floor_id="mw2-floor-1", x=-50, y=0, width=12, height=10),
            Zone(id="mw2-wingB-near", name="Wing B Near", floor_id="mw2-floor-1", x=40, y=0, width=12, height=10),
            Zone(id="mw2-wingB-far", name="Wing B Far", floor_id="mw2-floor-1", x=60, y=0, width=12, height=10),
            Zone(id="mw2-wingC-near", name="Wing C Near", floor_id="mw2-floor-1", x=20, y=45, width=12, height=10),
            Zone(id="mw2-wingC-far", name="Wing C Far", floor_id="mw2-floor-1", x=20, y=65, width=12, height=10),
        ],
        doors=[
            Door(id="mw2-door-core", normally_open=True, zone_a_id="mw2-core-a", zone_b_id="mw2-core-b"),
            Door(id="mw2-door-core-a-wing", normally_open=True, zone_a_id="mw2-core-a", zone_b_id="mw2-wingA-near"),
            Door(id="mw2-door-a-far", normally_open=True, zone_a_id="mw2-wingA-near", zone_b_id="mw2-wingA-far"),
            Door(id="mw2-door-core-b-wingb", normally_open=True, zone_a_id="mw2-core-b", zone_b_id="mw2-wingB-near"),
            Door(id="mw2-door-b-far", normally_open=True, zone_a_id="mw2-wingB-near", zone_b_id="mw2-wingB-far"),
            Door(id="mw2-door-core-b-wingc", normally_open=True, zone_a_id="mw2-core-b", zone_b_id="mw2-wingC-near"),
            Door(id="mw2-door-c-far", normally_open=True, zone_a_id="mw2-wingC-near", zone_b_id="mw2-wingC-far"),
        ],
        exits=[
            Exit(id="mw2-exit-a", zone_id="mw2-core-a"),
            Exit(id="mw2-exit-b", zone_id="mw2-core-b"),
        ],
    )

    building = Building(id="topology-multi-wing-asymmetric-exits", name="Multi-Wing Asymmetric Exits", floors=[floor1])

    zone_ids = (
        "mw2-core-a", "mw2-core-b", "mw2-wingA-near", "mw2-wingA-far",
        "mw2-wingB-near", "mw2-wingB-far", "mw2-wingC-near", "mw2-wingC-far",
    )
    door_ids = (
        "mw2-door-core", "mw2-door-core-a-wing", "mw2-door-a-far",
        "mw2-door-core-b-wingb", "mw2-door-b-far", "mw2-door-core-b-wingc", "mw2-door-c-far",
    )

    definition = ScenarioDefinition(
        fire=FireDefinition(
            growth_parameter_distribution=UniformRange(60.0, 380.0),
            ignition_zone_preference=WeightedOptions({zid: 1.0 / len(zone_ids) for zid in zone_ids}),
            allowed_fire_profiles=frozenset({"Flaming", "Smoldering"}),
        ),
        occupant=OccupantDefinition(
            occupancy_distribution={zid: UniformRange(3, 16, discrete=True) for zid in zone_ids},
            behaviour_profile_distribution={
                zid: WeightedOptions({"Adult_Default": 0.65, "Staff_Default": 0.2, "Elderly_Default": 0.15})
                for zid in zone_ids
            },
            assistance_pairing_probability=0.2,
        ),
        engineering=EngineeringConstraints(
            door_state_distribution={did: WeightedOptions({"OPEN": 0.75, "CLOSED": 0.2, "LOCKED": 0.05}) for did in door_ids},
            exit_state_distribution={
                "mw2-exit-a": WeightedOptions({True: 0.85, False: 0.15}),
                "mw2-exit-b": WeightedOptions({True: 0.85, False: 0.15}),
            },
            min_open_exits=1,
        ),
        firefighter=FirefighterDeploymentDefinition(
            team_count_distribution=WeightedOptions({0: 0.45, 1: 0.35, 2: 0.2}),
            team_size_distribution=UniformRange(2, 4, discrete=True),
            arrival_time_distribution=UniformRange(45.0, 180.0),
            entry_zone_ids=("mw2-core-a",),
            rescue_assignment_probability=0.6,
        ),
    )

    return TopologySpec(
        name="multi_wing_asymmetric_exits",
        description="Multi-wing variant: 2-zone-deep core (core-a/core-b) each with its own exit, "
                    "wings attached asymmetrically (1 on core-a, 2 on core-b) -- real exit-catchment imbalance.",
        building=building, definition=definition, scenario_count=150,
    )


def build_multi_wing_four_wings() -> TopologySpec:
    """4 wings (instead of 3), each 2 zones deep, ONE shared core exit --
    higher branching factor at the core than the base variant."""

    floor1 = Floor(
        id="mw4-floor-1", name="Ground", display_order=0,
        zones=[
            Zone(id="mw4-core", name="Core", floor_id="mw4-floor-1", x=0, y=20, width=14, height=12),
            Zone(id="mw4-wingA-near", name="Wing A Near", floor_id="mw4-floor-1", x=-30, y=0, width=12, height=10),
            Zone(id="mw4-wingA-far", name="Wing A Far", floor_id="mw4-floor-1", x=-50, y=0, width=12, height=10),
            Zone(id="mw4-wingB-near", name="Wing B Near", floor_id="mw4-floor-1", x=30, y=0, width=12, height=10),
            Zone(id="mw4-wingB-far", name="Wing B Far", floor_id="mw4-floor-1", x=50, y=0, width=12, height=10),
            Zone(id="mw4-wingC-near", name="Wing C Near", floor_id="mw4-floor-1", x=-15, y=45, width=12, height=10),
            Zone(id="mw4-wingC-far", name="Wing C Far", floor_id="mw4-floor-1", x=-15, y=65, width=12, height=10),
            Zone(id="mw4-wingD-near", name="Wing D Near", floor_id="mw4-floor-1", x=15, y=45, width=12, height=10),
            Zone(id="mw4-wingD-far", name="Wing D Far", floor_id="mw4-floor-1", x=15, y=65, width=12, height=10),
        ],
        doors=[
            Door(id="mw4-door-core-a", normally_open=True, zone_a_id="mw4-core", zone_b_id="mw4-wingA-near"),
            Door(id="mw4-door-a-far", normally_open=True, zone_a_id="mw4-wingA-near", zone_b_id="mw4-wingA-far"),
            Door(id="mw4-door-core-b", normally_open=True, zone_a_id="mw4-core", zone_b_id="mw4-wingB-near"),
            Door(id="mw4-door-b-far", normally_open=True, zone_a_id="mw4-wingB-near", zone_b_id="mw4-wingB-far"),
            Door(id="mw4-door-core-c", normally_open=True, zone_a_id="mw4-core", zone_b_id="mw4-wingC-near"),
            Door(id="mw4-door-c-far", normally_open=True, zone_a_id="mw4-wingC-near", zone_b_id="mw4-wingC-far"),
            Door(id="mw4-door-core-d", normally_open=True, zone_a_id="mw4-core", zone_b_id="mw4-wingD-near"),
            Door(id="mw4-door-d-far", normally_open=True, zone_a_id="mw4-wingD-near", zone_b_id="mw4-wingD-far"),
        ],
        exits=[Exit(id="mw4-exit-1", zone_id="mw4-core")],
    )

    building = Building(id="topology-multi-wing-four-wings", name="Multi-Wing Four Wings", floors=[floor1])

    zone_ids = (
        "mw4-core", "mw4-wingA-near", "mw4-wingA-far", "mw4-wingB-near", "mw4-wingB-far",
        "mw4-wingC-near", "mw4-wingC-far", "mw4-wingD-near", "mw4-wingD-far",
    )
    door_ids = (
        "mw4-door-core-a", "mw4-door-a-far", "mw4-door-core-b", "mw4-door-b-far",
        "mw4-door-core-c", "mw4-door-c-far", "mw4-door-core-d", "mw4-door-d-far",
    )

    definition = ScenarioDefinition(
        fire=FireDefinition(
            growth_parameter_distribution=UniformRange(60.0, 380.0),
            ignition_zone_preference=WeightedOptions({zid: 1.0 / len(zone_ids) for zid in zone_ids}),
            allowed_fire_profiles=frozenset({"Flaming", "Smoldering"}),
        ),
        occupant=OccupantDefinition(
            occupancy_distribution={zid: UniformRange(2, 12, discrete=True) for zid in zone_ids},
            behaviour_profile_distribution={
                zid: WeightedOptions({"Adult_Default": 0.65, "Staff_Default": 0.2, "Elderly_Default": 0.15})
                for zid in zone_ids
            },
            assistance_pairing_probability=0.2,
        ),
        engineering=EngineeringConstraints(
            door_state_distribution={did: WeightedOptions({"OPEN": 0.75, "CLOSED": 0.2, "LOCKED": 0.05}) for did in door_ids},
            exit_state_distribution={"mw4-exit-1": WeightedOptions({True: 0.9, False: 0.1})},
            min_open_exits=0,
        ),
        firefighter=FirefighterDeploymentDefinition(
            team_count_distribution=WeightedOptions({0: 0.45, 1: 0.35, 2: 0.2}),
            team_size_distribution=UniformRange(2, 4, discrete=True),
            arrival_time_distribution=UniformRange(45.0, 180.0),
            entry_zone_ids=("mw4-core",),
            rescue_assignment_probability=0.6,
        ),
    )

    return TopologySpec(
        name="multi_wing_four_wings",
        description="Multi-wing variant: 4 wings (2 zones deep each) instead of 3, ONE shared core exit -- "
                    "higher core branching factor than the base variant.",
        building=building, definition=definition, scenario_count=150,
    )


def build_multi_wing_vertical() -> TopologySpec:
    """Core + 2 ground-floor wings (2 zones deep each) + ONE wing that
    goes UP a Stair to a 2-zone sub-wing on floor 2 -- combines
    multi-wing branching with vertical connectivity, a combination no
    V3 variant has (multi_exit_wide is always single-floor;
    twin_stair_highrise has no wing-depth/funneling structure)."""

    ground = Floor(
        id="mwv-floor-ground", name="Ground", display_order=0,
        zones=[
            Zone(id="mwv-core", name="Core", floor_id="mwv-floor-ground", x=0, y=20, width=14, height=12),
            Zone(id="mwv-wingA-near", name="Wing A Near", floor_id="mwv-floor-ground", x=-30, y=0, width=12, height=10),
            Zone(id="mwv-wingA-far", name="Wing A Far", floor_id="mwv-floor-ground", x=-50, y=0, width=12, height=10),
            Zone(id="mwv-wingB-near", name="Wing B Near", floor_id="mwv-floor-ground", x=30, y=0, width=12, height=10),
            Zone(id="mwv-wingB-far", name="Wing B Far", floor_id="mwv-floor-ground", x=50, y=0, width=12, height=10),
        ],
        doors=[
            Door(id="mwv-door-core-a", normally_open=True, zone_a_id="mwv-core", zone_b_id="mwv-wingA-near"),
            Door(id="mwv-door-a-far", normally_open=True, zone_a_id="mwv-wingA-near", zone_b_id="mwv-wingA-far"),
            Door(id="mwv-door-core-b", normally_open=True, zone_a_id="mwv-core", zone_b_id="mwv-wingB-near"),
            Door(id="mwv-door-b-far", normally_open=True, zone_a_id="mwv-wingB-near", zone_b_id="mwv-wingB-far"),
        ],
        exits=[Exit(id="mwv-exit-1", zone_id="mwv-core")],
    )

    upper = Floor(
        id="mwv-floor-upper", name="Upper Wing", display_order=1,
        zones=[
            Zone(id="mwv-wingC-near", name="Wing C Near (Upper)", floor_id="mwv-floor-upper", x=0, y=45, width=12, height=10),
            Zone(id="mwv-wingC-far", name="Wing C Far (Upper)", floor_id="mwv-floor-upper", x=0, y=65, width=12, height=10),
        ],
        doors=[
            Door(id="mwv-door-c-far", normally_open=True, zone_a_id="mwv-wingC-near", zone_b_id="mwv-wingC-far"),
        ],
        stairs=[
            Staircase(
                id="mwv-stair-1", from_zone_id="mwv-wingC-near", to_zone_id="mwv-core",
                from_floor_id="mwv-floor-upper", to_floor_id="mwv-floor-ground",
            ),
        ],
    )

    building = Building(id="topology-multi-wing-vertical", name="Multi-Wing Vertical", floors=[ground, upper])

    zone_ids = (
        "mwv-core", "mwv-wingA-near", "mwv-wingA-far", "mwv-wingB-near", "mwv-wingB-far",
        "mwv-wingC-near", "mwv-wingC-far",
    )
    door_ids = ("mwv-door-core-a", "mwv-door-a-far", "mwv-door-core-b", "mwv-door-b-far", "mwv-door-c-far")

    definition = ScenarioDefinition(
        fire=FireDefinition(
            growth_parameter_distribution=UniformRange(60.0, 380.0),
            ignition_zone_preference=WeightedOptions({zid: 1.0 / len(zone_ids) for zid in zone_ids}),
            allowed_fire_profiles=frozenset({"Flaming", "Smoldering"}),
        ),
        occupant=OccupantDefinition(
            occupancy_distribution={zid: UniformRange(2, 12, discrete=True) for zid in zone_ids},
            behaviour_profile_distribution={
                zid: WeightedOptions({"Adult_Default": 0.65, "Staff_Default": 0.2, "Elderly_Default": 0.15})
                for zid in zone_ids
            },
            assistance_pairing_probability=0.2,
        ),
        engineering=EngineeringConstraints(
            door_state_distribution={did: WeightedOptions({"OPEN": 0.75, "CLOSED": 0.2, "LOCKED": 0.05}) for did in door_ids},
            exit_state_distribution={"mwv-exit-1": WeightedOptions({True: 0.9, False: 0.1})},
            min_open_exits=0,
        ),
        firefighter=FirefighterDeploymentDefinition(
            team_count_distribution=WeightedOptions({0: 0.45, 1: 0.35, 2: 0.2}),
            team_size_distribution=UniformRange(2, 4, discrete=True),
            arrival_time_distribution=UniformRange(45.0, 180.0),
            entry_zone_ids=("mwv-core",),
            rescue_assignment_probability=0.6,
        ),
    )

    return TopologySpec(
        name="multi_wing_vertical",
        description="Multi-wing variant: core + 2 ground-floor wings + 1 wing reached via a Stair to floor 2 -- "
                    "combines wing-depth/funneling with vertical connectivity.",
        building=building, definition=definition, scenario_count=150,
    )


# =====================================================
# Family 6 -- RING_CORRIDOR. Base: a genuine 4-zone RING (zone-to-zone
# cycle, never mediated by OUTSIDE) with one exit -- the structural
# region 15 of 16 Dataset V3 variants have zero of.
# =====================================================


def build_ring_corridor() -> TopologySpec:
    """4 zones connected in a closed RING (A-B-C-D-A, 4 doors), one exit
    on zone A -- a genuine zone-only cycle, unlike anything in V3 except
    v1_fixed_dual_stair's incidental one."""

    floor1 = Floor(
        id="rc-floor-1", name="Ground", display_order=0,
        zones=[
            Zone(id="rc-zone-a", name="Zone A", floor_id="rc-floor-1", x=0, y=0, width=14, height=10),
            Zone(id="rc-zone-b", name="Zone B", floor_id="rc-floor-1", x=20, y=0, width=14, height=10),
            Zone(id="rc-zone-c", name="Zone C", floor_id="rc-floor-1", x=20, y=16, width=14, height=10),
            Zone(id="rc-zone-d", name="Zone D", floor_id="rc-floor-1", x=0, y=16, width=14, height=10),
        ],
        doors=[
            Door(id="rc-door-ab", normally_open=True, zone_a_id="rc-zone-a", zone_b_id="rc-zone-b"),
            Door(id="rc-door-bc", normally_open=True, zone_a_id="rc-zone-b", zone_b_id="rc-zone-c"),
            Door(id="rc-door-cd", normally_open=True, zone_a_id="rc-zone-c", zone_b_id="rc-zone-d"),
            Door(id="rc-door-da", normally_open=True, zone_a_id="rc-zone-d", zone_b_id="rc-zone-a"),
        ],
        exits=[Exit(id="rc-exit-1", zone_id="rc-zone-a")],
    )

    building = Building(id="topology-ring-corridor", name="Ring Corridor", floors=[floor1])

    zone_ids = ("rc-zone-a", "rc-zone-b", "rc-zone-c", "rc-zone-d")
    door_ids = ("rc-door-ab", "rc-door-bc", "rc-door-cd", "rc-door-da")

    definition = ScenarioDefinition(
        fire=FireDefinition(
            growth_parameter_distribution=UniformRange(60.0, 380.0),
            ignition_zone_preference=WeightedOptions({zid: 1.0 / len(zone_ids) for zid in zone_ids}),
            allowed_fire_profiles=frozenset({"Flaming", "Smoldering"}),
        ),
        occupant=OccupantDefinition(
            occupancy_distribution={zid: UniformRange(3, 16, discrete=True) for zid in zone_ids},
            behaviour_profile_distribution={
                zid: WeightedOptions({"Adult_Default": 0.65, "Staff_Default": 0.2, "Elderly_Default": 0.15})
                for zid in zone_ids
            },
            assistance_pairing_probability=0.2,
        ),
        engineering=EngineeringConstraints(
            door_state_distribution={did: WeightedOptions({"OPEN": 0.7, "CLOSED": 0.25, "LOCKED": 0.05}) for did in door_ids},
            exit_state_distribution={"rc-exit-1": WeightedOptions({True: 0.9, False: 0.1})},
            min_open_exits=0,
        ),
        firefighter=FirefighterDeploymentDefinition(
            team_count_distribution=WeightedOptions({0: 0.45, 1: 0.35, 2: 0.2}),
            team_size_distribution=UniformRange(2, 4, discrete=True),
            arrival_time_distribution=UniformRange(45.0, 180.0),
            entry_zone_ids=("rc-zone-a",),
            rescue_assignment_probability=0.6,
        ),
    )

    return TopologySpec(
        name="ring_corridor",
        description="Ring-corridor family base: 4 zones in a closed ring (4 doors), one exit -- "
                    "genuine zone-to-zone route redundancy, near-absent from Dataset V3.",
        building=building, definition=definition, scenario_count=150,
    )


def build_ring_corridor_dual_exit() -> TopologySpec:
    """Same 4-zone ring, but TWO exits at OPPOSITE ring positions (zone A
    and zone C) -- clockwise vs. counterclockwise travel now genuinely
    favors different exits, a route-CHOICE structure no V3 variant has."""

    floor1 = Floor(
        id="rcde-floor-1", name="Ground", display_order=0,
        zones=[
            Zone(id="rcde-zone-a", name="Zone A", floor_id="rcde-floor-1", x=0, y=0, width=14, height=10),
            Zone(id="rcde-zone-b", name="Zone B", floor_id="rcde-floor-1", x=20, y=0, width=14, height=10),
            Zone(id="rcde-zone-c", name="Zone C", floor_id="rcde-floor-1", x=20, y=16, width=14, height=10),
            Zone(id="rcde-zone-d", name="Zone D", floor_id="rcde-floor-1", x=0, y=16, width=14, height=10),
        ],
        doors=[
            Door(id="rcde-door-ab", normally_open=True, zone_a_id="rcde-zone-a", zone_b_id="rcde-zone-b"),
            Door(id="rcde-door-bc", normally_open=True, zone_a_id="rcde-zone-b", zone_b_id="rcde-zone-c"),
            Door(id="rcde-door-cd", normally_open=True, zone_a_id="rcde-zone-c", zone_b_id="rcde-zone-d"),
            Door(id="rcde-door-da", normally_open=True, zone_a_id="rcde-zone-d", zone_b_id="rcde-zone-a"),
        ],
        exits=[
            Exit(id="rcde-exit-a", zone_id="rcde-zone-a"),
            Exit(id="rcde-exit-c", zone_id="rcde-zone-c"),
        ],
    )

    building = Building(id="topology-ring-corridor-dual-exit", name="Ring Corridor Dual Exit", floors=[floor1])

    zone_ids = ("rcde-zone-a", "rcde-zone-b", "rcde-zone-c", "rcde-zone-d")
    door_ids = ("rcde-door-ab", "rcde-door-bc", "rcde-door-cd", "rcde-door-da")

    definition = ScenarioDefinition(
        fire=FireDefinition(
            growth_parameter_distribution=UniformRange(60.0, 380.0),
            ignition_zone_preference=WeightedOptions({zid: 1.0 / len(zone_ids) for zid in zone_ids}),
            allowed_fire_profiles=frozenset({"Flaming", "Smoldering"}),
        ),
        occupant=OccupantDefinition(
            occupancy_distribution={zid: UniformRange(3, 16, discrete=True) for zid in zone_ids},
            behaviour_profile_distribution={
                zid: WeightedOptions({"Adult_Default": 0.65, "Staff_Default": 0.2, "Elderly_Default": 0.15})
                for zid in zone_ids
            },
            assistance_pairing_probability=0.2,
        ),
        engineering=EngineeringConstraints(
            door_state_distribution={did: WeightedOptions({"OPEN": 0.7, "CLOSED": 0.25, "LOCKED": 0.05}) for did in door_ids},
            exit_state_distribution={
                "rcde-exit-a": WeightedOptions({True: 0.85, False: 0.15}),
                "rcde-exit-c": WeightedOptions({True: 0.85, False: 0.15}),
            },
            min_open_exits=1,
        ),
        firefighter=FirefighterDeploymentDefinition(
            team_count_distribution=WeightedOptions({0: 0.45, 1: 0.35, 2: 0.2}),
            team_size_distribution=UniformRange(2, 4, discrete=True),
            arrival_time_distribution=UniformRange(45.0, 180.0),
            entry_zone_ids=("rcde-zone-a",),
            rescue_assignment_probability=0.6,
        ),
    )

    return TopologySpec(
        name="ring_corridor_dual_exit",
        description="Ring-corridor variant: same 4-zone ring, TWO exits at opposite ring positions -- "
                    "clockwise/counterclockwise travel favors different exits.",
        building=building, definition=definition, scenario_count=150,
    )


def build_ring_corridor_large() -> TopologySpec:
    """A LARGER 6-zone ring (A-B-C-D-E-F-A, 6 doors), one exit -- deeper
    corridor depth around the ring than the base 4-zone variant."""

    floor1 = Floor(
        id="rcl-floor-1", name="Ground", display_order=0,
        zones=[
            Zone(id="rcl-zone-a", name="Zone A", floor_id="rcl-floor-1", x=0, y=0, width=12, height=10),
            Zone(id="rcl-zone-b", name="Zone B", floor_id="rcl-floor-1", x=16, y=0, width=12, height=10),
            Zone(id="rcl-zone-c", name="Zone C", floor_id="rcl-floor-1", x=32, y=0, width=12, height=10),
            Zone(id="rcl-zone-d", name="Zone D", floor_id="rcl-floor-1", x=32, y=16, width=12, height=10),
            Zone(id="rcl-zone-e", name="Zone E", floor_id="rcl-floor-1", x=16, y=16, width=12, height=10),
            Zone(id="rcl-zone-f", name="Zone F", floor_id="rcl-floor-1", x=0, y=16, width=12, height=10),
        ],
        doors=[
            Door(id="rcl-door-ab", normally_open=True, zone_a_id="rcl-zone-a", zone_b_id="rcl-zone-b"),
            Door(id="rcl-door-bc", normally_open=True, zone_a_id="rcl-zone-b", zone_b_id="rcl-zone-c"),
            Door(id="rcl-door-cd", normally_open=True, zone_a_id="rcl-zone-c", zone_b_id="rcl-zone-d"),
            Door(id="rcl-door-de", normally_open=True, zone_a_id="rcl-zone-d", zone_b_id="rcl-zone-e"),
            Door(id="rcl-door-ef", normally_open=True, zone_a_id="rcl-zone-e", zone_b_id="rcl-zone-f"),
            Door(id="rcl-door-fa", normally_open=True, zone_a_id="rcl-zone-f", zone_b_id="rcl-zone-a"),
        ],
        exits=[Exit(id="rcl-exit-1", zone_id="rcl-zone-a")],
    )

    building = Building(id="topology-ring-corridor-large", name="Ring Corridor Large", floors=[floor1])

    zone_ids = ("rcl-zone-a", "rcl-zone-b", "rcl-zone-c", "rcl-zone-d", "rcl-zone-e", "rcl-zone-f")
    door_ids = ("rcl-door-ab", "rcl-door-bc", "rcl-door-cd", "rcl-door-de", "rcl-door-ef", "rcl-door-fa")

    definition = ScenarioDefinition(
        fire=FireDefinition(
            growth_parameter_distribution=UniformRange(60.0, 380.0),
            ignition_zone_preference=WeightedOptions({zid: 1.0 / len(zone_ids) for zid in zone_ids}),
            allowed_fire_profiles=frozenset({"Flaming", "Smoldering"}),
        ),
        occupant=OccupantDefinition(
            occupancy_distribution={zid: UniformRange(3, 14, discrete=True) for zid in zone_ids},
            behaviour_profile_distribution={
                zid: WeightedOptions({"Adult_Default": 0.65, "Staff_Default": 0.2, "Elderly_Default": 0.15})
                for zid in zone_ids
            },
            assistance_pairing_probability=0.2,
        ),
        engineering=EngineeringConstraints(
            door_state_distribution={did: WeightedOptions({"OPEN": 0.7, "CLOSED": 0.25, "LOCKED": 0.05}) for did in door_ids},
            exit_state_distribution={"rcl-exit-1": WeightedOptions({True: 0.9, False: 0.1})},
            min_open_exits=0,
        ),
        firefighter=FirefighterDeploymentDefinition(
            team_count_distribution=WeightedOptions({0: 0.45, 1: 0.35, 2: 0.2}),
            team_size_distribution=UniformRange(2, 4, discrete=True),
            arrival_time_distribution=UniformRange(45.0, 180.0),
            entry_zone_ids=("rcl-zone-a",),
            rescue_assignment_probability=0.6,
        ),
    )

    return TopologySpec(
        name="ring_corridor_large",
        description="Ring-corridor variant: a larger 6-zone ring (6 doors), one exit -- deeper "
                    "corridor depth around the ring than the base 4-zone variant.",
        building=building, definition=definition, scenario_count=150,
    )


def build_ring_corridor_partial_chord() -> TopologySpec:
    """The base 4-zone ring PLUS one CHORD door cutting directly across
    (zone A <-> zone C) -- 2 independent cycles (higher cyclomatic
    number), a genuinely more complex redundancy structure than a
    simple ring."""

    floor1 = Floor(
        id="rcpc-floor-1", name="Ground", display_order=0,
        zones=[
            Zone(id="rcpc-zone-a", name="Zone A", floor_id="rcpc-floor-1", x=0, y=0, width=14, height=10),
            Zone(id="rcpc-zone-b", name="Zone B", floor_id="rcpc-floor-1", x=20, y=0, width=14, height=10),
            Zone(id="rcpc-zone-c", name="Zone C", floor_id="rcpc-floor-1", x=20, y=16, width=14, height=10),
            Zone(id="rcpc-zone-d", name="Zone D", floor_id="rcpc-floor-1", x=0, y=16, width=14, height=10),
        ],
        doors=[
            Door(id="rcpc-door-ab", normally_open=True, zone_a_id="rcpc-zone-a", zone_b_id="rcpc-zone-b"),
            Door(id="rcpc-door-bc", normally_open=True, zone_a_id="rcpc-zone-b", zone_b_id="rcpc-zone-c"),
            Door(id="rcpc-door-cd", normally_open=True, zone_a_id="rcpc-zone-c", zone_b_id="rcpc-zone-d"),
            Door(id="rcpc-door-da", normally_open=True, zone_a_id="rcpc-zone-d", zone_b_id="rcpc-zone-a"),
            Door(id="rcpc-door-chord-ac", normally_open=True, zone_a_id="rcpc-zone-a", zone_b_id="rcpc-zone-c"),
        ],
        exits=[Exit(id="rcpc-exit-1", zone_id="rcpc-zone-a")],
    )

    building = Building(id="topology-ring-corridor-partial-chord", name="Ring Corridor Partial Chord", floors=[floor1])

    zone_ids = ("rcpc-zone-a", "rcpc-zone-b", "rcpc-zone-c", "rcpc-zone-d")
    door_ids = ("rcpc-door-ab", "rcpc-door-bc", "rcpc-door-cd", "rcpc-door-da", "rcpc-door-chord-ac")

    definition = ScenarioDefinition(
        fire=FireDefinition(
            growth_parameter_distribution=UniformRange(60.0, 380.0),
            ignition_zone_preference=WeightedOptions({zid: 1.0 / len(zone_ids) for zid in zone_ids}),
            allowed_fire_profiles=frozenset({"Flaming", "Smoldering"}),
        ),
        occupant=OccupantDefinition(
            occupancy_distribution={zid: UniformRange(3, 16, discrete=True) for zid in zone_ids},
            behaviour_profile_distribution={
                zid: WeightedOptions({"Adult_Default": 0.65, "Staff_Default": 0.2, "Elderly_Default": 0.15})
                for zid in zone_ids
            },
            assistance_pairing_probability=0.2,
        ),
        engineering=EngineeringConstraints(
            door_state_distribution={did: WeightedOptions({"OPEN": 0.7, "CLOSED": 0.25, "LOCKED": 0.05}) for did in door_ids},
            exit_state_distribution={"rcpc-exit-1": WeightedOptions({True: 0.9, False: 0.1})},
            min_open_exits=0,
        ),
        firefighter=FirefighterDeploymentDefinition(
            team_count_distribution=WeightedOptions({0: 0.45, 1: 0.35, 2: 0.2}),
            team_size_distribution=UniformRange(2, 4, discrete=True),
            arrival_time_distribution=UniformRange(45.0, 180.0),
            entry_zone_ids=("rcpc-zone-a",),
            rescue_assignment_probability=0.6,
        ),
    )

    return TopologySpec(
        name="ring_corridor_partial_chord",
        description="Ring-corridor variant: the base 4-zone ring PLUS one chord door (A<->C) -- "
                    "2 independent cycles, higher cyclomatic number than a simple ring.",
        building=building, definition=definition, scenario_count=150,
    )


# =====================================================
# Registry -- V3's 16 (REUSED, unmodified) + 8 new (2 families x 4
# variants) = 24 structural templates.
# =====================================================


def all_structural_variants_v4() -> Tuple[StructuralVariant, ...]:

    return all_structural_variants_v3() + (
        StructuralVariant("multi_wing", "multi_wing", "base", build_multi_wing()),
        StructuralVariant("multi_wing", "multi_wing_asymmetric_exits", "asymmetric_exits", build_multi_wing_asymmetric_exits()),
        StructuralVariant("multi_wing", "multi_wing_four_wings", "four_wings", build_multi_wing_four_wings()),
        StructuralVariant("multi_wing", "multi_wing_vertical", "vertical", build_multi_wing_vertical()),

        StructuralVariant("ring_corridor", "ring_corridor", "base", build_ring_corridor()),
        StructuralVariant("ring_corridor", "ring_corridor_dual_exit", "dual_exit", build_ring_corridor_dual_exit()),
        StructuralVariant("ring_corridor", "ring_corridor_large", "large", build_ring_corridor_large()),
        StructuralVariant("ring_corridor", "ring_corridor_partial_chord", "partial_chord", build_ring_corridor_partial_chord()),
    )


ALL_FAMILIES_V4: Tuple[str, ...] = tuple(sorted(set(v.family for v in all_structural_variants_v4())))
