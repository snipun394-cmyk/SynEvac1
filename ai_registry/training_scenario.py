from models.building import Building
from models.camera import Camera
from models.detector import Detector
from models.door import Door
from models.exit import Exit
from models.floor import Floor
from models.heat_detector import HeatDetector
from models.obstacle import Obstacle
from models.smoke_detector import SmokeDetector
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
# Live-Compatible AI Model Training & Model Registry milestone -- Phase
# 4's own required "substantially larger and more diverse" campaign,
# reusing the existing Scenario Generator/Building/ScenarioDefinition
# infrastructure exactly as-is (no new generation machinery). Deliberately
# NOT the tiny 2-zone fixture tests/training_dataset_fixtures.py uses for
# fast unit tests -- that fixture stays untouched and is still used by
# every existing ai_training/training_dataset test.
#
# Every varied factor below affects the SIMULATION and its LABELS
# (total_evacuation_time, bottleneck_occurrence) -- none of it becomes a
# model INPUT feature unless it is also part of ai_features.
# CANONICAL_LIVE_SCHEMA. Occupant counts/behaviour-profile mix/fire
# ignition/growth are exactly the kind of variation Phase 4 asks for
# precisely because they move the LABELS while remaining properly
# excluded from canonical INPUT features (see ai_features/feature_schema.py's
# own "deliberately excluded" list).
# =====================================================


def make_training_building() -> Building:

    floor1 = Floor(
        id="floor-1", name="Ground", display_order=0,
        zones=[
            Zone(id="zone-lobby", name="Lobby", floor_id="floor-1", x=0, y=0, width=12, height=8),
            Zone(id="zone-office-a", name="Office A", floor_id="floor-1", x=20, y=0, width=8, height=8),
            Zone(id="zone-office-b", name="Office B", floor_id="floor-1", x=32, y=0, width=8, height=8),
        ],
        doors=[
            Door(id="door-1", normally_open=True, zone_a_id="zone-lobby", zone_b_id="zone-office-a"),
            Door(id="door-2", normally_open=True, zone_a_id="zone-office-a", zone_b_id="zone-office-b"),
        ],
        exits=[
            Exit(id="exit-1", zone_id="zone-lobby"),
            Exit(id="exit-2", zone_id="zone-office-b"),
        ],
        stairs=[
            Staircase(id="stair-1", from_zone_id="zone-office-a", to_zone_id="zone-upper", to_floor_id="floor-2"),
        ],
        obstacles=[Obstacle(id="obs-1", active=False)],
        cameras=[
            Camera(id="cam-lobby", active=True, floor_id="floor-1", zone_ids=("zone-lobby",)),
            Camera(id="cam-office-a", active=True, floor_id="floor-1", zone_ids=("zone-office-a",)),
        ],
        smoke_detectors=[
            SmokeDetector(id="smoke-lobby", name="Smoke Lobby", floor_id="floor-1", zone_ids=("zone-lobby",)),
            SmokeDetector(id="smoke-office-a", name="Smoke Office A", floor_id="floor-1", zone_ids=("zone-office-a",)),
        ],
        heat_detectors=[
            HeatDetector(id="heat-office-b", name="Heat Office B", floor_id="floor-1", zone_ids=("zone-office-b",)),
        ],
        detectors=[
            Detector(id="legacy-det-1", detector_type="Smoke", active=True, floor_id="floor-1"),
        ],
    )

    floor2 = Floor(
        id="floor-2", name="Upper", display_order=1,
        zones=[
            Zone(id="zone-upper", name="Upper Hall", floor_id="floor-2", x=0, y=0, width=16, height=10),
        ],
        cameras=[
            Camera(id="cam-upper", active=True, floor_id="floor-2", zone_ids=("zone-upper",)),
        ],
        smoke_detectors=[
            SmokeDetector(id="smoke-upper", name="Smoke Upper", floor_id="floor-2", zone_ids=("zone-upper",)),
        ],
    )

    return Building(id="training-building-1", name="Live AI Training Building", floors=[floor1, floor2])


def make_training_definition() -> ScenarioDefinition:

    return ScenarioDefinition(
        fire=FireDefinition(
            growth_parameter_distribution=UniformRange(80.0, 420.0),
            ignition_zone_preference=WeightedOptions({
                "zone-lobby": 0.3, "zone-office-a": 0.3, "zone-office-b": 0.25, "zone-upper": 0.15,
            }),
            allowed_fire_profiles=frozenset({"Flaming", "Smoldering"}),
        ),
        occupant=OccupantDefinition(
            occupancy_distribution={
                "zone-lobby": UniformRange(2, 10, discrete=True),
                "zone-office-a": UniformRange(1, 8, discrete=True),
                "zone-office-b": UniformRange(1, 8, discrete=True),
                "zone-upper": UniformRange(0, 6, discrete=True),
            },
            behaviour_profile_distribution={
                "zone-lobby": WeightedOptions({
                    "Adult_Default": 0.55, "Elderly_Default": 0.15, "Wheelchair_Default": 0.1,
                    "Child_Default": 0.1, "Visitor_Default": 0.1,
                }),
                "zone-office-a": WeightedOptions({"Adult_Default": 0.7, "Staff_Default": 0.3}),
                "zone-office-b": WeightedOptions({"Adult_Default": 0.7, "Staff_Default": 0.3}),
                "zone-upper": WeightedOptions({"Adult_Default": 0.6, "Visitor_Default": 0.4}),
            },
            assistance_pairing_probability=0.2,
        ),
        engineering=EngineeringConstraints(
            door_state_distribution={
                "door-1": WeightedOptions({"OPEN": 0.7, "CLOSED": 0.25, "LOCKED": 0.05}),
                "door-2": WeightedOptions({"OPEN": 0.7, "CLOSED": 0.25, "LOCKED": 0.05}),
            },
            exit_state_distribution={
                "exit-1": WeightedOptions({True: 0.9, False: 0.1}),
                "exit-2": WeightedOptions({True: 0.85, False: 0.15}),
            },
            min_open_exits=1,
            stair_state_distribution={
                "stair-1": WeightedOptions({"AVAILABLE": 0.85, "CLOSED": 0.15}),
            },
            camera_state_distribution={
                "cam-lobby": WeightedOptions({"AVAILABLE": 0.9, "FAILED": 0.1}),
                "cam-office-a": WeightedOptions({"AVAILABLE": 0.9, "FAILED": 0.1}),
                "cam-upper": WeightedOptions({"AVAILABLE": 0.85, "FAILED": 0.15}),
            },
            # detector_state_distribution is deliberately omitted --
            # engineering_constraints validation only resolves it against
            # legacy Floor.detectors (models/detector.py), not the
            # canonical SmokeDetector/HeatDetector assets this building
            # uses (confirmed by DefinitionValidationReport rejecting
            # canonical detector ids as "unknown_id"). Camera/door/exit/
            # stair/occupancy/fire/firefighter variation already give
            # this campaign substantial diversity without it.
        ),
        firefighter=FirefighterDeploymentDefinition(
            team_count_distribution=WeightedOptions({0: 0.5, 1: 0.35, 2: 0.15}),
            team_size_distribution=UniformRange(2, 4, discrete=True),
            arrival_time_distribution=UniformRange(45.0, 180.0),
            entry_zone_ids=("zone-lobby",),
            rescue_assignment_probability=0.6,
        ),
    )
