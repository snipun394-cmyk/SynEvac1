from models.building import Building
from models.door import Door
from models.exit import Exit
from models.floor import Floor
from models.zone import Zone

from scenario_definition import FireDefinition, FixedValue, OccupantDefinition, ScenarioDefinition


# =====================================================
# Mirrors tests/test_simulation_interactive.py's own make_building()
# shape (two-path building: zone-start's shortest route is via
# door-a/exit-a; door-b/exit-b is deliberately far away), so RL tests
# get the same easy-to-reason-about, deterministic rerouting behavior
# without depending on that other test module directly (per the
# repo's own "factor shared fixtures into their own *_fixtures.py
# module" convention -- tests/training_dataset_fixtures.py,
# tests/ai_training_fixtures.py).
# =====================================================


def make_building() -> Building:

    floor1 = Floor(
        name="Ground", id="floor-1",
        zones=[
            Zone(id="zone-start", name="Start", x=0.0, y=0.0, width=2.0, height=2.0),
            Zone(id="zone-a", name="A", x=10.0, y=0.0, width=2.0, height=2.0),
            Zone(id="zone-b", name="B", x=200.0, y=0.0, width=2.0, height=2.0),
        ],
        doors=[
            Door(id="door-a", normally_open=True, zone_a_id="zone-start", zone_b_id="zone-a"),
            Door(id="door-b", normally_open=True, zone_a_id="zone-start", zone_b_id="zone-b"),
        ],
        exits=[
            Exit(id="exit-a", zone_id="zone-a"),
            Exit(id="exit-b", zone_id="zone-b"),
        ],
    )

    return Building(name="RL Training Test Building", id="rl-building-1", floors=[floor1])


def make_definition() -> ScenarioDefinition:

    return ScenarioDefinition(
        fire=FireDefinition(growth_parameter_distribution=FixedValue(200.0)),
        occupant=OccupantDefinition(
            occupancy_distribution={"zone-start": FixedValue(1)},
            behaviour_profile_distribution={"zone-start": FixedValue("Staff_Default")},
        ),
    )


DEFINITION_ID = "rl-def-1"
MASTER_SEED = 4242
