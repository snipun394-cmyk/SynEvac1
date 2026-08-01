from models.building import Building
from models.door import Door
from models.exit import Exit
from models.floor import Floor
from models.zone import Zone

from scenario_definition import FireDefinition, FixedValue, OccupantDefinition, ScenarioDefinition


# =====================================================
# Calibration Benchmark Framework V1 -- a deliberately single-path,
# single-bottleneck building (one zone, one door, one exit). Door/Exit
# width is widened to 1.4 m from their production defaults (Door
# 0.90 m, Exit 1.20 m) specifically so DefaultCapacityModel's own
# PEOPLE_PER_METER_OF_WIDTH=1.5 floors each edge's capacity to 2, not 1
# -- a capacity-1 edge never lets the simulator's own admission control
# admit a second occupant onto it at all, which makes Congestion
# Duration (ground_truth.bottleneck's own "2+ occupants concurrently on
# the same edge" definition) permanently None regardless of any
# candidate under test. A deliberately larger occupant count than
# tests/rl_training_fixtures.py's own building (which only ever seats
# one occupant) is used here on purpose: this milestone's own metrics
# (Queue Length, Maximum Density, Congestion Duration, Exit Utilization)
# are only meaningful when a scenario actually produces real queueing,
# which a single occupant crossing an uncontested door never does.
# =====================================================


def make_building() -> Building:

    floor = Floor(
        name="Ground", id="floor-1",
        zones=[
            Zone(id="zone-start", name="Start", x=0.0, y=0.0, width=20.0, height=10.0),
            Zone(id="zone-exit", name="Exit Lobby", x=25.0, y=0.0, width=5.0, height=5.0),
        ],
        doors=[
            Door(
                id="door-a", normally_open=True, zone_a_id="zone-start", zone_b_id="zone-exit",
                width=1.4,
            ),
        ],
        exits=[
            Exit(id="exit-a", zone_id="zone-exit", width=1.4),
        ],
    )

    return Building(name="Calibration Benchmark Test Building", id="calibration-building-1", floors=[floor])


def make_definition(occupant_count: int = 25) -> ScenarioDefinition:

    return ScenarioDefinition(
        fire=FireDefinition(growth_parameter_distribution=FixedValue(300.0)),
        occupant=OccupantDefinition(
            occupancy_distribution={"zone-start": FixedValue(occupant_count)},
            behaviour_profile_distribution={"zone-start": FixedValue("Adult_Default")},
        ),
    )


DEFINITION_ID = "calibration-benchmark-def-1"
MASTER_SEED = 90210
