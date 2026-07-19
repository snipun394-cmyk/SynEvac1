import sys

from PyQt6.QtWidgets import QApplication

# Module-level QApplication singleton -- CampaignWorker/CampaignConfig
# import PyQt6 machinery transitively (QThread/pyqtSignal), same
# convention tests/test_campaign_pipeline_integration.py already
# establishes. Shared here so every training_dataset test module can
# import make_campaign() without repeating this bootstrap.
_app = QApplication.instance() or QApplication(sys.argv)

from models.building import Building
from models.camera import Camera
from models.detector import Detector
from models.door import Door
from models.exit import Exit
from models.floor import Floor
from models.obstacle import Obstacle
from models.staircase import Staircase
from models.zone import Zone

from scenario_definition import (
    FireDefinition,
    FixedValue,
    OccupantDefinition,
    ScenarioDefinition,
    UniformRange,
)

from designer.campaign.campaign_worker import CampaignConfig, CampaignWorker


# =====================================================
# Real-pipeline fixtures for training_dataset's own tests -- this
# package only ever consumes campaign output, so its tests must run
# against a *real* Campaign Studio export (scenario_runner ->
# behaviour_profile_resolver -> ai_decision -> simulation_runtime ->
# dataset_builder/ground_truth/decision_policy), never a hand-authored
# CSV/JSON fixture that could silently drift from the real artifact
# shape. Same tiny Building/Definition shapes tests/test_campaign_
# pipeline_integration.py already uses, so a full campaign runs in
# well under a second.
# =====================================================


def make_building() -> Building:

    floor1 = Floor(
        name="Ground", id="floor-1",
        zones=[
            Zone(id="zone-1", name="Lobby", x=0.0, y=0.0, width=10.0, height=8.0),
            Zone(id="zone-2", name="Office", x=20.0, y=0.0, width=6.0, height=6.0),
        ],
        doors=[Door(id="door-1", normally_open=True, zone_a_id="zone-1", zone_b_id="zone-2")],
        exits=[Exit(id="exit-1", zone_id="zone-1")],
        obstacles=[Obstacle(id="obs-1", active=False)],
        cameras=[Camera(id="cam-1", active=True)],
        detectors=[Detector(id="det-1", active=True)],
        stairs=[Staircase(id="stair-1", from_zone_id="zone-1", to_zone_id="zone-3", to_floor_id="floor-2")],
    )
    floor2 = Floor(name="Upper", id="floor-2", zones=[Zone(id="zone-3", name="Attic")])

    return Building(name="Test Building", id="building-1", floors=[floor1, floor2])


def make_definition() -> ScenarioDefinition:

    # Occupants spread across both zones (rather than only one), so
    # zone_results.csv always has more than one non-empty origin zone.

    return ScenarioDefinition(
        fire=FireDefinition(growth_parameter_distribution=FixedValue(200.0)),
        occupant=OccupantDefinition(
            occupancy_distribution={
                "zone-1": UniformRange(1, 2, discrete=True),
                "zone-2": UniformRange(1, 2, discrete=True),
            },
            behaviour_profile_distribution={
                "zone-1": FixedValue("Staff_Default"),
                "zone-2": FixedValue("Staff_Default"),
            },
        ),
    )


def make_campaign(output_dir: str, *, count: int = 4, master_seed: int = 42, **overrides):

    defaults = dict(
        name="Training Dataset Test Campaign",
        building=make_building(),
        definition=make_definition(),
        definition_id="def-test",
        count=count,
        master_seed=master_seed,
        output_directory=output_dir,
        max_attempts=10,
        dt=10.0,
    )
    defaults.update(overrides)

    summary = CampaignWorker(CampaignConfig(**defaults)).execute()

    return summary
