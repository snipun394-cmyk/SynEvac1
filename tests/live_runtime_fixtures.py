from models.building import Building
from models.door import Door
from models.exit import Exit
from models.floor import Floor
from models.heat_detector import HeatDetector
from models.smoke_detector import SmokeDetector
from models.speaker import Speaker
from models.staircase import Staircase
from models.zone import Zone
from models.camera import Camera

from live_camera_pipeline.identity_resolver import MappingIdentityResolver
from live_camera_pipeline.replay_frame_source import ReplayFrameSource

from tests.live_camera_pipeline_fixtures import MockHumanDetector


# =====================================================
# Production Live Runtime Composition Root milestone -- one shared,
# richly-populated Building fixture (Phase 7's own required shape: 2+
# cameras, smoke/heat detectors, speakers, doors/exits/stairs, multiple
# zones) plus the offline camera collaborators every live_runtime test
# needs, so no test file re-derives this construction by hand. Mirrors
# tests.live_camera_pipeline_fixtures/tests.test_advisory_system's own
# established "one shared, hand-built fixture module per concern"
# convention.
# =====================================================


def make_demo_building() -> Building:

    ground = Floor(
        id="floor-1", name="Ground Floor",
        zones=[
            Zone(id="zone-lobby", name="Lobby", x=0, y=0, width=10, height=8, floor_id="floor-1"),
            Zone(id="zone-cafeteria", name="Cafeteria", x=20, y=0, width=10, height=8, floor_id="floor-1"),
        ],
        cameras=[
            Camera(id="CAM-LOBBY", name="Lobby Camera", floor_id="floor-1", zone_ids=("zone-lobby",)),
            Camera(id="CAM-CAFETERIA", name="Cafeteria Camera", floor_id="floor-1", zone_ids=("zone-cafeteria",)),
        ],
        smoke_detectors=[
            SmokeDetector(id="SMOKE-LOBBY", name="Smoke Lobby", floor_id="floor-1", zone_ids=("zone-lobby",)),
        ],
        heat_detectors=[
            HeatDetector(id="HEAT-CAFETERIA", name="Heat Cafeteria", floor_id="floor-1", zone_ids=("zone-cafeteria",)),
        ],
        speakers=[
            Speaker(id="SPK-LOBBY", name="Speaker Lobby", floor_id="floor-1", zone_ids=("zone-lobby",)),
            Speaker(id="SPK-CAFETERIA", name="Speaker Cafeteria", floor_id="floor-1", zone_ids=("zone-cafeteria",)),
        ],
        doors=[
            Door(id="DOOR-1", name="Lobby-Cafeteria Door", floor_id="floor-1", zone_a_id="zone-lobby", zone_b_id="zone-cafeteria"),
        ],
        exits=[
            Exit(id="EXIT-1", name="Main Exit", floor_id="floor-1", zone_id="zone-lobby"),
        ],
        stairs=[
            Staircase(id="STAIR-1", name="Stair 1", from_zone_id="zone-cafeteria", to_zone_id="zone-lobby", to_floor_id="floor-1"),
        ],
    )

    return Building(id="demo-building", name="Demo Building", floors=[ground])


def make_offline_frame_sources(building: Building):

    frame_sources = {}

    for floor in building.ordered_floors():
        for camera in floor.cameras:
            source = ReplayFrameSource(
                camera_id=camera.id, frames=[(0.0, [{"local_track_id": "1"}])],
            )
            frame_sources[camera.id] = source

    return frame_sources


def make_offline_identity_resolver(building: Building, occupant_prefix="P"):

    mapping = {}

    for index, floor in enumerate(building.ordered_floors()):
        for camera in floor.cameras:
            mapping[(camera.id, "1")] = f"{occupant_prefix}-{camera.id}"

    return MappingIdentityResolver(mapping)


def make_offline_human_detector():

    return MockHumanDetector()
