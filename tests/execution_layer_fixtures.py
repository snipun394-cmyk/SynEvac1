from models.building import Building
from models.door import Door
from models.floor import Floor
from models.speaker import Speaker
from models.zone import Zone

from speaker_manager.manager import SpeakerManager

from voice_evacuation.controller import VoiceEvacuationController
from voice_evacuation.provider import SimulationVoiceOutputProvider

from building_control.controller import BuildingControlController
from building_control.providers import SimulationControlProvider

from dynamic_signage.controller import DynamicSignageController
from dynamic_signage.provider import SimulationDynamicSignageProvider

from warden_notification.controller import WardenNotificationController
from warden_notification.provider import SimulationWardenNotificationProvider


# =====================================================
# execution_layer/ -- shared fixture builders, no real live_runtime/
# factory.py dependency. Mirrors tests/recommendation_layer_fixtures.
# py's own "plain builder functions, sensible defaults" convention.
# =====================================================


def make_building_with_door_and_speaker():

    floor = Floor(id="f1", name="F1")
    zone1 = Zone(id="z1", name="Z1", x=0, y=0, width=5, height=5, floor_id="f1")
    zone2 = Zone(id="z2", name="Z2", x=10, y=0, width=5, height=5, floor_id="f1")
    floor.zones.extend([zone1, zone2])
    door = Door(id="door-1", name="D1", floor_id="f1", zone_a_id="z1", zone_b_id="z2")
    floor.doors.append(door)
    speaker = Speaker(id="sp-1", name="S1", floor_id="f1", zone_ids=("z1",))
    floor.speakers.append(speaker)

    return Building(id="b", name="B", floors=[floor])


def make_voice_controller(building=None):

    building = building if building is not None else make_building_with_door_and_speaker()

    speaker_manager = SpeakerManager()
    speaker_manager.discover_speakers(building)

    return VoiceEvacuationController(speaker_manager, SimulationVoiceOutputProvider())


def make_control_controller(building=None):

    building = building if building is not None else make_building_with_door_and_speaker()

    return BuildingControlController(building, SimulationControlProvider(building))


def make_signage_controller():

    return DynamicSignageController(SimulationDynamicSignageProvider())


def make_warden_controller():

    return WardenNotificationController(SimulationWardenNotificationProvider())
