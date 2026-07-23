from dataclasses import dataclass, field
from uuid import uuid4

from models.zone import Zone
from models.exit import Exit
from models.staircase import Staircase
from models.elevator import Elevator
from models.camera import Camera
from models.detector import Detector
from models.heat_detector import HeatDetector
from models.smoke_detector import SmokeDetector
from models.speaker import Speaker
from models.assembly_point import AssemblyPoint
from models.obstacle import Obstacle
from models.door import Door
from models.dynamic_sign import DynamicEvacuationSign
from models.manual_call_point import ManualCallPoint
from models.emergency_light import EmergencyLight
from models.sprinkler import Sprinkler
from models.fire_extinguisher import FireExtinguisher
from models.fire_hydrant import FireHydrant
from models.hose_reel import HoseReel
from models.fire_water_tank import FireWaterTank
from models.fire_pump import FirePump
from models.jockey_pump import JockeyPump
from models.fire_service_inlet import FireServiceInlet


@dataclass
class Floor:

    name: str

    id: str = field(default_factory=lambda: str(uuid4()))

    # =====================================================
    # Ordering vs. physical position
    #
    # display_order controls UI ordering / active-floor
    # navigation. height is the vertical extent of the floor,
    # freely editable. Elevation is NOT stored here -- it is
    # always derived from the cumulative height of every floor
    # below this one in display_order (see
    # Building.floor_elevation()), the same "never stored,
    # always resolved through building" pattern already used by
    # Staircase.vertical_height(). This guarantees the lowest
    # floor is always at elevation 0.0 and that elevation can
    # never drift out of sync with height/order.
    # =====================================================

    display_order: int = 0

    height: float = 3.0

    floor_plan: str = ""

    visible: bool = True
    locked: bool = False

    zones: list[Zone] = field(default_factory=list)
    exits: list[Exit] = field(default_factory=list)
    stairs: list[Staircase] = field(default_factory=list)
    elevators: list[Elevator] = field(default_factory=list)

    cameras: list[Camera] = field(default_factory=list)
    detectors: list[Detector] = field(default_factory=list)

    # Building Sensor Network Framework -- additive, alongside (never
    # replacing) the pre-existing generic `detectors` list above.
    # SmokeDetector/HeatDetector are the new, first-class per-type
    # assets (models/sensor_asset.py); the old generic Detector class
    # is untouched and remains fully supported for backward
    # compatibility and for detector types with no dedicated class yet
    # (Flame, Gas).
    smoke_detectors: list[SmokeDetector] = field(default_factory=list)
    heat_detectors: list[HeatDetector] = field(default_factory=list)

    # Zoned Voice Evacuation & Speaker Network Framework -- additive,
    # same SensorAsset-based convention as smoke_detectors/heat_detectors
    # above. A Speaker is an output device, not a sensor, but shares the
    # exact same asset-management shape (zone_ids/active/health_status),
    # so it is placed alongside them here rather than in a parallel
    # per-type list scheme.
    speakers: list[Speaker] = field(default_factory=list)

    # Live Dynamic Evacuation Signage milestone -- additive, same
    # EngineeringAsset-based convention as speakers/cameras/detectors
    # above. A new list on an existing dataclass: old .syn files simply
    # have no "signs" key at all, and from_dict() below defaults that to
    # an empty list, so every pre-existing project keeps loading
    # unchanged (Phase 2's own backward-compatibility requirement).
    signs: list[DynamicEvacuationSign] = field(default_factory=list)

    # Manual Call Points & Emergency Lighting milestone -- additive,
    # same SensorAsset/EngineeringAsset-based conventions as above.
    manual_call_points: list[ManualCallPoint] = field(default_factory=list)
    emergency_lights: list[EmergencyLight] = field(default_factory=list)

    # Fire Suppression & Water-Based Safety Asset Digital Twin
    # milestone -- additive, same conventions as above.
    sprinklers: list[Sprinkler] = field(default_factory=list)
    fire_extinguishers: list[FireExtinguisher] = field(default_factory=list)
    fire_hydrants: list[FireHydrant] = field(default_factory=list)
    hose_reels: list[HoseReel] = field(default_factory=list)

    # Fire Water Supply & Suppression Infrastructure milestone --
    # additive, same conventions as above.
    fire_water_tanks: list[FireWaterTank] = field(default_factory=list)
    fire_pumps: list[FirePump] = field(default_factory=list)
    jockey_pumps: list[JockeyPump] = field(default_factory=list)
    fire_service_inlets: list[FireServiceInlet] = field(default_factory=list)

    assembly_points: list[AssemblyPoint] = field(default_factory=list)
    obstacles: list[Obstacle] = field(default_factory=list)
    doors: list[Door] = field(default_factory=list)

    # =====================================================

    def rename(self, name):

        self.name = name

    # =====================================================
    # Zones
    # =====================================================

    def add_zone(self, zone):

        self.zones.append(zone)

    def remove_zone(self, zone):

        if zone in self.zones:
            self.zones.remove(zone)

    # =====================================================
    # Exits
    # =====================================================

    def add_exit(self, exit_obj):

        self.exits.append(exit_obj)

    def remove_exit(self, exit_obj):

        if exit_obj in self.exits:
            self.exits.remove(exit_obj)

    # =====================================================
    # Stairs
    # =====================================================

    def add_stair(self, stair):

        self.stairs.append(stair)

    def remove_stair(self, stair):

        if stair in self.stairs:
            self.stairs.remove(stair)

    # =====================================================
    # Elevators
    # =====================================================

    def add_elevator(self, elevator):

        self.elevators.append(elevator)

    def remove_elevator(self, elevator):

        if elevator in self.elevators:
            self.elevators.remove(elevator)

    # =====================================================
    # Cameras
    # =====================================================

    def add_camera(self, camera):

        self.cameras.append(camera)

    def remove_camera(self, camera):

        if camera in self.cameras:
            self.cameras.remove(camera)

    # =====================================================
    # Detectors
    # =====================================================

    def add_detector(self, detector):

        self.detectors.append(detector)

    def remove_detector(self, detector):

        if detector in self.detectors:
            self.detectors.remove(detector)

    # =====================================================
    # Smoke Detectors
    # =====================================================

    def add_smoke_detector(self, smoke_detector):

        self.smoke_detectors.append(smoke_detector)

    def remove_smoke_detector(self, smoke_detector):

        if smoke_detector in self.smoke_detectors:
            self.smoke_detectors.remove(smoke_detector)

    # =====================================================
    # Heat Detectors
    # =====================================================

    def add_heat_detector(self, heat_detector):

        self.heat_detectors.append(heat_detector)

    def remove_heat_detector(self, heat_detector):

        if heat_detector in self.heat_detectors:
            self.heat_detectors.remove(heat_detector)

    # =====================================================
    # Speakers
    # =====================================================

    def add_speaker(self, speaker):

        self.speakers.append(speaker)

    def remove_speaker(self, speaker):

        if speaker in self.speakers:
            self.speakers.remove(speaker)

    # =====================================================
    # Dynamic Evacuation Signs
    # =====================================================

    def add_sign(self, sign):

        self.signs.append(sign)

    def remove_sign(self, sign):

        if sign in self.signs:
            self.signs.remove(sign)

    # =====================================================
    # Manual Call Points
    # =====================================================

    def add_manual_call_point(self, manual_call_point):

        self.manual_call_points.append(manual_call_point)

    def remove_manual_call_point(self, manual_call_point):

        if manual_call_point in self.manual_call_points:
            self.manual_call_points.remove(manual_call_point)

    # =====================================================
    # Emergency Lights
    # =====================================================

    def add_emergency_light(self, emergency_light):

        self.emergency_lights.append(emergency_light)

    def remove_emergency_light(self, emergency_light):

        if emergency_light in self.emergency_lights:
            self.emergency_lights.remove(emergency_light)

    # =====================================================
    # Sprinklers
    # =====================================================

    def add_sprinkler(self, sprinkler):

        self.sprinklers.append(sprinkler)

    def remove_sprinkler(self, sprinkler):

        if sprinkler in self.sprinklers:
            self.sprinklers.remove(sprinkler)

    # =====================================================
    # Fire Extinguishers
    # =====================================================

    def add_fire_extinguisher(self, fire_extinguisher):

        self.fire_extinguishers.append(fire_extinguisher)

    def remove_fire_extinguisher(self, fire_extinguisher):

        if fire_extinguisher in self.fire_extinguishers:
            self.fire_extinguishers.remove(fire_extinguisher)

    # =====================================================
    # Fire Hydrants / Landing Valves
    # =====================================================

    def add_fire_hydrant(self, fire_hydrant):

        self.fire_hydrants.append(fire_hydrant)

    def remove_fire_hydrant(self, fire_hydrant):

        if fire_hydrant in self.fire_hydrants:
            self.fire_hydrants.remove(fire_hydrant)

    # =====================================================
    # Hose Reels
    # =====================================================

    def add_hose_reel(self, hose_reel):

        self.hose_reels.append(hose_reel)

    def remove_hose_reel(self, hose_reel):

        if hose_reel in self.hose_reels:
            self.hose_reels.remove(hose_reel)

    # =====================================================
    # Fire Water Tanks
    # =====================================================

    def add_fire_water_tank(self, fire_water_tank):

        self.fire_water_tanks.append(fire_water_tank)

    def remove_fire_water_tank(self, fire_water_tank):

        if fire_water_tank in self.fire_water_tanks:
            self.fire_water_tanks.remove(fire_water_tank)

    # =====================================================
    # Fire Pumps
    # =====================================================

    def add_fire_pump(self, fire_pump):

        self.fire_pumps.append(fire_pump)

    def remove_fire_pump(self, fire_pump):

        if fire_pump in self.fire_pumps:
            self.fire_pumps.remove(fire_pump)

    # =====================================================
    # Jockey Pumps
    # =====================================================

    def add_jockey_pump(self, jockey_pump):

        self.jockey_pumps.append(jockey_pump)

    def remove_jockey_pump(self, jockey_pump):

        if jockey_pump in self.jockey_pumps:
            self.jockey_pumps.remove(jockey_pump)

    # =====================================================
    # Fire Service Inlets
    # =====================================================

    def add_fire_service_inlet(self, fire_service_inlet):

        self.fire_service_inlets.append(fire_service_inlet)

    def remove_fire_service_inlet(self, fire_service_inlet):

        if fire_service_inlet in self.fire_service_inlets:
            self.fire_service_inlets.remove(fire_service_inlet)

    # =====================================================
    # Assembly Points
    # =====================================================

    def add_assembly_point(self, assembly_point):

        self.assembly_points.append(assembly_point)

    def remove_assembly_point(self, assembly_point):

        if assembly_point in self.assembly_points:
            self.assembly_points.remove(assembly_point)

    # =====================================================
    # Obstacles
    # =====================================================

    def add_obstacle(self, obstacle):

        self.obstacles.append(obstacle)

    def remove_obstacle(self, obstacle):

        if obstacle in self.obstacles:
            self.obstacles.remove(obstacle)

    # =====================================================
    # Doors
    # =====================================================

    def add_door(self, door):

        self.doors.append(door)

    def remove_door(self, door):

        if door in self.doors:
            self.doors.remove(door)

    # =====================================================

    @property
    def zone_count(self):

        return len(self.zones)

    @property
    def exit_count(self):

        return len(self.exits)

    @property
    def stair_count(self):

        return len(self.stairs)

    @property
    def camera_count(self):

        return len(self.cameras)

    @property
    def detector_count(self):

        return len(self.detectors)

    @property
    def smoke_detector_count(self):

        return len(self.smoke_detectors)

    @property
    def heat_detector_count(self):

        return len(self.heat_detectors)

    @property
    def speaker_count(self):

        return len(self.speakers)

    @property
    def sign_count(self):

        return len(self.signs)

    @property
    def manual_call_point_count(self):

        return len(self.manual_call_points)

    @property
    def emergency_light_count(self):

        return len(self.emergency_lights)

    @property
    def sprinkler_count(self):

        return len(self.sprinklers)

    @property
    def fire_extinguisher_count(self):

        return len(self.fire_extinguishers)

    @property
    def fire_hydrant_count(self):

        return len(self.fire_hydrants)

    @property
    def hose_reel_count(self):

        return len(self.hose_reels)

    @property
    def fire_water_tank_count(self):

        return len(self.fire_water_tanks)

    @property
    def fire_pump_count(self):

        return len(self.fire_pumps)

    @property
    def jockey_pump_count(self):

        return len(self.jockey_pumps)

    @property
    def fire_service_inlet_count(self):

        return len(self.fire_service_inlets)

    @property
    def assembly_point_count(self):

        return len(self.assembly_points)

    @property
    def obstacle_count(self):

        return len(self.obstacles)

    @property
    def door_count(self):

        return len(self.doors)

    # =====================================================

    def to_dict(self):

        return {
            "id": self.id,
            "name": self.name,

            "display_order": self.display_order,

            "height": self.height,

            "floor_plan": self.floor_plan,

            "visible": self.visible,
            "locked": self.locked,

            "zones": [
                zone.to_dict()
                for zone in self.zones
            ],

            "exits": [
                exit_obj.to_dict()
                for exit_obj in self.exits
            ],

            "stairs": [
                stair.to_dict()
                for stair in self.stairs
            ],

            "elevators": [
                elevator.to_dict()
                for elevator in self.elevators
            ],

            "cameras": [
                camera.to_dict()
                for camera in self.cameras
            ],

            "detectors": [
                detector.to_dict()
                for detector in self.detectors
            ],

            "smoke_detectors": [
                smoke_detector.to_dict()
                for smoke_detector in self.smoke_detectors
            ],

            "heat_detectors": [
                heat_detector.to_dict()
                for heat_detector in self.heat_detectors
            ],

            "speakers": [
                speaker.to_dict()
                for speaker in self.speakers
            ],

            "signs": [
                sign.to_dict()
                for sign in self.signs
            ],

            "manual_call_points": [
                mcp.to_dict()
                for mcp in self.manual_call_points
            ],

            "emergency_lights": [
                light.to_dict()
                for light in self.emergency_lights
            ],

            "sprinklers": [
                sprinkler.to_dict()
                for sprinkler in self.sprinklers
            ],

            "fire_extinguishers": [
                fire_extinguisher.to_dict()
                for fire_extinguisher in self.fire_extinguishers
            ],

            "fire_hydrants": [
                fire_hydrant.to_dict()
                for fire_hydrant in self.fire_hydrants
            ],

            "hose_reels": [
                hose_reel.to_dict()
                for hose_reel in self.hose_reels
            ],

            "fire_water_tanks": [
                tank.to_dict()
                for tank in self.fire_water_tanks
            ],

            "fire_pumps": [
                pump.to_dict()
                for pump in self.fire_pumps
            ],

            "jockey_pumps": [
                pump.to_dict()
                for pump in self.jockey_pumps
            ],

            "fire_service_inlets": [
                inlet.to_dict()
                for inlet in self.fire_service_inlets
            ],

            "assembly_points": [
                assembly_point.to_dict()
                for assembly_point in self.assembly_points
            ],

            "obstacles": [
                obstacle.to_dict()
                for obstacle in self.obstacles
            ],

            "doors": [
                door.to_dict()
                for door in self.doors
            ],
        }

    # =====================================================

    @classmethod
    def from_dict(cls, data):

        floor = cls(

            id=data["id"],

            name=data.get(
                "name",
                "",
            ),

            display_order=data.get(
                "display_order",
                0,
            ),

            # "elevation" is deliberately not read here even if an
            # older .syn file still has it -- it is derived, not
            # loaded. See Building.floor_elevation().
            height=data.get(
                "height",
                3.0,
            ),

            floor_plan=data.get(
                "floor_plan",
                "",
            ),

            visible=data.get(
                "visible",
                True,
            ),

            locked=data.get(
                "locked",
                False,
            ),
        )

        for zone_data in data.get("zones", []):

            floor.zones.append(
                Zone.from_dict(zone_data)
            )

        for exit_data in data.get("exits", []):

            floor.exits.append(
                Exit.from_dict(exit_data)
            )

        for stair_data in data.get("stairs", []):

            floor.stairs.append(
                Staircase.from_dict(stair_data)
            )

        for elevator_data in data.get("elevators", []):

            floor.elevators.append(
                Elevator.from_dict(elevator_data)
            )

        for camera_data in data.get("cameras", []):

            floor.cameras.append(
                Camera.from_dict(camera_data)
            )

        for detector_data in data.get("detectors", []):

            floor.detectors.append(
                Detector.from_dict(detector_data)
            )

        for smoke_detector_data in data.get("smoke_detectors", []):

            floor.smoke_detectors.append(
                SmokeDetector.from_dict(smoke_detector_data)
            )

        for heat_detector_data in data.get("heat_detectors", []):

            floor.heat_detectors.append(
                HeatDetector.from_dict(heat_detector_data)
            )

        for speaker_data in data.get("speakers", []):

            floor.speakers.append(
                Speaker.from_dict(speaker_data)
            )

        for sign_data in data.get("signs", []):

            floor.signs.append(
                DynamicEvacuationSign.from_dict(sign_data)
            )

        for mcp_data in data.get("manual_call_points", []):

            floor.manual_call_points.append(
                ManualCallPoint.from_dict(mcp_data)
            )

        for emergency_light_data in data.get("emergency_lights", []):

            floor.emergency_lights.append(
                EmergencyLight.from_dict(emergency_light_data)
            )

        for sprinkler_data in data.get("sprinklers", []):

            floor.sprinklers.append(
                Sprinkler.from_dict(sprinkler_data)
            )

        for fire_extinguisher_data in data.get("fire_extinguishers", []):

            floor.fire_extinguishers.append(
                FireExtinguisher.from_dict(fire_extinguisher_data)
            )

        for fire_hydrant_data in data.get("fire_hydrants", []):

            floor.fire_hydrants.append(
                FireHydrant.from_dict(fire_hydrant_data)
            )

        for hose_reel_data in data.get("hose_reels", []):

            floor.hose_reels.append(
                HoseReel.from_dict(hose_reel_data)
            )

        for tank_data in data.get("fire_water_tanks", []):

            floor.fire_water_tanks.append(
                FireWaterTank.from_dict(tank_data)
            )

        for pump_data in data.get("fire_pumps", []):

            floor.fire_pumps.append(
                FirePump.from_dict(pump_data)
            )

        for jockey_pump_data in data.get("jockey_pumps", []):

            floor.jockey_pumps.append(
                JockeyPump.from_dict(jockey_pump_data)
            )

        for inlet_data in data.get("fire_service_inlets", []):

            floor.fire_service_inlets.append(
                FireServiceInlet.from_dict(inlet_data)
            )

        for assembly_point_data in data.get("assembly_points", []):

            floor.assembly_points.append(
                AssemblyPoint.from_dict(assembly_point_data)
            )

        for obstacle_data in data.get("obstacles", []):

            floor.obstacles.append(
                Obstacle.from_dict(obstacle_data)
            )

        for door_data in data.get("doors", []):

            floor.doors.append(
                Door.from_dict(door_data)
            )

        return floor
