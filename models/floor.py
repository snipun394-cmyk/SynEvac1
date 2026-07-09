from dataclasses import dataclass, field

from models.zone import Zone
from models.exit import Exit
from models.staircase import Staircase
from models.camera import Camera
from models.detector import Detector


@dataclass
class Floor:

    id: str
    name: str

    elevation: float = 0.0
    floor_plan: str = ""

    zones: list[Zone] = field(default_factory=list)
    exits: list[Exit] = field(default_factory=list)
    stairs: list[Staircase] = field(default_factory=list)
    elevators: list = field(default_factory=list)

    cameras: list[Camera] = field(default_factory=list)
    detectors: list[Detector] = field(default_factory=list)

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

    # =====================================================

    def to_dict(self):

        return {
            "id": self.id,
            "name": self.name,
            "elevation": self.elevation,
            "floor_plan": self.floor_plan,

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
        }