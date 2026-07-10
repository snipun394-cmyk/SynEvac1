from dataclasses import dataclass, field
from uuid import uuid4

from models.zone import Zone
from models.exit import Exit
from models.staircase import Staircase
from models.elevator import Elevator
from models.camera import Camera
from models.detector import Detector
from models.assembly_point import AssemblyPoint


@dataclass
class Floor:

    name: str

    id: str = field(default_factory=lambda: str(uuid4()))

    # =====================================================
    # Ordering vs. physical position
    #
    # display_order controls UI ordering / active-floor
    # navigation. elevation is the physical vertical
    # position. height is the vertical extent of the floor.
    # These are independent and must never be derived from
    # one another.
    # =====================================================

    display_order: int = 0

    elevation: float = 0.0
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

    assembly_points: list[AssemblyPoint] = field(default_factory=list)

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
    # Assembly Points
    # =====================================================

    def add_assembly_point(self, assembly_point):

        self.assembly_points.append(assembly_point)

    def remove_assembly_point(self, assembly_point):

        if assembly_point in self.assembly_points:
            self.assembly_points.remove(assembly_point)

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
    def assembly_point_count(self):

        return len(self.assembly_points)

    # =====================================================

    def to_dict(self):

        return {
            "id": self.id,
            "name": self.name,

            "display_order": self.display_order,

            "elevation": self.elevation,
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

            "assembly_points": [
                assembly_point.to_dict()
                for assembly_point in self.assembly_points
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

            elevation=data.get(
                "elevation",
                0.0,
            ),

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

        for assembly_point_data in data.get("assembly_points", []):

            floor.assembly_points.append(
                AssemblyPoint.from_dict(assembly_point_data)
            )

        return floor
