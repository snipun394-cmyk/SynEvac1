from dataclasses import dataclass, field

from models.floor import Floor
from models.zone import Zone
from models.exit import Exit
from models.staircase import Staircase
from models.camera import Camera
from models.detector import Detector


@dataclass
class Building:

    id: str
    name: str

    floors: list[Floor] = field(default_factory=list)

    # =====================================================

    def add_floor(self, floor: Floor):

        self.floors.append(floor)

    # =====================================================

    def remove_floor(self, floor: Floor):

        if floor in self.floors:
            self.floors.remove(floor)

    # =====================================================

    def get_floor(self, floor_id: str):

        for floor in self.floors:

            if floor.id == floor_id:
                return floor

        return None

    # =====================================================

    @property
    def floor_count(self):

        return len(self.floors)

    # =====================================================

    def to_dict(self):

        return {
            "id": self.id,
            "name": self.name,
            "floors": [
                floor.to_dict()
                for floor in self.floors
            ],
        }

    # =====================================================

    @classmethod
    def from_dict(cls, data):

        building = cls(
            id=data["id"],
            name=data["name"],
        )

        for floor_data in data.get("floors", []):

            floor = Floor(
                id=floor_data["id"],
                name=floor_data["name"],
                elevation=floor_data.get(
                    "elevation",
                    0.0,
                ),
                floor_plan=floor_data.get(
                    "floor_plan",
                    "",
                ),
            )

            # =============================================
            # Zones
            # =============================================

            for zone_data in floor_data.get(
                "zones",
                [],
            ):

                floor.zones.append(
                    Zone.from_dict(zone_data)
                )

            # =============================================
            # Exits
            # =============================================

            for exit_data in floor_data.get(
                "exits",
                [],
            ):

                floor.exits.append(
                    Exit.from_dict(exit_data)
                )

            # =============================================
            # Stairs
            # =============================================

            for stair_data in floor_data.get(
                "stairs",
                [],
            ):

                floor.stairs.append(
                    Staircase.from_dict(
                        stair_data
                    )
                )

            # =============================================
            # Cameras
            # =============================================

            for camera_data in floor_data.get(
                "cameras",
                [],
            ):

                floor.cameras.append(
                    Camera.from_dict(
                        camera_data
                    )
                )

            # =============================================
            # Detectors
            # =============================================

            for detector_data in floor_data.get(
                "detectors",
                [],
            ):

                floor.detectors.append(
                    Detector.from_dict(
                        detector_data
                    )
                )

            building.add_floor(floor)

        return building