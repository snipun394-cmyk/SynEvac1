from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4

from models.building import Building


@dataclass
class Project:

    id: str = field(default_factory=lambda: str(uuid4()))

    name: str = "Untitled Project"

    author: str = ""

    version: str = "1.0"

    created_at: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )

    modified_at: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )

    building: Building | None = None

    # =====================================================

    def set_building(self, building: Building):

        self.building = building

        self.touch()

    # =====================================================

    @classmethod
    def new_default(cls):

        project = cls(name="Untitled Project")

        building = Building(name="Building")
        building.create_floor(name="Ground Floor")

        project.set_building(building)

        return project

    # =====================================================

    def touch(self):

        self.modified_at = datetime.now().isoformat()

    # =====================================================

    def floor_count(self):

        if self.building is None:
            return 0

        return len(self.building.floors)

    # =====================================================

    def get_floor(self, floor_id):

        if self.building is None:
            return None

        for floor in self.building.floors:

            if floor.id == floor_id:
                return floor

        return None

    # =====================================================

    def to_dict(self):

        return {
            "id": self.id,
            "name": self.name,
            "author": self.author,
            "version": self.version,
            "created_at": self.created_at,
            "modified_at": self.modified_at,
            "building": (
                self.building.to_dict()
                if self.building
                else None
            ),
        }

    # =====================================================

    @classmethod
    def from_dict(cls, data):

        project = cls(
            id=data["id"],
            name=data.get(
                "name",
                "Untitled Project",
            ),
            author=data.get(
                "author",
                "",
            ),
            version=data.get(
                "version",
                "1.0",
            ),
            created_at=data.get(
                "created_at",
                datetime.now().isoformat(),
            ),
            modified_at=data.get(
                "modified_at",
                datetime.now().isoformat(),
            ),
        )

        if data.get("building"):

            project.building = Building.from_dict(
                data["building"]
            )

        return project