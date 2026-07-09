from dataclasses import dataclass, field

from models.floor import Floor


@dataclass
class Building:
    name: str
    floors: list[Floor] = field(default_factory=list)

    def add_floor(self, floor: Floor):
        self.floors.append(floor)

    def get_floor(self, floor_id: str):
        for floor in self.floors:
            if floor.id == floor_id:
                return floor
        return None