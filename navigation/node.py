from dataclasses import dataclass


@dataclass
class Node:

    id: str

    x: float
    y: float

    zone_id: str = ""

    def position(self):

        return (self.x, self.y)