from dataclasses import dataclass


@dataclass
class Edge:

    start: str
    end: str

    cost: float

    bidirectional: bool = True