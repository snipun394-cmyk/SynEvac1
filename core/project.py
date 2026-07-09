from dataclasses import dataclass, field

from core.building import Building


@dataclass
class Project:
    name: str
    version: str = "1.0.0"
    building: Building = field(default_factory=lambda: Building(name="New Building"))