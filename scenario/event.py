from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping


@dataclass(frozen=True)
class ScenarioEvent:

    # One resolved, sampled event inside a generated Scenario --
    # architecture doc §3.2/§6/§7. This is the *resolved* output of an
    # EventTemplate whose occurs Distribution sampled True -- a
    # concrete, scheduled Building-state trigger, never simulation
    # physics and never containing execution logic of its own. A
    # ScenarioEvent that exists always happens; there is no separate
    # occurs field here the way there is on the Definition-side
    # EventTemplate, because an event that did not occur is simply
    # never materialized as one of these in the first place.

    event_id: str

    target_type: str
    target_id: str

    event_type: str

    time: float

    parameters: Mapping[str, Any] = field(default_factory=dict)

    # =====================================================

    def __post_init__(self):

        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))

    # =====================================================

    def to_dict(self) -> dict:

        return {
            "event_id": self.event_id,
            "target_type": self.target_type,
            "target_id": self.target_id,
            "event_type": self.event_type,
            "time": self.time,
            "parameters": dict(self.parameters),
        }

    # =====================================================

    @classmethod
    def from_dict(cls, data: dict) -> "ScenarioEvent":

        return cls(
            event_id=data["event_id"],
            target_type=data["target_type"],
            target_id=data["target_id"],
            event_type=data["event_type"],
            time=data["time"],
            parameters=data.get("parameters", {}),
        )
