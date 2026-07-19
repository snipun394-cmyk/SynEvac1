from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from scenario_definition.distributions import Distribution, distribution_from_dict


@dataclass(frozen=True)
class EventTemplate:

    # Describes an event that MAY occur -- never a resolved
    # ScenarioEvent (architecture doc §3.1 finding 2/§3.2's Events row/
    # §6). "Door closure between 60-120s" is
    # EventTemplate(target_type="door", target_id=<id>,
    # event_type="close", occurs=FixedValue(True),
    # time=UniformRange(60, 120)); "Camera failure after 90s" is
    # occurs=WeightedOptions({True: p, False: 1 - p}),
    # time=UniformRange(90, scenario_horizon). A pinned/always-happens
    # event is simply occurs=FixedValue(True) -- the degenerate case
    # of the same mechanism, not a separate fixed_events list (§3.1
    # finding 2: that sibling list was a violation, already removed).
    #
    # This module never imports scenario.ScenarioEvent (the resolved-
    # output type) -- doing so would be exactly the violation §3.1
    # finding 2 corrected. Turning a sampled occurs=True/time draw into
    # an actual ScenarioEvent is exclusively the Generator's job.

    target_type: str
    target_id: str
    event_type: str

    occurs: Distribution
    time: Distribution

    parameters: Mapping[str, Any] = field(default_factory=dict)

    # =====================================================

    def __post_init__(self):

        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))

    # =====================================================

    def to_dict(self) -> dict:

        return {
            "target_type": self.target_type,
            "target_id": self.target_id,
            "event_type": self.event_type,
            "occurs": self.occurs.to_dict(),
            "time": self.time.to_dict(),
            "parameters": dict(self.parameters),
        }

    # =====================================================

    @classmethod
    def from_dict(cls, data: dict) -> "EventTemplate":

        return cls(
            target_type=data["target_type"],
            target_id=data["target_id"],
            event_type=data["event_type"],
            occurs=distribution_from_dict(data["occurs"]),
            time=distribution_from_dict(data["time"]),
            parameters=data.get("parameters", {}),
        )
