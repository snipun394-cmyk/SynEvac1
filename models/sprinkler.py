from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from models.sensor_asset import HealthStatus, SensorAsset


class SprinklerActivationState(Enum):

    # Deliberately its OWN type -- structurally identical in shape to
    # models.sensor_asset.DetectorState (NORMAL/ALARM->ACTIVATED/FAULT)
    # but NOT that type, and never DetectorState itself. This is the
    # same "structurally distinct concepts get structurally distinct
    # types" precedent facp.models.PanelState's own docstring already
    # establishes relative to DetectorState, applied here for the
    # opposite reason: a Sprinkler activating is a physical water-
    # discharge event, not a fire-ALARM report, and this codebase's
    # FACP model only ever consumes DetectorState via
    # DetectorConditionReport for genuine initiating-device alarm
    # sources (Smoke/Heat/ManualCallPoint). Naming this ALARM (or
    # reusing DetectorState directly) would silently invite exactly
    # the assumption Phase 13 of this milestone's own brief warns
    # against -- "do NOT assume sprinkler activation belongs in FACP".
    # A Sprinkler is never registered with SensorManager and never
    # produces a DetectorConditionReport -- see
    # docs/architecture/fire_suppression_safety_assets.md §"Sprinkler
    # vs FACP" for the full reasoning.

    NORMAL = auto()
    ACTIVATED = auto()
    FAULT = auto()


@dataclass
class Sprinkler(SensorAsset):

    # A fixed automatic sprinkler head -- reuses SensorAsset exactly as
    # SmokeDetector/HeatDetector/ManualCallPoint do (id/name/floor_id/
    # zone_ids/position/mount_height/active/mode/connection/
    # health_status/installation_date/last_activation_time): a
    # Sprinkler genuinely has an installation date and a last-
    # activation time the same way those three siblings do, unlike
    # EmergencyLight/FireExtinguisher/FireHydrant/HoseReel (passive
    # resources with no activation concept at all).
    #
    # activation_temperature mirrors HeatDetector.activation_threshold's
    # own fixed-temperature-only scope exactly (Rate-of-Rise is equally
    # out of scope here) -- 68.0C is a commonly documented standard
    # (ordinary-hazard) glass-bulb/fusible-link rating, restated here
    # independently rather than imported from HeatDetector (these are
    # two different physical devices that happen to share a threshold-
    # comparison shape, not one built on the other).
    #
    # ACTIVATED here means ONLY "the digital twin believes this
    # sprinkler head has discharged" -- it is a fact about the DEVICE,
    # never a claim about the fire it was pointed at. See
    # compute_state()'s own docstring below and docs/architecture/
    # fire_suppression_safety_assets.md's "digital-twin asset state !=
    # physical fire-suppression effect" boundary: nothing in this
    # class, or anywhere this class is consumed, feeds ACTIVATED back
    # into hazard/fire_growth/smoke_propagation.

    activation_temperature: float = 68.0

    # =====================================================

    def __post_init__(self):

        self.object_type = "Sprinkler"

    # =====================================================

    def move_to(self, x, y):

        self.position = (x, y)

    # =====================================================

    def compute_state(self, temperature: Optional[float], time: Optional[float] = None) -> SprinklerActivationState:

        # Same FAULT > threshold > NORMAL priority HeatDetector.
        # compute_state() already establishes -- a genuinely unhealthy
        # sprinkler head is never trusted to have activated correctly
        # regardless of what temperature was observed near it.

        if self.health_status != HealthStatus.OK:
            return SprinklerActivationState.FAULT

        if not self.active:
            return SprinklerActivationState.NORMAL

        if temperature is None:
            return SprinklerActivationState.NORMAL

        if temperature >= self.activation_temperature:

            if time is not None:
                self.last_activation_time = time

            return SprinklerActivationState.ACTIVATED

        return SprinklerActivationState.NORMAL

    # =====================================================

    def to_dict(self):

        data = self._sensor_dict()

        data.update({
            "activation_temperature": self.activation_temperature,
        })

        return data

    # =====================================================

    @classmethod
    def from_dict(cls, data):

        kwargs = cls._sensor_kwargs(data)

        kwargs.update(
            activation_temperature=data.get(
                "activation_temperature",
                68.0,
            ),
        )

        return cls(**kwargs)
