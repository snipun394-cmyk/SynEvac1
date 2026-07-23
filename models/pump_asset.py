from dataclasses import dataclass

from models.engineering_asset import EngineeringAsset
from models.sensor_asset import HealthStatus


class PumpOperationalState:

    # STOPPED/RUNNING are this pump's own reported run state; FAULT/
    # UNAVAILABLE mirror the same "device fault outranks anything else,
    # inactive means unavailable regardless of the running flag"
    # priority every other asset in this codebase already applies.
    # Deliberately plain string constants (Property Panel combo box
    # convention, same as HealthStatus/DeviceMode) rather than an Enum.
    #
    # RUNNING here means ONLY "the digital twin reports this pump as
    # running" -- never a claim about discharge pressure, flow, or
    # whether any downstream sprinkler/hydrant/hose reel actually
    # receives adequate water (see docs/architecture/
    # fire_water_infrastructure.md's own "operational state != hydraulic
    # performance" boundary). No pump curve, head, NPSH, or motor
    # performance calculation exists anywhere in this class.

    STOPPED = "STOPPED"
    RUNNING = "RUNNING"
    FAULT = "FAULT"
    UNAVAILABLE = "UNAVAILABLE"

    ALL = (STOPPED, RUNNING, FAULT, UNAVAILABLE)


class PumpControlMode:

    # Whether this pump is expected to start automatically (on a
    # pressure-drop signal a real fire pump controller would sense) or
    # only via manual operator action. Purely descriptive/informational
    # here -- this class has no pressure-sensing logic of any kind, so
    # nothing ever flips `running` based on control_mode; a caller
    # (Designer today; a future live integration) still sets `running`
    # directly, exactly like every other digital-twin asset in this
    # codebase reports its own state as an intrinsic fact rather than
    # something this framework derives from physics it does not model.

    AUTOMATIC = "Automatic"
    MANUAL = "Manual"

    ALL = (AUTOMATIC, MANUAL)


@dataclass
class PumpAsset(EngineeringAsset):

    # Shared foundation for FirePump/JockeyPump -- deliberately a real
    # subclass relationship (Phase 5 of the Fire Water Supply &
    # Suppression Infrastructure milestone's own "choose based on
    # actual shared semantics, not convenience" instruction), unlike
    # FireExtinguisher/FireHydrant/HoseReel (models/fire_safety_asset.py),
    # which only share a small computation, not a type. A jockey pump
    # IS a (smaller-duty, pressure-maintenance) fire pump -- the same
    # "is-a" relationship SmokeDetector/HeatDetector/ManualCallPoint
    # already share through SensorAsset, not a coincidental shape
    # overlap between otherwise-unrelated devices.
    #
    # Reuses EngineeringAsset exactly as every other asset in this
    # codebase does (id/name/floor_id/zone_ids/position/mount_height/
    # active/mode/connection); adds only what a pump genuinely needs
    # beyond that shape.

    health_status: str = HealthStatus.OK

    control_mode: str = PumpControlMode.AUTOMATIC

    # The pump's own intrinsic run state -- set directly (mirrors
    # models.manual_call_point.ManualCallPoint.activated exactly: a
    # direct, caller-reported fact, not derived from a continuous
    # external reading this framework does not have).
    running: bool = False

    # =====================================================

    def compute_state(self) -> str:

        if self.health_status != HealthStatus.OK:
            return PumpOperationalState.FAULT

        if not self.active:
            return PumpOperationalState.UNAVAILABLE

        if self.running:
            return PumpOperationalState.RUNNING

        return PumpOperationalState.STOPPED

    # =====================================================

    def _pump_dict(self):

        data = self._asset_dict()

        data.update({
            "health_status": self.health_status,
            "control_mode": self.control_mode,
            "running": self.running,
        })

        return data

    # =====================================================

    @classmethod
    def _pump_kwargs(cls, data):

        kwargs = cls._asset_kwargs(data)

        kwargs.update(
            health_status=data.get(
                "health_status",
                HealthStatus.OK,
            ),

            control_mode=data.get(
                "control_mode",
                PumpControlMode.AUTOMATIC,
            ),

            running=data.get(
                "running",
                False,
            ),
        )

        return kwargs
