from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ManualCallPointReading:

    # Manual Call Point -> Live Emergency Response Integration milestone
    # -- the same shape perception.models.smoke_detector_observation.
    # SmokeDetectorReading/heat_detector_observation.HeatDetectorReading
    # already establish, extended to the one remaining FACP alarm
    # source (see facp/models.py::DetectorConditionReport, which already
    # treats ManualCallPoint as a peer of Smoke/Heat). Unlike those two,
    # a Manual Call Point has no external hazard-threshold reading to
    # report -- ManualCallPoint.compute_state(time) is already the
    # complete, self-contained answer (a direct human action, not a
    # sensor threshold) -- so alarm_active here is always derived
    # directly from that same compute_state() call, never a second,
    # independent computation. confidence stays None always: a human
    # pressing a call point is a definite binary action, not a
    # probabilistic sensor reading, so there is no honest confidence
    # value to report.

    detector_id: str
    timestamp: float
    alarm_active: bool

    confidence: Optional[float] = None
