from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class SensorStatus:

    # A read-only snapshot of one sensor's management state -- same
    # "derived, never stored anywhere, produced fresh by the manager"
    # role as camera_manager.status.CameraStatus, independently
    # restated here rather than imported (SensorManager must not
    # couple to camera_manager -- see SensorManager's own docstring).

    sensor_id: str
    sensor_type: str  # Sensor's own object_type -- "SmokeDetector"/"HeatDetector"
    name: str
    floor_id: str
    zone_ids: Tuple[str, ...]

    active: bool
    mode: str
    health_status: str

    # The sensor's last-computed DetectorState, if the caller supplied
    # one to SensorManager.sensor_status()/all_statuses() -- None when
    # no state has been computed yet (SensorManager never computes a
    # state itself; see that class's own docstring on why).
    current_state: Optional[object] = None
