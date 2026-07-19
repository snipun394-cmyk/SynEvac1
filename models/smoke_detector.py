from dataclasses import dataclass
from typing import Optional

from models.sensor_asset import DetectorState, HealthStatus, SensorAsset


@dataclass
class SmokeDetector(SensorAsset):

    # Phase 3's Smoke Detector asset -- Designer/engineering data only,
    # exactly as pure as Camera is pure geometry. This class imports
    # nothing from hazard/hazard_evolution/perception: compute_state()
    # below takes a plain float smoke_level reading (whatever produced
    # it -- a real HazardSnapshot.node_state(zone_id).smoke_level, a
    # Designer "test" value, a future replay log), never a HazardSnapshot/
    # HazardProvider object itself. That's what keeps "Simulation mode
    # should use the existing hazard information" true without this
    # framework needing to import Hazard Evolution's own types -- the
    # caller (a test, a Designer panel, a future integration point)
    # is the one place that ever touches HazardSnapshot, exactly the
    # same responsibility split GroundTruthSmokeDetectorProvider
    # already established one layer up (it reads HazardProvider so
    # Detector itself never has to).

    activation_threshold: float = 0.2

    # =====================================================

    def __post_init__(self):

        self.object_type = "SmokeDetector"

    # =====================================================

    def move_to(self, x, y):

        self.position = (x, y)

    # =====================================================

    def compute_state(self, smoke_level: Optional[float], time: Optional[float] = None) -> DetectorState:

        # Health overrides everything -- a faulty/offline device's
        # reading (if it produces one at all) cannot be trusted, so it
        # is never allowed to report ALARM or NORMAL.
        if self.health_status != HealthStatus.OK:
            return DetectorState.FAULT

        # An inactive device is deliberately not watching at all --
        # NORMAL (not FAULT), the same "inactive means off, not
        # broken" distinction Camera.active/EngineeringAsset.active
        # already carry.
        if not self.active:
            return DetectorState.NORMAL

        # No reading available this instant (e.g. this zone has no
        # smoke_level Ground Truth recorded, same "None means not
        # modeled" convention GroundTruthSmokeDetectorProvider already
        # follows) -- nothing to alarm on, but this is not a fault
        # either.
        if smoke_level is None:
            return DetectorState.NORMAL

        if smoke_level >= self.activation_threshold:

            if time is not None:
                self.last_activation_time = time

            return DetectorState.ALARM

        return DetectorState.NORMAL

    # =====================================================

    def to_dict(self):

        data = self._sensor_dict()

        data.update({
            "activation_threshold": self.activation_threshold,
        })

        return data

    # =====================================================

    @classmethod
    def from_dict(cls, data):

        kwargs = cls._sensor_kwargs(data)

        kwargs.update(
            activation_threshold=data.get(
                "activation_threshold",
                0.2,
            ),
        )

        return cls(**kwargs)
