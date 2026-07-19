from dataclasses import dataclass

from models.sensor_asset import SensorAsset


@dataclass
class Speaker(SensorAsset):

    # A physical voice-evacuation / public-address speaker in the
    # Building Digital Twin -- reuses SensorAsset exactly as SmokeDetector/
    # HeatDetector already do (id/name/floor_id/zone_ids/position/
    # mount_height/active/mode/connection/health_status), since a speaker
    # is an output device with the same asset-management needs (which
    # zone(s) it serves, whether it is currently active/healthy) as an
    # input device, not a different shape. Deliberately adds only two
    # fields beyond that -- no acoustic simulation, no sound-pressure-
    # level modelling, no coverage_radius: a speaker's "coverage" is its
    # explicit zone_ids assignment, the same logical (not geometric)
    # routing convention Camera/Detector zone assignment already
    # establishes, and this milestone's own explicit scope boundary
    # ("logical zone coverage is sufficient").

    # Exposed for the Property Panel combo box, same convention
    # Detector.DETECTOR_TYPES/Zone.ZONE_TYPES already use for a small
    # closed set.
    SPEAKER_TYPES = (
        "Ceiling",
        "Wall",
        "Horn",
    )

    speaker_type: str = "Ceiling"

    # Decibels -- informational/logical only, never fed into any
    # acoustic model anywhere in this codebase. A future Live adapter's
    # own amplifier integration is the one place this would ever become
    # operationally meaningful.
    volume_level: float = 75.0

    # =====================================================

    def __post_init__(self):

        self.object_type = "Speaker"

    # =====================================================

    def move_to(self, x, y):

        self.position = (x, y)

    # =====================================================

    def to_dict(self):

        data = self._sensor_dict()

        data.update({
            "speaker_type": self.speaker_type,
            "volume_level": self.volume_level,
        })

        return data

    # =====================================================

    @classmethod
    def from_dict(cls, data):

        kwargs = cls._sensor_kwargs(data)

        kwargs.update(
            speaker_type=data.get(
                "speaker_type",
                "Ceiling",
            ),

            volume_level=data.get(
                "volume_level",
                75.0,
            ),
        )

        return cls(**kwargs)
