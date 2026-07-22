from dataclasses import dataclass, field
from typing import Tuple

from models.engineering_asset import EngineeringAsset


class SignIndication:

    # The complete, exhaustive vocabulary a Dynamic Evacuation Sign can
    # ever display -- Live Dynamic Evacuation Signage milestone, Phase 3.
    # Every member here is honestly derivable from real modeled geometry/
    # graph structure (see dynamic_signage/direction.py, dynamic_signage/
    # planner.py) -- nothing here is a fabricated richer semantic the
    # Building Model/Navigation Graph cannot actually support.

    LEFT = "LEFT"
    RIGHT = "RIGHT"
    STRAIGHT = "STRAIGHT"

    # Placard-style, not directional -- shown when the sign's own zone
    # IS the meaningful decision point itself (the sign is posted at the
    # stair/exit, not pointing toward one from elsewhere). See
    # dynamic_signage/route_mapping.py's own documented rule.
    EXIT_HERE = "EXIT_HERE"
    USE_STAIRS = "USE_STAIRS"

    # Honest degraded/negative states -- never a fabricated direction.
    DO_NOT_USE = "DO_NOT_USE"
    NO_SAFE_DIRECTION = "NO_SAFE_DIRECTION"
    UNAVAILABLE = "UNAVAILABLE"

    ALL = (
        LEFT, RIGHT, STRAIGHT, EXIT_HERE, USE_STAIRS,
        DO_NOT_USE, NO_SAFE_DIRECTION, UNAVAILABLE,
    )

    # The subset a real physical directional sign (arrows + a couple of
    # placards) can honestly be expected to show -- the default
    # `supported_indications` for a freshly placed sign. A user is free
    # to narrow this in the Property Panel (e.g. an arrow-only sign that
    # cannot show EXIT_HERE/USE_STAIRS placards); this is never widened
    # beyond SignIndication.ALL.
    DEFAULT_SUPPORTED = ALL


@dataclass
class DynamicEvacuationSign(EngineeringAsset):

    # The Building Digital Twin's own Dynamic Evacuation Sign asset --
    # Live Dynamic Evacuation Signage milestone, Phase 2. Reuses
    # EngineeringAsset exactly as Camera does (id/name/floor_id/
    # zone_ids/position/mount_height/active/mode/connection) -- a sign
    # is a physical device placed in the building with the same asset-
    # management needs as every other engineering asset, not a
    # different shape. Adds only ORIENTATION (Camera.rotation's own
    # convention, reused verbatim: 0 degrees points along +x, increasing
    # clockwise) and SUPPORTED_INDICATIONS -- no coverage geometry, no
    # acoustic/visual simulation, no hardware protocol fields of any
    # kind (no IP address, no Modbus/BACnet registers, no serial port,
    # no firmware assumptions -- this is a digital-twin-only asset, see
    # models/engineering_asset.py's own pre-existing ConnectionInfo for
    # where a real future integration would live instead).
    #
    # This class carries NO current indication/instruction state --
    # "what a sign currently shows" is dynamic_signage.provider.
    # DynamicSignageProvider's own responsibility (mirrors Speaker
    # carrying no VoiceMessage state, only zone_ids/active/health).

    orientation: float = 0.0

    supported_indications: Tuple[str, ...] = field(default_factory=lambda: SignIndication.DEFAULT_SUPPORTED)

    # =====================================================

    def __post_init__(self):

        self.object_type = "DynamicEvacuationSign"

    # =====================================================

    def move_to(self, x, y):

        self.position = (x, y)

    # =====================================================

    def rotate(self, angle):

        self.orientation = angle

    # =====================================================

    def to_dict(self):

        data = self._asset_dict()

        data.update({
            "orientation": self.orientation,
            "supported_indications": list(self.supported_indications),
        })

        return data

    # =====================================================

    @classmethod
    def from_dict(cls, data):

        kwargs = cls._asset_kwargs(data)

        kwargs.update(
            orientation=data.get(
                "orientation",
                0.0,
            ),

            supported_indications=tuple(
                data.get(
                    "supported_indications",
                    SignIndication.DEFAULT_SUPPORTED,
                )
            ),
        )

        return cls(**kwargs)
