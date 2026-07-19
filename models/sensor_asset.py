from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from models.engineering_asset import EngineeringAsset


class HealthStatus:

    # The physical device's own self-reported condition -- independent
    # of whatever hazard reading it's currently seeing. A device can be
    # perfectly healthy and quietly reading "no smoke" (Normal), or
    # unhealthy and unable to produce a trustworthy reading at all
    # (Fault/Offline) regardless of the real hazard level. Plain string
    # constants, not an Enum, matching Detector.DETECTOR_TYPES' own
    # convention for a small closed set exposed to a Property Panel
    # combo box.

    OK = "OK"
    FAULT = "Fault"
    OFFLINE = "Offline"

    ALL = (OK, FAULT, OFFLINE)


class DetectorState(Enum):

    # The one three-state model every fire-protection sensor in this
    # framework exposes (Phase 3's own requirement, reused as-is by
    # Phase 4's Heat Detector rather than invented a second time).
    # NORMAL and ALARM are the hazard-driven states; FAULT means the
    # device itself (HealthStatus != OK) cannot be trusted regardless
    # of what the hazard reading says -- a real panel shows a device
    # fault distinctly from "all clear" for exactly this reason.

    NORMAL = auto()
    ALARM = auto()
    FAULT = auto()


@dataclass
class SensorAsset(EngineeringAsset):

    # The reusable foundation Phase 2 asks for -- shared shape for
    # every fire-protection sensor this Building Digital Twin will
    # ever place (Smoke Detector and Heat Detector today; Flame
    # Detector, Gas Detector, Manual Call Point are future candidates
    # that fit this same shape without this class changing, the same
    # "future device" role models.engineering_asset.EngineeringAsset
    # already documents for itself one level down).
    #
    # Deliberately adds ONLY the three fields EngineeringAsset doesn't
    # already provide -- id/name (BaseObject), floor_id/zone_ids/
    # position/active/mode (EngineeringAsset) are reused completely
    # unchanged, never redeclared here. This is Phase 2's own explicit
    # rule: "Do not duplicate functionality already provided by
    # EngineeringAsset."
    #
    # This class is plain data plus its own to_dict()/from_dict()
    # support -- no hazard/perception logic of any kind lives here (see
    # SmokeDetector.compute_state()/HeatDetector.compute_state() for
    # why those stay pure functions of a plain float reading, never an
    # import of hazard/perception types into this framework at all).

    # The device's own self-reported condition -- see HealthStatus.
    health_status: str = HealthStatus.OK

    # Free-text/ISO-date metadata, optional -- "installed 2026-01-15",
    # or empty if never recorded. Never parsed or validated here; a
    # future maintenance-scheduling feature could read it, this
    # framework does not.
    installation_date: str = ""

    # The last time this sensor's own compute_state() (see subclasses)
    # produced ALARM -- None until it first does. Unlike every other
    # field on this class, this one IS mutated as a side effect of
    # compute_state() rather than only ever set by a user/Designer edit
    # -- the same "a Camera has geometry the Designer sets, but its
    # coverage_polygon() is still derived fresh each call" split, one
    # level further: here the derived thing (ALARM) is allowed to
    # leave a permanent mark (when it last happened), because "last
    # activation time" is explicitly asked for as sensor state, not
    # geometry.
    last_activation_time: Optional[float] = None

    # =====================================================

    def _sensor_dict(self):

        data = self._asset_dict()

        data.update({
            "health_status": self.health_status,
            "installation_date": self.installation_date,
            "last_activation_time": self.last_activation_time,
        })

        return data

    # =====================================================

    @classmethod
    def _sensor_kwargs(cls, data):

        kwargs = cls._asset_kwargs(data)

        kwargs.update(
            health_status=data.get(
                "health_status",
                HealthStatus.OK,
            ),

            installation_date=data.get(
                "installation_date",
                "",
            ),

            last_activation_time=data.get(
                "last_activation_time",
                None,
            ),
        )

        return kwargs
