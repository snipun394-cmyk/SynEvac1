from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Optional, Tuple
from uuid import uuid4

from hazard.severity import HazardSeverity

from models.sensor_asset import DetectorState

from occupancy.snapshot import OccupancySnapshot

from facp.models import FACPSnapshot

from building_control.snapshot import ControlStateSnapshot

from perception.models.camera_observation import CameraFrameObservation

from multi_camera_fusion.track import FusedTrack

from camera_manager.status import CameraStatus
from sensor_manager.status import SensorStatus


@dataclass(frozen=True)
class CameraAssetState:

    # One camera's asset-management status (online/offline, floor/zone
    # assignment -- see camera_manager.status.CameraStatus) paired with
    # whatever CameraFrameObservation that camera produced this cycle,
    # if any. Kept as two separate objects rather than one merged
    # dataclass so neither CameraManager nor Perception has to change
    # shape for this type to exist -- BuildingState only composes what
    # already exists.

    status: CameraStatus
    frame_observation: Optional[CameraFrameObservation] = None


@dataclass(frozen=True)
class DetectorAssetState:

    # One fire-protection sensor's asset-management status (see
    # sensor_manager.status.SensorStatus) paired with its current
    # typed reading -- a SmokeDetectorReading when this object lives in
    # BuildingState.smoke_detector_states, a HeatDetectorReading when it
    # lives in heat_detector_states. Not narrowed to either type here
    # (same "loosely typed, caller knows which mapping it came from"
    # convention SensorStatus.current_state already uses) so this one
    # class serves both mappings without duplication.

    status: SensorStatus
    reading: Optional[object] = None


@dataclass(frozen=True)
class HazardSummary:

    # A building-wide roll-up of HazardSnapshot, classified through
    # HazardSeverity.from_score() -- the one centralized place severity
    # cutoffs live (see hazard/severity.py) -- so nothing in this
    # package re-derives its own thresholds. Sparse: a zone absent from
    # zone_severities simply has no hazard reading this cycle, same
    # "absent means no reading" convention HazardSnapshot itself uses.

    overall_severity: HazardSeverity = HazardSeverity.NONE
    worst_zone_id: Optional[str] = None
    zone_severities: Mapping[str, HazardSeverity] = field(default_factory=dict)

    def __post_init__(self):

        object.__setattr__(self, "zone_severities", MappingProxyType(dict(self.zone_severities)))

    def severity_for(self, zone_id) -> HazardSeverity:

        return self.zone_severities.get(zone_id, HazardSeverity.NONE)


@dataclass(frozen=True)
class ActiveAssetsSummary:

    # Which engineering assets (Camera Assets, fire-protection sensors)
    # are currently active vs. offline -- derived straight from each
    # asset's own CameraStatus.active/SensorStatus.active, never a new
    # source of truth. A camera/sensor id appears in exactly one of the
    # active_*/offline_* tuples for its own kind.

    active_camera_ids: Tuple[str, ...] = field(default_factory=tuple)
    offline_camera_ids: Tuple[str, ...] = field(default_factory=tuple)

    active_sensor_ids: Tuple[str, ...] = field(default_factory=tuple)
    offline_sensor_ids: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class BuildingState:

    # The single authoritative snapshot of the Building Digital Twin's
    # CURRENT state -- fuses whatever HazardSnapshot/OccupancySnapshot/
    # BuildingObservation/FusionResult/CameraStatus/SensorStatus a
    # BuildingStateEstimator was handed this cycle into one immutable
    # object. Deliberately carries no AI/RL/decision-policy field of
    # any kind (contrast live_system.state_manager.LiveBuildingSnapshot,
    # which additionally carries ai_predictions/decision_policy/
    # recommendations) -- this type only ever answers "what is true
    # right now", never "what should happen next".
    #
    # Same recipe every other snapshot type in this codebase already
    # follows: uuid4 state_id (a stable identity independent of
    # timestamp/content, for future replay/logging), Mapping fields
    # MappingProxyType-wrapped in __post_init__, and total accessors
    # that return a usable default rather than raising, except where no
    # honest default exists (occupant_track/camera_observation/
    # smoke_detector_state/heat_detector_state below) -- mirroring
    # BuildingObservation.human_observation()'s own "returns None, the
    # one total accessor on this type that does" precedent.
    #
    # This type does not know, and must never be made to know, whether
    # its inputs came from a running Simulation, a Replay, or a Live
    # deployment -- see BuildingStateEstimator's own docstring.

    state_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: float = 0.0

    occupant_tracks: Mapping[str, FusedTrack] = field(default_factory=dict)

    zone_occupancy: OccupancySnapshot = field(default_factory=OccupancySnapshot)

    camera_observations: Mapping[str, CameraAssetState] = field(default_factory=dict)

    smoke_detector_states: Mapping[str, DetectorAssetState] = field(default_factory=dict)
    heat_detector_states: Mapping[str, DetectorAssetState] = field(default_factory=dict)

    hazard_summary: HazardSummary = field(default_factory=HazardSummary)

    building_alarm_status: DetectorState = DetectorState.NORMAL

    active_assets: ActiveAssetsSummary = field(default_factory=ActiveAssetsSummary)

    # Fire Alarm Control Panel integration (additive) -- an aggregation/
    # coordination view over the same smoke_detector_states/
    # heat_detector_states above (see facp/engine.py::SimulatedFACP,
    # which produces this value; BuildingStateEstimator only ever
    # passes it through, never computes it). None when no FACP is
    # configured for this cycle -- never fabricated. Deliberately does
    # not replace or duplicate smoke_detector_states/heat_detector_states:
    # those remain the per-device source of truth; this is the
    # panel-wide view over them (current panel state, acknowledgement/
    # silence status, and a bounded recent event window).
    facp_status: Optional[FACPSnapshot] = None

    # Building Control & Safety Systems Integration (additive) -- the
    # current CONFIRMED engineering-control state (door/exit/stair
    # pressurization/smoke exhaust/deluge), see building_control.
    # controller.BuildingControlController.snapshot(), which produces
    # this value; BuildingStateEstimator only ever passes it through,
    # never computes it, same discipline as facp_status above. None
    # when no BuildingControlController is configured for this cycle --
    # never fabricated. Only ever reflects provider-CONFIRMED state,
    # never a pending/requested one (RECOMMENDATION != COMMAND !=
    # CONFIRMED PHYSICAL ACTION).
    control_status: Optional[ControlStateSnapshot] = None

    # =====================================================

    def __post_init__(self):

        object.__setattr__(self, "occupant_tracks", MappingProxyType(dict(self.occupant_tracks)))
        object.__setattr__(self, "camera_observations", MappingProxyType(dict(self.camera_observations)))
        object.__setattr__(self, "smoke_detector_states", MappingProxyType(dict(self.smoke_detector_states)))
        object.__setattr__(self, "heat_detector_states", MappingProxyType(dict(self.heat_detector_states)))

    # =====================================================
    # Total accessors. occupant_track/camera_observation/
    # smoke_detector_state/heat_detector_state return None for an
    # unknown id -- there is no honest default FusedTrack/CameraAssetState/
    # DetectorAssetState to fabricate for an id this snapshot never saw,
    # the same reasoning BuildingObservation.human_observation() already
    # documents for itself. zone_occupancy/hazard severity below DO have
    # an honest "nothing reported" default, so they stay total.
    # =====================================================

    def occupant_track(self, track_id: str) -> Optional[FusedTrack]:

        return self.occupant_tracks.get(track_id)

    def camera_observation(self, camera_id: str) -> Optional[CameraAssetState]:

        return self.camera_observations.get(camera_id)

    def smoke_detector_state(self, sensor_id: str) -> Optional[DetectorAssetState]:

        return self.smoke_detector_states.get(sensor_id)

    def heat_detector_state(self, sensor_id: str) -> Optional[DetectorAssetState]:

        return self.heat_detector_states.get(sensor_id)

    def zone_severity(self, zone_id: str) -> HazardSeverity:

        return self.hazard_summary.severity_for(zone_id)
