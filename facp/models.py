from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple
from uuid import uuid4

from models.sensor_asset import DetectorState, HealthStatus


class PanelState(Enum):

    # The Fire Alarm Control Panel's own operator-facing displayed
    # state -- deliberately a DIFFERENT type from DetectorState below,
    # even though NORMAL/ALARM/FAULT overlap in name, for the same
    # "structurally distinct concepts get structurally distinct types"
    # reason HazardSeverity/PerceptionSeverity stay separate enums
    # elsewhere in this codebase: ALARM_ACKNOWLEDGED/ALARM_SILENCED
    # have no meaning for a single detector, only for the panel as a
    # whole. A panel showing ALARM_ACKNOWLEDGED or ALARM_SILENCED does
    # NOT mean the underlying fire condition has cleared -- see
    # FACPSnapshot.active_alarm_source_ids below, which stays populated
    # throughout both. FAULT deliberately has no acknowledged/silenced
    # variant in this version -- Phase 2's own event list (Detector
    # Fault/Restore, Panel Acknowledged, Alarm Silenced) is worded
    # entirely around ALARM, so extending ack/silence to FAULT is left
    # for a future phase rather than guessed at now.
    #
    # No DISABLED/ISOLATED member: no per-detector "isolate" capability
    # exists anywhere in this codebase (confirmed by investigation), and
    # SensorAsset.active already gives the identical effect -- an
    # inactive detector's own compute_state() already returns NORMAL,
    # never ALARM/FAULT (see SmokeDetector.compute_state()'s own
    # docstring) -- so it never reaches the panel as an alarm/fault
    # source in the first place. Adding a redundant panel-level
    # DISABLED state would duplicate a mechanism that already exists
    # one layer down, which Phase 2's own "if appropriate" qualifier
    # argues against.

    NORMAL = auto()
    FAULT = auto()
    ALARM = auto()
    ALARM_ACKNOWLEDGED = auto()
    ALARM_SILENCED = auto()


class PanelEventType(Enum):

    DETECTOR_ALARM = auto()
    DETECTOR_FAULT = auto()
    DETECTOR_RESTORE = auto()
    MANUAL_ALARM = auto()
    PANEL_ACKNOWLEDGED = auto()
    ALARM_SILENCED = auto()
    SYSTEM_RESET = auto()


class InvalidPanelOperation(Exception):

    # Raised by SimulatedFACP.acknowledge()/silence() when called
    # outside the state that operation is meaningful in (e.g.
    # acknowledging a panel that is not currently in ALARM) -- same
    # "an operation not legal from the current state raises immediately
    # rather than silently no-op-ing" convention
    # live_system.incident_manager.InvalidIncidentTransition already
    # establishes for a state machine one level up. reset() never
    # raises this -- Reset is always a legal operator action on a real
    # panel, only its RESULT depends on current conditions (see
    # SimulatedFACP.reset()'s own docstring).

    pass


@dataclass(frozen=True)
class PanelEvent:

    # One append-only entry in the panel's own event history (Phase 4).
    # source_asset_id/source_asset_type/floor_id/zone_ids are None/empty
    # for panel-level events (PANEL_ACKNOWLEDGED, ALARM_SILENCED,
    # SYSTEM_RESET) that originate from an operator action, not from a
    # specific device -- never fabricated to look device-sourced.
    # panel_state_after is this event's one authoritative link back to
    # the state machine: exactly what PanelState the panel held
    # immediately after this event was recorded, so a reader of the
    # event history alone (without re-running the state machine) can
    # always tell what the panel showed at any point in its history.

    event_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: float = 0.0

    event_type: PanelEventType = PanelEventType.SYSTEM_RESET

    source_asset_id: Optional[str] = None
    source_asset_type: Optional[str] = None

    floor_id: Optional[str] = None
    zone_ids: Tuple[str, ...] = field(default_factory=tuple)

    panel_state_after: PanelState = PanelState.NORMAL


@dataclass(frozen=True)
class DetectorConditionReport:

    # SimulatedFACP's one and only input shape -- an already-computed
    # DetectorState (see models/sensor_asset.py) for one detector this
    # cycle, plus the identity/location metadata a PanelEvent needs to
    # report it honestly. SimulatedFACP never sees a smoke_level,
    # temperature, HazardSnapshot, or SmokeDetector/HeatDetector object
    # directly -- only this. That is what "must NOT independently
    # recompute smoke or temperature physics" (Phase 3) means in
    # practice: there is no hazard-reading field on this type for the
    # FACP to compute anything from even if it wanted to.

    asset_id: str
    asset_type: str
    state: DetectorState

    floor_id: Optional[str] = None
    zone_ids: Tuple[str, ...] = field(default_factory=tuple)

    # =====================================================

    @classmethod
    def from_status_and_reading(cls, status, reading=None) -> "DetectorConditionReport":

        # The one place "health fault outranks an alarm reading, an
        # alarm reading outranks a clear reading" is decided -- the
        # exact same ALARM > FAULT > NORMAL-shaped priority
        # SensorManager.aggregate_alarm_state() already applies across
        # multiple sensors, applied here to ONE sensor's own
        # (SensorStatus, Optional[reading]) pair so every future caller
        # (Designer glue today; a SimulationRuntime tick, Replay, or a
        # Live adapter later) builds a DetectorConditionReport the same
        # way instead of re-deriving this priority itself. `reading` is
        # whatever this sensor's own SmokeDetectorReading/
        # HeatDetectorReading was this cycle (or None -- "no reading
        # available", e.g. a detector Perception's own Ground Truth
        # provider already skipped because it was FAULT -- see that
        # provider's own change; this method still reports FAULT
        # correctly in that case purely from status.health_status,
        # never needing the reading Perception withheld).

        if status.health_status != HealthStatus.OK:
            state = DetectorState.FAULT
        elif reading is not None and reading.alarm_active:
            state = DetectorState.ALARM
        else:
            state = DetectorState.NORMAL

        return cls(
            asset_id=status.sensor_id,
            asset_type=status.sensor_type,
            state=state,
            floor_id=status.floor_id,
            zone_ids=status.zone_ids,
        )


@dataclass(frozen=True)
class FACPSnapshot:

    # The panel's current status, read-only -- what BuildingState
    # exposes additively as `facp_status` (Phase 6). Deliberately
    # separate from BuildingState.smoke_detector_states/
    # heat_detector_states, which are untouched by this type's
    # existence -- this is a building-wide AGGREGATION/COORDINATION
    # view over those, never a replacement for them.

    panel_id: str
    timestamp: float

    panel_state: PanelState = PanelState.NORMAL

    # Always the CURRENT raw set of alarming/faulted source ids,
    # regardless of panel_state -- stays populated through
    # ALARM_ACKNOWLEDGED/ALARM_SILENCED exactly so acknowledging or
    # silencing can never be read as "the fire condition disappeared"
    # (Phase 5's own explicit warning).
    active_alarm_source_ids: Tuple[str, ...] = field(default_factory=tuple)
    active_fault_source_ids: Tuple[str, ...] = field(default_factory=tuple)

    acknowledged: bool = False
    silenced: bool = False

    # A bounded recent window -- the full, unbounded history is always
    # available through SimulatedFACP.event_log's own query methods
    # (Phase 4); this field exists only so a consumer of a single
    # FACPSnapshot (e.g. BuildingState) has *something* to show without
    # needing a separate reference to the engine that produced it.
    recent_events: Tuple[PanelEvent, ...] = field(default_factory=tuple)
