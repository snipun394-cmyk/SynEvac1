from typing import Any, Dict, Optional, Tuple

from hazard.snapshot import HazardSnapshot

from occupancy.snapshot import OccupancySnapshot

from models.building import Building
from models.sensor_asset import DetectorState

from perception.models.human_observation import HumanClassification
from perception.models.smoke_detector_observation import SmokeDetectorReading
from perception.models.heat_detector_observation import HeatDetectorReading

from camera_manager.manager import CameraManager
from sensor_manager.manager import SensorManager

from multi_camera_fusion.track import FusedTrack, FusionResult, TrackHistory

from facp.engine import SimulatedFACP
from facp.models import DetectorConditionReport

from building_state.estimator import BuildingStateEstimator
from building_state.models import BuildingState

from ai_features.building_state_extractor import extract_canonical_features


# =====================================================
# Simulation-side parity extractor -- Phase 4's own explicit preference:
# "If simulation can already construct BuildingState, route training-
# feature generation through that path where practical... strongly
# preferred over creating a second independent implementation." This
# module builds a REAL BuildingState from simulation-side facts using
# the exact same production classes designer/building_state_debug_
# runner.py already composes for the Designer (CameraManager,
# SensorManager, SimulatedFACP, BuildingStateEstimator) -- no camera/
# sensor/FACP aggregation logic is reimplemented here -- and then calls
# ai_features.building_state_extractor.extract_canonical_features(), the
# SAME function a genuine live deployment calls. This is what makes
# "same feature names and semantics from simulation data and
# BuildingState" true by construction rather than by convention.
#
# TEMPORAL FRAMING (Phase 7): every one of the four currently-trained
# models is a single-snapshot, pre-outcome predictor -- none of them
# operate on a per-tick trajectory (see docs/architecture/ai_live_
# feature_parity.md's own Phase 7 findings). The lowest-risk, most
# faithful live-compatible reinterpretation is therefore "capture one
# snapshot at T = alarm activation, predict the eventual outcome" --
# the same shape the existing models already use, not a new temporal-
# model redesign. build_building_state_at_alarm_activation() below
# builds exactly that snapshot: the full starting population (nobody
# has evacuated yet, by definition, at the instant alarm activates),
# every device's own configured active/offline status, and exactly the
# detector(s) covering the ignition zone (if known) reporting ALARM --
# an HONEST consequence of where the fire started (a real FACP would
# show precisely this), never a fabricated hazard reading and never the
# ignition zone identity itself exposed as a feature.
#
# LABELS remain simulation Ground Truth (Phase 6's own explicit rule:
# "TRAINING LABELS may come from simulation outcomes" -- only INPUT
# features must be live-compatible) -- this module produces feature
# rows only, never a target value; callers pair its output with
# whatever label (e.g. total_evacuation_time) their own experiment needs.
# =====================================================


def build_building_state_at_alarm_activation(
    building: Building,
    *,
    total_occupants: int,
    ignition_zone_id: Optional[str] = None,
    timestamp: float = 0.0,
) -> BuildingState:

    camera_manager = CameraManager()
    camera_manager.discover_cameras(building)

    sensor_manager = SensorManager()
    sensor_manager.discover_sensors(building)

    fusion_result = FusionResult(
        timestamp=timestamp, tracks=_synthetic_occupant_tracks(total_occupants, timestamp),
    )

    smoke_statuses, smoke_readings, heat_statuses, heat_readings, detector_conditions = (
        _detector_inputs(sensor_manager, ignition_zone_id, timestamp)
    )

    facp = SimulatedFACP(panel_id="FACP-TRAINING")
    facp.evaluate(detector_conditions, timestamp)

    return BuildingStateEstimator().estimate(
        timestamp,
        hazard_snapshot=HazardSnapshot(),
        occupancy_snapshot=OccupancySnapshot(),
        camera_statuses=camera_manager.all_statuses(),
        fusion_result=fusion_result,
        smoke_detector_statuses=smoke_statuses,
        smoke_detector_readings=smoke_readings,
        heat_detector_statuses=heat_statuses,
        heat_detector_readings=heat_readings,
        facp_snapshot=facp.current_snapshot(timestamp),
    )


# =====================================================


def extract_canonical_training_row(
    building: Building,
    *,
    total_occupants: int,
    ignition_zone_id: Optional[str] = None,
    timestamp: float = 0.0,
) -> Dict[str, Any]:

    # The one entry point Phase 6/8's dataset generation and Phase 9's
    # parity test both call -- builds the simulation-side BuildingState
    # above, then extracts it through the exact same canonical extractor
    # a live deployment would use.

    state = build_building_state_at_alarm_activation(
        building,
        total_occupants=total_occupants,
        ignition_zone_id=ignition_zone_id,
        timestamp=timestamp,
    )

    return extract_canonical_features(state)


# =====================================================


def _synthetic_occupant_tracks(total_occupants: int, timestamp: float) -> Tuple[FusedTrack, ...]:

    # A track per starting occupant -- count only, never a per-person
    # attribute (no classification/human_state claimed; UNKNOWN/None
    # match exactly what a real detector with no working classifier
    # would honestly report). Mirrors designer/building_state_debug_
    # runner.py's own _ground_truth_fusion_result() one level further
    # abstracted (that method reads live SandboxOccupant positions; this
    # one only needs a total count, since the canonical schema itself
    # never uses per-track zone/floor/classification, only the count and
    # mean confidence -- see ai_features.building_state_extractor).

    return tuple(
        FusedTrack(
            track_id=f"synthetic-occupant-{index}",
            source_camera_ids=(),
            floor_id=None,
            zone_id=None,
            classification=HumanClassification.UNKNOWN,
            human_state=None,
            confidence=1.0,
            timestamp=timestamp,
            history=TrackHistory(
                track_id=f"synthetic-occupant-{index}",
                previous_camera_id=None,
                current_camera_id=None,
                last_observation_time=timestamp,
                current_zone_id=None,
                current_floor_id=None,
            ),
        )
        for index in range(total_occupants)
    )


# =====================================================


def _detector_inputs(sensor_manager, ignition_zone_id: Optional[str], timestamp: float):

    smoke_statuses = []
    smoke_readings = []
    heat_statuses = []
    heat_readings = []
    detector_conditions = {}

    for sensor in sensor_manager.all_sensors():

        status = sensor_manager.sensor_status(sensor.id)

        # Honest consequence of where the fire started, never the
        # ignition zone identity itself exposed as a feature -- only a
        # covering, ACTIVE detector alarms; an inactive/offline detector
        # cannot report anything, alarmed or otherwise.
        alarm_active = (
            status.active
            and ignition_zone_id is not None
            and ignition_zone_id in sensor.zone_ids
        )

        if sensor.object_type == "SmokeDetector":

            reading = SmokeDetectorReading(detector_id=sensor.id, timestamp=timestamp, alarm_active=alarm_active)
            smoke_statuses.append(status)
            smoke_readings.append(reading)

        elif sensor.object_type == "HeatDetector":

            reading = HeatDetectorReading(detector_id=sensor.id, timestamp=timestamp, alarm_active=alarm_active)
            heat_statuses.append(status)
            heat_readings.append(reading)

        else:

            continue

        detector_conditions[sensor.id] = DetectorConditionReport(
            asset_id=sensor.id,
            asset_type=sensor.object_type,
            state=DetectorState.ALARM if alarm_active else DetectorState.NORMAL,
            floor_id=sensor.floor_id,
            zone_ids=sensor.zone_ids,
        )

    return (
        tuple(smoke_statuses), tuple(smoke_readings),
        tuple(heat_statuses), tuple(heat_readings),
        detector_conditions,
    )
