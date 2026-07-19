from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple

from hazard.node_state import HazardNodeState
from hazard.provider import ManualHazardProvider
from hazard.severity import HazardSeverity
from hazard.snapshot import HazardSnapshot

from occupancy.observation import OccupancyObservation
from occupancy.provider import ManualOccupancyProvider
from occupancy.snapshot import OccupancySnapshot

from perception.fusion.occupancy_estimation import EstimatedOccupancy, OccupancyEstimator
from perception.fusion.sensor_fusion import SensorFusion
from perception.models.building_observation import BuildingObservation
from perception.models.camera_observation import CameraFrameObservation
from perception.models.heat_detector_observation import HeatDetectorReading
from perception.models.smoke_detector_observation import SmokeDetectorReading
from perception.providers.ground_truth_camera_provider import GroundTruthCameraProvider
from perception.providers.ground_truth_heat_detector_provider import GroundTruthHeatDetectorProvider
from perception.providers.ground_truth_smoke_detector_provider import GroundTruthSmokeDetectorProvider


# =====================================================
# This module is Designer/debug-only glue -- it exists to drive the
# already-built Perception Layer (Phases 1-4) for the Perception Debug
# Panel and nothing else. It implements no new perception algorithm:
# every estimation/fusion decision here is delegated to the existing
# GroundTruth*Provider/OccupancyEstimator/SensorFusion classes exactly
# as built. What this module *does* add, deliberately kept separate
# from perception/, is:
#
#   1. A live Ground Truth occupancy source built from the running
#      Sandbox's actual SandboxOccupant positions (counted per zone) --
#      the "no live Ground Truth occupancy producer exists yet" gap
#      Round 1's architecture review already flagged. This is debug-
#      only: it is not a general OccupancyProvider implementation
#      offered to the rest of the app, only a value fed into
#      ManualOccupancyProvider for this panel's own use.
#   2. A small, developer-editable hazard override map (per zone
#      smoke_level/temperature), since no fire/smoke simulation is
#      wired into the Designer UI at all yet -- this lets a developer
#      manually pose a hazard condition to watch Perception react to
#      it, without a Scenario Engine. No time evolution, no physics --
#      a hand-set value only, exactly like ManualHazardProvider's own
#      "hand-authored, fixed" role in the frozen architecture.
#   3. Camera/Detector -> Zone topology resolution, reusing the exact
#      same geometry (Camera.coverage_polygon()/point-in-polygon,
#      Detector.position + Zone.contains()) GroundTruthCameraProvider/
#      GroundTruthSmokeDetectorProvider/GroundTruthHeatDetectorProvider
#      already use internally -- duplicated here (not imported from
#      perception/, which keeps those helpers private) purely to build
#      the same camera_id/detector_id -> zone_id mappings
#      OccupancyEstimator/SensorFusion need as constructor input. This
#      is topology plumbing, not a new estimation rule.
# =====================================================


def _point_in_polygon(point: Tuple[float, float], polygon_points: List[Tuple[float, float]]) -> bool:

    # Identical algorithm to
    # perception.providers.ground_truth_camera_provider._point_in_polygon
    # -- duplicated rather than imported since that helper is private
    # to its module. See the module docstring above for why this
    # duplication is debug-only plumbing, not a second implementation
    # of a perception algorithm.

    x, y = point

    if len(polygon_points) < 3:
        return False

    inside = False
    j = len(polygon_points) - 1

    for i in range(len(polygon_points)):

        xi, yi = polygon_points[i]
        xj, yj = polygon_points[j]

        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside

        j = i

    return inside


@dataclass(frozen=True)
class GroundTruthZoneReading:

    # The simulator-only comparison view for one zone -- deliberately
    # lives only in this debug module, never anywhere near
    # BuildingObservation or anything perception/ produces. This is
    # exactly the information the Perception Layer's whole design
    # exists to keep away from the Rule-Based Engine/RL -- constructing
    # it here, for a human-facing debug panel only, is the same
    # allowance Round 1's architecture already grants Designer/Sandbox
    # visualization.

    zone_id: str
    zone_name: str
    floor_name: str
    occupant_count: float
    smoke_level: Optional[float]
    temperature: Optional[float]
    hazard_score: float
    severity: HazardSeverity


@dataclass(frozen=True)
class PerceptionDebugSnapshot:

    timestamp: float

    ground_truth: Mapping[str, GroundTruthZoneReading]

    camera_observations: List[CameraFrameObservation]
    smoke_detector_readings: List[SmokeDetectorReading]
    heat_detector_readings: List[HeatDetectorReading]

    occupancy_estimates: Mapping[str, EstimatedOccupancy]

    building_observation: BuildingObservation

    # Geometric truth, straight from Camera.coverage_polygon()/
    # Zone.contains() -- every zone a device's engineering coverage
    # actually overlaps, independent of and possibly *wider than* the
    # single-zone assignment actually routed into OccupancyEstimator/
    # SensorFusion (see camera_assigned_zone_id below). Answers "is
    # this camera observing only within its configured coverage" on
    # its own, without needing to trust the routing step at all.
    camera_covered_zone_ids: Mapping[str, List[str]]
    smoke_detector_zone_id: Mapping[str, Optional[str]]
    heat_detector_zone_id: Mapping[str, Optional[str]]

    zone_names: Mapping[str, str] = field(default_factory=dict)


class PerceptionDebugRunner:

    def __init__(self):

        # zone_id -> (smoke_level, temperature), developer-injected,
        # hand-authored, never evolved over time. See module docstring
        # item 2.
        self._hazard_overrides: Dict[str, Tuple[Optional[float], Optional[float]]] = {}

    # =====================================================

    def set_zone_hazard(
        self, zone_id: str, smoke_level: Optional[float], temperature: Optional[float],
    ):

        self._hazard_overrides[zone_id] = (smoke_level, temperature)

    # =====================================================

    def clear_zone_hazard(self, zone_id: str):

        self._hazard_overrides.pop(zone_id, None)

    # =====================================================

    def clear_all_hazard_overrides(self):

        self._hazard_overrides = {}

    # =====================================================

    def run(self, building, sandbox_manager, time: float) -> PerceptionDebugSnapshot:

        floors = building.ordered_floors() if building is not None else []

        cameras = [camera for floor in floors for camera in floor.cameras]
        detectors = [detector for floor in floors for detector in floor.detectors]
        zones = [zone for floor in floors for zone in floor.zones]

        floor_name_by_id = {floor.id: floor.name for floor in floors}
        zone_names = {zone.id: zone.name or zone.id for zone in zones}

        camera_covered_zone_ids = self._resolve_camera_coverage(cameras, zones)
        camera_zone_assignments = {
            camera_id: covered[0]
            for camera_id, covered in camera_covered_zone_ids.items()
            if covered
        }

        # Detector Identity Unification -- combines legacy generic
        # Detector objects (floor.detectors, detector_type-based) with
        # the canonical SensorAsset-based SmokeDetector/HeatDetector
        # objects a Designer engineer may have placed instead
        # (floor.smoke_detectors/floor.heat_detectors). Both kinds are
        # accepted directly by GroundTruthSmokeDetectorProvider/
        # GroundTruthHeatDetectorProvider (see those classes' own
        # docstrings) -- a legacy Detector is adapted internally via
        # models.detector_migration.adapt_legacy_detector(), a
        # canonical asset is used exactly as given. This is what makes
        # a Smoke/Heat Detector placed with either Designer tool
        # actually produce a Ground Truth reading here, not just the
        # legacy tool.
        smoke_detectors = (
            [d for d in detectors if d.detector_type == "Smoke"]
            + [smoke_detector for floor in floors for smoke_detector in floor.smoke_detectors]
        )
        heat_detectors = (
            [d for d in detectors if d.detector_type == "Heat"]
            + [heat_detector for floor in floors for heat_detector in floor.heat_detectors]
        )

        smoke_detector_zone_id = {
            detector.id: self._zone_containing(detector, zones) for detector in smoke_detectors
        }
        heat_detector_zone_id = {
            detector.id: self._zone_containing(detector, zones) for detector in heat_detectors
        }

        smoke_zone_assignments = {
            detector_id: zone_id
            for detector_id, zone_id in smoke_detector_zone_id.items() if zone_id is not None
        }
        heat_zone_assignments = {
            detector_id: zone_id
            for detector_id, zone_id in heat_detector_zone_id.items() if zone_id is not None
        }

        occupancy_snapshot = self._ground_truth_occupancy(zones, sandbox_manager, time)
        hazard_snapshot = self._ground_truth_hazard(zones, time)

        camera_provider = GroundTruthCameraProvider(
            cameras=cameras, zones=zones,
            occupancy_provider=ManualOccupancyProvider(occupancy_snapshot),
        )
        camera_observations = [
            camera_provider.frame_observation_at(camera.id, time) for camera in cameras
        ]

        smoke_provider = GroundTruthSmokeDetectorProvider(
            detectors=smoke_detectors, zones=zones, hazard_provider=ManualHazardProvider(hazard_snapshot),
        )
        smoke_readings = smoke_provider.alarm_states_at(time)

        heat_provider = GroundTruthHeatDetectorProvider(
            detectors=heat_detectors, zones=zones, hazard_provider=ManualHazardProvider(hazard_snapshot),
        )
        heat_readings = heat_provider.alarm_states_at(time)

        occupancy_estimator = OccupancyEstimator(camera_zone_assignments)
        occupancy_estimates = occupancy_estimator.estimate(camera_observations)

        sensor_fusion = SensorFusion(
            smoke_detector_zone_assignments=smoke_zone_assignments,
            heat_detector_zone_assignments=heat_zone_assignments,
            camera_zone_assignments=camera_zone_assignments,
        )
        building_observation = sensor_fusion.fuse(
            timestamp=time,
            occupancy_estimates=occupancy_estimates,
            camera_observations=camera_observations,
            smoke_detector_readings=smoke_readings,
            heat_detector_readings=heat_readings,
        )

        ground_truth = {
            zone.id: self._ground_truth_reading(
                zone, floor_name_by_id.get(zone.floor_id, "-"), occupancy_snapshot, hazard_snapshot,
            )
            for zone in zones
        }

        return PerceptionDebugSnapshot(
            timestamp=time,
            ground_truth=ground_truth,
            camera_observations=camera_observations,
            smoke_detector_readings=smoke_readings,
            heat_detector_readings=heat_readings,
            occupancy_estimates=occupancy_estimates,
            building_observation=building_observation,
            camera_covered_zone_ids=camera_covered_zone_ids,
            smoke_detector_zone_id=smoke_detector_zone_id,
            heat_detector_zone_id=heat_detector_zone_id,
            zone_names=zone_names,
        )

    # =====================================================

    def _ground_truth_reading(
        self, zone, floor_name, occupancy_snapshot: OccupancySnapshot, hazard_snapshot: HazardSnapshot,
    ) -> GroundTruthZoneReading:

        occupant_count = occupancy_snapshot.observation_at(zone.id).occupant_count or 0.0
        node_state = hazard_snapshot.node_state(zone.id)

        return GroundTruthZoneReading(
            zone_id=zone.id,
            zone_name=zone.name or zone.id,
            floor_name=floor_name,
            occupant_count=occupant_count,
            smoke_level=node_state.smoke_level,
            temperature=node_state.temperature,
            hazard_score=node_state.hazard_score,
            severity=node_state.severity,
        )

    # =====================================================

    def _ground_truth_occupancy(self, zones, sandbox_manager, time: float) -> OccupancySnapshot:

        counts: Dict[str, int] = {}

        occupants = getattr(sandbox_manager, "occupants", []) if sandbox_manager is not None else []

        for occupant in occupants:

            # An occupant mid-traversal of a Door/Stair edge has no
            # current zone_id (see sandbox/occupant.py) -- not counted
            # anywhere by this debug view. A known simplification, not
            # a hidden one: see the Perception Debug Panel's own
            # reported limitations.
            if occupant.zone_id is None:
                continue

            counts[occupant.zone_id] = counts.get(occupant.zone_id, 0) + 1

        observations = {
            zone.id: OccupancyObservation(occupant_count=float(counts.get(zone.id, 0)))
            for zone in zones
        }

        return OccupancySnapshot(timestamp=time, observations=observations)

    # =====================================================

    def _ground_truth_hazard(self, zones, time: float) -> HazardSnapshot:

        node_states = {}

        for zone in zones:

            if zone.id not in self._hazard_overrides:
                continue

            smoke_level, temperature = self._hazard_overrides[zone.id]

            hazard_score = max(
                value for value in (smoke_level, 0.0) if value is not None
            )

            node_states[zone.id] = HazardNodeState(
                smoke_level=smoke_level, temperature=temperature, hazard_score=hazard_score,
            )

        return HazardSnapshot(timestamp=time, node_states=node_states)

    # =====================================================

    def _resolve_camera_coverage(self, cameras, zones) -> Dict[str, List[str]]:

        coverage = {}

        for camera in cameras:

            polygon = camera.coverage_polygon()

            coverage[camera.id] = [
                zone.id
                for zone in zones
                if zone.floor_id == camera.floor_id and _point_in_polygon(zone.center, polygon)
            ]

        return coverage

    # =====================================================

    def _zone_containing(self, detector, zones) -> Optional[str]:

        x, y = detector.position

        for zone in zones:

            if zone.floor_id == detector.floor_id and zone.contains(x, y):
                return zone.id

        return None
