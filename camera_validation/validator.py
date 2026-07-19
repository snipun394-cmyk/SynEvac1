from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from visibility.coverage import compute_floor_coverage
from visibility.engine import VisibilityEngine

from camera_validation.metrics import CameraPlacementMetrics, compute_camera_placement_metrics
from camera_validation.network import NetworkAnalysis, compute_network_analysis
from camera_validation.recommendations import (
    Recommendation,
    generate_camera_recommendations,
    generate_network_recommendations,
)


@dataclass(frozen=True)
class FloorPlacementReport:

    # One floor's complete Camera Calibration & Placement Validation
    # result -- Phase 2's per-camera metrics, Phase 3's network
    # analysis, and Phase 4's recommendations, all for this floor only
    # (Visibility Engine never lets a camera see across floors, so
    # neither does this report).

    floor_id: str

    camera_metrics: Dict[str, CameraPlacementMetrics] = field(default_factory=dict)
    network_analysis: Optional[NetworkAnalysis] = None
    recommendations: Tuple[Recommendation, ...] = ()


@dataclass(frozen=True)
class BuildingPlacementReport:

    # Every floor's FloorPlacementReport, plus one Building-wide
    # summary score -- Phase 5's "Display network score" is this
    # field; "Display placement score" is per_floor[...].camera_
    # metrics[camera_id].placement_score.

    per_floor: Dict[str, FloorPlacementReport] = field(default_factory=dict)

    # Area-weighted average of every floor's own network_score -- a
    # large, poorly-covered floor should drag the overall score down
    # more than a small one, the same area-weighting
    # FloorCoverage.total_floor_coverage_fraction already uses.
    overall_network_score: float = 0.0

    def floor_report(self, floor_id: str) -> Optional[FloorPlacementReport]:

        return self.per_floor.get(floor_id)


def validate_floor(cameras, building, floor, engine: Optional[VisibilityEngine] = None) -> FloorPlacementReport:

    engine = engine or VisibilityEngine()

    floor_cameras = [camera for camera in cameras if camera.floor_id == floor.id]

    floor_coverage = compute_floor_coverage(floor_cameras, building, floor, engine=engine)

    camera_by_id = {camera.id: camera for camera in floor_cameras}

    camera_metrics = {}
    recommendations = []

    for camera in floor_cameras:

        camera_visibility = floor_coverage.per_camera[camera.id]

        metrics = compute_camera_placement_metrics(camera, floor, camera_visibility)
        camera_metrics[camera.id] = metrics

        recommendations.extend(
            generate_camera_recommendations(camera, floor, camera_visibility, metrics)
        )

    network_analysis = compute_network_analysis(floor, floor_coverage)

    recommendations.extend(
        generate_network_recommendations(floor, network_analysis, camera_by_id=camera_by_id)
    )

    return FloorPlacementReport(
        floor_id=floor.id,
        camera_metrics=camera_metrics,
        network_analysis=network_analysis,
        recommendations=tuple(recommendations),
    )


def validate_building(cameras, building, engine: Optional[VisibilityEngine] = None) -> BuildingPlacementReport:

    engine = engine or VisibilityEngine()

    floors = building.ordered_floors() if building is not None else []

    per_floor = {
        floor.id: validate_floor(cameras, building, floor, engine=engine)
        for floor in floors
    }

    total_area = 0.0
    weighted_score_sum = 0.0

    for floor in floors:

        floor_area = sum(zone.area for zone in floor.zones)

        total_area += floor_area
        weighted_score_sum += floor_area * per_floor[floor.id].network_analysis.network_score

    overall_network_score = (weighted_score_sum / total_area) if total_area else 0.0

    return BuildingPlacementReport(per_floor=per_floor, overall_network_score=overall_network_score)
