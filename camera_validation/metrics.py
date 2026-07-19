from dataclasses import dataclass
from typing import Tuple


# Illustrative, documented scoring weights -- same "engineering
# judgment, not a validated life-safety figure" honesty this codebase
# already discloses everywhere a scoring constant appears (Hazard
# Severity's score cutoffs, StairCapacityModel's capacity formula,
# MultiCameraConfidenceBoostStrategy's boost-per-camera). A camera that
# covers more of its own floor's area scores higher; a camera that
# additionally watches a Door/Exit/Stair scores higher still, since
# observing a critical engineering asset is valuable even when it
# contributes little raw area.
COVERAGE_WEIGHT = 0.7
ASSET_WEIGHT = 0.3

# Each visible Door/Exit/Stair contributes this many asset-score
# points, capped at 100 -- five or more critical assets already
# saturate the asset component of the score.
ASSET_SCORE_PER_CRITICAL_ASSET = 20.0


@dataclass(frozen=True)
class CameraPlacementMetrics:

    # Phase 2's required per-camera engineering metrics -- mounting
    # height/rotation/horizontal FOV/max range are read straight off
    # the Camera Asset (models/camera.py); everything else is read
    # straight off that camera's own CameraVisibility (visibility/
    # engine.py), never re-derived. placement_score is the one new
    # number this module actually computes.

    camera_id: str

    mount_height: float
    rotation: float
    horizontal_fov: float
    max_range: float

    visible_door_ids: Tuple[str, ...]
    visible_exit_ids: Tuple[str, ...]
    visible_stair_ids: Tuple[str, ...]

    # Area-weighted percentage (0-100) of this camera's own floor that
    # its visibility polygon actually covers -- not just whether it
    # touches a zone at all.
    zone_coverage_percentage: float

    # Zones this camera sees none of at all -- CameraVisibility.
    # hidden_zone_ids, reused as-is.
    blind_zone_ids: Tuple[str, ...]

    # 0-100, COVERAGE_WEIGHT/ASSET_WEIGHT-blended -- a single number
    # summarizing "how good is this one camera's placement", not a
    # validated life-safety metric.
    placement_score: float


def compute_camera_placement_metrics(camera, floor, camera_visibility) -> CameraPlacementMetrics:

    total_area = sum(zone.area for zone in floor.zones)

    covered_area = sum(
        zone.area * camera_visibility.zone_coverage.get(zone.id, 0.0)
        for zone in floor.zones
    )

    coverage_percentage = (covered_area / total_area * 100.0) if total_area else 0.0

    asset_count = (
        len(camera_visibility.visible_door_ids)
        + len(camera_visibility.visible_exit_ids)
        + len(camera_visibility.visible_stair_ids)
    )
    asset_score = min(100.0, ASSET_SCORE_PER_CRITICAL_ASSET * asset_count)

    placement_score = COVERAGE_WEIGHT * coverage_percentage + ASSET_WEIGHT * asset_score

    return CameraPlacementMetrics(
        camera_id=camera.id,
        mount_height=camera.mount_height,
        rotation=camera.rotation,
        horizontal_fov=camera.horizontal_fov,
        max_range=camera.max_range,
        visible_door_ids=camera_visibility.visible_door_ids,
        visible_exit_ids=camera_visibility.visible_exit_ids,
        visible_stair_ids=camera_visibility.visible_stair_ids,
        zone_coverage_percentage=coverage_percentage,
        blind_zone_ids=camera_visibility.hidden_zone_ids,
        placement_score=placement_score,
    )
