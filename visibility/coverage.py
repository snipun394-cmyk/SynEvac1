from dataclasses import dataclass, field
from typing import Dict, Tuple

from visibility.engine import CameraVisibility, VisibilityEngine, zone_sample_points
from visibility.geometry import point_in_polygon
from visibility.segments import floor_opaque_segments


@dataclass(frozen=True)
class FloorCoverage:

    # The multi-camera picture for one floor -- built entirely from
    # each camera's own already-computed CameraVisibility, never
    # re-deriving wall geometry itself. Same "never permanently
    # stored, always recomputed on demand" contract as
    # CameraVisibility (see visibility/engine.py) -- nothing here is
    # serialized onto Floor/Building.

    floor_id: str

    per_camera: Dict[str, CameraVisibility] = field(default_factory=dict)

    # zone_id -> fraction of that zone's area seen by AT LEAST ONE
    # camera (the union across every camera on this floor).
    zone_combined_coverage: Dict[str, float] = field(default_factory=dict)

    # zone_id -> fraction of that zone's area seen by TWO OR MORE
    # cameras at once.
    zone_overlap_coverage: Dict[str, float] = field(default_factory=dict)

    # Zones with zero combined coverage -- no camera on this floor
    # sees any part of them.
    uncovered_zone_ids: Tuple[str, ...] = ()

    # Zones with nonzero overlap coverage -- at least some part of
    # them is doubly (or more) covered.
    overlapping_zone_ids: Tuple[str, ...] = ()

    # Area-weighted, not a plain average of per-zone fractions -- a
    # large fully-blind zone and a small fully-covered zone should not
    # count equally toward "the floor's" coverage.
    total_floor_coverage_fraction: float = 0.0


def compute_floor_coverage(cameras, building, floor, engine=None) -> FloorCoverage:

    engine = engine or VisibilityEngine()

    relevant_cameras = [
        camera for camera in cameras
        if camera.floor_id == floor.id
    ]

    # Built once and shared across every camera on this floor --
    # floor_opaque_segments() is O(zones * doors) and would otherwise
    # be paid again for every single camera (see visibility/engine.py's
    # own performance notes on compute_camera_visibility_on_floor()).
    segments = floor_opaque_segments(floor)

    per_camera = {
        camera.id: engine.compute_camera_visibility_on_floor(camera, floor, segments)
        for camera in relevant_cameras
    }

    polygons = [
        visibility.visibility_polygon
        for visibility in per_camera.values()
        if visibility.visibility_polygon
    ]

    zone_combined_coverage = {}
    zone_overlap_coverage = {}

    for zone in floor.zones:

        points = zone_sample_points(zone, engine.zone_sample_grid)

        if not points:
            zone_combined_coverage[zone.id] = 0.0
            zone_overlap_coverage[zone.id] = 0.0
            continue

        covered = 0
        overlapped = 0

        for point in points:

            hits = sum(
                1 for polygon in polygons if point_in_polygon(point, polygon)
            )

            if hits > 0:
                covered += 1

            if hits > 1:
                overlapped += 1

        zone_combined_coverage[zone.id] = covered / len(points)
        zone_overlap_coverage[zone.id] = overlapped / len(points)

    uncovered_zone_ids = tuple(
        zone_id for zone_id, fraction in zone_combined_coverage.items()
        if fraction <= 0.001
    )

    overlapping_zone_ids = tuple(
        zone_id for zone_id, fraction in zone_overlap_coverage.items()
        if fraction > 0.001
    )

    total_area = sum(zone.area for zone in floor.zones)

    covered_area = sum(
        zone.area * zone_combined_coverage.get(zone.id, 0.0)
        for zone in floor.zones
    )

    total_floor_coverage_fraction = (
        covered_area / total_area if total_area else 0.0
    )

    return FloorCoverage(
        floor_id=floor.id,
        per_camera=per_camera,
        zone_combined_coverage=zone_combined_coverage,
        zone_overlap_coverage=zone_overlap_coverage,
        uncovered_zone_ids=uncovered_zone_ids,
        overlapping_zone_ids=overlapping_zone_ids,
        total_floor_coverage_fraction=total_floor_coverage_fraction,
    )
