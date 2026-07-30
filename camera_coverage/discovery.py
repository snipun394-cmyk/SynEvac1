import math
from typing import Dict, Mapping, Optional, Sequence, Tuple

from visibility.geometry import point_in_polygon, segment_intersection

from camera_coverage.models import AssetCoverage, CameraCoverage, CameraCoverageSnapshot, CoverageState


# =====================================================
# Camera Coverage Intelligence & Observable Asset Mapping milestone --
# automatic Camera -> Observable Asset coverage discovery. Geometry is
# the source of truth (this milestone's own explicit instruction): no
# manual Camera-to-asset assignment is read or required anywhere in this
# module (the prior Observable Asset Perception Framework milestone
# already found models.camera.Camera.zone_ids/EngineeringAsset.zone_ids
# to be cosmetic/non-authoritative for the real live pipeline -- see
# camera_calibration.asset_lookup.covered_asset_ids()'s own docstring;
# this module deliberately does not repeat that mistake).
#
# Deliberately does NOT touch camera_calibration.camera_model,
# camera_calibration.projection, or camera_calibration.geometry --
# "do not redesign calibration mathematics" (this milestone's own
# instruction). Every value this module reads from a CalibrationProfile
# (extrinsics.position, extrinsics.yaw_degrees, intrinsics.image_width,
# intrinsics.focal_length_x) is an EXISTING public field; the horizontal-
# FOV-from-focal-length derivation below is simply the algebraic inverse
# of CameraIntrinsics.from_horizontal_fov() (camera_calibration/
# camera_model.py), computed locally here rather than added as a new
# method on that frozen dataclass.
#
# The coverage SECTOR itself reuses models.camera.Camera.coverage_polygon()'s
# own exact math (position, facing angle, half-FOV, max range, "0 degrees
# points along +x, increasing clockwise") -- not a new geometric model,
# just that same fan/wedge shape built from a camera's CALIBRATED pose
# (CalibrationProfile.extrinsics) rather than its raw Designer-authored
# fields, so a camera whose calibration has been refined independently of
# its logical Camera.position/rotation is honestly represented. Camera.
# max_range is reused as-is for the sector's radius -- the same "one
# distance an engineering camera asset needs" field
# perception.providers.ground_truth_camera_provider.GroundTruthCameraProvider
# already reuses for its own effective detection range (see
# models/camera.py's own docstring).
# =====================================================


DEFAULT_SEGMENTS = 24

NO_CALIBRATION = "camera has no calibration profile"
NO_USABLE_SECTOR = "camera calibration cannot yield a usable coverage sector (no focal length / no detection range)"
NO_OBSERVABLE_REGION = "asset has no observable region authored on this floor"
OVERLAPS_SECTOR = "asset's observable region overlaps the camera's calibrated coverage sector"
NO_OVERLAP = "asset's observable region does not overlap the camera's calibrated coverage sector"


def _derive_horizontal_fov_degrees(intrinsics) -> Optional[float]:

    # The algebraic inverse of CameraIntrinsics.from_horizontal_fov()'s
    # own fx = (width/2) / tan(fov/2) -- fov = 2 * atan((width/2) / fx).
    # None (never a fabricated angle) whenever focal_length_x is
    # non-positive -- geometrically meaningless, mirroring
    # from_horizontal_fov()'s own "no nonsense CameraIntrinsics" guard.

    if intrinsics.focal_length_x <= 0:
        return None

    half_fov_radians = math.atan((intrinsics.image_width / 2.0) / intrinsics.focal_length_x)

    return math.degrees(half_fov_radians) * 2.0


def _sector_polygon(
    center_x: float, center_y: float, yaw_degrees: float, horizontal_fov_degrees: float,
    max_range: float, segments: int = DEFAULT_SEGMENTS,
) -> Tuple[Tuple[float, float], ...]:

    half_fov = horizontal_fov_degrees / 2.0

    points = [(center_x, center_y)]

    for i in range(segments + 1):

        angle_degrees = yaw_degrees - half_fov + (horizontal_fov_degrees * i / segments)
        angle_radians = math.radians(angle_degrees)

        points.append((
            center_x + max_range * math.cos(angle_radians),
            center_y + max_range * math.sin(angle_radians),
        ))

    return tuple(points)


def _rectangle_corners(region) -> Tuple[Tuple[float, float], ...]:

    # Duck-typed against `.center_x`/`.center_y`/`.width`/`.depth` --
    # models.staircase.StairObservableRegion's own real shape, the only
    # observable-region type this codebase has today. A future non-
    # rectangular observable-region type would need its own corner/
    # boundary adapter here (a small, isolated addition -- exactly the
    # per-kind-adapter pattern camera_calibration.stair_lookup.
    # build_stairs_by_floor() already establishes for asset discovery),
    # never a change to the framework types in observable_assets/
    # camera_calibration.asset_lookup themselves.

    half_width = region.width / 2.0
    half_depth = region.depth / 2.0

    return (
        (region.center_x - half_width, region.center_y - half_depth),
        (region.center_x + half_width, region.center_y - half_depth),
        (region.center_x + half_width, region.center_y + half_depth),
        (region.center_x - half_width, region.center_y + half_depth),
    )


def _polygon_edges(points: Sequence[Tuple[float, float]]):

    return [(points[i], points[(i + 1) % len(points)]) for i in range(len(points))]


def _edges_intersect(edges_a, edges_b) -> bool:

    for p1, p2 in edges_a:

        for p3, p4 in edges_b:

            if segment_intersection(p1, p2, p3, p4) is not None:
                return True

    return False


def _classify_overlap(rect_corners, region, sector_polygon) -> CoverageState:

    # Phase 4's precise determination rule, in priority order:
    #
    #   1. All four of the asset region's corners fall inside the
    #      camera's coverage sector -> FULLY_VISIBLE (the whole footprint
    #      is within the calibrated FOV wedge and detection range --
    #      point_in_polygon() on a range-bounded sector fan inherently
    #      enforces both the angular AND the distance limit at once).
    #   2. Some, but not all, corners fall inside -> PARTIALLY_VISIBLE.
    #   3. No corner falls inside, but the sector nonetheless reaches
    #      into the rectangle (a narrow sector wedge can pass through a
    #      large asset region without ever containing one of its four
    #      corners) -- checked two ways: any sector-polygon vertex lies
    #      inside the rectangle, or any sector edge crosses any rectangle
    #      edge -> PARTIALLY_VISIBLE either way.
    #   4. None of the above -> NOT_VISIBLE, a provably empty
    #      intersection.

    corners_inside = sum(1 for corner in rect_corners if point_in_polygon(corner, sector_polygon))

    if corners_inside == len(rect_corners):
        return CoverageState.FULLY_VISIBLE

    if corners_inside > 0:
        return CoverageState.PARTIALLY_VISIBLE

    if any(region.contains(px, py) for px, py in sector_polygon):
        return CoverageState.PARTIALLY_VISIBLE

    if _edges_intersect(_polygon_edges(rect_corners), _polygon_edges(sector_polygon)):
        return CoverageState.PARTIALLY_VISIBLE

    return CoverageState.NOT_VISIBLE


def compute_camera_coverage(
    camera,
    calibration: Optional[object],
    candidates: Sequence[Tuple[str, object]],
    timestamp: float = 0.0,
    segments: int = DEFAULT_SEGMENTS,
) -> CameraCoverage:

    # camera: a Camera-shaped object exposing `.id`, `.floor_id`,
    # `.max_range` (models.camera.Camera's own real shape).
    # calibration: this camera's camera_calibration.camera_model.
    # CalibrationProfile, or None -- "missing calibration"/"legacy
    # project" cameras honestly produce UNKNOWN for every candidate,
    # never a guessed sector.
    # candidates: this camera's floor's own (asset_type, asset) pairs,
    # exactly the shape camera_calibration.asset_lookup.
    # build_assets_by_floor() produces (never re-derived from a Building
    # here).

    if calibration is None:

        return CameraCoverage(
            camera_id=camera.id, floor_id=getattr(camera, "floor_id", None) or None, timestamp=timestamp,
            sector_polygon=None,
            assets={
                asset.id: AssetCoverage(
                    asset_id=asset.id, asset_type=asset_type, state=CoverageState.UNKNOWN, provenance=NO_CALIBRATION,
                )
                for asset_type, asset in candidates
            },
        )

    horizontal_fov_degrees = _derive_horizontal_fov_degrees(calibration.intrinsics)
    max_range = getattr(camera, "max_range", None)

    if horizontal_fov_degrees is None or not max_range or max_range <= 0:

        return CameraCoverage(
            camera_id=camera.id, floor_id=calibration.floor_id, timestamp=timestamp, sector_polygon=None,
            assets={
                asset.id: AssetCoverage(
                    asset_id=asset.id, asset_type=asset_type, state=CoverageState.UNKNOWN, provenance=NO_USABLE_SECTOR,
                )
                for asset_type, asset in candidates
            },
        )

    center_x, center_y = calibration.extrinsics.position

    sector_polygon = _sector_polygon(
        center_x, center_y, calibration.extrinsics.yaw_degrees, horizontal_fov_degrees, max_range, segments,
    )

    assets: Dict[str, AssetCoverage] = {}

    for asset_type, asset in candidates:

        region = asset.observable_region_for_floor(calibration.floor_id)

        if region is None:

            assets[asset.id] = AssetCoverage(
                asset_id=asset.id, asset_type=asset_type, state=CoverageState.UNKNOWN, provenance=NO_OBSERVABLE_REGION,
            )
            continue

        rect_corners = _rectangle_corners(region)
        state = _classify_overlap(rect_corners, region, sector_polygon)
        determined = state in (CoverageState.FULLY_VISIBLE, CoverageState.PARTIALLY_VISIBLE)

        assets[asset.id] = AssetCoverage(
            asset_id=asset.id, asset_type=asset_type, state=state,
            region_polygon=rect_corners if determined else None,
            provenance=OVERLAPS_SECTOR if determined else NO_OVERLAP,
        )

    return CameraCoverage(
        camera_id=camera.id, floor_id=calibration.floor_id, timestamp=timestamp,
        sector_polygon=sector_polygon, assets=assets,
    )


def compute_camera_coverage_snapshot(
    cameras: Sequence[object],
    calibrations: Mapping[str, object],
    assets_by_floor: Mapping[str, Sequence[Tuple[str, object]]],
    timestamp: float = 0.0,
    segments: int = DEFAULT_SEGMENTS,
) -> CameraCoverageSnapshot:

    # cameras: every Camera-shaped object to compute coverage for (an
    # uncalibrated one is included too -- it simply reports UNKNOWN for
    # its floor's candidates, never silently dropped).
    # calibrations: camera_id -> CalibrationProfile, exactly the shape
    # camera_calibration.calibration.CalibrationRegistry.all_profiles()
    # (keyed) or camera_calibration.projection.WorldProjector's own
    # constructor parameter already uses.
    # assets_by_floor: floor_id -> (asset_type, asset) pairs, exactly
    # camera_calibration.asset_lookup.build_assets_by_floor()'s own
    # output -- never re-derived here.

    by_camera = {}

    for camera in cameras:

        calibration = calibrations.get(camera.id)
        floor_id = calibration.floor_id if calibration is not None else getattr(camera, "floor_id", None)
        candidates = assets_by_floor.get(floor_id, ()) if floor_id else ()

        by_camera[camera.id] = compute_camera_coverage(camera, calibration, candidates, timestamp, segments)

    return CameraCoverageSnapshot(timestamp=timestamp, by_camera=by_camera)
