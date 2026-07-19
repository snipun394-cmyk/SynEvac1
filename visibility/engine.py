import math

from dataclasses import dataclass, field
from typing import Dict, Tuple

from visibility.geometry import point_in_polygon, segment_intersection
from visibility.segments import floor_opaque_segments


# Rays cast across the camera's FOV wedge to build its occlusion-aware
# visibility polygon. A resolution/performance knob, not a validated
# sensor value -- higher values trace finer-grained wall corners at
# proportionally higher cost (cost is linear in ray_count *
# len(segments) per camera). 120 rays across a typical <=180 degree
# FOV is better than 1-degree resolution and stays well under a
# millisecond for a normal floor's segment count (see this module's
# own performance notes below).
DEFAULT_RAY_COUNT = 120

# NxN sample grid used to estimate what *fraction* of a Zone's area
# is actually visible (as opposed to whether any of it is). Fixed
# per zone regardless of the zone's physical size, so cost stays
# bounded (grid**2 point-in-polygon tests per zone per camera)
# independent of how large a building's zones are.
DEFAULT_ZONE_SAMPLE_GRID = 6

# A zone is classified "visible" at or above this coverage fraction,
# and "hidden" at or below it -- anything strictly between is
# "partially visible". Not exactly 1.0/0.0 because the sample grid
# is a discrete approximation of a continuous area; giving a small
# tolerance at both ends avoids a fully (or barely not-quite-fully)
# covered zone being mislabeled "partial" purely from grid rounding.
FULLY_VISIBLE_THRESHOLD = 0.999
HIDDEN_THRESHOLD = 0.001


@dataclass(frozen=True)
class CameraVisibility:

    # One camera's complete, occlusion-aware view of its own floor at
    # the moment compute_camera_visibility() was called -- never
    # stored on Camera itself (see this module's docstring-level
    # comment below on why), always recomputed on demand.

    camera_id: str

    # The actual occlusion-aware coverage shape: camera position
    # followed by one point per cast ray (either the wall it hit, or
    # max_range along that ray if nothing blocked it). Strictly a
    # subset of Camera.coverage_polygon()'s naive full-sector
    # polygon -- this is that same sector with walls/obstacles
    # carved out of it.
    visibility_polygon: Tuple[Tuple[float, float], ...] = ()

    # zone_id -> fraction of that zone's area visible to this camera,
    # in [0.0, 1.0].
    zone_coverage: Dict[str, float] = field(default_factory=dict)

    visible_zone_ids: Tuple[str, ...] = ()
    partially_visible_zone_ids: Tuple[str, ...] = ()
    hidden_zone_ids: Tuple[str, ...] = ()

    visible_door_ids: Tuple[str, ...] = ()
    visible_exit_ids: Tuple[str, ...] = ()
    visible_stair_ids: Tuple[str, ...] = ()

    max_visible_distance: float = 0.0


def zone_sample_points(zone, grid):

    # A grid of `grid` x `grid` points spread evenly inside the
    # zone's rectangle (never on the boundary itself, which could
    # sit exactly on a wall segment and make point-in-polygon
    # ambiguous). Exposed as a module-level function -- not private
    # to VisibilityEngine -- because visibility/coverage.py needs the
    # exact same points to test each zone against multiple cameras'
    # polygons without duplicating this grid math.

    if grid <= 0:
        return []

    points = []

    for i in range(grid):
        for j in range(grid):

            sx = zone.x + zone.width * (i + 0.5) / grid
            sy = zone.y + zone.height * (j + 0.5) / grid

            points.append((sx, sy))

    return points


def _stair_point_on_floor(stair, floor_id):

    # Staircase spans two floors (see models/staircase.py) and has no
    # single "position" -- only whichever endpoint sits on the
    # camera's own floor is a meaningful visibility target.

    if stair.from_floor_id == floor_id:
        return stair.from_position

    if stair.to_floor_id == floor_id:
        return stair.to_position

    return None


class VisibilityEngine:

    # The one geometric seam Simulation, Replay, and future Live CCTV
    # all share (see docs referenced from models/engineering_asset.py
    # DeviceMode) -- every method here answers a pure "given this
    # camera's geometry and this building's walls, what can/can't be
    # seen" question, entirely from Designer-authored geometry. No
    # computer vision, no CCTV connection, no AI: this is the same
    # kind of deliberately simple, honest, documented-as-simplified
    # geometry model as DefaultCongestionModel/DefaultCapacityModel
    # elsewhere in this codebase, not a validated real-world sensor
    # model.

    def __init__(
        self,
        ray_count: int = DEFAULT_RAY_COUNT,
        zone_sample_grid: int = DEFAULT_ZONE_SAMPLE_GRID,
    ):

        self.ray_count = ray_count
        self.zone_sample_grid = zone_sample_grid

    # =====================================================

    def compute_camera_visibility(self, camera, building) -> CameraVisibility:

        floor = building.get_floor(camera.floor_id)

        if floor is None:

            return CameraVisibility(camera_id=camera.id)

        return self.compute_camera_visibility_on_floor(
            camera, floor, floor_opaque_segments(floor),
        )

    # =====================================================

    def compute_camera_visibility_on_floor(self, camera, floor, segments) -> CameraVisibility:

        # Same computation as compute_camera_visibility(), but takes
        # an already-resolved Floor and already-built opaque-segment
        # list rather than a Building -- the reuse seam
        # visibility/coverage.py's compute_floor_coverage() needs so
        # that N cameras on one floor build that floor's wall segments
        # ONCE, not N times (floor_opaque_segments() is O(zones *
        # doors) -- see this module's performance notes). Any caller
        # that already has both in hand (a future live-Designer
        # update loop, for instance) can use this directly too.

        if not camera.active:

            # An inactive camera sees nothing -- every zone on its
            # own floor is hidden, not "unknown"; it simply isn't
            # looking. Mirrors Edge.traversable-style "inactive means
            # off, not absent" conventions elsewhere in this codebase.
            return CameraVisibility(
                camera_id=camera.id,
                zone_coverage={zone.id: 0.0 for zone in floor.zones},
                hidden_zone_ids=tuple(zone.id for zone in floor.zones),
            )

        polygon = self._sweep_visibility_polygon(camera, segments)

        zone_coverage = {
            zone.id: self._zone_coverage_fraction(zone, polygon)
            for zone in floor.zones
        }

        visible_zone_ids = tuple(
            zone_id for zone_id, fraction in zone_coverage.items()
            if fraction >= FULLY_VISIBLE_THRESHOLD
        )

        hidden_zone_ids = tuple(
            zone_id for zone_id, fraction in zone_coverage.items()
            if fraction <= HIDDEN_THRESHOLD
        )

        partially_visible_zone_ids = tuple(
            zone_id for zone_id in zone_coverage
            if zone_id not in visible_zone_ids and zone_id not in hidden_zone_ids
        )

        visible_door_ids = tuple(
            door.id for door in floor.doors
            if point_in_polygon(door.center, polygon)
        )

        visible_exit_ids = tuple(
            exit_obj.id for exit_obj in floor.exits
            if point_in_polygon(exit_obj.center, polygon)
        )

        visible_stair_ids = tuple(
            stair.id for stair in floor.stairs
            if self._stair_visible(stair, camera.floor_id, polygon)
        )

        max_visible_distance = max(
            (math.dist(camera.position, point) for point in polygon),
            default=0.0,
        )

        return CameraVisibility(
            camera_id=camera.id,
            visibility_polygon=tuple(polygon),
            zone_coverage=zone_coverage,
            visible_zone_ids=visible_zone_ids,
            partially_visible_zone_ids=partially_visible_zone_ids,
            hidden_zone_ids=hidden_zone_ids,
            visible_door_ids=visible_door_ids,
            visible_exit_ids=visible_exit_ids,
            visible_stair_ids=visible_stair_ids,
            max_visible_distance=max_visible_distance,
        )

    # =====================================================
    # Phase 6 -- Future Compatibility
    #
    # These two methods are deliberately the same geometric test under
    # two names: Simulation ("is this occupant visible, so a synthetic
    # detection should be generated") and Live ("does this reported
    # CCTV detection's position lie within what CAM-003 could actually
    # see") are the same question asked from opposite directions.
    # Replay asks it a third way (was this recorded detection's
    # position plausible for this camera at the time). No mode-specific
    # branching exists anywhere in this class -- whichever mode is
    # calling, it calls the same method.
    # =====================================================

    def point_is_visible(self, camera, building, point) -> bool:

        visibility = self.compute_camera_visibility(camera, building)

        return point_in_polygon(point, visibility.visibility_polygon)

    def is_within_expected_area(self, camera, building, reported_position) -> bool:

        return self.point_is_visible(camera, building, reported_position)

    # =====================================================
    # Internals
    # =====================================================

    def _sweep_visibility_polygon(self, camera, segments):

        # Same angle convention as Camera.coverage_polygon() (models/
        # camera.py) -- 0 degrees along +x, increasing clockwise --
        # so this occlusion-aware polygon is always a subset of that
        # naive sector, never a differently-oriented shape.

        cx, cy = camera.position

        half_fov = camera.horizontal_fov / 2

        polygon = [(cx, cy)]

        for i in range(self.ray_count + 1):

            angle_degrees = (
                camera.rotation
                - half_fov
                + (camera.horizontal_fov * i / self.ray_count)
            )

            angle_radians = math.radians(angle_degrees)

            far_point = (
                cx + camera.max_range * math.cos(angle_radians),
                cy + camera.max_range * math.sin(angle_radians),
            )

            hit = self._closest_intersection((cx, cy), far_point, segments)

            polygon.append(hit if hit is not None else far_point)

        return polygon

    # =====================================================

    def _closest_intersection(self, origin, far_point, segments):

        closest_point = None
        closest_distance = None

        for segment_start, segment_end in segments:

            point = segment_intersection(origin, far_point, segment_start, segment_end)

            if point is None:
                continue

            distance = math.dist(origin, point)

            if closest_distance is None or distance < closest_distance:
                closest_point = point
                closest_distance = distance

        return closest_point

    # =====================================================

    def _zone_coverage_fraction(self, zone, polygon):

        if len(polygon) < 3:
            return 0.0

        points = zone_sample_points(zone, self.zone_sample_grid)

        if not points:
            return 0.0

        visible = sum(
            1 for point in points if point_in_polygon(point, polygon)
        )

        return visible / len(points)

    # =====================================================

    def _stair_visible(self, stair, floor_id, polygon):

        point = _stair_point_on_floor(stair, floor_id)

        if point is None:
            return False

        return point_in_polygon(point, polygon)
