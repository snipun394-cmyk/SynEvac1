from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

from visibility.geometry import point_in_polygon

from camera_calibration.camera_model import CalibrationProfile
from camera_calibration.geometry import intersect_ray_with_floor, pixel_ray_direction


DEFAULT_GRAZING_ANGLE_RATIO = 0.15  # minimum |direction.z| / |direction| for full-confidence projection


@dataclass(frozen=True)
class WorldProjection:

    # Camera Calibration & World Coordinate Projection milestone,
    # Phase 4/7 -- one bounding box's complete projection result, as of
    # one project() call. Every field is Optional and NEVER fabricated
    # -- a missing calibration, a geometrically-impossible ray (see
    # camera_calibration.geometry.intersect_ray_with_floor), or a point
    # that lands outside every known Zone all honestly produce None for
    # the fields that genuinely cannot be determined, never a guessed
    # stand-in.

    world_position: Optional[Tuple[float, float]]
    floor_id: Optional[str]
    zone_id: Optional[str]
    projection_confidence: Optional[float]


class WorldProjector:

    # The Phase 4 pipeline in one class: bounding box -> ground contact
    # point -> world coordinate -> floor coordinate -> zone lookup.
    # Stateless/per-call -- holds only its own configuration
    # (calibrations, zones), never any per-track history (that remains
    # entirely behavior_recognition's job -- Phase 11's "Projection
    # depends only on geometry, camera calibration, building geometry"
    # deliberately excludes time/tracking).

    def __init__(
        self,
        calibrations: Mapping[str, CalibrationProfile],
        zones_by_floor: Mapping[str, Sequence[object]],
        grazing_angle_ratio: float = DEFAULT_GRAZING_ANGLE_RATIO,
    ):

        # zones_by_floor: floor_id -> a sequence of zone-shaped objects
        # (models.zone.Zone, or anything exposing the same `.id`,
        # `.floor_id`, `.contains(x, y)`, `.polygon` shape -- "building
        # geometry", Phase 11's own allowed dependency). Never mutated,
        # never re-derived from a Building here -- a caller supplies
        # whatever it already has.

        self._calibrations = dict(calibrations)
        self._zones_by_floor = {floor_id: tuple(zones) for floor_id, zones in zones_by_floor.items()}
        self.grazing_angle_ratio = grazing_angle_ratio

    # =====================================================

    def project(
        self,
        camera_id: str,
        bounding_box: Optional[Tuple[float, float, float, float]],
        detection_confidence: float,
    ) -> WorldProjection:

        calibration = self._calibrations.get(camera_id)

        if calibration is None or bounding_box is None:
            # No honest basis to project at all -- an uncalibrated
            # camera, or a detection with no geometry (Phase 7's own
            # "never fabricate values").
            return WorldProjection(world_position=None, floor_id=None, zone_id=None, projection_confidence=None)

        ground_pixel = self._ground_contact_point(bounding_box)

        direction = pixel_ray_direction(
            ground_pixel[0], ground_pixel[1], calibration.intrinsics, calibration.extrinsics,
        )
        origin = (
            calibration.extrinsics.position[0],
            calibration.extrinsics.position[1],
            calibration.extrinsics.mount_height,
        )

        world_position = intersect_ray_with_floor(origin, direction)

        if world_position is None:
            # Geometrically impossible (e.g. an implausible pitch/box
            # combination that never reaches the floor) -- still
            # honestly reports which floor this camera itself belongs
            # to (that much is certain regardless of the ray), but no
            # position/zone/confidence.
            return WorldProjection(
                world_position=None, floor_id=calibration.floor_id, zone_id=None, projection_confidence=None,
            )

        zone_id = self._lookup_zone(calibration.floor_id, world_position)
        confidence = self._projection_confidence(direction, detection_confidence)

        return WorldProjection(
            world_position=world_position, floor_id=calibration.floor_id,
            zone_id=zone_id, projection_confidence=confidence,
        )

    # =====================================================

    def _ground_contact_point(self, bounding_box: Tuple[float, float, float, float]) -> Tuple[float, float]:

        # "Assume feet touch the floor" (Phase 4's own explicit
        # instruction, and explicitly NOT a 3D skeleton/pose estimate):
        # the bottom-center of the bounding box, in the same
        # (x1, y1, x2, y2) image-pixel-space convention human_detection.
        # yolo_backend.BoundingBoxDetection already establishes (y
        # increasing downward, y2 = the bottom edge).

        x1, y1, x2, y2 = bounding_box

        return ((x1 + x2) / 2.0, y2)

    # =====================================================

    def _lookup_zone(self, floor_id: str, world_position: Tuple[float, float]) -> Optional[str]:

        zones = self._zones_by_floor.get(floor_id, ())

        x, y = world_position

        for zone in zones:

            if zone.polygon:

                if point_in_polygon((x, y), zone.polygon):
                    return zone.id

            elif zone.contains(x, y):

                return zone.id

        return None

    # =====================================================

    def _projection_confidence(self, direction, detection_confidence: float) -> Optional[float]:

        # A ray nearly parallel to the floor (direction.z close to 0)
        # is numerically unstable: a tiny pixel error near the horizon
        # translates into a huge world-position error, a well-known
        # real limitation of single-camera ground-plane projection --
        # this is honestly reflected as reduced confidence, never
        # hidden. Scales detection_confidence down as the ray
        # approaches horizontal, reaching full confidence once the
        # ray's own downward steepness clears grazing_angle_ratio.

        magnitude = (direction[0] ** 2 + direction[1] ** 2 + direction[2] ** 2) ** 0.5

        if magnitude == 0.0:
            return None

        steepness = abs(direction[2]) / magnitude

        scale = min(1.0, steepness / self.grazing_angle_ratio) if self.grazing_angle_ratio > 0 else 1.0

        return detection_confidence * scale


def nearest_navigation_node(navigation_graph, floor_id: str, world_position: Tuple[float, float]) -> Optional[str]:

    # Phase 4's own motivating question -- "which navigation node are
    # they closest to" -- answered as a light, standalone utility
    # (never a Detection field; Phase 7 names world_position/zone_id/
    # floor_id/world_velocity/projection_confidence as the fields to
    # add, not this). Duck-typed against navigation.graph.NavigationGraph's
    # own real shape (`.nodes`, a dict of Node objects each exposing
    # `.floor_id`/`.node_type`/`.reference`) without importing that
    # package at module scope -- "building geometry" (Phase 11's own
    # allowed dependency), consulted only when a caller actually passes
    # a real graph.
    #
    # Only Zone-type nodes are considered (a Node.ZONE's own `.reference`
    # is a models.zone.Zone, whose `.center` this function reuses
    # directly) -- Outside/AssemblyPoint nodes have no comparable
    # "occupiable position" to measure distance to in the same sense.

    best_node_id = None
    best_distance = None

    x, y = world_position

    for node in navigation_graph.nodes.values():

        if node.floor_id != floor_id:
            continue

        if node.node_type != "Zone":
            continue

        zone = node.reference

        if zone is None:
            continue

        zx, zy = zone.center
        distance = ((zx - x) ** 2 + (zy - y) ** 2) ** 0.5

        if best_distance is None or distance < best_distance or (distance == best_distance and node.id < best_node_id):
            best_distance = distance
            best_node_id = node.id

    return best_node_id
