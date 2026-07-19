import math

from dataclasses import dataclass

from visibility.engine import zone_sample_points
from visibility.geometry import point_in_polygon


# A gap this small (degrees) is not worth recommending a rotation for
# -- the camera is already aimed at the right area; any remaining
# shortfall is a range/FOV issue, not a direction issue.
MIN_MEANINGFUL_ROTATION_DEGREES = 10.0

# Beyond this angular gap, rotating the existing camera in place is
# unlikely to realistically fix coverage (the uncovered area is close
# to directly behind it) -- relocating is the more honest suggestion.
MAX_FIXABLE_BY_ROTATION_DEGREES = 150.0

# Suggested rotations are rounded to the nearest 5 degrees -- "rotate
# by approximately 20 degrees", never a falsely precise "23.7 degrees".
ROTATION_SUGGESTION_ROUND_DEGREES = 5.0

FOV_INCREASE_STEP_DEGREES = 20.0

# A camera whose own target zone(s) are covered at or above this
# fraction needs no directional recommendation at all.
SUFFICIENT_COVERAGE_FRACTION = 0.95

# Grid density used to locate the centroid of a zone's currently-
# uncovered area -- same default DEFAULT_ZONE_SAMPLE_GRID uses
# elsewhere, restated here since this module calls zone_sample_points()
# directly rather than through VisibilityEngine.
CENTROID_SAMPLE_GRID = 6


@dataclass(frozen=True)
class Recommendation:

    # Phase 4's engineering suggestions -- plain, human-readable
    # engineering text plus enough structure (subject_id/subject_type/
    # category) for a Designer panel to group, filter, or highlight
    # them without parsing the message string. These are advisory only
    # -- nothing in this codebase ever applies a Recommendation
    # automatically.

    subject_id: str
    subject_type: str  # "camera" | "door" | "exit" | "stair" | "zone"
    category: str
    message: str


def generate_camera_recommendations(camera, floor, camera_visibility, metrics) -> tuple:

    target_zones = [zone for zone in floor.zones if zone.id in camera.zone_ids] or list(floor.zones)

    worst_zone, worst_fraction = _worst_covered_zone(target_zones, camera_visibility)

    if worst_zone is None or worst_fraction >= SUFFICIENT_COVERAGE_FRACTION:
        return ()

    target_angle, target_distance = _hidden_centroid_polar(
        camera, worst_zone, camera_visibility.visibility_polygon,
    )

    if target_angle is None:
        return ()

    delta = _normalize_angle_difference(target_angle - camera.rotation)
    camera_label = camera.name or camera.id
    zone_label = worst_zone.name or worst_zone.id

    if abs(delta) <= MIN_MEANINGFUL_ROTATION_DEGREES:

        # Already facing the right way -- the shortfall is range or
        # FOV width, not direction.
        if target_distance > camera.max_range:

            return (
                Recommendation(
                    subject_id=camera.id, subject_type="camera", category="increase_range",
                    message=(
                        f"{camera_label} is aimed correctly but '{zone_label}' extends beyond "
                        f"its {camera.max_range:.1f}m range -- consider increasing Maximum Range."
                    ),
                ),
            )

        return (
            Recommendation(
                subject_id=camera.id, subject_type="camera", category="increase_fov",
                message=(
                    f"{camera_label} is aimed correctly but under-covers '{zone_label}' -- "
                    f"consider increasing Horizontal FOV by approximately "
                    f"{FOV_INCREASE_STEP_DEGREES:.0f}°."
                ),
            ),
        )

    if abs(delta) <= MAX_FIXABLE_BY_ROTATION_DEGREES:

        rounded = round(delta / ROTATION_SUGGESTION_ROUND_DEGREES) * ROTATION_SUGGESTION_ROUND_DEGREES
        direction = "clockwise" if rounded > 0 else "counter-clockwise"

        return (
            Recommendation(
                subject_id=camera.id, subject_type="camera", category="rotate",
                message=(
                    f"Rotate {camera_label} by approximately {abs(rounded):.0f}° {direction} "
                    f"to better face '{zone_label}'."
                ),
            ),
        )

    return (
        Recommendation(
            subject_id=camera.id, subject_type="camera", category="relocate",
            message=(
                f"{camera_label} cannot cover '{zone_label}' by rotation alone -- consider "
                f"relocating it."
            ),
        ),
    )


# =====================================================


def generate_network_recommendations(floor, network_analysis, camera_by_id=None) -> tuple:

    camera_by_id = camera_by_id or {}

    def camera_label(camera_id):
        camera = camera_by_id.get(camera_id)
        return (camera.name or camera_id) if camera is not None else camera_id

    recommendations = []

    for exit_obj in floor.exits:

        if exit_obj.id in network_analysis.uncovered_exit_ids:

            recommendations.append(Recommendation(
                subject_id=exit_obj.id, subject_type="exit", category="unmonitored",
                message=f"Exit '{exit_obj.name or exit_obj.id}' is not monitored -- add camera coverage.",
            ))

    for stair in floor.stairs:

        if stair.id in network_analysis.uncovered_stair_ids:

            recommendations.append(Recommendation(
                subject_id=stair.id, subject_type="stair", category="unmonitored",
                message=f"Stair '{stair.name or stair.id}' is not monitored -- add camera coverage.",
            ))

    for door in floor.doors:

        if door.id in network_analysis.uncovered_door_ids:

            recommendations.append(Recommendation(
                subject_id=door.id, subject_type="door", category="unmonitored",
                message=f"Door '{door.name or door.id}' is not monitored -- add camera coverage.",
            ))

    for zone in floor.zones:

        if zone.id in network_analysis.uncovered_zone_ids:

            recommendations.append(Recommendation(
                subject_id=zone.id, subject_type="zone", category="add_camera",
                message=f"Zone '{zone.name or zone.id}' has no camera coverage at all -- add another camera.",
            ))

    for camera_id in network_analysis.excessive_overlap_camera_ids:

        recommendations.append(Recommendation(
            subject_id=camera_id, subject_type="camera", category="excessive_overlap",
            message=(
                f"{camera_label(camera_id)} has excessive overlap with other cameras -- "
                f"consider relocating it or narrowing its field of view."
            ),
        ))

    for camera_id in network_analysis.redundant_camera_ids:

        recommendations.append(Recommendation(
            subject_id=camera_id, subject_type="camera", category="redundancy_ok",
            message=f"{camera_label(camera_id)} provides useful redundant coverage -- acceptable.",
        ))

    for camera_id in network_analysis.isolated_camera_ids:

        recommendations.append(Recommendation(
            subject_id=camera_id, subject_type="camera", category="isolated",
            message=(
                f"{camera_label(camera_id)} is the sole source of coverage for its zone(s) -- "
                f"consider adding a backup camera."
            ),
        ))

    for camera_id in network_analysis.no_coverage_camera_ids:

        recommendations.append(Recommendation(
            subject_id=camera_id, subject_type="camera", category="relocate",
            message=f"{camera_label(camera_id)} provides no meaningful zone coverage -- consider relocating it.",
        ))

    return tuple(recommendations)


# =====================================================
# Internals
# =====================================================


def _worst_covered_zone(zones, camera_visibility):

    worst_zone = None
    worst_fraction = 1.0

    for zone in zones:

        fraction = camera_visibility.zone_coverage.get(zone.id, 0.0)

        if fraction < worst_fraction:
            worst_fraction = fraction
            worst_zone = zone

    return worst_zone, worst_fraction


# =====================================================


def _hidden_centroid_polar(camera, zone, visibility_polygon):

    # The centroid of whichever of this zone's sample points fall
    # outside the camera's own visibility polygon -- "roughly where
    # the gap is", not a precise geometric target. Returns
    # (angle_degrees, distance) from the camera's own position, or
    # (None, None) if every sample point is already visible (the
    # camera has nothing left to gain from rotating/widening/ranging
    # further for this particular zone).

    points = zone_sample_points(zone, CENTROID_SAMPLE_GRID)

    hidden_points = [
        point for point in points if not point_in_polygon(point, visibility_polygon)
    ]

    if not hidden_points:
        return None, None

    centroid_x = sum(point[0] for point in hidden_points) / len(hidden_points)
    centroid_y = sum(point[1] for point in hidden_points) / len(hidden_points)

    camera_x, camera_y = camera.position

    dx = centroid_x - camera_x
    dy = centroid_y - camera_y

    distance = math.hypot(dx, dy)

    if distance < 1e-6:
        return None, None

    angle = math.degrees(math.atan2(dy, dx))

    return angle, distance


# =====================================================


def _normalize_angle_difference(delta):

    while delta > 180.0:
        delta -= 360.0

    while delta < -180.0:
        delta += 360.0

    return delta
