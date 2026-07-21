import math
from typing import Optional, Tuple

from camera_calibration.camera_model import CameraExtrinsics, CameraIntrinsics


Vector3 = Tuple[float, float, float]


# Camera Calibration & World Coordinate Projection milestone, Phase 4 --
# pure geometry only (this module's whole reason to exist separately
# from projection.py: no camera_id/Zone/building lookup of any kind
# here, just vectors and a floor plane). World axes: X/Y are the SAME
# horizontal floor-plan meters models.zone.Zone/models.camera.Camera
# already use (X right, Y "down" on the floor-plan, matching Camera.
# coverage_polygon()'s own rotation convention); Z is height above THIS
# FLOOR's own surface (Z=0 at the floor, increasing upward) --
# deliberately floor-LOCAL, never a building-wide absolute elevation
# (Zone/Detection never carry one either -- see Detection.position's
# own docstring on this codebase's existing "floor_id + 2D (x, y)" plan
# for occupant location, nothing 3D/absolute).


def _vector_length(v: Vector3) -> float:

    return math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)


def _cross(a: Vector3, b: Vector3) -> Vector3:

    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _add(a: Vector3, b: Vector3, scale_b: float = 1.0) -> Vector3:

    return (a[0] + scale_b * b[0], a[1] + scale_b * b[1], a[2] + scale_b * b[2])


# =====================================================
# Camera orientation basis
# =====================================================


def camera_basis_vectors(extrinsics: CameraExtrinsics) -> Tuple[Vector3, Vector3, Vector3]:

    # Returns (forward, right, down) -- three orthonormal, world-space
    # unit vectors describing this camera's own local frame. `forward`
    # is the direction pixel (principal_point_x, principal_point_y) --
    # dead image center -- looks toward. `right`/`down` are the
    # directions increasing image u/v (column/row) move toward, so a
    # pixel offset from center maps onto them directly (pixel_ray_
    # direction() below).
    #
    # Construction (all in radians internally, degrees at the public
    # boundary): start from yaw+pitch alone (roll = 0) using the
    # standard graphics "look-at" recipe -- forward from yaw/pitch,
    # right = normalize(forward x world_up), down = forward x right --
    # then rotate right/down about forward by roll. world_up = (0, 0, 1)
    # throughout (Z is always "up" in this module's convention).
    #
    # Verified by direct construction (not by citing an external
    # convention -- none existed in this codebase before this
    # milestone): at yaw=0, pitch=0, roll=0, forward=(1,0,0) (facing
    # +X, matching Camera.rotation=0's own existing meaning), right=
    # (0,-1,0), down=(0,0,-1) (increasing image row v moves the ray
    # toward -Z, i.e. downward toward the floor -- correct for a level
    # camera looking at a person's feet below the image center).

    yaw = math.radians(extrinsics.yaw_degrees)
    pitch = math.radians(extrinsics.pitch_degrees)
    roll = math.radians(extrinsics.roll_degrees)

    forward = (
        math.cos(yaw) * math.cos(pitch),
        math.sin(yaw) * math.cos(pitch),
        -math.sin(pitch),
    )

    world_up: Vector3 = (0.0, 0.0, 1.0)

    right_raw = _cross(forward, world_up)
    right_length = _vector_length(right_raw)

    if right_length == 0.0:
        # forward is exactly vertical (pitch = +/-90 degrees) -- yaw is
        # meaningless at that exact orientation (gimbal-lock case);
        # fall back to the yaw-0 horizontal reference so this never
        # raises, only degrades to an arbitrary-but-deterministic
        # choice of "right".
        right = (0.0, -1.0, 0.0)
    else:
        right = (right_raw[0] / right_length, right_raw[1] / right_length, right_raw[2] / right_length)

    down = _cross(forward, right)

    if roll != 0.0:

        cos_r, sin_r = math.cos(roll), math.sin(roll)

        rolled_right = (
            right[0] * cos_r + down[0] * sin_r,
            right[1] * cos_r + down[1] * sin_r,
            right[2] * cos_r + down[2] * sin_r,
        )
        rolled_down = (
            -right[0] * sin_r + down[0] * cos_r,
            -right[1] * sin_r + down[1] * cos_r,
            -right[2] * sin_r + down[2] * cos_r,
        )
        right, down = rolled_right, rolled_down

    return forward, right, down


# =====================================================
# Pixel -> world-space ray
# =====================================================


def pixel_ray_direction(
    pixel_x: float,
    pixel_y: float,
    intrinsics: CameraIntrinsics,
    extrinsics: CameraExtrinsics,
) -> Vector3:

    # The (unnormalized) world-space direction of the ray from this
    # camera's optical center through image pixel (pixel_x, pixel_y).
    # Standard pinhole model: a unit step along the optical axis
    # (`forward`) plus (pixel - principal_point) / focal_length steps
    # along `right`/`down` -- exactly the inverse of the usual
    # world-to-pixel projection equation.

    forward, right, down = camera_basis_vectors(extrinsics)

    dx = (pixel_x - intrinsics.principal_point_x) / intrinsics.focal_length_x
    dy = (pixel_y - intrinsics.principal_point_y) / intrinsics.focal_length_y

    direction = _add(forward, right, dx)
    direction = _add(direction, down, dy)

    return direction


# =====================================================
# Ray / floor-plane intersection
# =====================================================


def intersect_ray_with_floor(
    origin: Vector3,
    direction: Vector3,
    floor_z: float = 0.0,
) -> Optional[Tuple[float, float]]:

    # Solves for the (x, y) point where the ray (origin + t*direction,
    # t >= 0) crosses the horizontal plane Z = floor_z -- "assume feet
    # touch the floor" (Phase 4's own explicit assumption; this
    # function never estimates a 3D skeleton or body pose). Returns
    # None -- never a fabricated point -- whenever direction.z >= 0:
    # a ray pointing level with or above the horizon never reaches a
    # floor plane below the camera at any positive t, and forcing an
    # intersection anyway would silently invent a position for
    # geometry that does not honestly support one (e.g. a bounding box
    # whose "ground contact point" ended up above the camera's own
    # height due to a bad detection or extreme pitch).

    if direction[2] >= 0.0:
        return None

    t = (floor_z - origin[2]) / direction[2]

    if t < 0.0:
        # The floor plane is technically crossed only BEHIND the
        # camera along this ray -- geometrically possible for certain
        # origin/direction combinations, never an honest "in front of
        # the camera" observation.
        return None

    return (origin[0] + t * direction[0], origin[1] + t * direction[1])
