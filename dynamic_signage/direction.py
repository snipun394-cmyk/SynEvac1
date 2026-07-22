import math
from typing import Optional, Tuple

from models.dynamic_sign import SignIndication

from dynamic_signage.models import DynamicSignageConfig


# =====================================================
# Live Dynamic Evacuation Signage milestone, Phase 4 -- a PURE geometry
# layer. Input: sign position, sign orientation, next meaningful route
# target position. Output: a relative direction (LEFT/RIGHT/STRAIGHT),
# honestly derived from the signed angular difference between the
# sign's own facing direction and the vector toward the target -- never
# a fabricated direction, never north/south/compass wording.
#
# COORDINATE-SYSTEM CONVENTION -- verified against the existing
# Designer, not assumed (Phase 4's own explicit requirement):
#
#   models.camera.Camera.coverage_polygon()'s own docstring states
#   "0 degrees points along +x, increasing clockwise -- matching Qt's
#   own rotation convention", confirmed against designer.items.
#   camera_item.CameraItem (Qt QGraphicsItem.setRotation(), a
#   y-axis-down scene). DynamicEvacuationSign.orientation reuses this
#   EXACT SAME convention (see models/dynamic_sign.py) -- a sign is
#   just another oriented EngineeringAsset, not a new convention.
#
#   Facing direction vector at angle theta (degrees) is therefore
#   (cos(theta), sin(theta)) in this same (x right, y down) frame --
#   the identical formula Camera.coverage_polygon() already uses.
#
#   Derivation of LEFT/RIGHT (worked out directly from this
#   convention, not assumed): picture the floor plan as a map viewed
#   from directly above, "up" on screen (smaller y) read as compass
#   north. Facing north is theta = -90 (direction (0, -1)); standing
#   on the map facing north, your right hand points east, i.e.
#   direction (1, 0), theta = 0. Going from facing theta=-90 to
#   right-hand theta=0 is a change of +90 degrees. So: turning the
#   facing angle by +90 degrees (this codebase's own "increasing
#   clockwise") always yields the sign's own RIGHT-hand direction, and
#   -90 degrees always yields LEFT. Therefore a target whose bearing
#   requires a POSITIVE signed rotation away from the sign's facing
#   angle is to the sign's RIGHT; a NEGATIVE signed rotation is to its
#   LEFT. This is verified computationally in
#   tests/test_dynamic_signage_direction.py against concrete worked
#   positions, not merely asserted here.
# =====================================================


def _normalize_degrees(angle: float) -> float:

    # Wrap to (-180, 180] -- the one canonical representation every
    # comparison below relies on, so a threshold boundary is never
    # ambiguous between e.g. 180 and -180.

    wrapped = math.fmod(angle, 360.0)

    if wrapped <= -180.0:
        wrapped += 360.0

    if wrapped > 180.0:
        wrapped -= 360.0

    return wrapped


def bearing_to(from_position: Tuple[float, float], to_position: Tuple[float, float]) -> Optional[float]:

    # None means "the target is exactly at the sign's own position" --
    # no honest bearing exists (a zero-length vector has no direction).
    # Never fabricated as an arbitrary 0.0.

    dx = to_position[0] - from_position[0]
    dy = to_position[1] - from_position[1]

    if dx == 0.0 and dy == 0.0:
        return None

    return math.degrees(math.atan2(dy, dx))


def relative_direction(
    sign_position: Tuple[float, float],
    sign_orientation: float,
    target_position: Tuple[float, float],
    config: Optional[DynamicSignageConfig] = None,
) -> Optional[str]:

    # Returns one of SignIndication.STRAIGHT/LEFT/RIGHT, or None when no
    # honest direction can be derived (target coincides with the sign's
    # own position) -- never a fabricated default direction.

    config = config if config is not None else DynamicSignageConfig()

    target_bearing = bearing_to(sign_position, target_position)

    if target_bearing is None:
        return None

    delta = _normalize_degrees(target_bearing - sign_orientation)

    if abs(delta) <= config.straight_threshold_degrees:
        return SignIndication.STRAIGHT

    return SignIndication.RIGHT if delta > 0 else SignIndication.LEFT
