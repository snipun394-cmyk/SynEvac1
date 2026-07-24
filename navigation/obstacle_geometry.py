from typing import Sequence, Tuple

from visibility.geometry import segment_intersection


# =====================================================
# Obstacle -> Navigation & Evacuation Connectivity milestone, Phase 1/3
# -- the smallest correct integration point this milestone's own
# investigation found: Node/Edge are deliberately "thin, read-only
# views" with no dynamic state of their own (see navigation/node.py,
# navigation/edge.py's own docstrings), and Door.locked/Exit.is_blocked
# already reach Edge.traversable by LIVE getattr() on `reference` --
# never a value baked in at graph-build time, never requiring a graph
# rebuild to notice a change. Obstacle intersection follows the exact
# same discipline: Edge.blocking_obstacles holds LIVE references to the
# same Obstacle objects Floor.obstacles already owns (populated once by
# NavigationGraphGenerator, mirroring how `reference` itself is
# populated), and this module's own segment_blocked_by_obstacles() is
# called fresh on every Edge.traversable access -- an Obstacle's
# position/active/traversability can change at any time (Designer
# drag, a scenario activation event, a live building edit) and the very
# next traversable check honestly reflects it, with zero graph rebuild
# required. This is a stricter, MORE honest guarantee than "rebuild the
# graph to see the change" would have been.
#
# Deliberately reuses visibility.geometry.segment_intersection rather
# than reimplementing segment/rectangle intersection a second time in
# this package -- the one existing geometry utility in this codebase
# that already solves exactly this primitive (Phase 1's own "what
# existing geometry utilities can be reused" question).
# =====================================================


def _rectangle_corners(obstacle) -> Tuple[Tuple[float, float], ...]:

    x1, y1 = obstacle.x, obstacle.y
    x2, y2 = obstacle.x + obstacle.length, obstacle.y + obstacle.width

    return ((x1, y1), (x2, y1), (x2, y2), (x1, y2))


def _point_in_rectangle(point, obstacle) -> bool:

    px, py = point

    x1, y1 = obstacle.x, obstacle.y
    x2, y2 = obstacle.x + obstacle.length, obstacle.y + obstacle.width

    return (min(x1, x2) <= px <= max(x1, x2)) and (min(y1, y2) <= py <= max(y1, y2))


def _segment_intersects_obstacle(seg_start, seg_end, obstacle) -> bool:

    # A segment fully contained within the rectangle (both endpoints
    # inside, no edge crossing) must still count as blocked -- e.g. a
    # short Door segment placed entirely inside an Obstacle's footprint
    # -- so endpoint containment is checked in addition to the four
    # edge-crossing tests, not instead of them.
    if _point_in_rectangle(seg_start, obstacle) or _point_in_rectangle(seg_end, obstacle):
        return True

    corners = _rectangle_corners(obstacle)

    for i in range(4):

        corner_a = corners[i]
        corner_b = corners[(i + 1) % 4]

        if segment_intersection(seg_start, seg_end, corner_a, corner_b) is not None:
            return True

    return False


def segment_blocked_by_obstacles(
    seg_start: Tuple[float, float],
    seg_end: Tuple[float, float],
    obstacles: Sequence[object],
) -> bool:

    # Phase 2's own honest semantics: an Obstacle blocks a connection
    # ONLY when it is currently active AND its traversability is
    # exactly "Blocked" -- "Passable" and "Reduced Width" never affect
    # traversability at all (Phase 2's explicit "do not invent
    # congestion/cost semantics" boundary; Obstacle.traversal_cost
    # remains uninterpreted here, exactly as its own docstring already
    # states). A degenerate obstacle (zero/negative length or width)
    # is geometrically well-defined (an empty or inverted rectangle)
    # and simply never intersects anything -- never a crash, never a
    # fabricated block.

    if seg_start is None or seg_end is None:
        return False

    for obstacle in obstacles:

        if not getattr(obstacle, "active", False):
            continue

        if getattr(obstacle, "traversability", None) != "Blocked":
            continue

        if _segment_intersects_obstacle(seg_start, seg_end, obstacle):
            return True

    return False
