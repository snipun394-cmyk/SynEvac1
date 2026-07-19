import math

# Shared numerical tolerance for every near-parallel/near-zero-length
# check in this package -- one constant, not a magic number repeated
# at each call site.
EPSILON = 1e-9


def segment_intersection(p1, p2, p3, p4):

    # Standard parametric line-segment intersection. Returns the
    # (x, y) intersection point of segment p1->p2 and segment p3->p4,
    # or None when they don't cross (including the parallel/
    # collinear case -- treated as "no intersection" rather than
    # infinite intersections, since a ray grazing exactly along a
    # wall is not a meaningful line-of-sight block for this engine's
    # purposes).

    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    if abs(denominator) < EPSILON:
        return None

    t = (
        (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
    ) / denominator

    u = (
        (x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)
    ) / denominator

    if -EPSILON <= t <= 1 + EPSILON and -EPSILON <= u <= 1 + EPSILON:
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

    return None


def point_in_polygon(point, polygon_points):

    # Standard ray-casting point-in-polygon test. Deliberately
    # reimplemented here rather than imported from perception/
    # providers/ground_truth_camera_provider.py's private
    # _point_in_polygon() -- that helper is module-local to a
    # different package, and Visibility must not depend on
    # Perception (dependency direction stays one-way: Perception is
    # a future consumer of this engine, never the reverse).

    x, y = point

    if len(polygon_points) < 3:
        return False

    inside = False
    j = len(polygon_points) - 1

    for i in range(len(polygon_points)):

        xi, yi = polygon_points[i]
        xj, yj = polygon_points[j]

        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside

        j = i

    return inside


def closest_point_on_segment(point, seg_start, seg_end):

    # Returns (closest_x, closest_y, t), t in [0, 1] clamped --
    # the point on segment seg_start->seg_end nearest `point`, and
    # how far along the segment it falls.

    px, py = point
    x1, y1 = seg_start
    x2, y2 = seg_end

    dx, dy = x2 - x1, y2 - y1
    length_sq = dx * dx + dy * dy

    if length_sq < EPSILON:
        return (x1, y1, 0.0)

    t = ((px - x1) * dx + (py - y1) * dy) / length_sq
    t_clamped = max(0.0, min(1.0, t))

    return (x1 + t_clamped * dx, y1 + t_clamped * dy, t_clamped)


def perpendicular_distance_to_line(point, seg_start, seg_end):

    x1, y1 = seg_start
    x2, y2 = seg_end
    px, py = point

    length = math.hypot(x2 - x1, y2 - y1)

    if length < EPSILON:
        return math.hypot(px - x1, py - y1)

    return abs(
        (x2 - x1) * (y1 - py) - (x1 - px) * (y2 - y1)
    ) / length
