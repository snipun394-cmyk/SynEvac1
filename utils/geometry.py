from dataclasses import dataclass, field
from math import sqrt


# ==========================================================
# Point
# ==========================================================

@dataclass
class Point:

    x: float
    y: float

    def to_tuple(self):

        return (self.x, self.y)

    def distance_to(self, other):

        return sqrt(
            (self.x - other.x) ** 2
            + (self.y - other.y) ** 2
        )


# ==========================================================
# Line
# ==========================================================

@dataclass
class Line:

    start: Point
    end: Point

    @property
    def length(self):

        return self.start.distance_to(self.end)


# ==========================================================
# Polygon
# ==========================================================

@dataclass
class Polygon:

    points: list[Point] = field(default_factory=list)

    # ------------------------------------------------------

    def add(self, point: Point):

        self.points.append(point)

    # ------------------------------------------------------

    def insert(self, index, point):

        self.points.insert(index, point)

    # ------------------------------------------------------

    def remove(self, index):

        self.points.pop(index)

    # ------------------------------------------------------

    def move(self, dx, dy):

        for point in self.points:

            point.x += dx
            point.y += dy

    # ------------------------------------------------------

    @property
    def vertex_count(self):

        return len(self.points)

    # ------------------------------------------------------

    @property
    def area(self):

        if len(self.points) < 3:
            return 0

        area = 0

        n = len(self.points)

        for i in range(n):

            j = (i + 1) % n

            area += (
                self.points[i].x
                * self.points[j].y
            )

            area -= (
                self.points[j].x
                * self.points[i].y
            )

        return abs(area) / 2

    # ------------------------------------------------------

    @property
    def perimeter(self):

        if len(self.points) < 2:
            return 0

        total = 0

        n = len(self.points)

        for i in range(n):

            j = (i + 1) % n

            total += self.points[i].distance_to(
                self.points[j]
            )

        return total

    # ------------------------------------------------------

    @property
    def bounding_box(self):

        xs = [p.x for p in self.points]
        ys = [p.y for p in self.points]

        return (
            min(xs),
            min(ys),
            max(xs),
            max(ys),
        )

    # ------------------------------------------------------

    def to_list(self):

        return [
            (p.x, p.y)
            for p in self.points
        ]