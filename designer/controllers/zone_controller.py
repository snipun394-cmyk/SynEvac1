from PyQt6.QtCore import QPointF
from PyQt6.QtGui import QPolygonF


class ZoneController:

    GRID_SIZE = 50

    def __init__(self, item):

        self.item = item

    # =====================================================

    def move_vertex(self, index, position):

        polygon = self.item.polygon()

        polygon[index] = QPointF(position)

        self.item.setPolygon(
            QPolygonF(polygon)
        )

        self.item.sync_to_model()

        if self.item.geometry_changed_callback:

            self.item.geometry_changed_callback(
                self.item
            )

    # =====================================================

    def insert_vertex(self, index, position):

        polygon = self.item.polygon()

        polygon.insert(
            index,
            QPointF(position),
        )

        self.item.setPolygon(
            QPolygonF(polygon)
        )

        self.item.sync_to_model()

    # =====================================================

    def remove_vertex(self, index):

        polygon = self.item.polygon()

        if polygon.count() <= 3:
            return

        polygon.remove(index)

        self.item.setPolygon(
            QPolygonF(polygon)
        )

        self.item.sync_to_model()

    # =====================================================

    def vertices(self):

        polygon = self.item.polygon()

        return [
            polygon.at(i)
            for i in range(polygon.count())
        ]

    # =====================================================

    def vertex_count(self):

        return self.item.polygon().count()

    # =====================================================

    def polygon(self):

        return self.item.polygon()

    # =====================================================

    def move(self, dx, dy):

        self.item.moveBy(dx, dy)

        self.item.sync_to_model()

    # =====================================================

    def rename(self, name):

        self.item.rename(name)