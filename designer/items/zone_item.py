from PyQt6.QtCore import QPointF
from PyQt6.QtGui import QColor, QBrush, QPen, QPolygonF
from PyQt6.QtWidgets import QGraphicsItem, QGraphicsPolygonItem


class ZoneItem(QGraphicsPolygonItem):

    GRID_SIZE = 50

    def __init__(self, polygon, model=None):
        super().__init__(polygon)

        self.model = model

        if model is not None:
            self.object_id = model.id
            self.object_name = model.name
        else:
            self.object_id = ""
            self.object_name = ""

        self.default_brush = QBrush(
            QColor(0, 150, 255, 80)
        )

        self.selected_brush = QBrush(
            QColor(255, 255, 0, 120)
        )

        self.setBrush(self.default_brush)

        self.setPen(
            QPen(
                QColor(0, 170, 255),
                2,
            )
        )

        self.setFlag(
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable,
            True,
        )

        self.setFlag(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable,
            True,
        )

        self.setFlag(
            QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges,
            True,
        )

        self.geometry_changed_callback = None

    # =====================================================

    def itemChange(self, change, value):

        if (
            change
            == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged
        ):

            self.sync_to_model()

            if self.geometry_changed_callback:
                self.geometry_changed_callback(self)

        return super().itemChange(change, value)

    # =====================================================

    def sync_to_model(self):

        if self.model is None:
            return

        polygon = self.polygon()

        pts = []

        for i in range(polygon.count()):

            p = polygon.at(i)

            pts.append(
                (
                    (p.x() + self.pos().x())
                    / self.GRID_SIZE,
                    (p.y() + self.pos().y())
                    / self.GRID_SIZE,
                )
            )

        self.model.polygon = pts

    # =====================================================

    def rename(self, name):

        self.object_name = name

        if self.model:
            self.model.name = name

    # =====================================================

    def set_selected(self, selected):

        if selected:
            self.setBrush(self.selected_brush)
        else:
            self.setBrush(self.default_brush)

    # =====================================================

    def update_polygon(self, points):

        polygon = QPolygonF()

        for x, y in points:

            polygon.append(
                QPointF(
                    x * self.GRID_SIZE,
                    y * self.GRID_SIZE,
                )
            )

        self.setPolygon(polygon)

        self.sync_to_model()

    # =====================================================

    def to_dict(self):

        if self.model:
            return self.model.to_dict()

        return {}