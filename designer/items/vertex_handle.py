from PyQt6.QtCore import QPointF
from PyQt6.QtGui import QColor, QBrush, QPen
from PyQt6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
)


class VertexHandle(QGraphicsEllipseItem):

    SIZE = 10
    GRID_SIZE = 50

    def __init__(self, x, y, index, parent=None):
        super().__init__(
            -self.SIZE / 2,
            -self.SIZE / 2,
            self.SIZE,
            self.SIZE,
            parent,
        )

        self.index = index
        self.owner = parent

        self.setPos(x, y)

        self.setBrush(
            QBrush(QColor(255, 80, 80))
        )

        self.setPen(
            QPen(
                QColor(255, 255, 255),
                1,
            )
        )

        self.setFlag(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable,
            True,
        )

        self.setFlag(
            QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges,
            True,
        )

        self.setFlag(
            QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations,
            True,
        )

        self.setZValue(1000)

    # =====================================================

    def itemChange(self, change, value):

        if (
            change
            == QGraphicsItem.GraphicsItemChange.ItemPositionChange
        ):

            x = (
                round(value.x() / self.GRID_SIZE)
                * self.GRID_SIZE
            )

            y = (
                round(value.y() / self.GRID_SIZE)
                * self.GRID_SIZE
            )

            return QPointF(x, y)

        if (
            change
            == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged
        ):

            if self.owner:

                polygon = self.owner.polygon()

                polygon[self.index] = self.pos()

                self.owner.setPolygon(polygon)

                self.owner.sync_to_model()

                if self.owner.geometry_changed_callback:
                    self.owner.geometry_changed_callback(
                        self.owner
                    )

        return super().itemChange(change, value)