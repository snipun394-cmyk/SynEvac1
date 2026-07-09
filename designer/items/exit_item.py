from PyQt6.QtCore import QPointF
from PyQt6.QtGui import QColor, QPen
from PyQt6.QtWidgets import (
    QGraphicsItem,
    QGraphicsLineItem,
)


class ExitItem(QGraphicsLineItem):

    GRID_SIZE = 50

    def __init__(self, x1, y1, x2, y2, model=None):
        super().__init__(0, 0, x2 - x1, y2 - y1)

        self.model = model

        self.setPos(x1, y1)

        self.default_pen = QPen(
            QColor(0, 255, 0),
            6,
        )

        self.selected_pen = QPen(
            QColor(255, 255, 0),
            6,
        )

        self.setPen(self.default_pen)

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

                self.geometry_changed_callback(
                    self
                )

        return super().itemChange(change, value)

    # =====================================================

    def sync_to_model(self):

        if self.model is None:
            return

        line = self.line()

        self.model.start_point = (
            (self.pos().x() + line.x1())
            / self.GRID_SIZE,
            (self.pos().y() + line.y1())
            / self.GRID_SIZE,
        )

        self.model.end_point = (
            (self.pos().x() + line.x2())
            / self.GRID_SIZE,
            (self.pos().y() + line.y2())
            / self.GRID_SIZE,
        )

    # =====================================================

    def set_selected(self, selected):

        if selected:
            self.setPen(self.selected_pen)
        else:
            self.setPen(self.default_pen)