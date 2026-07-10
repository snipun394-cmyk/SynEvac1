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

        if self.model is not None:

            self.object_id = self.model.id
            self.object_name = self.model.name

        else:

            self.object_id = ""
            self.object_name = ""

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

            self.sync_to_model()

            if self.geometry_changed_callback:

                self.geometry_changed_callback(
                    self
                )

        return super().itemChange(change, value)

    # =====================================================

    def setLine(self, *args):

        super().setLine(*args)

        self.sync_to_model()

        if self.geometry_changed_callback:
            self.geometry_changed_callback(self)

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

        self.object_name = self.model.name

    # =====================================================

    def rename(self, name):

        self.object_name = name

        if self.model is not None:
            self.model.name = name

        if self.geometry_changed_callback:
            self.geometry_changed_callback(self)

    # =====================================================

    def set_selected(self, selected):

        if selected:
            self.setPen(self.selected_pen)
        else:
            self.setPen(self.default_pen)
