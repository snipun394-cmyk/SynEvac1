from PyQt6.QtCore import QPointF
from PyQt6.QtGui import QColor, QPen
from PyQt6.QtWidgets import (
    QGraphicsItem,
    QGraphicsLineItem,
)


class DoorItem(QGraphicsLineItem):

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

        self._selected = False
        self._highlighted = False

        # =====================================================
        # Appearance -- violet reads as "connector", distinct
        # from Exit's green and Stair's brown.
        # =====================================================

        self.default_pen = QPen(QColor(170, 100, 220), 6)
        self.selected_pen = QPen(QColor(255, 255, 0), 6)
        self.locked_pen = QPen(QColor(220, 60, 60), 6)
        self.inactive_pen = QPen(QColor(120, 120, 120), 6)

        # Manual Simulation Sandbox path visualization -- amber,
        # same convention ZoneRectangle/StairItem use.
        self.highlight_pen = QPen(QColor(255, 165, 0), 6)

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

        self._update_appearance()

    # =====================================================

    def _update_appearance(self):

        if self._selected:

            self.setPen(self.selected_pen)

            return

        if self._highlighted:

            self.setPen(self.highlight_pen)

            return

        active = (
            self.model.active
            if self.model is not None
            else True
        )

        if not active:

            self.setPen(self.inactive_pen)

            return

        locked = (
            self.model.locked
            if self.model is not None
            else False
        )

        if locked:

            self.setPen(self.locked_pen)

            return

        self.setPen(self.default_pen)

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
    # Called after the Property Panel writes Door Type/Normally
    # Open/Locked/Active/Zone A/Zone B straight onto the model --
    # none of those move the item, so only appearance (locked/
    # inactive tint) needs an explicit refresh.
    # =====================================================

    def refresh_geometry(self):

        self._update_appearance()

    # =====================================================

    def rename(self, name):

        self.object_name = name

        if self.model is not None:
            self.model.name = name

        if self.geometry_changed_callback:
            self.geometry_changed_callback(self)

    # =====================================================

    def set_selected(self, selected):

        self._selected = selected

        self._update_appearance()

    # =====================================================

    def set_highlighted(self, highlighted):

        self._highlighted = highlighted

        self._update_appearance()
