from PyQt6.QtCore import QPointF
from PyQt6.QtGui import QBrush, QColor, QPen
from PyQt6.QtWidgets import QGraphicsItem, QGraphicsRectItem


class AssemblyPointItem(QGraphicsRectItem):

    GRID_SIZE = 50

    def __init__(self, x, y, length, width, model=None):

        length_px = length * self.GRID_SIZE
        width_px = width * self.GRID_SIZE

        super().__init__(
            -length_px / 2,
            -width_px / 2,
            length_px,
            width_px,
        )

        self.model = model

        if self.model is not None:

            self.object_id = self.model.id
            self.object_name = self.model.name

        else:

            self.object_id = ""
            self.object_name = ""

        self.setPos(x, y)

        self._selected = False

        # =====================================================
        # Appearance -- green reads as "safe destination", the
        # same convention as the Exit tool's green line.
        # =====================================================

        self.default_brush = QBrush(QColor(0, 220, 120, 90))
        self.selected_brush = QBrush(QColor(255, 255, 0, 120))
        self.inactive_brush = QBrush(QColor(120, 120, 120, 60))

        self.default_pen = QPen(QColor(0, 220, 120), 2)
        self.selected_pen = QPen(QColor(255, 255, 0), 2)
        self.inactive_pen = QPen(QColor(120, 120, 120), 2)

        # =====================================================
        # Flags
        # =====================================================

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

            self.setBrush(self.selected_brush)
            self.setPen(self.selected_pen)

        elif self.model is not None and not self.model.active:

            self.setBrush(self.inactive_brush)
            self.setPen(self.inactive_pen)

        else:

            self.setBrush(self.default_brush)
            self.setPen(self.default_pen)

    # =====================================================

    def sync_to_model(self):

        if self.model is None:
            return

        self.model.position = (
            self.pos().x() / self.GRID_SIZE,
            self.pos().y() / self.GRID_SIZE,
        )

        self.model.length = (
            self.rect().width() / self.GRID_SIZE
        )

        self.model.width = (
            self.rect().height() / self.GRID_SIZE
        )

        self.object_name = self.model.name

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

    def set_size(self, length_m, width_m):

        length_px = length_m * self.GRID_SIZE
        width_px = width_m * self.GRID_SIZE

        self.setRect(
            -length_px / 2,
            -width_px / 2,
            length_px,
            width_px,
        )

    # =====================================================

    def setRect(self, *args):

        super().setRect(*args)

        self.sync_to_model()

        if self.geometry_changed_callback:
            self.geometry_changed_callback(self)

    # =====================================================
    # Called after the Property Panel writes Capacity/
    # Description/Active straight onto the model -- none of
    # those change the item's geometry, so only appearance
    # (active/inactive tint) needs an explicit refresh.
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
