from PyQt6.QtCore import QPointF
from PyQt6.QtGui import QBrush, QColor, QPen
from PyQt6.QtWidgets import QGraphicsItem, QGraphicsRectItem


class ObstacleItem(QGraphicsRectItem):

    GRID_SIZE = 50

    # Fill tint per Traversability -- a glance at the canvas
    # tells you how much a given obstacle impedes movement.
    TRAVERSABILITY_COLORS = {
        "Blocked": QColor(210, 60, 40),
        "Reduced Width": QColor(230, 160, 40),
        "Passable": QColor(130, 130, 140),
    }

    def __init__(self, x, y, length, width, model=None):
        super().__init__(
            0,
            0,
            length * self.GRID_SIZE,
            width * self.GRID_SIZE,
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
        # Appearance
        # =====================================================

        self.selected_brush = QBrush(QColor(255, 255, 0, 130))
        self.selected_pen = QPen(QColor(255, 255, 0), 2)

        self.inactive_brush = QBrush(QColor(120, 120, 120, 70))
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

    def _traversability_color(self):

        traversability = (
            self.model.traversability
            if self.model is not None
            else "Blocked"
        )

        return self.TRAVERSABILITY_COLORS.get(
            traversability,
            QColor(160, 160, 160),
        )

    # =====================================================

    def _update_appearance(self):

        if self._selected:

            self.setBrush(self.selected_brush)
            self.setPen(self.selected_pen)

            return

        active = (
            self.model.active
            if self.model is not None
            else True
        )

        if not active:

            self.setBrush(self.inactive_brush)
            self.setPen(self.inactive_pen)

            return

        color = self._traversability_color()

        self.setBrush(
            QBrush(
                QColor(
                    color.red(),
                    color.green(),
                    color.blue(),
                    140,
                )
            )
        )

        self.setPen(
            QPen(
                color,
                2,
            )
        )

    # =====================================================

    def sync_to_model(self):

        if self.model is None:
            return

        self.model.move_to(
            self.pos().x() / self.GRID_SIZE,
            self.pos().y() / self.GRID_SIZE,
        )

        self.model.resize(
            self.rect().width() / self.GRID_SIZE,
            self.rect().height() / self.GRID_SIZE,
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
                self.geometry_changed_callback(self)

        return super().itemChange(change, value)

    # =====================================================

    def setRect(self, *args):

        super().setRect(*args)

        self.sync_to_model()

        if self.geometry_changed_callback:
            self.geometry_changed_callback(self)

    # =====================================================
    # Called after the Property Panel writes Type/Traversability/
    # Traversal Cost/Active straight onto the model -- none of
    # those change the item's geometry, so only appearance needs
    # an explicit refresh.
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
