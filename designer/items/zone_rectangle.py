from PyQt6.QtCore import QPointF
from PyQt6.QtGui import QColor, QBrush, QPen
from PyQt6.QtWidgets import QGraphicsItem, QGraphicsRectItem


class ZoneRectangle(QGraphicsRectItem):

    GRID_SIZE = 50

    def __init__(self, x, y, w, h, model=None):
        super().__init__(0, 0, w, h)

        self.model = model

        self.setPos(x, y)

        # =====================================================
        # Appearance
        # =====================================================

        self.default_brush = QBrush(
            QColor(0, 150, 255, 80)
        )

        self.selected_brush = QBrush(
            QColor(255, 255, 0, 120)
        )

        # Manual Simulation Sandbox path visualization -- an amber
        # tint distinct from selection's yellow, so a highlighted
        # Zone on an occupant's planned route reads differently from
        # one the user has clicked on.
        self.highlight_brush = QBrush(
            QColor(255, 165, 0, 120)
        )

        self._selected = False
        self._highlighted = False

        self.setBrush(self.default_brush)

        self.setPen(
            QPen(
                QColor(0, 170, 255),
                2,
            )
        )

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

    # =====================================================
    # zone_id/zone_name are read live from the model rather than
    # cached at construction time -- a stale copy here is exactly
    # what left the Property Panel's ID field blank whenever this
    # item was built with model=None and had .model assigned
    # afterward (see both call sites in GraphicsScene).
    # =====================================================

    @property
    def zone_id(self):

        return (
            self.model.id
            if self.model is not None
            else ""
        )

    @property
    def zone_name(self):

        return (
            self.model.name
            if self.model is not None
            else ""
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

    @property
    def top_left(self):

        return (
            self.pos().x() / self.GRID_SIZE,
            self.pos().y() / self.GRID_SIZE,
        )

    @property
    def top_right(self):

        return (
            (self.pos().x() + self.rect().width())
            / self.GRID_SIZE,
            self.pos().y() / self.GRID_SIZE,
        )

    @property
    def bottom_right(self):

        return (
            (self.pos().x() + self.rect().width())
            / self.GRID_SIZE,
            (self.pos().y() + self.rect().height())
            / self.GRID_SIZE,
        )

    @property
    def bottom_left(self):

        return (
            self.pos().x() / self.GRID_SIZE,
            (self.pos().y() + self.rect().height())
            / self.GRID_SIZE,
        )

    @property
    def width_m(self):

        return self.rect().width() / self.GRID_SIZE

    @property
    def height_m(self):

        return self.rect().height() / self.GRID_SIZE

    @property
    def area_m2(self):

        return self.width_m * self.height_m

    # =====================================================

    def rename(self, name):

        if self.model is not None:
            self.model.name = name

        if self.geometry_changed_callback:
            self.geometry_changed_callback(self)

    # =====================================================

    def set_selected(self, selected):

        self._selected = selected

        self._update_brush()

    # =====================================================
    # Manual Simulation Sandbox path visualization -- independent of
    # selection, so a route can be highlighted while the user has
    # something else (or nothing) selected. Selection still wins
    # visually if both are true at once.
    # =====================================================

    def set_highlighted(self, highlighted):

        self._highlighted = highlighted

        self._update_brush()

    # =====================================================

    def _update_brush(self):

        if self._selected:
            self.setBrush(self.selected_brush)
        elif self._highlighted:
            self.setBrush(self.highlight_brush)
        else:
            self.setBrush(self.default_brush)

    # =====================================================

    def to_dict(self):

        if self.model is not None:
            return self.model.to_dict()

        return {
            "id": self.zone_id,
            "name": self.zone_name,
            "top_left": self.top_left,
            "top_right": self.top_right,
            "bottom_right": self.bottom_right,
            "bottom_left": self.bottom_left,
            "width": self.width_m,
            "height": self.height_m,
            "area": self.area_m2,
        }