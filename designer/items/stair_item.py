from PyQt6.QtCore import QPointF, QRectF
from PyQt6.QtGui import QBrush, QColor, QPainterPath, QPen
from PyQt6.QtWidgets import QGraphicsItem


class StairItem(QGraphicsItem):

    GRID_SIZE = 50

    BODY_RADIUS = 8

    # A Staircase is one shared object rendered as two markers --
    # this is the "from" (entrance) marker on from_floor_id, or
    # the "to" (landing) marker on to_floor_id. Both markers point
    # at the same model; itemChange only ever writes back to the
    # position field this particular marker represents.
    def __init__(self, x, y, width, role, model=None):
        super().__init__()

        self.model = model
        self.role = role

        if self.model is not None:

            self.object_id = self.model.id
            self.object_name = self.model.name

        else:

            self.object_id = ""
            self.object_name = ""

        self.setPos(x, y)

        self._selected = False
        self._highlighted = False

        # =====================================================
        # Appearance
        # =====================================================

        self.default_brush = QBrush(QColor(40, 40, 40))
        self.selected_brush = QBrush(QColor(255, 255, 0))
        self.highlight_brush = QBrush(QColor(255, 165, 0))

        self.default_pen = QPen(QColor(180, 120, 40), 4)
        self.selected_pen = QPen(QColor(255, 255, 0), 4)
        self.highlight_pen = QPen(QColor(255, 165, 0), 4)

        self.default_body_pen = QPen(QColor(220, 220, 220), 2)
        self.selected_body_pen = QPen(QColor(255, 255, 0), 2)

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
    # Width Indicator (derived, local space)
    #
    # A short horizontal tick centered on the origin, length =
    # Width in meters -- the physical width of the stair at this
    # end. There is no rotation concept here (unlike Camera), so
    # it is always drawn along local +x.
    # =====================================================

    def _width_px(self):

        if self.model is None:
            return 1.5 * self.GRID_SIZE

        return self.model.width * self.GRID_SIZE

    # =====================================================

    def boundingRect(self):

        half_width = self._width_px() / 2

        margin = self.BODY_RADIUS + 4

        radius = half_width + margin

        return QRectF(
            -radius,
            -radius,
            radius * 2,
            radius * 2,
        )

    # =====================================================
    # Hit-testing only the anchor body -- keeps the width tick
    # from stealing clicks meant for whatever is drawn under it.
    # =====================================================

    def shape(self):

        path = QPainterPath()

        path.addEllipse(
            QPointF(0, 0),
            self.BODY_RADIUS,
            self.BODY_RADIUS,
        )

        return path

    # =====================================================

    def paint(self, painter, option, widget=None):

        half_width = self._width_px() / 2

        if self._selected:
            line_pen, body_brush, body_pen = (
                self.selected_pen, self.selected_brush, self.selected_body_pen,
            )
        elif self._highlighted:
            line_pen, body_brush, body_pen = (
                self.highlight_pen, self.highlight_brush, self.selected_body_pen,
            )
        else:
            line_pen, body_brush, body_pen = (
                self.default_pen, self.default_brush, self.default_body_pen,
            )

        painter.setPen(line_pen)

        painter.drawLine(
            QPointF(-half_width, 0),
            QPointF(half_width, 0),
        )

        painter.setBrush(body_brush)
        painter.setPen(body_pen)

        painter.drawEllipse(
            QPointF(0, 0),
            self.BODY_RADIUS,
            self.BODY_RADIUS,
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

                self.geometry_changed_callback(
                    self
                )

        return super().itemChange(change, value)

    # =====================================================

    def sync_to_model(self):

        if self.model is None:
            return

        position = (
            self.pos().x() / self.GRID_SIZE,
            self.pos().y() / self.GRID_SIZE,
        )

        if self.role == "from":
            self.model.from_position = position
        else:
            self.model.to_position = position

        self.object_name = self.model.name

    # =====================================================
    # Called after the Property Panel writes Width (or the other
    # end's position) straight onto the model -- geometry that
    # doesn't move this item's own position, so no itemChange
    # fires on its own.
    # =====================================================

    def refresh_geometry(self):

        self.prepareGeometryChange()

        self.update()

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

        self.update()

    # =====================================================

    def set_highlighted(self, highlighted):

        self._highlighted = highlighted

        self.update()
