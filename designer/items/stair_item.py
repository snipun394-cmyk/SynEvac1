from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QPainterPath, QPen, QPolygonF
from PyQt6.QtWidgets import QGraphicsItem


class StairItem(QGraphicsItem):

    GRID_SIZE = 50

    # =====================================================
    # Symbol layout (local space, fixed regardless of Width --
    # Width is still a real, editable model field, it just isn't
    # visualized here; a horizontal tick sized to it read as a
    # confusing dumbbell shape and didn't communicate direction
    # or floor identity, which is what this symbol needs to show).
    #
    #      ^          <- arrowhead (points away from marker: "up")
    #      |          <- arrow shaft (fixed, same for both directions)
    #      o          <- anchor marker (the actual position)
    #      :
    #      :          <- dashed vertical guide (fixed length)
    # =====================================================

    MARKER_RADIUS = 7

    ARROW_TOP_Y = -26
    ARROW_BOTTOM_Y = -12
    ARROWHEAD_HALF = 5

    GUIDE_START_Y = MARKER_RADIUS + 4
    GUIDE_LENGTH = 26

    # A Staircase is one shared object rendered as two markers --
    # this is the "from" (entrance) marker on from_floor_id, or
    # the "to" (landing) marker on to_floor_id. Both markers point
    # at the same model; itemChange only ever writes back to the
    # position field this particular marker represents. `building`
    # is needed to derive which of the two connected floors is
    # physically lower/higher (see _direction()) -- never stored,
    # resolved fresh on every repaint, same as every other derived
    # value in this codebase.
    def __init__(self, x, y, role, building, model=None):
        super().__init__()

        self.model = model
        self.role = role
        self.building = building

        if self.model is not None:

            self.object_id = self.model.id
            self.object_name = self.model.name

        else:

            self.object_id = ""
            self.object_name = ""

        self.setPos(x, y)

        self._selected = False

        # =====================================================
        # Appearance -- one color for the whole symbol (marker,
        # arrow, guide) so it reads as one object, not several.
        # =====================================================

        self.default_color = QColor(180, 120, 40)
        self.selected_color = QColor(255, 255, 0)

        self.default_brush = QBrush(self.default_color)
        self.selected_brush = QBrush(self.selected_color)

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
    # Direction (derived, never stored, never user-editable)
    #
    # Compares THIS marker's own floor elevation against the
    # OTHER end's floor elevation -- not a hardcoded "from is
    # always lower" assumption, since the guided Stair Tool does
    # not restrict which direction a stair is drawn. Returns
    # "up" if the staircase continues upward from here, "down"
    # if it continues downward, or None if there's nothing to
    # compare yet (not connected, or an unresolvable floor).
    # =====================================================

    def _direction(self):

        if self.model is None or self.building is None:
            return None

        if self.role == "from":
            my_floor_id = self.model.from_floor_id
            other_floor_id = self.model.to_floor_id
        else:
            my_floor_id = self.model.to_floor_id
            other_floor_id = self.model.from_floor_id

        my_floor = self.building.get_floor(my_floor_id)
        other_floor = self.building.get_floor(other_floor_id)

        if my_floor is None or other_floor is None:
            return None

        my_elevation = self.building.floor_elevation(my_floor)
        other_elevation = self.building.floor_elevation(other_floor)

        if other_elevation > my_elevation:
            return "up"

        if other_elevation < my_elevation:
            return "down"

        return None

    # =====================================================

    def boundingRect(self):

        half_width = self.ARROWHEAD_HALF + 3

        top = self.ARROW_TOP_Y - 4
        bottom = self.GUIDE_START_Y + self.GUIDE_LENGTH + 4

        return QRectF(
            -half_width,
            top,
            half_width * 2,
            bottom - top,
        )

    # =====================================================
    # Hit-testing only the anchor marker -- keeps the arrow/
    # guide from stealing clicks meant for whatever is under them.
    # =====================================================

    def shape(self):

        path = QPainterPath()

        path.addEllipse(
            QPointF(0, 0),
            self.MARKER_RADIUS,
            self.MARKER_RADIUS,
        )

        return path

    # =====================================================

    def paint(self, painter, option, widget=None):

        color = (
            self.selected_color
            if self._selected
            else self.default_color
        )

        direction = self._direction()

        if direction is not None:

            self._paint_arrow(painter, color, direction)
            self._paint_guide(painter, color)

        painter.setBrush(
            self.selected_brush
            if self._selected
            else self.default_brush
        )

        painter.setPen(
            QPen(color, 2)
        )

        painter.drawEllipse(
            QPointF(0, 0),
            self.MARKER_RADIUS,
            self.MARKER_RADIUS,
        )

    # =====================================================
    # Arrow -- fixed shaft, same footprint for both directions;
    # only the arrowhead end/orientation changes. "up" points
    # away from the marker (the stair rises from here). "down"
    # points toward the marker (the stair descends into here).
    # =====================================================

    def _paint_arrow(self, painter, color, direction):

        pen = QPen(color, 3)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)

        painter.setPen(pen)
        painter.setBrush(QBrush(color))

        painter.drawLine(
            QPointF(0, self.ARROW_BOTTOM_Y),
            QPointF(0, self.ARROW_TOP_Y),
        )

        if direction == "up":
            tip_y = self.ARROW_TOP_Y
            apex_y = tip_y - self.ARROWHEAD_HALF
        else:
            tip_y = self.ARROW_BOTTOM_Y
            apex_y = tip_y + self.ARROWHEAD_HALF

        arrowhead = QPolygonF(
            [
                QPointF(0, apex_y),
                QPointF(-self.ARROWHEAD_HALF, tip_y),
                QPointF(self.ARROWHEAD_HALF, tip_y),
            ]
        )

        painter.drawPolygon(arrowhead)

    # =====================================================
    # Dashed vertical guide -- purely a visual hint that the two
    # markers belong to one staircase. Fixed length, stays inside
    # this floor's own view; it is not, and must never become, an
    # actual cross-scene connection.
    # =====================================================

    def _paint_guide(self, painter, color):

        guide_pen = QPen(color, 2, Qt.PenStyle.DashLine)

        painter.setPen(guide_pen)

        painter.drawLine(
            QPointF(0, self.GUIDE_START_Y),
            QPointF(0, self.GUIDE_START_Y + self.GUIDE_LENGTH),
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
    # Called after the Property Panel writes Width, the other
    # end's position, or To Floor straight onto the model --
    # none of those move this item's own position, so no
    # itemChange fires on its own, and a To Floor change can
    # flip this marker's arrow direction.
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
