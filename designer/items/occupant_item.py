from PyQt6.QtCore import QPointF, QRectF
from PyQt6.QtGui import QBrush, QColor, QPen
from PyQt6.QtWidgets import QGraphicsItem

from navigation.edge import Edge
from sandbox.occupant import SandboxOccupantState


class OccupantItem(QGraphicsItem):

    # The Manual Simulation Sandbox's on-canvas marker for one
    # SandboxOccupant -- a small filled circle, deliberately not a
    # QGraphicsRectItem/line like the engineering objects: an
    # Occupant isn't part of the Building Model, and nothing about it
    # is user-editable geometry the way a Zone/Door/Camera's shape is.
    #
    # Not movable (no ItemIsMovable, no itemChange position sync) --
    # unlike every engineering item, an Occupant's position is owned
    # entirely by SandboxManager's routing/movement logic, never by
    # dragging it on screen. Selectable only, so clicking one still
    # shows its properties in the Property Panel.
    #
    # `occupant` (a sandbox.occupant.SandboxOccupant) plays the same
    # role `model` plays on every other item, just named differently
    # to make the distinction from an engineering model obvious at a
    # glance.

    GRID_SIZE = 50

    BODY_RADIUS = 7

    STATE_COLORS = {
        SandboxOccupantState.IDLE: QColor(200, 200, 200),
        SandboxOccupantState.ROUTED: QColor(80, 160, 255),
        SandboxOccupantState.MOVING: QColor(0, 220, 120),
        SandboxOccupantState.ARRIVED: QColor(0, 140, 70),
        SandboxOccupantState.UNREACHABLE: QColor(220, 60, 60),
    }

    def __init__(self, occupant=None):
        super().__init__()

        self.occupant = occupant

        self.object_id = occupant.occupant_id if occupant is not None else ""
        self.object_name = occupant.name if occupant is not None else ""

        self._selected = False

        self.selected_brush = QBrush(QColor(255, 255, 0))
        self.default_pen = QPen(QColor(20, 20, 20), 1.5)
        self.selected_pen = QPen(QColor(255, 255, 0), 2)

        self.setFlag(
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable,
            True,
        )

        self.setZValue(500)

        # Set (unconditionally, for every selected item type) by
        # MainWindow.on_selection_changed() -- same convention every
        # other item already follows.
        self.geometry_changed_callback = None

        self.sync_from_occupant()

    # =====================================================

    def boundingRect(self):

        radius = self.BODY_RADIUS + 4

        return QRectF(-radius, -radius, radius * 2, radius * 2)

    # =====================================================

    def paint(self, painter, option, widget=None):

        state = (
            self.occupant.state
            if self.occupant is not None
            else SandboxOccupantState.IDLE
        )

        painter.setBrush(
            self.selected_brush
            if self._selected
            else QBrush(self.STATE_COLORS.get(state, QColor(200, 200, 200)))
        )

        painter.setPen(
            self.selected_pen
            if self._selected
            else self.default_pen
        )

        painter.drawEllipse(
            QPointF(0, 0),
            self.BODY_RADIUS,
            self.BODY_RADIUS,
        )

    # =====================================================
    # Called after every SandboxManager.place_occupant()/step()/
    # tick() call that touches this occupant -- position and state
    # (hence color) both come from the occupant, never the other way
    # around.
    # =====================================================

    def sync_from_occupant(self):

        if self.occupant is None:
            return

        self.prepareGeometryChange()

        x_m, y_m = self._render_position()

        self.setPos(
            x_m * self.GRID_SIZE,
            y_m * self.GRID_SIZE,
        )

        self.update()

    # =====================================================
    # occupant.position itself deliberately holds still for the whole
    # duration of a Stair edge (see SandboxManager._sync_position --
    # a Stair's two ends live in different floors' coordinate spaces,
    # so there is no single meaningful position to compute mid-edge).
    # That's correct for routing/state, but left on its own it makes a
    # occupant look frozen rather than "entering the stairs" for
    # however long the crossing takes. This is a display-only easing
    # toward the Stair's own on-floor landing marker as edge_progress
    # advances -- occupant.position, the Route, and every other piece
    # of simulation state are untouched.
    # =====================================================

    def _render_position(self):

        occupant = self.occupant
        route = occupant.route

        if route is None:
            return occupant.position

        node_index = occupant.node_index

        if node_index < len(route.edges):

            edge = route.edges[node_index]

            if edge.edge_type == Edge.STAIR:

                landing = self._stair_landing(edge.reference, occupant.floor_id)

                if landing is not None:

                    start_x, start_y = occupant.position
                    t = occupant.edge_progress

                    return (
                        start_x + (landing[0] - start_x) * t,
                        start_y + (landing[1] - start_y) * t,
                    )

        # Just arrived (edge_progress is reset to 0.0 the instant
        # _advance_one_node() fires) and the edge just completed was a
        # Stair -- show the occupant standing at this floor's landing
        # first, rather than snapping straight to the Zone center
        # normal movement resumes from.
        if node_index > 0 and occupant.edge_progress == 0.0:

            previous_edge = route.edges[node_index - 1]

            if previous_edge.edge_type == Edge.STAIR:

                landing = self._stair_landing(previous_edge.reference, occupant.floor_id)

                if landing is not None:
                    return landing

        return occupant.position

    # =====================================================

    def _stair_landing(self, stair, floor_id):

        if stair is None:
            return None

        if floor_id == stair.from_floor_id:
            return stair.from_position

        if floor_id == stair.to_floor_id:
            return stair.to_position

        return None

    # =====================================================
    # Property Panel's Name field is shared across every object type
    # and always calls current_item.rename() unconditionally -- an
    # Occupant supports it too, same as every engineering item, even
    # though the milestone doesn't otherwise ask for it.
    # =====================================================

    def rename(self, name):

        self.object_name = name

        if self.occupant is not None:
            self.occupant.name = name

        if self.geometry_changed_callback:
            self.geometry_changed_callback(self)

    # =====================================================

    def set_selected(self, selected):

        self._selected = selected

        self.update()
