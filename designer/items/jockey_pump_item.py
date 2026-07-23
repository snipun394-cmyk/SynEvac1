import math

from PyQt6.QtCore import Qt, QPointF, QRectF
from PyQt6.QtGui import QBrush, QColor, QPainterPath, QPen, QPolygonF
from PyQt6.QtWidgets import QGraphicsItem


class JockeyPumpItem(QGraphicsItem):

    # Fire Water Supply & Suppression Infrastructure milestone -- the
    # SAME octagon shape family as FirePumpItem (mirroring models.
    # jockey_pump.JockeyPump's own real subclass relationship to
    # models.pump_asset.PumpAsset -- a jockey pump IS a pump), at a
    # smaller BODY_RADIUS and its own distinct color, so the two remain
    # visually distinguishable at a glance. A fully independent class,
    # not a subclass of FirePumpItem -- graphics items in this codebase
    # are never subclassed from one another (every sibling item extends
    # QGraphicsItem directly), specifically so `isinstance(item,
    # FirePumpItem)` in designer/windows/main_window.py's own selection
    # dispatch can never accidentally match a JockeyPumpItem.

    GRID_SIZE = 50

    BODY_RADIUS = 7

    STATE_RING_COLORS = {
        "STOPPED": QColor(140, 140, 140),
        "RUNNING": QColor(80, 200, 120),
        "FAULT": QColor(230, 60, 60),
        "UNAVAILABLE": QColor(140, 140, 140),
    }

    UNKNOWN_STATE_RING_COLOR = QColor(140, 140, 140)

    def __init__(self, x, y, model=None):
        super().__init__()

        self.model = model

        if self.model is not None:

            self.object_id = self.model.id
            self.object_name = self.model.name

        else:

            self.object_id = ""
            self.object_name = ""

        self.setPos(x, y)

        self._selected = False
        self.current_state = None

        self.default_body_brush = QBrush(QColor(90, 140, 200))
        self.selected_body_brush = QBrush(QColor(255, 255, 0))
        self.inactive_body_brush = QBrush(QColor(130, 130, 130))

        self.default_body_pen = QPen(QColor(30, 30, 30), 2)
        self.selected_body_pen = QPen(QColor(255, 255, 0), 2)

        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)

        self.geometry_changed_callback = None

    # =====================================================

    def _octagon(self, radius):

        points = []
        for i in range(8):
            angle = math.radians(22.5 + i * 45)
            points.append(QPointF(radius * math.cos(angle), radius * math.sin(angle)))

        return QPolygonF(points)

    # =====================================================

    def boundingRect(self):

        radius = self.BODY_RADIUS + 6

        return QRectF(-radius, -radius, radius * 2, radius * 2)

    # =====================================================

    def shape(self):

        path = QPainterPath()
        path.addPolygon(self._octagon(self.BODY_RADIUS))
        path.closeSubpath()

        return path

    # =====================================================

    def paint(self, painter, option, widget=None):

        active = self.model.active if self.model is not None else True

        painter.setBrush(
            self.selected_body_brush if self._selected
            else (self.default_body_brush if active else self.inactive_body_brush)
        )
        painter.setPen(self.selected_body_pen if self._selected else self.default_body_pen)

        painter.drawPolygon(self._octagon(self.BODY_RADIUS))

        state_name = self.current_state if isinstance(self.current_state, str) else None
        ring_color = self.STATE_RING_COLORS.get(state_name, self.UNKNOWN_STATE_RING_COLOR)

        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(ring_color, 3))
        painter.drawPolygon(self._octagon(self.BODY_RADIUS + 4))

    # =====================================================

    def itemChange(self, change, value):

        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:

            x = round(value.x() / self.GRID_SIZE) * self.GRID_SIZE
            y = round(value.y() / self.GRID_SIZE) * self.GRID_SIZE

            return QPointF(x, y)

        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:

            self.sync_to_model()

            if self.geometry_changed_callback:
                self.geometry_changed_callback(self)

        return super().itemChange(change, value)

    # =====================================================

    def refresh_geometry(self):

        self.prepareGeometryChange()
        self.update()

    # =====================================================

    def sync_to_model(self):

        if self.model is None:
            return

        self.model.position = (
            self.pos().x() / self.GRID_SIZE,
            self.pos().y() / self.GRID_SIZE,
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

        self._selected = selected
        self.update()
