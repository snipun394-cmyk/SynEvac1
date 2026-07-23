from PyQt6.QtCore import Qt, QPointF, QRectF
from PyQt6.QtGui import QBrush, QColor, QPainterPath, QPen, QPolygonF
from PyQt6.QtWidgets import QGraphicsItem


class FireHydrantItem(QGraphicsItem):

    # Fire Suppression & Water-Based Safety Asset Digital Twin
    # milestone -- a pentagon body (a generic landing-valve/hydrant
    # outlet glyph, not a vendor reproduction), deliberately a
    # different SHAPE from every sibling item so it reads as its own
    # device class at a glance. Own point-object plumbing, same
    # reasoning as every sibling item in this milestone.
    #
    # The outer ring shows AVAILABLE/UNAVAILABLE/FAULT, set by the
    # Property Panel after calling the model's own
    # compute_availability() -- never computed here.

    GRID_SIZE = 50

    BODY_RADIUS = 10

    AVAILABILITY_RING_COLORS = {
        "AVAILABLE": QColor(80, 200, 120),
        "UNAVAILABLE": QColor(140, 140, 140),
        "FAULT": QColor(240, 190, 60),
    }

    UNKNOWN_AVAILABILITY_RING_COLOR = QColor(140, 140, 140)

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
        self.current_availability = None

        self.default_body_brush = QBrush(QColor(210, 60, 20))
        self.selected_body_brush = QBrush(QColor(255, 255, 0))
        self.inactive_body_brush = QBrush(QColor(130, 130, 130))

        self.default_body_pen = QPen(QColor(30, 30, 30), 2)
        self.selected_body_pen = QPen(QColor(255, 255, 0), 2)

        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)

        self.geometry_changed_callback = None

    # =====================================================

    def _pentagon(self, radius):

        import math

        points = []
        for i in range(5):
            angle = math.radians(-90 + i * 72)
            points.append(QPointF(radius * math.cos(angle), radius * math.sin(angle)))

        return QPolygonF(points)

    # =====================================================

    def boundingRect(self):

        radius = self.BODY_RADIUS + 6

        return QRectF(-radius, -radius, radius * 2, radius * 2)

    # =====================================================

    def shape(self):

        path = QPainterPath()
        path.addPolygon(self._pentagon(self.BODY_RADIUS))
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

        painter.drawPolygon(self._pentagon(self.BODY_RADIUS))

        ring_color = self.AVAILABILITY_RING_COLORS.get(self.current_availability, self.UNKNOWN_AVAILABILITY_RING_COLOR)

        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(ring_color, 3))
        painter.drawPolygon(self._pentagon(self.BODY_RADIUS + 4))

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
