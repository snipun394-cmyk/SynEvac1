from PyQt6.QtCore import Qt, QPointF, QRectF
from PyQt6.QtGui import QBrush, QColor, QPainterPath, QPen
from PyQt6.QtWidgets import QGraphicsItem


class FireWaterTankItem(QGraphicsItem):

    # Fire Water Supply & Suppression Infrastructure milestone -- a
    # wide rectangle body (a generic storage-tank glyph, not a vendor
    # reproduction), deliberately a different SHAPE from every sibling
    # item (a rectangle wider than tall, distinct from Manual Call
    # Point's own square body) so it reads as its own device class at
    # a glance. Own point-object plumbing, same reasoning as every
    # sibling item in this codebase for not reusing SensorItemBase.
    #
    # The outer ring shows TankOperationalState (models.fire_water_tank.
    # TankOperationalState), set by the Property Panel after calling
    # the model's own compute_state() -- never computed here.

    GRID_SIZE = 50

    HALF_WIDTH = 14
    HALF_HEIGHT = 8

    STATE_RING_COLORS = {
        "AVAILABLE": QColor(80, 200, 120),
        "LOW_LEVEL": QColor(240, 190, 60),
        "EMPTY": QColor(230, 60, 60),
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

        self.default_body_brush = QBrush(QColor(30, 90, 160))
        self.selected_body_brush = QBrush(QColor(255, 255, 0))
        self.inactive_body_brush = QBrush(QColor(130, 130, 130))

        self.default_body_pen = QPen(QColor(30, 30, 30), 2)
        self.selected_body_pen = QPen(QColor(255, 255, 0), 2)

        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)

        self.geometry_changed_callback = None

    # =====================================================

    def boundingRect(self):

        return QRectF(
            -self.HALF_WIDTH - 6, -self.HALF_HEIGHT - 6,
            (self.HALF_WIDTH + 6) * 2, (self.HALF_HEIGHT + 6) * 2,
        )

    # =====================================================

    def shape(self):

        path = QPainterPath()
        path.addRect(QRectF(-self.HALF_WIDTH, -self.HALF_HEIGHT, self.HALF_WIDTH * 2, self.HALF_HEIGHT * 2))

        return path

    # =====================================================

    def paint(self, painter, option, widget=None):

        active = self.model.active if self.model is not None else True

        painter.setBrush(
            self.selected_body_brush if self._selected
            else (self.default_body_brush if active else self.inactive_body_brush)
        )
        painter.setPen(self.selected_body_pen if self._selected else self.default_body_pen)

        painter.drawRect(QRectF(-self.HALF_WIDTH, -self.HALF_HEIGHT, self.HALF_WIDTH * 2, self.HALF_HEIGHT * 2))

        # Waterline glyph -- generic, not a vendor reproduction.
        line_pen = QPen(QColor(255, 255, 255), 2)
        painter.setPen(line_pen)
        painter.drawLine(QPointF(-self.HALF_WIDTH + 3, 0), QPointF(self.HALF_WIDTH - 3, 0))

        state_name = self.current_state if isinstance(self.current_state, str) else None
        ring_color = self.STATE_RING_COLORS.get(state_name, self.UNKNOWN_STATE_RING_COLOR)

        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(ring_color, 3))
        painter.drawRect(
            QRectF(
                -self.HALF_WIDTH - 4, -self.HALF_HEIGHT - 4,
                (self.HALF_WIDTH + 4) * 2, (self.HALF_HEIGHT + 4) * 2,
            )
        )

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
