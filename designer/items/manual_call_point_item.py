from PyQt6.QtCore import Qt, QPointF, QRectF
from PyQt6.QtGui import QBrush, QColor, QPainterPath, QPen
from PyQt6.QtWidgets import QGraphicsItem


class ManualCallPointItem(QGraphicsItem):

    # Manual Call Points & Emergency Lighting milestone, Phase 4 -- a
    # recognizable wall-mounted MCP/pull-station symbol: a red SQUARE
    # body (never a circle, deliberately distinct in SHAPE from Smoke/
    # HeatDetectorItem's own bodies, not just color, so it reads as a
    # genuinely different device class at a glance) with a white
    # diagonal "break glass" cross -- a generic call-point glyph, not a
    # reproduction of any real vendor product. Own point-object
    # plumbing (position sync, selection, itemChange) rather than
    # reusing designer.items.sensor_item.SensorItemBase, since that
    # base class hardcodes an ELLIPSE body/shape/paint -- the same
    # reasoning designer.items.sign_item.SignItem already gives for
    # building its own item instead of reusing SensorItemBase.
    #
    # Still shows the same "state ring" concept SensorItemBase
    # establishes for DetectorState (NORMAL/ALARM/FAULT), restated here
    # independently so this item has no import dependency on
    # SensorItemBase at all.

    GRID_SIZE = 50

    BODY_HALF_SIZE = 9

    STATE_RING_COLORS = {
        "NORMAL": QColor(80, 200, 120),
        "ALARM": QColor(230, 60, 60),
        "FAULT": QColor(240, 190, 60),
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

        # Display-only, set by the Property Panel after calling the
        # model's own compute_state() -- never computed here, mirrors
        # SensorItemBase.current_state exactly.
        self.current_state = None

        self.default_body_brush = QBrush(QColor(200, 30, 30))
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

        radius = self.BODY_HALF_SIZE + 6

        return QRectF(-radius, -radius, radius * 2, radius * 2)

    # =====================================================

    def shape(self):

        path = QPainterPath()

        path.addRect(
            QRectF(-self.BODY_HALF_SIZE, -self.BODY_HALF_SIZE, self.BODY_HALF_SIZE * 2, self.BODY_HALF_SIZE * 2)
        )

        return path

    # =====================================================

    def paint(self, painter, option, widget=None):

        active = self.model.active if self.model is not None else True

        painter.setBrush(
            self.selected_body_brush if self._selected
            else (self.default_body_brush if active else self.inactive_body_brush)
        )
        painter.setPen(self.selected_body_pen if self._selected else self.default_body_pen)

        painter.drawRect(
            QRectF(-self.BODY_HALF_SIZE, -self.BODY_HALF_SIZE, self.BODY_HALF_SIZE * 2, self.BODY_HALF_SIZE * 2)
        )

        # "Break glass" cross glyph -- generic, not a vendor reproduction.
        glass_pen = QPen(QColor(255, 255, 255), 2)
        painter.setPen(glass_pen)
        half = self.BODY_HALF_SIZE - 3
        painter.drawLine(QPointF(-half, -half), QPointF(half, half))
        painter.drawLine(QPointF(-half, half), QPointF(half, -half))

        # State ring -- outline only, same convention as SensorItemBase.
        state_name = self.current_state.name if self.current_state is not None else None
        ring_color = self.STATE_RING_COLORS.get(state_name, self.UNKNOWN_STATE_RING_COLOR)

        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(ring_color, 3))
        painter.drawRect(
            QRectF(
                -self.BODY_HALF_SIZE - 4, -self.BODY_HALF_SIZE - 4,
                (self.BODY_HALF_SIZE + 4) * 2, (self.BODY_HALF_SIZE + 4) * 2,
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
