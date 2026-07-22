from PyQt6.QtCore import QPointF, QRectF
from PyQt6.QtGui import QBrush, QColor, QPainterPath, QPen, QPolygonF
from PyQt6.QtWidgets import QGraphicsItem


class SignItem(QGraphicsItem):

    # Live Dynamic Evacuation Signage milestone, Phase 19/20 -- the
    # Designer's own click-to-place Dynamic Evacuation Sign asset.
    # Mirrors designer.items.camera_item.CameraItem's own orientation-
    # indicator plumbing (setRotation/facing triangle) exactly, since a
    # Sign is an oriented point object the same way a Camera is --
    # WITHOUT a coverage cone (a sign has no detection geometry to
    # draw, only a facing direction). This is design-time orientation
    # visualization ONLY (Phase 20) -- no live evacuation arrows are
    # rendered here; that is dynamic_signage/command_center's own
    # runtime concern, entirely separate from this static Designer item.

    GRID_SIZE = 50

    BODY_HALF_WIDTH = 9
    BODY_HALF_HEIGHT = 9

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

        if self.model is not None:
            self.setRotation(self.model.orientation)

        self._selected = False

        # =====================================================
        # Appearance -- a distinct green body (evacuation-signage
        # green, matching real-world exit-sign convention), never
        # confused with Camera's dark body or Speaker's amber body.
        # =====================================================

        self.default_body_brush = QBrush(QColor(30, 140, 60))
        self.selected_body_brush = QBrush(QColor(255, 255, 0))
        self.inactive_body_brush = QBrush(QColor(130, 130, 130))

        self.default_body_pen = QPen(QColor(220, 220, 220), 2)
        self.selected_body_pen = QPen(QColor(255, 255, 0), 2)

        # =====================================================
        # Flags
        # =====================================================

        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)

        self.geometry_changed_callback = None

    # =====================================================

    def boundingRect(self):

        radius = max(self.BODY_HALF_WIDTH, self.BODY_HALF_HEIGHT) + 14

        return QRectF(-radius, -radius, radius * 2, radius * 2)

    # =====================================================

    def shape(self):

        path = QPainterPath()

        path.addRect(
            QRectF(-self.BODY_HALF_WIDTH, -self.BODY_HALF_HEIGHT, self.BODY_HALF_WIDTH * 2, self.BODY_HALF_HEIGHT * 2)
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
            QRectF(-self.BODY_HALF_WIDTH, -self.BODY_HALF_HEIGHT, self.BODY_HALF_WIDTH * 2, self.BODY_HALF_HEIGHT * 2)
        )

        # Facing indicator -- small triangle pointing along local +x,
        # the direction the rotation transform aims (same convention,
        # same visual shape, as CameraItem's own facing triangle).
        facing = QPolygonF([
            QPointF(self.BODY_HALF_WIDTH, 0),
            QPointF(self.BODY_HALF_WIDTH + 12, -7),
            QPointF(self.BODY_HALF_WIDTH + 12, 7),
        ])

        painter.drawPolygon(facing)

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

    def set_orientation_degrees(self, degrees):

        self.prepareGeometryChange()

        self.setRotation(degrees)

        if self.model is not None:
            self.model.orientation = degrees

        if self.geometry_changed_callback:
            self.geometry_changed_callback(self)

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
