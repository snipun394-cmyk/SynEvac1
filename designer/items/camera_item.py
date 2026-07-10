import math

from PyQt6.QtCore import QPointF, QRectF
from PyQt6.QtGui import QBrush, QColor, QPainterPath, QPen, QPolygonF
from PyQt6.QtWidgets import QGraphicsItem


class CameraItem(QGraphicsItem):

    GRID_SIZE = 50

    BODY_RADIUS = 10

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
            self.setRotation(self.model.rotation)

        self._selected = False

        # =====================================================
        # Appearance
        # =====================================================

        self.default_body_brush = QBrush(QColor(40, 40, 40))
        self.selected_body_brush = QBrush(QColor(255, 255, 0))

        self.default_body_pen = QPen(QColor(220, 220, 220), 2)
        self.selected_body_pen = QPen(QColor(255, 255, 0), 2)

        self.active_cone_brush = QBrush(QColor(0, 200, 255, 60))
        self.inactive_cone_brush = QBrush(QColor(120, 120, 120, 40))
        self.selected_cone_brush = QBrush(QColor(255, 255, 0, 70))

        self.active_cone_pen = QPen(QColor(0, 200, 255, 160), 1)
        self.inactive_cone_pen = QPen(QColor(120, 120, 120, 160), 1)
        self.selected_cone_pen = QPen(QColor(255, 255, 0, 200), 1)

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
    # Coverage Geometry (derived)
    #
    # Computed in local space, always pointing along local +x --
    # Qt's own rotation transform (setRotation) aims the sector,
    # so this never needs to know the camera's facing angle
    # itself. Mirrors Camera.coverage_polygon() in models/camera.py,
    # but works in scene pixels instead of meters.
    # =====================================================

    def _fov(self):

        if self.model is None:
            return 90.0

        return self.model.horizontal_fov

    def _range_px(self):

        if self.model is None:
            return 25.0 * self.GRID_SIZE

        return self.model.max_range * self.GRID_SIZE

    def _coverage_polygon(self, segments=24):

        half_fov = self._fov() / 2

        radius = self._range_px()

        polygon = QPolygonF()

        polygon.append(QPointF(0, 0))

        for i in range(segments + 1):

            angle_degrees = (
                -half_fov
                + (self._fov() * i / segments)
            )

            angle_radians = math.radians(angle_degrees)

            polygon.append(
                QPointF(
                    radius * math.cos(angle_radians),
                    radius * math.sin(angle_radians),
                )
            )

        return polygon

    # =====================================================

    def boundingRect(self):

        radius = max(
            self._range_px(),
            self.BODY_RADIUS,
        ) + 4

        return QRectF(
            -radius,
            -radius,
            radius * 2,
            radius * 2,
        )

    # =====================================================
    # Hit-testing only the camera body -- keeps overlapping
    # coverage cones from stealing clicks meant for whatever
    # is drawn underneath them.
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

        active = (
            self.model.active
            if self.model is not None
            else True
        )

        if self._selected:
            cone_brush = self.selected_cone_brush
            cone_pen = self.selected_cone_pen
        elif active:
            cone_brush = self.active_cone_brush
            cone_pen = self.active_cone_pen
        else:
            cone_brush = self.inactive_cone_brush
            cone_pen = self.inactive_cone_pen

        painter.setBrush(cone_brush)
        painter.setPen(cone_pen)

        painter.drawPolygon(
            self._coverage_polygon()
        )

        painter.setBrush(
            self.selected_body_brush
            if self._selected
            else self.default_body_brush
        )

        painter.setPen(
            self.selected_body_pen
            if self._selected
            else self.default_body_pen
        )

        painter.drawEllipse(
            QPointF(0, 0),
            self.BODY_RADIUS,
            self.BODY_RADIUS,
        )

        # Facing indicator -- small triangle pointing along local
        # +x, the direction the rotation transform aims the cone.
        facing = QPolygonF(
            [
                QPointF(self.BODY_RADIUS, 0),
                QPointF(self.BODY_RADIUS + 10, -6),
                QPointF(self.BODY_RADIUS + 10, 6),
            ]
        )

        painter.drawPolygon(facing)

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

    def set_rotation_degrees(self, degrees):

        self.prepareGeometryChange()

        self.setRotation(degrees)

        if self.model is not None:
            self.model.rotation = degrees

        if self.geometry_changed_callback:
            self.geometry_changed_callback(self)

    # =====================================================
    # Called after the Property Panel writes FOV/Range/Mount
    # Height/Active straight onto the model -- none of those
    # move the item, so no itemChange fires on its own and the
    # coverage sector needs an explicit repaint.
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
