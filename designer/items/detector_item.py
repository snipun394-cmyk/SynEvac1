from PyQt6.QtCore import QPointF, QRectF
from PyQt6.QtGui import QBrush, QColor, QPainterPath, QPen
from PyQt6.QtWidgets import QGraphicsItem


class DetectorItem(QGraphicsItem):

    GRID_SIZE = 50

    BODY_RADIUS = 8

    # Body/coverage tint per Detector Type -- purely visual, so a
    # glance at the canvas tells you what kind of hazard a given
    # detector watches for.
    TYPE_COLORS = {
        "Smoke": QColor(150, 170, 255),
        "Heat": QColor(255, 150, 60),
        "Flame": QColor(255, 70, 70),
        "Gas": QColor(140, 255, 100),
    }

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

        # =====================================================
        # Appearance
        # =====================================================

        self.selected_body_brush = QBrush(QColor(255, 255, 0))
        self.selected_body_pen = QPen(QColor(255, 255, 0), 2)

        self.default_body_pen = QPen(QColor(30, 30, 30), 2)

        self.selected_coverage_brush = QBrush(QColor(255, 255, 0, 70))
        self.selected_coverage_pen = QPen(QColor(255, 255, 0, 200), 1)

        self.inactive_coverage_brush = QBrush(QColor(120, 120, 120, 40))
        self.inactive_coverage_pen = QPen(QColor(120, 120, 120, 160), 1)

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
    # Local space, centered on the origin -- there is no rotation
    # concept for Detector. Mirrors Detector.coverage_circle() in
    # models/detector.py, but works in scene pixels instead of
    # meters.
    # =====================================================

    def _coverage_radius_px(self):

        if self.model is None:
            return 3.0 * self.GRID_SIZE

        return self.model.coverage_radius * self.GRID_SIZE

    def _coverage_rect(self):

        radius = self._coverage_radius_px()

        return QRectF(
            -radius,
            -radius,
            radius * 2,
            radius * 2,
        )

    # =====================================================

    def _type_color(self):

        detector_type = (
            self.model.detector_type
            if self.model is not None
            else "Smoke"
        )

        return self.TYPE_COLORS.get(
            detector_type,
            QColor(200, 200, 200),
        )

    # =====================================================

    def boundingRect(self):

        margin = max(self.BODY_RADIUS, 4)

        radius = self._coverage_radius_px() + margin

        return QRectF(
            -radius,
            -radius,
            radius * 2,
            radius * 2,
        )

    # =====================================================
    # Hit-testing only the detector body -- keeps overlapping
    # coverage circles from stealing clicks meant for whatever is
    # drawn underneath them.
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

        type_color = self._type_color()

        if self._selected:
            coverage_brush = self.selected_coverage_brush
            coverage_pen = self.selected_coverage_pen
        elif active:
            coverage_brush = QBrush(
                QColor(
                    type_color.red(),
                    type_color.green(),
                    type_color.blue(),
                    60,
                )
            )
            coverage_pen = QPen(
                QColor(
                    type_color.red(),
                    type_color.green(),
                    type_color.blue(),
                    160,
                ),
                1,
            )
        else:
            coverage_brush = self.inactive_coverage_brush
            coverage_pen = self.inactive_coverage_pen

        painter.setBrush(coverage_brush)
        painter.setPen(coverage_pen)

        painter.drawEllipse(
            self._coverage_rect()
        )

        painter.setBrush(
            self.selected_body_brush
            if self._selected
            else QBrush(type_color)
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
    # Called after the Property Panel writes Coverage Radius/
    # Mount Height/Detector Type/Active straight onto the model --
    # none of those move the item, so no itemChange fires on its
    # own and the coverage circle needs an explicit repaint.
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
