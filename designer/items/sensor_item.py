from PyQt6.QtCore import Qt, QPointF, QRectF
from PyQt6.QtGui import QBrush, QColor, QPainterPath, QPen
from PyQt6.QtWidgets import QGraphicsItem


class SensorItemBase(QGraphicsItem):

    # Shared point-object boilerplate for every Building Sensor Network
    # graphics item (SmokeDetectorItem, HeatDetectorItem today; a
    # future FlameDetectorItem/GasDetectorItem/ManualCallPointItem
    # fits here too) -- position sync, selection, rename, and the
    # itemChange plumbing every point-object item in this package
    # (CameraItem, DetectorItem, AssemblyPointItem) already repeats
    # near-identically. Factored out here specifically because this
    # milestone is adding two new sibling items at once; existing
    # items are deliberately left untouched (this is additive, not a
    # refactor of working code).
    #
    # Subclasses supply exactly one thing: _fill_color(), their own
    # fixed body color. Everything else -- including the outer "state
    # ring" showing the sensor's last-known DetectorState -- is shared.

    GRID_SIZE = 50
    BODY_RADIUS = 8

    # DetectorState.name -> ring color. A plain string-keyed dict (not
    # importing models.sensor_asset.DetectorState itself) so this
    # rendering-only module never needs a hard dependency on the model
    # package beyond what `model` already gives it duck-typed.
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

        # Display-only hint the Property Panel sets after calling the
        # model's own compute_state() (see models/smoke_detector.py) --
        # never computed here, never persisted; None means "not yet
        # evaluated this session", rendered with a neutral ring.
        self.current_state = None

        # =====================================================
        # Appearance
        # =====================================================

        self.selected_body_brush = QBrush(QColor(255, 255, 0))
        self.selected_body_pen = QPen(QColor(255, 255, 0), 2)

        self.default_body_pen = QPen(QColor(30, 30, 30), 2)

        self.inactive_fill_color = QColor(130, 130, 130)

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
    # Subclass hook
    # =====================================================

    def _fill_color(self):

        raise NotImplementedError

    # =====================================================

    def boundingRect(self):

        radius = self.BODY_RADIUS + 6

        return QRectF(
            -radius,
            -radius,
            radius * 2,
            radius * 2,
        )

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

        painter.setBrush(
            self.selected_body_brush
            if self._selected
            else QBrush(self._fill_color() if active else self.inactive_fill_color)
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

        # State ring -- an outline only (no fill), so it reads as a
        # status indicator around the body rather than a second body.
        state_name = self.current_state.name if self.current_state is not None else None
        ring_color = self.STATE_RING_COLORS.get(state_name, self.UNKNOWN_STATE_RING_COLOR)

        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(ring_color, 3))

        painter.drawEllipse(
            QPointF(0, 0),
            self.BODY_RADIUS + 4,
            self.BODY_RADIUS + 4,
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
    # Called after the Property Panel writes Active/Health Status/
    # current_state straight onto the model/item -- none of those move
    # the item, so no itemChange fires on its own.
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
