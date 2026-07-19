from PyQt6.QtGui import QColor

from designer.items.sensor_item import SensorItemBase


class HeatDetectorItem(SensorItemBase):

    # Same body color DetectorItem.TYPE_COLORS already uses for "Heat",
    # kept visually consistent with the pre-existing generic Detector
    # tool.

    FILL_COLOR = QColor(255, 150, 60)

    def _fill_color(self):

        return self.FILL_COLOR
