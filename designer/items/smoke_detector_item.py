from PyQt6.QtGui import QColor

from designer.items.sensor_item import SensorItemBase


class SmokeDetectorItem(SensorItemBase):

    # Same body color DetectorItem.TYPE_COLORS already uses for
    # "Smoke", kept visually consistent with the pre-existing generic
    # Detector tool.

    FILL_COLOR = QColor(150, 170, 255)

    def _fill_color(self):

        return self.FILL_COLOR
