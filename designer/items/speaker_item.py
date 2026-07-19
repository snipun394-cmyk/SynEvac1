from PyQt6.QtGui import QColor

from designer.items.sensor_item import SensorItemBase


class SpeakerItem(SensorItemBase):

    # A distinct body color from Smoke/Heat Detector (Zoned Voice
    # Evacuation & Speaker Network Framework) -- otherwise fully reuses
    # SensorItemBase's shared point-object plumbing. The "state ring"
    # SensorItemBase draws around every item's body is meaningless for a
    # Speaker (an output device has no DetectorState of its own); this
    # item simply never sets current_state, so it always renders the
    # neutral UNKNOWN_STATE_RING_COLOR ring, honestly reflecting "no
    # state to show" rather than a fabricated one.

    FILL_COLOR = QColor(255, 180, 90)

    def _fill_color(self):

        return self.FILL_COLOR
