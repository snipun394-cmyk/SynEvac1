from typing import Sequence

from PyQt6.QtWidgets import QListWidget, QVBoxLayout, QWidget


# =====================================================
# LiveEventsPanel -- Phase 15's bounded recent-event list. Live-only;
# the full, unbounded RecommendationTimelinePanel remains Replay-only
# (never duplicated here). Shows exactly the already-formatted strings
# live_system.live_command_center_gateway.LiveCommandCenterDataSource
# supplies -- this widget never inspects an Event/EventBus itself, and
# never fabricates an entry when no event history is available.
# =====================================================


class LiveEventsPanel(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        self.list_widget = QListWidget()
        layout.addWidget(self.list_widget)

    # =====================================================

    def show_recent_events(self, events: Sequence[str]) -> None:

        self.list_widget.clear()

        if not events:
            self.list_widget.addItem("No recent events.")
            return

        # Most-recent-last input (see LiveCommandCenterDataSource._recent_events()'s
        # own docstring) -- shown most-recent-first, the natural reading
        # order for an operator glancing at "what just happened."
        for entry in reversed(events):
            self.list_widget.addItem(entry)
