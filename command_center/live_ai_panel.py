from typing import Optional

from PyQt6.QtWidgets import QGroupBox, QLabel, QVBoxLayout, QWidget


# =====================================================
# LiveAIPanel -- Phase 9's live AI status panel. Displays exactly the
# two signals live_system.live_ai_gateway.LiveAIPredictionSnapshot
# carries, visually and structurally distinct: the operational
# PRODUCTION_CANDIDATE bottleneck-occurrence prediction, and the
# EXPERIMENTAL evacuation-time estimate -- never presented with equal
# authority (see docs/architecture/live_ai_runtime_integration.md §6).
# No field here is ever silently retained as current once stale --
# staleness is passed in explicitly by the caller (Dashboard, comparing
# ai_prediction_timestamp against building_state_timestamp), never
# inferred by this widget itself.
# =====================================================


def _format_percent(value):

    return "-" if value is None else f"{value:.0%}"


def _format_seconds(value):

    if value is None:
        return "-"

    minutes, seconds = divmod(int(value), 60)
    return f"{minutes:02d}:{seconds:02d}"


class LiveAIPanel(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)

        operational_group = QGroupBox("Operational AI -- Bottleneck Occurrence")
        operational_layout = QVBoxLayout(operational_group)
        self.status_label = QLabel("AI prediction unavailable")
        self.status_label.setWordWrap(True)
        self.probability_label = QLabel("Bottleneck Risk Probability: -")
        self.model_label = QLabel("Model: -")
        self.timestamp_label = QLabel("Prediction Timestamp: -")
        for label in (self.status_label, self.probability_label, self.model_label, self.timestamp_label):
            operational_layout.addWidget(label)
        layout.addWidget(operational_group)

        experimental_group = QGroupBox("Experimental AI -- Evacuation-Time Estimate")
        experimental_layout = QVBoxLayout(experimental_group)
        self.experimental_status_label = QLabel(
            "Status: EXPERIMENTAL -- not validated for operational use",
        )
        self.experimental_status_label.setWordWrap(True)
        self.experimental_value_label = QLabel("Experimental Evacuation-Time Estimate: -")
        for label in (self.experimental_status_label, self.experimental_value_label):
            experimental_layout.addWidget(label)
        layout.addWidget(experimental_group)

        layout.addStretch(1)

    # =====================================================

    def show_prediction(self, ai_prediction_snapshot: Optional[object], *, stale: bool = False) -> None:

        if ai_prediction_snapshot is None:

            self.status_label.setText("AI prediction unavailable (no live_ai_gateway configured, or no cycle run yet)")
            self.probability_label.setText("Bottleneck Risk Probability: -")
            self.model_label.setText("Model: -")
            self.timestamp_label.setText("Prediction Timestamp: -")
            self.experimental_status_label.setText("Status: EXPERIMENTAL -- not validated for operational use")
            self.experimental_value_label.setText("Experimental Evacuation-Time Estimate: -")
            return

        bottleneck = ai_prediction_snapshot.bottleneck

        if bottleneck is None:

            reason = ", ".join(ai_prediction_snapshot.errors) or ai_prediction_snapshot.system_status.name
            self.status_label.setText(f"AI prediction unavailable ({reason})")
            self.probability_label.setText("Bottleneck Risk Probability: -")
            self.model_label.setText("Model: -")
            self.timestamp_label.setText("Prediction Timestamp: -")

        else:

            status_text = "STALE" if stale else ai_prediction_snapshot.system_status.name
            self.status_label.setText(f"Status: {status_text}")
            self.probability_label.setText(f"Bottleneck Risk Probability: {_format_percent(bottleneck.probability)}")
            self.model_label.setText(f"Model: {bottleneck.model_id} ({bottleneck.model_version})")
            self.timestamp_label.setText(f"Prediction Timestamp: {_format_seconds(bottleneck.timestamp)}")

        evacuation_time = ai_prediction_snapshot.evacuation_time_experimental

        if evacuation_time is None:

            self.experimental_status_label.setText("Status: EXPERIMENTAL -- unavailable this cycle")
            self.experimental_value_label.setText("Experimental Evacuation-Time Estimate: -")

        else:

            self.experimental_status_label.setText(
                "Status: EXPERIMENTAL -- not validated for operational use; do not treat as authoritative RSET",
            )
            self.experimental_value_label.setText(
                f"Experimental Evacuation-Time Estimate: {_format_seconds(evacuation_time.predicted_seconds)}",
            )
