from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
)


class BottomInfoBar(QFrame):

    def __init__(self):
        super().__init__()

        self.setObjectName("BottomInfoBar")

        # Tall enough that if the labels below ever wrap (narrow
        # window + word wrap engaging), text stays fully visible
        # instead of clipped, at 1366x768 and wider. Below that,
        # a label may still clip slightly in the rare case.
        self.setFixedHeight(58)

        self.setStyleSheet("""
        QFrame#BottomInfoBar{
            background:#2b2b2b;
            border-top:1px solid #505050;
        }

        QLabel{
            color:white;
            padding-left:4px;
            padding-right:4px;
        }
        """)

        layout = QHBoxLayout()

        layout.setContentsMargins(
            8,
            0,
            8,
            0,
        )

        layout.setSpacing(10)

        # =====================================================
        # Left
        # =====================================================

        self.tool_label = QLabel(
            "Tool : Select"
        )

        self.coord_label = QLabel(
            "X : 0.00 m  Y : 0.00 m"
        )

        self.zoom_label = QLabel(
            "Zoom : 100%"
        )

        self.grid_label = QLabel(
            "Grid : 1.0 m"
        )

        self.floor_label = QLabel(
            "Floor : Ground Floor"
        )

        layout.addWidget(self.tool_label)
        layout.addWidget(self.coord_label)
        layout.addWidget(self.zoom_label)
        layout.addWidget(self.grid_label)
        layout.addWidget(self.floor_label)

        layout.addStretch()

        # =====================================================
        # Right
        # =====================================================

        self.selection_label = QLabel(
            "Selection : None"
        )

        self.scale_label = QLabel(
            "Scale : Not Calibrated"
        )

        layout.addWidget(self.selection_label)
        layout.addWidget(self.scale_label)

        # A QLabel's minimum width defaults to its full text
        # width (minimumSizeHint == sizeHint when word wrap is
        # off), and QHBoxLayout can never shrink a child below
        # its minimum -- with 7 labels that summed to a ~1700px
        # hard floor on the whole MainWindow. Word wrap makes
        # minimumSizeHint based on the longest single word
        # instead, while sizeHint (the normal single-line
        # display at typical widths) is unchanged.
        for label in (
            self.tool_label,
            self.coord_label,
            self.zoom_label,
            self.grid_label,
            self.floor_label,
            self.selection_label,
            self.scale_label,
        ):

            label.setWordWrap(True)

        self.setLayout(layout)

    # =====================================================

    def update_coordinates(self, x, y):

        self.coord_label.setText(
            f"X : {x:.2f} m  Y : {y:.2f} m"
        )

    # =====================================================

    def update_tool(self, tool):

        self.tool_label.setText(
            f"Tool : {tool.capitalize()}"
        )

    # =====================================================

    def update_zoom(self, zoom):

        self.zoom_label.setText(
            f"Zoom : {zoom}%"
        )

    # =====================================================

    def update_floor(self, floor):

        self.floor_label.setText(
            f"Floor : {floor}"
        )

    # =====================================================

    def update_scale(self, scale):

        self.scale_label.setText(
            f"Scale : {scale}"
        )

    # =====================================================

    def update_selection(self, text):

        self.selection_label.setText(
            f"Selection : {text}"
        )