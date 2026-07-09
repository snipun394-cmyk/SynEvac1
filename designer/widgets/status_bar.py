from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QLabel, QStatusBar


class MainStatusBar(QStatusBar):

    def __init__(self):
        super().__init__()

        self.setSizeGripEnabled(False)
        self.setFixedHeight(28)

        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(45, 45, 45))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(255, 255, 255))
        self.setPalette(palette)
        self.setAutoFillBackground(True)

        self.coord_label = QLabel("X : 0.00 m     Y : 0.00 m")
        self.coord_label.setStyleSheet(
            "color: white; background: transparent;"
        )

        self.addPermanentWidget(self.coord_label)

    def update_coordinates(self, x, y):
        self.coord_label.setText(
            f"X : {x:.2f} m     Y : {y:.2f} m"
        )