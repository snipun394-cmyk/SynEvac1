import sys

from PyQt6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QTextEdit,
)


class TestWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Status Bar Test")
        self.resize(1000, 700)

        self.setCentralWidget(QTextEdit())

        self.status = self.statusBar()
        self.label = QLabel("X : 0.0   Y : 0.0")

        self.status.addPermanentWidget(self.label)


app = QApplication(sys.argv)

window = TestWindow()
window.show()

sys.exit(app.exec())