import sys

from PyQt6.QtWidgets import QApplication

from designer.windows.main_window import MainWindow


class SynEvacApp:

    def __init__(self):

        self.app = QApplication(sys.argv)

        self.window = MainWindow()

    def run(self):

        self.window.show()

        print("\n========== OBJECT TREE ==========\n")
        self.window.dumpObjectTree()
        print("\n=================================\n")

        sys.exit(self.app.exec())