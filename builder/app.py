import sys

from PyQt6.QtWidgets import QApplication

from builder.windows.builder_main_window import BuilderMainWindow


class BuilderApp:

    # Mirrors core.app.SynEvacApp exactly -- a thin bootstrap over its
    # own main window, nothing else. This is what makes Builder a
    # genuinely separate executable rather than a mode flag: BuilderApp
    # never imports core.app.SynEvacApp or designer.windows.main_window.
    # MainWindow, so launching it can never pull in any Simulation/AI/
    # Perception/Live-Runtime module Studio's own MainWindow eagerly
    # constructs.

    def __init__(self):

        self.app = QApplication(sys.argv)

        self.window = BuilderMainWindow()

    def run(self):

        self.window.show()

        sys.exit(self.app.exec())
