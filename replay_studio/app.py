import sys

from PyQt6.QtWidgets import QApplication

from command_center.main_window import MainWindow


# =====================================================
# ReplayStudioApp -- mirrors builder.app.BuilderApp exactly: a thin
# bootstrap over an existing, already-built main window, nothing else.
# This is what makes Simulation Replay Studio a genuinely separate,
# distinctly-launchable tool for researchers rather than a mode flag
# bolted onto the operator-facing Command Center's own launcher --
# while still reusing that same MainWindow/Dashboard/BuildingView code
# wholesale, as a library, exactly as builder.app.BuilderApp already
# reuses Studio's non-simulation packages instead of duplicating them.
#
# MainWindow already defaults to CommandCenterMode.REPLAY and already
# owns every Play/Pause/Stop/speed/scrub/Open-Scenario control this
# milestone needs -- there is nothing left for this class to add beyond
# the window title, which is the one honest signal to the user that
# they launched the research/replay tool, not the live operator one.
# =====================================================


class ReplayStudioApp:

    def __init__(self):

        self.app = QApplication(sys.argv)

        self.window = MainWindow()
        self.window.setWindowTitle("SynEvac Simulation Replay Studio")

    def run(self):

        self.window.show()

        sys.exit(self.app.exec())
