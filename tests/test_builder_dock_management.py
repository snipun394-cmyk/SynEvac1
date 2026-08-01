import gc
import sys
import unittest

from PyQt6.QtWidgets import QApplication, QDockWidget

_app = QApplication.instance() or QApplication(sys.argv)

from designer.widgets.dock_manager import DockManager
from builder.windows.builder_main_window import BuilderMainWindow


# =====================================================
# Dock Management Refactor V1 -- verifies the real SynEvac Builder
# window uses the exact same shared DockManager infrastructure as
# Studio, and that every Builder dock (none of which had ANY reopen
# mechanism before this refactor -- see the investigation) is now
# recoverable and survives the original bug trace. See
# tests/test_studio_dock_management.py's own module docstring for why
# each test constructs and explicitly tears down its own window.
# =====================================================


class _BuilderWindowTestCase(unittest.TestCase):

    def make_window(self):

        window = BuilderMainWindow()
        window.show()
        _app.processEvents()

        self.addCleanup(self._close_and_collect, window)

        return window

    def _close_and_collect(self, window):

        window.hide()
        window.deleteLater()
        _app.processEvents()
        gc.collect()


class EveryDockIsManagedTests(_BuilderWindowTestCase):

    def test_all_six_docks_share_one_dock_manager(self):

        window = self.make_window()

        self.assertIsInstance(window.dock_manager, DockManager)
        self.assertEqual(len(window._view_menu_actions), 6)

        for action in window._view_menu_actions:
            self.assertTrue(action.isCheckable())
            self.assertTrue(action.isChecked())  # every Builder dock starts visible

    # =====================================================

    def test_every_dock_title_has_a_view_menu_action(self):

        window = self.make_window()

        titles = {action.text() for action in window._view_menu_actions}

        for expected in (
            "Project Explorer", "Floors", "Properties",
            "Validation", "Project Summary", "Navigation Preview",
        ):
            self.assertIn(expected, titles)

    # =====================================================

    def test_view_menu_exists_and_contains_every_action(self):

        window = self.make_window()

        menubar = window.menuBar()
        view_menu = next(
            action.menu() for action in menubar.actions() if action.text() == "View"
        )

        menu_action_texts = {action.text() for action in view_menu.actions()}
        expected_texts = {action.text() for action in window._view_menu_actions}

        self.assertEqual(menu_action_texts, expected_texts)


class RecoverabilityTests(_BuilderWindowTestCase):

    # Requirement 5 -- Project Explorer, Floors, Properties, and every
    # other Builder dock must become recoverable: none had a toggle
    # action before this refactor, so closing one via its own native
    # close button (DockWidgetClosable is on by default, unrestricted
    # by any setFeatures() call) left no way back at all.

    def test_closing_a_dock_via_its_native_close_leaves_it_recoverable(self):

        window = self.make_window()

        docks_by_title = {dock.windowTitle(): dock for dock in window.findChildren(QDockWidget)}
        project_explorer_dock = docks_by_title["Project Explorer"]
        action = next(a for a in window._view_menu_actions if a.text() == "Project Explorer")

        # QDockWidget's own native close button calls close(), which is
        # equivalent to hide() for a dock -- never destroys it.
        project_explorer_dock.close()
        self.assertFalse(project_explorer_dock.isVisible())
        self.assertFalse(action.isChecked())

        action.trigger()
        _app.processEvents()

        self.assertTrue(project_explorer_dock.isVisible())


class OriginalBugTraceTests(_BuilderWindowTestCase):

    def test_every_builder_dock_survives_the_original_bug_trace(self):

        window = self.make_window()

        docks_by_title = {dock.windowTitle(): dock for dock in window.findChildren(QDockWidget)}

        for action in window._view_menu_actions:

            dock = docks_by_title[action.text()]

            with self.subTest(dock=action.text()):

                self.assertTrue(dock.isVisible())  # Builder docks start visible

                dock.setFloating(True)  # double-click the title bar
                _app.processEvents()
                self.assertTrue(dock.isVisible())

                action.trigger()  # "reopen" (actually closes it, since it was visible)
                self.assertFalse(dock.isVisible())

                action.trigger()  # reopen again
                _app.processEvents()

                self.assertTrue(dock.isVisible())
                self.assertFalse(dock.isFloating())  # re-docked -- never stuck invisible


if __name__ == "__main__":
    unittest.main()
