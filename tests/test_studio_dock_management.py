import gc
import sys
import unittest

from PyQt6.QtWidgets import QApplication, QDockWidget

_app = QApplication.instance() or QApplication(sys.argv)

from designer.widgets.dock_manager import DockManager
from designer.windows.main_window import MainWindow


# =====================================================
# Dock Management Refactor V1 -- verifies the real SynEvac Studio
# window (not a synthetic stand-in) uses the single, shared DockManager
# for every dock, and that the original bug (double-click a title bar
# -> dock permanently unrecoverable) cannot reproduce against the
# actual production window.
#
# Each test constructs its own full MainWindow (a real, heavy
# QMainWindow with 14 QDockWidgets whose signal connections capture
# `self` in closures, exactly as production code needs to). Left to
# Python's own cyclic garbage collector, several such windows'
# collection can be deferred and then run mid-construction of a LATER
# window in the same process -- a well-known PyQt/SIP hazard where a
# deferred collection pass touches an already-C++-deleted object and
# crashes the interpreter. Explicitly closing and force-collecting each
# window in tearDown() (never done by the production app itself, which
# only ever constructs one MainWindow for its entire process lifetime)
# keeps every test's window fully collected before the next test
# constructs its own.
# =====================================================


class _MainWindowTestCase(unittest.TestCase):

    def make_window(self):

        window = MainWindow()
        window.show()
        _app.processEvents()

        self.addCleanup(self._close_and_collect, window)

        return window

    def _close_and_collect(self, window):

        # Deliberately NOT window.close() -- MainWindow.closeEvent()
        # calls _confirm_discard_unsaved_changes(), which can show a
        # blocking QMessageBox; that interactive-close confirmation
        # logic has nothing to do with test teardown, whose only job is
        # freeing the window's Qt/Python objects deterministically.
        window.hide()
        window.deleteLater()
        _app.processEvents()
        gc.collect()


class EveryDockIsManagedTests(_MainWindowTestCase):

    def test_all_fourteen_docks_share_one_dock_manager(self):

        window = self.make_window()

        self.assertIsInstance(window.dock_manager, DockManager)
        self.assertEqual(len(window._view_menu_actions), 14)

        for action in window._view_menu_actions:
            self.assertTrue(action.isCheckable())

    # =====================================================

    def test_previously_unmanaged_docks_now_have_a_working_view_menu_action(self):

        # Project Explorer, Floors, Fire Water Systems, and Properties
        # had no toggle/reopen action anywhere before this refactor.

        window = self.make_window()

        titles = {action.text() for action in window._view_menu_actions}

        for expected in ("Project Explorer", "Floors", "Fire Water Systems", "Properties"):
            self.assertIn(expected, titles)

        docks_by_title = {dock.windowTitle(): dock for dock in window.findChildren(QDockWidget)}
        project_explorer_action = next(
            a for a in window._view_menu_actions if a.text() == "Project Explorer"
        )
        project_explorer_dock = docks_by_title["Project Explorer"]

        project_explorer_action.trigger()
        self.assertFalse(project_explorer_dock.isVisible())

        project_explorer_action.trigger()
        _app.processEvents()
        self.assertTrue(project_explorer_dock.isVisible())


class ToolbarMirrorTests(_MainWindowTestCase):

    def test_toolbar_simulation_action_and_dock_action_stay_in_sync(self):

        window = self.make_window()

        self.assertTrue(window.toolbar.simulation_action.isCheckable())
        self.assertEqual(
            window.toolbar.simulation_action.isChecked(),
            window.simulation_dock_action.isChecked(),
        )

        window.toolbar.simulation_action.trigger()
        _app.processEvents()

        self.assertTrue(window.simulation_dock.isVisible())
        self.assertTrue(window.simulation_dock_action.isChecked())
        self.assertTrue(window.toolbar.simulation_action.isChecked())

        window.simulation_dock_action.trigger()
        self.assertFalse(window.simulation_dock.isVisible())
        self.assertFalse(window.toolbar.simulation_action.isChecked())


class OriginalBugTraceTests(_MainWindowTestCase):

    def test_simulation_dock_survives_the_original_bug_trace(self):

        window = self.make_window()

        dock = window.simulation_dock
        action = window.simulation_dock_action

        self.assertFalse(dock.isVisible())  # hidden by default

        action.trigger()  # user opens it
        self.assertTrue(dock.isVisible())

        dock.setFloating(True)  # double-click the title bar
        _app.processEvents()
        self.assertTrue(dock.isVisible())  # never silently hidden by floating

        action.trigger()  # user tries to "reopen" (closes it, since it was visible)
        self.assertFalse(dock.isVisible())

        action.trigger()  # user tries again
        _app.processEvents()

        self.assertTrue(dock.isVisible())
        self.assertFalse(dock.isFloating())  # re-docked -- never stuck invisible again

    # =====================================================

    def test_every_tabbed_bottom_dock_survives_the_same_trace(self):

        window = self.make_window()

        for dock, action in (
            (window.perception_debug_dock, window.perception_debug_dock_action),
            (window.building_state_debug_dock, window.building_state_debug_dock_action),
            (window.camera_manager_dock, window.camera_manager_dock_action),
            (window.speaker_manager_dock, window.speaker_manager_dock_action),
            (window.camera_validation_dock, window.camera_validation_dock_action),
            (window.live_runtime_dock, window.live_runtime_dock_action),
            (window.recommendation_dock, window.recommendation_dock_action),
            (window.execution_dock, window.execution_dock_action),
            (window.live_camera_view_dock, window.live_camera_view_dock_action),
        ):
            with self.subTest(dock=dock.windowTitle()):

                action.trigger()
                self.assertTrue(dock.isVisible())

                dock.setFloating(True)
                _app.processEvents()
                self.assertTrue(dock.isVisible())

                action.trigger()
                self.assertFalse(dock.isVisible())

                action.trigger()
                _app.processEvents()
                self.assertTrue(dock.isVisible())
                self.assertFalse(dock.isFloating())

                # Leave it as we found it (hidden) for the next dock in
                # this loop, matching every dock's own default state.
                action.trigger()


class RefreshOnOpenPreservedTests(_MainWindowTestCase):

    # Dock Management Refactor V1 removed the ten duplicated
    # toggle_*_panel methods, three of which had a real, non-duplicate
    # side effect beyond visibility (refreshing their own panel content
    # the moment they became visible) -- confirms that behavior was
    # carried over onto the real dock actions, not silently dropped.

    def test_camera_manager_panel_refreshes_when_dock_becomes_visible(self):

        window = self.make_window()

        calls = []
        window._refresh_camera_manager_panel = lambda: calls.append(1)

        window.camera_manager_dock_action.trigger()

        self.assertEqual(len(calls), 1)

    # =====================================================

    def test_speaker_manager_panel_refreshes_when_dock_becomes_visible(self):

        window = self.make_window()

        calls = []
        window._refresh_speaker_manager_panel = lambda: calls.append(1)

        window.speaker_manager_dock_action.trigger()

        self.assertEqual(len(calls), 1)

    # =====================================================

    def test_camera_validation_panel_refreshes_when_dock_becomes_visible(self):

        window = self.make_window()

        calls = []
        window.camera_validation_panel.refresh = lambda building: calls.append(building)

        window.camera_validation_dock_action.trigger()

        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
