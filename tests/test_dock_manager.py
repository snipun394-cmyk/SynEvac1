import gc
import sys
import unittest

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QDockWidget, QLabel, QMainWindow

_app = QApplication.instance() or QApplication(sys.argv)

from designer.widgets.dock_manager import DockManager


def make_dock(window, title="Test Dock"):

    dock = QDockWidget(title, window)
    dock.setWidget(QLabel(title))

    return dock


# =====================================================
# DockManager -- Dock Management Refactor V1's single, reusable
# mechanism (designer/widgets/dock_manager.py), replacing the ten
# duplicated `dock.setVisible(not dock.isVisible())` handlers this
# milestone's own investigation root-caused. Every test here drives
# the manager the same way a real double-click/close/reopen would,
# never by inspecting private state.
#
# Every test constructs its own QMainWindow/QDockWidget/DockManager.
# DockManager's own signal connections (topLevelChanged, the reopen
# action's triggered) are closures over the dock they manage -- a
# normal, ordinary Python reference cycle (dock -> Qt connection ->
# closure -> dock), harmless on its own, but PyQt/SIP can crash
# (access violation) if such a cycle's deferred garbage collection runs
# after the underlying C++ object was already deleted. Explicitly
# tearing down each window (never needed by the production app, which
# only ever constructs one window for its whole process lifetime) keeps
# every test's objects fully collected before the next test's -- and,
# just as importantly, before any OTHER test file's own heavier windows
# construct theirs in the same pytest process.
# =====================================================


class _DockManagerTestCase(unittest.TestCase):

    def make_window(self):

        window = QMainWindow()
        window.resize(1400, 900)
        window.show()

        self.addCleanup(self._close_and_collect, window)

        return window

    def _close_and_collect(self, window):

        window.hide()
        window.deleteLater()
        _app.processEvents()
        gc.collect()


class RegisterTests(_DockManagerTestCase):

    def test_returns_a_checkable_action_matching_initial_visibility(self):

        window = self.make_window()
        manager = DockManager(window)
        dock = make_dock(window)

        action = manager.register(dock, Qt.DockWidgetArea.BottomDockWidgetArea)
        _app.processEvents()  # addDockWidget()'s own show is only fully reflected in isVisible() after a processed event

        self.assertTrue(action.isCheckable())
        self.assertEqual(action.isChecked(), dock.isVisible())
        self.assertTrue(dock.isVisible())

    # =====================================================

    def test_start_hidden_leaves_the_dock_hidden_and_unchecked(self):

        window = self.make_window()
        manager = DockManager(window)
        dock = make_dock(window)

        action = manager.register(dock, Qt.DockWidgetArea.BottomDockWidgetArea, start_hidden=True)

        self.assertFalse(dock.isVisible())
        self.assertFalse(action.isChecked())

    # =====================================================

    def test_title_overrides_the_action_text(self):

        window = self.make_window()
        manager = DockManager(window)
        dock = make_dock(window)

        action = manager.register(
            dock, Qt.DockWidgetArea.BottomDockWidgetArea, title="Custom Panel Name",
        )

        self.assertEqual(action.text(), "Custom Panel Name")


class ShowHideTests(_DockManagerTestCase):

    def test_action_shows_and_hides_the_dock_both_directions(self):

        window = self.make_window()
        manager = DockManager(window)
        dock = make_dock(window)

        action = manager.register(dock, Qt.DockWidgetArea.BottomDockWidgetArea, start_hidden=True)

        action.trigger()
        self.assertTrue(dock.isVisible())
        self.assertTrue(action.isChecked())

        action.trigger()
        self.assertFalse(dock.isVisible())
        self.assertFalse(action.isChecked())


class FloatingTests(_DockManagerTestCase):

    # These tests replay the exact mechanism a double-click on the
    # dock's own title bar uses internally (QDockWidget's default
    # title bar handles mouseDoubleClickEvent via toggleTopLevel(),
    # which is equivalent to setFloating(not isFloating()) and emits
    # topLevelChanged(bool) -- the same signal this suite drives).

    def test_floating_never_desyncs_the_action_from_real_visibility(self):

        window = self.make_window()
        manager = DockManager(window)
        dock = make_dock(window)

        action = manager.register(dock, Qt.DockWidgetArea.BottomDockWidgetArea)

        dock.setFloating(True)
        _app.processEvents()

        # The historical bug's own premise: floating never makes
        # isVisible() False -- confirm the action agrees, not just the
        # dock.
        self.assertTrue(dock.isVisible())
        self.assertTrue(dock.isFloating())
        self.assertTrue(action.isChecked())

    # =====================================================

    def test_floating_a_small_dock_grows_it_to_the_default_floating_size(self):

        window = self.make_window()
        manager = DockManager(window)
        dock = make_dock(window)

        manager.register(dock, Qt.DockWidgetArea.BottomDockWidgetArea)
        dock.resize(50, 30)

        dock.setFloating(True)
        _app.processEvents()

        min_width, min_height = DockManager.DEFAULT_FLOATING_SIZE
        self.assertGreaterEqual(dock.width(), min_width)
        self.assertGreaterEqual(dock.height(), min_height)

    # =====================================================

    def test_floating_a_large_dock_does_not_shrink_it(self):

        window = self.make_window()
        manager = DockManager(window)
        dock = make_dock(window)

        manager.register(dock, Qt.DockWidgetArea.BottomDockWidgetArea)
        dock.resize(900, 700)

        dock.setFloating(True)
        _app.processEvents()

        self.assertGreaterEqual(dock.width(), 900)
        self.assertGreaterEqual(dock.height(), 700)


class ReopenAndOriginalBugTraceTests(_DockManagerTestCase):

    def test_original_bug_scenario_cannot_reproduce(self):

        # Replays the investigation's own 5-step trace verbatim:
        # 1. dock hidden at construction.
        # 2. user opens it via the (one, real) action.
        # 3. user double-clicks the title bar (setFloating(True)).
        # 4. user clicks the action again, believing the panel is gone.
        # 5. user clicks it again.
        # Historical bug: step 4 genuinely hid it (isVisible() had
        # never gone False after step 3, so the naive `setVisible(not
        # isVisible())` handler hid a dock the user thought was already
        # hidden), and every later click just alternated between
        # "really hidden" and "floating at an easy-to-miss position" --
        # both indistinguishable to the user. Assert the dock is
        # visible after every click that re-checks the action, and that
        # by step 5 it is back in a known-good (docked) position.

        window = self.make_window()
        manager = DockManager(window)
        dock = make_dock(window)

        action = manager.register(dock, Qt.DockWidgetArea.BottomDockWidgetArea, start_hidden=True)

        self.assertFalse(dock.isVisible())  # step 1

        action.trigger()  # step 2
        self.assertTrue(dock.isVisible())

        dock.setFloating(True)  # step 3 (double-click emulation)
        _app.processEvents()
        self.assertTrue(dock.isVisible())  # never actually hidden by floating

        action.trigger()  # step 4 -- "reopen" (actually a close, since it was checked)
        self.assertFalse(dock.isVisible())
        self.assertFalse(action.isChecked())

        action.trigger()  # step 5 -- "reopen" again
        _app.processEvents()

        self.assertTrue(dock.isVisible())
        self.assertTrue(action.isChecked())
        self.assertFalse(dock.isFloating())  # re-docked, not stuck floating out of view

    # =====================================================

    def test_reopening_a_closed_floating_dock_redocks_it_by_default(self):

        window = self.make_window()
        manager = DockManager(window)
        dock = make_dock(window)

        action = manager.register(dock, Qt.DockWidgetArea.BottomDockWidgetArea)

        dock.setFloating(True)
        _app.processEvents()

        action.trigger()  # close it while floating (native close-button equivalent)
        self.assertFalse(dock.isVisible())
        self.assertTrue(dock.isFloating())  # Qt remembers it was floating

        action.trigger()  # reopen
        _app.processEvents()

        self.assertTrue(dock.isVisible())
        self.assertFalse(dock.isFloating())

    # =====================================================

    def test_re_dock_on_reopen_can_be_disabled(self):

        window = self.make_window()
        manager = DockManager(window)
        dock = make_dock(window)

        action = manager.register(
            dock, Qt.DockWidgetArea.BottomDockWidgetArea, re_dock_on_reopen=False,
        )

        dock.setFloating(True)
        _app.processEvents()

        action.trigger()  # close
        action.trigger()  # reopen
        _app.processEvents()

        self.assertTrue(dock.isVisible())
        self.assertTrue(dock.isFloating())  # left floating -- opted out of re-docking

    # =====================================================

    def test_setFloating_alone_never_triggers_re_docking(self):

        # Regression guard for a real bug found while building this
        # module: connecting the re-dock logic to the action's
        # `toggled` signal (instead of `triggered`) caused a bare
        # setFloating(True) to immediately undo itself, because
        # QDockWidget's own internal float/re-dock reparenting bounces
        # toggleViewAction()'s toggled signal (False/True/False/True) as
        # pure internal noise with no user action involved -- `triggered`
        # does not fire for that internal bookkeeping, only for a real
        # activation of the action.

        window = self.make_window()
        manager = DockManager(window)
        dock = make_dock(window)

        manager.register(dock, Qt.DockWidgetArea.BottomDockWidgetArea)

        dock.setFloating(True)
        _app.processEvents()

        self.assertTrue(dock.isFloating())


class MirrorActionsTests(_DockManagerTestCase):

    def test_mirrored_action_toggles_the_dock_and_stays_in_sync(self):

        from PyQt6.QtGui import QAction

        window = self.make_window()
        manager = DockManager(window)
        dock = make_dock(window)

        toolbar_action = QAction("Toolbar Button", window)

        real_action = manager.register(
            dock, Qt.DockWidgetArea.BottomDockWidgetArea,
            start_hidden=True, mirror_actions=(toolbar_action,),
        )

        self.assertTrue(toolbar_action.isCheckable())
        self.assertFalse(toolbar_action.isChecked())

        # Toggling the mirrored (toolbar) action moves the real action
        # and the dock together.
        toolbar_action.trigger()
        _app.processEvents()
        self.assertTrue(dock.isVisible())
        self.assertTrue(real_action.isChecked())
        self.assertTrue(toolbar_action.isChecked())

        # Toggling the real action moves the mirrored one back.
        real_action.trigger()
        self.assertFalse(dock.isVisible())
        self.assertFalse(toolbar_action.isChecked())

    # =====================================================

    def test_mirroring_does_not_recurse_infinitely(self):

        from PyQt6.QtGui import QAction

        window = self.make_window()
        manager = DockManager(window)
        dock = make_dock(window)

        toolbar_action = QAction("Toolbar Button", window)
        real_action = manager.register(
            dock, Qt.DockWidgetArea.BottomDockWidgetArea, mirror_actions=(toolbar_action,),
        )

        call_count = 0

        def count(_checked):
            nonlocal call_count
            call_count += 1

        toolbar_action.toggled.connect(count)

        real_action.trigger()  # a single state change should fire the mirror once

        self.assertEqual(call_count, 1)


if __name__ == "__main__":
    unittest.main()
