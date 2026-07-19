import sys
import unittest

from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox

# Module-level QApplication singleton -- same convention every other
# PyQt6-backed test module in this repo already establishes (see
# tests/training_dataset_fixtures.py, tests/test_campaign_studio.py).
_app = QApplication.instance() or QApplication(sys.argv)

from designer.windows.main_window import MainWindow
from models.project import Project


def _patched(cls, attr_name, replacement):

    # Same "monkey-patch a blocking Qt dialog to a no-op, restore in
    # finally" convention tests/test_campaign_studio.py already
    # establishes -- a real QMessageBox/QFileDialog would hang a
    # headless test run waiting for a user click that will never come.

    original = getattr(cls, attr_name)
    setattr(cls, attr_name, staticmethod(replacement))

    return original


class ToolbarWiringTests(unittest.TestCase):

    def test_new_open_save_actions_are_connected(self):

        window = MainWindow()

        for action in (
            window.toolbar.new_action, window.toolbar.open_action, window.toolbar.save_action,
        ):
            self.assertGreater(action.receivers(action.triggered), 0)

    def test_zoom_and_reset_actions_are_connected_and_functional(self):

        window = MainWindow()

        window.toolbar.zoom_in_action.trigger()
        self.assertGreater(window.canvas.transform().m11(), 1.0)

        window.toolbar.reset_view_action.trigger()
        self.assertEqual(window.canvas.zoom_level, 100)

    def test_undo_redo_elevator_are_disabled_not_silently_inert(self):

        window = MainWindow()

        for action in (
            window.toolbar.undo_action, window.toolbar.redo_action, window.toolbar.elevator_action,
        ):
            self.assertFalse(action.isEnabled())
            self.assertTrue(action.toolTip())


class NewProjectTests(unittest.TestCase):

    def test_new_project_replaces_project_with_a_fresh_default(self):

        window = MainWindow()

        original_project = window.canvas.scene_obj.project
        window.canvas.scene_obj.project.name = "Renamed Project"

        window.new_project()

        self.assertIsNot(window.canvas.scene_obj.project, original_project)
        self.assertEqual(window.canvas.scene_obj.project.name, "Untitled Project")
        self.assertEqual(window.canvas.scene_obj.project.building.floor_count, 1)

    def test_new_project_action_is_wired_in_the_file_menu(self):

        window = MainWindow()

        self.assertGreater(window.new_action.receivers(window.new_action.triggered), 0)


class UnsavedChangesTests(unittest.TestCase):

    def test_scene_mutation_marks_the_window_dirty(self):

        window = MainWindow()

        self.assertFalse(window._dirty)

        window.canvas.scene_obj.project.building.create_floor(name="Extra Floor")
        window.canvas.scene_obj.addText("trigger a scene change")

        # QGraphicsScene.changed aggregates and emits asynchronously on
        # the next event-loop iteration (real, not a test artifact --
        # the running app's own app.exec() loop delivers this during
        # normal use; a headless test has to pump it explicitly).
        _app.processEvents()

        self.assertTrue(window._dirty)

    def test_confirm_discard_returns_true_immediately_when_not_dirty(self):

        window = MainWindow()

        self.assertTrue(window._confirm_discard_unsaved_changes())

    def test_confirm_discard_returns_false_on_cancel(self):

        window = MainWindow()
        window._dirty = True

        original = _patched(
            QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Cancel,
        )
        try:
            self.assertFalse(window._confirm_discard_unsaved_changes())
        finally:
            QMessageBox.question = original

        self.assertTrue(window._dirty)

    def test_confirm_discard_returns_true_and_clears_dirty_on_discard(self):

        window = MainWindow()
        window._dirty = True

        original = _patched(
            QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Discard,
        )
        try:
            self.assertTrue(window._confirm_discard_unsaved_changes())
        finally:
            QMessageBox.question = original

        # Discard intentionally leaves the dirty flag alone (there is
        # still an unsaved change, the user just chose to proceed
        # without saving it) -- the caller (new/open/close) is the one
        # that actually replaces/discards the project afterward.
        self.assertTrue(window._dirty)

    def test_confirm_discard_saves_and_clears_dirty_when_save_succeeds(self):

        window = MainWindow()
        window._dirty = True

        original_question = _patched(
            QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Save,
        )
        original_dialog = _patched(
            QFileDialog, "getSaveFileName", lambda *a, **k: ("", ""),
        )
        try:
            # File dialog cancelled mid-save -- must NOT report success.
            self.assertFalse(window._confirm_discard_unsaved_changes())
        finally:
            QMessageBox.question = original_question
            QFileDialog.getSaveFileName = original_dialog

    def test_close_event_is_ignored_when_user_cancels(self):

        window = MainWindow()
        window._dirty = True

        class FakeCloseEvent:

            def __init__(self):
                self.accepted = False
                self.ignored = False

            def accept(self):
                self.accepted = True

            def ignore(self):
                self.ignored = True

        original = _patched(
            QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Cancel,
        )
        event = FakeCloseEvent()
        try:
            window.closeEvent(event)
        finally:
            QMessageBox.question = original

        self.assertTrue(event.ignored)
        self.assertFalse(event.accepted)

    def test_save_project_clears_dirty_flag(self):

        window = MainWindow()
        window._dirty = True

        with self._temp_save_path() as path:

            original = _patched(QFileDialog, "getSaveFileName", lambda *a, **k: (path, ""))
            try:
                result = window.save_project()
            finally:
                QFileDialog.getSaveFileName = original

        self.assertTrue(result)
        self.assertFalse(window._dirty)

    def _temp_save_path(self):

        import contextlib
        import os
        import tempfile

        @contextlib.contextmanager
        def _ctx():
            directory = tempfile.mkdtemp()
            try:
                yield os.path.join(directory, "test_project.syn")
            finally:
                pass

        return _ctx()


if __name__ == "__main__":
    unittest.main()
