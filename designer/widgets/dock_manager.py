from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QDockWidget, QMainWindow


# =====================================================
# Dock Management Refactor V1 -- the one place a QDockWidget's own
# show/hide/float/reopen behavior is wired, for both SynEvac Studio
# (designer/windows/main_window.py) and SynEvac Builder
# (builder/windows/builder_main_window.py). Pure PyQt6 -- zero
# SynEvac-domain imports -- so both apps can use it identically without
# either importing the other or pulling in anything heavier (this
# module lives in designer/widgets/, which already has no __init__.py
# of its own and is already how Builder reuses other dependency-clean
# Designer widgets -- bottom_info_bar.py, floor_list.py,
# project_tree.py).
#
# Root cause this replaces (see docs/architecture/
# dock_management_refactor_v1.md for the full investigation): every
# dock in this codebase either had no reopen action at all, or
# reimplemented one as `dock.setVisible(not dock.isVisible())` -- a
# boolean flip that assumes the ONLY way a dock's visibility ever
# changes is through that same handler. That assumption breaks the
# moment the user double-clicks the dock's own native title bar (Qt's
# built-in toggleTopLevel(), which floats the dock WITHOUT changing
# isVisible() at all) or uses its own native close button. This class
# never reimplements that tracking -- it wraps Qt's own already-correct
# QDockWidget.toggleViewAction() (kept perfectly in sync with real dock
# state by Qt itself, through every one of those native interactions)
# and adds exactly two small, additive behaviors Qt does not provide:
#
#   1. dock.topLevelChanged(True) -- fires the instant a dock becomes
#      floating, by ANY means (double-click, drag-out, or code) -- Qt
#      reuses the dock's last DOCKED geometry verbatim for the new
#      floating window (e.g. a full-width, 93px-tall strip pinned to
#      the exact screen position the dock already occupied), which is
#      trivially easy to miss. Deferred by one event-loop tick (Qt is
#      still finishing its own internal float transition when this
#      signal fires) to give the dock a sane minimum size and bring it
#      to the front -- never undoes the user's own deliberate float.
#
#   2. The dock's own toggleViewAction().toggled(True) -- fires when
#      the dock is re-shown via that action (a deliberate "bring this
#      panel back" gesture, distinct from a plain double-click). By
#      default (re_dock_on_reopen=True) this re-docks the panel to its
#      original area first -- the most foolproof choice, since it can
#      never again reappear at a possibly-invisible floating position;
#      a caller that genuinely wants a dock to stay floating across a
#      hide/show cycle can opt out per-dock.
# =====================================================


class DockManager:

    DEFAULT_FLOATING_SIZE = (420, 320)

    def __init__(self, main_window: QMainWindow):

        self._main_window = main_window

        # dock -> the Qt.DockWidgetArea it was registered with, used
        # only to re-dock a floating dock back to where it started
        # (setFloating(False) alone re-docks to the last non-floating
        # area Qt itself remembers, which is already correct -- this
        # map exists for callers that want to introspect it, not
        # because re-docking itself needs it).
        self._areas = {}

    # =====================================================

    def register(
        self,
        dock: QDockWidget,
        area,
        *,
        title=None,
        start_hidden=False,
        re_dock_on_reopen=True,
        mirror_actions=(),
    ) -> QAction:

        self._main_window.addDockWidget(area, dock)
        self._areas[dock] = area

        dock.topLevelChanged.connect(
            lambda floating, d=dock: self._on_top_level_changed(d, floating)
        )

        action = dock.toggleViewAction()

        if title is not None:
            action.setText(title)

        if re_dock_on_reopen:

            # Deliberately `triggered`, not `toggled`. QDockWidget's own
            # setFloating()/toggleTopLevel() implementation reparents the
            # dock into (or out of) a top-level window, and that
            # reparenting transiently hides/reshows the dock as a side
            # effect -- which bounces toggleViewAction()'s own `toggled`
            # signal (False/True/False/True) purely as internal Qt
            # noise, with no user action involved at all (confirmed
            # empirically while building this module: connecting to
            # `toggled` here caused a plain setFloating(True) to
            # immediately re-dock itself, undoing the very float it was
            # supposed to perform). `triggered` fires only in response
            # to an actual activation of the action (a real click, or an
            # explicit trigger() call) -- exactly "the user asked to
            # reopen this dock" and nothing else.
            action.triggered.connect(
                lambda checked, d=dock: self._on_toggled(d, checked)
            )

        for extra in mirror_actions:
            self._mirror(action, extra)

        if start_hidden:
            dock.hide()

        return action

    # =====================================================

    def _on_top_level_changed(self, dock: QDockWidget, floating: bool) -> None:

        if not floating:
            return

        # Deferred: Qt is still completing its own internal float
        # transition (creating the top-level window) when this signal
        # fires -- resizing/raising immediately can be a no-op on some
        # platforms. A zero-delay singleShot runs after Qt finishes.
        QTimer.singleShot(0, lambda d=dock: self._raise_and_resize(d))

    # =====================================================

    def _raise_and_resize(self, dock: QDockWidget) -> None:

        if not dock.isFloating():
            # Already re-docked (e.g. by _on_toggled below) before this
            # deferred call ran -- nothing to raise/resize.
            return

        min_width, min_height = self.DEFAULT_FLOATING_SIZE
        dock.resize(max(dock.width(), min_width), max(dock.height(), min_height))

        dock.raise_()
        dock.activateWindow()

    # =====================================================

    def _on_toggled(self, dock: QDockWidget, checked: bool) -> None:

        if checked and dock.isFloating():
            dock.setFloating(False)

    # =====================================================

    def _mirror(self, real_action: QAction, extra_action: QAction) -> None:

        # Asymmetric on purpose -- discovered empirically while building
        # this module: QDockWidget wires its OWN internal show/hide slot
        # to toggleViewAction()'s `triggered` signal, not `toggled` (the
        # same reason register()'s own re-dock-on-reopen logic above
        # must use `triggered`). Calling real_action.setChecked(...)
        # alone flips its checked flag but never actually shows/hides
        # the dock -- only real_action.trigger() does. So:
        #
        #   extra -> real: extra_action.triggered calls real_action.
        #     trigger() (never setChecked()) so the dock genuinely
        #     follows a click on the mirrored (e.g. toolbar) action.
        #   real -> extra: real_action.toggled (fired for every actual
        #     state change, by any means -- a real_action click, a
        #     dock's own close button, ...) drives extra_action.
        #     setChecked(), which is sufficient here since extra_action
        #     is a plain QAction with no further Qt-internal wiring of
        #     its own to cascade into.
        #
        # Loop-safe: setChecked() never emits `triggered`, only
        # `toggled` -- so the real->extra direction can never loop back
        # into the extra->real handler above.
        extra_action.setCheckable(True)
        extra_action.setChecked(real_action.isChecked())

        extra_action.triggered.connect(lambda _checked, ra=real_action: ra.trigger())
        real_action.toggled.connect(extra_action.setChecked)
