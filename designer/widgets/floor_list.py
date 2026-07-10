from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QGridLayout,
    QInputDialog,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class FloorList(QWidget):

    # =====================================================
    # Presentation only. Every operation below delegates to
    # Building -- this widget never reads or writes
    # display_order, and never reorders/sorts floors itself.
    # =====================================================

    def __init__(self):
        super().__init__()

        self.building = None
        self.active_floor = None

        self.active_floor_changed_callback = None
        self.floors_changed_callback = None

        layout = QVBoxLayout()

        self.list_widget = QListWidget()

        self.list_widget.currentItemChanged.connect(
            self._on_selection_changed
        )

        layout.addWidget(self.list_widget)

        # =================================================
        # Controls
        #
        # Laid out as a 2-column grid rather than one wide row:
        # 6 buttons in a single QHBoxLayout forced a ~640px
        # minimum panel width, which pushed the whole MainWindow
        # (and the Property Panel dock with it) off typical
        # desktop resolutions.
        # =================================================

        button_layout = QGridLayout()

        self.add_button = QPushButton("Add")
        self.rename_button = QPushButton("Rename")
        self.duplicate_button = QPushButton("Duplicate")
        self.delete_button = QPushButton("Delete")
        self.move_up_button = QPushButton("Move Up")
        self.move_down_button = QPushButton("Move Down")

        self.add_button.clicked.connect(self.add_floor)
        self.rename_button.clicked.connect(self.rename_selected_floor)
        self.duplicate_button.clicked.connect(self.duplicate_selected_floor)
        self.delete_button.clicked.connect(self.delete_selected_floor)
        self.move_up_button.clicked.connect(self.move_selected_floor_up)
        self.move_down_button.clicked.connect(self.move_selected_floor_down)

        button_layout.addWidget(self.add_button, 0, 0)
        button_layout.addWidget(self.duplicate_button, 0, 1)

        button_layout.addWidget(self.rename_button, 1, 0)
        button_layout.addWidget(self.delete_button, 1, 1)

        button_layout.addWidget(self.move_up_button, 2, 0)
        button_layout.addWidget(self.move_down_button, 2, 1)

        layout.addLayout(button_layout)

        self.setLayout(layout)

    # =====================================================
    # Binding
    # =====================================================

    def set_building(self, building):

        self.building = building
        self.active_floor = None

        self.refresh()

    # =====================================================

    def refresh(self):

        self.list_widget.blockSignals(True)

        self.list_widget.clear()

        if self.building is not None:

            for floor in self.building.ordered_floors():

                item = QListWidgetItem(floor.name)

                item.setData(
                    Qt.ItemDataRole.UserRole,
                    floor,
                )

                self.list_widget.addItem(item)

                if floor is self.active_floor:
                    self.list_widget.setCurrentItem(item)

        self.list_widget.blockSignals(False)

    # =====================================================
    # Selection
    # =====================================================

    def selected_floor(self):

        item = self.list_widget.currentItem()

        if item is None:
            return None

        return item.data(Qt.ItemDataRole.UserRole)

    # =====================================================

    def _on_selection_changed(self, current, previous):

        floor = (
            current.data(Qt.ItemDataRole.UserRole)
            if current
            else None
        )

        self.set_active_floor(floor)

    # =====================================================

    def set_active_floor(self, floor):

        self.active_floor = floor

        if self.active_floor_changed_callback:
            self.active_floor_changed_callback(floor)

    # =====================================================
    # Actions -- each delegates entirely to Building
    # =====================================================

    def add_floor(self):

        if self.building is None:
            return

        suggested_name = (
            f"Floor {self.building.floor_count + 1}"
        )

        name, ok = QInputDialog.getText(
            self,
            "Add Floor",
            "Floor name:",
            text=suggested_name,
        )

        if not ok or not name.strip():
            return

        # Elevation is derived (see Building.floor_elevation()), so
        # it is never asked for here -- only Height, which is the
        # one physical measurement a floor actually needs at
        # creation time.
        height, ok = QInputDialog.getDouble(
            self,
            "Add Floor",
            "Floor height (m):",
            3.0,
            0.1,
            100.0,
            2,
        )

        if not ok:
            return

        floor = self.building.create_floor(
            name=name.strip(),
            height=height,
        )

        self.set_active_floor(floor)
        self.refresh()

        self._notify_floors_changed()

    # =====================================================

    def rename_selected_floor(self):

        floor = self.selected_floor()

        if floor is None or self.building is None:
            return

        name, ok = QInputDialog.getText(
            self,
            "Rename Floor",
            "Floor name:",
            text=floor.name,
        )

        if not ok or not name.strip():
            return

        self.building.rename_floor(floor, name.strip())

        self.refresh()

        self._notify_floors_changed()

    # =====================================================

    def duplicate_selected_floor(self):

        floor = self.selected_floor()

        if floor is None or self.building is None:
            return

        new_floor = self.building.duplicate_floor(floor)

        if new_floor is None:
            return

        self.set_active_floor(new_floor)
        self.refresh()

        self._notify_floors_changed()

    # =====================================================

    def delete_selected_floor(self):

        floor = self.selected_floor()

        if floor is None or self.building is None:
            return

        confirm = QMessageBox.question(
            self,
            "Delete Floor",
            f'Delete "{floor.name}"? This cannot be undone.',
        )

        if confirm != QMessageBox.StandardButton.Yes:
            return

        fallback = self.building.delete_floor(floor)

        self.set_active_floor(fallback)
        self.refresh()

        self._notify_floors_changed()

    # =====================================================

    def move_selected_floor_up(self):

        floor = self.selected_floor()

        if floor is None or self.building is None:
            return

        self.building.move_floor_up(floor)

        self.refresh()

        self._notify_floors_changed()

    # =====================================================

    def move_selected_floor_down(self):

        floor = self.selected_floor()

        if floor is None or self.building is None:
            return

        self.building.move_floor_down(floor)

        self.refresh()

        self._notify_floors_changed()

    # =====================================================

    def _notify_floors_changed(self):

        if self.floors_changed_callback:
            self.floors_changed_callback()
