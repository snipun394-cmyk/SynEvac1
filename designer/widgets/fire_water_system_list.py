from PyQt6.QtWidgets import (
    QInputDialog,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from PyQt6.QtCore import Qt


class FireWaterSystemList(QWidget):

    # Fire Water Supply & Suppression Infrastructure milestone, Phase
    # 10 -- the cleanest existing UI precedent for creating/renaming/
    # deleting a small set of NAMED, non-spatial entities is designer.
    # widgets.floor_list.FloorList (a QListWidget + Add/Rename/Delete
    # buttons, delegating every operation straight to Building). This
    # widget follows that exact shape for FireWaterSystem
    # (models/fire_water_system.py) -- a FireWaterSystem is likewise
    # never drawn/placed on the canvas (it is a logical grouping, not a
    # spatial object), so a dockable list panel is the right authoring
    # surface, not a toolbar tool. No JSON editing is ever required of
    # the operator.
    #
    # Presentation only -- every operation delegates to Building; this
    # widget never mutates a FireWaterSystem's own membership fields
    # directly (that happens through each asset's own Property Panel
    # "Fire Water System" combo instead, see designer/widgets/
    # property_panel.py's own _populate_fire_water_system_combo()).

    def __init__(self):
        super().__init__()

        self.building = None

        layout = QVBoxLayout()

        self.list_widget = QListWidget()

        layout.addWidget(self.list_widget)

        self.add_button = QPushButton("Add System")
        self.rename_button = QPushButton("Rename System")
        self.delete_button = QPushButton("Delete System")

        self.add_button.clicked.connect(self.add_system)
        self.rename_button.clicked.connect(self.rename_selected_system)
        self.delete_button.clicked.connect(self.delete_selected_system)

        layout.addWidget(self.add_button)
        layout.addWidget(self.rename_button)
        layout.addWidget(self.delete_button)

        self.setLayout(layout)

    # =====================================================
    # Binding
    # =====================================================

    def set_building(self, building):

        self.building = building

        self.refresh()

    # =====================================================

    def refresh(self):

        self.list_widget.blockSignals(True)

        self.list_widget.clear()

        if self.building is not None:

            for system in self.building.fire_water_systems:

                item = QListWidgetItem(system.name)
                item.setData(Qt.ItemDataRole.UserRole, system)

                self.list_widget.addItem(item)

        self.list_widget.blockSignals(False)

    # =====================================================
    # Selection
    # =====================================================

    def selected_system(self):

        item = self.list_widget.currentItem()

        if item is None:
            return None

        return item.data(Qt.ItemDataRole.UserRole)

    # =====================================================
    # Actions -- each delegates entirely to Building
    # =====================================================

    def add_system(self):

        if self.building is None:
            return

        suggested_name = f"Fire Water System {len(self.building.fire_water_systems) + 1}"

        name, ok = QInputDialog.getText(
            self,
            "Add Fire Water System",
            "System name:",
            text=suggested_name,
        )

        if not ok or not name.strip():
            return

        self.building.create_fire_water_system(name.strip())

        self.refresh()

    # =====================================================

    def rename_selected_system(self):

        system = self.selected_system()

        if system is None or self.building is None:
            return

        name, ok = QInputDialog.getText(
            self,
            "Rename Fire Water System",
            "System name:",
            text=system.name,
        )

        if not ok or not name.strip():
            return

        self.building.rename_fire_water_system(system, name.strip())

        self.refresh()

    # =====================================================

    def delete_selected_system(self):

        system = self.selected_system()

        if system is None or self.building is None:
            return

        confirm = QMessageBox.question(
            self,
            "Delete Fire Water System",
            f'Delete "{system.name}"? Assets referencing it will become '
            f"unassigned, not deleted. This cannot be undone.",
        )

        if confirm != QMessageBox.StandardButton.Yes:
            return

        self.building.remove_fire_water_system(system)

        self.refresh()
