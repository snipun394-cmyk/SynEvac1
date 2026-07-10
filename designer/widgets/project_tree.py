from PyQt6.QtWidgets import (
    QTreeWidget,
    QTreeWidgetItem,
)


class ProjectTree(QTreeWidget):

    def __init__(self):
        super().__init__()

        self.setHeaderHidden(True)

        self.project = None

    # =====================================================

    def set_project(self, project):

        self.project = project

        self.refresh()

    # =====================================================

    def refresh(self):

        self.clear()

        if self.project is None:
            return

        if self.project.building is None:
            return

        building = QTreeWidgetItem(
            [self.project.building.name]
        )

        self.addTopLevelItem(building)

        # =================================================
        # Floors
        # =================================================

        for floor in self.project.building.floors:

            floor_item = QTreeWidgetItem(
                [floor.name]
            )

            building.addChild(floor_item)

            # =============================================
            # Zones
            # =============================================

            zones_root = QTreeWidgetItem(
                ["Zones"]
            )

            floor_item.addChild(zones_root)

            for zone in floor.zones:

                zone_item = QTreeWidgetItem(
                    [zone.name]
                )

                zones_root.addChild(zone_item)

            # =============================================
            # Exits
            # =============================================

            exits_root = QTreeWidgetItem(
                ["Exits"]
            )

            floor_item.addChild(exits_root)

            for exit_obj in floor.exits:

                exits_root.addChild(
                    QTreeWidgetItem(
                        [exit_obj.name]
                    )
                )

            # =============================================
            # Doors
            # =============================================

            doors_root = QTreeWidgetItem(
                ["Doors"]
            )

            floor_item.addChild(doors_root)

            for door in floor.doors:

                doors_root.addChild(
                    QTreeWidgetItem(
                        [door.name]
                    )
                )

            # =============================================
            # Staircases
            # =============================================

            stairs_root = QTreeWidgetItem(
                ["Staircases"]
            )

            floor_item.addChild(stairs_root)

            for stair in floor.stairs:

                stairs_root.addChild(
                    QTreeWidgetItem(
                        [stair.name]
                    )
                )

            # =============================================
            # Elevators
            # =============================================

            elevators_root = QTreeWidgetItem(
                ["Elevators"]
            )

            floor_item.addChild(elevators_root)

            for elevator in floor.elevators:

                elevators_root.addChild(
                    QTreeWidgetItem(
                        [elevator.name]
                    )
                )

            # =============================================
            # Cameras
            # =============================================

            cameras_root = QTreeWidgetItem(
                ["Cameras"]
            )

            floor_item.addChild(cameras_root)

            for camera in floor.cameras:

                cameras_root.addChild(
                    QTreeWidgetItem(
                        [camera.name]
                    )
                )

            # =============================================
            # Detectors
            # =============================================

            detectors_root = QTreeWidgetItem(
                ["Detectors"]
            )

            floor_item.addChild(detectors_root)

            for detector in floor.detectors:

                detectors_root.addChild(
                    QTreeWidgetItem(
                        [detector.name]
                    )
                )

            # =============================================
            # Assembly Points
            # =============================================

            assembly_points_root = QTreeWidgetItem(
                ["Assembly Points"]
            )

            floor_item.addChild(assembly_points_root)

            for assembly_point in floor.assembly_points:

                assembly_points_root.addChild(
                    QTreeWidgetItem(
                        [assembly_point.name]
                    )
                )

            # =============================================
            # Obstacles
            # =============================================

            obstacles_root = QTreeWidgetItem(
                ["Obstacles"]
            )

            floor_item.addChild(obstacles_root)

            for obstacle in floor.obstacles:

                obstacles_root.addChild(
                    QTreeWidgetItem(
                        [obstacle.name]
                    )
                )

        building.setExpanded(True)

        for i in range(building.childCount()):
            building.child(i).setExpanded(True)