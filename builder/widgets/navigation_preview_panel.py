from PyQt6.QtCore import Qt
from PyQt6.QtGui import QBrush, QColor, QPainter, QPen
from PyQt6.QtWidgets import QGraphicsScene, QGraphicsView, QLabel, QVBoxLayout, QWidget

from navigation.graph_builder import NavigationGraphGenerator
from navigation.node import Node


class NavigationPreviewPanel(QWidget):

    # "Builder shall NOT run evacuation simulation. However, provide a
    # lightweight navigation preview showing: graph connectivity,
    # reachable exits, navigation graph visualization." (milestone
    # brief). Reuses navigation.graph_builder.NavigationGraphGenerator
    # UNCHANGED -- the exact same derived, read-only graph Studio's own
    # Navigation Graph tooling builds from a Building (see
    # docs/architecture/synevac_builder_feasibility_investigation.md,
    # Phase 2 -- confirmed independent, zero Simulation/AI coupling).
    # This panel only draws it; it never runs a pathfinder, never
    # simulates an occupant, and reachability below is a plain BFS
    # over Edge.traversable -- the same notion NavigationGraph.
    # validate()'s own (private) reachability check already uses, not
    # duplicated logic imported from navigation/pathfinder.py.
    #
    # Deliberately per-floor, not a single all-floors rendering: Zone/
    # AssemblyPoint positions are only meaningful within their own
    # floor's local coordinate plane (same reason Staircase.
    # from_position/to_position have no shared coordinate system --
    # see models/staircase.py). A Stair or Exit edge whose other end
    # is off this floor is drawn as a short labelled stub rather than
    # a real line to nowhere.

    PREVIEW_SCALE = 20  # pixels per meter, independent of GRID_SIZE -- this is its own small preview canvas, not the main authoring canvas.

    REACHABLE_COLOR = QColor(40, 180, 60)
    UNREACHABLE_COLOR = QColor(200, 60, 40)

    def __init__(self):
        super().__init__()

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.info_label = QLabel("No project loaded.")
        layout.addWidget(self.info_label)

        self.scene = QGraphicsScene()

        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing)

        layout.addWidget(self.view)

    # =====================================================

    def refresh(self, building, floor):

        self.scene.clear()

        if building is None or floor is None:

            self.info_label.setText("No project loaded.")

            return

        graph = NavigationGraphGenerator().build(building)

        reachable = self._compute_reachable(graph)

        node_positions = {}

        for node in graph.nodes.values():

            if node.floor_id != floor.id:
                continue

            position = self._position_of(node.reference)

            if position is None:
                continue

            node_positions[node.id] = position

        for edge in graph.edges:

            self._draw_edge(graph, edge, node_positions)

        for node_id, (x, y) in node_positions.items():

            node = graph.find_node(node_id)

            self._draw_node(node, x, y, node_id in reachable)

        reachable_count = sum(1 for node_id in node_positions if node_id in reachable)

        self.info_label.setText(
            f"{len(node_positions)} space(s) on '{floor.name}', "
            f"{reachable_count} reachable from Outside."
        )

        self.view.fitInView(self.scene.itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)

    # =====================================================

    def _position_of(self, reference):

        # Same duck-typed lookup NavigationGraphGenerator._position_of()
        # already uses (Zone exposes `center`, AssemblyPoint exposes
        # `position`) -- restated here rather than imported since that
        # method is private to graph_builder.py.

        if reference is None:
            return None

        if hasattr(reference, "center"):
            return reference.center

        if hasattr(reference, "position"):
            return reference.position

        return None

    # =====================================================

    def _compute_reachable(self, graph):

        reachable = set()
        frontier = [Node.OUTSIDE_NODE_ID]

        while frontier:

            current_id = frontier.pop()

            if current_id in reachable:
                continue

            reachable.add(current_id)

            current_node = graph.find_node(current_id)

            if current_node is None:
                continue

            for neighbor, edge in graph.find_neighbors(current_node):

                if edge.traversable and neighbor.id not in reachable:
                    frontier.append(neighbor.id)

        return reachable

    # =====================================================

    def _draw_node(self, node, x, y, is_reachable):

        px = x * self.PREVIEW_SCALE
        py = y * self.PREVIEW_SCALE

        color = self.REACHABLE_COLOR if is_reachable else self.UNREACHABLE_COLOR

        self.scene.addEllipse(
            px - 6, py - 6, 12, 12,
            QPen(QColor(20, 20, 20)),
            QBrush(color),
        )

        label = self.scene.addText(node.name if node is not None else "?")
        label.setPos(px + 8, py - 10)

    # =====================================================

    def _draw_edge(self, graph, edge, node_positions):

        pen = QPen(
            QColor(40, 150, 220) if edge.traversable else QColor(160, 160, 160),
            2,
        )

        from_position = node_positions.get(edge.from_node)
        to_position = node_positions.get(edge.to_node)

        if from_position is not None and to_position is not None:

            self.scene.addLine(
                from_position[0] * self.PREVIEW_SCALE, from_position[1] * self.PREVIEW_SCALE,
                to_position[0] * self.PREVIEW_SCALE, to_position[1] * self.PREVIEW_SCALE,
                pen,
            )

            return

        # One end (or both) is off this floor -- Exit leads to the
        # single shared "Outside" node (no position at all), or a
        # Stair's other end is a Zone on a different floor. Drawn as a
        # short labelled stub from whichever end IS on this floor.
        anchor = from_position if from_position is not None else to_position

        if anchor is None:
            return

        other_node_id = edge.to_node if from_position is not None else edge.from_node
        other_node = graph.find_node(other_node_id)

        label_text = "Outside" if other_node_id == Node.OUTSIDE_NODE_ID else (
            other_node.name if other_node is not None else "?"
        )

        stub_x = anchor[0] * self.PREVIEW_SCALE + 30
        stub_y = anchor[1] * self.PREVIEW_SCALE - 30

        self.scene.addLine(
            anchor[0] * self.PREVIEW_SCALE, anchor[1] * self.PREVIEW_SCALE,
            stub_x, stub_y,
            pen,
        )

        stub_label = self.scene.addText(f"-> {label_text}")
        stub_label.setPos(stub_x, stub_y)
