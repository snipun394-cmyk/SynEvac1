class NavigationGraph:

    def __init__(self):

        self.nodes = {}

        self.edges = []

    # ===================================================

    def add_node(self, node):

        self.nodes[node.id] = node

    # ===================================================

    def add_edge(self, edge):

        self.edges.append(edge)

    # ===================================================

    def get_node(self, node_id):

        return self.nodes.get(node_id)

    # ===================================================

    def neighbours(self, node_id):

        result = []

        for edge in self.edges:

            if edge.start == node_id:

                result.append(
                    (
                        edge.end,
                        edge.cost,
                    )
                )

            elif (
                edge.bidirectional
                and edge.end == node_id
            ):

                result.append(
                    (
                        edge.start,
                        edge.cost,
                    )
                )

        return result