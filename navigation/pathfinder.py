from navigation.cost import DefaultCostModel


class Pathfinder:

    def __init__(self, graph, cost_model=None):

        self.graph = graph

        # Edge cost is always obtained through this interface, never
        # by reading Door/Stair/Exit directly -- see navigation/cost.py.
        self.cost_model = cost_model or DefaultCostModel()

    def shortest_path(
        self,
        start,
        goal,
    ):

        # Dijkstra / A* will be implemented later, against
        # self.cost_model.cost(edge).

        return []