from navigation.graph import NavigationGraph


class GraphBuilder:

    def __init__(self):

        self.graph = NavigationGraph()

    def build(self, project):

        # Build navigation graph from
        # Zones, Exits, Stairs, Doors

        return self.graphs