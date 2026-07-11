class CapacityModel:

    # The one interface MultiAgentSimulation depends on for "how many
    # occupants can be on this edge at once" -- mirrors CostModel's
    # role in Pathfinding. A future capacity model (e.g. one that
    # reduces effective capacity under a Dynamic Hazard layer) plugs
    # in here without any change to the coordinator.

    def capacity(self, edge):

        raise NotImplementedError


class DefaultCapacityModel(CapacityModel):

    # Exit already carries a real, authored capacity (Exit.capacity)
    # -- used as-is. Door and Staircase have no capacity field, only
    # width, so capacity is derived from it using a simple, documented
    # "people per meter of width" assumption -- not a validated
    # life-safety flow-rate model, just a reasonable default.
    #
    # Always floored at 1: a door/stair/exit of any positive width can
    # always be crossed by one person at a time. This floor is what
    # guarantees MultiAgentSimulation's event queue can never deadlock
    # -- every capacity-constrained edge always eventually admits the
    # occupant at the head of its queue.

    PEOPLE_PER_METER_OF_WIDTH = 1.5

    MINIMUM_CAPACITY = 1

    def capacity(self, edge):

        explicit_capacity = edge.capacity

        if explicit_capacity is not None:
            return max(int(explicit_capacity), self.MINIMUM_CAPACITY)

        width = edge.width

        if width is None:
            return self.MINIMUM_CAPACITY

        derived_capacity = int(width * self.PEOPLE_PER_METER_OF_WIDTH)

        return max(derived_capacity, self.MINIMUM_CAPACITY)
