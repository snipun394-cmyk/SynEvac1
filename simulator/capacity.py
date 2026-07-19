from navigation.edge import Edge


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


def derive_stair_capacity(width, vertical_travel_distance=0.0):

    # The shared formula behind StairCapacityModel below, pulled out
    # as a plain function so Ground Truth (which only ever has a
    # Staircase/Edge, never a live MultiAgentSimulation) can compute
    # the exact same number without constructing a capacity model.
    #
    # Both constants are an arbitrary, documented placeholder, not a
    # validated life-safety flow-rate value: PEOPLE_PER_METER_OF_WIDTH
    # is deliberately lower than DefaultCapacityModel's flat-floor
    # figure, on the engineering judgment that vertical movement
    # (climbing/descending, handrail clearance, tread-width boundary
    # layers) uses more effective width per person than a flat
    # doorway of the same nominal width. VERTICAL_DISTANCE_STEP_M then
    # removes one more unit of capacity for every additional stretch
    # of stair travel, on the judgment that a longer stairwell has
    # more opportunity for a queue to compress at its pinch point.
    # Always floored at MINIMUM_CAPACITY, same reasoning as
    # DefaultCapacityModel: capacity can narrow throughput but must
    # never reach zero, or a queued occupant would be stranded
    # forever (routes are planned once and never revised).

    if width is None:
        return StairCapacityModel.MINIMUM_CAPACITY

    base_capacity = width * StairCapacityModel.PEOPLE_PER_METER_OF_WIDTH
    distance_penalty = (
        (vertical_travel_distance or 0.0) // StairCapacityModel.VERTICAL_DISTANCE_STEP_M
    )

    derived_capacity = int(base_capacity - distance_penalty)

    return max(derived_capacity, StairCapacityModel.MINIMUM_CAPACITY)


class StairCapacityModel(CapacityModel):

    # Wraps DefaultCapacityModel and narrows it further, but only for
    # Stair edges -- every Door/Exit crossing is delegated straight
    # through, unchanged. Same "wrap the base model, special-case only
    # your own concern" shape as HazardAwareCapacityModel.

    PEOPLE_PER_METER_OF_WIDTH = 1.2
    VERTICAL_DISTANCE_STEP_M = 4.0
    MINIMUM_CAPACITY = DefaultCapacityModel.MINIMUM_CAPACITY

    def __init__(self, base_model: CapacityModel = None):

        self.base_model = base_model or DefaultCapacityModel()

    def capacity(self, edge):

        if edge.edge_type != Edge.STAIR:
            return self.base_model.capacity(edge)

        explicit_capacity = edge.capacity

        if explicit_capacity is not None:
            return max(int(explicit_capacity), self.MINIMUM_CAPACITY)

        return derive_stair_capacity(edge.width, edge.walking_distance)
