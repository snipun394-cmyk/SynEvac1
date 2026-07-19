from navigation.edge import Edge


class CongestionModel:

    # The one interface MultiAgentSimulation depends on for "how much
    # does current crowding on this edge slow someone down" -- mirrors
    # CostModel/CapacityModel's role. Returns a multiplier in (0, 1]
    # applied to an occupant's own walking_speed; 1.0 means no
    # slowdown at all.
    #
    # `other_occupants` deliberately excludes the occupant whose speed
    # is being computed -- a lone occupant crossing an empty-but-wide
    # edge should walk at full speed even though capacity isn't
    # infinite; slowdown should come from sharing the edge with other
    # people, not merely from capacity being finite.
    #
    # `opposing_occupants` is a second, optional headcount: occupants
    # sharing the same edge but travelling the opposite direction
    # (relevant to Stair edges, which V1 models as inherently
    # bidirectional -- see navigation/edge.py). Defaults to 0 so every
    # existing caller/override that only knows about same-direction
    # crowding keeps working unchanged.

    def speed_factor(self, edge, other_occupants, capacity, opposing_occupants=0):

        raise NotImplementedError


class DefaultCongestionModel(CongestionModel):

    # V1's whole congestion model: linear degradation from full speed
    # (1.0, no one else on the edge) down to a floored minimum (edge
    # at capacity). This is a simple, explicitly documented
    # simplification -- not a validated pedestrian-dynamics/
    # fundamental-diagram model -- same honesty as Building Analysis's
    # "no hardcoded life-safety thresholds". Congestion only ever
    # applies to edges, never to nodes (matches the requirement's own
    # split between tracked-only node occupancy and enforced edge
    # capacity).

    MINIMUM_SPEED_FACTOR = 0.3

    def speed_factor(self, edge, other_occupants, capacity, opposing_occupants=0):

        if capacity <= 0:
            return self.MINIMUM_SPEED_FACTOR

        occupancy_ratio = min(other_occupants / capacity, 1.0)

        factor = 1.0 - occupancy_ratio * (1.0 - self.MINIMUM_SPEED_FACTOR)

        return max(factor, self.MINIMUM_SPEED_FACTOR)


class StairAwareCongestionModel(CongestionModel):

    # Wraps DefaultCongestionModel (or any other CongestionModel) and
    # adds a counter-flow penalty, but only for Stair edges -- every
    # Door/Exit crossing is delegated straight through, unchanged.
    # Same "wrap the base model, special-case only your own concern"
    # shape as StairCapacityModel/HazardAwareCapacityModel.
    #
    # COUNTERFLOW_PENALTY_PER_OPPOSING is an arbitrary, documented
    # placeholder, not a validated pedestrian-counterflow value: each
    # occupant travelling the opposite way on the same stair shaves a
    # further fixed slice off the base speed factor, on the
    # engineering judgment that a stair -- unlike a wide corridor --
    # gives two opposing lines of people very little room to pass
    # each other without both slowing down.

    COUNTERFLOW_PENALTY_PER_OPPOSING = 0.15

    def __init__(self, base_model: CongestionModel = None):

        self.base_model = base_model or DefaultCongestionModel()

    def speed_factor(self, edge, other_occupants, capacity, opposing_occupants=0):

        base_factor = self.base_model.speed_factor(edge, other_occupants, capacity)

        if edge.edge_type != Edge.STAIR or opposing_occupants <= 0:
            return base_factor

        minimum = getattr(self.base_model, "MINIMUM_SPEED_FACTOR", DefaultCongestionModel.MINIMUM_SPEED_FACTOR)

        penalty = min(
            opposing_occupants * self.COUNTERFLOW_PENALTY_PER_OPPOSING,
            base_factor - minimum,
        )

        return max(base_factor - penalty, minimum)
