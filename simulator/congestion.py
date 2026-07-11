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

    def speed_factor(self, edge, other_occupants, capacity):

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

    def speed_factor(self, edge, other_occupants, capacity):

        if capacity <= 0:
            return self.MINIMUM_SPEED_FACTOR

        occupancy_ratio = min(other_occupants / capacity, 1.0)

        factor = 1.0 - occupancy_ratio * (1.0 - self.MINIMUM_SPEED_FACTOR)

        return max(factor, self.MINIMUM_SPEED_FACTOR)
