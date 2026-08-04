class BufferModel:

    # Admission Control V7 -- Hybrid Buffer-Service Architecture. The
    # new, independent BUFFER constraint MultiAgentSimulation's
    # buffer_model parameter accepts, alongside (never in place of)
    # capacity_model (service-zone storage) and discharge_model
    # (service-zone throughput, Admission Control V4). Mirrors their
    # own role: the one interface MultiAgentSimulation depends on for
    # "how many occupants can physically be present at this landing at
    # once" -- the missing quantity Admission Control V5/V6 identified
    # (queueing theory's own `K`, buffer/waiting-room size), gated at
    # NODE granularity (a landing), never at Edge/FlowRegion
    # granularity (that remains capacity_model's own, sole concern).

    def buffer_capacity(self, node):

        raise NotImplementedError


class DefaultBufferModel(BufferModel):

    # A literature-grounded default, per the Admission Control V5
    # investigation's own survey: Pathfinder's Max Room Density
    # (3.55 persons/m^2) and buildingEXODUS's own packing-density
    # ceiling (4 people/m^2, from its 0.5m grid cells) converge closely
    # -- 3.8 persons/m^2 is the midpoint of that triangulation, not an
    # arbitrary number.
    #
    # Deliberately uses Node.area (navigation/node.py), not a new
    # geometry computation -- Node.area already reads straight through
    # to Zone.area (= width * height, models/zone.py), which is already
    # authored by every existing Designer building with no new field
    # required. This is the literal mechanism behind this milestone's
    # own "Landing area must be derived automatically from existing
    # Zone geometry" requirement -- there was nothing new to derive,
    # only something existing to finally consult.
    #
    # None area (the "Outside" node, or any node whose reference has no
    # derivable area) returns None -- fails OPEN, the same "not
    # derivable, no constraint" convention DischargeModel/
    # DefaultCapacityModel already use. This is physically correct, not
    # a fallback of convenience: the exterior world has no meaningful
    # landing capacity, and every Exit edge's own to_node is exactly
    # the single shared Outside node (navigation/node.py's own
    # OUTSIDE_NODE_ID) -- so exiting the building is never buffer-
    # gated, by construction.

    MAX_DENSITY_PERS_PER_SQ_M = 3.8

    MINIMUM_BUFFER_CAPACITY = 1

    def buffer_capacity(self, node):

        area = node.area

        if area is None or area <= 0:
            return None

        derived_capacity = int(area * self.MAX_DENSITY_PERS_PER_SQ_M)

        return max(derived_capacity, self.MINIMUM_BUFFER_CAPACITY)
