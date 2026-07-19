from typing import Dict, Optional

from occupancy.observation import OccupancyObservation
from occupancy.provider import OccupancyProvider
from occupancy.snapshot import OccupancySnapshot

from simulator import MultiAgentSimulationResult, OccupantTimeline


# Simulation Runtime -- architecture doc docs/architecture/simulation_runtime.md
# SS11: a required, new, additive implementation of the already-frozen
# OccupancyProvider interface -- the exact extension mechanism
# EvolutionBackedHazardProvider already demonstrates for HazardProvider, not a
# redesign of occupancy/ or simulator/. Neither package is modified anywhere
# in this module.
#
# The core question SS11/SS21 explicitly left open -- "where is occupant X at
# time T" for an occupant strictly between two OccupantTimelineSteps -- is
# resolved here not by invention but by faithfully reconstructing
# MultiAgentSimulation's own, already-frozen node-occupancy bookkeeping
# (simulator/coordinator.py):
#
#   - _register() only ever calls _add_node_occupant(start_node.id, ...) for
#     an occupant with a *non-trivial*, *reachable* route (route.edges
#     non-empty and reached_goal True) -- an UNREACHABLE occupant, or one
#     whose trivial (zero-length) route means they were already at their
#     goal, is *never* added to node occupancy at all, at any point in the
#     live run. This bridge preserves that fact rather than inventing a
#     "they must be somewhere" fallback the live simulation itself does not
#     provide.
#   - _admit_onto_edge() removes the occupant from from_node's occupancy the
#     instant they are admitted onto an edge (OccupantTimelineStep.start_
#     time), and _handle_arrive_at_node() does not add them back to any
#     node's occupancy until that edge's OccupantTimelineStep.end_time --
#     meaning a live occupant contributes to *no* zone while mid-traversal.
#     This is not a new "mid-traversal" policy invented here; it is exactly
#     what the coordinator's own _node_occupancy set already does, replayed
#     against the persisted timeline instead of live heap state.
#   - _handle_arrive_at_node() removes the occupant from node occupancy
#     again, permanently, the instant they reach their *final* destination
#     (arrival_time == the last step's end_time) -- an arrived occupant has
#     evacuated, and contributes to no zone thereafter.
#
# Only OBSERVED-equivalent (here: "occupant found") node ids ever get an
# entry in the returned OccupancySnapshot -- a node no occupant is at right
# now is simply absent, exactly the "only touched ids get an entry"
# convention HazardEvolutionEngine/SensorFusion already use elsewhere in
# this codebase; OccupancySnapshot.observation_at()'s own total accessor
# already defaults an absent node_id to "no reading", which is the correct
# reading here since this bridge has no independent list of every zone in
# the building to assert an explicit zero against.


class MovementTimelineOccupancyProvider(OccupancyProvider):

    def __init__(self, movement_result: MultiAgentSimulationResult):

        self._movement_result = movement_result

    # =====================================================

    def snapshot_at(self, time: float) -> OccupancySnapshot:

        counts: Dict[str, float] = {}

        for timeline in self._movement_result.occupants.values():

            node_id = self._node_at(timeline, time)

            if node_id is not None:
                counts[node_id] = counts.get(node_id, 0.0) + 1.0

        observations = {
            node_id: OccupancyObservation(occupant_count=count)
            for node_id, count in counts.items()
        }

        return OccupancySnapshot(timestamp=time, observations=observations)

    # =====================================================

    def _node_at(self, timeline: OccupantTimeline, time: float) -> Optional[str]:

        # STATIONARY (route is None), UNREACHABLE, and trivial-route ARRIVED
        # occupants (steps == []) are never tracked in live node occupancy
        # either (see module docstring) -- not a gap this bridge introduces.
        if timeline.route is None or not timeline.route.edges or not timeline.steps:
            return None

        for step in timeline.steps:

            if time < step.start_time:
                # Not yet admitted onto this hop -- still at (or waiting,
                # possibly queued, at) the node this step departs from.
                # For the first step this is the occupant's start node.
                return step.from_node.id

            if time < step.end_time:
                # Mid-traversal -- removed from from_node's occupancy at
                # admission, not yet added to to_node's. Contributes to
                # neither zone, matching _node_occupancy exactly.
                return None

        # time >= the final step's end_time == arrival_time -- evacuated,
        # removed from node occupancy permanently.
        return None
