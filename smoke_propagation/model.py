from typing import Optional

from navigation.edge import Edge

from hazard.node_state import HazardNodeState

from hazard_evolution.contribution import HazardContribution
from hazard_evolution.source import HazardSource

from smoke_propagation.graph_distance import shortest_graph_distances
from smoke_propagation.growth_curve import SmokeGrowthCurve, TSquaredSmokeGrowthCurve
from smoke_propagation.visibility import visibility_from_smoke_level


class SmokePropagationModel(HazardSource):

    # A single smoke source spreading outward from one ignition node
    # through explicit Door/Stair connectivity -- implements
    # HazardSource exactly like FireGrowthModel, so it plugs into
    # HazardEvolutionEngine with zero engine changes (see
    # hazard_evolution/source.py). Produces HazardContribution objects
    # only -- propose() has no other effect and no other output, so
    # this class never imports (and never could reach) Simulation,
    # Behavior, Pathfinding, or Building Analysis.
    #
    # The one thing this class reads from Navigation is the graph's
    # own static topology (Node.id, Edge.edge_type, Edge.
    # walking_distance), taken once at construction via
    # shortest_graph_distances() and cached -- never PathfindingEngine,
    # never a CostModel, and the graph reference itself is never
    # touched again after __init__. That is a one-time structural
    # read of "what connects to what", not a dependency on Navigation's
    # routing behavior.
    #
    # Deliberately stateless with respect to simulation history, same
    # as FireGrowthModel: propose() never reads previous_snapshot,
    # every node's smoke_level is recomputed fresh from elapsed
    # wall-clock time since the smoke front reached that node, so
    # repeated small evolve() steps land on the same result as one
    # larger step at the same absolute time.
    #
    # Multiple ignition points, exactly like FireGrowthModel: register
    # one SmokePropagationModel per ignition point as separate entries
    # in HazardEvolutionEngine(sources=[...]).
    #
    # No CFD, no ventilation physics, no temperature/toxicity/CO/CO2/
    # FED -- all deliberately absent; only smoke_level and visibility
    # are ever written to a HazardNodeState here.
    #
    # Dilution: smoke that has traveled further from its source has
    # measurably mixed with more clean air than smoke still near the
    # source -- without this, every reached node eventually saturates
    # at the identical peak intensity, merely later, which is not
    # dilution at all. dilution_half_distance scales the growth curve's
    # own output by 1 / (1 + distance / dilution_half_distance): 1.0
    # (no attenuation) at the ignition node itself, 0.5 at exactly one
    # half-distance away, approaching 0 further out. An arbitrary,
    # documented placeholder shape (not a validated concentration/
    # mixing model), same honesty as every other constant in this
    # class. Defaults to None (dilution off, exact previous behavior)
    # so this class's existing constructor signature and every existing
    # test/caller is completely unaffected -- the production wiring in
    # scenario_runner/fire_initializer.py is the one place that opts in
    # explicitly with a real value.

    DEFAULT_GROWTH_TIME = 180.0  # seconds -- an arbitrary, documented
                                  # placeholder (smoke logging a space
                                  # plausibly builds up faster than the
                                  # fire itself reaches full
                                  # development there), not a validated
                                  # design-fire value. Always
                                  # overridable via the growth_curve
                                  # argument.

    # An arbitrary, documented placeholder for how fast the visible
    # smoke front advances along graph connectivity -- not a validated
    # fluid-dynamics value. Always overridable via the front_speed
    # argument; a future ventilation-aware model would compute this
    # per-edge instead of as one constant.
    DEFAULT_FRONT_SPEED_M_PER_S = 0.5

    # Smoke propagates through Door and Stair edges only -- never
    # Exit (smoke logging the exterior "Outside" node has no meaning
    # here) and never inferred from drawing geometry. Stair edges
    # already connect nodes on different floors directly (see
    # NavigationGraphGenerator._add_stair_edges), so multi-floor
    # spread falls out of this list with no floor-specific code
    # anywhere in this class.
    PROPAGATED_EDGE_TYPES = (Edge.DOOR, Edge.STAIR)

    def __init__(
        self,
        graph,
        ignition_node_id: str,
        ignition_time: float,
        growth_curve: Optional[SmokeGrowthCurve] = None,
        front_speed: float = DEFAULT_FRONT_SPEED_M_PER_S,
        dilution_half_distance: Optional[float] = None,
    ):

        if front_speed <= 0:
            raise ValueError(
                f"SmokePropagationModel.front_speed must be > 0, "
                f"got {front_speed!r} -- a non-positive speed would "
                f"never let the smoke front arrive anywhere."
            )

        if dilution_half_distance is not None and dilution_half_distance <= 0:
            raise ValueError(
                f"SmokePropagationModel.dilution_half_distance must be > 0 or None, "
                f"got {dilution_half_distance!r}."
            )

        self.ignition_node_id = ignition_node_id
        self.ignition_time = ignition_time
        self.growth_curve = growth_curve or TSquaredSmokeGrowthCurve(self.DEFAULT_GROWTH_TIME)
        self.front_speed = front_speed
        self.dilution_half_distance = dilution_half_distance

        # Static graph topology, read once and cached -- see class
        # docstring. Never recomputed in propose(), never re-reads
        # `graph` again after this line.
        self._distances = shortest_graph_distances(
            graph, ignition_node_id, self.PROPAGATED_EDGE_TYPES,
        )

    # =====================================================

    def _dilution_factor(self, distance: float) -> float:

        if self.dilution_half_distance is None:
            return 1.0

        return 1.0 / (1.0 + distance / self.dilution_half_distance)

    # =====================================================

    def propose(self, previous_snapshot, time, dt) -> HazardContribution:

        step_end_time = time + dt

        node_states = {}

        for node_id, distance in self._distances.items():

            arrival_time = self.ignition_time + distance / self.front_speed

            if step_end_time < arrival_time:
                continue

            elapsed_time = step_end_time - arrival_time
            smoke_level = self.growth_curve.intensity_at(elapsed_time) * self._dilution_factor(distance)

            node_states[node_id] = HazardNodeState(
                smoke_level=smoke_level,
                visibility=visibility_from_smoke_level(smoke_level),
                hazard_score=smoke_level,
            )

        return HazardContribution(node_states=node_states)
