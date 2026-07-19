from dataclasses import dataclass, field
from typing import Tuple

from models.building import Building

from navigation.graph import NavigationGraph

from pathfinding.engine import PathfindingEngine

from simulator.coordinator import MultiAgentSimulation

from hazard.snapshot import HazardSnapshot
from hazard_evolution.engine import HazardEvolutionEngine

from scenario import ScenarioEvent, ScenarioFirefighter, ScenarioMetadata, ScenarioOccupant


@dataclass(frozen=True)
class SimulationContext:

    # The Scenario Runner's one output -- architecture doc
    # docs/architecture/scenario_runner.md §4, this pass. An immutable
    # bundle of already-existing types, not a new simulation engine of
    # its own, and NOT run-ready: `simulation` is constructed but
    # empty (no add_occupant()/submit_decision() call has been made,
    # §6), and `occupants`' behaviour_profile_id is carried through
    # exactly as opaque as it arrived (§7) -- completing registration
    # is the future Behaviour Profile Resolver's job, entirely outside
    # this package.
    #
    # `simulation` being a *mutable* MultiAgentSimulation instance
    # living inside an *immutable* SimulationContext is the same
    # pattern scenario_pipeline.PipelineResult already establishes
    # (frozen, holding a mutable ScenarioValidationReport) -- this
    # dataclass's own fields are never reassigned after construction;
    # what a later phase does to the objects those fields point to is
    # expected, ordinary runtime mutation of a component designed to
    # accumulate state, not a violation of the container's own
    # immutability (§4).

    building: Building
    graph: NavigationGraph
    engine: PathfindingEngine
    simulation: MultiAgentSimulation

    hazard_engine: HazardEvolutionEngine
    initial_hazard_snapshot: HazardSnapshot

    occupants: Tuple[ScenarioOccupant, ...] = field(default_factory=tuple)
    scheduled_events: Tuple[ScenarioEvent, ...] = field(default_factory=tuple)

    # Human Population & Assisted Evacuation Modeling Phase 5 --
    # additive, carried through verbatim from Scenario.firefighters the
    # same "no interpretation here" way `occupants` already is (§6/§7
    # apply identically: only behaviour_profile_resolver interprets
    # either). Every existing SimulationContext construction that
    # predates this field is unaffected, defaulting to no firefighters
    # deployed.
    firefighters: Tuple[ScenarioFirefighter, ...] = field(default_factory=tuple)

    # Carried through verbatim from Scenario.metadata (§4's own
    # "verbatim carry-through, never interpreted" principle, applied
    # here too) -- the provenance a caller needs to know *which*
    # Scenario this context was built from, without this package
    # re-deriving or reshaping anything ScenarioMetadata already
    # states.
    metadata: ScenarioMetadata = None

    # =====================================================

    def __post_init__(self):

        object.__setattr__(self, "occupants", tuple(self.occupants))
        object.__setattr__(self, "scheduled_events", tuple(self.scheduled_events))
        object.__setattr__(self, "firefighters", tuple(self.firefighters))
