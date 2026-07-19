from simulator.capacity import StairCapacityModel
from simulator.congestion import StairAwareCongestionModel
from simulator.coordinator import MultiAgentSimulation


# Occupant initialization -- architecture doc
# docs/architecture/scenario_runner.md §6. "Positioning uniformly"
# does not mean re-sampling: ScenarioOccupant.position is already a
# uniformly-sampled point, frozen onto the Scenario by
# scenario_generator. This module draws no new random value and
# resolves nothing -- it carries scenario.occupants through field for
# field, and constructs the MultiAgentSimulation those occupants will
# eventually be registered into, empty.
#
# behaviour_profile_id is passed through completely unread -- this
# module (and this package) must never import behavior/
# behavior_library, never construct a BehaviorProfile, and never call
# HumanBehaviorLayer.register()/MultiAgentSimulation.submit_decision().
# Interpreting behaviour_profile_id is exclusively the future
# Behaviour Profile Resolver's job (§7).


def build_occupants(scenario):

    # No new type, no transformation -- ScenarioOccupant already has
    # exactly the shape a later phase needs (§6).
    return tuple(scenario.occupants)


def build_firefighters(scenario):

    # Human Population & Assisted Evacuation Modeling Phase 5 --
    # carried through exactly as opaquely as build_occupants() already
    # carries scenario.occupants (behaviour_profile_id, rescue
    # assignments, and every other field are interpreted exclusively by
    # behaviour_profile_resolver.firefighter_registrar, never here).
    return tuple(scenario.firefighters)


def build_simulation(engine):

    # Constructed and wrapping `engine` -- otherwise empty. No
    # add_occupant()/submit_decision() call is ever made here.
    #
    # capacity_model/congestion_model are the one production wiring
    # point for stair-aware capacity/congestion (Stair Capacity
    # Modeling) -- both wrap the same DefaultCapacityModel/
    # DefaultCongestionModel used everywhere else and only special-
    # case Stair edges, so every Door/Exit crossing behaves exactly
    # as before.
    return MultiAgentSimulation(
        engine,
        capacity_model=StairCapacityModel(),
        congestion_model=StairAwareCongestionModel(),
    )
