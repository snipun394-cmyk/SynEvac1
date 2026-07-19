# Event initialization -- architecture doc
# docs/architecture/scenario_runner.md §10. No execution mechanism for
# a ScenarioEvent exists anywhere in the codebase today (§2/§10) --
# this module does not invent one. It only transports scenario.events,
# already resolved and already time-ordered (scenario_validator's own
# Event Validation already checks this), into the SimulationContext
# verbatim. No scheduling data structure, no event engine, no
# execution.


def build_scheduled_events(scenario):

    return tuple(scenario.events)
