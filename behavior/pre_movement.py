class PreMovementDelayStrategy:

    # Navigation Stage interface: "how long before this occupant
    # actually starts walking" -- only ever consulted when the
    # Decision Stage determined movement is required.

    def delay(self, context) -> float:

        raise NotImplementedError


class NoPreMovementDelay(PreMovementDelayStrategy):

    # The trivial default -- proves the interface without implementing
    # any real behavior. Every occupant departs immediately, exactly
    # what MultiAgentSimulation.evacuate() already does today.

    def delay(self, context) -> float:

        return 0.0
