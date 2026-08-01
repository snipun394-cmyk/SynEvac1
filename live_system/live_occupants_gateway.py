from typing import Optional, Protocol

from live_occupants.models import LiveOccupantsSnapshot


# =====================================================
# Expose Real Live Camera Occupant State In SynEvac UI milestone -- the
# seam LiveOrchestrator uses to reach live_occupants/, mirroring every
# sibling Engine*Gateway in this module (crowd_intelligence_gateway,
# evacuation_progress_gateway, trajectory_intelligence_gateway, ...)
# exactly: thin Protocol + a real adapter that never raises out of
# compute() and crash the live cycle.
#
# Unlike its siblings, there is no separate "Engine" to wrap here --
# live_occupants.manager.LiveOccupantManager is ALREADY the correctly-
# computed source of truth (canonical_occupancy()/active_occupants()),
# updated as a side effect of camera_pipeline.run_cycle() inside this
# same overall cycle's building_state_gateway.collect() call. This
# gateway performs NO tracking/fusion/lifecycle logic of its own -- it
# only ever reads already-computed LiveOccupantManager state and
# packages it into the one immutable snapshot type Command Center can
# render (Phase requirement: "the UI consumes state, it does not
# determine state").
# =====================================================


class LiveOccupantsGateway(Protocol):

    def compute(self, time: float) -> Optional[LiveOccupantsSnapshot]: ...


# =====================================================


class EngineLiveOccupantsGateway:

    # `live_occupant_manager` is deliberately untyped (not `live_occupants.
    # manager.LiveOccupantManager`) -- live_occupants.manager itself
    # imports live_system.event_bus, so importing it here at module level
    # would create a live_system <-> live_occupants import cycle the
    # instant live_system's own package __init__ eagerly imports
    # orchestrator (which imports this file). Every sibling Engine*Gateway
    # avoids this because crowd_intelligence/trajectory_intelligence/etc.
    # never import live_system themselves; live_occupants is the one
    # domain package that does, so this is the one gateway that must stay
    # untyped here (duck-typed: anything exposing all_occupants()/
    # canonical_occupancy() works, exactly LiveOccupantManager's own
    # public surface).

    def __init__(self, live_occupant_manager):

        self._live_occupant_manager = live_occupant_manager

    # =====================================================

    def compute(self, time: float) -> Optional[LiveOccupantsSnapshot]:

        try:

            return LiveOccupantsSnapshot(
                timestamp=time,
                occupants=self._live_occupant_manager.all_occupants(),
                current_occupant_count=self._live_occupant_manager.canonical_occupancy(time).total_observed_count,
            )

        except Exception:  # noqa: BLE001 -- an unexpected occupant-state failure must never crash the live cycle

            return None
