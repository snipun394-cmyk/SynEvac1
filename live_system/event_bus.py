# Shadow-Mode Predictive AI Integration milestone, Phase 1 -- the real
# implementation moved to event_bus/bus.py (see that module's own
# docstring for why: this type has zero project imports of its own, but
# living inside live_system/, whose __init__.py eagerly imports the
# entire package, forced every consumer to transitively load ai_registry/
# orchestrator/sensor_registry just to get EventBus/EventType -- the
# Core Architecture Freeze Review's own Finding 1).
#
# This module is now a THIN, BACKWARD-COMPATIBLE re-export only -- every
# existing `from live_system.event_bus import EventBus` (or `from
# live_system import EventBus`) call site keeps working completely
# unchanged, byte-identical behavior, zero API break. New/updated
# production call sites (live_occupants, evacuation_progress,
# command_center, live_runtime) import from event_bus.bus directly
# instead, which is what actually removes the package-level circular
# dependency for THEM -- this shim exists purely so every other existing
# caller (tests, scripts, and live_system's own internal modules) never
# had to change.
from event_bus.bus import Event, EventBus, EventType, Handler

__all__ = ["Event", "EventBus", "EventType", "Handler"]
