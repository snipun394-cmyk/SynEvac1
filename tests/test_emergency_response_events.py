import unittest

from behavior_recognition.observation import RecognizedBehavior

from models.building import Building
from models.camera import Camera
from models.engineering_asset import DeviceMode
from models.floor import Floor
from models.zone import Zone

from live_runtime.factory import build_offline_demo_runtime

from live_system.event_bus import EventBus, EventType

from live_occupants.manager import LiveOccupantManager

from evacuation_progress.engine import EvacuationProgressEngine
from crowd_intelligence.trends import TrendConfig


# =====================================================
# Live Emergency Response & Rescue Priority Intelligence milestone,
# Phase 22 items 14, 15, 17 -- temporal event behavior, driven through
# the real production LiveOrchestrator (not a hand-rolled snapshot
# comparison), so this proves the ACTUAL transition-detection wiring in
# live_system.orchestrator.LiveOrchestrator._emit_emergency_response_
# transition_events(), not merely the engine's own scoring in isolation.
# =====================================================


def make_runtime():

    floor = Floor(
        id="f1", name="Floor 1",
        zones=[Zone(id="z1", name="Z1", x=0.0, y=0.0, width=10.0, height=10.0, floor_id="f1")],
        cameras=[Camera(id="CAM-1", name="C1", floor_id="f1", zone_ids=("z1",))],
    )
    building = Building(id="b1", name="B", floors=[floor])

    # A deliberately sub-cycle-length trend_window_seconds -- these
    # tests move at whole-second cadence, and this package's own
    # zone-clearance trend (see evacuation_progress/engine.py's
    # _map_zone_trend()) honestly reports STALLED the moment the SAME
    # occupant is observed twice with no clearance change, which would
    # otherwise inject EVACUATION_STALLED evidence into every one of
    # these single-occupant scenarios and mask the assistance-signal
    # escalation/de-escalation behavior these tests exist to isolate.
    event_bus = EventBus()
    live_occupant_manager = LiveOccupantManager(event_bus=event_bus, exits=building.floors[0].exits)
    evacuation_progress_engine = EvacuationProgressEngine(
        building, live_occupant_manager, event_bus, trend_config=TrendConfig(trend_window_seconds=0.5),
    )

    runtime = build_offline_demo_runtime(
        building, event_bus=event_bus, live_occupant_manager=live_occupant_manager,
        evacuation_progress_engine=evacuation_progress_engine,
    )
    runtime.start()
    runtime.camera_manager.set_camera_mode("CAM-1", DeviceMode.LIVE)

    return runtime


class HeldHighNoSpamTests(unittest.TestCase):

    def test_14_held_high_for_multiple_cycles_produces_no_repeated_escalation_spam(self):

        runtime = make_runtime()
        manager = runtime.live_occupant_manager

        manager.update("OCC-1", "CAM-1", "T1", "z1", "f1", (1.0, 1.0), 0.0, RecognizedBehavior.WALKING, 0.9, 0.0)
        runtime.orchestrator.run_cycle(0.0)

        manager.update("OCC-1", "CAM-1", "T1", "z1", "f1", (1.0, 1.0), 0.0, RecognizedBehavior.POSSIBLY_FALLEN, 0.9, 1.0)
        runtime.orchestrator.run_cycle(1.0)  # escalates once here

        for t in (2.0, 3.0, 4.0):
            manager.update("OCC-1", "CAM-1", "T1", "z1", "f1", (1.0, 1.0), 0.0, RecognizedBehavior.POSSIBLY_FALLEN, 0.9, t)
            runtime.orchestrator.run_cycle(t)

        escalations = runtime.orchestrator.event_bus.history_of(EventType.ZONE_RESPONSE_ESCALATED)
        assistance_events = runtime.orchestrator.event_bus.history_of(EventType.POSSIBLE_ASSISTANCE_DETECTED)

        self.assertEqual(len(escalations), 1)
        self.assertEqual(len(assistance_events), 1)

        runtime.stop()

    def test_13_zone_escalation_fires_exactly_once_on_the_transition_cycle(self):

        runtime = make_runtime()
        manager = runtime.live_occupant_manager

        manager.update("OCC-1", "CAM-1", "T1", "z1", "f1", (1.0, 1.0), 0.0, RecognizedBehavior.WALKING, 0.9, 0.0)
        runtime.orchestrator.run_cycle(0.0)

        manager.update("OCC-1", "CAM-1", "T1", "z1", "f1", (1.0, 1.0), 0.0, RecognizedBehavior.POSSIBLY_FALLEN, 0.9, 1.0)
        runtime.orchestrator.run_cycle(1.0)

        escalations = runtime.orchestrator.event_bus.history_of(EventType.ZONE_RESPONSE_ESCALATED)
        self.assertEqual(len(escalations), 1)
        self.assertEqual(escalations[0].payload.zone_id, "z1")

        runtime.stop()


class DeescalationTests(unittest.TestCase):

    def test_15_zone_deescalates_after_clearance(self):

        runtime = make_runtime()
        manager = runtime.live_occupant_manager

        manager.update("OCC-1", "CAM-1", "T1", "z1", "f1", (1.0, 1.0), 0.0, RecognizedBehavior.POSSIBLY_FALLEN, 0.9, 0.0)
        runtime.orchestrator.run_cycle(0.0)

        # Occupant seen again, clearly walking (no more assistance signal).
        manager.update("OCC-1", "CAM-1", "T1", "z1", "f1", (1.0, 1.0), 0.0, RecognizedBehavior.WALKING, 0.9, 1.0)
        runtime.orchestrator.run_cycle(1.0)

        deescalations = runtime.orchestrator.event_bus.history_of(EventType.ZONE_RESPONSE_DEESCALATED)
        self.assertGreaterEqual(len(deescalations), 1)

        runtime.stop()


class ReappearanceTests(unittest.TestCase):

    def test_17_reappearing_occupant_updates_zone_priority_correctly(self):

        runtime = make_runtime()
        manager = runtime.live_occupant_manager

        manager.update("OCC-1", "CAM-1", "T1", "z1", "f1", (1.0, 1.0), 0.0, RecognizedBehavior.WALKING, 0.9, 0.0)
        runtime.orchestrator.run_cycle(0.0)
        with_occupant = runtime.orchestrator.latest_emergency_response.zone("z1").known_occupant_count
        self.assertEqual(with_occupant, 1)

        manager.sweep_missing(60.0, seen_occupant_ids=set())  # occupant goes missing (away from any exit)
        runtime.orchestrator.run_cycle(60.0)
        while_missing = runtime.orchestrator.latest_emergency_response.zone("z1").known_occupant_count
        self.assertEqual(while_missing, 0)

        manager.update("OCC-1", "CAM-1", "T1", "z1", "f1", (1.0, 1.0), 0.0, RecognizedBehavior.WALKING, 0.9, 61.0)
        runtime.orchestrator.run_cycle(61.0)
        after_reappearance = runtime.orchestrator.latest_emergency_response.zone("z1").known_occupant_count
        self.assertEqual(after_reappearance, 1)

        runtime.stop()


if __name__ == "__main__":
    unittest.main()
