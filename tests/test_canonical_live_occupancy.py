import pathlib
import re
import unittest

from live_system.event_bus import EventBus, EventType
from live_system.live_command_center_gateway import frame_from_building_state

from live_occupants.manager import LiveOccupantManager
from live_occupants.occupancy import OccupancyFacts, compute_occupancy_facts

from crowd_intelligence.engine import CrowdIntelligenceEngine
from evacuation_progress.engine import EvacuationProgressEngine
from emergency_response.engine import EmergencyResponseIntelligenceEngine

from ai_features.building_state_extractor import extract_canonical_features

from models.building import Building
from models.door import Door
from models.exit import Exit
from models.floor import Floor
from models.zone import Zone

from live_runtime.factory import build_live_runtime

from tests.test_live_perception_double_counting import make_building, make_world_projector
from tests.live_camera_pipeline_fixtures import MockHumanDetector
from tests.human_detection_fixtures import FakeYOLOBackend, person
from human_detection.yolo_human_detector import YOLOHumanDetector
from tracking.simple_tracker import SimpleSingleCameraTracker
from live_camera_pipeline.identity_resolver import MappingIdentityResolver
from live_camera_pipeline.replay_frame_source import ReplayFrameSource


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


# =====================================================
# Canonical Live Occupancy Source of Truth milestone -- Phase 1's own
# investigation found FOUR independent "filter NEW/ACTIVE occupants,
# group by current_zone_id" implementations (live_perception.providers.
# LiveOccupantObservationProvider, crowd_intelligence.density.
# compute_zone_metrics, evacuation_progress.engine.
# EvacuationProgressEngine, emergency_response.engine.
# EmergencyResponseIntelligenceEngine) that happened to numerically
# agree today only by coincidence of independent, parallel construction
# -- never mechanically tied together. This file proves the fix:
# live_occupants.manager.LiveOccupantManager.canonical_occupancy() is
# now the ONE grouping every one of those reads.
# =====================================================


class OccupancyFactsTests(unittest.TestCase):

    class _FakeOccupant:

        def __init__(self, occupant_id, zone_id, floor_id):
            self.occupant_id = occupant_id
            self.current_zone_id = zone_id
            self.current_floor_id = floor_id

    def test_groups_by_zone_and_floor(self):

        occupants = [
            self._FakeOccupant("OCC-1", "zone-1", "floor-1"),
            self._FakeOccupant("OCC-2", "zone-1", "floor-1"),
            self._FakeOccupant("OCC-3", "zone-2", "floor-1"),
        ]

        facts = compute_occupancy_facts(occupants, timestamp=5.0)

        self.assertEqual(facts.timestamp, 5.0)
        self.assertEqual(set(facts.occupant_ids_by_zone["zone-1"]), {"OCC-1", "OCC-2"})
        self.assertEqual(facts.occupant_ids_by_zone["zone-2"], ("OCC-3",))
        self.assertEqual(set(facts.occupant_ids_by_floor["floor-1"]), {"OCC-1", "OCC-2", "OCC-3"})
        self.assertEqual(facts.total_observed_count, 3)
        self.assertEqual(facts.zone_count("zone-1"), 2)
        self.assertEqual(facts.zone_count("zone-nonexistent"), 0)

    def test_no_zone_id_is_unlocalized_not_fabricated(self):

        occupants = [self._FakeOccupant("OCC-1", None, "floor-1")]

        facts = compute_occupancy_facts(occupants, timestamp=1.0)

        self.assertEqual(facts.unlocalized_occupant_ids, ("OCC-1",))
        self.assertEqual(facts.occupant_ids_by_zone, {})
        self.assertEqual(facts.total_observed_count, 1)
        self.assertEqual(facts.unlocalized_count, 1)

    def test_empty_input_is_a_valid_zero_not_an_error(self):

        facts = compute_occupancy_facts([], timestamp=0.0)

        self.assertEqual(facts.total_observed_count, 0)
        self.assertEqual(facts.occupant_ids_by_zone, {})
        self.assertEqual(facts.unlocalized_occupant_ids, ())

    def test_is_immutable(self):

        facts = compute_occupancy_facts([self._FakeOccupant("OCC-1", "zone-1", "floor-1")], timestamp=1.0)

        with self.assertRaises(TypeError):
            facts.occupant_ids_by_zone["zone-2"] = ("OCC-X",)


class CanonicalOccupancyMemoizationTests(unittest.TestCase):

    def test_same_timestamp_returns_the_same_cached_object(self):

        manager = LiveOccupantManager()
        manager.update("OCC-1", "CAM-A", "T1", "zone-1", "floor-1", (1.0, 1.0), None, None, 0.9, 1.0)

        first = manager.canonical_occupancy(1.0)
        second = manager.canonical_occupancy(1.0)

        self.assertIs(first, second)

    def test_a_mutation_invalidates_the_cache_even_at_the_same_timestamp(self):

        manager = LiveOccupantManager()
        manager.update("OCC-1", "CAM-A", "T1", "zone-1", "floor-1", (1.0, 1.0), None, None, 0.9, 1.0)

        first = manager.canonical_occupancy(1.0)

        manager.update("OCC-2", "CAM-B", "T2", "zone-1", "floor-1", (2.0, 1.0), None, None, 0.9, 1.0)

        second = manager.canonical_occupancy(1.0)

        self.assertIsNot(first, second)
        self.assertEqual(first.total_observed_count, 1)
        self.assertEqual(second.total_observed_count, 2)

    def test_a_different_timestamp_recomputes(self):

        manager = LiveOccupantManager()
        manager.update("OCC-1", "CAM-A", "T1", "zone-1", "floor-1", (1.0, 1.0), None, None, 0.9, 1.0)

        first = manager.canonical_occupancy(1.0)
        second = manager.canonical_occupancy(2.0)

        self.assertIsNot(first, second)
        self.assertEqual(second.timestamp, 2.0)


# =====================================================
# Phase 10 -- THE required worked example, extended across every
# subsystem in ONE place (the pre-existing per-subsystem proofs in
# tests/test_live_perception_double_counting.py, tests/
# test_crowd_intelligence_double_counting.py, tests/
# test_evacuation_progress_double_counting.py, tests/
# test_emergency_response_double_counting.py each already prove their
# own slice in isolation -- unmodified, still passing -- this test is
# the NEW, additional proof that they all agree SIMULTANEOUSLY, from
# the SAME cycle, with the SAME occupant ids, because they now all read
# the SAME LiveOccupantManager.canonical_occupancy() result).
# =====================================================


class MultiCameraFourToThreeAllSubsystemsTests(unittest.TestCase):

    def setUp(self):

        self.building = make_building()

        backend_a = FakeYOLOBackend()
        backend_a.queue_result(
            person(confidence=0.9, box=(310.0, 230.0, 330.0, 250.0)),
            person(confidence=0.9, box=(320.0, 240.0, 340.0, 260.0)),
        )

        backend_b = FakeYOLOBackend()
        backend_b.queue_result(
            person(confidence=0.9, box=(310.0, 230.0, 330.0, 250.0)),
            person(confidence=0.9, box=(320.0, 240.0, 340.0, 260.0)),
        )

        frame_sources = {
            "CAM-A": ReplayFrameSource(camera_id="CAM-A", frames=[(0.0, "frame")]),
            "CAM-B": ReplayFrameSource(camera_id="CAM-B", frames=[(0.0, "frame")]),
        }
        for source in frame_sources.values():
            source.start()

        class _DispatchingDetector:
            def __init__(self, detectors_by_camera):
                self._by_camera = detectors_by_camera

            def detect(self, frame):
                return self._by_camera[frame.camera_id].detect(frame)

        human_detector = _DispatchingDetector({
            "CAM-A": YOLOHumanDetector(backend_a),
            "CAM-B": YOLOHumanDetector(backend_b),
        })

        identity_resolver = MappingIdentityResolver({
            ("CAM-A", "CAM-A-T1"): "OCC-ONLY-A",
            ("CAM-A", "CAM-A-T2"): "OCC-SHARED",
            ("CAM-B", "CAM-B-T1"): "OCC-SHARED",
            ("CAM-B", "CAM-B-T2"): "OCC-ONLY-B",
        })

        self.runtime = build_live_runtime(
            self.building,
            frame_sources=frame_sources,
            human_detector=human_detector,
            identity_resolver=identity_resolver,
            tracker=SimpleSingleCameraTracker(),
            world_projector=make_world_projector(),
        )

        self.runtime.start()

    def tearDown(self):
        self.runtime.stop()

    def test_every_subsystem_agrees_on_three_with_the_same_occupant_ids(self):

        self.runtime.run_cycle(0.0)

        expected_ids = {"OCC-ONLY-A", "OCC-SHARED", "OCC-ONLY-B"}

        # 1. Canonical source itself.
        facts = self.runtime.live_occupant_manager.canonical_occupancy(0.0)
        self.assertEqual(facts.total_observed_count, 3)
        self.assertEqual(set(facts.occupant_ids_by_zone["zone-1"]), expected_ids)

        # 2. BuildingState.zone_occupancy (via live_perception provider).
        building_state = self.runtime.orchestrator.latest_building_state
        occupancy = building_state.zone_occupancy.observation_at("zone-1").occupant_count
        self.assertEqual(occupancy, 3.0)

        # 3. Crowd Intelligence.
        crowd_snapshot = self.runtime.crowd_intelligence_engine.compute(0.0)
        self.assertEqual(crowd_snapshot.zone("zone-1").occupant_count, 3)
        self.assertEqual(crowd_snapshot.building_summary.total_observed_occupants, 3)

        # 4. Evacuation Progress.
        progress_snapshot = self.runtime.evacuation_progress_engine.compute(0.0, building_state, crowd_snapshot)
        self.assertEqual(progress_snapshot.known_active_occupants, 3)
        self.assertEqual(progress_snapshot.zone("zone-1").current_active_count, 3)

        # 5. Emergency Response.
        emergency_snapshot = self.runtime.emergency_response_engine.compute(
            0.0, building_state, crowd_snapshot, progress_snapshot,
        )
        self.assertEqual(emergency_snapshot.zone("zone-1").known_occupant_count, 3)
        occupant_evidence_ids = set()  # not populated without classification/state evidence in this fixture

        # 6. AI feature extraction -- total_occupant_count is DELIBERATELY
        # sourced from BuildingState.occupant_tracks (identity truth,
        # MultiCameraFusionEngine), never from zone_occupancy/canonical
        # occupancy directly (Phase 9's own "do not alter feature
        # schemas" instruction) -- verified here to still agree with the
        # canonical count in this common case, not modified to consume
        # it.
        features = extract_canonical_features(building_state)
        self.assertEqual(features["total_occupant_count"], 3)

        # 7. Command Center (Live mode frame conversion).
        frame = frame_from_building_state(building_state)
        self.assertEqual(frame.zone_occupancy["zone-1"], 3.0)

        # Same physical people everywhere -- not merely the same COUNT.
        self.assertEqual({o.occupant_id for o in self.runtime.live_occupant_manager.active_occupants()}, expected_ids)
        self.assertEqual(set(building_state.occupant_tracks.keys()), expected_ids)


# =====================================================
# A small, hand-built building for direct LiveOccupantManager.update()-
# driven lifecycle tests (Phase 11/12/13/14) -- these test lifecycle
# TRANSITIONS across cycles, which the camera-pipeline-driven fixture
# above is not built for (it only ever exercises a single cycle).
# =====================================================


def _make_lifecycle_building():

    floor = Floor(
        id="floor-1", name="Ground Floor",
        zones=[
            Zone(id="zone-1", name="Z1", x=0.0, y=0.0, width=10.0, height=10.0, floor_id="floor-1"),
            Zone(id="zone-2", name="Z2", x=20.0, y=0.0, width=10.0, height=10.0, floor_id="floor-1"),
        ],
        doors=[Door(id="DOOR-1", name="D1", floor_id="floor-1", zone_a_id="zone-1", zone_b_id="zone-2")],
        exits=[Exit(id="EXIT-1", name="Main Exit", floor_id="floor-1", zone_id="zone-1",
                    start_point=(0.0, 3.0), end_point=(0.0, 7.0))],
    )

    return Building(id="lifecycle-building", name="Lifecycle Building", floors=[floor])


class _Engines:

    def __init__(self, building, manager, event_bus):
        self.crowd = CrowdIntelligenceEngine(building, manager)
        self.progress = EvacuationProgressEngine(building, manager, event_bus)
        self.emergency = EmergencyResponseIntelligenceEngine(building, manager)

    def counts_for_zone(self, time, zone_id, building_state=None):

        crowd_snapshot = self.crowd.compute(time)
        progress_snapshot = self.progress.compute(time, building_state, crowd_snapshot)
        emergency_snapshot = self.emergency.compute(time, building_state, crowd_snapshot, progress_snapshot)

        return {
            "crowd": crowd_snapshot.zone(zone_id).occupant_count,
            "progress": progress_snapshot.zone(zone_id).current_active_count,
            "emergency": emergency_snapshot.zone(zone_id).known_occupant_count,
        }


class TemporarilyLostOccupantTests(unittest.TestCase):

    # Phase 11 -- cycle 1 ACTIVE in Z1, cycle 2 disappears (far from any
    # exit -> TEMPORARILY_LOST, per live_occupants.lifecycle's own
    # geometry-based rule), cycle 3 reappears with the SAME global
    # identity. No subsystem may independently decide whether OCC-1
    # counts -- every one of them must agree, every cycle, because every
    # one of them reads the same canonical_occupancy().

    def setUp(self):

        self.building = _make_lifecycle_building()
        self.event_bus = EventBus()
        self.manager = LiveOccupantManager(event_bus=self.event_bus, exits=[], expire_after_seconds=100.0)
        self.engines = _Engines(self.building, self.manager, self.event_bus)

    def test_lost_occupant_stops_counting_everywhere_then_recovers_everywhere(self):

        # Cycle 1 -- OCC-1 ACTIVE in zone-1.
        self.manager.update("OCC-1", "CAM-A", "T1", "zone-1", "floor-1", (5.0, 5.0), None, None, 0.9, 1.0)
        facts_1 = self.manager.canonical_occupancy(1.0)
        self.assertEqual(facts_1.zone_count("zone-1"), 1)
        counts_1 = self.engines.counts_for_zone(1.0, "zone-1")
        self.assertEqual(counts_1, {"crowd": 1, "progress": 1, "emergency": 1})

        # Cycle 2 -- nobody seen this cycle -> sweep_missing() demotes
        # OCC-1 to TEMPORARILY_LOST (far from the one exit at (0,3)-(0,7)).
        self.manager.sweep_missing(2.0, seen_occupant_ids=set())
        facts_2 = self.manager.canonical_occupancy(2.0)
        self.assertEqual(facts_2.zone_count("zone-1"), 0)
        self.assertEqual(facts_2.total_observed_count, 0)
        counts_2 = self.engines.counts_for_zone(2.0, "zone-1")
        self.assertEqual(counts_2, {"crowd": 0, "progress": 0, "emergency": 0})

        # Cycle 3 -- OCC-1 reappears, same global identity, same zone.
        self.manager.update("OCC-1", "CAM-A", "T1", "zone-1", "floor-1", (5.0, 5.0), None, None, 0.9, 3.0)
        facts_3 = self.manager.canonical_occupancy(3.0)
        self.assertEqual(facts_3.zone_count("zone-1"), 1)
        self.assertEqual(facts_3.occupant_ids_by_zone["zone-1"], ("OCC-1",))
        counts_3 = self.engines.counts_for_zone(3.0, "zone-1")
        self.assertEqual(counts_3, {"crowd": 1, "progress": 1, "emergency": 1})


class ZoneTransitionTests(unittest.TestCase):

    # Phase 12 -- OCC-1: Z1 -> Z1 -> Z2. At the transition cycle, Z1
    # decreases exactly once and Z2 increases exactly once, with every
    # subsystem agreeing -- occupant membership is single-zone by
    # construction (LiveOccupant.current_zone_id is one value, not a
    # set), so this is mechanically guaranteed once every subsystem
    # reads the SAME canonical grouping instead of independently
    # deciding membership.

    def setUp(self):

        self.building = _make_lifecycle_building()
        self.event_bus = EventBus()
        self.manager = LiveOccupantManager(event_bus=self.event_bus, exits=[], expire_after_seconds=100.0)
        self.engines = _Engines(self.building, self.manager, self.event_bus)

    def test_transition_moves_exactly_once_no_double_count_no_stale_membership(self):

        self.manager.update("OCC-1", "CAM-A", "T1", "zone-1", "floor-1", (5.0, 5.0), None, None, 0.9, 1.0)
        self.manager.update("OCC-1", "CAM-A", "T1", "zone-1", "floor-1", (5.0, 5.0), None, None, 0.9, 2.0)

        facts_before = self.manager.canonical_occupancy(2.0)
        self.assertEqual(facts_before.zone_count("zone-1"), 1)
        self.assertEqual(facts_before.zone_count("zone-2"), 0)

        # Transition cycle.
        self.manager.update("OCC-1", "CAM-A", "T1", "zone-2", "floor-1", (25.0, 5.0), None, None, 0.9, 3.0)

        facts_after = self.manager.canonical_occupancy(3.0)
        self.assertEqual(facts_after.zone_count("zone-1"), 0)  # decreased exactly once
        self.assertEqual(facts_after.zone_count("zone-2"), 1)  # increased exactly once
        self.assertEqual(facts_after.total_observed_count, 1)  # never double-counted across both zones

        counts = self.engines.counts_for_zone(3.0, "zone-2")
        self.assertEqual(counts, {"crowd": 1, "progress": 1, "emergency": 1})

        counts_z1 = self.engines.counts_for_zone(3.0, "zone-1")
        self.assertEqual(counts_z1, {"crowd": 0, "progress": 0, "emergency": 0})


class UnlocalizedOccupantTests(unittest.TestCase):

    # Phase 13 -- an occupant tracked (NEW/ACTIVE) but with no resolved
    # current_zone_id must survive in TOTAL OBSERVED, be visible as
    # UNLOCALIZED, and never be silently assigned a fake zone.

    def setUp(self):

        self.building = _make_lifecycle_building()
        self.event_bus = EventBus()
        self.manager = LiveOccupantManager(event_bus=self.event_bus, exits=[])
        self.engines = _Engines(self.building, self.manager, self.event_bus)

    def test_unlocalized_occupant_counted_in_total_never_in_a_zone(self):

        self.manager.update("OCC-LOCALIZED", "CAM-A", "T1", "zone-1", "floor-1", (5.0, 5.0), None, None, 0.9, 1.0)
        self.manager.update("OCC-UNLOCALIZED", "CAM-B", "T2", None, "floor-1", None, None, None, 0.9, 1.0)

        facts = self.manager.canonical_occupancy(1.0)

        self.assertEqual(facts.total_observed_count, 2)  # both survive
        self.assertEqual(facts.unlocalized_occupant_ids, ("OCC-UNLOCALIZED",))
        self.assertEqual(facts.occupant_ids_by_zone["zone-1"], ("OCC-LOCALIZED",))

        # No zone anywhere contains the unlocalized occupant -- checked
        # across every known zone, not fabricated into any of them.
        for zone_id, ids in facts.occupant_ids_by_zone.items():
            self.assertNotIn("OCC-UNLOCALIZED", ids)

        # Crowd Intelligence: zone-1 density reflects only the localized
        # occupant; the building-wide total still counts both.
        crowd_snapshot = self.engines.crowd.compute(1.0)
        self.assertEqual(crowd_snapshot.zone("zone-1").occupant_count, 1)
        self.assertEqual(crowd_snapshot.building_summary.total_observed_occupants, 2)


class ExitedExpiredSemanticsTests(unittest.TestCase):

    # Phase 14 -- ACTIVE -> EXITED -> EXPIRED, per live_occupants'
    # actual vocabulary. Current occupancy must decrease at the
    # semantically correct point (the moment they stop being counted as
    # NEW/ACTIVE, i.e. as soon as they go missing near an exit --
    # OccupantStatus.EXITED is itself already excluded from
    # active_occupants(), same as TEMPORARILY_LOST). Evacuation
    # Progress's own durable ledger (evacuation_progress.ledger.
    # EvacuationLedger, event-driven, unmodified by this milestone) must
    # still remember them as historically exited even after EXPIRED
    # removes them from LiveOccupantManager's own live store entirely.

    def setUp(self):

        self.building = _make_lifecycle_building()
        self.event_bus = EventBus()

        exit_obj = self.building.ordered_floors()[0].exits[0]

        # near_exit is computed from world_position vs Exit geometry --
        # (1.0, 5.0) is within the default 2.0m threshold of the exit
        # segment (0,3)-(0,7).
        self.manager = LiveOccupantManager(
            event_bus=self.event_bus, exits=[exit_obj], expire_after_seconds=5.0,
        )
        self.progress = EvacuationProgressEngine(self.building, self.manager, self.event_bus)

    def test_current_occupancy_drops_at_exit_survives_in_progress_ledger_after_expiry(self):

        self.manager.update("OCC-1", "CAM-A", "T1", "zone-1", "floor-1", (1.0, 5.0), None, None, 0.9, 1.0)

        facts_active = self.manager.canonical_occupancy(1.0)
        self.assertEqual(facts_active.total_observed_count, 1)

        # Cycle 2: OCC-1 not seen, was near the exit -> EXITED (not
        # TEMPORARILY_LOST) -- current occupancy drops to 0 immediately.
        self.manager.sweep_missing(2.0, seen_occupant_ids=set())

        facts_exited = self.manager.canonical_occupancy(2.0)
        self.assertEqual(facts_exited.total_observed_count, 0)

        progress_snapshot = self.progress.compute(2.0, None, None)
        self.assertEqual(progress_snapshot.known_active_occupants, 0)
        self.assertEqual(progress_snapshot.known_exited_occupants, 1)  # durable ledger already recorded it

        # Cycle far enough later that OCC-1 exceeds expire_after_seconds
        # (5.0s) -- sweep_missing() removes them from the manager
        # entirely (EXPIRED, terminal).
        self.manager.sweep_missing(10.0, seen_occupant_ids=set())

        self.assertIsNone(self.manager.get("OCC-1"))  # gone from the live store

        facts_expired = self.manager.canonical_occupancy(10.0)
        self.assertEqual(facts_expired.total_observed_count, 0)  # still honestly zero, not re-added

        # The durable ledger still remembers -- historical evacuation
        # progress survives even though the occupant is fully gone from
        # LiveOccupantManager's own live store.
        final_progress = self.progress.compute(10.0, None, None)
        self.assertEqual(final_progress.known_exited_occupants, 1)
        self.assertEqual(final_progress.known_total_observed_occupants, 1)


# =====================================================
# Phase 18 -- architecture guards: mechanically prove every consumer
# that needs a "current live occupancy, grouped by zone" answer reads
# LiveOccupantManager.canonical_occupancy() rather than re-implementing
# its own independent grouping loop.
# =====================================================


class NoDuplicatedOccupancyGroupingGuardTests(unittest.TestCase):

    _MUST_CALL_CANONICAL_OCCUPANCY = (
        ("live_perception", "providers.py"),
        ("crowd_intelligence", "engine.py"),
        ("evacuation_progress", "engine.py"),
        ("emergency_response", "engine.py"),
    )

    def test_every_occupancy_consumer_calls_canonical_occupancy(self):

        for package, filename in self._MUST_CALL_CANONICAL_OCCUPANCY:

            text = (REPO_ROOT / package / filename).read_text(encoding="utf-8")

            self.assertIn(
                "canonical_occupancy(", text,
                f"{package}/{filename} must read live_occupant_manager.canonical_occupancy() -- "
                f"never independently re-filter/group active_occupants() by current_zone_id itself "
                f"(Canonical Live Occupancy Source of Truth milestone, Phase 18).",
            )

    def test_live_occupants_package_still_imports_nothing_forbidden(self):

        # The new live_occupants/occupancy.py module must obey the SAME
        # package-boundary guard every other live_occupants/ file
        # already does (tests/test_live_occupants_architecture_guards.py,
        # unmodified) -- re-verified here directly for this one new file.

        text = (REPO_ROOT / "live_occupants" / "occupancy.py").read_text(encoding="utf-8")

        forbidden = (
            r"^\s*(from|import)\s+("
            r"ai_engine|reinforcement_learning|advisory_system|command_center|"
            r"building_state|multi_camera_fusion|camera_manager|live_system\."
            r")\b"
        )

        self.assertIsNone(re.search(forbidden, text, re.MULTILINE))


# =====================================================
# Phase 16 -- Application-level E2E. Exercises the ACTUAL application
# entry path this repository now has (designer.windows.main_window.
# MainWindow -> its real live_runtime_panel/live_runtime_controller ->
# live_runtime_launcher.session.LiveRuntimeSession, the Application Live
# Runtime Launcher milestone's own new production seam), never
# build_live_runtime() called directly -- proving the canonical
# occupancy fix is reachable from, and internally consistent within,
# the real application, not merely from a hand-assembled test chain.
# =====================================================


class ApplicationLevelOccupancyConsistencyE2ETests(unittest.TestCase):

    def setUp(self):

        import sys
        from PyQt6.QtWidgets import QApplication

        self._app = QApplication.instance() or QApplication(sys.argv)

        from models.project import Project
        from designer.windows.main_window import MainWindow
        from tests.live_runtime_fixtures import make_demo_building

        self.window = MainWindow()
        self.window.canvas.scene_obj.project = Project(name="Occupancy Demo", building=make_demo_building())

    def tearDown(self):

        self.window.live_runtime_controller.shutdown()

    def test_command_center_reaches_internally_consistent_occupancy_via_the_real_app_entry_path(self):

        panel = self.window.live_runtime_panel
        panel.mode_combo.setCurrentIndex(1)  # Offline Demo
        panel.start_button.click()

        session = self.window.live_runtime_controller.session
        self.assertTrue(session.is_running)

        runtime = session.runtime

        # Perception bypassed on purpose (same "ProductionWiringOfflineE2ETests"
        # precedent tests/test_live_dynamic_signage_operator_workflow.py
        # already established) -- occupants driven directly into the
        # SAME live_occupant_manager the application-constructed
        # LiveRuntime itself owns, exactly as a real camera pipeline
        # would via LiveCameraPipeline.run_cycle().
        runtime.live_occupant_manager.update(
            "OCC-1", "CAM-LOBBY", "T1", "zone-lobby", "floor-1", (1.0, 1.0), None, None, 0.9, 1.0,
        )
        runtime.live_occupant_manager.update(
            "OCC-2", "CAM-LOBBY", "T2", "zone-lobby", "floor-1", (2.0, 1.0), None, None, 0.9, 1.0,
        )

        runtime.run_cycle(1.0)

        snapshot = runtime.command_center_data_source.current_snapshot()

        self.assertEqual(snapshot.building_state.zone_occupancy.observation_at("zone-lobby").occupant_count, 2.0)
        self.assertEqual(snapshot.evacuation_progress.known_active_occupants, 2)
        self.assertEqual(snapshot.emergency_response.zone("zone-lobby").known_occupant_count, 2)

        crowd_snapshot = runtime.crowd_intelligence_engine.compute(1.0)
        self.assertEqual(crowd_snapshot.zone("zone-lobby").occupant_count, 2)

        # Open the real Command Center window against this SAME runtime
        # (Application Live Runtime Launcher milestone's own Phase 6
        # guarantee) and confirm it is looking at the identical snapshot.
        panel.open_command_center_button.click()
        command_center_window = session._command_center_window
        self.assertIs(command_center_window.live_data_source, runtime.command_center_data_source)


if __name__ == "__main__":
    unittest.main()
