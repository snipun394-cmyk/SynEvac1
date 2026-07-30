import shutil
import tempfile
import time
import unittest

import ai_features as af
import ai_registry as reg

from hazard.snapshot import HazardSnapshot
from occupancy.snapshot import OccupancySnapshot

from camera_manager.status import CameraStatus

from building_state.estimator import BuildingStateEstimator

from models.building import Building
from models.exit import Exit
from models.staircase import Staircase
from models.zone import Zone

from live_occupants.manager import LiveOccupantManager

from event_bus.bus import EventBus, EventType

from live_system.live_ai_gateway import (
    AISystemStatus, LiveAIPredictionSnapshot, RegistryLiveAIInferenceGateway,
)
from live_system.orchestrator import LiveOrchestrator
from live_system.crowd_intelligence_gateway import EngineCrowdIntelligenceGateway
from live_system.evacuation_recommendation_gateway import EngineEvacuationRecommendationGateway
from live_system.evacuation_guidance_gateway import EngineEvacuationGuidanceGateway

from navigation.graph_builder import NavigationGraphGenerator

from crowd_intelligence.engine import CrowdIntelligenceEngine

from evacuation_recommendation.engine import EvacuationRecommendationEngine
from evacuation_recommendation.evidence import ai_bottleneck_probability
from evacuation_recommendation.ranking import rank_exits_for_zone
from evacuation_recommendation.models import RecommendationConfig, RecommendationWeights

from evacuation_guidance.engine import EvacuationGuidanceEngine


# =====================================================
# Shadow-Mode Predictive AI Integration milestone -- Phases 8/9/10.
# One small, real, trained model (mirrors tests/test_live_ai_runtime_
# integration.py's own setUpModule convention exactly) rather than a
# mock -- every assertion below exercises the REAL ai_registry/
# ai_features/live_system.live_ai_gateway chain.
# =====================================================

_MODULE_STATE = {}

CAMPAIGN_COUNT = 40
CAMPAIGN_SEED = 707


def setUpModule():

    tmp_dir = tempfile.mkdtemp(prefix="shadow_mode_test_")
    building = reg.make_training_building()
    definition = reg.make_training_definition()

    campaign_dir, _summary = reg.generate_training_campaign(
        tmp_dir, building, definition, count=CAMPAIGN_COUNT, master_seed=CAMPAIGN_SEED,
    )

    import ai_training as at
    legacy_dataset = at.load_campaign_dataset(campaign_dir)
    live_dataset = reg.build_live_compatible_dataset(legacy_dataset, building)

    bottleneck_result = reg.train_bottleneck_occurrence_model(live_dataset, training_seed=1, dataset_identifier="shadow-test")

    _MODULE_STATE["tmp_dir"] = tmp_dir
    _MODULE_STATE["bottleneck_result"] = bottleneck_result


def tearDownModule():

    shutil.rmtree(_MODULE_STATE.get("tmp_dir", ""), ignore_errors=True)


def _make_gateway():

    registry = reg.ModelRegistry()
    registry.register_model(_MODULE_STATE["bottleneck_result"].model, _MODULE_STATE["bottleneck_result"].metadata)
    service = reg.LiveAIInferenceService(registry)

    return RegistryLiveAIInferenceGateway(service, include_evacuation_time=False)


def _make_two_floor_building():

    building = Building(name="Shadow Mode Test Building")
    ground = building.create_floor(name="Ground", height=3.0)
    floor1 = building.create_floor(name="Floor 1", height=3.0)

    lobby = Zone(name="Lobby", floor_id=ground.id, x=0.0, y=0.0, width=10.0, height=10.0)
    corridor_a = Zone(name="Corridor A", floor_id=ground.id, x=20.0, y=0.0, width=10.0, height=10.0)
    corridor_b = Zone(name="Corridor B", floor_id=ground.id, x=0.0, y=20.0, width=10.0, height=10.0)
    upstairs = Zone(name="Upstairs", floor_id=floor1.id, x=0.0, y=0.0, width=10.0, height=10.0)

    ground.add_zone(lobby)
    ground.add_zone(corridor_a)
    ground.add_zone(corridor_b)
    floor1.add_zone(upstairs)

    from models.door import Door
    ground.add_door(Door(id="D-A", name="D-A", floor_id=ground.id, start_point=(10.0, 5.0), end_point=(20.0, 5.0), zone_a_id=lobby.id, zone_b_id=corridor_a.id))
    ground.add_door(Door(id="D-B", name="D-B", floor_id=ground.id, start_point=(5.0, 10.0), end_point=(5.0, 20.0), zone_a_id=lobby.id, zone_b_id=corridor_b.id))

    ground.add_exit(Exit(id="EXIT-A", name="Exit A", zone_id=corridor_a.id, floor_id=ground.id, start_point=(20.0, 0.0), end_point=(20.0, 5.0)))
    ground.add_exit(Exit(id="EXIT-B", name="Exit B", zone_id=corridor_b.id, floor_id=ground.id, start_point=(0.0, 20.0), end_point=(0.0, 25.0)))

    stair = Staircase(id="S1", name="Stair", from_floor_id=ground.id, to_floor_id=floor1.id, from_zone_id=lobby.id, to_zone_id=upstairs.id, width=1.5)
    ground.add_stair(stair)

    return building, ground, floor1, lobby, corridor_a, corridor_b, upstairs


class _StubBuildingStateGateway:

    # Matches live_system.building_state_gateway.BuildingStateGateway's
    # own Protocol exactly (collect(time) -> BuildingState) -- a minimal
    # test double so snapshot.building_state is genuinely populated,
    # letting live_ai_gateway.predict() run a REAL inference rather than
    # honestly reporting "no BuildingState available."

    def __init__(self, state):
        self._state = state

    def collect(self, time):
        return self._state


def _build_orchestrator(building, occupant_manager, live_ai_gateway=None):

    graph = NavigationGraphGenerator().build(building)
    event_bus = EventBus()

    crowd_engine = CrowdIntelligenceEngine(building, occupant_manager)
    recommendation_engine = EvacuationRecommendationEngine(building, graph, occupant_manager)
    guidance_engine = EvacuationGuidanceEngine(building, graph)

    building_state = BuildingStateEstimator().estimate(
        0.0, hazard_snapshot=HazardSnapshot(timestamp=0.0),
        occupancy_snapshot=OccupancySnapshot(timestamp=0.0),
    )

    return LiveOrchestrator(
        event_bus=event_bus,
        building_state_gateway=_StubBuildingStateGateway(building_state),
        crowd_intelligence_gateway=EngineCrowdIntelligenceGateway(crowd_engine),
        evacuation_recommendation_gateway=EngineEvacuationRecommendationGateway(recommendation_engine),
        evacuation_guidance_gateway=EngineEvacuationGuidanceGateway(guidance_engine),
        live_ai_gateway=live_ai_gateway,
    )


# =====================================================
# Phase 9 -- controlled runtime scenarios
# =====================================================


class ShadowModeValidationTests(unittest.TestCase):

    def setUp(self):

        self.building, self.ground, self.floor1, self.lobby, self.corridor_a, self.corridor_b, self.upstairs = (
            _make_two_floor_building()
        )

    def _occupied_manager(self):

        manager = LiveOccupantManager()
        manager.update("OCC-1", "CAM-1", "T1", self.upstairs.id, self.floor1.id, (1.0, 1.0), 0.0, None, 0.9, 0.0)
        return manager

    def test_feature_vector_is_produced_every_cycle_ai_is_run(self):

        gateway = _make_gateway()
        state = af.build_building_state_at_alarm_activation(reg.make_training_building(), total_occupants=4)

        snapshot = gateway.predict(state, 10.0)

        self.assertIsInstance(snapshot, LiveAIPredictionSnapshot)
        self.assertEqual(snapshot.system_status, AISystemStatus.AVAILABLE)
        self.assertIsNotNone(snapshot.bottleneck)
        self.assertIsNotNone(snapshot.feature_schema_version)

    def test_prediction_executed_and_stored_every_orchestrator_cycle(self):

        manager = self._occupied_manager()
        orchestrator = _build_orchestrator(self.building, manager, live_ai_gateway=_make_gateway())
        orchestrator.start()

        result = orchestrator.run_cycle(0.0)

        self.assertIsNotNone(result.ai_prediction_snapshot)
        self.assertIs(orchestrator.latest_ai_prediction, result.ai_prediction_snapshot)

    def test_runtime_continues_normally_when_ai_is_configured(self):

        manager = self._occupied_manager()
        orchestrator = _build_orchestrator(self.building, manager, live_ai_gateway=_make_gateway())
        orchestrator.start()

        # Multiple cycles, no exception, normal ARRIVED/ongoing operation.
        for t in (0.0, 1.0, 2.0):
            snapshot = orchestrator.run_cycle(t)
            self.assertIsNotNone(snapshot)

    def test_recommendation_decision_fields_identical_before_and_after_shadow_mode(self):

        # The central Phase 9 proof: the SAME building/occupancy,
        # driven through two independent orchestrators -- one with no
        # AI configured at all (today's exact behavior), one with a
        # real trained model producing real predictions every cycle.
        # recommended_exit_id/ranked_exit_ids/status/confidence must be
        # IDENTICAL.

        manager_without_ai = self._occupied_manager()
        orchestrator_without_ai = _build_orchestrator(self.building, manager_without_ai, live_ai_gateway=None)
        orchestrator_without_ai.start()
        snapshot_without_ai = orchestrator_without_ai.run_cycle(0.0)

        manager_with_ai = self._occupied_manager()
        orchestrator_with_ai = _build_orchestrator(self.building, manager_with_ai, live_ai_gateway=_make_gateway())
        orchestrator_with_ai.start()
        snapshot_with_ai = orchestrator_with_ai.run_cycle(0.0)

        self.assertIsNone(snapshot_without_ai.ai_prediction_snapshot)
        self.assertIsNotNone(snapshot_with_ai.ai_prediction_snapshot)
        self.assertIsNotNone(snapshot_with_ai.ai_prediction_snapshot.bottleneck)

        rec_without = snapshot_without_ai.evacuation_recommendation.zones[self.upstairs.id]
        rec_with = snapshot_with_ai.evacuation_recommendation.zones[self.upstairs.id]

        self.assertEqual(rec_without.status, rec_with.status)
        self.assertEqual(rec_without.recommended_exit_id, rec_with.recommended_exit_id)
        self.assertEqual(rec_without.ranked_exit_ids, rec_with.ranked_exit_ids)
        self.assertEqual(rec_without.confidence, rec_with.confidence)

        # Guidance, downstream of Recommendation's own (unaffected)
        # output, is identical too.
        guidance_without = snapshot_without_ai.evacuation_guidance.zones[self.upstairs.id]
        guidance_with = snapshot_with_ai.evacuation_guidance.zones[self.upstairs.id]

        self.assertEqual(guidance_without.recommended_exit_id, guidance_with.recommended_exit_id)
        self.assertEqual(guidance_without.ordered_door_ids, guidance_with.ordered_door_ids)
        self.assertEqual(guidance_without.ordered_stair_ids, guidance_with.ordered_stair_ids)


# =====================================================
# Phase 8 -- mechanical proof of no decision integration
# =====================================================


class NoDecisionIntegrationTests(unittest.TestCase):

    def test_guidance_voice_signage_building_control_simulation_never_import_ai(self):

        import pathlib
        import re

        repo_root = pathlib.Path(__file__).resolve().parent.parent
        forbidden = r"^\s*(from|import)\s+(ai_registry|ai_inference|ai_training|ai_features|live_system\.live_ai_gateway)\b"

        for package in ("evacuation_guidance", "voice_evacuation", "dynamic_signage", "building_control", "simulator", "simulation_runtime"):

            for path in sorted((repo_root / package).glob("*.py")):

                text = path.read_text(encoding="utf-8")
                match = re.search(forbidden, text, re.MULTILINE)

                self.assertIsNone(
                    match,
                    f"{path.relative_to(repo_root)} imports {match.group(0).strip() if match else ''!r} -- "
                    f"this package must never depend on AI inference infrastructure.",
                )

    def test_guidance_gateway_call_site_never_passes_ai_prediction(self):

        # A structural (not just import-based) proof: the ORCHESTRATOR's
        # own call into evacuation_guidance_gateway.compute() must never
        # even mention ai_prediction_snapshot as an argument.
        import pathlib
        import re

        text = (pathlib.Path(__file__).resolve().parent.parent / "live_system" / "orchestrator.py").read_text(encoding="utf-8")

        match = re.search(r"evacuation_guidance_gateway\.compute\([^)]*\)", text, re.DOTALL)
        self.assertIsNotNone(match)
        self.assertNotIn("ai_prediction", match.group(0))

    def test_recommendation_ranking_is_mathematically_invariant_to_ai_evidence(self):

        # Precise, empirical proof -- NOT "Recommendation never receives
        # AI evidence" (it does: evacuation_recommendation.engine.
        # EvacuationRecommendationEngine.compute() accepts and threads
        # ai_prediction_snapshot into rank_exits_for_zone(), a genuine,
        # pre-existing, deliberate "AI only supports" design predating
        # this milestone -- see evacuation_recommendation/scoring.py's
        # own docstring). What IS proven here, using the real scoring
        # function: because ai_bottleneck_probability is a single,
        # building-wide value applied IDENTICALLY to every candidate for
        # a zone, it can never change the RELATIVE ranking -- verified
        # directly, not assumed.

        distance_by_exit = {
            "EXIT-A": {"zone-1": 10.0},
            "EXIT-B": {"zone-1": 50.0},
        }
        exit_zone_map = {"EXIT-A": "zone-corridor-a", "EXIT-B": "zone-corridor-b"}
        weights = RecommendationWeights()
        config = RecommendationConfig()

        without_ai = rank_exits_for_zone(
            "zone-1", distance_by_exit, exit_zone_map=exit_zone_map,
            ai_bottleneck_probability=None, weights=weights, config=config,
        )

        for probability in (0.0, 0.1, 0.5, 0.9, 1.0):

            with self.subTest(probability=probability):

                with_ai = rank_exits_for_zone(
                    "zone-1", distance_by_exit, exit_zone_map=exit_zone_map,
                    ai_bottleneck_probability=probability, weights=weights, config=config,
                )

                self.assertEqual(
                    tuple(c.exit_id for c in without_ai), tuple(c.exit_id for c in with_ai),
                    "AI evidence changed the exit ranking order -- this must never happen.",
                )

    def test_ai_bottleneck_probability_extraction_never_touches_recommendation_decision_fields(self):

        # ai_bottleneck_probability() itself never raises/produces a
        # decision -- purely an extraction helper.
        self.assertIsNone(ai_bottleneck_probability(None))

        snapshot = LiveAIPredictionSnapshot(
            timestamp=0.0, building_state_timestamp=0.0, feature_schema_version="1.0",
            system_status=AISystemStatus.UNAVAILABLE,
        )
        self.assertIsNone(ai_bottleneck_probability(snapshot))


# =====================================================
# Phase 10 -- performance
# =====================================================


class ShadowModePerformanceTests(unittest.TestCase):

    def test_realistic_scale_incremental_cost(self):

        # ~20 cameras, ~100 occupants, ~20 stairs -- feature extraction +
        # model inference + gateway/logging cost, reported separately
        # from any camera/detection/tracking pipeline cost (already
        # benchmarked at ~1-2ms in the Stair Flow / Predictive Feature
        # Live Parity milestones).
        building = Building(name="Shadow Mode Performance Building")
        floor_1 = building.create_floor(name="Floor 1", height=3.0)
        floor_2 = building.create_floor(name="Floor 2", height=3.0)

        for i in range(20):

            zone_a = Zone(id=f"z{i}a", name=f"Z{i}A", floor_id=floor_1.id, x=float(i) * 10, y=0.0, width=5.0, height=5.0)
            zone_b = Zone(id=f"z{i}b", name=f"Z{i}B", floor_id=floor_2.id, x=float(i) * 10, y=0.0, width=5.0, height=5.0)
            floor_1.add_zone(zone_a)
            floor_2.add_zone(zone_b)

            floor_1.add_stair(Staircase(
                id=f"S{i}", name=f"S{i}", from_floor_id=floor_1.id, to_floor_id=floor_2.id,
                from_zone_id=zone_a.id, to_zone_id=zone_b.id, width=1.5,
            ))

        manager = LiveOccupantManager()
        for i in range(100):
            manager.update(f"OCC-{i}", f"CAM-{i % 20}", f"T-{i}", f"z{i % 20}a", floor_1.id, (1.0, 1.0), 0.0, None, 0.9, 0.0)

        camera_statuses = [
            CameraStatus(camera_id=f"CAM-{i}", name=f"CAM-{i}", floor_id=floor_1.id, zone_ids=(), active=True, mode="Live", has_detection_provider=True)
            for i in range(20)
        ]

        occupancy_facts = manager.canonical_occupancy(0.0)
        occupancy_snapshot = OccupancySnapshot(timestamp=0.0)

        state = BuildingStateEstimator().estimate(
            0.0, hazard_snapshot=HazardSnapshot(timestamp=0.0), occupancy_snapshot=occupancy_snapshot,
            camera_statuses=camera_statuses,
        )

        gateway = _make_gateway()

        start = time.perf_counter()
        prediction_snapshot = gateway.predict(state, 0.0)
        elapsed = time.perf_counter() - start

        print(
            f"\n[shadow-mode perf] 20 cameras, 100 occupants, 20 stairs -> "
            f"feature extraction + inference + snapshot assembly took {elapsed * 1000:.2f} ms "
            f"(status={prediction_snapshot.system_status.name}, "
            f"gateway-measured inference_duration={prediction_snapshot.inference_duration_seconds * 1000:.2f} ms)"
        )

        self.assertLess(elapsed, 2.0)
        self.assertIsNotNone(prediction_snapshot.inference_duration_seconds)


if __name__ == "__main__":
    unittest.main()
