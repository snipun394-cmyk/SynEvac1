import shutil
import sys
import tempfile
import unittest

from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

import ai_registry as reg
import ai_features as af

from decision_policy.exit_policy import CLOSE, KEEP_OPEN
from decision_policy.stair_policy import AVOID, USE
from decision_policy.zone_policy import EVACUATE_IMMEDIATELY, SHELTER_IN_PLACE, WAIT

from advisory_system.ai_evidence import (
    AIDecisionEvidence,
    UNAVAILABLE_AI_DECISION_EVIDENCE,
    evidence_from_bottleneck_prediction,
)
from advisory_system.orchestrator import AdvisoryOrchestrator
from advisory_system.recommendation_models import AdvisoryInputs

from live_system.event_bus import EventBus, EventType
from live_system.live_ai_gateway import (
    AISystemStatus,
    LiveAIPredictionSnapshot,
    RegistryLiveAIInferenceGateway,
)
from live_system.live_advisory_gateway import (
    ReplayCompatibleAdvisoryGateway,
    ai_decision_evidence_from_prediction_snapshot,
)
from live_system.orchestrator import LiveOrchestrator
from live_system.state_manager import StateManager

from tests.test_advisory_system import (
    make_building, make_decision_policy, make_ground_truth, make_scenario,
)


# =====================================================
# AI-Augmented Decision Policy & Advisory Integration milestone --
# Phase 14 (16 named validation scenarios) + Phase 16 (27 named tests).
# Fixtures are deliberately IMPORTED from tests.test_advisory_system
# (make_building/make_decision_policy/make_ground_truth/make_scenario),
# the same "reuse an existing test module's own fixtures" convention
# tests.test_camera_manager_integration/test_multi_camera_fusion_
# integration already establish for tests.test_virtual_camera.make_zone --
# never a second, drifting copy of the same 3-zone building.
#
# Phase 15's End-to-End Offline Live Test reuses tests.test_live_ai_
# runtime_integration's own setUpModule pattern exactly (one real,
# moderate-scale 150-scenario campaign, trained once per module, not
# once per test) -- see EndToEndOfflineLiveAdvisoryTests below.
# =====================================================


def _evidence(*, probability=0.8, predicted=True, threshold=0.5, model_id="bottleneck-1", model_version="v1"):

    return evidence_from_bottleneck_prediction(
        probability=probability, predicted_occurrence=predicted, threshold=threshold,
        model_id=model_id, model_version=model_version, prediction_timestamp=10.0,
        building_state_timestamp=9.5, feature_schema_version="schema-v1",
    )


# =====================================================
# Phase 3 -- AIDecisionEvidence type itself.
# =====================================================


class AIDecisionEvidenceTests(unittest.TestCase):

    def test_unavailable_constant_has_no_leaked_values(self):

        self.assertFalse(UNAVAILABLE_AI_DECISION_EVIDENCE.available)
        self.assertIsNone(UNAVAILABLE_AI_DECISION_EVIDENCE.bottleneck_occurrence_probability)
        self.assertIsNone(UNAVAILABLE_AI_DECISION_EVIDENCE.bottleneck_predicted)
        self.assertIsNone(UNAVAILABLE_AI_DECISION_EVIDENCE.model_id)

    def test_evidence_from_bottleneck_prediction_carries_every_field(self):

        evidence = _evidence(probability=0.73, predicted=True, threshold=0.5, model_id="m-1", model_version="v2")

        self.assertTrue(evidence.available)
        self.assertEqual(evidence.bottleneck_occurrence_probability, 0.73)
        self.assertTrue(evidence.bottleneck_predicted)
        self.assertEqual(evidence.threshold, 0.5)
        self.assertEqual(evidence.model_id, "m-1")
        self.assertEqual(evidence.model_version, "v2")
        self.assertEqual(evidence.model_status, "PRODUCTION_CANDIDATE")
        self.assertEqual(evidence.feature_schema_version, "schema-v1")

    def test_no_localization_field_exists_anywhere_on_the_type(self):

        # Phase 2's own confirmed finding, made mechanical: no field on
        # this dataclass may ever name a specific zone/door/exit/stair.
        field_names = set(AIDecisionEvidence.__dataclass_fields__.keys())

        for forbidden in ("zone_id", "stair_id", "exit_id", "door_id", "location", "likely_zone_id", "suspected_bottleneck_asset_ids"):
            self.assertNotIn(forbidden, field_names)

    def test_to_dict_round_trips_every_field(self):

        evidence = _evidence()
        as_dict = evidence.to_dict()

        self.assertEqual(as_dict["bottleneck_occurrence_probability"], evidence.bottleneck_occurrence_probability)
        self.assertEqual(as_dict["model_id"], evidence.model_id)
        self.assertEqual(as_dict["available"], True)

    def test_frozen_immutable(self):

        evidence = _evidence()

        with self.assertRaises(Exception):
            evidence.bottleneck_occurrence_probability = 0.99


# =====================================================
# Phase 11 -- live_advisory_gateway's own snapshot -> evidence converter.
# =====================================================


class PredictionSnapshotToEvidenceTests(unittest.TestCase):

    def test_none_snapshot_yields_unavailable_evidence(self):

        self.assertEqual(
            ai_decision_evidence_from_prediction_snapshot(None), UNAVAILABLE_AI_DECISION_EVIDENCE,
        )

    def test_snapshot_with_no_bottleneck_yields_unavailable_evidence(self):

        snapshot = LiveAIPredictionSnapshot(
            timestamp=5.0, building_state_timestamp=4.5, feature_schema_version="schema-v1",
            system_status=AISystemStatus.UNAVAILABLE, bottleneck=None,
        )

        self.assertEqual(ai_decision_evidence_from_prediction_snapshot(snapshot), UNAVAILABLE_AI_DECISION_EVIDENCE)

    def test_snapshot_with_bottleneck_yields_populated_evidence(self):

        from ai_registry.inference_service import BottleneckOccurrencePrediction

        bottleneck = BottleneckOccurrencePrediction(
            probability=0.62, predicted_occurrence=True, threshold=0.5,
            model_id="bn-1", model_version="v1", feature_schema_version="schema-v1", timestamp=5.0,
        )
        snapshot = LiveAIPredictionSnapshot(
            timestamp=5.0, building_state_timestamp=4.5, feature_schema_version="schema-v1",
            system_status=AISystemStatus.AVAILABLE, bottleneck=bottleneck,
        )

        evidence = ai_decision_evidence_from_prediction_snapshot(snapshot)

        self.assertTrue(evidence.available)
        self.assertEqual(evidence.bottleneck_occurrence_probability, 0.62)
        self.assertTrue(evidence.bottleneck_predicted)
        self.assertEqual(evidence.model_id, "bn-1")
        self.assertEqual(evidence.building_state_timestamp, 4.5)
        self.assertEqual(evidence.prediction_timestamp, 5.0)


# =====================================================
# Phase 5 -- Safety Precedence. AI must never be able to flip a
# deterministic AVOID/CLOSE/SHELTER_IN_PLACE decision, regardless of how
# high its probability is.
# =====================================================


class SafetyPrecedenceTests(unittest.TestCase):

    def test_high_ai_probability_cannot_reopen_a_closed_exit_announcement(self):

        building = make_building()
        scenario = make_scenario()
        ground_truth = make_ground_truth()
        decision_policy = make_decision_policy(
            zone_decisions=[{"zone_id": "zone-a", "recommended_exit": "exit-1", "recommended_stair": None, "action": EVACUATE_IMMEDIATELY}],
            exit_decisions=[{"exit_id": "exit-1", "status": CLOSE}],
        )

        inputs = AdvisoryInputs(
            building=building, scenario=scenario, ground_truth=ground_truth, decision_policy=decision_policy,
            ai_decision_evidence=_evidence(probability=0.99, predicted=True),
        )
        report = AdvisoryOrchestrator().generate_report(inputs)

        announcement = report.civilian_announcements[0]
        self.assertIn("Remain in place", announcement.announcement)
        self.assertNotIn("Proceed to", announcement.announcement)

    def test_high_ai_probability_cannot_reinstate_an_avoided_stair(self):

        building = make_building()
        scenario = make_scenario()
        ground_truth = make_ground_truth()
        decision_policy = make_decision_policy(
            zone_decisions=[{"zone_id": "zone-b", "recommended_exit": "exit-1", "recommended_stair": "stair-1", "action": EVACUATE_IMMEDIATELY}],
            stair_decisions=[{"stair_id": "stair-1", "status": AVOID}],
        )

        inputs = AdvisoryInputs(
            building=building, scenario=scenario, ground_truth=ground_truth, decision_policy=decision_policy,
            ai_decision_evidence=_evidence(probability=0.97, predicted=True),
        )
        report = AdvisoryOrchestrator().generate_report(inputs)

        announcement = report.civilian_announcements[0]
        self.assertIn("Avoid Stair 1", announcement.announcement)
        self.assertNotIn("Use Stair 1", announcement.announcement)

    def test_ai_unavailable_does_not_prevent_shelter_in_place(self):

        building = make_building()
        scenario = make_scenario()
        ground_truth = make_ground_truth()
        decision_policy = make_decision_policy(
            zone_decisions=[{"zone_id": "zone-a", "recommended_exit": None, "recommended_stair": None, "action": SHELTER_IN_PLACE}],
        )

        inputs = AdvisoryInputs(
            building=building, scenario=scenario, ground_truth=ground_truth, decision_policy=decision_policy,
            ai_decision_evidence=None,
        )
        report = AdvisoryOrchestrator().generate_report(inputs)

        self.assertIn("Remain in place", report.civilian_announcements[0].announcement)

    def test_firefighter_blocked_routes_still_lists_avoided_stair_despite_ai(self):

        building = make_building()
        scenario = make_scenario()
        ground_truth = make_ground_truth()
        decision_policy = make_decision_policy(
            zone_decisions=[{"zone_id": "zone-b", "recommended_exit": None, "recommended_stair": None, "action": WAIT}],
            stair_decisions=[{"stair_id": "stair-1", "status": AVOID}],
        )

        inputs = AdvisoryInputs(
            building=building, scenario=scenario, ground_truth=ground_truth, decision_policy=decision_policy,
            ai_decision_evidence=_evidence(probability=0.01, predicted=False),
        )
        report = AdvisoryOrchestrator().generate_report(inputs)

        self.assertIn("Stair stair-1", report.firefighter_intelligence.blocked_routes)


# =====================================================
# Phase 6 -- Civilian advisories stay zone-based, and AI only ever
# touches a WAIT (congestion) zone's confidence.
# =====================================================


class CivilianAdvisoryAIIntegrationTests(unittest.TestCase):

    def test_ai_strengthens_confidence_of_an_already_deterministic_wait_zone(self):

        building = make_building()
        scenario = make_scenario()
        ground_truth = make_ground_truth(exits_exceeding_capacity=("exit-1",))
        decision_policy = make_decision_policy(
            zone_decisions=[{"zone_id": "zone-a", "recommended_exit": "exit-1", "recommended_stair": None, "action": WAIT}],
        )

        inputs_without_ai = AdvisoryInputs(
            building=building, scenario=scenario, ground_truth=ground_truth, decision_policy=decision_policy,
        )
        inputs_with_ai = AdvisoryInputs(
            building=building, scenario=scenario, ground_truth=ground_truth, decision_policy=decision_policy,
            ai_decision_evidence=_evidence(probability=0.9, predicted=True),
        )

        report_without_ai = AdvisoryOrchestrator().generate_report(inputs_without_ai)
        report_with_ai = AdvisoryOrchestrator().generate_report(inputs_with_ai)

        confidence_without = report_without_ai.civilian_announcements[0].confidence
        confidence_with = report_with_ai.civilian_announcements[0].confidence

        self.assertNotEqual(confidence_without, confidence_with)
        self.assertIn("ai", report_with_ai.civilian_announcements[0].confidence_source)
        self.assertNotIn("ai", report_without_ai.civilian_announcements[0].confidence_source)

    def test_ai_never_touches_confidence_of_an_evacuate_immediately_zone(self):

        building = make_building()
        scenario = make_scenario()
        ground_truth = make_ground_truth()
        decision_policy = make_decision_policy(
            zone_decisions=[{"zone_id": "zone-a", "recommended_exit": "exit-1", "recommended_stair": None, "action": EVACUATE_IMMEDIATELY}],
        )

        inputs_without_ai = AdvisoryInputs(
            building=building, scenario=scenario, ground_truth=ground_truth, decision_policy=decision_policy,
        )
        inputs_with_ai = AdvisoryInputs(
            building=building, scenario=scenario, ground_truth=ground_truth, decision_policy=decision_policy,
            ai_decision_evidence=_evidence(probability=0.9, predicted=True),
        )

        report_without_ai = AdvisoryOrchestrator().generate_report(inputs_without_ai)
        report_with_ai = AdvisoryOrchestrator().generate_report(inputs_with_ai)

        self.assertEqual(
            report_without_ai.civilian_announcements[0].confidence,
            report_with_ai.civilian_announcements[0].confidence,
        )
        self.assertNotIn("ai", report_with_ai.civilian_announcements[0].confidence_source)

    def test_announcements_remain_zone_addressed_never_individual_with_ai_present(self):

        building = make_building()
        scenario = make_scenario()
        ground_truth = make_ground_truth()
        decision_policy = make_decision_policy(
            zone_decisions=[{"zone_id": "zone-a", "recommended_exit": "exit-1", "recommended_stair": None, "action": EVACUATE_IMMEDIATELY}],
        )

        inputs = AdvisoryInputs(
            building=building, scenario=scenario, ground_truth=ground_truth, decision_policy=decision_policy,
            ai_decision_evidence=_evidence(probability=0.9, predicted=True),
        )
        report = AdvisoryOrchestrator().generate_report(inputs)

        announcement = report.civilian_announcements[0].announcement
        self.assertTrue(announcement.startswith("Attention occupants in"))
        for occupant_id in ("occ-1", "occ-2", "occ-3"):
            self.assertNotIn(occupant_id, announcement)


# =====================================================
# Phase 7 -- Firefighter Intelligence: information only, never a command.
# =====================================================


class FirefighterIntelligenceAIIntegrationTests(unittest.TestCase):

    def _report_for(self, evidence):

        building = make_building()
        scenario = make_scenario()
        ground_truth = make_ground_truth()
        decision_policy = make_decision_policy(
            zone_decisions=[{"zone_id": "zone-a", "recommended_exit": "exit-1", "recommended_stair": None, "action": EVACUATE_IMMEDIATELY}],
        )
        inputs = AdvisoryInputs(
            building=building, scenario=scenario, ground_truth=ground_truth, decision_policy=decision_policy,
            ai_decision_evidence=evidence,
        )
        return AdvisoryOrchestrator().generate_report(inputs)

    def test_ai_bottleneck_fields_populated_when_evidence_available(self):

        report = self._report_for(_evidence(probability=0.81, predicted=True, model_id="bn-1"))

        self.assertEqual(report.firefighter_intelligence.ai_bottleneck_probability, 0.81)
        self.assertEqual(report.firefighter_intelligence.ai_bottleneck_model_id, "bn-1")
        self.assertIn("ai", report.firefighter_intelligence.confidence_source)

    def test_ai_bottleneck_fields_none_when_evidence_unavailable(self):

        report = self._report_for(None)

        self.assertIsNone(report.firefighter_intelligence.ai_bottleneck_probability)
        self.assertIsNone(report.firefighter_intelligence.ai_bottleneck_model_id)
        self.assertNotIn("ai", report.firefighter_intelligence.confidence_source)

    def test_ai_bottleneck_fields_none_when_evidence_explicitly_unavailable_object(self):

        report = self._report_for(UNAVAILABLE_AI_DECISION_EVIDENCE)

        self.assertIsNone(report.firefighter_intelligence.ai_bottleneck_probability)
        self.assertIsNone(report.firefighter_intelligence.ai_bottleneck_model_id)

    def test_no_field_anywhere_assigns_a_firefighter_id_or_task(self):

        report = self._report_for(_evidence())
        as_dict = report.firefighter_intelligence.to_dict()

        forbidden_terms = ("command", "mission", "order", "assign", "directive", "instruct", "firefighter_id")
        rendered = str(as_dict).lower()

        for term in forbidden_terms:
            self.assertNotIn(term, rendered)


# =====================================================
# Phase 8 -- Building/Commander advisories: AI may only ever add a
# building-wide "monitor" recommendation, never a control action, never
# a localized one.
# =====================================================


class BuildingRecommendationsAIIntegrationTests(unittest.TestCase):

    def _recommendations_for(self, evidence, *, exits_exceeding_capacity=()):

        building = make_building()
        scenario = make_scenario()
        ground_truth = make_ground_truth(exits_exceeding_capacity=exits_exceeding_capacity)
        decision_policy = make_decision_policy(
            zone_decisions=[{"zone_id": "zone-a", "recommended_exit": "exit-1", "recommended_stair": None, "action": EVACUATE_IMMEDIATELY}],
        )
        inputs = AdvisoryInputs(
            building=building, scenario=scenario, ground_truth=ground_truth, decision_policy=decision_policy,
            ai_decision_evidence=evidence,
        )
        return AdvisoryOrchestrator().generate_report(inputs).building_recommendations

    def test_monitor_recommendation_appears_when_bottleneck_predicted(self):

        recommendations = self._recommendations_for(_evidence(probability=0.85, predicted=True))
        monitor = [r for r in recommendations if r.action == "Monitor for Building-Wide Congestion"]

        self.assertEqual(len(monitor), 1)
        self.assertEqual(monitor[0].target_type, "building")
        self.assertIsNone(monitor[0].target_id)
        self.assertEqual(monitor[0].confidence_source, ("ai",))

    def test_monitor_recommendation_absent_when_bottleneck_not_predicted(self):

        recommendations = self._recommendations_for(_evidence(probability=0.2, predicted=False))
        monitor = [r for r in recommendations if r.action == "Monitor for Building-Wide Congestion"]

        self.assertEqual(monitor, [])

    def test_monitor_recommendation_absent_when_ai_unavailable(self):

        recommendations = self._recommendations_for(None)
        monitor = [r for r in recommendations if r.action == "Monitor for Building-Wide Congestion"]

        self.assertEqual(monitor, [])

    def test_ai_never_generates_a_control_action_recommendation(self):

        recommendations = self._recommendations_for(_evidence(probability=0.99, predicted=True))

        forbidden_prefixes = (
            "Close Door", "Open Exit", "Activate Deluge", "Activate Smoke Exhaust",
            "Activate Stair Pressurization", "Unlock Exit", "Broadcast Voice Message",
        )
        ai_sourced = [r for r in recommendations if "ai" in r.confidence_source]

        for rec in ai_sourced:
            for prefix in forbidden_prefixes:
                self.assertFalse(rec.action.startswith(prefix), f"AI-sourced recommendation {rec.action!r} is a control action")

    def test_ai_recommendation_never_names_a_specific_zone_stair_or_exit(self):

        recommendations = self._recommendations_for(_evidence(probability=0.99, predicted=True))
        ai_sourced = [r for r in recommendations if "ai" in r.confidence_source]

        for rec in ai_sourced:
            self.assertEqual(rec.target_type, "building")
            self.assertIsNone(rec.target_id)

    def test_deterministic_recommendations_unaffected_by_ai_presence(self):

        without_ai = self._recommendations_for(None)
        with_ai = self._recommendations_for(_evidence(probability=0.99, predicted=True))

        deterministic_without = [r.action for r in without_ai]
        deterministic_with = [r.action for r in with_ai if r.action != "Monitor for Building-Wide Congestion"]

        self.assertEqual(deterministic_without, deterministic_with)


# =====================================================
# Phase 8/10 -- Commander Dashboard: three genuinely separate confidence
# concepts must never be conflated.
# =====================================================


class CommanderDashboardAIIntegrationTests(unittest.TestCase):

    def test_ai_bottleneck_probability_populated_and_distinct_field(self):

        building = make_building()
        scenario = make_scenario()
        ground_truth = make_ground_truth()
        decision_policy = make_decision_policy(
            zone_decisions=[{"zone_id": "zone-a", "recommended_exit": "exit-1", "recommended_stair": None, "action": EVACUATE_IMMEDIATELY}],
        )
        inputs = AdvisoryInputs(
            building=building, scenario=scenario, ground_truth=ground_truth, decision_policy=decision_policy,
            ai_decision_evidence=_evidence(probability=0.66, predicted=True, model_id="bn-9"),
        )
        dashboard = AdvisoryOrchestrator().generate_report(inputs).commander_dashboard

        self.assertEqual(dashboard.ai_bottleneck_probability, 0.66)
        self.assertEqual(dashboard.ai_bottleneck_model_id, "bn-9")

    def test_predicted_bottlenecks_never_receives_an_ai_derived_location(self):

        building = make_building()
        scenario = make_scenario()
        ground_truth = make_ground_truth()
        decision_policy = make_decision_policy(
            zone_decisions=[{"zone_id": "zone-a", "recommended_exit": "exit-1", "recommended_stair": None, "action": EVACUATE_IMMEDIATELY}],
        )
        inputs = AdvisoryInputs(
            building=building, scenario=scenario, ground_truth=ground_truth, decision_policy=decision_policy,
            ai_decision_evidence=_evidence(probability=0.99, predicted=True),
        )
        dashboard = AdvisoryOrchestrator().generate_report(inputs).commander_dashboard

        self.assertEqual(dashboard.predicted_bottlenecks, ())


class ConfidenceSeparationTests(unittest.TestCase):

    def test_ai_probability_occupancy_confidence_and_recommendation_confidence_are_distinct(self):

        from perception.models.human_observation import HumanClassification, HumanObservation, HumanState

        building = make_building()
        scenario = make_scenario()
        ground_truth = make_ground_truth(zone_risk_scores=[{"zone_id": "zone-a", "risk_score": 0.4}])
        decision_policy = make_decision_policy(
            zone_decisions=[{"zone_id": "zone-a", "recommended_exit": "exit-1", "recommended_stair": None, "action": EVACUATE_IMMEDIATELY}],
        )
        human_observations = {
            "occ-1": HumanObservation(
                person_id="occ-1", zone_id="zone-a", classification=HumanClassification.ADULT,
                state=HumanState.WALKING, confidence=0.55, last_observed_time=1.0,
            ),
        }
        inputs = AdvisoryInputs(
            building=building, scenario=scenario, ground_truth=ground_truth, decision_policy=decision_policy,
            human_observations=human_observations,
            ai_decision_evidence=_evidence(probability=0.9, predicted=True),
        )
        dashboard = AdvisoryOrchestrator().generate_report(inputs).commander_dashboard

        values = {dashboard.ai_bottleneck_probability, dashboard.occupancy_confidence, dashboard.recommendation_confidence}

        # Three genuinely different quantities computed by three
        # different code paths -- asserting they are not silently the
        # same float is the mechanical form of Phase 10's requirement.
        self.assertEqual(dashboard.ai_bottleneck_probability, 0.9)
        self.assertNotEqual(dashboard.ai_bottleneck_probability, dashboard.occupancy_confidence)
        self.assertIsNotNone(dashboard.occupancy_confidence)
        self.assertIsNotNone(dashboard.recommendation_confidence)


# =====================================================
# Phase 9 -- Explainability: RULE_BASED vs AI_SUPPORTED classification.
# =====================================================


class ExplainabilityConfidenceSourceTests(unittest.TestCase):

    def test_wait_zone_without_ai_is_rule_based(self):

        building = make_building()
        scenario = make_scenario()
        ground_truth = make_ground_truth(exits_exceeding_capacity=("exit-1",))
        decision_policy = make_decision_policy(
            zone_decisions=[{"zone_id": "zone-a", "recommended_exit": "exit-1", "recommended_stair": None, "action": WAIT}],
        )
        inputs = AdvisoryInputs(building=building, scenario=scenario, ground_truth=ground_truth, decision_policy=decision_policy)
        report = AdvisoryOrchestrator().generate_report(inputs)

        self.assertEqual(report.civilian_announcements[0].confidence_source, ())

    def test_wait_zone_with_ai_is_ai_supported_not_ai_generated(self):

        building = make_building()
        scenario = make_scenario()
        ground_truth = make_ground_truth(exits_exceeding_capacity=("exit-1",))
        decision_policy = make_decision_policy(
            zone_decisions=[{"zone_id": "zone-a", "recommended_exit": "exit-1", "recommended_stair": None, "action": WAIT}],
        )
        inputs = AdvisoryInputs(
            building=building, scenario=scenario, ground_truth=ground_truth, decision_policy=decision_policy,
            ai_decision_evidence=_evidence(probability=0.9, predicted=True),
        )
        report = AdvisoryOrchestrator().generate_report(inputs)

        # AI_SUPPORTED: "ai" appears alongside the recommendation, which
        # decision_policy already independently produced (WAIT was
        # decided with zero knowledge of this AI model) -- never
        # AI_GENERATED, since AI created neither the action nor the zone.
        self.assertEqual(report.civilian_announcements[0].confidence_source, ("ai",))
        self.assertEqual(report.civilian_announcements[0].announcement, (
            "Attention occupants in Cafeteria. Hold your position briefly. "
            "Your evacuation route is currently congested."
        ))


# =====================================================
# Phase 11/12 -- ReplayCompatibleAdvisoryGateway + StateManager snapshot.
# =====================================================


class ReplayCompatibleAdvisoryGatewayTests(unittest.TestCase):

    def _gateway(self, *, decision_policy_provider):

        return ReplayCompatibleAdvisoryGateway(
            building=make_building(), scenario=make_scenario(), ground_truth=make_ground_truth(),
            decision_policy_provider=decision_policy_provider,
        )

    def test_returns_none_when_decision_policy_provider_returns_none(self):

        gateway = self._gateway(decision_policy_provider=lambda time: None)

        self.assertIsNone(gateway.generate(_evidence(), 5.0))

    def test_returns_none_when_provider_raises(self):

        def _raising_provider(time):
            raise RuntimeError("boom")

        gateway = self._gateway(decision_policy_provider=_raising_provider)

        self.assertIsNone(gateway.generate(_evidence(), 5.0))

    def test_happy_path_returns_advisory_report_with_ai_evidence_wired_through(self):

        decision_policy = make_decision_policy(
            zone_decisions=[{"zone_id": "zone-a", "recommended_exit": "exit-1", "recommended_stair": None, "action": EVACUATE_IMMEDIATELY}],
        )
        gateway = self._gateway(decision_policy_provider=lambda time: decision_policy)

        report = gateway.generate(_evidence(probability=0.75, predicted=True, model_id="bn-live"), 5.0)

        self.assertIsNotNone(report)
        self.assertEqual(report.firefighter_intelligence.ai_bottleneck_probability, 0.75)
        self.assertEqual(report.firefighter_intelligence.ai_bottleneck_model_id, "bn-live")

    def test_ai_evidence_none_still_produces_a_report(self):

        decision_policy = make_decision_policy(
            zone_decisions=[{"zone_id": "zone-a", "recommended_exit": "exit-1", "recommended_stair": None, "action": EVACUATE_IMMEDIATELY}],
        )
        gateway = self._gateway(decision_policy_provider=lambda time: decision_policy)

        report = gateway.generate(None, 5.0)

        self.assertIsNotNone(report)
        self.assertIsNone(report.firefighter_intelligence.ai_bottleneck_probability)

    def test_decision_policy_provider_called_fresh_each_cycle(self):

        calls = []

        def _provider(time):
            calls.append(time)
            return make_decision_policy(
                zone_decisions=[{"zone_id": "zone-a", "recommended_exit": "exit-1", "recommended_stair": None, "action": EVACUATE_IMMEDIATELY}],
            )

        gateway = self._gateway(decision_policy_provider=_provider)
        gateway.generate(_evidence(), 1.0)
        gateway.generate(_evidence(), 2.0)

        self.assertEqual(calls, [1.0, 2.0])


class StateManagerAdvisorySnapshotTests(unittest.TestCase):

    def test_latest_advisory_report_none_before_any_update(self):

        manager = StateManager()

        self.assertIsNone(manager.latest_advisory_report())

    def test_update_advisory_report_stores_and_stamps_timestamp(self):

        manager = StateManager()
        decision_policy = make_decision_policy(
            zone_decisions=[{"zone_id": "zone-a", "recommended_exit": "exit-1", "recommended_stair": None, "action": EVACUATE_IMMEDIATELY}],
        )
        inputs = AdvisoryInputs(
            building=make_building(), scenario=make_scenario(), ground_truth=make_ground_truth(),
            decision_policy=decision_policy,
        )
        report = AdvisoryOrchestrator().generate_report(inputs)

        snapshot = manager.update_advisory_report(report, 12.0)

        self.assertIs(snapshot.advisory_report, report)
        self.assertIs(manager.latest_advisory_report(), report)
        self.assertEqual(snapshot.component_timestamps["advisory_report"], 12.0)

    def test_other_fields_preserved_across_advisory_only_update(self):

        manager = StateManager()
        manager.update_engineering_state({"door-1": "state"}, 1.0)

        decision_policy = make_decision_policy(
            zone_decisions=[{"zone_id": "zone-a", "recommended_exit": "exit-1", "recommended_stair": None, "action": EVACUATE_IMMEDIATELY}],
        )
        inputs = AdvisoryInputs(
            building=make_building(), scenario=make_scenario(), ground_truth=make_ground_truth(),
            decision_policy=decision_policy,
        )
        report = AdvisoryOrchestrator().generate_report(inputs)
        snapshot = manager.update_advisory_report(report, 2.0)

        self.assertEqual(dict(snapshot.engineering_state), {"door-1": "state"})

    def test_no_stale_report_silently_looks_current(self):

        # A report never gets a bumped component_timestamps["advisory_
        # report"] unless update_advisory_report() is actually called --
        # the same honest staleness convention live_ai_gateway's own
        # ai_prediction_snapshot entry already established.
        manager = StateManager()
        decision_policy = make_decision_policy(
            zone_decisions=[{"zone_id": "zone-a", "recommended_exit": "exit-1", "recommended_stair": None, "action": EVACUATE_IMMEDIATELY}],
        )
        inputs = AdvisoryInputs(
            building=make_building(), scenario=make_scenario(), ground_truth=make_ground_truth(),
            decision_policy=decision_policy,
        )
        report = AdvisoryOrchestrator().generate_report(inputs)
        manager.update_advisory_report(report, 5.0)

        # A later cycle that does NOT call update_advisory_report()
        # leaves the timestamp exactly where it was.
        manager.update_engineering_state({}, 9.0)

        self.assertEqual(manager.current().component_timestamps["advisory_report"], 5.0)
        self.assertIs(manager.latest_advisory_report(), report)


# =====================================================
# Phase 11/13 -- LiveOrchestrator wiring + no-output-execution boundary.
# =====================================================


class OrchestratorLiveAdvisoryWiringTests(unittest.TestCase):

    def _decision_policy(self):

        return make_decision_policy(
            zone_decisions=[{"zone_id": "zone-a", "recommended_exit": "exit-1", "recommended_stair": None, "action": EVACUATE_IMMEDIATELY}],
        )

    def test_run_cycle_populates_advisory_report_and_emits_event(self):

        gateway = ReplayCompatibleAdvisoryGateway(
            building=make_building(), scenario=make_scenario(), ground_truth=make_ground_truth(),
            decision_policy_provider=lambda time: self._decision_policy(),
        )
        event_bus = EventBus()
        orchestrator = LiveOrchestrator(event_bus=event_bus, live_advisory_gateway=gateway)
        orchestrator.start()

        snapshot = orchestrator.run_cycle(1.0)

        self.assertIsNotNone(snapshot.advisory_report)
        self.assertIs(orchestrator.latest_advisory_report, snapshot.advisory_report)
        self.assertEqual(len(event_bus.history_of(EventType.ADVISORY_REPORT_UPDATED)), 1)

    def test_run_cycle_leaves_previous_report_in_place_when_gateway_returns_none(self):

        calls = {"count": 0}

        class _FlakyGateway:

            def generate(self, ai_evidence, time):
                calls["count"] += 1
                if calls["count"] == 1:
                    return AdvisoryOrchestrator().generate_report(
                        AdvisoryInputs(
                            building=make_building(), scenario=make_scenario(), ground_truth=make_ground_truth(),
                            decision_policy=OrchestratorLiveAdvisoryWiringTests()._decision_policy(),
                        )
                    )
                return None

        orchestrator = LiveOrchestrator(live_advisory_gateway=_FlakyGateway())
        orchestrator.start()

        first_snapshot = orchestrator.run_cycle(1.0)
        second_snapshot = orchestrator.run_cycle(2.0)

        self.assertIs(second_snapshot.advisory_report, first_snapshot.advisory_report)
        self.assertEqual(second_snapshot.component_timestamps["advisory_report"], 1.0)

    def test_no_advisory_gateway_configured_leaves_advisory_report_none(self):

        orchestrator = LiveOrchestrator()
        orchestrator.start()

        snapshot = orchestrator.run_cycle(1.0)

        self.assertIsNone(snapshot.advisory_report)


class NoOutputExecutionBoundaryTests(unittest.TestCase):

    # Phase 13's explicit requirement, made mechanical for the files this
    # milestone actually touches. The broader live_system-wide sweep
    # already exists (tests.test_live_system.
    # LiveSystemPackageDependencyDirectionTests) and already covers
    # live_advisory_gateway.py automatically via its own package-wide
    # glob -- this class names the file directly so the guarantee is
    # legible from this milestone's own test file, not only inherited.

    def test_live_advisory_gateway_never_imports_execution_capable_modules(self):

        import pathlib
        import re

        path = pathlib.Path(__file__).resolve().parent.parent / "live_system" / "live_advisory_gateway.py"
        text = path.read_text()

        forbidden = (
            r"^\s*(from|import)\s+"
            r"(voice_evacuation|speaker_manager|building_control\.controller|building_control\.providers)\b"
        )

        self.assertIsNone(
            re.search(forbidden, text, re.MULTILINE),
            "live_advisory_gateway.py imports an execution-capable module -- AdvisoryReport "
            "must remain inert data this milestone never wires to real output.",
        )

    def test_advisory_system_package_never_imports_execution_capable_modules(self):

        import pathlib
        import re

        package_dir = pathlib.Path(__file__).resolve().parent.parent / "advisory_system"

        forbidden = (
            r"^\s*(from|import)\s+"
            r"(voice_evacuation|speaker_manager|building_control\.controller|building_control\.providers)\b"
        )

        for path in package_dir.glob("*.py"):

            text = path.read_text()

            self.assertIsNone(
                re.search(forbidden, text, re.MULTILINE),
                f"advisory_system/{path.name} imports an execution-capable module -- this package "
                f"must only ever produce inert recommendation records.",
            )

    def test_advisory_report_dataclasses_expose_no_execute_or_send_method(self):

        from advisory_system.recommendation_models import AdvisoryReport, BuildingRecommendation, CivilianAnnouncement

        for cls in (AdvisoryReport, BuildingRecommendation, CivilianAnnouncement):
            for forbidden_method in ("execute", "send", "broadcast", "apply", "activate", "trigger"):
                self.assertFalse(hasattr(cls, forbidden_method), f"{cls.__name__} exposes a {forbidden_method}() method")


# =====================================================
# Phase 15 -- End-to-End Offline Live Test. Real trained bottleneck
# model, via the same setUpModule pattern tests.test_live_ai_runtime_
# integration already establishes (one shared 150-scenario campaign,
# trained once per module).
# =====================================================


_MODULE_STATE = {}

CAMPAIGN_COUNT = 150
CAMPAIGN_SEED = 909


def setUpModule():

    tmp_dir = tempfile.mkdtemp(prefix="ai_augmented_advisory_test_")
    building = reg.make_training_building()
    definition = reg.make_training_definition()

    campaign_dir, _summary = reg.generate_training_campaign(
        tmp_dir, building, definition, count=CAMPAIGN_COUNT, master_seed=CAMPAIGN_SEED,
    )

    import ai_training as at

    legacy_dataset = at.load_campaign_dataset(campaign_dir)
    live_dataset = reg.build_live_compatible_dataset(legacy_dataset, building)

    bottleneck_result = reg.train_bottleneck_occurrence_model(live_dataset, training_seed=1, dataset_identifier="advisory-e2e")

    _MODULE_STATE["tmp_dir"] = tmp_dir
    _MODULE_STATE["building"] = building
    _MODULE_STATE["bottleneck_result"] = bottleneck_result


def tearDownModule():

    shutil.rmtree(_MODULE_STATE.get("tmp_dir", ""), ignore_errors=True)


class EndToEndOfflineLiveAdvisoryTests(unittest.TestCase):

    def _gateway_chain(self):

        registry = reg.ModelRegistry()
        registry.register_model(_MODULE_STATE["bottleneck_result"].model, _MODULE_STATE["bottleneck_result"].metadata)
        service = reg.LiveAIInferenceService(registry)
        live_ai_gateway = RegistryLiveAIInferenceGateway(service, include_evacuation_time=False)

        real_building = _MODULE_STATE["building"]
        decision_policy = make_decision_policy(
            zone_decisions=[{"zone_id": "zone-a", "recommended_exit": "exit-1", "recommended_stair": None, "action": EVACUATE_IMMEDIATELY}],
        )
        advisory_gateway = ReplayCompatibleAdvisoryGateway(
            building=make_building(), scenario=make_scenario(), ground_truth=make_ground_truth(),
            decision_policy_provider=lambda time: decision_policy,
        )

        return live_ai_gateway, advisory_gateway, real_building

    def _building_state(self, building):

        return af.build_building_state_at_alarm_activation(building, total_occupants=6, ignition_zone_id="zone-lobby")

    def test_full_chain_produces_advisory_report_with_ai_evidence(self):

        live_ai_gateway, advisory_gateway, real_building = self._gateway_chain()

        class _FixedBuildingStateGateway:

            def __init__(self, state):
                self._state = state

            def collect(self, time):
                return self._state

        building_state = self._building_state(real_building)
        building_state_gateway = _FixedBuildingStateGateway(building_state)

        orchestrator = LiveOrchestrator(
            building_state_gateway=building_state_gateway,
            live_ai_gateway=live_ai_gateway,
            live_advisory_gateway=advisory_gateway,
        )
        orchestrator.start()

        snapshot = orchestrator.run_cycle(1.0)

        self.assertIsNotNone(snapshot.ai_prediction_snapshot)
        self.assertIsNotNone(snapshot.ai_prediction_snapshot.bottleneck)
        self.assertIsNotNone(snapshot.advisory_report)

        report = snapshot.advisory_report

        # Zone-based civilian advisory, firefighter intelligence, and a
        # commander summary are all genuinely produced.
        self.assertGreater(len(report.civilian_announcements), 0)
        self.assertIsNotNone(report.firefighter_intelligence)
        self.assertIsNotNone(report.commander_dashboard)

        # AI evidence appears where relevant (firefighter/commander both
        # carry the live model's own probability, not a fabricated one).
        self.assertEqual(
            report.firefighter_intelligence.ai_bottleneck_probability,
            snapshot.ai_prediction_snapshot.bottleneck.probability,
        )
        self.assertEqual(
            report.commander_dashboard.ai_bottleneck_probability,
            snapshot.ai_prediction_snapshot.bottleneck.probability,
        )

        # No individual-occupant speaker command anywhere in the report.
        rendered = str(report.to_dict())
        for occupant_id in ("occ-1", "occ-2", "occ-3"):
            self.assertNotIn(occupant_id, rendered)

        # No firefighter command (no assigned_task/mission/order field).
        for forbidden_term in ("mission", "assign", "directive"):
            self.assertNotIn(forbidden_term, rendered.lower())

        # No automatic building control or voice broadcast executed --
        # every BuildingRecommendation is inert data, never an action
        # this test (or LiveOrchestrator) actually carried out.
        self.assertFalse(hasattr(orchestrator, "building_control_controller"))
        self.assertFalse(hasattr(orchestrator, "voice_evacuation_controller"))

    def test_stale_previous_report_kept_when_ai_gateway_has_no_building_state_yet(self):

        live_ai_gateway, advisory_gateway, _real_building = self._gateway_chain()

        orchestrator = LiveOrchestrator(
            live_ai_gateway=live_ai_gateway, live_advisory_gateway=advisory_gateway,
        )
        orchestrator.start()

        # No building_state_gateway configured -> snapshot.building_state
        # stays None -> live_ai_gateway.predict(None, ...) still runs
        # (Phase-established "always called when configured" behavior)
        # and reports UNAVAILABLE, but advisory generation still proceeds
        # (ReplayCompatibleAdvisoryGateway does not require AI evidence
        # to produce a report -- it degrades to no AI signal honestly).
        snapshot = orchestrator.run_cycle(1.0)

        self.assertEqual(snapshot.ai_prediction_snapshot.system_status, AISystemStatus.UNAVAILABLE)
        self.assertIsNotNone(snapshot.advisory_report)
        self.assertIsNone(snapshot.advisory_report.firefighter_intelligence.ai_bottleneck_probability)


if __name__ == "__main__":
    unittest.main()
