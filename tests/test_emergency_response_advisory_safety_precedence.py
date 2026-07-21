import unittest

from advisory_system.emergency_response_evidence import EmergencyResponseEvidence, ZoneResponseDetail
from advisory_system.orchestrator import AdvisoryOrchestrator
from advisory_system.recommendation_models import AdvisoryInputs

from decision_policy.exit_policy import KEEP_OPEN
from decision_policy.zone_policy import EVACUATE_IMMEDIATELY, SHELTER_IN_PLACE, WAIT

from tests.test_advisory_system import make_building, make_decision_policy, make_ground_truth, make_scenario


# =====================================================
# Live Emergency Response & Rescue Priority Intelligence milestone,
# Phase 22 items 25-30 -- emergency-response-specific safety
# precedence. Mirrors tests/test_crowd_advisory_safety_precedence.py/
# test_evacuation_advisory_safety_precedence.py's own structure exactly:
# response priority is SECONDARY, SUPPORTING evidence that can never
# override a deterministic safety restriction, and it never dispatches,
# broadcasts, or executes anything.
# =====================================================


def response_evidence(*, critical_zone_ids=(), high_priority_zone_ids=(), zone_details=None) -> EmergencyResponseEvidence:

    return EmergencyResponseEvidence(
        available=True, timestamp=0.0,
        critical_zone_ids=tuple(critical_zone_ids), high_priority_zone_ids=tuple(high_priority_zone_ids),
        zone_details=zone_details or {},
    )


def generate(*, decision_policy, response=None):

    inputs = AdvisoryInputs(
        building=make_building(), scenario=make_scenario(), ground_truth=make_ground_truth(),
        decision_policy=decision_policy, emergency_response_evidence=response,
    )

    return AdvisoryOrchestrator().generate_report(inputs)


class Test25UnsafeExitRemainsUnsafe(unittest.TestCase):

    def test_unsafe_exit_remains_unsafe_regardless_of_response_priority(self):

        from decision_policy.exit_policy import CLOSE

        policy = make_decision_policy(
            zone_decisions=[{"zone_id": "zone-a", "action": EVACUATE_IMMEDIATELY, "recommended_exit": "exit-1"}],
            exit_decisions=[{"exit_id": "exit-1", "status": CLOSE}, {"exit_id": "exit-2", "status": KEEP_OPEN}],
        )

        report = generate(
            decision_policy=policy,
            response=response_evidence(critical_zone_ids=("zone-a",), zone_details={
                "zone-a": ZoneResponseDetail(priority_level="CRITICAL", reason_codes=("KNOWN_OCCUPANTS_PRESENT",)),
            }),
        )

        announcement = report.civilian_announcements[0]
        self.assertIn("Remain in place", announcement.announcement)
        self.assertNotIn("Proceed to", announcement.announcement)


class Test26EvacuateImmediatelyUnchanged(unittest.TestCase):

    def test_evacuate_immediately_never_changed_by_response_priority(self):

        policy = make_decision_policy(
            zone_decisions=[{"zone_id": "zone-a", "action": EVACUATE_IMMEDIATELY, "recommended_exit": "exit-1"}],
            exit_decisions=[{"exit_id": "exit-1", "status": KEEP_OPEN}],
        )

        report = generate(
            decision_policy=policy,
            response=response_evidence(critical_zone_ids=("zone-a",), zone_details={
                "zone-a": ZoneResponseDetail(priority_level="CRITICAL", reason_codes=("EVACUATION_STALLED",)),
            }),
        )

        announcement = report.civilian_announcements[0]
        self.assertIn("Proceed to", announcement.announcement)
        self.assertNotIn("Remain in place", announcement.announcement)
        self.assertNotIn("Hold your position", announcement.announcement)


class Test27ShelterInPlaceUnchanged(unittest.TestCase):

    def test_shelter_in_place_never_changed_by_response_priority(self):

        policy = make_decision_policy(
            zone_decisions=[{"zone_id": "zone-a", "action": SHELTER_IN_PLACE}],
            exit_decisions=[{"exit_id": "exit-1", "status": KEEP_OPEN}],
        )

        # No critical/high finding at all for zone-a -- an unambiguously
        # "good" observed state -- must not upgrade SHELTER_IN_PLACE.
        report = generate(decision_policy=policy, response=response_evidence())

        announcement = report.civilian_announcements[0]
        self.assertIn("Remain in place", announcement.announcement)


class Test28NoAutomaticVoice(unittest.TestCase):

    def test_no_voice_evacuation_controller_reachable_from_advisory_recommendations(self):

        policy = make_decision_policy(
            zone_decisions=[{"zone_id": "zone-a", "action": WAIT, "recommended_exit": "exit-1"}],
            exit_decisions=[{"exit_id": "exit-1", "status": KEEP_OPEN}],
        )

        report = generate(
            decision_policy=policy,
            response=response_evidence(critical_zone_ids=("zone-a",), zone_details={
                "zone-a": ZoneResponseDetail(priority_level="CRITICAL", reason_codes=("KNOWN_OCCUPANTS_PRESENT",)),
            }),
        )

        # Every BuildingRecommendation is an inert, informational record --
        # there is no "execute"/"broadcast" method anywhere on it.
        for rec in report.building_recommendations:
            self.assertFalse(hasattr(rec, "execute"))
            self.assertFalse(hasattr(rec, "broadcast"))


class Test29NoAutomaticBuildingControl(unittest.TestCase):

    def test_no_building_control_execution_reachable_from_advisory_recommendations(self):

        policy = make_decision_policy(
            zone_decisions=[{"zone_id": "zone-a", "action": WAIT, "recommended_exit": "exit-1"}],
            exit_decisions=[{"exit_id": "exit-1", "status": KEEP_OPEN}],
        )

        report = generate(
            decision_policy=policy,
            response=response_evidence(critical_zone_ids=("zone-a",), zone_details={
                "zone-a": ZoneResponseDetail(priority_level="CRITICAL", reason_codes=("HAZARD_PRESENT",)),
            }),
        )

        for rec in report.building_recommendations:
            self.assertFalse(hasattr(rec, "execute_control"))
            self.assertFalse(hasattr(rec, "confirm"))


class Test30NoAutomaticFirefighterDispatch(unittest.TestCase):

    def test_firefighter_intelligence_report_never_carries_a_dispatch_or_assignment_field(self):

        policy = make_decision_policy(
            zone_decisions=[{"zone_id": "zone-a", "action": WAIT, "recommended_exit": "exit-1"}],
            exit_decisions=[{"exit_id": "exit-1", "status": KEEP_OPEN}],
        )

        report = generate(
            decision_policy=policy,
            response=response_evidence(critical_zone_ids=("zone-a",), zone_details={
                "zone-a": ZoneResponseDetail(priority_level="CRITICAL", reason_codes=("KNOWN_OCCUPANTS_PRESENT",)),
            }),
        )

        report_dict = report.firefighter_intelligence.to_dict()
        self.assertNotIn("assigned_task", report_dict)
        self.assertNotIn("dispatch", report_dict)
        self.assertNotIn("mission", report_dict)

        # Response priority is reported as informational zone id lists
        # only -- never an instruction/order.
        self.assertIn("zone-a", report.firefighter_intelligence.live_priority_zone_ids)


class NoEvidencePreservesBehaviorTests(unittest.TestCase):

    def test_no_response_evidence_produces_identical_output_to_pre_milestone_behavior(self):

        policy = make_decision_policy(
            zone_decisions=[{"zone_id": "zone-a", "action": WAIT, "recommended_exit": "exit-1"}],
            exit_decisions=[{"exit_id": "exit-1", "status": KEEP_OPEN}],
        )

        without_field = generate(decision_policy=policy, response=None)
        without_arg_at_all = AdvisoryOrchestrator().generate_report(
            AdvisoryInputs(building=make_building(), scenario=make_scenario(), ground_truth=make_ground_truth(), decision_policy=policy)
        )

        self.assertEqual(without_field.civilian_announcements, without_arg_at_all.civilian_announcements)
        self.assertEqual(without_field.building_recommendations, without_arg_at_all.building_recommendations)


if __name__ == "__main__":
    unittest.main()
