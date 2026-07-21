import unittest

from advisory_system.evacuation_progress_evidence import EvacuationExitDetail, EvacuationProgressEvidence
from advisory_system.orchestrator import AdvisoryOrchestrator
from advisory_system.recommendation_models import AdvisoryInputs

from decision_policy.exit_policy import CLOSE, KEEP_OPEN
from decision_policy.zone_policy import EVACUATE_IMMEDIATELY, SHELTER_IN_PLACE, WAIT

from tests.test_advisory_system import make_building, make_decision_policy, make_ground_truth, make_scenario


# =====================================================
# Live Evacuation Progress, Flow & Clearance Intelligence milestone,
# Phase 19 items 20-25 -- evacuation-progress-specific safety
# precedence. Mirrors tests/test_crowd_advisory_safety_precedence.py's
# own structure exactly: progress evidence is SECONDARY, SUPPORTING
# evidence that can never override a deterministic safety restriction.
# =====================================================


def progress_evidence(*, high_queue_low_flow_exit_ids=(), stalled_zone_ids=(), exit_details=None) -> EvacuationProgressEvidence:

    return EvacuationProgressEvidence(
        available=True, timestamp=0.0,
        high_queue_low_flow_exit_ids=tuple(high_queue_low_flow_exit_ids),
        stalled_zone_ids=tuple(stalled_zone_ids),
        exit_details=exit_details or {},
    )


def generate(*, decision_policy, progress=None, crowd=None, ai=None):

    inputs = AdvisoryInputs(
        building=make_building(), scenario=make_scenario(), ground_truth=make_ground_truth(),
        decision_policy=decision_policy, ai_decision_evidence=ai, crowd_decision_evidence=crowd,
        evacuation_progress_evidence=progress,
    )

    return AdvisoryOrchestrator().generate_report(inputs)


class Test20UnsafeExitGoodFlowRemainsUnsafe(unittest.TestCase):

    def test_unsafe_exit_with_excellent_observed_flow_stays_unsafe(self):

        policy = make_decision_policy(
            zone_decisions=[{"zone_id": "zone-a", "action": EVACUATE_IMMEDIATELY, "recommended_exit": "exit-1"}],
            exit_decisions=[{"exit_id": "exit-1", "status": CLOSE}, {"exit_id": "exit-2", "status": KEEP_OPEN}],
        )

        # exit-1 is unsafe (CLOSE) but has "excellent" observed flow
        # (not flagged as high-queue-low-flow at all) -- must not matter.
        report = generate(decision_policy=policy, progress=progress_evidence(high_queue_low_flow_exit_ids=()))

        announcement = report.civilian_announcements[0]
        self.assertIn("Remain in place", announcement.announcement)
        self.assertNotIn("Proceed to", announcement.announcement)


class Test21SafeCongestedExitLowFlowMaySupport(unittest.TestCase):

    def test_safe_exit_with_low_flow_may_influence_supporting_evidence(self):

        policy = make_decision_policy(
            zone_decisions=[{"zone_id": "zone-a", "action": WAIT, "recommended_exit": "exit-1"}],
            exit_decisions=[{"exit_id": "exit-1", "status": KEEP_OPEN}],
        )

        baseline = generate(decision_policy=policy, progress=None).civilian_announcements[0]

        report = generate(
            decision_policy=policy,
            progress=progress_evidence(
                high_queue_low_flow_exit_ids=("exit-1",),
                exit_details={"exit-1": EvacuationExitDetail(queue_candidate_count=4)},
            ),
        )
        augmented = report.civilian_announcements[0]

        self.assertIn("Hold your position", baseline.announcement)
        self.assertIn("Hold your position", augmented.announcement)  # action/text unchanged
        self.assertGreater(augmented.confidence, baseline.confidence)
        self.assertIn("progress", augmented.confidence_source)
        self.assertIn("exit-1", augmented.reason)

        review_actions = [rec.action for rec in report.building_recommendations if "progress" in rec.confidence_source]
        self.assertIn("Review Exit exit-1 Throughput", review_actions)


class Test22EvacuateImmediatelyUnchanged(unittest.TestCase):

    def test_evacuate_immediately_never_changed_by_stalled_progress_evidence(self):

        policy = make_decision_policy(
            zone_decisions=[{"zone_id": "zone-a", "action": EVACUATE_IMMEDIATELY, "recommended_exit": "exit-1"}],
            exit_decisions=[{"exit_id": "exit-1", "status": KEEP_OPEN}],
        )

        report = generate(
            decision_policy=policy,
            progress=progress_evidence(stalled_zone_ids=("zone-a",), high_queue_low_flow_exit_ids=("exit-1",)),
        )

        announcement = report.civilian_announcements[0]
        self.assertIn("Proceed to", announcement.announcement)
        self.assertNotIn("Remain in place", announcement.announcement)
        self.assertNotIn("Hold your position", announcement.announcement)


class Test23ShelterInPlaceUnchanged(unittest.TestCase):

    def test_shelter_in_place_never_changed_by_good_exit_flow(self):

        policy = make_decision_policy(
            zone_decisions=[{"zone_id": "zone-a", "action": SHELTER_IN_PLACE}],
            exit_decisions=[{"exit_id": "exit-1", "status": KEEP_OPEN}],
        )

        # No stalled/high-queue-low-flow findings at all -- an
        # unambiguously "good" observed state -- must not upgrade
        # SHELTER_IN_PLACE to evacuation.
        report = generate(decision_policy=policy, progress=progress_evidence())

        announcement = report.civilian_announcements[0]
        self.assertIn("Remain in place", announcement.announcement)


class Test24AIUnavailableProgressStillWorks(unittest.TestCase):

    def test_progress_evidence_functions_without_ai(self):

        policy = make_decision_policy(
            zone_decisions=[{"zone_id": "zone-a", "action": WAIT, "recommended_exit": "exit-1"}],
            exit_decisions=[{"exit_id": "exit-1", "status": KEEP_OPEN}],
        )

        report = generate(
            decision_policy=policy, ai=None,
            progress=progress_evidence(high_queue_low_flow_exit_ids=("exit-1",)),
        )

        announcement = report.civilian_announcements[0]
        self.assertIn("progress", announcement.confidence_source)
        self.assertNotIn("ai", announcement.confidence_source)


class Test25CrowdUnavailableProgressStillWorks(unittest.TestCase):

    def test_progress_evidence_functions_without_crowd_evidence(self):

        policy = make_decision_policy(
            zone_decisions=[{"zone_id": "zone-a", "action": WAIT, "recommended_exit": "exit-1"}],
            exit_decisions=[{"exit_id": "exit-1", "status": KEEP_OPEN}],
        )

        report = generate(
            decision_policy=policy, crowd=None,
            progress=progress_evidence(high_queue_low_flow_exit_ids=("exit-1",)),
        )

        announcement = report.civilian_announcements[0]
        self.assertIn("progress", announcement.confidence_source)
        self.assertNotIn("crowd", announcement.confidence_source)


class NoProgressEvidencePreservesBehaviorTests(unittest.TestCase):

    def test_no_progress_evidence_produces_identical_output_to_pre_milestone_behavior(self):

        policy = make_decision_policy(
            zone_decisions=[{"zone_id": "zone-a", "action": WAIT, "recommended_exit": "exit-1"}],
            exit_decisions=[{"exit_id": "exit-1", "status": KEEP_OPEN}],
        )

        without_field = generate(decision_policy=policy, progress=None)
        without_arg_at_all = AdvisoryOrchestrator().generate_report(
            AdvisoryInputs(building=make_building(), scenario=make_scenario(), ground_truth=make_ground_truth(), decision_policy=policy)
        )

        self.assertEqual(without_field.civilian_announcements, without_arg_at_all.civilian_announcements)
        self.assertEqual(without_field.building_recommendations, without_arg_at_all.building_recommendations)


if __name__ == "__main__":
    unittest.main()
