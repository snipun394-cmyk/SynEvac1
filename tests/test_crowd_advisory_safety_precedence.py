import unittest

from advisory_system.ai_evidence import AIDecisionEvidence
from advisory_system.crowd_evidence import CrowdAssetDetail, CrowdDecisionEvidence
from advisory_system.orchestrator import AdvisoryOrchestrator
from advisory_system.recommendation_models import AdvisoryInputs

from decision_policy.exit_policy import CLOSE, KEEP_OPEN
from decision_policy.stair_policy import AVOID, USE
from decision_policy.zone_policy import EVACUATE_IMMEDIATELY, SHELTER_IN_PLACE, WAIT

from tests.test_advisory_system import make_building, make_decision_policy, make_ground_truth, make_scenario


# =====================================================
# Live Crowd Intelligence -> Operational Advisory Integration milestone,
# Phase 16 -- THE required safety test matrix. The single, non-
# negotiable invariant every test here proves in some form: crowd
# congestion is SECONDARY, SUPPORTING evidence that can never override
# a deterministic safety restriction, create/remove a CLOSE/AVOID
# status, or change an EVACUATE_IMMEDIATELY/SHELTER_IN_PLACE decision.
# =====================================================


def crowd_evidence(
    *, congested_exit_ids=(), congested_stair_ids=(), position_coverage_fraction=1.0,
    position_unavailable_asset_ids=(), asset_details=None,
) -> CrowdDecisionEvidence:

    return CrowdDecisionEvidence(
        available=True,
        timestamp=0.0,
        congested_exit_ids=tuple(congested_exit_ids),
        congested_stair_ids=tuple(congested_stair_ids),
        position_coverage_fraction=position_coverage_fraction,
        position_unavailable_asset_ids=tuple(position_unavailable_asset_ids),
        asset_details=asset_details or {},
    )


def generate(*, decision_policy, crowd=None, ai=None):

    inputs = AdvisoryInputs(
        building=make_building(), scenario=make_scenario(), ground_truth=make_ground_truth(),
        decision_policy=decision_policy, ai_decision_evidence=ai, crowd_decision_evidence=crowd,
    )

    return AdvisoryOrchestrator().generate_report(inputs)


def _crowd_sourced_exit_targets(report):

    # Deliberately restricted to CROWD-sourced recommendations
    # (confidence_source == ("crowd",)) -- pre-existing, unrelated
    # building recommendations (e.g. "Unlock Exit exit-1", a physical-
    # unlock suggestion for a CLOSE exit that predates this milestone
    # entirely) legitimately mention an unsafe exit's id without
    # claiming it is usable/preferred; this helper only inspects what
    # THIS milestone's own crowd-intelligence logic produced.
    return {
        rec.target_id for rec in report.building_recommendations
        if rec.target_type == "exit" and "crowd" in rec.confidence_source
    }


def _prefer_recommendations(report):

    return [rec for rec in report.building_recommendations if rec.action.startswith("Prefer ")]


class Test1UnsafeExitNoCongestion(unittest.TestCase):

    def test_unsafe_exit_stays_unsafe_with_no_congestion_evidence(self):

        policy = make_decision_policy(
            zone_decisions=[{"zone_id": "zone-a", "action": EVACUATE_IMMEDIATELY, "recommended_exit": "exit-1"}],
            exit_decisions=[{"exit_id": "exit-1", "status": CLOSE}, {"exit_id": "exit-2", "status": KEEP_OPEN}],
        )

        report = generate(decision_policy=policy, crowd=crowd_evidence())

        announcement = report.civilian_announcements[0]
        self.assertEqual(announcement.announcement.count("Remain in place"), 1)  # degraded to SHELTER_IN_PLACE
        self.assertNotIn("exit-1", _crowd_sourced_exit_targets(report))


class Test2UnsafeExitLowCongestion(unittest.TestCase):

    def test_unsafe_exit_stays_unsafe_when_crowd_reports_it_as_uncongested(self):

        policy = make_decision_policy(
            zone_decisions=[{"zone_id": "zone-a", "action": EVACUATE_IMMEDIATELY, "recommended_exit": "exit-1"}],
            exit_decisions=[{"exit_id": "exit-1", "status": CLOSE}, {"exit_id": "exit-2", "status": KEEP_OPEN}],
        )

        # exit-1 is NOT in congested_exit_ids -- a "low congestion" reading.
        report = generate(decision_policy=policy, crowd=crowd_evidence(congested_exit_ids=()))

        announcement = report.civilian_announcements[0]
        self.assertIn("Remain in place", announcement.announcement)
        self.assertNotIn("Proceed to", announcement.announcement)


class Test3UnsafeExitVsAlternativeSafeCongestedExit(unittest.TestCase):

    def test_unsafe_exit_never_selected_even_when_the_only_alternative_is_congested(self):

        policy = make_decision_policy(
            zone_decisions=[{"zone_id": "zone-a", "action": EVACUATE_IMMEDIATELY, "recommended_exit": "exit-1"}],
            exit_decisions=[{"exit_id": "exit-1", "status": CLOSE}, {"exit_id": "exit-2", "status": KEEP_OPEN}],
        )

        report = generate(decision_policy=policy, crowd=crowd_evidence(congested_exit_ids=("exit-2",)))

        announcement = report.civilian_announcements[0]
        self.assertIn("Remain in place", announcement.announcement)

        # No CROWD-sourced recommendation ever targets or names the
        # unsafe exit -- the pre-existing, unrelated "Unlock Exit
        # exit-1" recommendation (a physical building-control suggestion
        # for a CLOSE exit) is untouched by this milestone and is not
        # what this assertion is about.
        for rec in report.building_recommendations:
            if "crowd" in rec.confidence_source:
                self.assertNotIn("exit-1", rec.action)
        self.assertNotIn("exit-1", _crowd_sourced_exit_targets(report))


class Test4TwoSafeExitsOneCongested(unittest.TestCase):

    def test_congestion_evidence_may_support_the_clear_safe_exit(self):

        policy = make_decision_policy(
            zone_decisions=[{"zone_id": "zone-a", "action": EVACUATE_IMMEDIATELY, "recommended_exit": "exit-1"}],
            exit_decisions=[{"exit_id": "exit-1", "status": KEEP_OPEN}, {"exit_id": "exit-2", "status": KEEP_OPEN}],
        )

        report = generate(
            decision_policy=policy,
            crowd=crowd_evidence(
                congested_exit_ids=("exit-1",),
                asset_details={"exit-1": CrowdAssetDetail(asset_type="Exit", congestion_level="HIGH", trend="RISING")},
            ),
        )

        preferences = _prefer_recommendations(report)
        self.assertEqual(len(preferences), 1)
        self.assertEqual(preferences[0].target_id, "exit-2")
        self.assertIn("exit-1", preferences[0].reason)


class Test5TwoSafeStairsOneCongested(unittest.TestCase):

    def test_congestion_evidence_may_support_the_clear_safe_stair(self):

        policy = make_decision_policy(
            zone_decisions=[{"zone_id": "zone-b", "action": WAIT, "recommended_stair": "stair-1"}],
            stair_decisions=[{"stair_id": "stair-1", "status": USE}, {"stair_id": "stair-2", "status": USE}],
        )

        report = generate(
            decision_policy=policy,
            crowd=crowd_evidence(
                congested_stair_ids=("stair-1",),
                asset_details={"stair-1": CrowdAssetDetail(asset_type="Stair", congestion_level="HIGH", trend="RISING")},
            ),
        )

        preferences = [rec for rec in _prefer_recommendations(report) if rec.target_type == "stair"]
        self.assertEqual(len(preferences), 1)
        self.assertEqual(preferences[0].target_id, "stair-2")


class Test6EvacuateImmediatelyExtremeCongestion(unittest.TestCase):

    def test_evacuate_immediately_never_downgraded_by_extreme_congestion(self):

        policy = make_decision_policy(
            zone_decisions=[{"zone_id": "zone-a", "action": EVACUATE_IMMEDIATELY, "recommended_exit": "exit-1"}],
            exit_decisions=[{"exit_id": "exit-1", "status": KEEP_OPEN}, {"exit_id": "exit-2", "status": KEEP_OPEN}],
        )

        report = generate(
            decision_policy=policy,
            crowd=crowd_evidence(
                congested_exit_ids=("exit-1", "exit-2"),
                asset_details={
                    "exit-1": CrowdAssetDetail(asset_type="Exit", congestion_level="CRITICAL", trend="RISING"),
                    "exit-2": CrowdAssetDetail(asset_type="Exit", congestion_level="CRITICAL", trend="RISING"),
                },
            ),
        )

        announcement = report.civilian_announcements[0]
        self.assertIn("Proceed to", announcement.announcement)
        self.assertNotIn("Remain in place", announcement.announcement)
        self.assertNotIn("Hold your position", announcement.announcement)


class Test7ShelterInPlaceClearRoute(unittest.TestCase):

    def test_shelter_in_place_never_changed_by_a_clear_route(self):

        policy = make_decision_policy(
            zone_decisions=[{"zone_id": "zone-a", "action": SHELTER_IN_PLACE}],
            exit_decisions=[{"exit_id": "exit-1", "status": KEEP_OPEN}, {"exit_id": "exit-2", "status": KEEP_OPEN}],
        )

        report = generate(decision_policy=policy, crowd=crowd_evidence(congested_exit_ids=("exit-2",)))

        announcement = report.civilian_announcements[0]
        self.assertIn("Remain in place", announcement.announcement)


class Test8WaitZoneHighRisingCongestion(unittest.TestCase):

    def test_wait_zone_confidence_and_reason_change_but_action_stays_wait(self):

        policy = make_decision_policy(
            zone_decisions=[{"zone_id": "zone-a", "action": WAIT, "recommended_exit": "exit-1"}],
            exit_decisions=[{"exit_id": "exit-1", "status": KEEP_OPEN}, {"exit_id": "exit-2", "status": KEEP_OPEN}],
        )

        baseline = generate(decision_policy=policy, crowd=None).civilian_announcements[0]

        report = generate(
            decision_policy=policy,
            crowd=crowd_evidence(
                congested_exit_ids=("exit-1",),
                asset_details={"exit-1": CrowdAssetDetail(asset_type="Exit", congestion_level="CRITICAL", trend="RISING")},
            ),
        )
        augmented = report.civilian_announcements[0]

        self.assertIn("Hold your position", baseline.announcement)
        self.assertIn("Hold your position", augmented.announcement)  # action/text unchanged
        self.assertGreater(augmented.confidence, baseline.confidence)  # confidence may rise
        self.assertIn("crowd", augmented.confidence_source)
        self.assertIn("exit-1", augmented.reason)  # reason may explain why


class Test9NoCrowdEvidencePreservesExistingBehavior(unittest.TestCase):

    def test_no_crowd_evidence_produces_byte_identical_civilian_output_to_pre_milestone_behavior(self):

        policy = make_decision_policy(
            zone_decisions=[{"zone_id": "zone-a", "action": WAIT, "recommended_exit": "exit-1"}],
            exit_decisions=[{"exit_id": "exit-1", "status": KEEP_OPEN}],
        )

        without_field = generate(decision_policy=policy, crowd=None)
        without_crowd_arg_at_all = AdvisoryOrchestrator().generate_report(
            AdvisoryInputs(building=make_building(), scenario=make_scenario(), ground_truth=make_ground_truth(), decision_policy=policy)
        )

        self.assertEqual(without_field.civilian_announcements, without_crowd_arg_at_all.civilian_announcements)
        self.assertEqual(without_field.building_recommendations, without_crowd_arg_at_all.building_recommendations)
        self.assertEqual(without_field.civilian_announcements[0].confidence_source, ())


class Test10PartialCalibrationNoFalseClear(unittest.TestCase):

    def test_partial_coverage_carries_an_explicit_caveat_never_a_bare_clear_claim(self):

        policy = make_decision_policy(
            zone_decisions=[{"zone_id": "zone-a", "action": EVACUATE_IMMEDIATELY, "recommended_exit": "exit-1"}],
            exit_decisions=[{"exit_id": "exit-1", "status": KEEP_OPEN}, {"exit_id": "exit-2", "status": KEEP_OPEN}],
        )

        report = generate(
            decision_policy=policy,
            crowd=crowd_evidence(
                congested_exit_ids=("exit-1",), position_coverage_fraction=0.35,
                asset_details={"exit-1": CrowdAssetDetail(asset_type="Exit", congestion_level="HIGH", trend="STABLE")},
            ),
        )

        preferences = _prefer_recommendations(report)
        self.assertEqual(len(preferences), 1)
        self.assertIn("35%", preferences[0].reason)
        self.assertIn("does not confirm", preferences[0].reason)


class Test11ZeroPositionCoverageTreatedUnavailable(unittest.TestCase):

    def test_asset_with_no_position_coverage_is_never_treated_as_a_confirmed_clear_alternative(self):

        policy = make_decision_policy(
            zone_decisions=[{"zone_id": "zone-a", "action": EVACUATE_IMMEDIATELY, "recommended_exit": "exit-1"}],
            exit_decisions=[{"exit_id": "exit-1", "status": KEEP_OPEN}, {"exit_id": "exit-2", "status": KEEP_OPEN}],
        )

        # exit-1 congested (real position data); exit-2 has ZERO position
        # coverage -- absence from congested_exit_ids must NOT make it a
        # recommended "clear" alternative.
        report = generate(
            decision_policy=policy,
            crowd=crowd_evidence(
                congested_exit_ids=("exit-1",),
                position_unavailable_asset_ids=("exit-2",),
                position_coverage_fraction=0.5,
                asset_details={"exit-1": CrowdAssetDetail(asset_type="Exit", congestion_level="HIGH", trend="STABLE")},
            ),
        )

        self.assertEqual(_prefer_recommendations(report), [])


class Test12AIAndCrowdBothAvailable(unittest.TestCase):

    def test_ai_and_crowd_evidence_are_both_visible_independently(self):

        policy = make_decision_policy(
            zone_decisions=[{"zone_id": "zone-a", "action": WAIT, "recommended_exit": "exit-1"}],
            exit_decisions=[{"exit_id": "exit-1", "status": KEEP_OPEN}],
        )

        ai = AIDecisionEvidence(
            available=True, bottleneck_occurrence_probability=0.8, bottleneck_predicted=True,
            model_id="m1", model_version="v1", model_status="PRODUCTION_CANDIDATE",
        )
        crowd = crowd_evidence(
            congested_exit_ids=("exit-1",),
            asset_details={"exit-1": CrowdAssetDetail(asset_type="Exit", congestion_level="HIGH", trend="RISING")},
        )

        report = generate(decision_policy=policy, crowd=crowd, ai=ai)
        announcement = report.civilian_announcements[0]

        self.assertIn("ai", announcement.confidence_source)
        self.assertIn("crowd", announcement.confidence_source)

        building_monitor_actions = [rec.action for rec in report.building_recommendations]
        self.assertIn("Monitor for Building-Wide Congestion", building_monitor_actions)
        self.assertIn("Monitor Congestion at Exit exit-1", building_monitor_actions)


class Test13AIUnavailableCrowdAvailable(unittest.TestCase):

    def test_crowd_support_works_without_ai(self):

        policy = make_decision_policy(
            zone_decisions=[{"zone_id": "zone-a", "action": WAIT, "recommended_exit": "exit-1"}],
            exit_decisions=[{"exit_id": "exit-1", "status": KEEP_OPEN}],
        )

        crowd = crowd_evidence(
            congested_exit_ids=("exit-1",),
            asset_details={"exit-1": CrowdAssetDetail(asset_type="Exit", congestion_level="HIGH", trend="RISING")},
        )

        report = generate(decision_policy=policy, crowd=crowd, ai=None)
        announcement = report.civilian_announcements[0]

        self.assertIn("crowd", announcement.confidence_source)
        self.assertNotIn("ai", announcement.confidence_source)


class Test14CrowdUnavailableAIAvailable(unittest.TestCase):

    def test_ai_support_still_works_without_crowd(self):

        policy = make_decision_policy(
            zone_decisions=[{"zone_id": "zone-a", "action": WAIT, "recommended_exit": "exit-1"}],
            exit_decisions=[{"exit_id": "exit-1", "status": KEEP_OPEN}],
        )

        ai = AIDecisionEvidence(
            available=True, bottleneck_occurrence_probability=0.9, bottleneck_predicted=True,
            model_id="m1", model_version="v1", model_status="PRODUCTION_CANDIDATE",
        )

        report = generate(decision_policy=policy, crowd=None, ai=ai)
        announcement = report.civilian_announcements[0]

        self.assertIn("ai", announcement.confidence_source)
        self.assertNotIn("crowd", announcement.confidence_source)


class Test15BothUnavailable(unittest.TestCase):

    def test_deterministic_advisory_still_works_with_neither(self):

        policy = make_decision_policy(
            zone_decisions=[{"zone_id": "zone-a", "action": EVACUATE_IMMEDIATELY, "recommended_exit": "exit-1"}],
            exit_decisions=[{"exit_id": "exit-1", "status": KEEP_OPEN}],
        )

        report = generate(decision_policy=policy, crowd=None, ai=None)
        announcement = report.civilian_announcements[0]

        self.assertIn("Proceed to", announcement.announcement)
        self.assertEqual(announcement.confidence_source, ())
        self.assertIsNotNone(announcement.confidence)


if __name__ == "__main__":
    unittest.main()
