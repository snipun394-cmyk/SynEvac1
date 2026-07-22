import unittest

from evacuation_guidance.models import EvacuationGuidanceSnapshot

from dynamic_signage.consistency import detect_inconsistencies
from dynamic_signage.models import DynamicSignageSnapshot, SignageInconsistency, SignageInstruction, SignageStatus
from dynamic_signage.planner import DynamicSignagePlanner

from tests.dynamic_signage_fixtures import make_guidance_snapshot, make_sign, make_signage_building


class ConsistencyTests(unittest.TestCase):

    def test_no_mismatch_for_matching_exit(self):

        building = make_signage_building()
        _, guidance_snapshot = make_guidance_snapshot(building, zone_id="z2", floor_id="f1", exit_id="EXIT-1")

        sign = make_sign("SIGN-1", zone_ids=("z2",), position=(22.0, 5.0), orientation=180.0)
        signage_snapshot = DynamicSignagePlanner(building).compute(0.0, guidance_snapshot, [sign])

        result = detect_inconsistencies(guidance_snapshot, signage_snapshot)

        self.assertEqual(result, {})

    def test_signage_guidance_mismatch_detected(self):

        building = make_signage_building()
        _, guidance_snapshot = make_guidance_snapshot(building, zone_id="z2", floor_id="f1", exit_id="EXIT-1")

        plan = guidance_snapshot.zone("z2")

        # Hand-construct a signage instruction that disagrees with the
        # real guidance plan's own recommended exit -- simulating a
        # stale/incorrectly-wired instruction, never something the
        # planner itself would normally produce.
        instruction = SignageInstruction(
            sign_id="SIGN-1", zone_id="z2", recommended_exit_id="EXIT-WRONG",
            guidance_revision=plan.revision, indication="STRAIGHT", status=SignageStatus.ACTIVE,
        )
        signage_snapshot = DynamicSignageSnapshot(timestamp=0.0, instructions={"SIGN-1": instruction}, conflicts={})

        result = detect_inconsistencies(guidance_snapshot, signage_snapshot)

        self.assertIn("SIGN-1", result)
        self.assertIn(SignageInconsistency.SIGNAGE_GUIDANCE_MISMATCH, result["SIGN-1"])

    def test_stale_signage_revision_detected(self):

        building = make_signage_building()
        _, guidance_snapshot = make_guidance_snapshot(building, zone_id="z2", floor_id="f1", exit_id="EXIT-1")

        plan = guidance_snapshot.zone("z2")

        instruction = SignageInstruction(
            sign_id="SIGN-1", zone_id="z2", recommended_exit_id=plan.recommended_exit_id,
            guidance_revision=plan.revision - 1 if plan.revision > 0 else plan.revision + 1,
            indication="STRAIGHT", status=SignageStatus.ACTIVE,
        )
        signage_snapshot = DynamicSignageSnapshot(timestamp=0.0, instructions={"SIGN-1": instruction}, conflicts={})

        result = detect_inconsistencies(guidance_snapshot, signage_snapshot)

        self.assertIn(SignageInconsistency.STALE_SIGNAGE_REVISION, result["SIGN-1"])

    def test_voice_signage_exit_mismatch_detected(self):

        building = make_signage_building()
        _, guidance_snapshot = make_guidance_snapshot(building, zone_id="z2", floor_id="f1", exit_id="EXIT-1")

        voice_plan = guidance_snapshot.voice_plan("z2")
        self.assertIsNotNone(voice_plan)

        instruction = SignageInstruction(
            sign_id="SIGN-1", zone_id="z2", recommended_exit_id="EXIT-DIFFERENT",
            guidance_revision=guidance_snapshot.zone("z2").revision, indication="STRAIGHT",
            status=SignageStatus.ACTIVE,
        )
        signage_snapshot = DynamicSignageSnapshot(timestamp=0.0, instructions={"SIGN-1": instruction}, conflicts={})

        result = detect_inconsistencies(guidance_snapshot, signage_snapshot)

        self.assertIn(SignageInconsistency.VOICE_SIGNAGE_EXIT_MISMATCH, result["SIGN-1"])

    def test_unavailable_instruction_never_flagged(self):

        instruction = SignageInstruction(sign_id="SIGN-1", zone_id=None, indication="UNAVAILABLE", status=SignageStatus.UNAVAILABLE)
        signage_snapshot = DynamicSignageSnapshot(timestamp=0.0, instructions={"SIGN-1": instruction}, conflicts={})

        result = detect_inconsistencies(EvacuationGuidanceSnapshot(timestamp=0.0, zones={}, voice_plans={}), signage_snapshot)

        self.assertEqual(result, {})


if __name__ == "__main__":
    unittest.main()
