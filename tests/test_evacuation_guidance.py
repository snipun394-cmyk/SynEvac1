import unittest

from hazard.severity import HazardSeverity

from evacuation_guidance.models import DeliveryStatus, GuidanceInconsistency, NavigationStepType, RouteStatus
from evacuation_recommendation.models import RecommendationStatus

from tests.evacuation_guidance_fixtures import FakeSpeakerManager, make_engine, make_recommendation_snapshot
from tests.trajectory_intelligence_fixtures import make_building_state


# =====================================================
# Live Evacuation Guidance & Zoned Message Planning milestone, Phase 25
# -- deterministic engine-level unit coverage. No randomness anywhere
# in this file.
# =====================================================


class BasicRouteTests(unittest.TestCase):

    def test_1_zone_already_at_exit(self):

        engine = make_engine()
        recommendation = make_recommendation_snapshot("z1", "f1", "EXIT-1")

        snapshot = engine.compute(0.0, recommendation, make_building_state())
        plan = snapshot.zone("z1")

        self.assertEqual(plan.route_status, RouteStatus.ALREADY_AT_EXIT)
        self.assertEqual(plan.recommended_exit_id, "EXIT-1")
        self.assertEqual(plan.ordered_door_ids, ())

    def test_2_multi_zone_route_via_door(self):

        engine = make_engine()
        recommendation = make_recommendation_snapshot("z2", "f1", "EXIT-1")

        snapshot = engine.compute(0.0, recommendation, make_building_state())
        plan = snapshot.zone("z2")

        self.assertEqual(plan.route_status, RouteStatus.ROUTE_AVAILABLE)
        self.assertEqual(plan.ordered_zone_ids, ("z2", "z1"))

    def test_3_door_appears_in_ordered_route(self):

        engine = make_engine()
        recommendation = make_recommendation_snapshot("z2", "f1", "EXIT-1")

        plan = engine.compute(0.0, recommendation, make_building_state()).zone("z2")

        self.assertEqual(plan.ordered_door_ids, ("DOOR-1",))

    def test_4_stair_appears_in_ordered_route(self):

        engine = make_engine()
        recommendation = make_recommendation_snapshot("z3", "f2", "EXIT-1")

        plan = engine.compute(0.0, recommendation, make_building_state()).zone("z3")

        self.assertEqual(plan.ordered_stair_ids, ("STAIR-1",))

    def test_5_multi_floor_route(self):

        engine = make_engine()
        recommendation = make_recommendation_snapshot("z3", "f2", "EXIT-1")

        plan = engine.compute(0.0, recommendation, make_building_state()).zone("z3")

        self.assertEqual(plan.route_status, RouteStatus.ROUTE_AVAILABLE)
        self.assertEqual(plan.ordered_zone_ids, ("z3", "z1"))

    def test_6_route_terminates_at_recommended_exit(self):

        engine = make_engine()
        recommendation = make_recommendation_snapshot("z2", "f1", "EXIT-1")

        plan = engine.compute(0.0, recommendation, make_building_state()).zone("z2")

        exit_steps = [s for s in plan.ordered_navigation_steps if s.step_type == NavigationStepType.CONTINUE_TO_EXIT]
        self.assertEqual(len(exit_steps), 1)
        self.assertEqual(exit_steps[0].exit_id, "EXIT-1")
        self.assertEqual(plan.ordered_navigation_steps[-1].step_type, NavigationStepType.EXIT_BUILDING)


class InvalidationTests(unittest.TestCase):

    def test_7_blocked_door_invalidates_route(self):

        engine = make_engine()
        recommendation = make_recommendation_snapshot("z2", "f1", "EXIT-1")

        for door in engine.building.ordered_floors()[0].doors:
            if door.id == "DOOR-1":
                door.active = False

        plan = engine.compute(0.0, recommendation, make_building_state()).zone("z2")

        self.assertNotEqual(plan.route_status, RouteStatus.ROUTE_AVAILABLE)
        self.assertFalse(plan.is_valid())

    def test_8_hazardous_intermediate_zone_invalidates_route(self):

        # z3 -> STAIR-1 -> z1 -> DOOR-1 -> z2 -> DOOR-2 -> z4 -> EXIT-2:
        # z1 is a pure intermediate hop here (not the exit's own zone),
        # proving hazard exclusion is checked along the WHOLE route, not
        # only at the terminal exit zone.
        engine = make_engine()
        recommendation = make_recommendation_snapshot("z3", "f2", "EXIT-2")

        plan = engine.compute(0.0, recommendation, make_building_state({"z1": HazardSeverity.HIGH})).zone("z3")

        self.assertFalse(plan.is_valid())
        self.assertIn(GuidanceInconsistency.RECOMMENDED_EXIT_UNREACHABLE, plan.inconsistencies)

    def test_9_unsafe_recommended_exit_never_receives_guidance(self):

        # Defensive mismatch scenario: the recommendation itself names
        # an exit whose own zone is (this cycle) hazardous -- Guidance's
        # own independent validation (Phase 10) must never fabricate a
        # route through it regardless.
        engine = make_engine()
        recommendation = make_recommendation_snapshot("z2", "f1", "EXIT-1")

        plan = engine.compute(0.0, recommendation, make_building_state({"z1": HazardSeverity.HIGH})).zone("z2")

        self.assertFalse(plan.is_valid())
        self.assertEqual(plan.ordered_door_ids, ())
        self.assertEqual(plan.ordered_navigation_steps, ())

    def test_10_unreachable_recommended_exit_reports_inconsistency(self):

        engine = make_engine()
        # EXIT-2 genuinely exists but z1 (a required hop for z3->EXIT-2)
        # is hazard-excluded.
        recommendation = make_recommendation_snapshot("z3", "f2", "EXIT-2")

        plan = engine.compute(0.0, recommendation, make_building_state({"z1": HazardSeverity.HIGH})).zone("z3")

        self.assertEqual(plan.route_status, RouteStatus.ROUTE_UNAVAILABLE)
        self.assertIn(GuidanceInconsistency.RECOMMENDED_EXIT_UNREACHABLE, plan.inconsistencies)

    def test_11_no_fabricated_route_when_exit_missing_from_graph(self):

        engine = make_engine()
        recommendation = make_recommendation_snapshot("z2", "f1", "EXIT-DOES-NOT-EXIST")

        plan = engine.compute(0.0, recommendation, make_building_state()).zone("z2")

        self.assertEqual(plan.route_status, RouteStatus.ROUTE_UNCERTAIN)
        self.assertFalse(plan.is_valid())
        self.assertEqual(plan.ordered_navigation_steps, ())

    def test_12_no_safe_exit_recommendation_produces_no_safe_exit_guidance(self):

        engine = make_engine()
        recommendation = make_recommendation_snapshot(
            "z1", "f1", None, status=RecommendationStatus.NO_SAFE_EXIT_AVAILABLE,
        )

        plan = engine.compute(0.0, recommendation, make_building_state()).zone("z1")

        self.assertEqual(plan.route_status, RouteStatus.NO_SAFE_EXIT)


class NoSilentSubstitutionTests(unittest.TestCase):

    def test_13_guidance_never_substitutes_a_different_exit(self):

        engine = make_engine()
        recommendation = make_recommendation_snapshot("z2", "f1", "EXIT-1")

        plan = engine.compute(0.0, recommendation, make_building_state({"z1": HazardSeverity.HIGH})).zone("z2")

        # EXIT-2 is genuinely reachable and safe from z2, but Guidance
        # must never silently switch to it behind Recommendation
        # Engine's back (Phase 22).
        self.assertEqual(plan.recommended_exit_id, "EXIT-1")
        self.assertNotEqual(plan.route_status, RouteStatus.ROUTE_AVAILABLE)


class DynamicReplanningTests(unittest.TestCase):

    def test_14_hazard_change_invalidates_old_route(self):

        engine = make_engine()
        recommendation = make_recommendation_snapshot("z2", "f1", "EXIT-1")

        before = engine.compute(0.0, recommendation, make_building_state()).zone("z2")
        self.assertTrue(before.is_valid())

        after = engine.compute(1.0, recommendation, make_building_state({"z1": HazardSeverity.HIGH})).zone("z2")
        self.assertFalse(after.is_valid())

    def test_15_recommendation_change_causes_route_change(self):

        engine = make_engine()

        first = engine.compute(0.0, make_recommendation_snapshot("z2", "f1", "EXIT-1"), make_building_state()).zone("z2")
        second = engine.compute(1.0, make_recommendation_snapshot("z2", "f1", "EXIT-2"), make_building_state()).zone("z2")

        self.assertNotEqual(first.recommended_exit_id, second.recommended_exit_id)
        self.assertNotEqual(first.ordered_door_ids, second.ordered_door_ids)


class RevisionTests(unittest.TestCase):

    def test_16_identical_recommendation_does_not_create_new_revision(self):

        engine = make_engine()
        recommendation = make_recommendation_snapshot("z2", "f1", "EXIT-1")

        first = engine.compute(0.0, recommendation, make_building_state()).zone("z2")
        second = engine.compute(1.0, recommendation, make_building_state()).zone("z2")

        self.assertEqual(first.revision, second.revision)

    def test_17_changed_route_increments_revision(self):

        engine = make_engine()

        first = engine.compute(0.0, make_recommendation_snapshot("z2", "f1", "EXIT-1"), make_building_state()).zone("z2")
        second = engine.compute(1.0, make_recommendation_snapshot("z2", "f1", "EXIT-2"), make_building_state()).zone("z2")

        self.assertGreater(second.revision, first.revision)

    def test_18_revision_is_deterministic_sequence_not_random(self):

        engine = make_engine()
        recommendation = make_recommendation_snapshot("z2", "f1", "EXIT-1")

        plan = engine.compute(0.0, recommendation, make_building_state()).zone("z2")

        self.assertEqual(plan.revision, 1)


class DeterminismTests(unittest.TestCase):

    def test_19_structured_route_deterministic(self):

        engine_a = make_engine()
        engine_b = make_engine()
        recommendation = make_recommendation_snapshot("z3", "f2", "EXIT-1")

        plan_a = engine_a.compute(0.0, recommendation, make_building_state()).zone("z3")
        plan_b = engine_b.compute(0.0, recommendation, make_building_state()).zone("z3")

        self.assertEqual(plan_a.to_dict(), plan_b.to_dict())

    def test_20_occupant_count_cannot_change_structural_route(self):

        engine = make_engine()

        few = make_recommendation_snapshot("z2", "f1", "EXIT-1")
        many = make_recommendation_snapshot("z2", "f1", "EXIT-1")
        object.__setattr__(list(many.zones.values())[0], "occupant_count", 50)

        plan_few = engine.compute(0.0, few, make_building_state()).zone("z2")
        plan_many = engine.compute(1.0, many, make_building_state()).zone("z2")

        self.assertEqual(plan_few.ordered_door_ids, plan_many.ordered_door_ids)
        self.assertEqual(plan_few.route_status, plan_many.route_status)


class SpeakerCoverageTests(unittest.TestCase):

    def test_21_speaker_coverage_found_for_originating_zone(self):

        engine = make_engine()
        recommendation = make_recommendation_snapshot("z2", "f1", "EXIT-1")
        speaker_manager = FakeSpeakerManager({"z2": ("SPK-2",)})

        snapshot = engine.compute(0.0, recommendation, make_building_state(), speaker_manager=speaker_manager)
        voice_plan = snapshot.voice_plan("z2")

        self.assertEqual(voice_plan.speaker_ids, ("SPK-2",))
        self.assertEqual(voice_plan.delivery_status, DeliveryStatus.PLANNED)

    def test_22_multiple_speakers_supported(self):

        engine = make_engine()
        recommendation = make_recommendation_snapshot("z2", "f1", "EXIT-1")
        speaker_manager = FakeSpeakerManager({"z2": ("SPK-2A", "SPK-2B")})

        voice_plan = engine.compute(0.0, recommendation, make_building_state(), speaker_manager=speaker_manager).voice_plan("z2")

        self.assertEqual(voice_plan.speaker_ids, ("SPK-2A", "SPK-2B"))

    def test_23_missing_speaker_coverage_does_not_invalidate_route(self):

        engine = make_engine()
        recommendation = make_recommendation_snapshot("z2", "f1", "EXIT-1")

        plan = engine.compute(0.0, recommendation, make_building_state(), speaker_manager=FakeSpeakerManager({})).zone("z2")

        self.assertTrue(plan.is_valid())
        self.assertEqual(plan.route_status, RouteStatus.ROUTE_AVAILABLE)

    def test_24_missing_speaker_coverage_reported_honestly(self):

        engine = make_engine()
        recommendation = make_recommendation_snapshot("z2", "f1", "EXIT-1")

        snapshot = engine.compute(0.0, recommendation, make_building_state(), speaker_manager=FakeSpeakerManager({}))
        plan = snapshot.zone("z2")
        voice_plan = snapshot.voice_plan("z2")

        self.assertIn(GuidanceInconsistency.NO_SPEAKER_COVERAGE, plan.inconsistencies)
        self.assertEqual(voice_plan.delivery_status, DeliveryStatus.NO_SPEAKER_COVERAGE)


class VoicePlanTests(unittest.TestCase):

    def test_25_voice_plan_generated_from_valid_guidance(self):

        engine = make_engine()
        recommendation = make_recommendation_snapshot("z2", "f1", "EXIT-1")

        snapshot = engine.compute(0.0, recommendation, make_building_state())

        self.assertIsNotNone(snapshot.voice_plan("z2"))

    def test_26_voice_plan_not_generated_from_invalid_guidance(self):

        engine = make_engine()
        recommendation = make_recommendation_snapshot(
            "z1", "f1", None, status=RecommendationStatus.NO_SAFE_EXIT_AVAILABLE,
        )

        snapshot = engine.compute(0.0, recommendation, make_building_state())

        self.assertIsNone(snapshot.voice_plan("z1"))

    def test_27_recommended_exit_survives_into_voice_plan(self):

        engine = make_engine()
        recommendation = make_recommendation_snapshot("z2", "f1", "EXIT-1")

        voice_plan = engine.compute(0.0, recommendation, make_building_state()).voice_plan("z2")

        self.assertEqual(voice_plan.recommended_exit_id, "EXIT-1")

    def test_28_revision_survives_into_voice_plan(self):

        engine = make_engine()
        recommendation = make_recommendation_snapshot("z2", "f1", "EXIT-1")

        plan = engine.compute(0.0, recommendation, make_building_state()).zone("z2")
        voice_plan = engine.compute(0.0, recommendation, make_building_state()).voice_plan("z2")

        self.assertEqual(plan.revision, voice_plan.guidance_revision)

    def test_29_voice_message_text_matches_deterministic_format(self):

        # Mirrors the milestone's own worked example exactly: "Zone Z3
        # -- D4 -> S2 -> E2" style route produces "Proceed through
        # Door D4 toward Stair S2 and continue to Exit E2."
        engine = make_engine()
        recommendation = make_recommendation_snapshot("z3", "f2", "EXIT-1")

        voice_plan = engine.compute(0.0, recommendation, make_building_state()).voice_plan("z3")

        self.assertEqual(voice_plan.message_text, "Zone z3: Proceed toward Stair 1 and continue to Main Exit.")

    def test_30_unsafe_exit_cannot_appear_through_text_generation_bug(self):

        engine = make_engine()
        recommendation = make_recommendation_snapshot("z2", "f1", "EXIT-1")

        snapshot = engine.compute(0.0, recommendation, make_building_state({"z1": HazardSeverity.HIGH}))

        self.assertIsNone(snapshot.voice_plan("z2"))


class InstructionAndStructureTests(unittest.TestCase):

    def test_31_full_route_reconstructable_from_structured_steps(self):

        engine = make_engine()
        recommendation = make_recommendation_snapshot("z3", "f2", "EXIT-1")

        plan = engine.compute(0.0, recommendation, make_building_state()).zone("z3")

        reconstructed_stairs = tuple(
            step.stair_id for step in plan.ordered_navigation_steps if step.step_type == NavigationStepType.USE_STAIR
        )
        reconstructed_exit = next(
            step.exit_id for step in plan.ordered_navigation_steps if step.step_type == NavigationStepType.CONTINUE_TO_EXIT
        )

        self.assertEqual(reconstructed_stairs, plan.ordered_stair_ids)
        self.assertEqual(reconstructed_exit, plan.recommended_exit_id)

    def test_32_cross_floor_instructions_preserve_stair_identity(self):

        engine = make_engine()
        recommendation = make_recommendation_snapshot("z3", "f2", "EXIT-1")

        plan = engine.compute(0.0, recommendation, make_building_state()).zone("z3")

        stair_instruction = next(text for text in plan.instructions if "Stair" in text)

        # The instruction uses the Staircase's own human-readable name
        # ("Stair 1"), not the raw graph id ("STAIR-1") -- the id is
        # still preserved structurally on the step/plan itself.
        self.assertIn("Stair 1", stair_instruction)
        self.assertEqual(plan.ordered_stair_ids, ("STAIR-1",))

    def test_33_no_fabricated_directions_or_landmarks(self):

        engine = make_engine()
        recommendation = make_recommendation_snapshot("z3", "f2", "EXIT-1")

        plan = engine.compute(0.0, recommendation, make_building_state()).zone("z3")

        forbidden = ("left", "right", "north", "south", "east", "west")

        for instruction in plan.instructions:

            lowered = instruction.lower()
            for word in forbidden:
                self.assertNotIn(word, lowered)

    def test_34_instruction_text_deterministic(self):

        engine_a = make_engine()
        engine_b = make_engine()
        recommendation = make_recommendation_snapshot("z2", "f1", "EXIT-1")

        instructions_a = engine_a.compute(0.0, recommendation, make_building_state()).zone("z2").instructions
        instructions_b = engine_b.compute(0.0, recommendation, make_building_state()).zone("z2").instructions

        self.assertEqual(instructions_a, instructions_b)


if __name__ == "__main__":
    unittest.main()
