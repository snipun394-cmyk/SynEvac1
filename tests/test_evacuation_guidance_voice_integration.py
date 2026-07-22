import unittest

from voice_evacuation.adapter import guidance_plan_to_voice_message
from voice_evacuation.controller import VoiceEvacuationController
from voice_evacuation.models import BroadcastStatus, VoiceMessageType

from speaker_manager.manager import SpeakerManager

from command_center.live_operator_action_gateway import (
    LiveOperatorActionGateway, OperatorActionUnavailable, VOICE_STATUS_RECOMMENDED,
    VOICE_STATUS_REJECTED, VOICE_STATUS_SENT,
)

from models.building import Building
from models.floor import Floor
from models.speaker import Speaker
from models.zone import Zone

from tests.evacuation_guidance_fixtures import FakeSpeakerManager, make_engine, make_recommendation_snapshot
from tests.trajectory_intelligence_fixtures import make_building_state


# =====================================================
# Live Evacuation Guidance & Zoned Message Planning milestone, Phase 15
# -- proves the "one final operator-visible message per zone" reconciliation
# decision (guidance produces its own separately-labelled, VoiceMessageType.
# ROUTE_GUIDANCE candidate, reconciled via VoiceEvacuationController's
# own pre-existing priority supersession, never a second competing
# broadcast path) end to end, plus operator-approval/rejection/history
# (Phase 16/19).
# =====================================================


class FakeSimulationProvider:

    is_simulation_only = True

    def __init__(self):
        self.sent = []

    def send(self, instruction):
        self.sent.append(instruction)
        return instruction


def make_speaker_manager():

    floor = Floor(
        id="f1", name="Floor 1",
        zones=[Zone(id="z2", name="Hall", x=20.0, y=0.0, width=10.0, height=10.0, floor_id="f1")],
        speakers=[Speaker(id="SPK-2", name="Speaker Hall", floor_id="f1", zone_ids=("z2",))],
    )
    building = Building(id="b", name="B", floors=[floor])

    manager = SpeakerManager()
    manager.discover_speakers(building)

    return manager


class AdapterTests(unittest.TestCase):

    def test_guidance_plan_becomes_route_guidance_voice_message(self):

        engine = make_engine()
        recommendation = make_recommendation_snapshot("z2", "f1", "EXIT-1")
        voice_plan = engine.compute(0.0, recommendation, make_building_state()).voice_plan("z2")

        message = guidance_plan_to_voice_message(voice_plan, timestamp=1.0)

        self.assertEqual(message.message_type, VoiceMessageType.ROUTE_GUIDANCE)
        self.assertEqual(message.target_zone_ids, ("z2",))
        self.assertEqual(message.message_text, voice_plan.message_text)


class OperatorApprovalTests(unittest.TestCase):

    def setUp(self):

        self.engine = make_engine()
        self.provider = FakeSimulationProvider()
        self.controller = VoiceEvacuationController(make_speaker_manager(), self.provider)
        self.gateway = LiveOperatorActionGateway(voice_controller=self.controller)

    def _voice_plan(self, exit_id="EXIT-1", time=0.0):

        recommendation = make_recommendation_snapshot("z2", "f1", exit_id, timestamp=time)
        return self.engine.compute(time, recommendation, make_building_state()).voice_plan("z2")

    def test_no_automatic_broadcast(self):

        self._voice_plan()
        self.assertEqual(self.provider.sent, [])

    def test_status_starts_recommended(self):

        plan = self._voice_plan()
        self.assertEqual(self.gateway.guidance_recommendation_status(plan), VOICE_STATUS_RECOMMENDED)

    def test_operator_approval_reaches_provider(self):

        plan = self._voice_plan()
        self.gateway.approve_guidance_message(plan, 1.0)

        self.assertEqual(len(self.provider.sent), 1)
        self.assertEqual(self.provider.sent[0].message.message_text, plan.message_text)
        self.assertEqual(self.gateway.guidance_recommendation_status(plan), VOICE_STATUS_SENT)

    def test_operator_rejection_causes_no_broadcast(self):

        plan = self._voice_plan()
        self.gateway.reject_guidance_message(plan, 1.0)

        self.assertEqual(self.provider.sent, [])
        self.assertEqual(self.gateway.guidance_recommendation_status(plan), VOICE_STATUS_REJECTED)

    def test_no_provider_configured_is_an_honest_failure(self):

        gateway = LiveOperatorActionGateway(voice_controller=None)
        plan = self._voice_plan()

        with self.assertRaises(OperatorActionUnavailable):
            gateway.approve_guidance_message(plan, 1.0)

    def test_changed_guidance_supersedes_unsent_old_guidance(self):

        old_plan = self._voice_plan(exit_id="EXIT-1", time=0.0)
        new_plan = self._voice_plan(exit_id="EXIT-2", time=1.0)

        self.assertNotEqual(old_plan.guidance_revision, new_plan.guidance_revision)

        # The old, never-approved revision was never sent, and approving
        # the NEW one never resurrects it -- a fresh RECOMMENDED status.
        self.assertEqual(self.gateway.guidance_recommendation_status(old_plan), VOICE_STATUS_RECOMMENDED)
        self.assertEqual(self.gateway.guidance_recommendation_status(new_plan), VOICE_STATUS_RECOMMENDED)

        self.gateway.approve_guidance_message(new_plan, 2.0)

        self.assertEqual(self.gateway.guidance_recommendation_status(new_plan), VOICE_STATUS_SENT)
        # The old revision's own key is untouched (still RECOMMENDED,
        # never silently flipped to APPROVED/SUPERSEDED by this call).
        self.assertEqual(self.gateway.guidance_recommendation_status(old_plan), VOICE_STATUS_RECOMMENDED)

    def test_sent_old_guidance_remains_in_history(self):

        old_plan = self._voice_plan(exit_id="EXIT-1", time=0.0)
        self.gateway.approve_guidance_message(old_plan, 0.5)

        new_plan = self._voice_plan(exit_id="EXIT-2", time=1.0)
        self.gateway.approve_guidance_message(new_plan, 1.5)

        history = self.gateway.voice_broadcast_history()
        texts = [instruction.message.message_text for instruction in history if instruction.message is not None]

        self.assertIn(old_plan.message_text, texts)
        self.assertIn(new_plan.message_text, texts)

    def test_duplicate_approval_click_does_not_rebroadcast(self):

        plan = self._voice_plan()

        self.gateway.approve_guidance_message(plan, 1.0)
        self.gateway.approve_guidance_message(plan, 2.0)

        self.assertEqual(len(self.provider.sent), 1)


class ReconciliationWithCivilianAnnouncementTests(unittest.TestCase):

    # Phase 15 -- "one final operator-visible message per zone/action
    # context": if a civilian EVACUATE announcement AND a guidance
    # ROUTE_GUIDANCE message both target the same zone, the controller's
    # own pre-existing priority supersession (EVACUATE=90 > ROUTE_
    # GUIDANCE=50) is what reconciles them -- never a second, independent
    # broadcast path, never both remaining simultaneously "active".

    def test_higher_priority_announcement_supersedes_guidance_message_for_same_zone(self):

        from advisory_system.recommendation_models import CivilianAnnouncement
        from voice_evacuation.adapter import civilian_announcement_to_voice_message
        from decision_policy.zone_policy import EVACUATE_IMMEDIATELY

        provider = FakeSimulationProvider()
        controller = VoiceEvacuationController(make_speaker_manager(), provider)

        engine = make_engine()
        recommendation = make_recommendation_snapshot("z2", "f1", "EXIT-1")
        guidance_plan = engine.compute(0.0, recommendation, make_building_state()).voice_plan("z2")

        guidance_message = guidance_plan_to_voice_message(guidance_plan, timestamp=0.0)
        controller.broadcast(guidance_message, 0.0)

        self.assertIsNotNone(controller.active_message_for_zone("z2"))
        self.assertEqual(controller.active_message_for_zone("z2").message_type, VoiceMessageType.ROUTE_GUIDANCE)

        announcement = CivilianAnnouncement(
            zone_id="z2", zone_name="Hall", announcement="Attention occupants in Hall. Evacuate immediately via Exit E1.",
            reason="test", confidence=0.9, predicted_rset_improvement_seconds=None,
        )
        evacuate_message = civilian_announcement_to_voice_message(announcement, timestamp=1.0, zone_action=EVACUATE_IMMEDIATELY)
        instructions = controller.broadcast(evacuate_message, 1.0)

        self.assertEqual(instructions[0].status, BroadcastStatus.BROADCAST)

        # Exactly one message is now active for the zone -- the higher-
        # priority civilian EVACUATE message -- never both simultaneously.
        active = controller.active_message_for_zone("z2")
        self.assertEqual(active.message_type, VoiceMessageType.EVACUATE)
        self.assertNotEqual(active.message_type, VoiceMessageType.ROUTE_GUIDANCE)

        # The superseded guidance broadcast remains in history, not erased.
        history = controller.broadcast_log.by_zone("z2")
        self.assertTrue(any(
            i.message is not None and i.message.message_type == VoiceMessageType.ROUTE_GUIDANCE for i in history
        ))


if __name__ == "__main__":
    unittest.main()
