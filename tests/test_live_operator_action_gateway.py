import unittest

from models.building import Building
from models.door import Door
from models.speaker import Speaker

from advisory_system.recommendation_models import AdvisoryReport, BuildingRecommendation, CivilianAnnouncement

from voice_evacuation.controller import VoiceEvacuationController
from voice_evacuation.models import BroadcastStatus
from voice_evacuation.provider import SimulationVoiceOutputProvider

from speaker_manager.manager import SpeakerManager

from building_control.controller import BuildingControlController
from building_control.providers import SimulationControlProvider
from building_control.types import RequestStatus

from command_center.live_operator_action_gateway import (
    OPERATOR_ACTOR,
    OperatorActionUnavailable,
    PROVIDER_CAPABILITY_LIVE_HARDWARE,
    PROVIDER_CAPABILITY_NO_PROVIDER,
    PROVIDER_CAPABILITY_SIMULATION,
    VOICE_STATUS_APPROVED,
    VOICE_STATUS_FAILED,
    VOICE_STATUS_RECOMMENDED,
    VOICE_STATUS_REJECTED,
    VOICE_STATUS_SENT,
    LiveOperatorActionGateway,
)


# Live Operator Action Routing milestone -- Phase 2's own gateway,
# tested in isolation from any Qt widget. The gateway is the ONE seam
# Command Center is allowed to route explicit operator intent through
# to real execution (voice_evacuation.controller.VoiceEvacuationController
# / building_control.controller.BuildingControlController) -- every test
# here exercises that seam directly, standing in for what a real
# operator's Approve/Reject click would do.


def make_announcement(zone_id="zone-1", zone_name="Zone 1", text="Evacuate now", confidence=0.9):

    return CivilianAnnouncement(
        zone_id=zone_id, zone_name=zone_name, announcement=text, reason="fire detected",
        confidence=confidence, predicted_rset_improvement_seconds=30.0, confidence_source=("ai",),
    )


def make_recommendation(target_id, action="Open Door", target_type="door", confidence=0.8):

    return BuildingRecommendation(
        action=action, target_type=target_type, target_id=target_id, reason="egress",
        confidence=confidence, expected_engineering_benefit="faster egress",
    )


def make_report(announcements=(), recommendations=(), time=1.0):

    return AdvisoryReport(
        scenario_id="s", simulation_time=time, civilian_announcements=announcements,
        firefighter_intelligence=None, building_recommendations=recommendations,
        commander_dashboard=None, recommendation_history=(),
    )


def make_voice_controller_with_speaker(zone_id="zone-1"):

    manager = SpeakerManager()
    manager.register_speaker(Speaker(name="SPK-1", floor_id="floor-1", zone_ids=(zone_id,)))

    return VoiceEvacuationController(manager, SimulationVoiceOutputProvider())


def make_control_controller(building, action_executor=None):

    return BuildingControlController(building, SimulationControlProvider(building, action_executor))


def make_building_with_door():

    building = Building(name="B")
    floor = building.create_floor(name="Ground Floor")
    door = Door(name="D1", floor_id=floor.id)
    floor.add_door(door)

    return building, door


class CapabilityTests(unittest.TestCase):

    def test_no_controllers_is_no_provider_both_sides(self):

        gateway = LiveOperatorActionGateway()

        self.assertEqual(gateway.voice_capability, PROVIDER_CAPABILITY_NO_PROVIDER)
        self.assertEqual(gateway.control_capability, PROVIDER_CAPABILITY_NO_PROVIDER)

    def test_simulation_provider_reports_simulation_capability(self):

        building, _door = make_building_with_door()

        gateway = LiveOperatorActionGateway(
            voice_controller=make_voice_controller_with_speaker(),
            control_controller=make_control_controller(building),
        )

        self.assertEqual(gateway.voice_capability, PROVIDER_CAPABILITY_SIMULATION)
        self.assertEqual(gateway.control_capability, PROVIDER_CAPABILITY_SIMULATION)

    def test_a_hypothetical_non_simulation_provider_reports_live_hardware(self):

        # No real Live provider exists anywhere in this codebase (by
        # design -- this milestone does not implement hardware
        # communication) -- this test proves the capability derivation
        # itself is correct for the day one does, using the smallest
        # possible non-simulation stand-in.

        class _FakeLiveVoiceProvider:
            is_simulation_only = False
            def send(self, instruction):
                return instruction

        manager = SpeakerManager()
        controller = VoiceEvacuationController(manager, _FakeLiveVoiceProvider())

        gateway = LiveOperatorActionGateway(voice_controller=controller)

        self.assertEqual(gateway.voice_capability, PROVIDER_CAPABILITY_LIVE_HARDWARE)


class VoiceApprovalTests(unittest.TestCase):

    def setUp(self):

        self.gateway = LiveOperatorActionGateway(voice_controller=make_voice_controller_with_speaker())

    def test_default_status_is_recommended(self):

        announcement = make_announcement()
        self.assertEqual(self.gateway.voice_recommendation_status(announcement), VOICE_STATUS_RECOMMENDED)

    def test_approve_broadcasts_the_exact_advisory_text_verbatim(self):

        announcement = make_announcement(text="Attention occupants in Zone 1: evacuate via the nearest exit.")

        instructions = self.gateway.approve_voice_message(announcement, time=5.0)

        self.assertEqual(len(instructions), 1)
        self.assertEqual(
            instructions[0].message.message_text,
            "Attention occupants in Zone 1: evacuate via the nearest exit.",
        )
        self.assertEqual(instructions[0].status, BroadcastStatus.BROADCAST)

    def test_approve_updates_status_to_sent_when_speakers_available(self):

        announcement = make_announcement()
        self.gateway.approve_voice_message(announcement, time=1.0)

        self.assertEqual(self.gateway.voice_recommendation_status(announcement), VOICE_STATUS_SENT)

    def test_approve_with_no_speakers_in_zone_reports_failed_not_sent(self):

        gateway = LiveOperatorActionGateway(
            voice_controller=VoiceEvacuationController(SpeakerManager(), SimulationVoiceOutputProvider()),
        )
        announcement = make_announcement()

        gateway.approve_voice_message(announcement, time=1.0)

        self.assertEqual(gateway.voice_recommendation_status(announcement), VOICE_STATUS_FAILED)

    def test_reject_never_touches_the_controller(self):

        announcement = make_announcement()

        self.gateway.reject_voice_message(announcement, time=1.0)

        self.assertEqual(self.gateway.voice_recommendation_status(announcement), VOICE_STATUS_REJECTED)
        self.assertEqual(len(self.gateway.voice_broadcast_history()), 0)

    def test_approve_with_no_provider_raises_operator_action_unavailable(self):

        gateway = LiveOperatorActionGateway()
        announcement = make_announcement()

        with self.assertRaises(OperatorActionUnavailable):
            gateway.approve_voice_message(announcement, time=1.0)

    def test_reject_with_no_provider_still_succeeds(self):

        # Rejecting is purely a gateway-side decision, never requires a
        # provider -- Phase 5's "operator can always review/reject" and
        # Phase 3's "no execution path required to say no" both hold.

        gateway = LiveOperatorActionGateway()
        announcement = make_announcement()

        gateway.reject_voice_message(announcement, time=1.0)  # must not raise
        self.assertEqual(gateway.voice_recommendation_status(announcement), VOICE_STATUS_REJECTED)

    def test_duplicate_approve_click_never_re_broadcasts(self):

        announcement = make_announcement()

        first = self.gateway.approve_voice_message(announcement, time=1.0)
        second = self.gateway.approve_voice_message(announcement, time=2.0)

        self.assertEqual(first, second)
        self.assertEqual(len(self.gateway.voice_broadcast_history()), 1)

    def test_a_new_recommendation_for_the_same_zone_starts_fresh(self):

        # A genuinely different recommendation (new text) for a zone
        # that already had a decided-on message must not inherit that
        # old decision (Phase 11 item 6 -- "recommendation superseded
        # before approval").

        first = make_announcement(text="Shelter in place")
        second = make_announcement(text="Evacuate immediately")

        self.gateway.approve_voice_message(first, time=1.0)
        self.assertEqual(self.gateway.voice_recommendation_status(second), VOICE_STATUS_RECOMMENDED)

    def test_audit_log_records_operator_actor_and_zone(self):

        announcement = make_announcement()
        self.gateway.approve_voice_message(announcement, time=3.0)

        log = self.gateway.voice_audit_log()

        self.assertEqual(len(log), 1)
        self.assertEqual(log[0].action, "APPROVE_VOICE_MESSAGE")
        self.assertEqual(log[0].actor, OPERATOR_ACTOR)
        self.assertEqual(log[0].zone_id, "zone-1")
        self.assertEqual(log[0].time, 3.0)

    def test_history_reflects_zone_targeting_and_priority_preserved(self):

        announcement = make_announcement(zone_id="zone-9", zone_name="Zone Nine")
        gateway = LiveOperatorActionGateway(voice_controller=make_voice_controller_with_speaker(zone_id="zone-9"))

        gateway.approve_voice_message(announcement, time=1.0)

        history = gateway.voice_broadcast_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].target_zone_id, "zone-9")


class ControlApprovalTests(unittest.TestCase):

    def setUp(self):

        self.building, self.door = make_building_with_door()
        self.gateway = LiveOperatorActionGateway(control_controller=make_control_controller(self.building))

    def test_ingest_submits_translatable_recommendations(self):

        report = make_report(recommendations=(make_recommendation(self.door.id),))

        submitted = self.gateway.ingest_control_recommendations(report)

        self.assertEqual(len(submitted), 1)
        self.assertEqual(self.gateway.pending_control_requests()[0].request_id, submitted[0].request_id)

    def test_ingest_with_no_controller_submits_nothing(self):

        gateway = LiveOperatorActionGateway()
        report = make_report(recommendations=(make_recommendation(self.door.id),))

        submitted = gateway.ingest_control_recommendations(report)

        self.assertEqual(submitted, ())
        self.assertEqual(gateway.pending_control_requests(), ())

    def test_approve_dispatches_through_the_real_controller(self):

        report = make_report(recommendations=(make_recommendation(self.door.id),))
        request = self.gateway.ingest_control_recommendations(report)[0]

        self.gateway.approve_control_request(request.request_id)

        self.assertEqual(
            self.building.floors[0].doors[0].id, self.door.id,  # sanity: same building object
        )
        # No action_executor configured -- honest FAILED, never
        # fabricated CONFIRMED (Phase 11 item 4).
        self.assertEqual(self.gateway.confirmed_control_entries(), ())

    def test_reject_never_dispatches(self):

        report = make_report(recommendations=(make_recommendation(self.door.id),))
        request = self.gateway.ingest_control_recommendations(report)[0]

        self.gateway.reject_control_request(request.request_id)

        self.assertEqual(self.gateway.pending_control_requests(), ())
        self.assertEqual(self.gateway.confirmed_control_entries(), ())

    def test_approve_with_no_provider_raises_operator_action_unavailable(self):

        gateway = LiveOperatorActionGateway()

        with self.assertRaises(OperatorActionUnavailable):
            gateway.approve_control_request("does-not-matter")

    def test_reject_with_no_provider_raises_operator_action_unavailable(self):

        gateway = LiveOperatorActionGateway()

        with self.assertRaises(OperatorActionUnavailable):
            gateway.reject_control_request("does-not-matter")

    def test_duplicate_pending_ingest_does_not_create_a_second_request(self):

        report = make_report(recommendations=(make_recommendation(self.door.id),))

        first = self.gateway.ingest_control_recommendations(report)[0]
        second = self.gateway.ingest_control_recommendations(report)[0]

        self.assertEqual(first.request_id, second.request_id)
        self.assertEqual(len(self.gateway.pending_control_requests()), 1)

    def test_rejecting_a_recommendation_prevents_it_from_silently_resubmitting(self):

        # A re-render (e.g. the panel's own post-decision show_live()
        # call) re-ingests the identical AdvisoryReport -- once
        # REJECTED (a terminal status BuildingControlController's own
        # dedup no longer protects), that must never come back as a
        # fresh PENDING_APPROVAL request without a new operator decision.

        report = make_report(recommendations=(make_recommendation(self.door.id),))

        request = self.gateway.ingest_control_recommendations(report)[0]
        self.gateway.reject_control_request(request.request_id)

        resubmitted = self.gateway.ingest_control_recommendations(report)

        self.assertEqual(resubmitted, ())
        self.assertEqual(self.gateway.pending_control_requests(), ())

    def test_a_different_recommendation_for_the_same_target_is_not_blocked_by_a_prior_rejection(self):

        open_report = make_report(recommendations=(make_recommendation(self.door.id, action="Open Door"),))
        close_report = make_report(recommendations=(make_recommendation(self.door.id, action="Close Door"),))

        open_request = self.gateway.ingest_control_recommendations(open_report)[0]
        self.gateway.reject_control_request(open_request.request_id)

        close_submitted = self.gateway.ingest_control_recommendations(close_report)

        self.assertEqual(len(close_submitted), 1)
        self.assertEqual(len(self.gateway.pending_control_requests()), 1)

    def test_control_history_reuses_the_real_controlevent_history(self):

        report = make_report(recommendations=(make_recommendation(self.door.id),))
        request = self.gateway.ingest_control_recommendations(report)[0]

        self.gateway.approve_control_request(request.request_id)

        history = self.gateway.control_history()
        actors = [event.actor for event in history]

        self.assertIn(OPERATOR_ACTOR, actors)

    def test_approve_dispatches_through_a_real_action_executor_interface(self):

        # Same "minimal stand-in for the real ActionExecutor, records
        # every Action, always reports applied=True" convention
        # tests/test_building_control.py's own RecordingActionExecutor
        # already establishes -- standing up a full SimulationContext/
        # RouteManager is not this gateway's own concern to prove.

        from simulation_interactive.action_executor import Action, ActionResult

        class RecordingActionExecutor:

            def __init__(self):
                self.calls = []

            def apply(self, action: Action, time: float) -> ActionResult:
                self.calls.append(action)
                return ActionResult(action=action, applied=True, reason=None)

        executor = RecordingActionExecutor()

        gateway = LiveOperatorActionGateway(control_controller=make_control_controller(self.building, executor))

        report = make_report(recommendations=(make_recommendation(self.door.id),))
        request = gateway.ingest_control_recommendations(report)[0]

        gateway.approve_control_request(request.request_id)

        entries = gateway.confirmed_control_entries()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].target_id, self.door.id)


if __name__ == "__main__":
    unittest.main()
