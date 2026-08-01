import unittest

from voice_evacuation.models import VoiceMessage

from execution_layer.adapters import voice_adapter
from execution_layer.models import ExecutionCategory, ExecutionStatus, RecommendationIdProvenance

from tests.execution_layer_fixtures import make_voice_controller


class VoiceAdapterTests(unittest.TestCase):

    def test_broadcast_produces_confirmed_execution_request(self):

        controller = make_voice_controller()
        message = VoiceMessage(target_zone_ids=("z1",), message_text="Evacuate now")
        controller.broadcast(message, 1.0)

        requests = voice_adapter.build_execution_requests(controller)

        self.assertEqual(len(requests), 1)
        request = requests[0]
        self.assertEqual(request.category, ExecutionCategory.VOICE_EVACUATION)
        self.assertEqual(request.status, ExecutionStatus.CONFIRMED)
        self.assertEqual(request.dispatched_at, 1.0)
        self.assertEqual(request.completed_at, 1.0)
        self.assertIsNone(request.created_at)
        self.assertIsNone(request.approved_at)

    def test_zone_with_no_speaker_reports_failed(self):

        controller = make_voice_controller()
        message = VoiceMessage(target_zone_ids=("no-speaker-zone",), message_text="Evacuate now")
        controller.broadcast(message, 1.0)

        requests = voice_adapter.build_execution_requests(controller)

        self.assertEqual(requests[0].status, ExecutionStatus.FAILED)

    def test_source_recommendation_id_tagged_as_advisory_system(self):

        controller = make_voice_controller()
        message = VoiceMessage(target_zone_ids=("z1",), message_text="Evacuate now", source_recommendation_id="rec-abc")
        controller.broadcast(message, 1.0)

        requests = voice_adapter.build_execution_requests(controller)

        self.assertEqual(requests[0].originating_recommendation_id, "rec-abc")
        self.assertEqual(requests[0].recommendation_id_provenance, RecommendationIdProvenance.ADVISORY_SYSTEM)

    def test_no_recommendation_id_is_unavailable(self):

        controller = make_voice_controller()
        message = VoiceMessage(target_zone_ids=("z1",), message_text="Evacuate now")
        controller.broadcast(message, 1.0)

        requests = voice_adapter.build_execution_requests(controller)

        self.assertEqual(requests[0].recommendation_id_provenance, RecommendationIdProvenance.UNAVAILABLE)

    def test_none_controller_produces_no_requests(self):

        self.assertEqual(voice_adapter.build_execution_requests(None), ())


if __name__ == "__main__":
    unittest.main()
