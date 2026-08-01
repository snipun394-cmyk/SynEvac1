import unittest
from unittest.mock import patch

from voice_evacuation.models import VoiceMessage

from building_control.requests import ControlRequest
from building_control.types import ControlAction, ControlSystemType, RequestSource

from execution_layer.layer import ExecutionLayer
from execution_layer.models import ExecutionCategory

from tests.execution_layer_fixtures import make_control_controller, make_voice_controller, make_warden_controller


class ExecutionLayerTests(unittest.TestCase):

    def test_compute_aggregates_across_categories(self):

        voice_controller = make_voice_controller()
        voice_controller.broadcast(VoiceMessage(target_zone_ids=("z1",), message_text="Evacuate"), 1.0)

        control_controller = make_control_controller()
        control_request = ControlRequest(
            system_type=ControlSystemType.DOOR, target_id="door-1", requested_action=ControlAction.CLOSE,
            reason="hazard", source=RequestSource.OPERATOR,
        )
        control_controller.submit(control_request)

        layer = ExecutionLayer(voice_controller=voice_controller, control_controller=control_controller)
        execution_set = layer.compute(1.0)

        categories = {r.category for r in execution_set.requests}
        self.assertEqual(categories, {ExecutionCategory.VOICE_EVACUATION, ExecutionCategory.BUILDING_CONTROL})

    def test_compute_always_returns_an_execution_set_never_none(self):

        layer = ExecutionLayer()

        execution_set = layer.compute(1.0)

        self.assertIsNotNone(execution_set)
        self.assertEqual(execution_set.requests, ())

    def test_a_raising_adapter_never_blanks_the_others(self):

        warden_controller = make_warden_controller()

        voice_controller = make_voice_controller()
        voice_controller.broadcast(VoiceMessage(target_zone_ids=("z1",), message_text="Evacuate"), 1.0)

        layer = ExecutionLayer(voice_controller=voice_controller, warden_controller=warden_controller)

        with patch("execution_layer.layer.warden_adapter.build_execution_requests", side_effect=RuntimeError("boom")):
            execution_set = layer.compute(1.0)

        categories = {r.category for r in execution_set.requests}
        self.assertIn(ExecutionCategory.VOICE_EVACUATION, categories)

    def test_latest_property_reflects_most_recent_compute(self):

        layer = ExecutionLayer()

        self.assertIsNone(layer.latest)

        execution_set = layer.compute(1.0)

        self.assertIs(layer.latest, execution_set)


if __name__ == "__main__":
    unittest.main()
