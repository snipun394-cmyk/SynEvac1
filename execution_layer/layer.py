from typing import Optional

from execution_layer.adapters import building_control_adapter, signage_adapter, voice_adapter, warden_adapter
from execution_layer.models import ExecutionSet


# =====================================================
# ExecutionLayer -- the ONE public facade this package exposes. It is
# an ORCHESTRATION/COORDINATING layer, not a replacement execution
# engine: voice_evacuation.VoiceEvacuationController, building_control.
# BuildingControlController, dynamic_signage.DynamicSignageController,
# and warden_notification.WardenNotificationController remain the sole
# execution authority. compute() never calls a provider itself -- it
# only reads what those controllers already recorded and normalizes it
# into one unified, cross-category ExecutionSet.
# =====================================================


class ExecutionLayer:

    def __init__(
        self, *, voice_controller=None, control_controller=None, signage_controller=None, warden_controller=None,
    ):

        self._voice_controller = voice_controller
        self._control_controller = control_controller
        self._signage_controller = signage_controller
        self._warden_controller = warden_controller

        self._latest: Optional[ExecutionSet] = None

    # =====================================================

    @property
    def latest(self) -> Optional[ExecutionSet]:

        return self._latest

    # =====================================================

    def compute(self, time: float) -> ExecutionSet:

        requests = []

        adapter_calls = (
            lambda: voice_adapter.build_execution_requests(self._voice_controller),
            lambda: building_control_adapter.build_execution_requests(self._control_controller),
            lambda: signage_adapter.build_execution_requests(self._signage_controller),
            lambda: warden_adapter.build_execution_requests(self._warden_controller),
        )

        for call in adapter_calls:

            try:
                requests.extend(call())
            except Exception:  # noqa: BLE001 -- one category's bug must never blank the other three
                continue

        self._latest = ExecutionSet(timestamp=time, requests=tuple(requests))

        return self._latest
