from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from dynamic_signage.models import SignageInstruction


@dataclass(frozen=True)
class SignageApplyResult:

    # What actually happened when a Provider tried to apply one
    # approved SignageInstruction to a physical (or simulated) sign.
    # Mirrors building_control.requests.ControlResult/voice_evacuation.
    # models.BroadcastStatus's own "honest outcome, never assume success"
    # role -- DynamicSignageController must never assume "dispatched"
    # means "displayed".

    confirmed: bool
    message: str = ""


class DynamicSignageProvider:

    # The one seam a future real digital-signage integration plugs
    # into (Phase 14/15) -- mirrors voice_evacuation.provider.
    # VoiceOutputProvider/building_control.providers.BuildingControlProvider's
    # own established "one interface, no concrete implementation yet,
    # raise NotImplementedError" pattern. A future Live vendor adapter
    # (Modbus/BACnet/MQTT/proprietary sign-controller SDK -- none
    # implemented here, none imported here, none imported anywhere in
    # this package) satisfies this exact contract:
    #
    #   SignageInstruction -> LiveDynamicSignageProvider -> Vendor Adapter
    #       -> Physical Digital Sign Hardware
    #
    # DynamicSignageController never imports vendor-specific libraries
    # and never needs to know sign manufacturer, protocol, IP address,
    # or vendor SDK -- it only ever calls apply(instruction) against
    # whichever DynamicSignageProvider it was constructed with.

    is_simulation_only: bool = False

    def apply(self, instruction: SignageInstruction) -> SignageApplyResult:

        raise NotImplementedError


class SimulationDynamicSignageProvider(DynamicSignageProvider):

    # Simulation mode's own output provider (Phase 14) -- records the
    # current indication per sign, never anything physical. Deliberately
    # does nothing beyond bookkeeping: no display rendering, no
    # hardware I/O, no network access of any kind. Gives a
    # deterministic, inspectable record of "which sign currently shows
    # what, and every applied instruction over time" for validation,
    # research, and Command Center display.

    is_simulation_only = True

    def __init__(self):

        self._current: Dict[str, SignageInstruction] = {}
        self._applied: list = []

    # =====================================================

    def apply(self, instruction: SignageInstruction) -> SignageApplyResult:

        self._current[instruction.sign_id] = instruction
        self._applied.append(instruction)

        return SignageApplyResult(confirmed=True, message="Applied in simulation.")

    # =====================================================

    def current_indication(self, sign_id: str) -> Optional[SignageInstruction]:

        return self._current.get(sign_id)

    # =====================================================

    def all_current(self) -> Dict[str, SignageInstruction]:

        return dict(self._current)

    # =====================================================

    def applied_instructions(self) -> Tuple[SignageInstruction, ...]:

        return tuple(self._applied)
