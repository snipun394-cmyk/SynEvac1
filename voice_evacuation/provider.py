from typing import List, Tuple

from voice_evacuation.models import BroadcastInstruction


class VoiceOutputProvider:

    # The one seam a future real voice-evacuation/PA integration plugs
    # into (Phase 9) -- mirrors facp.provider.FACPEventProvider/
    # HazardProvider/PerceptionProvider's own established "one
    # interface, no concrete implementation yet, raise NotImplementedError"
    # pattern. A future Live vendor adapter (Modbus/BACnet/MQTT/SIP/
    # VoIP/Dante/ONVIF/serial/proprietary amplifier SDK -- none
    # implemented here, none imported here, none imported anywhere in
    # this package) satisfies this exact contract:
    #
    #   BroadcastInstruction -> LiveVoiceOutputProvider -> Vendor Adapter
    #       -> Physical Voice Alarm / PA System
    #
    # VoiceEvacuationController (controller.py) never imports vendor-
    # specific libraries and never needs to know panel manufacturer,
    # protocol, IP address, serial port, or vendor SDK -- it only ever
    # calls send(instruction) against whichever VoiceOutputProvider it
    # was constructed with.
    #
    # send() returns the instruction the provider actually delivered
    # (or the same instruction unchanged, for a provider with nothing to
    # add) -- a Live provider is free to report delivery failure by
    # returning an instruction with a different BroadcastStatus than it
    # was given; VoiceEvacuationController must never assume "sent" means
    # "successfully played" (Phase 11's own explicit Command Center
    # requirement, applying equally here).
    #
    # is_simulation_only mirrors building_control.providers.
    # BuildingControlProvider's own established flag exactly -- False
    # for every provider except an explicitly simulation-only one, so a
    # Live Operator Action Gateway (command_center/live_operator_action_
    # gateway.py) can honestly report NO_PROVIDER/SIMULATION/
    # LIVE_HARDWARE capability without importing, inspecting, or
    # guessing anything about a specific provider implementation.

    is_simulation_only: bool = False

    def send(self, instruction: BroadcastInstruction) -> BroadcastInstruction:

        raise NotImplementedError


class SimulationVoiceOutputProvider(VoiceOutputProvider):

    # Simulation mode's own output provider (Phase 8) -- records, never
    # plays. Deliberately does nothing beyond bookkeeping: no audio
    # engine, no text-to-speech, no sound of any kind. Gives a
    # deterministic, inspectable record of "which message was broadcast,
    # to which zone(s), through which speakers, when, with what status"
    # -- exactly Phase 8's own required example shape -- for validation,
    # research, and Command Center playback.

    is_simulation_only = True

    def __init__(self):

        self._sent: List[BroadcastInstruction] = []

    # =====================================================

    def send(self, instruction: BroadcastInstruction) -> BroadcastInstruction:

        self._sent.append(instruction)

        return instruction

    # =====================================================

    def sent_instructions(self) -> Tuple[BroadcastInstruction, ...]:

        return tuple(self._sent)
