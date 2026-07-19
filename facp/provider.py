from typing import Tuple

from facp.models import PanelEvent


class FACPEventProvider:

    # The one seam a future real Fire Alarm Control Panel integration
    # plugs into -- mirrors HazardProvider/PerceptionProvider/
    # HumanObservationProvider's own established "one interface, no
    # concrete implementation yet, raise NotImplementedError" pattern
    # (Phase 8's own explicit instruction). A future vendor adapter
    # (Modbus/BACnet/MQTT/OPC-UA/serial/vendor SDK -- none implemented
    # here, none imported here, none imported anywhere in this package)
    # satisfies this exact contract by translating whatever its
    # protocol delivers into the SAME canonical facp.models.PanelEvent
    # shape SimulatedFACP already produces:
    #
    #   Real Panel -> Vendor Protocol -> Live FACP Adapter -> events_at()
    #
    # The rest of SynEvac (BuildingState, Perception, Advisory, Command
    # Center) consumes PanelEvent/FACPSnapshot either way and never
    # needs to know panel manufacturer, protocol, IP address, serial
    # port, or vendor SDK -- none of those concepts exist anywhere in
    # this file or this package.
    #
    # Deliberately does not extend SimulatedFACP or share its class
    # hierarchy: SimulatedFACP's public surface (evaluate/acknowledge/
    # silence/reset) is a stateful COORDINATOR that consumes structured
    # DetectorConditionReport input, whereas a Live adapter's job is
    # narrower and different in shape -- "hand me whatever new canonical
    # events exist since last time", closer to
    # live_system.sensor_registry.FireAlarmControlPanelSensor's own
    # raw-reading role one layer below. A future integration composes
    # both: a Live FACP Adapter implementing this interface feeds
    # already-translated PanelEvents to whatever downstream consumer
    # replaces SimulatedFACP's role in Live mode, exactly the same
    # "only the source changes, the downstream shape stays identical"
    # principle already established for HazardProvider/OccupancyProvider/
    # PerceptionProvider (Simulation/Replay/Live all implement the same
    # provider interface; nothing downstream is coupled to which one is
    # wired up).

    def events_at(self, time: float) -> Tuple[PanelEvent, ...]:

        raise NotImplementedError
