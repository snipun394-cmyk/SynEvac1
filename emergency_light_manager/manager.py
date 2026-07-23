from typing import Dict, Optional, Tuple

from emergency_light_manager.status import EmergencyLightStatus


class EmergencyLightManager:

    # The single place Emergency Light assets are managed -- mirrors
    # sign_manager.manager.SignManager's own design principles point for
    # point (discover/register/lookup/enable-disable/status) WITHOUT
    # importing it (independent sibling packages that happen to share a
    # shape, not one built on another -- the same discipline every
    # existing manager already holds toward every other one).
    #
    # A NEW, small manager is justified here (Phase 9's own explicit
    # "avoid manager proliferation without reason" instruction,
    # weighed deliberately): EmergencyLight is not a fire-protection
    # sensor (no DetectorState/alarm concept -- SensorManager's whole
    # purpose is aggregating ALARM/FAULT conditions for FACP, which this
    # asset has none of), not a Camera, not a Speaker (no broadcast
    # concept), and not a Sign (no route-indication concept) -- none of
    # the four existing managers is a semantically honest fit, so a
    # fifth, equally small manager is the correct choice rather than
    # forcing this asset into an unrelated one.
    #
    # This class performs no lux/photometric computation and no route-
    # safety judgment of any kind -- it is pure bookkeeping over
    # EmergencyLight assets, the same "never decides what should happen,
    # only reports what already exists" discipline every sibling manager
    # already holds for itself.

    def __init__(self):

        self._lights: Dict[str, object] = {}

    # =====================================================
    # Discovery / Registration
    # =====================================================

    def discover_lights(self, building) -> Tuple[object, ...]:

        self._lights = {}

        if building is None:
            return ()

        for floor in building.ordered_floors():

            for light in floor.emergency_lights:
                self.register_light(light)

        return self.all_lights()

    # =====================================================

    def register_light(self, light) -> None:

        self._lights[light.id] = light

    # =====================================================

    def remove_light(self, light_id: str) -> None:

        self._lights.pop(light_id, None)

    # =====================================================

    def all_lights(self) -> Tuple[object, ...]:

        return tuple(self._lights.values())

    # =====================================================
    # Lookup
    # =====================================================

    def get_light(self, light_id: str) -> Optional[object]:

        return self._lights.get(light_id)

    # =====================================================

    def lights_on_floor(self, floor_id: str) -> Tuple[object, ...]:

        return tuple(light for light in self._lights.values() if light.floor_id == floor_id)

    # =====================================================

    def lights_in_zone(self, zone_id: str) -> Tuple[object, ...]:

        return tuple(light for light in self._lights.values() if zone_id in light.zone_ids)

    # =====================================================

    def available_lights(self) -> Tuple[object, ...]:

        from models.emergency_light import EmergencyLightAvailability

        return tuple(
            light for light in self._lights.values()
            if light.compute_availability() == EmergencyLightAvailability.AVAILABLE
        )

    # =====================================================
    # Enable / Disable
    # =====================================================

    def enable_light(self, light_id: str) -> None:

        self._require_light(light_id).active = True

    # =====================================================

    def disable_light(self, light_id: str) -> None:

        self._require_light(light_id).active = False

    # =====================================================
    # Status
    # =====================================================

    def light_status(self, light_id: str) -> EmergencyLightStatus:

        light = self._require_light(light_id)

        return EmergencyLightStatus(
            light_id=light.id, name=light.name, floor_id=light.floor_id, zone_ids=light.zone_ids,
            active=light.active, health_status=light.health_status, light_type=light.light_type,
            availability=light.compute_availability(),
        )

    # =====================================================

    def all_statuses(self) -> Tuple[EmergencyLightStatus, ...]:

        return tuple(self.light_status(light.id) for light in self.all_lights())

    # =====================================================
    # Internals
    # =====================================================

    def _require_light(self, light_id: str):

        light = self._lights.get(light_id)

        if light is None:
            raise KeyError(f"EmergencyLightManager has no light registered with id {light_id!r}.")

        return light
