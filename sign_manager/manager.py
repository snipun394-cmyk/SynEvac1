from typing import Dict, Optional, Tuple

from sign_manager.status import SignStatus


class SignManager:

    # The single place Dynamic Evacuation Sign assets are managed --
    # mirrors speaker_manager.manager.SpeakerManager's own design
    # principles point for point (discover/register/remove, lookup by
    # id/floor/zone, enable/disable, status) WITHOUT importing it --
    # independent sibling packages that happen to share a shape, not
    # one built on another (same discipline speaker_manager itself
    # holds toward camera_manager/sensor_manager).
    #
    # This class performs no routing/planning reasoning and no direction
    # geometry of any kind -- it is pure bookkeeping over Sign assets,
    # the same "never decides what should be shown or when" discipline
    # SpeakerManager already holds for itself. signs_in_zone() is the
    # one lookup dynamic_signage.planner.DynamicSignagePlanner actually
    # depends on.

    def __init__(self):

        self._signs: Dict[str, object] = {}

    # =====================================================
    # Discovery / Registration
    # =====================================================

    def discover_signs(self, building) -> Tuple[object, ...]:

        self._signs = {}

        if building is None:
            return ()

        for floor in building.ordered_floors():

            for sign in floor.signs:
                self.register_sign(sign)

        return self.all_signs()

    # =====================================================

    def register_sign(self, sign) -> None:

        self._signs[sign.id] = sign

    # =====================================================

    def remove_sign(self, sign_id: str) -> None:

        self._signs.pop(sign_id, None)

    # =====================================================

    def all_signs(self) -> Tuple[object, ...]:

        return tuple(self._signs.values())

    # =====================================================
    # Lookup
    # =====================================================

    def get_sign(self, sign_id: str) -> Optional[object]:

        return self._signs.get(sign_id)

    # =====================================================

    def signs_on_floor(self, floor_id: str) -> Tuple[object, ...]:

        return tuple(sign for sign in self._signs.values() if sign.floor_id == floor_id)

    # =====================================================

    def signs_in_zone(self, zone_id: str) -> Tuple[object, ...]:

        return tuple(sign for sign in self._signs.values() if zone_id in sign.zone_ids)

    # =====================================================

    def active_signs(self) -> Tuple[object, ...]:

        return tuple(sign for sign in self._signs.values() if sign.active)

    # =====================================================
    # Enable / Disable
    # =====================================================

    def enable_sign(self, sign_id: str) -> None:

        self._require_sign(sign_id).active = True

    # =====================================================

    def disable_sign(self, sign_id: str) -> None:

        self._require_sign(sign_id).active = False

    # =====================================================
    # Status
    # =====================================================

    def sign_status(self, sign_id: str) -> SignStatus:

        sign = self._require_sign(sign_id)

        return SignStatus(
            sign_id=sign.id, name=sign.name, floor_id=sign.floor_id, zone_ids=sign.zone_ids,
            active=sign.active, orientation=sign.orientation, supported_indications=sign.supported_indications,
        )

    # =====================================================

    def all_statuses(self) -> Tuple[SignStatus, ...]:

        return tuple(self.sign_status(sign.id) for sign in self.all_signs())

    # =====================================================
    # Internals
    # =====================================================

    def _require_sign(self, sign_id: str):

        sign = self._signs.get(sign_id)

        if sign is None:
            raise KeyError(f"SignManager has no sign registered with id {sign_id!r}.")

        return sign
