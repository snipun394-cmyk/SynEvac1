from typing import Dict, Optional, Tuple

from speaker_manager.status import SpeakerStatus


class SpeakerManager:

    # The single place voice-evacuation Speaker assets are managed --
    # mirrors camera_manager.manager.CameraManager/sensor_manager.
    # manager.SensorManager's own design principles point for point
    # (discover/register/remove, lookup by id/floor/zone, enable/
    # disable, status) WITHOUT importing either (enforced by this
    # package's own dependency-direction test, see
    # tests.test_speaker_manager.SpeakerManagerPackageDependencyDirectionTests)
    # -- three independent packages that happen to share a shape, not
    # one built on another.
    #
    # This class performs no recommendation reasoning, no message
    # routing decision, and no broadcast of any kind -- it is pure
    # bookkeeping and routing over Speaker assets, exactly the
    # "never performs computer vision"/"never computes hazard/physics"
    # discipline CameraManager/SensorManager already hold for
    # themselves, restated here as "never decides what should be
    # announced or when." speakers_in_zone() below is the one lookup
    # voice_evacuation.controller.VoiceEvacuationController actually
    # depends on -- "which speakers serve this zone" -- and it is a
    # plain membership query over Speaker.zone_ids, nothing more.

    def __init__(self):

        self._speakers: Dict[str, object] = {}

    # =====================================================
    # Discovery / Registration
    # =====================================================

    def discover_speakers(self, building) -> Tuple[object, ...]:

        # Full, idempotent rescan -- same "rebuild rather than
        # incrementally patch" convention CameraManager.discover_
        # cameras()/SensorManager.discover_sensors() already establish.

        self._speakers = {}

        if building is None:
            return ()

        for floor in building.ordered_floors():

            for speaker in floor.speakers:
                self.register_speaker(speaker)

        return self.all_speakers()

    # =====================================================

    def register_speaker(self, speaker) -> None:

        self._speakers[speaker.id] = speaker

    # =====================================================

    def remove_speaker(self, speaker_id: str) -> None:

        self._speakers.pop(speaker_id, None)

    # =====================================================

    def all_speakers(self) -> Tuple[object, ...]:

        return tuple(self._speakers.values())

    # =====================================================
    # Lookup
    # =====================================================

    def get_speaker(self, speaker_id: str) -> Optional[object]:

        return self._speakers.get(speaker_id)

    # =====================================================

    def speakers_on_floor(self, floor_id: str) -> Tuple[object, ...]:

        return tuple(
            speaker for speaker in self._speakers.values() if speaker.floor_id == floor_id
        )

    # =====================================================

    def speakers_in_zone(self, zone_id: str) -> Tuple[object, ...]:

        # "Which speakers serve a requested zone" -- Phase 3's own
        # required lookup. A single speaker may appear for more than
        # one zone_id (Speaker.zone_ids, inherited from EngineeringAsset,
        # already supports multiple assigned zones); multiple speakers
        # may serve the same zone -- both are plain consequences of this
        # being an ordinary membership test, never a geometric/acoustic
        # computation.

        return tuple(
            speaker for speaker in self._speakers.values() if zone_id in speaker.zone_ids
        )

    # =====================================================

    def active_speakers_in_zone(self, zone_id: str) -> Tuple[object, ...]:

        # The one filtered variant voice_evacuation.controller.
        # VoiceEvacuationController actually needs -- an inactive/
        # disabled speaker must never be selected for a broadcast (Phase
        # 13's own "disabled speaker is not selected" requirement).
        # Kept as its own method (rather than making every caller filter
        # speakers_in_zone() itself) since "currently deliverable" is a
        # distinct, reusable question from "assigned to this zone at
        # all."

        return tuple(speaker for speaker in self.speakers_in_zone(zone_id) if speaker.active)

    # =====================================================
    # Enable / Disable
    # =====================================================

    def enable_speaker(self, speaker_id: str) -> None:

        self._require_speaker(speaker_id).active = True

    # =====================================================

    def disable_speaker(self, speaker_id: str) -> None:

        self._require_speaker(speaker_id).active = False

    # =====================================================
    # Status
    # =====================================================

    def speaker_status(self, speaker_id: str) -> SpeakerStatus:

        speaker = self._require_speaker(speaker_id)

        return SpeakerStatus(
            speaker_id=speaker.id,
            name=speaker.name,
            floor_id=speaker.floor_id,
            zone_ids=speaker.zone_ids,
            active=speaker.active,
            mode=speaker.mode,
            health_status=speaker.health_status,
        )

    # =====================================================

    def all_statuses(self) -> Tuple[SpeakerStatus, ...]:

        return tuple(self.speaker_status(speaker.id) for speaker in self.all_speakers())

    # =====================================================
    # Internals
    # =====================================================

    def _require_speaker(self, speaker_id: str):

        speaker = self._speakers.get(speaker_id)

        if speaker is None:

            raise KeyError(
                f"SpeakerManager has no speaker registered with id {speaker_id!r}."
            )

        return speaker
