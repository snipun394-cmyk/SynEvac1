from evacuation_guidance.engine import EvacuationGuidanceEngine
from evacuation_guidance.models import GuidanceConfig

from evacuation_recommendation.models import EvacuationRecommendationSnapshot, RecommendationStatus, ZoneEvacuationRecommendation

from tests.trajectory_intelligence_fixtures import make_trajectory_building, make_trajectory_graph


# =====================================================
# Live Evacuation Guidance & Zoned Message Planning milestone -- reuses
# the SAME shared building topology tests.trajectory_intelligence_fixtures/
# tests.evacuation_recommendation_fixtures already established (z1/
# Lobby/EXIT-1 -- DOOR-1 -- z2/Hall -- DOOR-2 -- z4/Annex/EXIT-2, plus
# z3/Room2F on floor f2 via STAIR-1), so this milestone's own tests
# reason about identical route distances/topology.
# =====================================================


def make_engine(building=None, graph=None, config=None):

    building = building if building is not None else make_trajectory_building()
    graph = graph if graph is not None else make_trajectory_graph(building)
    config = config if config is not None else GuidanceConfig()

    return EvacuationGuidanceEngine(building, graph, config=config)


def make_recommendation_snapshot(zone_id, floor_id, exit_id, timestamp=0.0, status=RecommendationStatus.RECOMMENDED, confidence=0.8):

    recommendation = ZoneEvacuationRecommendation(
        zone_id=zone_id, floor_id=floor_id, status=status, recommended_exit_id=exit_id,
        ranked_exit_ids=(exit_id,) if exit_id else (), confidence=confidence, occupant_count=1, timestamp=timestamp,
    )

    return EvacuationRecommendationSnapshot(timestamp=timestamp, zones={zone_id: recommendation}, safe_exit_ids=(exit_id,) if exit_id else ())


class FakeSpeaker:

    def __init__(self, speaker_id):
        self.id = speaker_id


class FakeSpeakerManager:

    def __init__(self, speakers_by_zone=None):
        self._speakers_by_zone = speakers_by_zone or {}

    def active_speakers_in_zone(self, zone_id):
        return tuple(FakeSpeaker(sid) for sid in self._speakers_by_zone.get(zone_id, ()))
