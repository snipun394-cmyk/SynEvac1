from typing import Optional

from advisory_system.advisory_engine import (
    build_building_recommendations, build_civilian_announcements,
    build_commander_dashboard, build_firefighter_intelligence,
)
from advisory_system.recommendation_history import RecommendationHistory
from advisory_system.recommendation_models import AdvisoryInputs, AdvisoryReport


# =====================================================
# Phase 1 -- Advisory Core. AdvisoryOrchestrator is the one place that
# combines every existing platform output into one AdvisoryReport --
# it contains no recommendation logic of its own (every rule lives in
# advisory_engine.py/confidence_engine.py/explanation_engine.py); its
# only job is calling those four builders in the fixed order below and
# threading the resulting civilian announcements into a
# RecommendationHistory so Phase 7's change-log is maintained
# automatically, once per report, rather than by every caller
# separately.
# =====================================================


class AdvisoryOrchestrator:

    def __init__(self, history: Optional[RecommendationHistory] = None):

        # A caller running one long-lived incident (successive snapshots
        # of the same scenario over simulation time) should construct
        # one AdvisoryOrchestrator and call generate_report() repeatedly
        # so RecommendationHistory accumulates across calls; a caller
        # analyzing many independent scenarios (e.g. a campaign) should
        # construct a fresh one per scenario -- history is never shared
        # across scenario_ids implicitly.
        self.history = history if history is not None else RecommendationHistory()

    # =====================================================

    def generate_report(self, inputs: AdvisoryInputs) -> AdvisoryReport:

        civilian_announcements = build_civilian_announcements(inputs)
        firefighter_intelligence = build_firefighter_intelligence(inputs)
        building_recommendations = build_building_recommendations(inputs)
        commander_dashboard = build_commander_dashboard(
            inputs, civilian_announcements, building_recommendations,
        )

        changes = []

        for announcement in civilian_announcements:

            change = self.history.record(
                timestamp=inputs.simulation_time,
                zone_id=announcement.zone_id,
                recommendation=announcement.announcement,
                confidence=announcement.confidence,
                reason_for_change=announcement.reason,
            )

            if change is not None:
                changes.append(change)

        return AdvisoryReport(
            scenario_id=inputs.scenario_id,
            simulation_time=inputs.simulation_time,
            civilian_announcements=civilian_announcements,
            firefighter_intelligence=firefighter_intelligence,
            building_recommendations=building_recommendations,
            commander_dashboard=commander_dashboard,
            recommendation_history=tuple(changes),
        )


# =====================================================


def generate_advisory_report(
    inputs: AdvisoryInputs, history: Optional[RecommendationHistory] = None,
) -> AdvisoryReport:

    # Convenience function for a caller that only wants one, one-shot
    # report and does not care about carrying a RecommendationHistory
    # across calls -- equivalent to
    # AdvisoryOrchestrator(history).generate_report(inputs).

    return AdvisoryOrchestrator(history).generate_report(inputs)
