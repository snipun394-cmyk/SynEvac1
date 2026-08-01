import sys
import unittest

from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

from designer.widgets.recommendation_panel import RecommendationPanel

from recommendation_layer.models import Recommendation, RecommendationPriority, RecommendationSet, RecommendationType, TriggerCondition


def make_recommendation(recommendation_id, priority=RecommendationPriority.MEDIUM, type_=RecommendationType.OCCUPANT_ROUTING, zone_id="zone-1"):

    return Recommendation(
        recommendation_id=recommendation_id, type=type_, priority=priority,
        trigger_condition=TriggerCondition.ZONE_EXIT_RECOMMENDED, affected_zones=(zone_id,),
        explanation="explanation", technical_reason="reason", recommended_action="action",
        primary_source="evacuation_recommendation", updated_at=1.0,
    )


class RecommendationPanelTests(unittest.TestCase):

    def test_refresh_populates_rows(self):

        panel = RecommendationPanel()

        recommendation_set = RecommendationSet(timestamp=1.0, recommendations=(
            make_recommendation("r1", priority=RecommendationPriority.CRITICAL),
            make_recommendation("r2", priority=RecommendationPriority.LOW),
        ))

        panel.refresh(recommendation_set)

        self.assertEqual(panel.recommendation_table.rowCount(), 2)

    def test_refresh_with_none_produces_empty_table(self):

        panel = RecommendationPanel()

        panel.refresh(None)

        self.assertEqual(panel.recommendation_table.rowCount(), 0)

    def test_priority_filter_narrows_rows(self):

        panel = RecommendationPanel()

        recommendation_set = RecommendationSet(timestamp=1.0, recommendations=(
            make_recommendation("r1", priority=RecommendationPriority.CRITICAL),
            make_recommendation("r2", priority=RecommendationPriority.LOW),
        ))

        panel.refresh(recommendation_set)

        index = panel.priority_filter.findData(RecommendationPriority.CRITICAL)
        panel.priority_filter.setCurrentIndex(index)

        self.assertEqual(panel.recommendation_table.rowCount(), 1)

    def test_selecting_a_row_invokes_callback_with_the_right_recommendation(self):

        panel = RecommendationPanel()

        selected = []
        panel.on_recommendation_selected = lambda recommendation: selected.append(recommendation)

        recommendation_set = RecommendationSet(timestamp=1.0, recommendations=(
            make_recommendation("r1"),
        ))

        panel.refresh(recommendation_set)

        panel.recommendation_table.setCurrentCell(0, 0)

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].recommendation_id, "r1")


if __name__ == "__main__":
    unittest.main()
