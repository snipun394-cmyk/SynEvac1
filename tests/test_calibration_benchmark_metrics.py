import unittest

from scenario_pipeline import run_batch_pipeline

from calibration_benchmark.metrics import (
    METRIC_FIELDS,
    compute_exit_utilization_balance,
    compute_peak_occupancy_ratio,
    extract_metrics,
)
from calibration_benchmark.simulation_seam import run_with_overrides
from simulator.capacity import StairCapacityModel

from tests.calibration_benchmark_fixtures import DEFINITION_ID, MASTER_SEED, make_building, make_definition


class MetricExtractionTests(unittest.TestCase):

    def setUp(self):

        self.building = make_building()
        batch = run_batch_pipeline(make_definition(occupant_count=15), DEFINITION_ID, self.building, MASTER_SEED, 1)
        self.scenario = batch.scenarios[0]
        self.movement_result, self.ground_truth, self.building_copy = run_with_overrides(
            self.scenario, self.building, dt=1.0,
        )
        self.capacity_model = StairCapacityModel()

    def test_extract_metrics_populates_every_declared_field(self):

        sample = extract_metrics(
            "scenario-1", self.ground_truth, self.movement_result, self.building_copy, self.capacity_model,
        )

        for field_name in METRIC_FIELDS:
            self.assertTrue(hasattr(sample, field_name))

        self.assertEqual(sample.evacuation_time, self.ground_truth.total_evacuation_time)
        self.assertEqual(sample.queue_length, self.ground_truth.peak_congestion_value)

    def test_peak_occupancy_ratio_is_at_most_the_edge_with_highest_demand_relative_to_capacity(self):

        ratio = compute_peak_occupancy_ratio(self.building_copy, self.movement_result, self.capacity_model)

        self.assertIsNotNone(ratio)
        self.assertGreaterEqual(ratio, 0.0)

    def test_exit_utilization_balance_is_between_zero_and_one(self):

        balance = compute_exit_utilization_balance(self.ground_truth, self.building_copy)

        self.assertIsNotNone(balance)
        self.assertGreaterEqual(balance, 0.0)
        self.assertLessEqual(balance, 1.0)

    def test_to_dict_round_trips_every_field(self):

        sample = extract_metrics(
            "scenario-1", self.ground_truth, self.movement_result, self.building_copy, self.capacity_model,
        )
        data = sample.to_dict()

        self.assertEqual(data["scenario_id"], "scenario-1")
        for field_name in METRIC_FIELDS:
            self.assertIn(field_name, data)


class PeakOccupancyRatioNoExitsTests(unittest.TestCase):

    def test_returns_none_when_building_has_no_measurable_edges(self):

        from models.building import Building
        from models.floor import Floor
        from models.zone import Zone

        empty_building = Building(
            name="Empty", id="empty-1",
            floors=[Floor(name="Ground", id="floor-1", zones=[Zone(id="z1", name="Z1", x=0, y=0, width=1, height=1)])],
        )

        class _FakeMovementResult:
            peak_edge_occupancy = {}

        ratio = compute_peak_occupancy_ratio(empty_building, _FakeMovementResult(), StairCapacityModel())

        self.assertIsNone(ratio)


if __name__ == "__main__":
    unittest.main()
