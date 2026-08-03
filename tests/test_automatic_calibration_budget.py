import unittest

from automatic_calibration.budget import AutoCalibrationBudget


class AutoCalibrationBudgetTests(unittest.TestCase):

    def test_max_evaluations_is_stored(self):

        self.assertEqual(AutoCalibrationBudget(max_evaluations=10).max_evaluations, 10)

    def test_zero_or_negative_max_evaluations_raises(self):

        with self.assertRaises(ValueError):
            AutoCalibrationBudget(max_evaluations=0)

        with self.assertRaises(ValueError):
            AutoCalibrationBudget(max_evaluations=-1)

    def test_to_dict_from_dict_round_trip(self):

        budget = AutoCalibrationBudget(max_evaluations=25)
        restored = AutoCalibrationBudget.from_dict(budget.to_dict())

        self.assertEqual(restored, budget)

    def test_is_frozen(self):

        budget = AutoCalibrationBudget(max_evaluations=5)

        with self.assertRaises(Exception):
            budget.max_evaluations = 10


if __name__ == "__main__":
    unittest.main()
