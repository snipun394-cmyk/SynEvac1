import unittest

from automatic_calibration.strategy import AutoCalibrationStrategy


class AutoCalibrationStrategyInterfaceTests(unittest.TestCase):

    def test_propose_is_not_implemented_on_the_base_class(self):

        with self.assertRaises(NotImplementedError):
            AutoCalibrationStrategy().propose(search_space=None, objective=None, history=())

    def test_describe_is_not_implemented_on_the_base_class(self):

        with self.assertRaises(NotImplementedError):
            AutoCalibrationStrategy().describe()


if __name__ == "__main__":
    unittest.main()
