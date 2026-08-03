import unittest

from calibration_benchmark import WalkingSpeedCandidate

from automatic_calibration.search_space import ParameterDimension, SearchSpace


def _walking_speed_dimension():

    return ParameterDimension(
        name="Adult_Default.walking_speed",
        bounds=(0.8, 1.6),
        build=lambda value: WalkingSpeedCandidate("Adult_Default", value, "test", "test"),
    )


class ParameterDimensionTests(unittest.TestCase):

    def test_build_candidate_constructs_a_real_parameter_candidate(self):

        dimension = _walking_speed_dimension()
        candidate = dimension.build_candidate(1.1)

        self.assertIsInstance(candidate, WalkingSpeedCandidate)
        self.assertEqual(candidate.candidate_value, 1.1)

    def test_build_candidate_is_called_fresh_every_time(self):

        dimension = _walking_speed_dimension()

        a = dimension.build_candidate(1.0)
        b = dimension.build_candidate(1.0)

        self.assertIsNot(a, b)

    def test_describe_is_json_safe_and_never_includes_the_live_build_callable(self):

        dimension = _walking_speed_dimension()
        description = dimension.describe()

        self.assertEqual(description, {"name": "Adult_Default.walking_speed", "bounds": [0.8, 1.6]})
        self.assertNotIn("build", description)


class SearchSpaceTests(unittest.TestCase):

    def test_single_dimension_search_space_iterates_and_reports_length(self):

        space = SearchSpace(dimensions=(_walking_speed_dimension(),))

        self.assertEqual(len(space), 1)
        self.assertEqual(list(space), list(space.dimensions))

    def test_describe_returns_one_entry_per_dimension(self):

        space = SearchSpace(dimensions=(_walking_speed_dimension(),))
        description = space.describe()

        self.assertEqual(len(description), 1)
        self.assertEqual(description[0]["name"], "Adult_Default.walking_speed")

    def test_empty_search_space_raises(self):

        with self.assertRaises(ValueError):
            SearchSpace(dimensions=())

    def test_duplicate_dimension_names_raise(self):

        dimension = _walking_speed_dimension()

        with self.assertRaises(ValueError):
            SearchSpace(dimensions=(dimension, dimension))


if __name__ == "__main__":
    unittest.main()
