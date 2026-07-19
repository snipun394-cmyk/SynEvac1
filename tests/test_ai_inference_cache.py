import unittest

from ai_inference.cache import CacheKey, PredictionCache, compute_feature_hash


class ComputeFeatureHashTests(unittest.TestCase):

    def test_same_row_always_hashes_the_same(self):

        row = {"a": 1, "b": "OPEN"}

        self.assertEqual(compute_feature_hash(row), compute_feature_hash(row))

    def test_key_order_does_not_affect_the_hash(self):

        first = {"a": 1, "b": 2}
        second = {"b": 2, "a": 1}

        self.assertEqual(compute_feature_hash(first), compute_feature_hash(second))

    def test_different_values_hash_differently(self):

        first = {"a": 1}
        second = {"a": 2}

        self.assertNotEqual(compute_feature_hash(first), compute_feature_hash(second))


class PredictionCacheTests(unittest.TestCase):

    def setUp(self):

        self.cache = PredictionCache()

    def test_starts_empty(self):

        self.assertEqual(len(self.cache), 0)
        self.assertEqual(self.cache.hits, 0)
        self.assertEqual(self.cache.misses, 0)
        self.assertIsNone(self.cache.hit_rate)

    def test_get_or_compute_calls_compute_fn_only_once_for_repeated_keys(self):

        calls = []

        def compute():
            calls.append(1)
            return "result"

        first = self.cache.get_or_compute("model-v1", {"a": 1}, "evacuation_time", compute)
        second = self.cache.get_or_compute("model-v1", {"a": 1}, "evacuation_time", compute)

        self.assertEqual(first, "result")
        self.assertEqual(second, "result")
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.cache.hits, 1)
        self.assertEqual(self.cache.misses, 1)
        self.assertEqual(len(self.cache), 1)

    def test_different_model_version_is_a_cache_miss(self):

        self.cache.get_or_compute("model-v1", {"a": 1}, "evacuation_time", lambda: "v1-result")
        self.cache.get_or_compute("model-v2", {"a": 1}, "evacuation_time", lambda: "v2-result")

        self.assertEqual(self.cache.misses, 2)
        self.assertEqual(len(self.cache), 2)

    def test_different_feature_row_is_a_cache_miss(self):

        self.cache.get_or_compute("model-v1", {"a": 1}, "evacuation_time", lambda: "a")
        self.cache.get_or_compute("model-v1", {"a": 2}, "evacuation_time", lambda: "b")

        self.assertEqual(self.cache.misses, 2)

    def test_different_prediction_type_is_a_cache_miss(self):

        self.cache.get_or_compute("model-v1", {"a": 1}, "evacuation_time", lambda: "a")
        self.cache.get_or_compute("model-v1", {"a": 1}, "bottleneck_location", lambda: "b")

        self.assertEqual(self.cache.misses, 2)

    def test_contains_reflects_whether_a_key_has_been_populated(self):

        self.assertFalse(self.cache.contains("model-v1", {"a": 1}, "evacuation_time"))

        self.cache.get_or_compute("model-v1", {"a": 1}, "evacuation_time", lambda: "result")

        self.assertTrue(self.cache.contains("model-v1", {"a": 1}, "evacuation_time"))

    def test_make_key_is_deterministic_and_matches_cache_keys(self):

        key = self.cache.make_key("model-v1", {"a": 1}, "evacuation_time")

        self.assertIsInstance(key, CacheKey)
        self.assertEqual(key, self.cache.make_key("model-v1", {"a": 1}, "evacuation_time"))

    def test_invalidate_model_version_removes_only_that_versions_entries(self):

        self.cache.get_or_compute("model-v1", {"a": 1}, "evacuation_time", lambda: "a")
        self.cache.get_or_compute("model-v2", {"a": 1}, "evacuation_time", lambda: "b")

        removed = self.cache.invalidate_model_version("model-v1")

        self.assertEqual(removed, 1)
        self.assertEqual(len(self.cache), 1)
        self.assertFalse(self.cache.contains("model-v1", {"a": 1}, "evacuation_time"))
        self.assertTrue(self.cache.contains("model-v2", {"a": 1}, "evacuation_time"))

    def test_clear_resets_everything(self):

        self.cache.get_or_compute("model-v1", {"a": 1}, "evacuation_time", lambda: "a")

        self.cache.clear()

        self.assertEqual(len(self.cache), 0)
        self.assertEqual(self.cache.hits, 0)
        self.assertEqual(self.cache.misses, 0)

    def test_hit_rate_reflects_hits_over_total(self):

        self.cache.get_or_compute("model-v1", {"a": 1}, "evacuation_time", lambda: "a")  # miss
        self.cache.get_or_compute("model-v1", {"a": 1}, "evacuation_time", lambda: "a")  # hit
        self.cache.get_or_compute("model-v1", {"a": 1}, "evacuation_time", lambda: "a")  # hit

        self.assertAlmostEqual(self.cache.hit_rate, 2.0 / 3.0)

    def test_no_persistence_a_fresh_cache_instance_starts_empty(self):

        self.cache.get_or_compute("model-v1", {"a": 1}, "evacuation_time", lambda: "a")

        fresh_cache = PredictionCache()

        self.assertEqual(len(fresh_cache), 0)
        self.assertFalse(fresh_cache.contains("model-v1", {"a": 1}, "evacuation_time"))


if __name__ == "__main__":
    unittest.main()
