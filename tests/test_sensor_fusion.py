import unittest
from enum import auto

from sensor_fusion.engine import SensorFusionEngine
from sensor_fusion.observation import FusedObservation, Observation, ObservationKind
from sensor_fusion.provider import ManualObservationProvider, ObservationProvider


# =====================================================
# Sensor Fusion Engine milestone, Phase 9 -- deterministic, offline
# unit tests. No randomness anywhere in this file.
# =====================================================


def obs(source, kind, location, timestamp, confidence, measurement):

    return Observation(
        source=source, kind=kind, location=location, timestamp=timestamp,
        confidence=confidence, measurement=measurement,
    )


class FailingProvider(ObservationProvider):

    def collect(self, time):
        raise RuntimeError("simulated provider failure")


class StaticProvider(ObservationProvider):

    def __init__(self, observations):
        self._observations = tuple(observations)

    def collect(self, time):
        return self._observations


class SingleProviderTests(unittest.TestCase):

    def test_1_single_provider_single_observation_fuses_cleanly(self):

        engine = SensorFusionEngine()

        observations = (obs("smoke-1", ObservationKind.SMOKE, "zone-1", 0.0, 0.9, False),)
        fused = engine.fuse(observations, time=0.0)

        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0].kind, ObservationKind.SMOKE)
        self.assertEqual(fused[0].location, "zone-1")
        self.assertFalse(fused[0].measurement)
        self.assertAlmostEqual(fused[0].confidence, 0.9)
        self.assertFalse(fused[0].conflict)
        self.assertEqual(fused[0].contributing_sources, ("smoke-1",))


class MultipleProviderTests(unittest.TestCase):

    def test_2_multiple_providers_collected_and_fused_together(self):

        engine = SensorFusionEngine()

        provider_a = StaticProvider([obs("camera", ObservationKind.OCCUPANCY, "zone-1", 0.0, 0.9, 2.0)])
        provider_b = StaticProvider([obs("manual", ObservationKind.OCCUPANCY, "zone-1", 0.0, 0.8, 2.0)])

        observations = engine.collect([provider_a, provider_b], time=0.0)
        fused = engine.fuse(observations, time=0.0)

        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0].measurement, 2.0)
        self.assertEqual(set(fused[0].contributing_sources), {"camera", "manual"})

    def test_2_multiple_locations_and_kinds_fuse_independently(self):

        engine = SensorFusionEngine()

        observations = (
            obs("camera", ObservationKind.OCCUPANCY, "zone-1", 0.0, 0.9, 3.0),
            obs("smoke-1", ObservationKind.SMOKE, "zone-1", 0.0, 0.9, False),
            obs("camera", ObservationKind.OCCUPANCY, "zone-2", 0.0, 0.9, 1.0),
        )
        fused = engine.fuse(observations, time=0.0)

        self.assertEqual(len(fused), 3)
        keys = {(f.location, f.kind) for f in fused}
        self.assertEqual(keys, {
            ("zone-1", ObservationKind.OCCUPANCY),
            ("zone-1", ObservationKind.SMOKE),
            ("zone-2", ObservationKind.OCCUPANCY),
        })


class ConflictingProviderTests(unittest.TestCase):

    def test_3_agreeing_occupancy_sources_get_an_agreement_bonus(self):

        engine = SensorFusionEngine(agreement_bonus=0.15)

        observations = (
            obs("camera", ObservationKind.OCCUPANCY, "zone-1", 0.0, 0.7, 2.0),
            obs("manual", ObservationKind.OCCUPANCY, "zone-1", 0.0, 0.7, 2.0),
        )
        fused = engine.fuse(observations, time=0.0)[0]

        self.assertFalse(fused.conflict)
        self.assertAlmostEqual(fused.confidence, 0.7 + 0.15)

    def test_camera_occupied_smoke_silent_manual_occupied_worked_example(self):

        # Phase 7's own worked example: camera+manual AGREE on
        # occupancy; the smoke detector's silence is a totally separate
        # SMOKE observation, never treated as contradicting occupancy.
        engine = SensorFusionEngine()

        observations = (
            obs("camera", ObservationKind.OCCUPANCY, "zone-1", 0.0, 0.8, 1.0),
            obs("manual", ObservationKind.OCCUPANCY, "zone-1", 0.0, 0.8, 1.0),
            obs("smoke-1", ObservationKind.SMOKE, "zone-1", 0.0, 0.9, False),
        )
        fused = engine.fuse(observations, time=0.0)
        by_kind = {f.kind: f for f in fused}

        self.assertFalse(by_kind[ObservationKind.OCCUPANCY].conflict)
        self.assertEqual(by_kind[ObservationKind.OCCUPANCY].measurement, 1.0)
        self.assertFalse(by_kind[ObservationKind.SMOKE].measurement)
        self.assertFalse(by_kind[ObservationKind.SMOKE].conflict)

    def test_4_disagreeing_alarm_sources_produce_a_conflict_and_penalty(self):

        engine = SensorFusionEngine(conflict_penalty=0.3)

        observations = (
            obs("smoke-1", ObservationKind.SMOKE, "zone-1", 0.0, 0.9, True),
            obs("smoke-2", ObservationKind.SMOKE, "zone-1", 0.0, 0.9, False),
        )
        fused = engine.fuse(observations, time=0.0)[0]

        self.assertTrue(fused.conflict)
        self.assertTrue(fused.measurement)  # ANY source alarming wins -- life-safety worst-case
        self.assertAlmostEqual(fused.confidence, 0.9 - 0.3)

    def test_4_numeric_disagreement_within_tolerance_is_not_a_conflict(self):

        engine = SensorFusionEngine()

        observations = (
            obs("camera", ObservationKind.OCCUPANCY, "zone-1", 0.0, 0.8, 3.0),
            obs("manual", ObservationKind.OCCUPANCY, "zone-1", 0.0, 0.8, 4.0),  # within default tolerance (2.0)
        )
        fused = engine.fuse(observations, time=0.0)[0]

        self.assertFalse(fused.conflict)
        self.assertEqual(fused.measurement, 4.0)  # worst-case MAX

    def test_4_numeric_disagreement_beyond_tolerance_is_a_conflict(self):

        engine = SensorFusionEngine()

        observations = (
            obs("camera", ObservationKind.OCCUPANCY, "zone-1", 0.0, 0.8, 1.0),
            obs("manual", ObservationKind.OCCUPANCY, "zone-1", 0.0, 0.8, 10.0),  # way beyond tolerance
        )
        fused = engine.fuse(observations, time=0.0)[0]

        self.assertTrue(fused.conflict)


class MissingProviderTests(unittest.TestCase):

    def test_5_no_providers_produces_no_fused_observations(self):

        engine = SensorFusionEngine()

        observations = engine.collect([], time=0.0)
        fused = engine.fuse(observations, time=0.0)

        self.assertEqual(fused, ())

    def test_5_heat_alarm_camera_unavailable_retains_confidence_honestly(self):

        # Phase 7's own second worked example -- a SINGLE contributing
        # source must neither be penalized (no conflict exists) nor
        # boosted (no corroboration exists either).
        engine = SensorFusionEngine()

        observations = (obs("heat-1", ObservationKind.HEAT, "zone-1", 0.0, 0.85, True),)
        fused = engine.fuse(observations, time=0.0)[0]

        self.assertFalse(fused.conflict)
        self.assertAlmostEqual(fused.confidence, 0.85)  # unchanged -- neither bonus nor penalty


class StaleObservationTests(unittest.TestCase):

    def test_6_stale_observation_confidence_decays(self):

        engine = SensorFusionEngine(staleness_half_life_seconds=10.0)

        observations = (obs("smoke-1", ObservationKind.SMOKE, "zone-1", 0.0, 0.8, True),)

        fresh = engine.fuse(observations, time=0.0)[0]
        stale = engine.fuse(observations, time=10.0)[0]  # one half-life later

        self.assertAlmostEqual(fresh.confidence, 0.8)
        self.assertAlmostEqual(stale.confidence, 0.4)


class ConfidenceChangeTests(unittest.TestCase):

    def test_7_source_weighting_scales_confidence(self):

        engine = SensorFusionEngine(source_weights={"unverified-manual": 0.5})

        observations = (obs("unverified-manual", ObservationKind.OCCUPANCY, "zone-1", 0.0, 0.8, 1.0),)
        fused = engine.fuse(observations, time=0.0)[0]

        self.assertAlmostEqual(fused.confidence, 0.4)


class ProviderFailureTests(unittest.TestCase):

    def test_8_a_failing_provider_never_blocks_other_providers(self):

        engine = SensorFusionEngine()

        good_provider = StaticProvider([obs("camera", ObservationKind.OCCUPANCY, "zone-1", 0.0, 0.9, 1.0)])

        observations = engine.collect([FailingProvider(), good_provider], time=0.0)

        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0].source, "camera")


class LateObservationTests(unittest.TestCase):

    def test_9_observation_timestamped_after_fusion_time_is_never_penalized(self):

        engine = SensorFusionEngine(staleness_half_life_seconds=10.0)

        # timestamp (5.0) is AFTER the fusion time (0.0) -- a slight
        # clock-skew/late-arriving-but-stamped-earlier case; age must
        # clamp to 0, never apply a nonsensical "negative decay" boost
        # past 1.0 or crash.
        observations = (obs("smoke-1", ObservationKind.SMOKE, "zone-1", 5.0, 0.8, True),)
        fused = engine.fuse(observations, time=0.0)[0]

        self.assertAlmostEqual(fused.confidence, 0.8)


class DuplicateObservationTests(unittest.TestCase):

    def test_10_duplicate_observations_from_the_same_source_do_not_inflate_agreement(self):

        engine = SensorFusionEngine(agreement_bonus=0.15)

        # The SAME source reporting twice this cycle -- must be treated
        # as ONE opinion, not two independent corroborating ones.
        observations = (
            obs("camera", ObservationKind.OCCUPANCY, "zone-1", 0.0, 0.7, 2.0),
            obs("camera", ObservationKind.OCCUPANCY, "zone-1", 0.0, 0.75, 2.0),
        )
        fused = engine.fuse(observations, time=0.0)[0]

        self.assertEqual(fused.contributing_sources, ("camera",))
        self.assertAlmostEqual(fused.confidence, 0.75)  # the better of the two duplicate reports, no bonus applied


class FutureProviderCompatibilityTests(unittest.TestCase):

    def test_11_a_kind_with_no_dedicated_merge_rule_still_fuses_via_the_generic_fallback(self):

        # Extending the real ObservationKind enum at runtime is not
        # possible (it's a stdlib Enum), so this exercises merge.py's
        # own documented generic-fallback branch directly -- exactly
        # the code path a genuinely future ObservationKind (Phase 3's
        # own "Future sensors" requirement) would take, proving new
        # sensor kinds plug in without any change to merge.py/conflict.py.
        from sensor_fusion.merge import _highest_confidence_value

        high_confidence_obs = obs("sensor-b", ObservationKind.ALARM, "zone-1", 0.0, 0.95, True)
        low_confidence_obs = obs("sensor-a", ObservationKind.ALARM, "zone-1", 0.0, 0.4, True)

        result = _highest_confidence_value([low_confidence_obs, high_confidence_obs])

        self.assertEqual(result, True)

    def test_11_new_provider_implementation_requires_only_the_collect_method(self):

        class FutureSensorProvider(ObservationProvider):

            def collect(self, time):
                return (obs("future-ble-beacon-1", ObservationKind.OCCUPANCY, "zone-1", time, 0.6, 1.0),)

        engine = SensorFusionEngine()
        observations = engine.collect([FutureSensorProvider()], time=0.0)
        fused = engine.fuse(observations, time=0.0)

        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0].contributing_sources, ("future-ble-beacon-1",))


class DeterminismTests(unittest.TestCase):

    def test_12_output_order_is_deterministic_regardless_of_input_order(self):

        engine = SensorFusionEngine()

        observations_a = (
            obs("camera", ObservationKind.OCCUPANCY, "zone-2", 0.0, 0.9, 1.0),
            obs("smoke-1", ObservationKind.SMOKE, "zone-1", 0.0, 0.9, False),
        )
        observations_b = tuple(reversed(observations_a))

        fused_a = engine.fuse(observations_a, time=0.0)
        fused_b = engine.fuse(observations_b, time=0.0)

        self.assertEqual(
            [(f.location, f.kind) for f in fused_a],
            [(f.location, f.kind) for f in fused_b],
        )


class ManualObservationProviderTests(unittest.TestCase):

    def test_13_manual_provider_reports_then_clears(self):

        provider = ManualObservationProvider()

        provider.report(ObservationKind.OCCUPANCY, "zone-1", 3.0, 0.9, 0.0)

        first = provider.collect(0.0)
        self.assertEqual(len(first), 1)

        second = provider.collect(1.0)
        self.assertEqual(second, ())  # cleared after being collected once


if __name__ == "__main__":
    unittest.main()
