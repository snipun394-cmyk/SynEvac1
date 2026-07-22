import unittest

from models.dynamic_sign import SignIndication

from dynamic_signage.direction import bearing_to, relative_direction
from dynamic_signage.models import DynamicSignageConfig


# =====================================================
# Live Dynamic Evacuation Signage milestone, Phase 27 -- test matrix
# items 7-11: straight/left/right direction geometry, threshold
# boundary determinism, orientation convention.
#
# WORKED EXAMPLE (documented in dynamic_signage/direction.py's own
# module docstring, independently re-derived here via real-world
# compass reasoning as a cross-check, not merely re-asserting the
# implementation):
#
#   Sign at (22, 5), target at (15, 5) -- target bearing 180 degrees
#   (due "west" on the floor plan, i.e. target is directly to the
#   sign's own -x side).
#
#   - Sign facing theta=180 (also pointing at -x, i.e. STRAIGHT at the
#     target) -> delta=0 -> STRAIGHT.
#   - Sign facing theta=90 ("south", +y) -> compass rule: facing south,
#     right hand = west = the target's own bearing -> RIGHT.
#   - Sign facing theta=270/-90 ("north", -y) -> compass rule: facing
#     north, left hand = west = the target's own bearing -> LEFT.
# =====================================================


SIGN_POSITION = (22.0, 5.0)
TARGET_POSITION = (15.0, 5.0)


class BearingTests(unittest.TestCase):

    def test_bearing_due_west_is_180(self):

        self.assertAlmostEqual(bearing_to(SIGN_POSITION, TARGET_POSITION), 180.0)

    def test_bearing_to_same_position_is_none(self):

        self.assertIsNone(bearing_to(SIGN_POSITION, SIGN_POSITION))


class RelativeDirectionTests(unittest.TestCase):

    def test_straight(self):

        self.assertEqual(
            relative_direction(SIGN_POSITION, 180.0, TARGET_POSITION), SignIndication.STRAIGHT,
        )

    def test_right(self):

        self.assertEqual(
            relative_direction(SIGN_POSITION, 90.0, TARGET_POSITION), SignIndication.RIGHT,
        )

    def test_left(self):

        self.assertEqual(
            relative_direction(SIGN_POSITION, 270.0, TARGET_POSITION), SignIndication.LEFT,
        )

    def test_left_using_negative_orientation(self):

        # -90 and 270 are the same physical orientation -- confirms
        # angle normalization treats them identically.
        self.assertEqual(
            relative_direction(SIGN_POSITION, -90.0, TARGET_POSITION), SignIndication.LEFT,
        )

    def test_co_located_target_yields_no_honest_direction(self):

        self.assertIsNone(relative_direction(SIGN_POSITION, 0.0, SIGN_POSITION))

    def test_threshold_boundary_inclusive_is_straight(self):

        config = DynamicSignageConfig(straight_threshold_degrees=20.0)

        # Facing exactly 20 degrees off the target bearing -- the
        # boundary itself is documented as inclusive (STRAIGHT), never
        # ambiguous between STRAIGHT and RIGHT.
        self.assertEqual(
            relative_direction(SIGN_POSITION, 200.0, TARGET_POSITION, config), SignIndication.STRAIGHT,
        )

    def test_just_past_threshold_boundary_is_right(self):

        config = DynamicSignageConfig(straight_threshold_degrees=20.0)

        # orientation=159.99 -> delta = 180 - 159.99 = +20.01, just past
        # the 20-degree threshold on the positive (RIGHT) side.
        self.assertEqual(
            relative_direction(SIGN_POSITION, 159.99, TARGET_POSITION, config), SignIndication.RIGHT,
        )

    def test_threshold_boundary_deterministic_across_repeated_calls(self):

        config = DynamicSignageConfig(straight_threshold_degrees=20.0)

        results = {
            relative_direction(SIGN_POSITION, 200.0, TARGET_POSITION, config)
            for _ in range(10)
        }

        self.assertEqual(results, {SignIndication.STRAIGHT})

    def test_custom_threshold_configurable(self):

        config = DynamicSignageConfig(straight_threshold_degrees=5.0)

        # orientation=170 -> delta = 180 - 170 = +10, which exceeds a
        # narrowed 5-degree threshold (RIGHT) even though it would have
        # been STRAIGHT under the default 20-degree threshold.
        self.assertEqual(
            relative_direction(SIGN_POSITION, 170.0, TARGET_POSITION, config), SignIndication.RIGHT,
        )


if __name__ == "__main__":
    unittest.main()
