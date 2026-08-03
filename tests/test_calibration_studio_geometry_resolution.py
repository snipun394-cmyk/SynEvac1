import unittest

from calibration_studio.benchmark import GeometryVersion
from calibration_studio.geometry_resolution import resolve_geometry_reference

from tests.calibration_benchmark_fixtures import make_building


class ResolveGeometryReferenceTests(unittest.TestCase):

    def test_none_returns_none(self):

        self.assertIsNone(resolve_geometry_reference(None))

    def test_valid_ref_resolves_to_a_real_building(self):

        # Reuses the existing calibration_benchmark test fixture --
        # never a new building/dataset.
        ref = GeometryVersion(version="v1", ref="tests.calibration_benchmark_fixtures.make_building")

        building = resolve_geometry_reference(ref)

        expected = make_building()
        self.assertEqual(building.id, expected.id)
        self.assertEqual(building.name, expected.name)

    def test_ref_without_a_dot_raises_value_error(self):

        ref = GeometryVersion(version="v1", ref="not_a_dotted_path")

        with self.assertRaises(ValueError):
            resolve_geometry_reference(ref)

    def test_unresolvable_module_raises_import_error(self):

        ref = GeometryVersion(version="v1", ref="this.module.does.not.exist.build")

        with self.assertRaises(ImportError):
            resolve_geometry_reference(ref)

    def test_unresolvable_attribute_raises_attribute_error(self):

        ref = GeometryVersion(version="v1", ref="tests.calibration_benchmark_fixtures.this_function_does_not_exist")

        with self.assertRaises(AttributeError):
            resolve_geometry_reference(ref)


if __name__ == "__main__":
    unittest.main()
