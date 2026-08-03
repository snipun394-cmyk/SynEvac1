import importlib
from typing import Optional

from calibration_studio.benchmark import GeometryVersion


# =====================================================
# Calibration Studio Phase 4 -- Calibration Runner.
#
# PublishedBenchmark.geometry_reference.ref (calibration_studio/
# benchmark.py) was deliberately left an opaque string in Phase 3, its
# own docstring naming exactly this as "the future Calibration
# Runner's job, not this phase's." This is that job, and nothing more:
# a dotted "module.attribute" path is resolved to the real Python
# callable it names (today, one of the existing NIST recreation
# scripts' own bespoke building-construction functions -- e.g.
# "scripts.run_nist_10story_validation.build_nist_10story_building")
# and called with no arguments to get a real Building. No new
# geometry-construction capability is added anywhere -- this only
# imports and calls a function that already exists.
#
# Failures are never swallowed: a missing/broken ref surfaces as a
# real ImportError/AttributeError from the import machinery itself
# (or whatever the factory function's own body raises), not a vague
# "could not resolve geometry" message -- the caller sees exactly what
# went wrong. Only geometry_reference itself being None (the honest,
# legitimate case for a DATASET_VALIDATION benchmark, which has none)
# returns None rather than raising.
# =====================================================


def resolve_geometry_reference(geometry_reference: Optional[GeometryVersion]):

    if geometry_reference is None:
        return None

    module_path, separator, attribute_name = geometry_reference.ref.rpartition(".")

    if not separator:
        raise ValueError(
            f"geometry_reference.ref {geometry_reference.ref!r} is not a dotted "
            f"'module.attribute' path -- cannot resolve a building factory from it.",
        )

    module = importlib.import_module(module_path)
    factory = getattr(module, attribute_name)

    return factory()
