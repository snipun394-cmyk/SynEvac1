import os

from typing import Dict, Optional, Tuple


# =====================================================
# Simulation Replay Studio V1 -- "Open a generated scenario" by id,
# against the one campaign output directory layout designer/campaign/
# campaign_worker.py::_export_scenario_artifacts() and
# research_framework/runner.py::run_scenario_artifacts() both already
# write (that second function's own docstring: "byte-for-byte the same
# layout" -- verified true for every subdirectory this module reads).
#
# This module performs no generation, validation, or simulation -- it
# only locates already-written files by their own already-established
# naming convention, then hands their paths to
# command_center.incident_data.load_incident(), which does the actual
# reading. A distinct, small module rather than a modification of
# training_dataset/loader.py: that loader serves a different purpose
# (CSV-oriented ML dataset loading) and does not know about this
# milestone's own occupant_routes/decision_events artifacts or the JSON
# (not CSV) timeline_rows shape load_incident() itself expects.
# =====================================================

_DATASETS_SUBDIRECTORY = "datasets"

_ARTIFACT_LAYOUT = {
    "ground_truth_path": ("ground_truth", "ground_truth.json"),
    "decision_policy_path": ("decision_policy", "decision_policy.json"),
    "timeline_rows_path": ("timelines", "timeline_rows.json"),
    "occupant_routes_path": ("occupant_routes", "occupant_routes.json"),
    "decision_events_path": ("decision_events", "decision_events.json"),
}


def discover_scenario_ids(output_dir: str) -> Tuple[str, ...]:

    # Every scenario a campaign run produced has its own subdirectory
    # under output_dir/datasets/<scenario_id>/ (Dataset Builder V1's own
    # per-scenario CSV set, written unconditionally for every scenario
    # this campaign accepted) -- the one artifact both known producers
    # never skip, making it the most reliable "which scenario ids exist
    # here" signal.

    datasets_dir = os.path.join(output_dir, _DATASETS_SUBDIRECTORY)

    if not os.path.isdir(datasets_dir):
        return ()

    return tuple(sorted(
        entry for entry in os.listdir(datasets_dir)
        if os.path.isdir(os.path.join(datasets_dir, entry))
    ))


def resolve_scenario_artifacts(output_dir: str, scenario_id: str) -> Dict[str, Optional[str]]:

    # Returns exactly the keyword arguments
    # command_center.incident_data.load_incident() accepts -- a path
    # that does not exist on disk resolves to None (that function's own
    # "optional -- Cancel to skip" convention), never a fabricated path
    # to a file that isn't there.

    paths: Dict[str, Optional[str]] = {
        "project_path": _existing_path(os.path.join(output_dir, "building.syn")),
        "scenario_id": scenario_id,
        "scenario_storage_root": output_dir,
    }

    for keyword, (subdirectory, filename) in _ARTIFACT_LAYOUT.items():

        candidate = os.path.join(output_dir, subdirectory, scenario_id, filename)
        paths[keyword] = candidate if os.path.isfile(candidate) else None

    return paths


def _existing_path(path: str) -> Optional[str]:

    return path if os.path.isfile(path) else None
