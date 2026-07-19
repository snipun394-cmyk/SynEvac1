import os

from typing import Any, Dict, List, Sequence

from dataset_builder.exporters import export_csv
from dataset_builder.timeline import TimelineRun, extract_timeline_rows


def export_timeline_csv(rows: Sequence[Dict[str, Any]], file_path: str) -> None:

    # Column set is Building-dependent (Zone_N/Door_N/... counts vary
    # per Building), same reasoning as V1's Scenario Features export --
    # the ordered union of every row's own keys is the column list, so
    # export_csv is called with columns=None.

    export_csv(rows, file_path)


class TimelineDatasetBuilder:

    # One row per timestep, across one or more completed runs -- each
    # row already carries its own scenario_id, so rows from different
    # scenarios (or different Buildings) can share a single CSV.

    def __init__(self, runs: Sequence[TimelineRun]):

        self.runs = list(runs)

    # =====================================================

    def build(self) -> List[Dict[str, Any]]:

        rows = []

        for run in self.runs:
            rows.extend(extract_timeline_rows(run))

        return rows

    # =====================================================

    def export(self, output_dir: str, file_name: str = "timeline.csv") -> str:

        os.makedirs(output_dir, exist_ok=True)

        path = os.path.join(output_dir, file_name)
        export_timeline_csv(self.build(), path)

        return path
