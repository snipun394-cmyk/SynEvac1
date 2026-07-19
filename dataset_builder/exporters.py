import csv

from typing import Any, Mapping, Optional, Sequence

from dataset_builder.schema import SIMULATION_OUTCOME_COLUMNS, ZONE_RESULT_COLUMNS


# =====================================================
# CSV export only, for V1 -- one CSV per dataset. Parquet, analytics,
# and any other export format belong to a future version.
# =====================================================


def export_csv(
    rows: Sequence[Mapping[str, Any]],
    file_path: str,
    columns: Optional[Sequence[str]] = None,
) -> None:

    # columns is required for the two fixed-shape datasets (Simulation
    # Outcomes, Zone-Level Results), whose column set never depends on
    # a particular Building. Left as None for Dataset 1 (Scenario
    # Features), whose column set is Building-dependent (Zone_N/Door_N/
    # ... counts vary per Building) -- there the ordered union of every
    # row's own keys is the column list, so rows produced from
    # different Buildings can still share one CSV without losing data
    # (a row missing a column just gets "" for it).

    fieldnames = list(columns) if columns is not None else _infer_columns(rows)

    with open(file_path, "w", newline="", encoding="utf-8") as csv_file:

        writer = csv.DictWriter(
            csv_file, fieldnames=fieldnames, restval="", extrasaction="ignore",
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(_sanitized_row(row, fieldnames))


# =====================================================


def export_scenario_features_csv(rows: Sequence[Mapping[str, Any]], file_path: str) -> None:

    export_csv(rows, file_path)


def export_simulation_outcomes_csv(rows: Sequence[Mapping[str, Any]], file_path: str) -> None:

    export_csv(rows, file_path, columns=SIMULATION_OUTCOME_COLUMNS)


def export_zone_results_csv(rows: Sequence[Mapping[str, Any]], file_path: str) -> None:

    export_csv(rows, file_path, columns=ZONE_RESULT_COLUMNS)


# =====================================================


def _infer_columns(rows: Sequence[Mapping[str, Any]]):

    columns = []
    seen = set()

    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                columns.append(key)

    return columns


# =====================================================


def _sanitized_row(row: Mapping[str, Any], fieldnames: Sequence[str]) -> dict:

    # A missing key and a present-but-None value both mean "no value"
    # for CSV purposes -- both become "", never the literal text
    # "None".

    return {
        field: ("" if row.get(field) is None else row.get(field))
        for field in fieldnames
    }
