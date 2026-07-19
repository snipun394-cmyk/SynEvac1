from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# =====================================================
# Phase 1 -- feature selection, categorical encoding, normalization,
# missing-value handling, all deterministic and leak-free. Every row
# here is a plain Dict[str, Any] (a TrainingDataset accessor's own
# shape) -- this module never assumes pandas/polars is installed.
# =====================================================


def select_features(
    rows: Sequence[Dict[str, Any]],
    *,
    exclude: Sequence[str] = (),
    include: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:

    # Explicit feature-selection step: drop identifier columns
    # (exclude) and/or restrict to only an explicit allow-list
    # (include). Deliberately dumb -- it never guesses which columns
    # "look like" features, so a caller always knows exactly what a
    # model was trained on.

    exclude_set = set(exclude)

    def _select(row: Dict[str, Any]) -> Dict[str, Any]:

        keys = include if include is not None else row.keys()

        return {key: row.get(key) for key in keys if key not in exclude_set}

    return [_select(row) for row in rows]


# =====================================================


def _is_numeric_value(value: Any) -> bool:

    if value is None:
        return True  # missingness never disqualifies a column from being numeric

    if isinstance(value, bool):
        return False  # True/False are states here, not magnitudes -- treated as categorical

    if isinstance(value, (int, float)):
        return True

    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


@dataclass(frozen=True)
class FeatureSchema:

    # A column is numeric if every non-missing value observed for it
    # (across the rows schema was inferred from) parses as a float;
    # otherwise it's categorical. Column order is always alphabetical
    # -- deterministic regardless of dict insertion order or which
    # rows happened to be sampled first.

    numeric_columns: Tuple[str, ...]
    categorical_columns: Tuple[str, ...]

    @property
    def columns(self) -> Tuple[str, ...]:

        return self.numeric_columns + self.categorical_columns

    @classmethod
    def infer(cls, rows: Sequence[Dict[str, Any]], *, exclude: Sequence[str] = ()) -> "FeatureSchema":

        exclude_set = set(exclude)
        columns = sorted({key for row in rows for key in row.keys()} - exclude_set)

        numeric: List[str] = []
        categorical: List[str] = []

        for column in columns:

            observed = [row.get(column) for row in rows if row.get(column) is not None]

            if observed and all(_is_numeric_value(value) for value in observed):
                numeric.append(column)
            else:
                categorical.append(column)

        return cls(numeric_columns=tuple(numeric), categorical_columns=tuple(categorical))


# =====================================================


def _rows_to_table(rows: Sequence[Dict[str, Any]], schema: FeatureSchema) -> np.ndarray:

    columns = schema.columns
    categorical_set = set(schema.categorical_columns)

    table = np.empty((len(rows), len(columns)), dtype=object)

    for row_index, row in enumerate(rows):
        for column_index, column in enumerate(columns):

            value = row.get(column)

            if value is None:
                table[row_index, column_index] = np.nan
            elif column in categorical_set:
                table[row_index, column_index] = str(value)
            else:
                table[row_index, column_index] = float(value)

    return table


class Preprocessor:

    # Deterministic, leak-free preprocessing: numeric columns are
    # median-imputed then standardized; categorical columns are
    # most-frequent-imputed then one-hot encoded (unseen categories at
    # transform time map to an all-zero row, never raise). Every
    # statistic (medians, means/stds, category vocabularies) is
    # captured once, at fit() time, from whatever rows are passed in --
    # transform() only ever *applies* those fixed statistics, so
    # fit() must only ever be called with a TRAIN split's rows for
    # there to be no leakage from validation/test data into scaling or
    # encoding.

    def __init__(self, schema: FeatureSchema):

        self.schema = schema
        self._column_transformer: Optional[ColumnTransformer] = None

    # =====================================================

    def fit(self, rows: Sequence[Dict[str, Any]]) -> "Preprocessor":

        if not self.schema.columns:
            raise ValueError("FeatureSchema has no columns to fit a Preprocessor against")

        table = _rows_to_table(rows, self.schema)

        n_numeric = len(self.schema.numeric_columns)
        numeric_positions = list(range(n_numeric))
        categorical_positions = list(range(n_numeric, len(self.schema.columns)))

        transformers = []

        if numeric_positions:

            transformers.append((
                "numeric",
                Pipeline([
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]),
                numeric_positions,
            ))

        if categorical_positions:

            transformers.append((
                "categorical",
                Pipeline([
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]),
                categorical_positions,
            ))

        self._column_transformer = ColumnTransformer(transformers, remainder="drop")
        self._column_transformer.fit(table)

        return self

    # =====================================================

    def transform(self, rows: Sequence[Dict[str, Any]]) -> np.ndarray:

        if self._column_transformer is None:
            raise RuntimeError("Preprocessor.transform() called before fit()")

        table = _rows_to_table(rows, self.schema)

        return np.asarray(self._column_transformer.transform(table), dtype=float)

    # =====================================================

    def fit_transform(self, rows: Sequence[Dict[str, Any]]) -> np.ndarray:

        return self.fit(rows).transform(rows)

    # =====================================================

    def feature_names_out(self) -> List[str]:

        if self._column_transformer is None:
            raise RuntimeError("Preprocessor.feature_names_out() called before fit()")

        names: List[str] = list(self.schema.numeric_columns)

        if self.schema.categorical_columns:

            encoder = self._column_transformer.named_transformers_["categorical"].named_steps["encode"]

            for column, categories in zip(self.schema.categorical_columns, encoder.categories_):
                names.extend(f"{column}={category}" for category in categories)

        return names
