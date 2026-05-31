from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

import pandas as pd

# ---------------------------------------------------------------------------
# Column detection: datetime
# ---------------------------------------------------------------------------

# Exact full-column-name matches (case-insensitive)
_DATETIME_EXACT = {
    "year", "month", "day", "hour", "minute", "second",
    "date", "time", "datetime", "timestamp", "week", "quarter",
    "weekday", "dayofweek",
}
# Token-boundary matching: datetime word must be the FIRST or LAST token.
# Prevents false positives like "two_year_recid" or "hours_per_week".
_DATETIME_BOUNDARY_TOKENS = {
    "year", "month", "day", "hour", "minute", "second",
    "date", "time", "datetime", "timestamp",
}
# Regex patterns for formatted date strings in cell values
_DATE_REGEXES = [
    re.compile(r"^\d{4}-\d{2}-\d{2}"),   # YYYY-MM-DD (ISO)
    re.compile(r"^\d{2}/\d{2}/\d{4}"),   # MM/DD/YYYY
    re.compile(r"^\d{4}/\d{2}/\d{2}"),   # YYYY/MM/DD
    re.compile(r"^\d{2}-\d{2}-\d{4}"),   # DD-MM-YYYY
]

# ---------------------------------------------------------------------------
# Column detection: serial / ID
# ---------------------------------------------------------------------------

_SERIAL_EXACT = {
    "id", "no", "index", "row", "serial", "rownum", "row_id",
    "record_id", "row_num", "idx",
}
_SERIAL_ENDSWITH = ["_id", "_no", "_num", "_number", "_index", "_serial", "_key"]
_SERIAL_STARTSWITH = ["id_", "row_"]

# ---------------------------------------------------------------------------
# Column detection: protected attributes
# ---------------------------------------------------------------------------

_PROTECTED_EXACT = {
    "race", "gender", "sex", "age", "ethnicity", "religion",
    "nationality", "marital", "marital_status", "disability",
}


def _is_datetime_col_by_name(col: str) -> bool:
    lower = col.lower()
    if lower in _DATETIME_EXACT:
        return True
    tokens = [t for t in lower.replace("-", "_").split("_") if t]
    if not tokens:
        return False
    return tokens[0] in _DATETIME_BOUNDARY_TOKENS or tokens[-1] in _DATETIME_BOUNDARY_TOKENS


def _is_serial_col_by_name(col: str) -> bool:
    lower = col.lower()
    if lower in _SERIAL_EXACT:
        return True
    if any(lower.endswith(s) for s in _SERIAL_ENDSWITH):
        return True
    if any(lower.startswith(s) for s in _SERIAL_STARTSWITH):
        return True
    return False


def _is_protected_col(col: str) -> bool:
    return col.lower() in _PROTECTED_EXACT


def _is_sequential_integers(values: List[str]) -> bool:
    """Return True if values form a contiguous +1 integer sequence.

    Accepts integer-valued float strings (e.g. '4.0', which pandas produces for an
    integer CSV column that has a missing value) but rejects genuine fractional
    values so '1.5','2.5' is never treated as sequential.
    """
    try:
        ints: list[int] = []
        for v in values:
            if not v:
                continue
            f = float(v)
            if not f.is_integer():
                return False
            ints.append(int(f))
        if len(ints) < 2:
            return False
        return all(ints[i + 1] - ints[i] == 1 for i in range(len(ints) - 1))
    except (ValueError, TypeError):
        return False


def _has_date_string_values(values: List[str]) -> bool:
    """Return True if >50% of sampled non-empty values match a date format regex."""
    non_empty = [v for v in values if v]
    if not non_empty:
        return False
    matches = sum(1 for v in non_empty if any(r.match(v) for r in _DATE_REGEXES))
    return (matches / len(non_empty)) > 0.5


@dataclass(frozen=True)
class ColumnFlags:
    datetime: List[str]
    serial: List[str]
    protected: List[str]


def _sample_strings(series: pd.Series, sample: int) -> List[str]:
    """Stripped string values for the first `sample` rows; NA becomes ''.

    Reproduces 1_dataset_status.py's csv.reader + .strip() semantics so the
    name/value heuristics behave identically regardless of the source dtype.
    """
    return ["" if pd.isna(v) else str(v).strip() for v in series.head(sample).tolist()]


def flag_columns(
    df: pd.DataFrame,
    numerical_columns: List[str] | None = None,
    sample: int = 200,
) -> ColumnFlags:
    """Heuristically flag datetime / serial-ID / protected columns.

    Rules (ported verbatim from scripts/1_dataset_status.py):
      - datetime: name heuristic (exact + first/last token boundary) OR >50% of the
        first `sample` non-empty values match a date regex.
      - serial: name heuristic (exact / endswith / startswith) OR
        (col in numerical_columns AND sampled values are a contiguous +1 sequence).
        When numerical_columns is None the sequential check is skipped (name-only).
      - protected: name-only.
    Per-column priority matches the original: a datetime-flagged column is not also
    tested for serial (elif chain); protected is independent.
    """
    numerical = set(numerical_columns or [])
    datetime_cols: List[str] = []
    serial_cols: List[str] = []
    protected_cols: List[str] = []

    for col in df.columns.astype(str):
        values = _sample_strings(df[col], sample)

        if _is_protected_col(col):
            protected_cols.append(col)

        if _is_datetime_col_by_name(col):
            datetime_cols.append(col)
        elif _has_date_string_values(values):
            datetime_cols.append(col)
        elif _is_serial_col_by_name(col) or (
            col in numerical and _is_sequential_integers(values)
        ):
            serial_cols.append(col)

    return ColumnFlags(datetime=datetime_cols, serial=serial_cols, protected=protected_cols)
