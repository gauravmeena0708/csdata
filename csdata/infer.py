from __future__ import annotations
import pandas as pd
import numpy as np
from csdata.spec import DatasetSpec

# Integer target with at most this many distinct values is treated as a
# discrete label (multiclass) rather than a regression target.
_MAX_MULTICLASS = 20


def _infer_task_type(s: pd.Series) -> str:
    """Guess a task_type from the target column's values."""
    nun = int(s.nunique(dropna=True))
    if nun == 2:
        return "binclass"
    if pd.api.types.is_float_dtype(s):
        # All-integer floats (e.g. 0.0/1.0/2.0 labels) read as discrete.
        non_null = s.dropna()
        if non_null.empty or not (non_null % 1 == 0).all():
            return "regression"
    if pd.api.types.is_numeric_dtype(s):
        return "multiclass" if nun <= _MAX_MULTICLASS else "regression"
    return "multiclass"


def _is_id_like(s: pd.Series, name: str, n_rows: int) -> bool:
    """Heuristic: does this column look like a row identifier?"""
    lname = str(name).strip().lower()
    name_hit = lname == "id" or lname.endswith("_id") or lname in {"uuid", "guid", "index", "key"}
    # Unique-per-row integers/strings look like keys; unique floats are usually
    # genuine measurements, so they are not treated as ids on values alone.
    value_hit = s.nunique(dropna=False) == n_rows and (
        pd.api.types.is_integer_dtype(s) or s.dtype == object
    )
    return name_hit or value_hit


def infer_spec(
    df: pd.DataFrame,
    target: str,
    name: str = "custom",
    task_type: str | None = None,
    drop_ids: bool = False,
) -> DatasetSpec:
    """Infer a DatasetSpec from a DataFrame using heuristics.

    task_type is inferred from the target when not given. When drop_ids is set,
    columns that look like row identifiers are dropped instead of modelled.
    """
    if target not in df.columns:
        raise ValueError(f"Target column {target!r} not found in DataFrame columns")

    if task_type is None:
        task_type = _infer_task_type(df[target])

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # Heuristic: Some numericals might be categorical (low cardinality integers)
    for col in num_cols.copy():
        if col == target:
            continue
        if df[col].nunique() < 10 and pd.api.types.is_integer_dtype(df[col]):
            cat_cols.append(col)
            num_cols.remove(col)

    if target in num_cols:
        num_cols.remove(target)
    if target in cat_cols:
        cat_cols.remove(target)

    # Opt-in: route id-like columns out of the feature lists so they fall into
    # `dropped` below rather than being modelled.
    if drop_ids:
        n_rows = len(df)
        id_like = {
            c for c in df.columns
            if c != target and _is_id_like(df[c], c, n_rows)
        }
        num_cols = [c for c in num_cols if c not in id_like]
        cat_cols = [c for c in cat_cols if c not in id_like]

    # Dedup and preserve order relative to df.columns
    num_cols = [c for c in df.columns if c in num_cols]
    cat_cols = [c for c in df.columns if c in cat_cols]

    # Columns whose dtype is neither numeric nor categorical (e.g. datetime,
    # timedelta), plus any id-like columns removed above, can't be modelled, so
    # drop them. A valid DatasetSpec partitions column_names into
    # {target} + numerical + categorical, so dropped columns must also be
    # excluded from column_names.
    classified = set(num_cols) | set(cat_cols) | {target}
    dropped = [c for c in df.columns if c not in classified]
    column_names = [c for c in df.columns if c not in dropped]

    return DatasetSpec(
        name=name,
        task_type=task_type,
        target=target,
        column_names=column_names,
        numerical=num_cols,
        categorical=cat_cols,
        dropped=dropped,
        default_naming="real",
        source={"url": "custom"}
    )
