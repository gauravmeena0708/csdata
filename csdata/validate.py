from __future__ import annotations

from typing import List

import pandas as pd
from pandas.api.types import is_numeric_dtype

from csdata.render import render_idx, render_name
from csdata.spec import DatasetSpec


def validate(
    info: dict,
    spec: DatasetSpec,
    df: pd.DataFrame | None = None,
    schema: str = "name",
    naming: str | None = None,
) -> List[str]:
    msgs: list[str] = []
    if schema == "name":
        expected = render_name(spec, naming)
        target_key = "target_column"
    elif schema == "idx":
        expected = render_idx(spec, naming)
        target_key = "target_col_idx"
    else:
        raise ValueError(f"bad schema {schema!r}")

    for key in ("numerical_columns", "categorical_columns"):
        actual_cols = set(info.get(key, []))
        expected_cols = set(expected[key])
        if actual_cols != expected_cols:
            msgs.append(
                f"{key} mismatch vs spec: "
                f"+{actual_cols - expected_cols} "
                f"-{expected_cols - actual_cols}"
            )

    if info.get(target_key) != expected[target_key]:
        msgs.append(f"{target_key} mismatch: {info.get(target_key)} != {expected[target_key]}")

    if df is not None:
        for col in info.get("numerical_columns", []):
            if col in df.columns and not is_numeric_dtype(df[col]):
                msgs.append(
                    f"column {col!r} declared numerical but CSV dtype is "
                    f"{df[col].dtype} (dtype mismatch)"
                )
    return msgs
