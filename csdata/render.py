from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd

from csdata.spec import DatasetSpec


def _alias_map(spec: DatasetSpec) -> Dict[str, str]:
    """Map real column names to col0..colN, with the target as label."""
    out: dict[str, str] = {}
    i = 0
    for name in spec.column_names:
        if name == spec.target:
            out[name] = "label"
        else:
            out[name] = f"col{i}"
            i += 1
    return out


def _names(spec: DatasetSpec, naming: str) -> Tuple[List[str], List[str], str]:
    if naming == "real":
        return list(spec.numerical), list(spec.categorical), spec.target
    if naming == "anonymized":
        aliases = _alias_map(spec)
        return (
            [aliases[col] for col in spec.numerical],
            [aliases[col] for col in spec.categorical],
            aliases[spec.target],
        )
    raise ValueError(f"bad naming {naming!r}")


def render_name(spec: DatasetSpec, naming: str | None = None) -> dict:
    naming = naming or spec.default_naming
    num, cat, target = _names(spec, naming)
    if spec.task_type == "regression":
        numerical_columns = num + [target]
        categorical_columns = list(cat)
    else:
        numerical_columns = list(num)
        categorical_columns = cat + [target]
    return {
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
        "target_column": target,
    }


def _ordered_names(spec: DatasetSpec, naming: str) -> List[str]:
    if naming == "real":
        return list(spec.column_names)
    if naming == "anonymized":
        aliases = _alias_map(spec)
        return [aliases[col] for col in spec.column_names]
    raise ValueError(f"bad naming {naming!r}")


def render_idx(spec: DatasetSpec, naming: str | None = None, df: pd.DataFrame | None = None) -> dict:
    naming = naming or spec.default_naming
    column_names = _ordered_names(spec, naming)
    num, cat, target = _names(spec, naming)
    pos = {name: i for i, name in enumerate(column_names)}

    num_col_idx = [pos[col] for col in num]
    cat_col_idx = [pos[col] for col in cat]
    target_col_idx = [pos[target]]

    idx_mapping: dict[int, int] = {}
    curr_num = 0
    curr_cat = len(num_col_idx)
    curr_target = curr_cat + len(cat_col_idx)
    for idx in range(len(column_names)):
        if idx in num_col_idx:
            idx_mapping[idx] = curr_num
            curr_num += 1
        elif idx in cat_col_idx:
            idx_mapping[idx] = curr_cat
            curr_cat += 1
        else:
            idx_mapping[idx] = curr_target
            curr_target += 1

    inverse_idx_mapping = {value: key for key, value in idx_mapping.items()}
    idx_name_mapping = {idx: column_names[idx] for idx in range(len(column_names))}

    info = {
        "column_names": column_names,
        "num_col_idx": num_col_idx,
        "cat_col_idx": cat_col_idx,
        "target_col_idx": target_col_idx,
        "task_type": spec.task_type,
        "numerical_columns": list(num),
        "categorical_columns": list(cat),
        "idx_mapping": idx_mapping,
        "inverse_idx_mapping": inverse_idx_mapping,
        "idx_name_mapping": idx_name_mapping,
    }
    if df is not None:
        info["column_info"] = _column_info(spec, df, naming, num, cat, target)
    return info


def _column_info(
    spec: DatasetSpec,
    df: pd.DataFrame,
    naming: str,
    num: List[str],
    cat: List[str],
    target: str,
) -> dict:
    aliases = _alias_map(spec) if naming == "anonymized" else {col: col for col in spec.column_names}
    real_by_rendered = {rendered: real for real, rendered in aliases.items()}

    column_info = {}
    for col in num:
        source_col = _source_column(df, col, real_by_rendered)
        series = df[source_col]
        column_info[col] = {
            "type": "numerical",
            "max": float(series.max()),
            "min": float(series.min()),
        }
    for col in cat:
        source_col = _source_column(df, col, real_by_rendered)
        series = df[source_col]
        column_info[col] = {
            "type": "categorical",
            "categories": sorted(set(series.astype(str))),
        }

    target_source_col = _source_column(df, target, real_by_rendered)
    target_series = df[target_source_col]
    if spec.task_type == "regression":
        column_info[target] = {
            "type": "numerical",
            "max": float(target_series.max()),
            "min": float(target_series.min()),
        }
    else:
        column_info[target] = {
            "type": "categorical",
            "categories": sorted(set(target_series.astype(str))),
        }
    return column_info


def _source_column(df: pd.DataFrame, rendered_col: str, real_by_rendered: dict[str, str]) -> str:
    if rendered_col in df.columns:
        return rendered_col
    return real_by_rendered.get(rendered_col, rendered_col)
