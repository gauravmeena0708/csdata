from __future__ import annotations

from typing import Dict, List, Tuple

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


def render_idx(*args, **kwargs):
    raise NotImplementedError
