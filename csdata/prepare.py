from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from csdata.download import fetch, fetch_health_heritage
from csdata.health_heritage import build_health_heritage_frame
from csdata.registry import load_spec
from csdata.render import _alias_map, render_idx, render_name
from csdata.transforms import apply_transform
from csdata.validate import validate


def _impute(df: pd.DataFrame, num_cols: list[str], cat_cols: list[str]) -> pd.DataFrame:
    for col in cat_cols:
        if col not in df.columns:
            continue
        if df[col].isnull().any():
            mode = df[col].mode(dropna=True)
            df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else "nan")
        df[col] = df[col].astype(str).str.strip()

    for col in num_cols:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    return df


def prepare(
    name: str,
    out_dir: str | Path,
    schema: str = "name",
    naming: str | None = None,
    raw_df: pd.DataFrame | None = None,
    cache_dir: str | Path | None = None,
) -> str:
    spec = load_spec(name)
    naming = naming or spec.default_naming
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if raw_df is not None:
        df = raw_df.copy()
    elif name == "health_heritage":
        raw_paths = fetch_health_heritage(spec.source["url"], name=name, cache_dir=cache_dir)
        df = build_health_heritage_frame(raw_paths)
    else:
        raw_path = fetch(spec.source["url"], name, cache_dir, spec.source.get("archive_member"))
        read_excel_opts = spec.source.get("read_excel")
        if read_excel_opts is not None or Path(raw_path).suffix.lower() in {".xls", ".xlsx"}:
            df = pd.read_excel(raw_path, **dict(read_excel_opts or {}))
        else:
            df = pd.read_csv(raw_path, **dict(spec.source.get("read_csv", {})))

    df = apply_transform(name, df)
    if naming == "anonymized":
        df = df.rename(columns=_alias_map(spec))
    elif naming != "real":
        raise ValueError(f"bad naming {naming!r}")

    if schema == "name":
        info = render_name(spec, naming)
    elif schema == "idx":
        info = render_idx(spec, naming)
    else:
        raise ValueError(f"bad schema {schema!r}")

    num_cols = info["numerical_columns"]
    cat_cols = list(info.get("categorical_columns", []))
    df = _impute(df, num_cols, cat_cols)

    df.to_csv(out_path / "full.csv", index=False)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
    train_df.to_csv(out_path / "train.csv", index=False)
    test_df.to_csv(out_path / "test.csv", index=False)

    if schema == "idx":
        info = render_idx(spec, naming, df=train_df)
        info["train_num"] = int(train_df.shape[0])
        info["test_num"] = int(test_df.shape[0])

    (out_path / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    for msg in validate(info, spec=spec, df=df, schema=schema, naming=naming):
        print(f"[csdata] drift warning: {msg}")

    return str(out_path)
