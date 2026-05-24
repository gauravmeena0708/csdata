import json
from pathlib import Path

import pandas as pd

from csdata.prepare import prepare


def _raw_adult(n=20):
    return pd.DataFrame({
        "age": list(range(20, 20 + n)),
        "workclass": ["P", "G"] * (n // 2),
        "fnlwgt": list(range(n)),
        "education": ["B", "M"] * (n // 2),
        "education-num": [9, 13] * (n // 2),
        "marital-status": ["S", "M"] * (n // 2),
        "occupation": ["X", "Y"] * (n // 2),
        "relationship": ["H", "W"] * (n // 2),
        "race": ["A", "B"] * (n // 2),
        "sex": ["M", "F"] * (n // 2),
        "capital-gain": [0] * n,
        "capital-loss": [0] * n,
        "hours-per-week": [40] * n,
        "native-country": ["US", "IN"] * (n // 2),
        "income": ["<=50K", ">50K"] * (n // 2),
    })


def test_prepare_name_schema_writes_artifacts(tmp_path):
    out = prepare(
        "adult",
        out_dir=tmp_path / "adult",
        schema="name",
        naming="real",
        raw_df=_raw_adult(),
    )
    out = Path(out)
    assert (out / "train.csv").exists()
    assert (out / "test.csv").exists()
    assert (out / "full.csv").exists()
    info = json.loads((out / "info.json").read_text())
    assert info["target_column"] == "income"
    assert "income" in info["categorical_columns"]


def test_prepare_validates_clean(tmp_path, capsys):
    prepare("adult", out_dir=tmp_path / "adult", schema="name", naming="real", raw_df=_raw_adult())
    assert "mismatch" not in capsys.readouterr().out


def test_prepare_idx_schema_emits_idx_fields(tmp_path):
    out = Path(
        prepare(
            "adult",
            out_dir=tmp_path / "a",
            schema="idx",
            naming="real",
            raw_df=_raw_adult(),
        )
    )
    info = json.loads((out / "info.json").read_text())
    assert "num_col_idx" in info and "column_info" in info
