import json
import importlib
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


def test_prepare_passes_read_csv_opts(tmp_path, monkeypatch):
    # A semicolon-separated raw file must be parsed using source.read_csv opts.
    prep = importlib.import_module("csdata.prepare")
    raw = tmp_path / "raw.csv"
    raw.write_text(
        "fixed acidity;volatile acidity;citric acid;residual sugar;chlorides;"
        "free sulfur dioxide;total sulfur dioxide;density;pH;sulphates;alcohol;quality\n"
        "7.4;0.7;0.0;1.9;0.076;11;34;0.9978;3.51;0.56;9.4;5\n"
        "7.8;0.88;0.0;2.6;0.098;25;67;0.9968;3.20;0.68;9.8;5\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        prep,
        "fetch",
        lambda url, name, cache_dir=None, archive_member=None: str(raw),
    )
    out = Path(prep.prepare("wine_quality_red", out_dir=tmp_path / "w", schema="idx", naming="real"))
    full = pd.read_csv(out / "full.csv")
    assert "quality" in full.columns
    assert full.shape[1] == 12  # parsed by ';', not a single column


def _raw_heart(n=20):
    df = pd.DataFrame({i: [float(i)] * n for i in range(13)})
    df[13] = [0, 2] * (n // 2)
    return df


def test_prepare_heart_idx_binclass(tmp_path):
    out = Path(
        prepare(
            "heart",
            out_dir=tmp_path / "h",
            schema="idx",
            naming="real",
            raw_df=_raw_heart(),
        )
    )
    info = json.loads((out / "info.json").read_text())
    assert info["task_type"] == "binclass"
    assert info["column_names"][-1] == "target"
    train = pd.read_csv(out / "train.csv")
    assert set(train["target"].unique()) <= {0, 1}


def _raw_wine(n=20):
    cols = [
        "fixed acidity",
        "volatile acidity",
        "citric acid",
        "residual sugar",
        "chlorides",
        "free sulfur dioxide",
        "total sulfur dioxide",
        "density",
        "pH",
        "sulphates",
        "alcohol",
    ]
    df = pd.DataFrame({c: [1.0] * n for c in cols})
    df["quality"] = [5, 6] * (n // 2)
    return df


def test_prepare_wine_idx_multiclass(tmp_path):
    out = Path(
        prepare(
            "wine_quality_red",
            out_dir=tmp_path / "w2",
            schema="idx",
            naming="real",
            raw_df=_raw_wine(),
        )
    )
    info = json.loads((out / "info.json").read_text())
    assert info["task_type"] == "multiclass"
    assert info["target_col_idx"] == [11]
    assert len(info["num_col_idx"]) == 11
