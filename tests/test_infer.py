import json
from pathlib import Path

import numpy as np
import pandas as pd

from csdata.infer import infer_spec
from csdata.prepare import prepare


def _frame(n=40):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "age": rng.integers(20, 70, n),
        "income": rng.normal(5e4, 1e4, n),
        "city": rng.choice(["a", "b", "c"], n),
        "y": rng.integers(0, 2, n),
    })


def test_infer_then_prepare_writes_artifacts(tmp_path):
    df = _frame()
    spec = infer_spec(df, target="y", name="custom")
    out = Path(prepare("custom", out_dir=tmp_path / "c", raw_df=df, spec=spec, schema="idx"))

    for f in ("full.csv", "train.csv", "test.csv", "info.json"):
        assert (out / f).exists(), f"missing artifact {f}"
    info = json.loads((out / "info.json").read_text())
    assert info["target_col_idx"] == [3]
    full = pd.read_csv(out / "full.csv")
    assert list(full.columns) == ["age", "income", "city", "y"]


def test_infer_drops_unsupported_dtype_columns():
    df = pd.DataFrame({
        "t": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"]),
        "income": [1.0, 2.0, 3.0, 4.0],
        "y": [0, 1, 0, 1],
    })
    spec = infer_spec(df, target="y")
    assert spec.dropped == ["t"]
    assert "t" not in spec.column_names
    assert spec.column_names == ["income", "y"]


def test_infer_then_prepare_with_dropped_column(tmp_path):
    df = pd.DataFrame({
        "t": pd.to_datetime(["2020-01-01", "2020-01-02"] * 20),
        "income": list(range(40)),
        "y": [0, 1] * 20,
    })
    spec = infer_spec(df, target="y", name="custom")
    out = Path(prepare("custom", out_dir=tmp_path / "d", raw_df=df, spec=spec, schema="name"))
    full = pd.read_csv(out / "full.csv")
    assert "t" not in full.columns


# --- (1) prepare logs dropped columns instead of dropping them silently ---

def test_prepare_logs_dropped_columns(tmp_path, capsys):
    df = pd.DataFrame({
        "t": pd.to_datetime(["2020-01-01", "2020-01-02"] * 20),
        "income": list(range(40)),
        "y": [0, 1] * 20,
    })
    spec = infer_spec(df, target="y", name="custom")
    prepare("custom", out_dir=tmp_path / "d", raw_df=df, spec=spec)
    out = capsys.readouterr().out
    assert "dropped" in out and "t" in out


# --- (3) task_type is inferred from the target when not given ---

def test_infer_task_type_binclass_for_binary_target():
    df = pd.DataFrame({"x": [1.0, 2, 3, 4], "y": [0, 1, 0, 1]})
    assert infer_spec(df, target="y").task_type == "binclass"


def test_infer_task_type_multiclass_for_few_int_classes():
    df = pd.DataFrame({"x": list(range(12)), "y": [0, 1, 2, 3] * 3})
    assert infer_spec(df, target="y").task_type == "multiclass"


def test_infer_task_type_regression_for_continuous_target():
    df = pd.DataFrame({"x": list(range(50)), "y": np.linspace(0.0, 1.0, 50)})
    assert infer_spec(df, target="y").task_type == "regression"


def test_infer_task_type_explicit_overrides():
    df = pd.DataFrame({"x": [1.0, 2, 3, 4], "y": [0, 1, 2, 3]})
    assert infer_spec(df, target="y", task_type="regression").task_type == "regression"


# --- (2) ID detection behind an opt-in flag ---

def test_ids_kept_as_features_by_default():
    df = pd.DataFrame({"id_int": list(range(40)), "y": [0, 1] * 20})
    spec = infer_spec(df, target="y")
    assert spec.dropped == []
    assert "id_int" in spec.numerical


def test_drop_ids_flag_routes_id_columns_to_dropped():
    df = pd.DataFrame({
        "id_int": list(range(40)),                  # unique-per-row int
        "ID": [f"U{i}" for i in range(40)],          # unique-per-row str, id name
        "amount": [float(i) for i in range(40)],     # unique float -> kept
        "y": [0, 1] * 20,
    })
    spec = infer_spec(df, target="y", drop_ids=True)
    assert set(spec.dropped) == {"id_int", "ID"}
    assert "id_int" not in spec.column_names and "ID" not in spec.column_names
    assert spec.numerical == ["amount"]
