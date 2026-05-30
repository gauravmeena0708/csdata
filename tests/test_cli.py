import json
from pathlib import Path

import pandas as pd

from csdata.cli import main


def test_cli_prepare_csv_writes_artifacts(tmp_path):
    csv = tmp_path / "raw.csv"
    pd.DataFrame({
        "age": list(range(20, 60)),
        "income": [float(i) for i in range(40)],
        "city": ["a", "b"] * 20,
        "y": [0, 1] * 20,
    }).to_csv(csv, index=False)
    out = tmp_path / "out"

    rc = main(["prepare-csv", str(csv), "--target", "y", "--out", str(out), "--schema", "idx"])

    assert rc == 0
    for f in ("full.csv", "train.csv", "test.csv", "info.json"):
        assert (Path(out) / f).exists(), f"missing artifact {f}"
    info = json.loads((Path(out) / "info.json").read_text())
    assert info["target_col_idx"] == [3]


def test_cli_prepare_csv_infers_task_type_and_drops_ids(tmp_path):
    csv = tmp_path / "raw.csv"
    pd.DataFrame({
        "user_id": list(range(40)),
        "amount": [float(i) for i in range(40)],
        "y": [0, 1, 2, 3] * 10,
    }).to_csv(csv, index=False)
    out = tmp_path / "out"

    rc = main([
        "prepare-csv", str(csv), "--target", "y",
        "--out", str(out), "--schema", "idx", "--drop-ids",
    ])

    assert rc == 0
    info = json.loads((Path(out) / "info.json").read_text())
    assert info["task_type"] == "multiclass"          # inferred, not the binclass default
    full = pd.read_csv(Path(out) / "full.csv")
    assert "user_id" not in full.columns              # dropped via --drop-ids


def test_cli_prepare_csv_bad_target_errors(tmp_path):
    csv = tmp_path / "raw.csv"
    pd.DataFrame({"a": [1, 2], "y": [0, 1]}).to_csv(csv, index=False)
    rc = main(["prepare-csv", str(csv), "--target", "nope", "--out", str(tmp_path / "o")])
    assert rc != 0


def test_cli_list(capsys):
    rc = main(["list"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "adult" in out and "compas" in out


def test_cli_prepare_unknown_dataset_errors():
    rc = main(["prepare", "nope", "--out", "/tmp/x"])
    assert rc != 0
