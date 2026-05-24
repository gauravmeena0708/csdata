import json
import importlib
from pathlib import Path

import pandas as pd

from csdata.download import fetch_health_heritage
from csdata.health_heritage import build_health_heritage_frame
from csdata.prepare import prepare


def _write_health_heritage_raw(root: Path) -> dict[str, str]:
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "Year": [1, 1],
        "MemberID": [101, 102],
        "ProviderID": ["p1", "p2"],
        "Vendor": ["v1", "v2"],
        "PCP": ["pcp1", "pcp2"],
        "PayDelay": ["162+", "5"],
        "DSFS": ["0- 1 month", "1- 2 months"],
        "CharlsonIndex": ["3-4", "0"],
        "LengthOfStay": ["1 day", "2 days"],
        "PrimaryConditionGroup": ["AMI", "COPD"],
        "Specialty": ["Other", "Surgery"],
        "ProcedureGroup": ["EM", "RAD"],
        "PlaceSvc": ["Office", "Home"],
    }).to_csv(root / "Claims.csv", index=False)
    pd.DataFrame({
        "Year": [1, 1],
        "MemberID": [101, 102],
        "DSFS": ["0- 1 month", "1- 2 months"],
        "DrugCount": ["2+", "1"],
    }).to_csv(root / "DrugCount.csv", index=False)
    pd.DataFrame({
        "Year": [1, 1],
        "MemberID": [101, 102],
        "DSFS": ["0- 1 month", "1- 2 months"],
        "LabCount": ["3+", "1"],
    }).to_csv(root / "LabCount.csv", index=False)
    pd.DataFrame({
        "MemberID": [101, 102],
        "AgeAtFirstClaim": ["60-69", "30-39"],
        "Sex": ["F", "M"],
    }).to_csv(root / "Members.csv", index=False)
    return {path.name: str(path) for path in root.glob("*.csv")}


def test_fetch_health_heritage_downloads_four_local_files(tmp_path):
    source = tmp_path / "source"
    _write_health_heritage_raw(source)
    paths = fetch_health_heritage(source.as_uri(), cache_dir=tmp_path / "cache")
    assert set(paths) == {"Claims.csv", "DrugCount.csv", "LabCount.csv", "Members.csv"}
    assert all(Path(path).exists() for path in paths.values())


def test_build_health_heritage_frame_matches_csdata_spec_columns(tmp_path):
    paths = _write_health_heritage_raw(tmp_path / "raw")
    df = build_health_heritage_frame(paths)
    assert list(df.columns) == [
        "LabCount_total",
        "LabCount_months",
        "DrugCount_total",
        "DrugCount_months",
        "no_Claims",
        "no_Providers",
        "no_Vendors",
        "no_PCPs",
        "PayDelay_total",
        "PayDelay_max",
        "PayDelay_min",
        "PrimaryConditionGroup",
        "Specialty",
        "ProcedureGroup",
        "PlaceSvc",
        "AgeAtFirstClaim",
        "Sex",
        "max_CharlsonIndex",
    ]
    assert pd.api.types.is_numeric_dtype(df["no_PCPs"])
    assert set(df["max_CharlsonIndex"]) == {"=0", ">0"}


def test_prepare_health_heritage_assembles_without_raw_df(tmp_path, monkeypatch):
    paths = _write_health_heritage_raw(tmp_path / "raw")

    prepare_module = importlib.import_module("csdata.prepare")

    monkeypatch.setattr(
        prepare_module,
        "fetch_health_heritage",
        lambda url, name, cache_dir: paths,
    )
    out = Path(prepare("health_heritage", tmp_path / "out", schema="name", naming="real"))
    info = json.loads((out / "info.json").read_text())
    assert (out / "train.csv").exists()
    assert info["target_column"] == "max_CharlsonIndex"
    assert "no_PCPs" in info["numerical_columns"]
