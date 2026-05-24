import pandas as pd
import pytest

from csdata.transforms import apply_transform, has_transform


def test_default_transform_renames_and_prefixes():
    # 25 columns: ID + 23 features + target (numeric), like raw default.
    raw = pd.DataFrame({
        **{"ID": [1, 2]},
        **{f"x{i}": [1, 2] for i in range(23)},
        "Y": [0, 1],
    })
    out = apply_transform("default", raw)
    assert "ID" not in out.columns
    assert list(out.columns)[-1] == "label"
    assert out["col1"].iloc[0].startswith("S")
    assert out["label"].iloc[0].startswith("L")


def test_telco_drops_id_and_coerces_totalcharges():
    raw = pd.DataFrame({
        "customerID": ["a", "b"],
        "gender": ["M", "F"],
        "TotalCharges": ["10.5", " "],
        "SeniorCitizen": [0, 1],
        "Churn": ["No", "Yes"],
    })
    out = apply_transform("telco_churn", raw)
    assert "customerID" not in out.columns
    assert pd.isna(out["TotalCharges"].iloc[1])
    assert out["SeniorCitizen"].dtype == object


def test_has_transform_for_every_spec():
    for name in (
        "adult",
        "default",
        "beijing",
        "shoppers",
        "german",
        "bank_marketing",
        "telco_churn",
        "compas",
        "law_school",
        "health_heritage",
    ):
        assert has_transform(name), name


def test_unknown_transform_raises_keyerror():
    with pytest.raises(KeyError):
        apply_transform("missing", pd.DataFrame({"x": [1]}))
