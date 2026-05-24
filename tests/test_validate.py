import pandas as pd

from csdata.registry import load_spec
from csdata.render import render_name
from csdata.validate import validate


def _adult_df():
    return pd.DataFrame({
        "age": [25, 40],
        "workclass": ["P", "G"],
        "fnlwgt": [1, 2],
        "education": ["B", "M"],
        "education-num": [9, 13],
        "marital-status": ["S", "M"],
        "occupation": ["X", "Y"],
        "relationship": ["H", "W"],
        "race": ["A", "B"],
        "sex": ["M", "F"],
        "capital-gain": [0, 1],
        "capital-loss": [0, 0],
        "hours-per-week": [40, 50],
        "native-country": ["US", "IN"],
        "income": ["<=50K", ">50K"],
    })


def test_validate_clean_when_info_matches_spec_and_df():
    spec = load_spec("adult")
    info = render_name(spec, naming="real")
    assert validate(info, spec=spec, df=_adult_df(), schema="name") == []


def test_validate_flags_numeric_column_typed_as_categorical():
    spec = load_spec("adult")
    info = render_name(spec, naming="real")
    info["numerical_columns"].remove("age")
    info["categorical_columns"].append("age")
    msgs = validate(info, spec=spec, df=_adult_df(), schema="name")
    assert any("age" in msg for msg in msgs)


def test_validate_flags_dtype_mismatch_against_csv():
    spec = load_spec("adult")
    info = render_name(spec, naming="real")
    df = _adult_df()
    df["age"] = df["age"].astype(str)
    msgs = validate(info, spec=spec, df=df, schema="name")
    assert any("age" in msg and "dtype" in msg.lower() for msg in msgs)
