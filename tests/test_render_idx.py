import pandas as pd

from csdata.registry import load_spec
from csdata.render import render_idx


def test_render_idx_adult_indices_match_columns():
    info = render_idx(load_spec("adult"), naming="real")
    assert info["task_type"] == "binclass"
    assert info["column_names"][info["target_col_idx"][0]] == "income"

    column_names = info["column_names"]
    assert [column_names[i] for i in info["num_col_idx"]] == info["numerical_columns"]
    assert [column_names[i] for i in info["cat_col_idx"]] == info["categorical_columns"]
    assert sorted(info["idx_mapping"].values()) == list(range(len(column_names)))


def test_render_idx_beijing_anonymized_regression():
    info = render_idx(load_spec("beijing"), naming="anonymized")
    assert info["task_type"] == "regression"
    assert info["column_names"][-1] == "label"
    assert info["target_col_idx"] == [11]


def test_render_idx_column_info_from_dataframe():
    spec = load_spec("compas")
    df = pd.DataFrame({
        "age": [25, 40],
        "sex": ["M", "F"],
        "race": ["A", "B"],
        "diff_custody": [1.0, 2.0],
        "diff_jail": [3.0, 4.0],
        "priors_count": [0, 5],
        "c_charge_degree": ["F", "M"],
        "v_score_text": ["Low", "High"],
        "two_year_recid": [0, 1],
    })[spec.column_names]
    info = render_idx(spec, naming="real", df=df)
    column_info = info["column_info"]
    assert column_info["age"]["type"] == "numerical"
    assert column_info["age"]["max"] == 40.0
    assert column_info["sex"]["type"] == "categorical"
    assert set(column_info["sex"]["categories"]) == {"M", "F"}
