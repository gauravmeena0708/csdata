from csdata.registry import load_spec
from csdata.render import render_name


def test_render_name_adult_folds_target_into_categorical():
    info = render_name(load_spec("adult"), naming="real")
    assert info["target_column"] == "income"
    assert "income" in info["categorical_columns"]
    assert "income" not in info["numerical_columns"]
    assert info["numerical_columns"][0] == "age"


def test_render_name_beijing_regression_folds_target_into_numerical():
    info = render_name(load_spec("beijing"), naming="anonymized")
    assert info["target_column"] == "label"
    assert "label" in info["numerical_columns"]
    assert "label" not in info["categorical_columns"]


def test_render_name_anonymized_uses_generic_names_for_adult():
    info = render_name(load_spec("adult"), naming="anonymized")
    assert info["target_column"] == "label"
    assert info["numerical_columns"][0] == "col0"
    assert "label" in info["categorical_columns"]
