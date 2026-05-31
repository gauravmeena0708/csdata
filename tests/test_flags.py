import pandas as pd

from csdata.flags import ColumnFlags, flag_columns


def test_datetime_by_name():
    df = pd.DataFrame({"date": ["a", "b"], "year": [1, 2], "name": ["x", "y"]})
    flags = flag_columns(df)
    assert "date" in flags.datetime
    assert "year" in flags.datetime
    assert "name" not in flags.datetime


def test_datetime_name_boundary_guard():
    # datetime token must be FIRST or LAST token; these must NOT be flagged.
    df = pd.DataFrame({"two_year_recid": [0, 1], "hours_per_week": [40, 38]})
    flags = flag_columns(df)
    assert flags.datetime == []


def test_datetime_by_value_regex():
    df = pd.DataFrame({"col": ["2020-01-01", "2021-06-15", "2022-12-31"]})
    flags = flag_columns(df)
    assert "col" in flags.datetime


def test_serial_by_name():
    df = pd.DataFrame({"id": [1, 2], "user_id": [3, 4], "row_num": [5, 6], "age": [20, 30]})
    flags = flag_columns(df)
    assert "id" in flags.serial
    assert "user_id" in flags.serial
    assert "row_num" in flags.serial
    assert "age" not in flags.serial


def test_serial_by_sequential_values_requires_numerical_list():
    df = pd.DataFrame({"seq": [1, 2, 3, 4, 5], "amount": [10, 90, 30, 70, 50]})
    flags = flag_columns(df, numerical_columns=["seq", "amount"])
    assert "seq" in flags.serial
    assert "amount" not in flags.serial
    flags_noinfo = flag_columns(df, numerical_columns=None)
    assert "seq" not in flags_noinfo.serial


def test_protected_by_name():
    df = pd.DataFrame({"sex": ["M", "F"], "race": ["a", "b"], "salary": [1, 2]})
    flags = flag_columns(df)
    assert "sex" in flags.protected
    assert "race" in flags.protected
    assert "salary" not in flags.protected


def test_empty_dataframe():
    flags = flag_columns(pd.DataFrame())
    assert flags == ColumnFlags(datetime=[], serial=[], protected=[])
