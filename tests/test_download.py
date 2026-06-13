from pathlib import Path
import zipfile

import pandas as pd

from csdata.download import fetch


def test_fetch_local_csv(tmp_path):
    src = tmp_path / "raw.csv"
    pd.DataFrame({"a": [1], "b": [2]}).to_csv(src, index=False)
    cache = tmp_path / "cache"
    out = fetch(url=src.as_uri(), name="toy", cache_dir=cache)
    assert Path(out).exists()
    assert pd.read_csv(out).shape == (1, 2)


def test_fetch_is_cached(tmp_path):
    src = tmp_path / "raw.csv"
    pd.DataFrame({"a": [1]}).to_csv(src, index=False)
    cache = tmp_path / "cache"
    first = fetch(url=src.as_uri(), name="toy", cache_dir=cache)
    second = fetch(url="http://unused", name="toy", cache_dir=cache)
    assert first == second


def test_fetch_extracts_named_non_csv_zip_member(tmp_path):
    archive = tmp_path / "raw.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("toy.data", "1 2\n3 4\n")
    out = fetch(url=archive.as_uri(), name="toy_zip", cache_dir=tmp_path / "cache", archive_member="toy.data")
    assert Path(out).suffix == ".data"
    assert Path(out).read_text() == "1 2\n3 4\n"


def _nested_zip_bytes(inner_name: str, payload: str) -> bytes:
    import io

    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as zf:
        zf.writestr(inner_name, payload)
    return inner.getvalue()


def test_fetch_extracts_member_from_nested_zip(tmp_path):
    # Mirrors UCI bank_marketing: outer zip holds only inner zips, the CSV lives
    # inside one of them. archive_member is matched inside the nested archive.
    archive = tmp_path / "raw.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("bank.zip", _nested_zip_bytes("bank-full.csv", "a;b\n1;2\n"))
        zf.writestr("bank-additional.zip", _nested_zip_bytes("bank-additional.csv", "x;y\n9;9\n"))
    out = fetch(
        url=archive.as_uri(), name="nested_zip",
        cache_dir=tmp_path / "cache", archive_member="bank-full.csv",
    )
    assert Path(out).suffix == ".csv"
    assert pd.read_csv(out, sep=";").to_dict("records") == [{"a": 1, "b": 2}]


def test_fetch_nested_zip_without_member_picks_first_tabular(tmp_path):
    archive = tmp_path / "raw.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("inner.zip", _nested_zip_bytes("only.csv", "a\n1\n"))
    out = fetch(url=archive.as_uri(), name="nested_anon", cache_dir=tmp_path / "cache")
    assert Path(out).read_text() == "a\n1\n"
