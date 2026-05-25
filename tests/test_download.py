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
