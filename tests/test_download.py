from pathlib import Path

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
