from __future__ import annotations

import os
import zipfile
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen


DEFAULT_CACHE = Path(os.environ.get("CSDATA_CACHE", Path.home() / ".cache" / "csdata"))


def fetch(url: str, name: str, cache_dir: Path | str | None = None) -> str:
    dataset_cache = Path(cache_dir or DEFAULT_CACHE) / name
    dataset_cache.mkdir(parents=True, exist_ok=True)

    marker = dataset_cache / "raw.csv"
    if marker.exists():
        return str(marker)

    filename = Path(urlparse(url).path).name or "raw"
    raw = dataset_cache / filename
    tmp = raw.with_name(raw.name + ".part")

    with urlopen(url) as response, tmp.open("wb") as handle:
        handle.write(response.read())

    if raw.suffix == ".zip":
        tmp.replace(raw)
        with zipfile.ZipFile(raw) as archive:
            csv_names = [item for item in archive.namelist() if item.lower().endswith(".csv")]
            if not csv_names:
                raise ValueError(f"no csv inside {url}")
            with archive.open(csv_names[0]) as src, marker.open("wb") as dst:
                dst.write(src.read())
    else:
        tmp.replace(marker)

    return str(marker)
