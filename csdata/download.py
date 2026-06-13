from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path
from urllib.parse import quote, urljoin, urlparse
from urllib.request import urlopen


DEFAULT_CACHE = Path(os.environ.get("CSDATA_CACHE", Path.home() / ".cache" / "csdata"))
TABULAR_SUFFIXES = {".csv", ".data", ".xls", ".xlsx"}
HEALTH_HERITAGE_FILES = ("Claims.csv", "DrugCount.csv", "LabCount.csv", "Members.csv")
HEALTH_HERITAGE_BASE_URLS = (
    "https://files.sri.inf.ethz.ch/tableak/Health_Heritage",
    "http://files.sri.inf.ethz.ch/tableak/Health_Heritage",
)


def fetch(
    url: str,
    name: str,
    cache_dir: Path | str | None = None,
    archive_member: str | None = None,
) -> str:
    dataset_cache = Path(cache_dir or DEFAULT_CACHE) / name
    dataset_cache.mkdir(parents=True, exist_ok=True)

    if archive_member:
        marker = dataset_cache / f"raw{Path(archive_member).suffix or '.data'}"
        if marker.exists():
            return str(marker)
    else:
        cached = sorted(dataset_cache.glob("raw.*"))
        if cached:
            return str(cached[0])

    filename = Path(urlparse(url).path).name or "raw"
    raw = dataset_cache / filename
    tmp = raw.with_name(raw.name + ".part")

    if not raw.exists():
        with urlopen(url) as response, tmp.open("wb") as handle:
            handle.write(response.read())
        tmp.replace(raw)

    if raw.suffix == ".zip":
        member, payload = _read_zip_member(raw.read_bytes(), archive_member, url)
        marker = dataset_cache / f"raw{Path(member).suffix or '.data'}"
        marker.write_bytes(payload)
    else:
        marker = dataset_cache / f"raw{raw.suffix or '.data'}"
        if raw != marker:
            raw.replace(marker)

    return str(marker)


def _read_zip_member(
    data: bytes, archive_member: str | None, url: str, *, _depth: int = 0
) -> tuple[str, bytes]:
    """Locate a tabular file inside a (possibly nested) zip.

    Returns ``(member_name, member_bytes)``. Some UCI archives wrap their data
    in inner zips — e.g. bank_marketing's ``bank+marketing.zip`` contains
    ``bank.zip`` / ``bank-additional.zip`` rather than a top-level CSV — so when
    no match is found at the current level this recurses into nested ``.zip``
    members. ``archive_member`` is matched by exact name first, then by basename
    at any depth; without it the first tabular file found wins (the current
    level is preferred over nested archives).
    """
    if _depth > 3:
        raise ValueError(f"zip nesting too deep inside {url}")

    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = [n for n in archive.namelist() if not n.endswith("/")]

        if archive_member:
            if archive_member in names:
                return archive_member, archive.read(archive_member)
            target = Path(archive_member).name
            for name in names:
                if Path(name).name == target and Path(name).suffix.lower() in TABULAR_SUFFIXES:
                    return name, archive.read(name)
        else:
            for name in names:
                if Path(name).suffix.lower() in TABULAR_SUFFIXES:
                    return name, archive.read(name)

        # Skip macOS resource-fork cruft (__MACOSX/...) when recursing.
        for name in names:
            if name.lower().endswith(".zip") and not name.startswith("__MACOSX"):
                try:
                    return _read_zip_member(archive.read(name), archive_member, url, _depth=_depth + 1)
                except ValueError:
                    continue

    if archive_member:
        raise ValueError(f"{archive_member!r} not found inside {url}")
    raise ValueError(f"no tabular data file inside {url}")


def fetch_health_heritage(
    url: str | None = None,
    name: str = "health_heritage",
    cache_dir: Path | str | None = None,
) -> dict[str, str]:
    dataset_cache = Path(cache_dir or DEFAULT_CACHE) / name
    dataset_cache.mkdir(parents=True, exist_ok=True)

    base_urls = _health_heritage_base_urls(url)
    paths: dict[str, str] = {}
    for filename in HEALTH_HERITAGE_FILES:
        target = dataset_cache / filename
        if _usable_file(target):
            paths[filename] = str(target)
            continue

        tmp = target.with_suffix(target.suffix + ".part")
        last_error: Exception | None = None
        for base_url in base_urls:
            file_url = urljoin(base_url.rstrip("/") + "/", quote(filename))
            try:
                if tmp.exists():
                    tmp.unlink()
                with urlopen(file_url, timeout=60) as response, tmp.open("wb") as handle:
                    handle.write(response.read())
                if tmp.stat().st_size == 0:
                    raise RuntimeError("downloaded empty file")
                tmp.replace(target)
                if not _usable_file(target):
                    target.unlink(missing_ok=True)
                    raise RuntimeError("downloaded file looks like an error page")
                paths[filename] = str(target)
                break
            except Exception as exc:
                last_error = exc
                if tmp.exists():
                    tmp.unlink()
        else:
            raise RuntimeError(f"could not fetch Health Heritage {filename}: {last_error}")

    return paths


def _health_heritage_base_urls(url: str | None) -> list[str]:
    candidates = [url] if url else []
    if url and url.startswith("https://"):
        candidates.append("http://" + url[len("https://"):])
    candidates.extend(HEALTH_HERITAGE_BASE_URLS)

    out = []
    for candidate in candidates:
        if candidate and candidate not in out:
            out.append(candidate)
    return out


def _usable_file(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    if path.stat().st_size >= 1000:
        return True
    head = path.read_text(encoding="utf-8", errors="ignore")[:500]
    error_markers = ("proxy.cgi Error", "<html", "<!doctype html", "Error")
    return not any(marker in head for marker in error_markers)
