from __future__ import annotations
from importlib import resources
from typing import List
import yaml
from csdata.spec import DatasetSpec

_SPECS_PKG = "csdata.specs"


def list_specs() -> List[str]:
    out = []
    for entry in resources.files(_SPECS_PKG).iterdir():
        if entry.name.endswith(".yaml"):
            out.append(entry.name[:-len(".yaml")])
    return sorted(out)


def load_spec(name: str) -> DatasetSpec:
    res = resources.files(_SPECS_PKG).joinpath(f"{name}.yaml")
    if not res.is_file():
        raise KeyError(f"no spec for dataset {name!r}")
    data = yaml.safe_load(res.read_text(encoding="utf-8"))
    return DatasetSpec.from_dict(data)
