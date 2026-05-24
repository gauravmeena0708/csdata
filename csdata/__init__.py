"""csdata: single-source dataset onboarding."""
from csdata.spec import DatasetSpec
from csdata.registry import load_spec, list_specs
from csdata.render import render_name, render_idx
from csdata.validate import validate
from csdata.prepare import prepare

__version__ = "0.1.0"
__all__ = [
    "DatasetSpec", "load_spec", "list_specs",
    "render_name", "render_idx", "validate", "prepare",
]
