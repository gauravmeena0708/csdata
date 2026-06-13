"""csdata: single-source dataset onboarding."""
from csdata.spec import DatasetSpec
from csdata.registry import load_spec, list_specs
from csdata.render import render_name, render_idx
from csdata.validate import validate
from csdata.prepare import prepare
from csdata.infer import infer_spec, parse_date_columns
from csdata.flags import flag_columns, ColumnFlags

__version__ = "1.0.1"
__all__ = [
    "DatasetSpec", "load_spec", "list_specs",
    "render_name", "render_idx", "validate", "prepare",
    "infer_spec", "parse_date_columns",
    "flag_columns", "ColumnFlags",
]
