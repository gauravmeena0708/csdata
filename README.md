# csdata

**Single source of truth for tabular dataset onboarding** — download, clean,
train/test split, and `info.json` rendering — across the synthetic-data project.

`csdata` exists because dataset column typing (which columns are numerical vs
categorical, which is the target) was previously re-derived by fragile per-repo
heuristics (`select_dtypes` + "low-cardinality integer → categorical"), producing
inconsistent and sometimes wrong `info.json` files. `csdata` replaces those
heuristics with **curated, human-reviewed specs** that every consumer renders
from, so the metadata is correct and identical everywhere.

## Install

```bash
pip install -e .                      # editable, for development
pip install git+https://github.com/gauravmeena0708/csdata.git   # from GitHub
pip install -e ".[dev]"               # + pytest
pip install -e ".[audit]"             # + ucimlrepo (for `csdata audit`)
```

Requires Python ≥ 3.10.

## Quickstart

```python
import csdata

# Inspect the curated spec
spec = csdata.load_spec("adult")
spec.numerical          # ['age', 'fnlwgt', 'education-num', ...]
spec.categorical        # ['workclass', 'education', ...]  (target excluded)
spec.target             # 'income'

# Render the info.json a consumer needs (pass the spec object)
csdata.render_name(spec)                      # parent format (dict)
csdata.render_idx(spec, naming="anonymized")  # TabSyn format (dict)

# Full pipeline: download -> clean -> split -> write train/test/full.csv + info.json -> validate
csdata.prepare("adult", out_dir="out/adult", schema="name", naming="real")
```

CLI:

```bash
csdata list                                                  # the 10 datasets
csdata prepare adult --out out/adult --schema name --naming real
csdata validate adult --dir out/adult --schema name          # drift check vs spec + CSV dtypes
csdata audit adult                                           # cross-check spec vs UCI metadata
```

## The canonical spec

One curated YAML per dataset under `csdata/specs/<dataset>.yaml`, shipped as
package data. The target is kept **separate** from the feature lists, and
`numerical + categorical + target` must exactly partition `column_names`
(enforced by `DatasetSpec`).

```yaml
name: adult
task_type: binclass            # binclass | multiclass | regression
target: income
column_names: [age, workclass, fnlwgt, ...]   # authoritative order
numerical:   [age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week]
categorical: [workclass, education, marital-status, ...]   # target NOT included
dropped:     []                                # IDs / text / leakage columns excluded
default_naming: real           # real | anonymized
source: {url: "https://archive.ics.uci.edu/...", transform: adult}
notes: "fnlwgt is a survey weight; education-num is ordinal kept numerical; ..."
```

## Schemas and naming

Two render schemas × two naming modes are derivable from one spec:

| | `schema="name"` (parent format) | `schema="idx"` (TabSyn format) |
|---|---|---|
| Fields | `numerical_columns`, `categorical_columns` (target **folded in**), `target_column` | `column_names`, `num_col_idx`, `cat_col_idx`, `target_col_idx`, `task_type`, `numerical_columns`/`categorical_columns`, `idx_mapping`, `inverse_idx_mapping`, `idx_name_mapping`; plus `column_info` when a DataFrame is supplied |
| Target | duplicated into the feature list per `task_type` | kept separate in `target_col_idx` |

| `naming="real"` | `naming="anonymized"` |
|---|---|
| real column names (`age`, `workclass`, …) | `col0…colN` for features, `label` for target, with a recoverable alias map |

`default_naming` per dataset chooses the default (e.g. `default`/`beijing` ship
`anonymized` to preserve existing on-disk artifacts; the rest are `real`).

## Components

- **`registry`** — `load_spec(name)` / `list_specs()`; loads and validates packaged YAML.
- **`transforms`** — per-dataset cleaning hooks (drop IDs, coerce dtypes, derive columns, apply headers), registered by name.
- **`download`** — cached fetch of raw data (`.zip`/`.csv`) by URL.
- **`render`** — `render_name` / `render_idx` (× real/anonymized).
- **`validate`** — checks an on-disk `info.json` against the spec and the CSV dtypes; reports drift.
- **`prepare`** — orchestrates download → transform → impute → 80/20 split (seed 42) → render → write → validate.
- **`audit`** — offline `csdata audit` cross-check of specs against UCI metadata (see below).

## Datasets (10)

`adult`, `default`, `bank_marketing`, `shoppers`, `beijing`, `german`, `compas`,
`telco_churn`, `health_heritage`, `law_school`.

Notable curated decisions (the answers to "what is the correct type"):
- **shoppers `SpecialDay`** is modeled **categorical**, not numerical. UCI documents
  it as numerical (a 6-level proximity score 0.0–1.0), but generative decoders only
  snap to valid discrete values for categorical columns; a continuous numerical
  column emits invalid in-between/out-of-range values. `csdata audit` intentionally
  flags this deliberate divergence.
- **shoppers `Browser`/`TrafficType`/`OperatingSystems`/`Region`** are categorical
  (integer-coded IDs), not numerical.
- **german `installment_rate`/`residence_since`/`existing_credits`/`maintenance_people`**
  are numerical (ordinal integer buckets), per UCI's 7-numerical/13-categorical split.
- **health_heritage** target `max_CharlsonIndex` is binary (`=0`/`>0`) → `binclass`.
- **default`/`beijing** use anonymized `col0…/label` names; `beijing` is regression
  (target `pm2.5`→`label`); the row-id `No` is dropped (quoted in YAML so it isn't
  parsed as boolean `false`).

## `csdata audit`

A **dev/reconciliation aid** (not a runtime input). For datasets sourced from
UCI with matching column names, it fetches the UCI variable table via the
optional `ucimlrepo` package and flags clear contradictions against the curated
spec. Integer-typed columns are never flagged — integer-coded columns are
genuinely ambiguous (ordinal-numeric vs category code), which is exactly the
human judgment the spec records.

```bash
pip install -e ".[audit]"
csdata audit --all
```

## Testing

```bash
python -m pytest -q
```

The suite covers spec validation, the registry, both render schemas × namings,
`validate`, `prepare`, the per-dataset transforms, the audit comparison logic,
and **reconciliation guards** that lock in the corrected splits so a future edit
can't silently reintroduce a heuristic misclassification. CI runs the suite on
every push/PR.
