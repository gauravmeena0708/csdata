"""Offline cross-check of curated specs against UCI ML Repository metadata.

This is a DEV/reconciliation aid, not a runtime input: it fetches the variable
table for datasets sourced from UCI (via the optional ``ucimlrepo`` package) and
flags clear contradictions against the curated spec. Integer-typed columns are
intentionally never flagged -- integer-coded columns are genuinely ambiguous
(ordinal-numeric vs category code), which is exactly the human judgment the
curated spec exists to record. Deliberate divergences (e.g. shoppers SpecialDay
modeled categorical though UCI types it Continuous) WILL surface here by design.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

from csdata.registry import load_spec
from csdata.spec import DatasetSpec

# csdata dataset -> UCI ML Repository id. Only datasets sourced from UCI whose
# column names line up with UCI's variable table are auditable; others (renamed
# col0.., derived compas columns, non-UCI sources) are intentionally excluded.
UCI_IDS: Dict[str, int] = {
    "adult": 2,
    "german": 144,
    "shoppers": 468,
    "bank_marketing": 222,
}

_NUMERIC_UCI = {"Continuous"}
_CATEGORICAL_UCI = {"Categorical", "Binary"}


def compare(spec: DatasetSpec, variables: List[Dict]) -> List[str]:
    """Pure cross-check of a spec against UCI variable records.

    ``variables``: list of ``{"name", "role", "type"}`` dicts. Returns advisory
    mismatch messages; empty means no clear contradictions.
    """
    msgs: List[str] = []
    numerical = set(spec.numerical)
    categorical = set(spec.categorical)
    for var in variables:
        name = var.get("name")
        role = var.get("role")
        vtype = var.get("type")
        if name not in spec.column_names and name not in spec.dropped:
            continue  # renamed / derived / absent column -> not auditable
        if role == "ID" and name not in spec.dropped:
            msgs.append(f"{name!r}: UCI role=ID but spec does not list it as dropped")
            continue
        if name in spec.dropped:
            continue
        if vtype in _NUMERIC_UCI and name in categorical:
            msgs.append(f"{name!r}: UCI type={vtype} (numeric) but spec marks it categorical")
        elif vtype in _CATEGORICAL_UCI and name in numerical:
            msgs.append(f"{name!r}: UCI type={vtype} but spec marks it numerical")
    return msgs


def audit_types(name: str, fetcher: Optional[Callable[[int], List[Dict]]] = None) -> List[str]:
    """Cross-check a csdata spec against UCI metadata.

    Requires the optional ``ucimlrepo`` package unless ``fetcher`` is injected
    (``fetcher(uci_id) -> [{"name","role","type"}, ...]``). Raises ``KeyError``
    for datasets with no UCI id mapping.
    """
    spec = load_spec(name)
    if name not in UCI_IDS:
        raise KeyError(f"no UCI id mapping for {name!r} (not auditable)")
    if fetcher is None:
        fetcher = _ucimlrepo_fetcher
    return compare(spec, fetcher(UCI_IDS[name]))


def auditable() -> List[str]:
    return sorted(UCI_IDS)


def _ucimlrepo_fetcher(uci_id: int) -> List[Dict]:
    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=uci_id)
    return [
        {"name": row["name"], "role": row["role"], "type": row["type"]}
        for _, row in dataset.variables.iterrows()
    ]
