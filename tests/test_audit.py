import pytest

from csdata.audit import compare, audit_types
from csdata.registry import load_spec


def test_compare_flags_continuous_marked_categorical():
    # shoppers SpecialDay is modeled categorical in the spec, but UCI types it
    # Continuous -> the audit should surface this deliberate divergence.
    spec = load_spec("shoppers")
    variables = [{"name": "SpecialDay", "role": "Feature", "type": "Continuous"}]
    assert any("SpecialDay" in m for m in compare(spec, variables))


def test_compare_flags_categorical_marked_numerical():
    spec = load_spec("german")  # installment_rate is numerical in the spec
    variables = [{"name": "installment_rate", "role": "Feature", "type": "Categorical"}]
    assert any("installment_rate" in m for m in compare(spec, variables))


def test_compare_clean_when_consistent():
    spec = load_spec("adult")
    variables = [
        {"name": "age", "role": "Feature", "type": "Continuous"},
        {"name": "workclass", "role": "Feature", "type": "Categorical"},
    ]
    assert compare(spec, variables) == []


def test_compare_ignores_integer_ambiguity():
    # Integer-typed columns are genuinely ambiguous; the audit must NOT flag them.
    spec = load_spec("german")  # installment_rate numerical
    variables = [{"name": "installment_rate", "role": "Feature", "type": "Integer"}]
    assert compare(spec, variables) == []


def test_compare_flags_id_not_dropped():
    spec = load_spec("adult")  # adult has no dropped columns
    variables = [{"name": "age", "role": "ID", "type": "Integer"}]
    assert any("age" in m and "ID" in m for m in compare(spec, variables))


def test_audit_types_with_injected_fetcher():
    msgs = audit_types(
        "shoppers",
        fetcher=lambda uid: [{"name": "SpecialDay", "role": "Feature", "type": "Continuous"}],
    )
    assert any("SpecialDay" in m for m in msgs)


def test_audit_types_unmappable_raises():
    with pytest.raises(KeyError):
        audit_types("law_school")
