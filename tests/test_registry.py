import pytest
from csdata.registry import load_spec, list_specs
from csdata.spec import DatasetSpec


def test_list_specs_has_all_ten():
    names = set(list_specs())
    assert names == {"adult", "default", "beijing", "shoppers", "german",
                     "bank_marketing", "telco_churn", "compas", "law_school",
                     "health_heritage"}


def test_load_spec_returns_validated_dataset_spec():
    spec = load_spec("adult")
    assert isinstance(spec, DatasetSpec)
    assert spec.target == "income"


def test_load_spec_unknown_raises():
    with pytest.raises(KeyError):
        load_spec("does_not_exist")


def test_all_specs_load_and_validate():
    for name in list_specs():
        load_spec(name)
