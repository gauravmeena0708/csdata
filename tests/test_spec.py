import pytest
from csdata.spec import DatasetSpec


def test_spec_roundtrip_and_validation():
    spec = DatasetSpec.from_dict({
        "name": "toy",
        "task_type": "binclass",
        "target": "y",
        "column_names": ["a", "b", "y"],
        "numerical": ["a"],
        "categorical": ["b"],
        "dropped": [],
        "default_naming": "real",
        "source": {"url": "http://x", "transform": "toy"},
        "notes": "",
    })
    assert spec.name == "toy"
    assert spec.target == "y"
    assert set(spec.numerical + spec.categorical + [spec.target]) == set(spec.column_names)


def test_spec_rejects_target_in_feature_lists():
    with pytest.raises(ValueError):
        DatasetSpec.from_dict({
            "name": "bad", "task_type": "binclass", "target": "y",
            "column_names": ["a", "y"], "numerical": ["a"],
            "categorical": ["y"], "dropped": [], "default_naming": "real",
            "source": {"url": "u", "transform": "t"}, "notes": "",
        })


def test_spec_rejects_unknown_task_type():
    with pytest.raises(ValueError):
        DatasetSpec.from_dict({
            "name": "bad", "task_type": "clustering", "target": "y",
            "column_names": ["a", "y"], "numerical": ["a"], "categorical": [],
            "dropped": [], "default_naming": "real",
            "source": {"url": "u", "transform": "t"}, "notes": "",
        })
