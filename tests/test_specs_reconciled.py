from csdata.registry import load_spec


def test_shoppers_specialday_is_categorical():
    # SpecialDay is a 6-level proximity score. Modeled categorical (not numerical)
    # so generators emit only the valid discrete levels; see shoppers.yaml notes.
    s = load_spec("shoppers")
    assert "SpecialDay" in s.categorical
    assert "SpecialDay" not in s.numerical


def test_shoppers_browser_traffictype_are_categorical():
    s = load_spec("shoppers")
    for c in ("Browser", "TrafficType", "OperatingSystems", "Region"):
        assert c in s.categorical, c


def test_german_count_columns_are_numerical():
    s = load_spec("german")
    for c in ("installment_rate", "residence_since", "existing_credits", "maintenance_people"):
        assert c in s.numerical, c


def test_health_heritage_no_pcps_is_numerical():
    s = load_spec("health_heritage")
    assert "no_PCPs" in s.numerical


def test_dropped_columns_never_appear_in_features():
    for name in ("telco_churn", "compas", "default", "beijing"):
        s = load_spec(name)
        feat = set(s.numerical) | set(s.categorical) | {s.target}
        assert feat.isdisjoint(set(s.dropped)), name
