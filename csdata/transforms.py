from __future__ import annotations

from collections.abc import Callable

import pandas as pd


HEADERS = {
    "adult": [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "income",
    ],
    "german": [
        "status",
        "duration",
        "credit_history",
        "purpose",
        "credit_amount",
        "savings",
        "employment",
        "installment_rate",
        "personal_status",
        "debtors",
        "residence_since",
        "property",
        "age",
        "installment_plans",
        "housing",
        "existing_credits",
        "job",
        "maintenance_people",
        "telephone",
        "foreign_worker",
        "class",
    ],
    "heart": [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
        "target",
    ],
}


_REGISTRY: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {}


def register(name: str) -> Callable[[Callable[[pd.DataFrame], pd.DataFrame]], Callable[[pd.DataFrame], pd.DataFrame]]:
    def deco(fn: Callable[[pd.DataFrame], pd.DataFrame]) -> Callable[[pd.DataFrame], pd.DataFrame]:
        _REGISTRY[name] = fn
        return fn

    return deco


def has_transform(name: str) -> bool:
    return name in _REGISTRY


def apply_transform(name: str, df: pd.DataFrame) -> pd.DataFrame:
    if name not in _REGISTRY:
        raise KeyError(f"no transform registered for {name!r}")
    out = _REGISTRY[name](df.copy())
    if name in HEADERS and len(out.columns) == len(HEADERS[name]):
        out.columns = HEADERS[name]
    if hasattr(out.columns, "str"):
        out.columns = out.columns.str.strip()
    return out


@register("default")
def _default(df: pd.DataFrame) -> pd.DataFrame:
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])
    df.columns = [f"col{i}" for i in range(len(df.columns) - 1)] + ["label"]
    df["col1"] = "S" + df["col1"].astype(int).astype(str)
    df["col2"] = "E" + df["col2"].astype(int).astype(str)
    df["col3"] = "M" + df["col3"].astype(int).astype(str)
    for i in range(5, 11):
        df[f"col{i}"] = "P" + df[f"col{i}"].astype(int).astype(str)
    df["label"] = "L" + df["label"].astype(int).astype(str)
    return df


@register("beijing")
def _beijing(df: pd.DataFrame) -> pd.DataFrame:
    if "No" in df.columns:
        df = df.drop(columns=["No"])
    if "pm2.5" in df.columns:
        df = df.dropna(subset=["pm2.5"])
        cols = [c for c in df.columns if c != "pm2.5"] + ["pm2.5"]
        df = df[cols]
    df.columns = [f"col{i}" for i in range(len(df.columns) - 1)] + ["label"]
    return df


@register("telco_churn")
def _telco(df: pd.DataFrame) -> pd.DataFrame:
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)
    return _move_target_last(df, "Churn")


@register("compas")
def _compas(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "days_b_screening_arrest",
        "is_recid",
        "c_charge_degree",
        "score_text",
        "in_custody",
        "out_custody",
        "c_jail_in",
        "c_jail_out",
        "two_year_recid",
        "age",
        "sex",
        "race",
        "priors_count",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"COMPAS raw file missing required columns: {missing}")

    score_col = "v_score_text" if "v_score_text" in df.columns else "score_text"
    df = df[df["days_b_screening_arrest"].between(-30, 30)]
    df = df[df["is_recid"] != -1]
    df = df[df["c_charge_degree"] != "0"]
    df = df[df[score_col] != "N/A"]

    for col in ("in_custody", "out_custody", "c_jail_in", "c_jail_out"):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    df["diff_custody"] = (df["out_custody"] - df["in_custody"]).dt.total_seconds()
    df["diff_jail"] = (df["c_jail_out"] - df["c_jail_in"]).dt.total_seconds()

    keep = [
        "age",
        "sex",
        "race",
        "diff_custody",
        "diff_jail",
        "priors_count",
        "c_charge_degree",
        score_col,
        "two_year_recid",
    ]
    df = df[keep].copy()
    if score_col != "v_score_text":
        df = df.rename(columns={score_col: "v_score_text"})
    return df


@register("law_school")
def _law_school(df: pd.DataFrame) -> pd.DataFrame:
    for col in ("cluster", "fulltime"):
        if col in df.columns:
            df[col] = df[col].map(lambda v: str(int(v)) if pd.notna(v) else v)

    for col in df.columns:
        unique_vals = {str(v).lower() for v in df[col].unique() if pd.notna(v)}
        if df[col].dtype in [bool, "bool"] or unique_vals <= {"true", "false", "1", "0", "1.0", "0.0"}:
            df[col] = df[col].map(
                lambda v: "TRUE"
                if str(v).lower() in {"true", "1", "1.0"}
                else ("FALSE" if str(v).lower() in {"false", "0", "0.0"} else v)
            )
    return _move_target_last(df, "bar")


@register("adult")
def _adult(df: pd.DataFrame) -> pd.DataFrame:
    return _move_target_last(df, "income")


@register("german")
def _german(df: pd.DataFrame) -> pd.DataFrame:
    return _move_target_last(df, "class")


@register("bank_marketing")
def _bank_marketing(df: pd.DataFrame) -> pd.DataFrame:
    return _move_target_last(df, "y")


@register("shoppers")
def _shoppers(df: pd.DataFrame) -> pd.DataFrame:
    return _move_target_last(df, "Revenue")


@register("wine_quality_red")
def _wine_quality_red(df: pd.DataFrame) -> pd.DataFrame:
    return _move_target_last(df, "quality")


@register("heart")
def _heart(df: pd.DataFrame) -> pd.DataFrame:
    target_col = df.columns[-1]
    df[target_col] = (pd.to_numeric(df[target_col], errors="coerce") > 0).astype(int)
    return df


@register("health_heritage")
def _health_heritage(df: pd.DataFrame) -> pd.DataFrame:
    return _move_target_last(df, "max_CharlsonIndex")


def _move_target_last(df: pd.DataFrame, target: str) -> pd.DataFrame:
    if target in df.columns:
        return df[[c for c in df.columns if c != target] + [target]]
    return df
