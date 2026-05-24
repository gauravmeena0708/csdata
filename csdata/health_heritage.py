from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd


TARGET = "max_CharlsonIndex"


def build_health_heritage_frame(paths: Mapping[str, str | Path]) -> pd.DataFrame:
    required = {"Claims.csv", "DrugCount.csv", "LabCount.csv", "Members.csv"}
    missing = sorted(required - set(paths))
    if missing:
        raise ValueError(f"Health Heritage raw files missing: {missing}")

    df_claims = preprocess_claims(
        pd.read_csv(paths["Claims.csv"]),
        ["PrimaryConditionGroup", "Specialty", "ProcedureGroup", "PlaceSvc"],
    )
    df_drugs = preprocess_drugs(pd.read_csv(paths["DrugCount.csv"]))
    df_labs = preprocess_labs(pd.read_csv(paths["LabCount.csv"]))
    df_members = preprocess_members(pd.read_csv(paths["Members.csv"]))

    df_labs_drugs = pd.merge(df_labs, df_drugs, on=["MemberID", "Year"], how="outer")
    df_labs_drugs_claims = pd.merge(df_labs_drugs, df_claims, on=["MemberID", "Year"], how="outer")
    df_health = pd.merge(df_labs_drugs_claims, df_members, on=["MemberID"], how="outer")
    df_health = df_health.drop(["Year", "MemberID"], axis=1)
    df_health = df_health.fillna(0)

    features = get_features(binary_age=False)
    encoded = _encoded_columns(df_health, features)
    mixed = _to_categorical(encoded.to_numpy(), features)
    df_mixed = pd.DataFrame(mixed, columns=list(features.keys()))
    for col, domain in features.items():
        if domain is None:
            df_mixed[col] = pd.to_numeric(df_mixed[col], errors="coerce")

    df_mixed[TARGET] = df_mixed[TARGET].apply(lambda value: ">0" if float(value) > 0 else "=0")
    ordered_columns = [col for col in features if col != TARGET] + [TARGET]
    return df_mixed[ordered_columns]


def get_features(binary_age: bool = False) -> dict[str, list[str] | None]:
    features: dict[str, list[str] | None] = {
        "LabCount_total": None,
        "LabCount_months": None,
        "DrugCount_total": None,
        "DrugCount_months": None,
        "no_Claims": None,
        "no_Providers": None,
        "no_Vendors": None,
        "no_PCPs": None,
        TARGET: None,
        "PayDelay_total": None,
        "PayDelay_max": None,
        "PayDelay_min": None,
        "PrimaryConditionGroup": [
            "AMI", "APPCHOL", "ARTHSPIN", "CANCRA", "CANCRB", "CANCRM", "CATAST",
            "CHF", "COPD", "FLaELEC", "FXDISLC", "GIBLEED", "GIOBSENT", "GYNEC1",
            "GYNECA", "HEART2", "HEART4", "HEMTOL", "HIPFX", "INFEC4", "LIVERDZ",
            "METAB1", "METAB3", "MISCHRT", "MISCL1", "MISCL5", "MSC2a3",
            "NEUMENT", "ODaBNCA", "PERINTL", "PERVALV", "PNCRDZ", "PNEUM",
            "PRGNCY", "PrimaryConditionGroup_?", "RENAL1", "RENAL2", "RENAL3",
            "RESPR4", "ROAMI", "SEIZURE", "SEPSIS", "SKNAUT", "STROKE",
            "TRAUMA", "UTI",
        ],
        "Specialty": [
            "Anesthesiology", "Diagnostic Imaging", "Emergency", "General Practice",
            "Internal", "Laboratory", "Obstetrics and Gynecology", "Other",
            "Pathology", "Pediatrics", "Rehabilitation", "Specialty_?", "Surgery",
        ],
        "ProcedureGroup": [
            "ANES", "EM", "MED", "PL", "ProcedureGroup_?", "RAD", "SAS", "SCS",
            "SDS", "SEOA", "SGS", "SIS", "SMCD", "SMS", "SNS", "SO", "SRS", "SUS",
        ],
        "PlaceSvc": [
            "Ambulance", "Home", "Independent Lab", "Inpatient Hospital", "Office",
            "Other", "Outpatient Hospital", "PlaceSvc_?", "Urgent Care",
        ],
        "AgeAtFirstClaim": ["0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+", "?"],
        "Sex": ["?", "F", "M"],
    }

    if binary_age:
        features["AgeAtFirstClaim"] = [">=60", "<60"]
    return features


def preprocess_claims(df_claims: pd.DataFrame, cat_names: list[str]) -> pd.DataFrame:
    df_claims = df_claims.copy()
    df_claims.loc[df_claims["PayDelay"] == "162+", "PayDelay"] = 162
    df_claims["PayDelay"] = df_claims["PayDelay"].astype(int)

    dsfs_map = {
        "0- 1 month": 1,
        "1- 2 months": 2,
        "2- 3 months": 3,
        "3- 4 months": 4,
        "4- 5 months": 5,
        "5- 6 months": 6,
        "6- 7 months": 7,
        "7- 8 months": 8,
        "8- 9 months": 9,
        "9-10 months": 10,
        "10-11 months": 11,
        "11-12 months": 12,
    }
    df_claims["DSFS"] = _map_values(df_claims["DSFS"], dsfs_map)

    charlson_map = {"0": 0, "1-2": 1, "3-4": 2, "5+": 3}
    df_claims["CharlsonIndex"] = _map_values(df_claims["CharlsonIndex"], charlson_map)

    stay_map = {
        "1 day": 1,
        "2 days": 2,
        "3 days": 3,
        "4 days": 4,
        "5 days": 5,
        "6 days": 6,
        "1- 2 weeks": 11,
        "2- 4 weeks": 21,
        "4- 8 weeks": 42,
        "26+ weeks": 180,
    }
    df_claims["LengthOfStay"] = _map_values(df_claims["LengthOfStay"], stay_map).fillna(0).astype(int)

    for cat_name in cat_names:
        df_claims[cat_name] = df_claims[cat_name].fillna(f"{cat_name}_?")
    df_claims = pd.get_dummies(df_claims, columns=cat_names, prefix_sep="=")

    one_hot_cols = [col for col in df_claims if "=" in col]
    agg = {
        "ProviderID": ["count", "nunique"],
        "Vendor": "nunique",
        "PCP": "nunique",
        "CharlsonIndex": "max",
        "PayDelay": ["sum", "max", "min"],
    }
    for col in one_hot_cols:
        agg[col] = "sum"

    df_group = df_claims.groupby(["Year", "MemberID"])
    out = df_group.agg(agg).reset_index()
    out.columns = [
        "Year",
        "MemberID",
        "no_Claims",
        "no_Providers",
        "no_Vendors",
        "no_PCPs",
        TARGET,
        "PayDelay_total",
        "PayDelay_max",
        "PayDelay_min",
    ] + one_hot_cols
    return out


def preprocess_drugs(df_drugs: pd.DataFrame) -> pd.DataFrame:
    df_drugs = df_drugs.copy().drop(columns=["DSFS"])
    df_drugs["DrugCount"] = df_drugs["DrugCount"].apply(lambda value: int(str(value).replace("+", "")))
    out = df_drugs.groupby(["Year", "MemberID"]).agg({"DrugCount": ["sum", "count"]}).reset_index()
    out.columns = ["Year", "MemberID", "DrugCount_total", "DrugCount_months"]
    return out


def preprocess_labs(df_labs: pd.DataFrame) -> pd.DataFrame:
    df_labs = df_labs.copy().drop(columns=["DSFS"])
    df_labs["LabCount"] = df_labs["LabCount"].apply(lambda value: int(str(value).replace("+", "")))
    out = df_labs.groupby(["Year", "MemberID"]).agg({"LabCount": ["sum", "count"]}).reset_index()
    out.columns = ["Year", "MemberID", "LabCount_total", "LabCount_months"]
    return out


def preprocess_members(df_members: pd.DataFrame) -> pd.DataFrame:
    df_members = df_members.copy()
    df_members["AgeAtFirstClaim"] = df_members["AgeAtFirstClaim"].fillna("?")
    df_members["Sex"] = df_members["Sex"].fillna("?")
    return pd.get_dummies(df_members, columns=["AgeAtFirstClaim", "Sex"], prefix_sep="=")


def _encoded_columns(df: pd.DataFrame, features: dict[str, list[str] | None]) -> pd.DataFrame:
    columns = {}
    for feature, domain in features.items():
        if domain is None:
            columns[feature] = df[feature] if feature in df.columns else 0
        else:
            for category in domain:
                column = f"{feature}={category}"
                columns[column] = df[column] if column in df.columns else 0
    return pd.DataFrame(columns, index=df.index)


def _to_categorical(data: np.ndarray, features: dict[str, list[str] | None]) -> np.ndarray:
    columns = []
    pointer = 0
    for feature, domain in features.items():
        if domain is None:
            columns.append(np.floor(data[:, pointer].astype(float) + 0.5))
            pointer += 1
            continue

        start = pointer
        end = pointer + len(domain)
        hot_args = np.argmax(data[:, start:end].astype(float), axis=1)
        columns.append([domain[arg] for arg in hot_args])
        pointer = end

    out = None
    for column in columns:
        reshaped = np.reshape(np.array(column, dtype=object), (data.shape[0], -1))
        out = reshaped if out is None else np.concatenate((out, reshaped), axis=1)
    return out


def _map_values(series: pd.Series, mapping: dict) -> pd.Series:
    return series.map(lambda value: mapping.get(value, value))
