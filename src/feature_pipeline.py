from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from config import ProjectConfig, get_config


MEDICATION_COLS = [
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "examide",
    "citoglipton",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
]


@dataclass
class FeatureArtifacts:
    train_encoded: pd.DataFrame
    test_encoded: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    y_train_binary: pd.Series
    y_test_binary: pd.Series
    encoder: ColumnTransformer


@dataclass
class FeatureSelectionArtifacts:
    chi2_results: pd.DataFrame
    features_to_keep: list[str]
    features_to_drop: list[str]
    train_df_selected: pd.DataFrame
    test_df_selected: pd.DataFrame


def clean_missing(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()

    cleaned = cleaned.replace("?", np.nan)
    cleaned = cleaned.drop(columns=["weight", "max_glu_serum"], errors="ignore")

    fill_map = {
        "payer_code": "Unknown",
        "medical_specialty": "Unknown",
        "race": "Unknown",
        "A1Cresult": "None",
    }
    for column, value in fill_map.items():
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].fillna(value)

    diag_cols = ["diag_1", "diag_2", "diag_3"]
    for column in diag_cols:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].fillna("Unknown_Diagnosis")

    cleaned["gender"] = cleaned["gender"].fillna("Unknown")

    return cleaned


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    featured = df.copy()

    available_med_cols = [col for col in MEDICATION_COLS if col in featured.columns]
    if available_med_cols:
        adjusted_values = {"Up", "Down"}
        featured["med_was_adjusted"] = (
            featured[available_med_cols]
            .apply(lambda row: any(str(v).strip() in adjusted_values for v in row), axis=1)
            .astype(int)
        )

    featured["num_prev_visits"] = (
        featured["number_outpatient"].fillna(0)
        + featured["number_emergency"].fillna(0)
        + featured["number_inpatient"].fillna(0)
    )

    featured["intensity_of_care"] = (
        featured["num_lab_procedures"] + featured["num_procedures"]
    ) / featured["time_in_hospital"].clip(lower=1)

    featured["lab_med_ratio"] = featured["num_lab_procedures"] / featured[
        "num_medications"
    ].clip(lower=1)

    featured["is_a1c_high"] = featured["A1Cresult"].isin([">7", ">8"]).astype(int)

    if "age" in featured.columns:
        featured["age_band"] = featured["age"].astype(str).str.strip()

    return featured


def drop_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=["encounter_id", "patient_nbr"], errors="ignore")


def encode_targets(
    y_train_raw: pd.Series,
    y_test_raw: pd.Series,
    config: ProjectConfig,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    y_train = y_train_raw.map(config.multiclass_map)
    y_test = y_test_raw.map(config.multiclass_map)
    y_train_binary = y_train_raw.map(config.binary_map)
    y_test_binary = y_test_raw.map(config.binary_map)

    if y_train.isna().any() or y_test.isna().any():
        raise ValueError("Unmapped labels found in target column.")

    return (
        y_train.astype(int),
        y_test.astype(int),
        y_train_binary.astype(int),
        y_test_binary.astype(int),
    )


def one_hot_encode(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    drop: str | None = None,
    min_frequency: int | float | None = None,
    max_categories: int | None = None,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, ColumnTransformer]:
    categorical_cols = (
        train_df.select_dtypes(include=["object", "category"]).columns.tolist()
    )

    encoder = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    drop=drop,
                    min_frequency=min_frequency,
                    max_categories=max_categories,
                ),
                categorical_cols,
            )
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    encoder.fit(train_df)

    train_encoded = pd.DataFrame(
        encoder.transform(train_df),
        columns=encoder.get_feature_names_out(),
        index=train_df.index,
    )
    test_encoded = pd.DataFrame(
        encoder.transform(test_df),
        columns=encoder.get_feature_names_out(),
        index=test_df.index,
    )

    if verbose:
        print("Encoding complete")
        print(f"train: {train_df.shape} -> {train_encoded.shape}")
        print(f"test : {test_df.shape} -> {test_encoded.shape}")

    return train_encoded, test_encoded, encoder


def save_feature_artifacts(artifacts: FeatureArtifacts, config: ProjectConfig) -> None:
    artifacts.train_encoded.to_csv(config.train_encoded_path, index=False)
    artifacts.test_encoded.to_csv(config.test_encoded_path, index=False)
    artifacts.y_train.to_csv(config.y_train_path, index=False)
    artifacts.y_test.to_csv(config.y_test_path, index=False)
    artifacts.y_train_binary.to_csv(config.y_train_binary_path, index=False)
    artifacts.y_test_binary.to_csv(config.y_test_binary_path, index=False)
    joblib.dump(artifacts.encoder, config.encoder_path)


def run_feature_pipeline(
    config: ProjectConfig | None = None,
    save: bool = True,
) -> FeatureArtifacts:
    cfg = config or get_config()

    train_df = pd.read_csv(cfg.interim_train_path)
    test_df = pd.read_csv(cfg.interim_test_path)

    y_train_raw = train_df[cfg.target_col]
    y_test_raw = test_df[cfg.target_col]

    train_X = train_df.drop(columns=[cfg.target_col])
    test_X = test_df.drop(columns=[cfg.target_col])

    train_X = drop_id_columns(add_engineered_features(clean_missing(train_X)))
    test_X = drop_id_columns(add_engineered_features(clean_missing(test_X)))

    y_train, y_test, y_train_binary, y_test_binary = encode_targets(
        y_train_raw,
        y_test_raw,
        cfg,
    )

    train_encoded, test_encoded, encoder = one_hot_encode(train_X, test_X)

    artifacts = FeatureArtifacts(
        train_encoded=train_encoded,
        test_encoded=test_encoded,
        y_train=y_train,
        y_test=y_test,
        y_train_binary=y_train_binary,
        y_test_binary=y_test_binary,
        encoder=encoder,
    )

    if save:
        save_feature_artifacts(artifacts, cfg)

    return artifacts


def clean_missing_notebook(df: pd.DataFrame) -> pd.DataFrame:
    """Notebook-compatible missing-value cleaning logic used in 02 notebook."""
    out = df.copy()
    out = out.drop(columns=["weight", "max_glu_serum"], errors="ignore")

    if {"medical_specialty", "payer_code"}.issubset(out.columns):
        out[["medical_specialty", "payer_code"]] = (
            out[["medical_specialty", "payer_code"]].replace("?", np.nan)
        )
        out[["medical_specialty", "payer_code"]] = (
            out[["medical_specialty", "payer_code"]].fillna("Unknown")
        )

    if "A1Cresult" in out.columns:
        out["A1Cresult"] = out["A1Cresult"].replace({"?": "None", pd.NA: "None"}).fillna(
            "None"
        )

    if "diag_1" in out.columns:
        out = out[out["diag_1"] != "?"]
    if {"diag_2", "number_diagnoses"}.issubset(out.columns):
        out = out[~((out["diag_2"] == "?") & (out["number_diagnoses"] > 1))]
    if {"diag_3", "number_diagnoses"}.issubset(out.columns):
        out = out[~((out["diag_3"] == "?") & (out["number_diagnoses"] > 2))]

    if "diag_2" in out.columns:
        out["diag_2"] = out["diag_2"].replace("?", "no_diag")
    if "diag_3" in out.columns:
        out["diag_3"] = out["diag_3"].replace("?", "no_diag")

    if "race" in out.columns:
        out = out[out["race"] != "?"]

    return out


def audit_missing(df: pd.DataFrame, sentinel: str = "?") -> pd.DataFrame:
    """Returns missing-value summary treating sentinel strings as NaN."""
    df_audit = df.replace(sentinel, np.nan)
    missing_counts = df_audit.isnull().sum()
    missing_pct = (missing_counts / len(df) * 100).round(2)

    report = (
        pd.concat([missing_counts, missing_pct], axis=1)
        .set_axis(["Missing Count", "Missing %"], axis=1)
        .query("`Missing Count` > 0")
        .sort_values("Missing Count", ascending=False)
    )
    return report


class EncodeCategorialFeatures:
    """Notebook-compatible categorical feature engineering block."""

    ICD9_RANGES = [
        (1, 139, "Infectious_Parasitic"),
        (140, 239, "Neoplasms"),
        (240, 249, "Endocrine_Non_Diabetes"),
        (250, 250, "Diabetes"),
        (251, 279, "Endocrine_Metabolic_Other"),
        (280, 289, "Blood_Diseases"),
        (290, 319, "Mental_Disorders"),
        (320, 389, "Nervous_System"),
        (390, 459, "Circulatory"),
        (460, 519, "Respiratory"),
        (580, 629, "Genitourinary"),
        (800, 999, "Injury_Poisoning"),
    ]

    DIAG_REFINEMENT_MAP = {
        "Endocrine_Non_Diabetes": "Endocrine_Metabolic_Other",
    }
    DIAG_COLS = ["diag_1", "diag_2", "diag_3"]

    SPECIALTY_MAP = {
        "InternalMedicine": "Internal_Medicine",
        "Hospitalist": "Internal_Medicine",
        "Nephrology": "Internal_Medicine",
        "Pulmonology": "Internal_Medicine",
        "Gastroenterology": "Internal_Medicine",
        "Endocrinology": "Internal_Medicine",
        "Endocrinology-Metabolism": "Internal_Medicine",
        "InfectiousDiseases": "Internal_Medicine",
        "Rheumatology": "Internal_Medicine",
        "Hematology": "Internal_Medicine",
        "Surgery-General": "Surgery",
        "Surgeon": "Surgery",
        "SurgicalSpecialty": "Surgery",
        "Surgery-Neuro": "Surgery",
        "Surgery-Vascular": "Surgery",
        "Surgery-Thoracic": "Surgery",
        "Surgery-Cardiovascular/Thoracic": "Surgery",
        "Surgery-Cardiovascular": "Surgery",
        "Surgery-Maxillofacial": "Surgery",
        "Surgery-Plastic": "Surgery",
        "Surgery-Colon&Rectal": "Surgery",
        "Orthopedics": "Surgery",
        "Orthopedics-Reconstructive": "Surgery",
        "Urology": "Surgery",
        "Ophthalmology": "Surgery",
        "Podiatry": "Surgery",
        "Dentistry": "Surgery",
        "Proctology": "Surgery",
        "Cardiology": "Cardiology",
        "Emergency/Trauma": "Emergency_Critical_Care",
        "Anesthesiology": "Emergency_Critical_Care",
        "Anesthesiology-Pediatric": "Emergency_Critical_Care",
        "Pediatrics": "Pediatrics",
        "Pediatrics-CriticalCare": "Pediatrics",
        "Pediatrics-Endocrinology": "Pediatrics",
        "Pediatrics-Pulmonology": "Pediatrics",
        "Cardiology-Pediatric": "Pediatrics",
        "Surgery-Pediatric": "Pediatrics",
        "Pediatrics-Hematology-Oncology": "Pediatrics",
        "Pediatrics-Neurology": "Pediatrics",
        "Pediatrics-EmergencyMedicine": "Pediatrics",
        "Pediatrics-AllergyandImmunology": "Pediatrics",
        "Psychiatry": "Psych_Neuro",
        "Psychology": "Psych_Neuro",
        "Psychiatry-Child/Adolescent": "Psych_Neuro",
        "Neurology": "Psych_Neuro",
        "Neurophysiology": "Psych_Neuro",
        "ObstetricsandGynecology": "OBGYN",
        "Gynecology": "OBGYN",
        "Obstetrics": "OBGYN",
        "Obsterics&Gynecology-GynecologicOnco": "OBGYN",
        "Perinatology": "OBGYN",
        "Family/GeneralPractice": "Family_Outpatient",
        "Osteopath": "Family_Outpatient",
        "OutreachServices": "Family_Outpatient",
        "AllergyandImmunology": "Family_Outpatient",
        "Dermatology": "Family_Outpatient",
        "PhysicalMedicineandRehabilitation": "Family_Outpatient",
        "Radiologist": "Diagnostics_Other",
        "Radiology": "Diagnostics_Other",
        "Pathology": "Diagnostics_Other",
        "Oncology": "Diagnostics_Other",
        "Hematology/Oncology": "Diagnostics_Other",
        "Speech": "Diagnostics_Other",
    }

    def __init__(self, drop_raw_medication_columns: bool = True):
        self.drop_raw_medication_columns = drop_raw_medication_columns

    @classmethod
    def _categorize_icd9(cls, series: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(series, errors="coerce")
        result = pd.Series("Other", index=series.index)
        result[series.isna() | (series.astype(str).str.strip() == "no_diag")] = "None"
        result[series.astype(str).str.upper().str.startswith(("E", "V"))] = (
            "External_Supplemental"
        )
        for start, end, label in cls.ICD9_RANGES:
            result[(numeric >= start) & (numeric <= end)] = label
        return result

    def transform_specialty(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["medical_specialty"] = out["medical_specialty"].map(self.SPECIALTY_MAP).fillna(
            "Other_Unknown"
        )
        return out

    def bin_admissiontype(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        source_map = {
            **dict.fromkeys(["1", "2", "3"], "Physician_Referral"),
            **dict.fromkeys(
                ["4", "5", "6", "10", "18", "22", "25"], "Transfer_Hospital"
            ),
            **dict.fromkeys(["7"], "Emergency_Room"),
            **dict.fromkeys(["8", "9"], "Court_Law"),
            **dict.fromkeys(["11", "12", "13", "14"], "Transfer_Facility"),
        }
        out["admission_source_id"] = (
            out["admission_source_id"].astype(str).map(source_map).fillna("Other_Source")
        )
        out["admission_source_id"] = out["admission_source_id"].replace(
            ["Court_Law", "Transfer_Facility"], "Other_Source"
        )
        adm_type_map = {"1": "Emergency", "2": "Urgent", "3": "Elective"}
        out["admission_type_id"] = (
            out["admission_type_id"]
            .astype(str)
            .map(adm_type_map)
            .fillna("Other_Admission")
        )
        return out

    def transform_clinical_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        a1c_map = {"None": "Not_Tested", "Norm": "Normal", ">7": "High", ">8": "High"}
        out["A1Cresult"] = out["A1Cresult"].map(a1c_map).fillna("Not_Tested")
        for col in self.DIAG_COLS:
            out[col] = self._categorize_icd9(out[col])
        out[self.DIAG_COLS] = out[self.DIAG_COLS].replace(self.DIAG_REFINEMENT_MAP)
        return out

    def transform_medications(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        med_cols = [c for c in MEDICATION_COLS if c in out.columns]
        med_cols = [c for c in med_cols if c in out.columns]
        if not med_cols:
            return out

        out["med_was_adjusted"] = out[med_cols].isin(["Up", "Down"]).any(axis=1).astype(int)
        out["count_titrated_up"] = (out[med_cols] == "Up").sum(axis=1)
        out["count_titrated_down"] = (out[med_cols] == "Down").sum(axis=1)
        out["adjustment_intensity"] = (
            (out["count_titrated_up"] + out["count_titrated_down"])
            / out["num_medications"].clip(lower=1)
        )
        out["any_diabetes_medication"] = out[med_cols].isin(
            ["Up", "Down", "Steady"]
        ).any(axis=1).astype(int)
        if "insulin" in out.columns:
            out["insulin_active"] = out["insulin"].isin(["Up", "Down", "Steady"]).astype(int)

        if self.drop_raw_medication_columns:
            out = out.drop(columns=med_cols, errors="ignore")
        return out

    def transform_payer_codes(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        s = out["payer_code"].fillna("Unknown").astype(str).str.upper()
        conditions = [
            s == "UNKNOWN",
            s.str.contains("MC|MEDICARE"),
            s.str.contains("MD|MEDICAID"),
            s.str.contains("BC|BLUE|HM|COMMERCIAL|PO|UN"),
            s.str.contains("SP|SELF"),
        ]
        choices = ["Unknown", "Medicare", "Medicaid", "Commercial", "Self_Pay"]
        out["payer_group"] = np.select(conditions, choices, default="Other_Insurance")
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return (
            df.pipe(self.bin_admissiontype)
            .pipe(self.transform_clinical_metrics)
            .pipe(self.transform_medications)
            .pipe(self.transform_payer_codes)
            .pipe(self.transform_specialty)
        )


def apply_age_specialty_interaction(df: pd.DataFrame) -> pd.Series:
    spec = df["medical_specialty"].fillna("Unknown").astype(str)
    age = df["age"].fillna("Unknown").astype(str)
    return spec + "_" + age


def add_a1c_med_interaction(df: pd.DataFrame) -> pd.Series:
    a1c = df["A1Cresult"].fillna("None").astype(str)
    med = df["med_was_adjusted"].map({1: "MedAdjusted", 0: "NoMedChange"})
    return a1c + "_" + med


def add_diag_combination(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    out = df.copy()
    combos = out[["diag_1", "diag_2", "diag_3"]].apply(
        lambda row: " + ".join(sorted(set(row.dropna()))), axis=1
    )
    top_combos = combos.value_counts().nlargest(top_n).index
    out["diag_combination"] = combos.where(combos.isin(top_combos), "Other_Combination")
    return out


def apply_train_fitted_category_mappings(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    columns: list[str],
    min_count_map: dict[str, int] | None = None,
    top_n_map: dict[str, int] | None = None,
    other_label_map: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fit categorical value support on train and apply the same mapping to test.

    Categories not supported by the train-derived mapping are collapsed to an
    explicit fallback label, preventing train/test drift from exploding OHE space.
    """
    min_count_map = min_count_map or {}
    top_n_map = top_n_map or {}
    other_label_map = other_label_map or {}

    train_out = train_df.copy()
    test_out = test_df.copy()
    report_rows: list[dict[str, int | str]] = []

    for col in columns:
        if col not in train_out.columns or col not in test_out.columns:
            continue

        default_other = f"{col}_Other"
        other_label = other_label_map.get(col, default_other)
        min_count = int(min_count_map.get(col, 1))
        top_n = top_n_map.get(col)

        train_series = train_out[col].astype(str).fillna(other_label)
        test_series = test_out[col].astype(str).fillna(other_label)

        counts = train_series.value_counts(dropna=False)
        allowed = counts[counts >= min_count]
        if top_n is not None:
            allowed = allowed.head(top_n)
        allowed_values = set(allowed.index.astype(str).tolist())

        train_before_unique = int(train_series.nunique(dropna=False))
        test_before_unique = int(test_series.nunique(dropna=False))
        unknown_test_before = int((~test_series.isin(set(counts.index.astype(str)))).sum())

        train_out[col] = np.where(train_series.isin(allowed_values), train_series, other_label)
        test_out[col] = np.where(test_series.isin(allowed_values), test_series, other_label)

        train_other_rows = int((train_out[col] == other_label).sum())
        test_other_rows = int((test_out[col] == other_label).sum())

        # Ensure fallback bucket exists in train when test contains that bucket.
        # This prevents "unknown category during transform" warnings at OHE time.
        if test_other_rows > 0 and train_other_rows == 0 and len(train_out) > 0:
            rare_value = counts.tail(1).index[0]
            fallback_idx = train_series[train_series == rare_value].index[:1]
            if len(fallback_idx) == 0:
                fallback_idx = train_out.index[:1]
            train_out.loc[fallback_idx, col] = other_label
            train_other_rows = int((train_out[col] == other_label).sum())

        train_after_unique = int(pd.Series(train_out[col]).nunique(dropna=False))
        test_after_unique = int(pd.Series(test_out[col]).nunique(dropna=False))

        report_rows.append(
            {
                "feature": col,
                "min_count": int(min_count),
                "top_n": int(top_n) if top_n is not None else -1,
                "train_unique_before": train_before_unique,
                "train_unique_after": train_after_unique,
                "test_unique_before": test_before_unique,
                "test_unique_after": test_after_unique,
                "unknown_test_rows_before_mapping": unknown_test_before,
                "train_rows_mapped_to_other": train_other_rows,
                "test_rows_mapped_to_other": test_other_rows,
                "other_label": other_label,
            }
        )

    report = pd.DataFrame(report_rows)
    return train_out, test_out, report


def add_intensity_of_care(df: pd.DataFrame) -> pd.Series:
    return (df["num_lab_procedures"] + df["num_procedures"]) / df["time_in_hospital"].clip(
        lower=1
    )


def add_lab_med_ratio(df: pd.DataFrame) -> pd.Series:
    return df["num_lab_procedures"] / df["num_medications"].clip(lower=1)


def transform_skewed_numerical_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    skew_threshold: float = 1.0,
    exclude_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply log1p to heavily right-skewed numeric features and return an audit table."""
    exclude = set(exclude_cols or [])
    train_out = train_df.copy()
    test_out = test_df.copy()

    ##Change here what are the numerical Columns
    numeric_cols = ['time_in_hospital',
            'num_lab_procedures', 'num_procedures', 'num_medications',
            'number_outpatient', 'number_emergency', 'number_inpatient',
            'number_diagnoses'
            ]
    candidate_cols = [c for c in numeric_cols if c not in exclude]

    rows: list[dict[str, object]] = []
    for col in candidate_cols:
        before_skew = float(train_out[col].dropna().skew())
        min_train = train_out[col].min()
        min_test = test_out[col].min() if col in test_out.columns else min_train
        use_log = before_skew >= skew_threshold and min(min_train, min_test) >= 0

        if use_log:
            train_out[col] = np.log1p(train_out[col])
            test_out[col] = np.log1p(test_out[col])

        after_skew = float(train_out[col].dropna().skew())
        rows.append(
            {
                "feature": col,
                "train_skew_before": round(before_skew, 4),
                "train_skew_after": round(after_skew, 4),
                "transform_applied": "log1p" if use_log else "none",
            }
        )

    report = pd.DataFrame(rows)
    if not report.empty:
        report = report.sort_values(
            "train_skew_before", key=lambda s: s.abs(), ascending=False
        ).reset_index(drop=True)
    return train_out, test_out, report


def run_chi2_analysis(
    df: pd.DataFrame,
    target_col: str,
    alpha: float = 0.02,
    min_cramers_v: float = 0.02,
) -> pd.DataFrame:
    categorical_cols = [
        col for col in df.columns if df[col].dtype == "object" and col != target_col
    ]

    from scipy.stats import chi2_contingency

    results: list[dict[str, object]] = []
    for col in categorical_cols:
        contingency = pd.crosstab(df[col], df[target_col])
        chi2, p, dof, _ = chi2_contingency(contingency)
        n = contingency.values.sum()
        k = min(contingency.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * k)) if k > 0 else 0

        results.append(
            {
                "Feature": col,
                "Chi2": round(chi2, 2),
                "p_value": round(p, 6),
                "Degrees_of_Freedom": dof,
                "CramersV": round(cramers_v, 4),
                "Significant": p < alpha,
                "Keep": p < alpha and cramers_v > min_cramers_v,
            }
        )

    return (
        pd.DataFrame(results).sort_values("CramersV", ascending=False).reset_index(drop=True)
    )


def apply_feature_drop(
    train_df: pd.DataFrame, test_df: pd.DataFrame, features_to_drop: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_selected = train_df.drop(columns=features_to_drop, errors="ignore")
    test_selected = test_df.drop(columns=features_to_drop, errors="ignore")
    return train_selected, test_selected


def run_chi2_feature_selection(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_train: pd.Series,
    target_col: str = "readmitted",
    alpha: float = 0.02,
    min_cramers_v: float = 0.02,
    max_features_to_keep: int | None = None,
    protected_features: list[str] | None = None,
) -> FeatureSelectionArtifacts:
    analysis_df = train_df.copy()
    analysis_df[target_col] = y_train
    chi2_results = run_chi2_analysis(
        analysis_df,
        target_col=target_col,
        alpha=alpha,
        min_cramers_v=min_cramers_v,
    )

    keep_df = chi2_results[chi2_results["Keep"]].copy()
    protected = set(protected_features or [])
    all_ranked_features = chi2_results["Feature"].tolist()
    protected_present = [feature for feature in all_ranked_features if feature in protected]

    # Protected features are kept regardless of statistical cutoff.
    if protected_present:
        protected_df = chi2_results[chi2_results["Feature"].isin(protected_present)]
        keep_df = pd.concat([keep_df, protected_df], ignore_index=True).drop_duplicates(
            subset=["Feature"]
        )

    if max_features_to_keep is not None and len(keep_df) > max_features_to_keep:
        protected_df = keep_df[keep_df["Feature"].isin(protected_present)]
        ranked_df = keep_df[~keep_df["Feature"].isin(protected)].sort_values(
            "CramersV", ascending=False
        )
        # If protected features exceed the cap, keep all protected features.
        remaining = max(max_features_to_keep - len(protected_df), 0)
        keep_df = pd.concat([protected_df, ranked_df.head(remaining)], ignore_index=True)
        keep_df = keep_df.drop_duplicates(subset=["Feature"])

    features_to_keep = keep_df.sort_values("CramersV", ascending=False)["Feature"].tolist()
    features_to_drop = [
        feature
        for feature in chi2_results["Feature"].tolist()
        if feature not in set(features_to_keep)
    ]
    train_selected, test_selected = apply_feature_drop(train_df, test_df, features_to_drop)

    return FeatureSelectionArtifacts(
        chi2_results=chi2_results,
        features_to_keep=features_to_keep,
        features_to_drop=features_to_drop,
        train_df_selected=train_selected,
        test_df_selected=test_selected,
    )


def summarize_ohe_cardinality(df: pd.DataFrame) -> tuple[dict[str, int], pd.DataFrame]:
    ohe_cols = [col for col in df.columns if "_" in col]
    non_ohe_cols = [col for col in df.columns if "_" not in col]
    categorical_cols = [col for col in df.columns if df[col].dtype == "object"]

    cardinality = pd.DataFrame(
        {
            "feature": categorical_cols,
            "n_unique": [df[col].nunique() for col in categorical_cols],
        }
    ).sort_values("n_unique", ascending=False)

    summary = {
        "total_columns": df.shape[1],
        "ohe_like_columns": len(ohe_cols),
        "non_ohe_columns": len(non_ohe_cols),
    }
    return summary, cardinality


def display_chi2_selection_dashboard(
    chi2_results: pd.DataFrame,
    alpha_default: float = 0.02,
    min_cramers_default: float = 0.02,
) -> None:
    import ipywidgets as widgets
    import plotly.graph_objects as go
    from IPython.display import display

    alpha_slider = widgets.FloatSlider(
        value=alpha_default,
        min=0.001,
        max=0.10,
        step=0.001,
        description="alpha:",
        readout_format=".3f",
        layout=widgets.Layout(width="360px"),
    )
    cramers_slider = widgets.FloatSlider(
        value=min_cramers_default,
        min=0.01,
        max=0.30,
        step=0.01,
        description="min V:",
        readout_format=".2f",
        layout=widgets.Layout(width="360px"),
    )
    output = widgets.Output()

    def _render(_change=None):
        alpha = alpha_slider.value
        min_v = cramers_slider.value
        df = chi2_results.copy()
        df["Keep"] = (df["p_value"] < alpha) & (df["CramersV"] > min_v)
        keep_mask = df["Keep"]

        with output:
            output.clear_output(wait=True)
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=df[keep_mask]["Feature"],
                    y=df[keep_mask]["CramersV"],
                    name="Keep",
                    marker_color="#2ecc71",
                )
            )
            fig.add_trace(
                go.Bar(
                    x=df[~keep_mask]["Feature"],
                    y=df[~keep_mask]["CramersV"],
                    name="Drop",
                    marker_color="#e74c3c",
                )
            )
            fig.add_hline(
                y=min_v,
                line_dash="dash",
                line_color="gray",
                annotation_text=f"Min effect size ({min_v})",
                annotation_position="top right",
            )
            fig.update_layout(
                title=f"Feature Importance via Cramers V | alpha={alpha} | keep={keep_mask.sum()} drop={(~keep_mask).sum()}",
                xaxis_title="Feature",
                yaxis_title="Cramers V",
                xaxis_tickangle=-35,
                template="plotly_white",
                height=460,
            )
            fig.show()

            keep_list = df[keep_mask]["Feature"].tolist()
            drop_list = df[~keep_mask]["Feature"].tolist()
            print(f"KEEP ({len(keep_list)}): {keep_list}")
            print(f"DROP ({len(drop_list)}): {drop_list}")

    alpha_slider.observe(_render, names="value")
    cramers_slider.observe(_render, names="value")
    display(
        widgets.VBox(
            [
                widgets.HTML("<b>Chi-Square Feature Selection</b>"),
                widgets.HBox([alpha_slider, cramers_slider]),
                output,
            ]
        )
    )
    _render()


def save_final_feature_outputs(
    train_df_encoded: pd.DataFrame,
    test_df_encoded: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    y_train_binary: pd.Series,
    y_test_binary: pd.Series,
    encoder: ColumnTransformer,
    config: ProjectConfig | None = None,
) -> dict[str, Path]:
    cfg = config or get_config()
    artifacts = FeatureArtifacts(
        train_encoded=train_df_encoded,
        test_encoded=test_df_encoded,
        y_train=y_train,
        y_test=y_test,
        y_train_binary=y_train_binary,
        y_test_binary=y_test_binary,
        encoder=encoder,
    )
    save_feature_artifacts(artifacts, cfg)
    return {
        "train_encoded": cfg.train_encoded_path,
        "test_encoded": cfg.test_encoded_path,
        "y_train": cfg.y_train_path,
        "y_test": cfg.y_test_path,
        "y_train_binary": cfg.y_train_binary_path,
        "y_test_binary": cfg.y_test_binary_path,
        "encoder": cfg.encoder_path,
    }

## ADDING FEATURE ENGINNEREING FEATURES

def add_comorbidity_interaction(
    df: pd.DataFrame, diag_cols: list[str] | None = None
) -> pd.Series:
    """Build an order-agnostic diagnosis interaction string."""
    diag_cols = diag_cols or ["diag_1", "diag_2", "diag_3"]
    temp_df = df[diag_cols].fillna("Missing")
    return temp_df.apply(
        lambda row: "_".join(sorted(row.astype(str).str.replace(" ", ""))), axis=1
    )


def add_procedure_diversification(df: pd.DataFrame) -> pd.Series:
    """Procedures per diagnosis, clipped to avoid divide-by-zero."""
    return df["num_procedures"] / df["number_diagnoses"].clip(lower=1)


def add_interactive_features(df: pd.DataFrame, top_n_diag_combinations: int = 20) -> pd.DataFrame:
    """Create interaction-heavy categorical features in one place."""
    out = df.copy()
    out["age_specialty_interaction"] = apply_age_specialty_interaction(out)
    out["a1c_med_interaction"] = add_a1c_med_interaction(out)
    out["comorbidity_interaction"] = add_comorbidity_interaction(out)
    out = add_diag_combination(out, top_n=top_n_diag_combinations)
    return out

## ADDING FEATURE ENGINNEREING FEATURES

def add_domain_specific_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create domain-driven numerical features in one place."""
    out = df.copy()
    out["intensity_of_care"] = add_intensity_of_care(out)
    out["lab_med_ratio"] = add_lab_med_ratio(out)
    out["procedure_diversification"] = add_procedure_diversification(out)
    return out


def prepare_age_specialty_impact(
    train_df: pd.DataFrame, y_train: pd.Series, min_count: int = 50, top_n: int = 15
) -> pd.DataFrame:
    analysis_df = train_df[["age_specialty_interaction"]].copy()
    analysis_df["target"] = y_train
    analysis_df["is_readmitted"] = (analysis_df["target"] > 0).astype(int)

    feature_impact = (
        analysis_df.groupby("age_specialty_interaction")
        .agg(Risk_Rate=("is_readmitted", "mean"), Patient_Count=("is_readmitted", "count"))
        .reset_index()
    )
    reliable = feature_impact[feature_impact["Patient_Count"] >= min_count].sort_values(
        "Risk_Rate", ascending=False
    )
    return pd.concat([reliable.head(top_n), reliable.tail(top_n)])


def prepare_a1c_med_impact(train_df: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    vis_df = pd.DataFrame(
        {
            "A1Cresult": train_df["A1Cresult"],
            "Med_Adjusted": train_df["med_was_adjusted"],
            "Target": y_train,
        }
    )
    vis_df["Med_Status"] = vis_df["Med_Adjusted"].map(
        {1: "Adjusted (Up/Down)", 0: "No Change/Steady"}
    )
    vis_df["is_readmitted"] = (vis_df["Target"] > 0).astype(int)
    plot_data = (
        vis_df.groupby(["A1Cresult", "Med_Status"])["is_readmitted"]
        .agg(["mean", "count"])
        .reset_index()
    )
    plot_data.columns = ["A1C_Result", "Medication_Action", "Risk_Rate", "Sample_Size"]
    return plot_data


def prepare_comorbidity_triad_impact(
    train_df: pd.DataFrame, y_train: pd.Series, min_count: int = 100
) -> pd.DataFrame:
    diag_cols = ["diag_1", "diag_2", "diag_3"]
    vis_df = train_df[diag_cols].copy()
    vis_df["target"] = y_train
    vis_df["is_readmitted"] = (vis_df["target"] > 0).astype(int)
    vis_df["comorbidity_triad"] = vis_df[diag_cols].apply(
        lambda x: " + ".join(sorted(x.astype(str))), axis=1
    )
    triad_stats = (
        vis_df.groupby("comorbidity_triad")["is_readmitted"].agg(["mean", "count"]).reset_index()
    )
    triad_stats.columns = ["Triad", "Risk_Rate", "Sample_Size"]
    return triad_stats[triad_stats["Sample_Size"] >= min_count].sort_values(
        "Risk_Rate", ascending=False
    )


### PLOTTING FEATURE ENGINEERING FEATURES

def plot_age_specialty_impact(
    train_df: pd.DataFrame,
    y_train: pd.Series,
    benchmark: float = 0.112,
    min_count: int = 50,
    top_n: int = 15,
):
    import plotly.express as px

    plot_df = prepare_age_specialty_impact(train_df, y_train, min_count=min_count, top_n=top_n)
    fig = px.bar(
        plot_df,
        x="age_specialty_interaction",
        y="Risk_Rate",
        color="Risk_Rate",
        title="Feature Rationale: Risk Variance Across Age-Specialty Interactions",
        labels={
            "age_specialty_interaction": "Engineered Interaction Feature",
            "Risk_Rate": "Readmission Probability",
        },
        color_continuous_scale="RdBu_r",
        hover_data=["Patient_Count"],
    )
    fig.add_hline(
        y=benchmark, line_dash="dot", annotation_text=f"Benchmark ({benchmark:.1%})", line_color="black"
    )
    fig.update_layout(xaxis_tickangle=-45, height=600)
    return fig


def plot_a1c_med_impact(train_df: pd.DataFrame, y_train: pd.Series, benchmark: float = 0.112):
    import plotly.express as px

    plot_data = prepare_a1c_med_impact(train_df, y_train)
    fig = px.bar(
        plot_data,
        x="A1C_Result",
        y="Risk_Rate",
        color="Medication_Action",
        barmode="group",
        title="Rationale: Readmission Risk by A1C Result and Medication Adjustment",
        labels={"Risk_Rate": "Readmission Probability", "A1C_Result": "A1C Test Result"},
        text_auto=".1%",
        hover_data=["Sample_Size"],
    )
    fig.add_hline(
        y=benchmark,
        line_dash="dot",
        annotation_text=f"Hospital Avg ({benchmark:.1%})",
        line_color="black",
    )
    fig.update_layout(template="plotly_white", height=500)
    return fig


def plot_comorbidity_triad_impact(
    train_df: pd.DataFrame, y_train: pd.Series, benchmark: float = 0.112, min_count: int = 100
):
    import plotly.express as px

    reliable = prepare_comorbidity_triad_impact(train_df, y_train, min_count=min_count)
    fig = px.bar(
        reliable.head(15),
        x="Risk_Rate",
        y="Triad",
        orientation="h",
        color="Risk_Rate",
        title="Rationale: Readmission Risk by Comorbidity Triad",
        labels={"Risk_Rate": "Readmission Probability", "Triad": "Diagnosis Combination"},
        text_auto=".1%",
        color_continuous_scale="Reds",
    )
    fig.add_vline(
        x=benchmark, line_dash="dot", annotation_text="Hospital Avg", line_color="black"
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, template="plotly_white", height=600)
    return fig


def plot_intensity_of_care_effect(train_df: pd.DataFrame, y_train: pd.Series):
    import plotly.express as px

    vis_df = train_df.copy()
    vis_df["target"] = y_train
    vis_df["intensity_of_care"] = add_intensity_of_care(vis_df)
    vis_df["Readmission_Status"] = vis_df["target"].map({0: "No", 1: ">30 Days", 2: "<30 Days"})
    fig = px.box(
        vis_df,
        x="admission_type_id",
        y="intensity_of_care",
        color="Readmission_Status",
        title="Rationale: Healthcare Utilization and Service Intensity",
        notched=True,
        labels={
            "intensity_of_care": "Service Intensity (Procedures/Day)",
            "admission_type_id": "Admission Type",
        },
        category_orders={"Readmission_Status": ["No", ">30 Days", "<30 Days"]},
        color_discrete_map={"No": "#636EFA", ">30 Days": "#EF553B", "<30 Days": "#00CC96"},
    )
    fig.update_layout(template="plotly_white", height=600)
    return fig


def plot_lab_med_ratio_effect(train_df: pd.DataFrame):
    import plotly.express as px

    vis_df = train_df.copy()
    vis_df["lab_med_ratio"] = add_lab_med_ratio(vis_df)
    fig = px.scatter(
        vis_df,
        x="num_medications",
        y="num_lab_procedures",
        color="lab_med_ratio",
        size="time_in_hospital",
        hover_data=["age"],
        title="Rationale: Diagnostic vs. Maintenance Intensity (Lab/Med Ratio)",
        labels={
            "num_lab_procedures": "Total Lab Tests (Diagnostic Effort)",
            "num_medications": "Number of Medications (Management Effort)",
            "lab_med_ratio": "Lab/Med Ratio",
        },
        color_continuous_scale="RdBu_r",
        opacity=0.6,
    )
    avg_labs = vis_df["num_lab_procedures"].mean()
    fig.add_hline(
        y=avg_labs, line_dash="dash", annotation_text=f"Avg Labs: {avg_labs:.1f}", line_color="black"
    )
    fig.update_layout(template="plotly_white", height=600)
    return fig
