from __future__ import annotations

from datetime import datetime, timezone
from importlib import import_module
import platform

import pandas as pd


DEFAULT_TARGET_LABELS = ("NO", ">30", "<30")


def assert_expected_schema(
    df: pd.DataFrame,
    required_cols: list[str],
    dataset_name: str,
) -> None:
    missing = sorted(set(required_cols) - set(df.columns))
    assert not missing, f"[{dataset_name}] Missing required columns: {missing}"


def assert_valid_target_labels(
    y: pd.Series,
    dataset_name: str,
    allowed_labels: tuple[object, ...] = DEFAULT_TARGET_LABELS,
) -> None:
    observed = set(pd.Series(y).dropna().unique().tolist())
    unexpected = sorted(observed - set(allowed_labels))
    assert not unexpected, f"[{dataset_name}] Unexpected target labels: {unexpected}"


def assert_no_missing_values(df: pd.DataFrame, dataset_name: str) -> None:
    missing = int(df.isna().sum().sum())
    assert missing == 0, f"[{dataset_name}] Found {missing} missing values."


def assert_feature_alignment(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    train_cols = list(train_df.columns)
    test_cols = list(test_df.columns)
    assert train_cols == test_cols, "Train/Test feature columns are misaligned."


def assert_no_leakage_by_id(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    id_cols: list[str],
) -> None:
    for col in id_cols:
        if col in train_df.columns and col in test_df.columns:
            overlap = set(train_df[col].dropna().tolist()) & set(test_df[col].dropna().tolist())
            assert len(overlap) == 0, f"Leakage detected by overlap in '{col}' ({len(overlap)} rows)."


def class_distribution_report(y: pd.Series) -> pd.DataFrame:
    counts = y.value_counts(dropna=False)
    pct = (counts / len(y) * 100).round(2)
    return pd.DataFrame({"count": counts, "percentage": pct})


def run_basic_data_quality_gates(
    train_df: pd.DataFrame,
    target_col: str,
    test_df: pd.DataFrame | None = None,
    required_cols: list[str] | None = None,
    allowed_labels: tuple[object, ...] = DEFAULT_TARGET_LABELS,
    leakage_id_cols: list[str] | None = None,
) -> pd.DataFrame:
    required = required_cols or [target_col]
    assert_expected_schema(train_df, required, "train_df")
    assert_valid_target_labels(train_df[target_col], "train_df", allowed_labels=allowed_labels)

    if test_df is not None:
        assert_expected_schema(test_df, required, "test_df")
        assert_valid_target_labels(test_df[target_col], "test_df", allowed_labels=allowed_labels)
        if leakage_id_cols:
            assert_no_leakage_by_id(train_df, test_df, leakage_id_cols)

    report = class_distribution_report(train_df[target_col]).reset_index()
    report = report.rename(columns={"index": "target_class"})
    report["dataset"] = "train"
    if test_df is not None:
        test_report = class_distribution_report(test_df[target_col]).reset_index()
        test_report = test_report.rename(columns={"index": "target_class"})
        test_report["dataset"] = "test"
        report = pd.concat([report, test_report], ignore_index=True)
    return report


def assert_target_mapping_complete(
    y_raw: pd.Series,
    mapping: dict[str, int],
    dataset_name: str,
) -> None:
    mapped = y_raw.map(mapping)
    assert not mapped.isna().any(), f"[{dataset_name}] Target mapping produced NaN values."


def build_reproducibility_footer(random_seed: int) -> pd.DataFrame:
    packages = ["numpy", "pandas", "sklearn", "matplotlib", "seaborn"]
    package_versions: dict[str, str] = {}
    for pkg in packages:
        module = import_module(pkg)
        package_versions[pkg] = getattr(module, "__version__", "unknown")

    footer = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "random_seed": int(random_seed),
        **package_versions,
    }
    return pd.DataFrame([footer])
