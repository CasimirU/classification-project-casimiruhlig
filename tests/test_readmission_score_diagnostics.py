from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.metrics import f1_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _require_paths(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        pytest.skip(
            "Required model-choice artifacts are missing. Run 03_model_choice.ipynb first. "
            f"Missing: {missing}"
        )


def _sanitize_feature_columns(columns: list[str]) -> list[str]:
    cleaned: list[str] = []
    seen: dict[str, int] = {}
    for raw in columns:
        base = re.sub(r"[^0-9a-zA-Z_]+", "_", str(raw)).strip("_")
        if not base:
            base = "feature"
        if base[0].isdigit():
            base = f"f_{base}"

        count = seen.get(base, 0)
        final = base if count == 0 else f"{base}_{count}"
        seen[base] = count + 1
        cleaned.append(final)
    return cleaned


def test_binary_readmission_f1_is_much_higher_than_multiclass_macro_f1() -> None:
    """Low macro F1 is mostly an objective mismatch, not a complete readmission failure."""
    x_test_path = PROJECT_ROOT / "data" / "final" / "test_encoded.csv"
    y_test_path = PROJECT_ROOT / "data" / "final" / "y_test.csv"
    y_test_binary_path = PROJECT_ROOT / "data" / "final" / "y_test_binary.csv"
    tuned_model_path = PROJECT_ROOT / "data" / "final" / "logisticregression_tuned.joblib"

    _require_paths([x_test_path, y_test_path, y_test_binary_path, tuned_model_path])

    x_test = pd.read_csv(x_test_path)
    x_test.columns = _sanitize_feature_columns(x_test.columns.tolist())
    y_test = pd.read_csv(y_test_path).squeeze("columns").astype(int)
    y_test_binary = pd.read_csv(y_test_binary_path).squeeze("columns").astype(int)

    model = joblib.load(tuned_model_path)
    y_pred_multiclass = model.predict(x_test)

    proba = model.predict_proba(x_test)
    classes = list(model.classes_)
    no_readmission_idx = classes.index(0)
    readmission_score = 1.0 - proba[:, no_readmission_idx]
    y_pred_binary = (readmission_score >= 0.5).astype(int)

    multiclass_macro_f1 = f1_score(y_test, y_pred_multiclass, average="macro")
    binary_f1 = f1_score(y_test_binary, y_pred_binary)

    assert binary_f1 > multiclass_macro_f1 + 0.20


def test_saved_reports_show_lt30_class_as_primary_bottleneck() -> None:
    """Class <30 metrics are the dominant reason macro score stays low."""
    report_path = PROJECT_ROOT / "output" / "model_choice" / "classification_reports.json"
    _require_paths([report_path])

    with report_path.open("r", encoding="utf-8") as handle:
        reports = json.load(handle)

    logistic = reports["LogisticRegression"]
    xgboost = reports["XGBoost"]

    assert logistic["2"]["precision"] < 0.25
    assert logistic["2"]["f1-score"] < logistic["1"]["f1-score"]
    assert xgboost["2"]["recall"] < 0.10


def test_confidence_threshold_table_shows_very_low_high_confidence_coverage() -> None:
    """At 0.8 confidence, model auto-decisions are extremely limited."""
    confidence_table = (
        PROJECT_ROOT
        / "src"
        / "eda"
        / "tables"
        / "03_S08_confidence_strategy_thresholds.csv"
    )
    _require_paths([confidence_table])

    df = pd.read_csv(confidence_table)
    row = df.loc[(df["threshold"] - 0.8).abs() < 1e-9].iloc[0]

    assert row["high_confidence_share"] < 0.01
    assert row["high_confidence_accuracy"] > row["low_confidence_accuracy"]
