from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
from sklearn.dummy import DummyClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import model_pipeline as mp


def test_rank_models_for_multiclass_priority_orders_by_recall_then_precision() -> None:
    cv_results = pd.DataFrame(
        [
            {
                "model": "A",
                "cv_recall_lt30_mean": 0.60,
                "cv_precision_macro_mean": 0.40,
                "cv_f1_macro_mean": 0.41,
                "cv_accuracy_mean": 0.50,
            },
            {
                "model": "B",
                "cv_recall_lt30_mean": 0.60,
                "cv_precision_macro_mean": 0.55,
                "cv_f1_macro_mean": 0.39,
                "cv_accuracy_mean": 0.58,
            },
            {
                "model": "C",
                "cv_recall_lt30_mean": 0.30,
                "cv_precision_macro_mean": 0.90,
                "cv_f1_macro_mean": 0.70,
                "cv_accuracy_mean": 0.92,
            },
        ]
    )

    ranked = mp.rank_models_for_multiclass_priority(cv_results)

    assert ranked["model"].tolist() == ["B", "A", "C"]


def test_rank_models_for_multiclass_priority_requires_expected_columns() -> None:
    incomplete = pd.DataFrame([{"model": "A", "cv_f1_macro_mean": 0.4}])

    try:
        mp.rank_models_for_multiclass_priority(incomplete)
    except ValueError as exc:
        assert "Missing ranking columns" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing ranking columns")


def test_fit_and_evaluate_test_emits_lt30_recall_and_precision_columns() -> None:
    # Tiny synthetic multiclass sample where class 2 exists.
    x_train = pd.DataFrame({"x1": [0, 1, 2, 3, 4, 5], "x2": [1, 1, 0, 0, 1, 0]})
    y_train = pd.Series([0, 0, 1, 1, 2, 2])

    x_test = pd.DataFrame({"x1": [0, 2, 4, 5], "x2": [1, 0, 1, 0]})
    y_test = pd.Series([0, 1, 2, 2])

    models = {
        "MostFrequent": DummyClassifier(strategy="most_frequent"),
        "Uniform": DummyClassifier(strategy="uniform", random_state=42),
    }

    test_results, _ = mp.fit_and_evaluate_test(
        models=models,
        X_train=x_train,
        y_train=y_train,
        X_test=x_test,
        y_test=y_test,
        lt30_label=2,
    )

    required_cols = {
        "model",
        "test_precision_macro",
        "test_precision_weighted",
        "test_recall_lt30",
        "test_f1_macro",
    }
    assert required_cols.issubset(set(test_results.columns))
    assert test_results.iloc[0]["test_recall_lt30"] >= test_results.iloc[-1]["test_recall_lt30"]
