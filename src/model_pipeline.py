from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import re
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    IMBLEARN_AVAILABLE = True
    IMBLEARN_IMPORT_ERROR: Exception | None = None
except Exception as exc:
    # Fallback keeps the notebook runnable when imblearn/sklearn versions are incompatible.
    from sklearn.pipeline import Pipeline as ImbPipeline

    SMOTE = None
    IMBLEARN_AVAILABLE = False
    IMBLEARN_IMPORT_ERROR = exc
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    make_scorer,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve
from config import ProjectConfig, get_config
from sklearn.utils import resample

@dataclass
class ModelData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    y_train_binary: pd.Series | None = None
    y_test_binary: pd.Series | None = None


@dataclass
class ModelArtifacts:
    baseline_results: pd.DataFrame
    cv_results: pd.DataFrame
    test_results: pd.DataFrame
    best_model_name: str
    best_model: ImbPipeline
    model_reports: dict[str, dict[str, Any]]


def _safe_read_series(path: Path) -> pd.Series | None:
    if not path.exists():
        return None
    return pd.read_csv(path).squeeze("columns")


def _sanitize_feature_columns(columns: list[str]) -> list[str]:
    """Sanitize column names for libraries that enforce strict feature-name rules."""
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


def load_model_data(config: ProjectConfig, include_binary: bool = True) -> ModelData:
    X_train = pd.read_csv(config.train_encoded_path)
    X_test = pd.read_csv(config.test_encoded_path)

    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(
            f"Train/Test encoded feature count mismatch: {X_train.shape[1]} vs {X_test.shape[1]}"
        )
    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("Train/Test encoded feature columns are not aligned.")

    # Keep train/test column names safe and fully aligned for all candidate models.
    cleaned_columns = _sanitize_feature_columns(X_train.columns.tolist())
    if len(set(cleaned_columns)) != len(cleaned_columns):
        raise ValueError("Feature name sanitization produced duplicate columns.")
    X_train.columns = cleaned_columns
    X_test.columns = cleaned_columns

    y_train = pd.read_csv(config.y_train_path).squeeze("columns").astype(int)
    y_test = pd.read_csv(config.y_test_path).squeeze("columns").astype(int)

    y_train_binary = None
    y_test_binary = None
    if include_binary:
        y_train_binary = _safe_read_series(config.y_train_binary_path)
        y_test_binary = _safe_read_series(config.y_test_binary_path)
        if y_train_binary is not None:
            y_train_binary = y_train_binary.astype(int)
        if y_test_binary is not None:
            y_test_binary = y_test_binary.astype(int)

    return ModelData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        y_train_binary=y_train_binary,
        y_test_binary=y_test_binary,
    )


def establish_dummy_baselines(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = 5,
    random_state: int = 42,
    n_jobs: int = -1,
) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scoring = {
        "f1_macro": "f1_macro",
        "recall_macro": "recall_macro",
        "accuracy": "accuracy",
    }

    rows: list[dict[str, float | str]] = []
    for strategy in ["most_frequent", "stratified", "uniform"]:
        model = DummyClassifier(strategy=strategy, random_state=random_state)
        scores = _cross_validate_with_fallback(
            estimator=model,
            X_train=X_train,
            y_train=y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
        )
        rows.append(
            {
                "model": f"Dummy::{strategy}",
                "cv_f1_macro_mean": float(scores["test_f1_macro"].mean()),
                "cv_recall_macro_mean": float(scores["test_recall_macro"].mean()),
                "cv_accuracy_mean": float(scores["test_accuracy"].mean()),
            }
        )

    return pd.DataFrame(rows).sort_values("cv_f1_macro_mean", ascending=False)


def _cross_validate_with_fallback(
    estimator: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: Any,
    scoring: dict[str, Any],
    n_jobs: int = -1,
) -> dict[str, np.ndarray]:
    try:
        return cross_validate(
            estimator,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            return_train_score=False,
        )
    except PermissionError:
        if n_jobs == 1:
            raise
        warnings.warn(
            "Parallel cross-validation is not available in this environment. "
            "Falling back to n_jobs=1.",
            RuntimeWarning,
        )
        return cross_validate(
            estimator,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            return_train_score=False,
        )


def build_model_candidates(random_state: int = 42) -> dict[str, ImbPipeline]:
    smote_steps: list[tuple[str, Any]] = []
    if IMBLEARN_AVAILABLE and SMOTE is not None:
        smote_steps.append(("smote", SMOTE(random_state=random_state)))
    else:
        warnings.warn(
            "imblearn is unavailable or incompatible; proceeding without SMOTE. "
            f"Import error: {IMBLEARN_IMPORT_ERROR}",
            RuntimeWarning,
        )

    models: dict[str, ImbPipeline] = {
        "LogisticRegression": ImbPipeline(
            steps=[
                *smote_steps,
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1500,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "RandomForest": ImbPipeline(
            steps=[
                *smote_steps,
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=None,
                        class_weight="balanced",
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }

    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = ImbPipeline(
            steps=[
                *smote_steps,
                (
                    "model",
                    XGBClassifier(
                        n_estimators=300,
                        learning_rate=0.05,
                        max_depth=6,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        objective="multi:softprob",
                        eval_metric="mlogloss",
                        random_state=random_state,
                    ),
                ),
            ]
        )
    except Exception:
        # Keep notebook runnable even when xgboost is unavailable.
        pass

    return models


def _build_multiclass_scoring(lt30_label: int = 2) -> dict[str, Any]:
    return {
        "f1_macro": "f1_macro",
        "precision_macro": "precision_macro",
        "precision_weighted": "precision_weighted",
        "recall_macro": "recall_macro",
        "recall_lt30": make_scorer(
            recall_score,
            labels=[lt30_label],
            average="macro",
            zero_division=0,
        ),
        "accuracy": "accuracy",
    }


def rank_models_for_multiclass_priority(cv_results: pd.DataFrame) -> pd.DataFrame:
    """Rank models for multiclass use-cases prioritizing <30 recall then precision."""
    required = {
        "model",
        "cv_recall_lt30_mean",
        "cv_precision_macro_mean",
        "cv_f1_macro_mean",
        "cv_accuracy_mean",
    }
    missing = required - set(cv_results.columns)
    if missing:
        raise ValueError(f"Missing ranking columns in cv_results: {sorted(missing)}")

    return cv_results.sort_values(
        [
            "cv_recall_lt30_mean",
            "cv_precision_macro_mean",
            "cv_f1_macro_mean",
            "cv_accuracy_mean",
        ],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def evaluate_models_cv(
    models: dict[str, ImbPipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = 5,
    random_state: int = 42,
    lt30_label: int = 2,
    n_jobs: int = -1,
) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scoring = _build_multiclass_scoring(lt30_label=lt30_label)

    rows: list[dict[str, float | str]] = []
    for model_name, pipeline in models.items():
        scores = _cross_validate_with_fallback(
            estimator=pipeline,
            X_train=X_train,
            y_train=y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
        )

        rows.append(
            {
                "model": model_name,
                "cv_f1_macro_mean": float(scores["test_f1_macro"].mean()),
                "cv_f1_macro_std": float(scores["test_f1_macro"].std()),
                "cv_precision_macro_mean": float(scores["test_precision_macro"].mean()),
                "cv_precision_weighted_mean": float(scores["test_precision_weighted"].mean()),
                "cv_recall_macro_mean": float(scores["test_recall_macro"].mean()),
                "cv_recall_lt30_mean": float(scores["test_recall_lt30"].mean()),
                "cv_accuracy_mean": float(scores["test_accuracy"].mean()),
            }
        )

    return rank_models_for_multiclass_priority(pd.DataFrame(rows))


def fit_and_evaluate_test(
    models: dict[str, ImbPipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    lt30_label: int = 2,
) -> tuple[pd.DataFrame, dict[str, ImbPipeline]]:
    rows: list[dict[str, float | str]] = []
    fitted_models: dict[str, ImbPipeline] = {}

    for model_name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        record: dict[str, float | str] = {
            "model": model_name,
            "test_accuracy": float(accuracy_score(y_test, y_pred)),
            "test_precision_macro": float(
                precision_score(y_test, y_pred, average="macro", zero_division=0)
            ),
            "test_precision_weighted": float(
                precision_score(y_test, y_pred, average="weighted", zero_division=0)
            ),
            "test_recall_macro": float(recall_score(y_test, y_pred, average="macro")),
            "test_recall_lt30": float(
                recall_score(y_test, y_pred, labels=[lt30_label], average="macro", zero_division=0)
            ),
            "test_f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        }

        if hasattr(pipeline, "predict_proba"):
            y_proba = pipeline.predict_proba(X_test)
            record["test_auc_ovr_macro"] = float(
                roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
            )

        fitted_models[model_name] = pipeline
        rows.append(record)

    test_results = pd.DataFrame(rows).sort_values(
        ["test_recall_lt30", "test_precision_macro", "test_f1_macro", "test_accuracy"],
        ascending=[False, False, False, False],
    )
    return test_results, fitted_models


def get_readmission_probability(model: ImbPipeline, X: pd.DataFrame) -> np.ndarray:
    if not hasattr(model, "predict_proba"):
        raise ValueError("Model does not support predict_proba.")
    proba = model.predict_proba(X)
    if proba.shape[1] == 2:
        return proba[:, 1]
    # Multiclass setting: class 0 = NO readmission. Use complement as readmission probability.
    return 1.0 - proba[:, 0]


def compute_binary_pr_curve(
    model: ImbPipeline,
    X: pd.DataFrame,
    y_binary: pd.Series,
) -> tuple[pd.DataFrame, float]:
    scores = get_readmission_probability(model, X)
    precision, recall, thresholds = precision_recall_curve(y_binary, scores)
    ap = float(average_precision_score(y_binary, scores))
    pr_df = pd.DataFrame(
        {
            "precision": precision[:-1],
            "recall": recall[:-1],
            "threshold": thresholds,
        }
    )
    return pr_df, ap

def analyze_errors(model, X_test, y_test):
    """Find patterns in model errors"""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # Identify errors
    errors = (y_test != y_pred)

    print(f"Error Rate: {errors.mean():.3f}")
    print(f"Total Errors: {errors.sum()} / {len(y_test)}")

    # Confidence analysis
    max_confidence = y_proba.max(axis=1)
    correct_confidence = max_confidence[~errors].mean()
    error_confidence = max_confidence[errors].mean()

    print(f"Average confidence (correct): {correct_confidence:.3f}")
    print(f"Average confidence (errors): {error_confidence:.3f}")

    # Low confidence predictions
    low_confidence = max_confidence < 0.6
    print(f"Low confidence predictions: {low_confidence.sum()} ({low_confidence.mean():.1%})")

## Feature-Based Error Analysis
def feature_error_patterns(X_test, y_test, y_pred, top_features):
    """Analyze how features relate to errors"""
    errors = (y_test != y_pred)

    print("Feature Analysis for Errors:")
    for feature in top_features[:3]:  # Top 3 features
        correct_mean = X_test[feature][~errors].mean()
        error_mean = X_test[feature][errors].mean()

        print(f"\n{feature}:")
        print(f"  Correct predictions: {correct_mean:.3f}")
        print(f"  Error predictions: {error_mean:.3f}")
        print(f"  Difference: {abs(correct_mean - error_mean):.3f}")

##Confidence-Based Decision Framework
def confidence_strategy(model, X_test, confidence_threshold=0.8):
    """Design confidence-based business strategy"""
    y_proba = model.predict_proba(X_test)
    max_confidence = y_proba.max(axis=1)

    high_confidence = max_confidence >= confidence_threshold
    low_confidence = max_confidence < confidence_threshold

    print(f"Business Strategy (threshold: {confidence_threshold}):")
    print(f"Auto-process: {high_confidence.sum()} samples ({high_confidence.mean():.1%})")
    print(f"Manual review: {low_confidence.sum()} samples ({low_confidence.mean():.1%})")
    print(f"Workload reduction: {high_confidence.mean():.1%}")

##Simple Calibration Analysis
def check_calibration(model, X_test, y_test):
    """Simple calibration analysis"""
    y_proba = get_readmission_probability(model, X_test)

    # Calculate calibration
    fraction_positives, mean_predicted = calibration_curve(
        y_test, y_proba, n_bins=10
    )

    # Plot calibration curve
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], 'k:', label='Perfect calibration')
    plt.plot(mean_predicted, fraction_positives, 's-', label='Model')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Plot')
    plt.legend()
    plt.show()

    # Calculate calibration error
    calibration_error = abs(fraction_positives - mean_predicted).mean()
    print(f"Calibration Error: {calibration_error:.3f}")

    if calibration_error < 0.05:
        print("✅ Well calibrated")
    elif calibration_error < 0.1:
        print("⚠️ Reasonably calibrated")
    else:
        print("🔴 Poorly calibrated")


##Multi-Class Performance Analysis

def compute_error_analysis(
    model: ImbPipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    confidence_threshold: float = 0.8,
) -> dict[str, Any]:
    y_pred = model.predict(X_test)
    errors = y_test != y_pred

    result: dict[str, Any] = {
        "error_rate": float(errors.mean()),
        "total_errors": int(errors.sum()),
        "total_rows": int(len(y_test)),
    }

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
        max_conf = y_proba.max(axis=1)
        high_conf = max_conf >= confidence_threshold
        low_conf = ~high_conf
        result.update(
            {
                "high_confidence_share": float(high_conf.mean()),
                "high_confidence_accuracy": float((y_pred[high_conf] == y_test[high_conf]).mean())
                if high_conf.any()
                else float("nan"),
                "low_confidence_accuracy": float((y_pred[low_conf] == y_test[low_conf]).mean())
                if low_conf.any()
                else float("nan"),
            }
        )

    return result




def collect_model_reports(
    fitted_models: dict[str, ImbPipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for model_name, model in fitted_models.items():
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        reports[model_name] = report
    return reports


def save_model_choice_artifacts(
    baseline_results: pd.DataFrame,
    cv_results: pd.DataFrame,
    test_results: pd.DataFrame,
    model_reports: dict[str, dict[str, Any]],
    best_model: ImbPipeline,
    best_model_name: str,
    config: ProjectConfig,
) -> dict[str, Path]:
    output_dir = config.output_dir / "model_choice"
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_path = output_dir / "baseline_results.csv"
    cv_path = output_dir / "cv_results.csv"
    test_path = output_dir / "test_results.csv"
    report_path = output_dir / "classification_reports.json"
    best_model_summary_path = output_dir / "best_model.txt"

    baseline_results.to_csv(baseline_path, index=False)
    cv_results.to_csv(cv_path, index=False)
    test_results.to_csv(test_path, index=False)

    pd.Series({"best_model": best_model_name}).to_csv(best_model_summary_path, index=False)

    import json

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(model_reports, f, indent=2)

    joblib.dump(best_model, config.model_path)

    return {
        "baseline_results": baseline_path,
        "cv_results": cv_path,
        "test_results": test_path,
        "classification_reports": report_path,
        "best_model_summary": best_model_summary_path,
        "best_model": config.model_path,
    }


def run_model_pipeline(
    config: ProjectConfig | None = None,
    cv_folds: int = 5,
    include_baselines: bool = True,
    save_artifacts: bool = True,
    n_jobs: int = -1,
) -> dict[str, Any]:
    cfg = config or get_config()
    data = load_model_data(cfg, include_binary=True)

    baseline_results = pd.DataFrame()
    if include_baselines:
        baseline_results = establish_dummy_baselines(
            data.X_train,
            data.y_train,
            cv_folds=cv_folds,
            random_state=cfg.random_state,
            n_jobs=n_jobs,
        )

    models = build_model_candidates(cfg.random_state)
    lt30_label = int(cfg.multiclass_map.get("<30", 2))
    cv_results = evaluate_models_cv(
        models,
        data.X_train,
        data.y_train,
        cv_folds=cv_folds,
        random_state=cfg.random_state,
        lt30_label=lt30_label,
        n_jobs=n_jobs,
    )
    test_results, fitted_models = fit_and_evaluate_test(
        models,
        data.X_train,
        data.y_train,
        data.X_test,
        data.y_test,
        lt30_label=lt30_label,
    )

    best_model_name = str(cv_results.iloc[0]["model"])
    best_model = fitted_models[best_model_name]

    model_reports = collect_model_reports(fitted_models, data.X_test, data.y_test)

    saved_paths: dict[str, Path] = {}
    if save_artifacts:
        saved_paths = save_model_choice_artifacts(
            baseline_results=baseline_results,
            cv_results=cv_results,
            test_results=test_results,
            model_reports=model_reports,
            best_model=best_model,
            best_model_name=best_model_name,
            config=cfg,
        )

    artifacts = ModelArtifacts(
        baseline_results=baseline_results,
        cv_results=cv_results,
        test_results=test_results,
        best_model_name=best_model_name,
        best_model=best_model,
        model_reports=model_reports,
    )

    return {
        "data": data,
        "baseline_results": artifacts.baseline_results,
        "cv_results": artifacts.cv_results,
        "test_results": artifacts.test_results,
        "fitted_models": fitted_models,
        "best_model_name": artifacts.best_model_name,
        "best_model": artifacts.best_model,
        "model_reports": artifacts.model_reports,
        "saved_paths": saved_paths,
    }


## Feature Importance Analysis
def analyze_feature_importance(model, X_test, feature_names):
    """Extract and analyze feature importance"""

    # Method 1: Built-in importance (for tree-based models)
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        method = "Built-in importance"
    elif hasattr(model, 'coef_'):
        importance = abs(model.coef_[0])
        method = "Coefficient magnitude"
    else:
        print("Model doesn't have built-in feature importance")
        return None

    # Create importance ranking
    feature_importance = list(zip(feature_names, importance))
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    print(f"Feature Importance ({method}):")
    for i, (feature, imp) in enumerate(feature_importance[:10]):
        print(f"{i + 1:2d}. {feature}: {imp:.4f}")

    return feature_importance

##Advanced Evalutation Techniques
### Bootstrap Confidence Intervals
def bootstrap_confidence_intervals(model, X_test, y_test, n_bootstrap=1000):
    """Calculate confidence intervals for performance metrics"""

    bootstrap_scores = []

    for i in range(n_bootstrap):
        # Bootstrap sample
        X_boot, y_boot = resample(X_test, y_test, random_state=i)

        # Calculate metric
        y_pred_boot = model.predict(X_boot)
        score = accuracy_score(y_boot, y_pred_boot)
        bootstrap_scores.append(score)

    # Calculate confidence intervals
    confidence_level = 0.95
    alpha = 1 - confidence_level
    lower = np.percentile(bootstrap_scores, (alpha / 2) * 100)
    upper = np.percentile(bootstrap_scores, (1 - alpha / 2) * 100)
    mean_score = np.mean(bootstrap_scores)

    print(f"Bootstrap Confidence Interval ({confidence_level:.0%}):")
    print(f"Mean Accuracy: {mean_score:.3f}")
    print(f"95% CI: [{lower:.3f}, {upper:.3f}]")

    return mean_score, (lower, upper)
