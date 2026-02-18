from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from config import ProjectConfig, get_config


@dataclass
class LoadingArtifacts:
    raw_df: pd.DataFrame
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    class_distribution: pd.DataFrame


def _build_class_distribution(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    counts = df[target_col].value_counts(dropna=False)
    percentages = (counts / len(df) * 100).round(2)
    distribution = pd.DataFrame({"count": counts, "percentage": percentages})
    return distribution


def load_raw_data(config: ProjectConfig) -> pd.DataFrame:
    return pd.read_csv(config.raw_data_path)


def split_train_test(
    df: pd.DataFrame,
    config: ProjectConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_col = config.target_col
    X = df.drop(columns=[target_col])
    y = df[target_col]

    stratify = y if config.stratify_split else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=stratify,
    )

    train_df = X_train.copy()
    test_df = X_test.copy()
    train_df[target_col] = y_train
    test_df[target_col] = y_test

    return train_df, test_df


def save_interim_sets(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: ProjectConfig,
) -> None:
    train_df.to_csv(config.interim_train_path, index=False)
    test_df.to_csv(config.interim_test_path, index=False)


def run_loading_pipeline(
    config: ProjectConfig | None = None,
    save: bool = True,
) -> LoadingArtifacts:
    cfg = config or get_config()
    raw_df = load_raw_data(cfg)
    train_df, test_df = split_train_test(raw_df, cfg)

    if save:
        save_interim_sets(train_df, test_df, cfg)

    class_distribution = _build_class_distribution(raw_df, cfg.target_col)
    return LoadingArtifacts(
        raw_df=raw_df,
        train_df=train_df,
        test_df=test_df,
        class_distribution=class_distribution,
    )
