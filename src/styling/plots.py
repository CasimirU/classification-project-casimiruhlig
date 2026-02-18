from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency

from styling.styling import save_figure


def _to_path(path: Path | str | None) -> Path | None:
    if path is None:
        return None
    return Path(path)


def _annotate_bars(ax, min_height: float = 0.0, y_offset: int = 7, fontsize: int = 9) -> None:
    for patch in ax.patches:
        height = patch.get_height()
        if height > min_height:
            ax.annotate(
                f"{height:.1f}%",
                (patch.get_x() + patch.get_width() / 2.0, height),
                ha="center",
                va="center",
                xytext=(0, y_offset),
                textcoords="offset points",
                fontsize=fontsize,
            )


def _add_benchmarks(avg_under_30: float, avg_over_30: float) -> None:
    plt.axhline(
        y=avg_under_30,
        color="#c0392b",
        linestyle="--",
        linewidth=1.5,
        label=f"Avg <30 ({avg_under_30}%)",
    )
    plt.axhline(
        y=avg_over_30,
        color="#2980b9",
        linestyle="--",
        linewidth=1.5,
        label=f"Avg >30 ({avg_over_30}%)",
    )


def plot_target_distribution(
    df: pd.DataFrame,
    target_col: str = "readmitted",
    order: Iterable[str] | None = None,
    threshold_pct: float | None = None,
    save_path: Path | str | None = None,
):
    order = list(order) if order else ["NO", ">30", "<30"]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(
        data=df,
        x=target_col,
        hue=target_col,
        order=order,
        palette="viridis",
        ax=ax,
    )

    total_rows = len(df)
    if threshold_pct is not None:
        threshold = total_rows * threshold_pct
        ax.axhline(
            threshold,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"{int(threshold_pct * 100)}% threshold ({int(threshold)})",
        )

    for patch in ax.patches:
        pct = 100 * patch.get_height() / total_rows
        x = patch.get_x() + patch.get_width() / 2
        y = patch.get_height()
        ax.annotate(
            f"{pct:.1f}%",
            (x, y),
            ha="center",
            va="bottom",
            fontsize=10,
            xytext=(0, 4),
            textcoords="offset points",
        )

    ax.set_title("Distribution of Readmission Classes", fontsize=15)
    ax.set_xlabel("Readmitted Status", fontsize=12)
    ax.set_ylabel("Number of Samples", fontsize=12)
    ax.legend(loc="best")

    save_path = _to_path(save_path)
    if save_path:
        save_figure(fig, save_path)

    return ax


def plot_missing_values(
    df: pd.DataFrame,
    top_n: int = 20,
    save_path: Path | str | None = None,
):
    missing_pct = df.replace("?", pd.NA).isna().mean().mul(100).sort_values(ascending=False)
    missing_pct = missing_pct[missing_pct > 0].head(top_n)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=missing_pct.values, y=missing_pct.index, palette="mako", ax=ax)
    ax.set_title(f"Top {top_n} Columns by Missing Percentage")
    ax.set_xlabel("Missing (%)")
    ax.set_ylabel("Column")

    save_path = _to_path(save_path)
    if save_path:
        save_figure(fig, save_path)

    return ax


def plot_numeric_distributions(
    df: pd.DataFrame,
    columns: list[str],
    n_cols: int = 3,
    save_path: Path | str | None = None,
):
    valid_cols = [col for col in columns if col in df.columns]
    n_rows = (len(valid_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx, column in enumerate(valid_cols):
        sns.histplot(df[column], kde=True, ax=axes[idx], color="#3b8bc2")
        axes[idx].set_title(column)

    for idx in range(len(valid_cols), len(axes)):
        axes[idx].axis("off")

    save_path = _to_path(save_path)
    if save_path:
        save_figure(fig, save_path)

    return axes


def plot_numeric_kde_by_readmitted(
    data: pd.DataFrame,
    num_cols: list[str],
    hue: str = "readmitted",
    n_cols: int = 3,
    save_path: Path | str | None = None,
):
    n_rows = (len(num_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(25, 7 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(num_cols):
        sns.kdeplot(
            data=data,
            x=col,
            hue=hue,
            fill=True,
            common_norm=False,
            palette="viridis",
            ax=axes[i],
        )
        axes[i].set_title(f"Density of {col}")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    save_path = _to_path(save_path)
    if save_path:
        save_figure(fig, save_path)

    return axes


def plot_correlation_heatmap(
    data: pd.DataFrame,
    num_cols: list[str],
    target_col: str = "readmitted",
    target_map: dict[str, int] | None = None,
    save_path: Path | str | None = None,
) -> pd.DataFrame:
    target_map = target_map or {"NO": 0, ">30": 1, "<30": 2}
    plot_df = data[num_cols].copy()
    plot_df[f"{target_col}_target"] = data[target_col].map(target_map)
    corr_matrix = plot_df.corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        square=True,
        ax=ax,
    )
    ax.set_title("Correlation Heatmap: Clinical Features vs Readmission", fontsize=15)

    save_path = _to_path(save_path)
    if save_path:
        save_figure(fig, save_path)

    return corr_matrix


def plot_categorical_impact(
    df: pd.DataFrame,
    column: str,
    target: str = "readmitted",
    save_path: Path | str | None = None,
):
    props = df.groupby(column)[target].value_counts(normalize=True).unstack() * 100
    if "<30" in props.columns:
        props = props.sort_values(by="<30", ascending=False)

    ax = props.plot(kind="bar", stacked=True, figsize=(12, 6), colormap="viridis")
    plt.title(f"Risk Profile: {column} vs {target}", fontsize=14, fontweight="bold")
    plt.ylabel("Percentage of Patients (%)")
    plt.xlabel(column.replace("_", " ").title())
    plt.xticks(rotation=45, ha="right")
    plt.legend(title=target, bbox_to_anchor=(1.05, 1), loc="upper left")

    save_path = _to_path(save_path)
    if save_path:
        save_figure(plt.gcf(), save_path)

    return ax


def plot_benchmark_grouped_risk(
    df: pd.DataFrame,
    group_col: str,
    avg_under_30: float = 11.2,
    avg_over_30: float = 35.2,
    title: str | None = None,
    x_label: str | None = None,
    rotate_xticks: int = 45,
    save_path: Path | str | None = None,
):
    risk_data = (
        df.groupby(group_col)["readmitted"]
        .value_counts(normalize=True)
        .unstack()
        .fillna(0)
        * 100
    )

    for col in ["<30", ">30"]:
        if col not in risk_data.columns:
            risk_data[col] = 0

    plot_df = risk_data[["<30", ">30"]].reset_index()
    plot_df_melted = plot_df.melt(id_vars=group_col, var_name="Readmission_Type", value_name="Percentage")

    plt.figure(figsize=(12, 7))
    ax = sns.barplot(
        data=plot_df_melted,
        x=group_col,
        y="Percentage",
        hue="Readmission_Type",
        palette=["#e74c3c", "#3498db"],
    )

    _add_benchmarks(avg_under_30, avg_over_30)
    plt.title(title or f"Readmission Risk Profiles by {group_col.title()}", fontsize=15, fontweight="bold")
    plt.ylabel("Percentage of Patients (%)")
    plt.xlabel(x_label or group_col.title())
    plt.xticks(rotation=rotate_xticks, ha="right")
    plt.legend(title="Category", bbox_to_anchor=(1.05, 1), loc="upper left")

    _annotate_bars(ax)

    save_path = _to_path(save_path)
    if save_path:
        save_figure(plt.gcf(), save_path)

    return ax


def plot_med_adjustment_binary(
    df: pd.DataFrame,
    avg_under_30: float = 11.2,
    avg_over_30: float = 35.2,
    save_path: Path | str | None = None,
):
    adj_risk = (
        df.groupby("med_adjusted_binary")["readmitted"]
        .value_counts(normalize=True)
        .unstack()
        .fillna(0)
        * 100
    )

    for col in ["<30", ">30"]:
        if col not in adj_risk.columns:
            adj_risk[col] = 0

    plot_adj = adj_risk[["<30", ">30"]].reset_index()
    plot_adj_melted = plot_adj.melt(id_vars="med_adjusted_binary", var_name="Type", value_name="Pct")

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=plot_adj_melted, x="med_adjusted_binary", y="Pct", hue="Type", palette="Set2")
    _add_benchmarks(avg_under_30, avg_over_30)
    plt.title("Readmission Risk: Was any Medication Adjusted?", fontweight="bold")
    plt.xlabel("Medication Adjusted (0=No, 1=Yes)")
    plt.ylabel("Percentage of Patients (%)")
    plt.legend(title="Readmission", bbox_to_anchor=(1.05, 1))

    save_path = _to_path(save_path)
    if save_path:
        save_figure(plt.gcf(), save_path)

    return ax


def plot_med_adjustment_panels(
    df: pd.DataFrame,
    order: list[str] | None = None,
    save_path: Path | str | None = None,
):
    order = order or ["NO", ">30", "<30"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    sns.boxplot(ax=axes[0], data=df, x="readmitted", y="adj_med_ratio", order=order, palette="mako")
    axes[0].set_title("Adjustment Ratio Distribution", fontweight="bold")
    axes[0].set_ylabel("Ratio (Adjustments / Total Meds)")

    sns.pointplot(ax=axes[1], data=df, x="readmitted", y="count_adjusted_up", order=order, color="#e74c3c")
    axes[1].set_title("Avg. Medications Adjusted UP", fontweight="bold")
    axes[1].set_ylabel('Count of "Up" Changes')

    sns.pointplot(ax=axes[2], data=df, x="readmitted", y="count_adjusted_down", order=order, color="#3498db")
    axes[2].set_title("Avg. Medications Adjusted DOWN", fontweight="bold")
    axes[2].set_ylabel('Count of "Down" Changes')

    save_path = _to_path(save_path)
    if save_path:
        save_figure(fig, save_path)

    return axes


def plot_medication_risk_yes_only(
    df: pd.DataFrame,
    med_cols: list[str],
    avg_under_30: float = 11.2,
    avg_over_30: float = 35.2,
    save_path: Path | str | None = None,
):
    work_df = df.copy()
    for col in med_cols:
        work_df[col] = work_df[col].apply(lambda x: "No" if str(x).strip() == "No" else "Yes")

    melted = work_df.melt(
        id_vars=["readmitted"],
        value_vars=med_cols,
        var_name="Medication",
        value_name="Taking_Med",
    )
    taking_meds = melted[melted["Taking_Med"] == "Yes"].copy()

    med_risk = taking_meds.groupby("Medication")["readmitted"].value_counts(normalize=True).unstack().fillna(0)
    for col in ["<30", ">30", "NO"]:
        if col not in med_risk.columns:
            med_risk[col] = 0.0
    med_risk = (med_risk * 100).sort_values(by="<30", ascending=False)

    plot_df = med_risk[["<30", ">30"]].reset_index()
    plot_df_melted = plot_df.melt(id_vars="Medication", var_name="Readmission_Type", value_name="Percentage")

    plt.figure(figsize=(16, 8))
    ax = sns.barplot(
        data=plot_df_melted,
        x="Medication",
        y="Percentage",
        hue="Readmission_Type",
        palette=["#e74c3c", "#3498db"],
    )

    plt.axhline(
        y=avg_under_30,
        color="#c0392b",
        linestyle="--",
        linewidth=2,
        label=f"Avg <30 ({avg_under_30}%)",
    )
    plt.axhline(
        y=avg_over_30,
        color="#2980b9",
        linestyle="--",
        linewidth=2,
        label=f"Avg >30 ({avg_over_30}%)",
    )

    plt.title("Readmission Risk by Medication (Taking = Yes)", fontsize=16, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Readmission Rate (%)")
    plt.legend(title="Category", bbox_to_anchor=(1.05, 1))

    save_path = _to_path(save_path)
    if save_path:
        save_figure(plt.gcf(), save_path)

    return ax


def plot_a1c_bin_distribution(
    df: pd.DataFrame,
    avg_under_30: float = 11.2,
    avg_over_30: float = 35.2,
    a1c_order: list[str] | None = None,
    save_path: Path | str | None = None,
):
    a1c_order = a1c_order or ["Missing", "Norm", ">7", ">8"]
    work_df = df.copy()
    work_df["A1Cresult"] = work_df["A1Cresult"].fillna("Missing")

    plot_data = work_df.groupby(["A1Cresult", "readmitted"]).size().reset_index(name="count")
    bin_totals = work_df.groupby("A1Cresult").size().reset_index(name="bin_total")
    plot_data = plot_data.merge(bin_totals, on="A1Cresult")
    plot_data["Percentage"] = (plot_data["count"] / plot_data["bin_total"]) * 100

    plt.figure(figsize=(16, 8))
    ax = sns.barplot(
        data=plot_data,
        x="A1Cresult",
        y="Percentage",
        hue="readmitted",
        order=a1c_order,
        palette={"<30": "#e74c3c", ">30": "#3498db", "NO": "#2ecc71"},
    )

    plt.axhline(
        y=avg_under_30,
        color="#c0392b",
        linestyle="--",
        linewidth=2,
        label=f"Global Avg <30 ({avg_under_30}%)",
    )
    plt.axhline(
        y=avg_over_30,
        color="#2980b9",
        linestyle="--",
        linewidth=2,
        label=f"Global Avg >30 ({avg_over_30}%)",
    )

    plt.title("Readmission Risk by A1C Result (Percentage within Bin)", fontsize=16, fontweight="bold")
    plt.xlabel("A1C Test Result", fontsize=12)
    plt.ylabel("Percentage of Admissions (%)", fontsize=12)
    plt.legend(title="Readmission Status", bbox_to_anchor=(1.05, 1), loc="upper left")

    _annotate_bars(ax, min_height=0, y_offset=9, fontsize=10)

    save_path = _to_path(save_path)
    if save_path:
        save_figure(plt.gcf(), save_path)

    return ax


def plot_specialty_logic_risk(
    df: pd.DataFrame,
    avg_under_30: float = 11.2,
    avg_over_30: float = 35.2,
    save_path: Path | str | None = None,
):
    logic_stats = df.groupby(["specialty_logic", "readmitted"]).size().unstack(fill_value=0)
    logic_pct = logic_stats.div(logic_stats.sum(axis=1), axis=0) * 100
    logic_pct = logic_pct.reset_index().melt(id_vars="specialty_logic", var_name="Readmission", value_name="Percentage")

    plt.figure(figsize=(16, 8))
    ax = sns.barplot(
        data=logic_pct,
        x="specialty_logic",
        y="Percentage",
        hue="Readmission",
        palette={"<30": "#e74c3c", ">30": "#3498db", "NO": "#2ecc71"},
    )

    plt.axhline(
        y=avg_under_30,
        color="#c0392b",
        linestyle="--",
        linewidth=2,
        label=f"Avg <30 ({avg_under_30}%)",
    )
    plt.axhline(
        y=avg_over_30,
        color="#2980b9",
        linestyle="--",
        linewidth=2,
        label=f"Avg >30 ({avg_over_30}%)",
    )

    plt.title("Readmission Risk by Clinical Groupings", fontsize=16, fontweight="bold")
    plt.ylabel("Percentage within Group (%)")
    plt.xticks(rotation=30, ha="right")
    plt.legend(title="Status", bbox_to_anchor=(1.05, 1))

    _annotate_bars(ax, min_height=0, y_offset=5)

    save_path = _to_path(save_path)
    if save_path:
        save_figure(plt.gcf(), save_path)

    return ax


def get_cramers_v_matrix(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    n = len(cols)
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            col1, col2 = cols[i], cols[j]
            confusion = pd.crosstab(df[col1], df[col2])

            chi2 = chi2_contingency(confusion)[0]
            total = confusion.sum().sum()
            phi2 = chi2 / total
            r, k = confusion.shape

            phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (total - 1))
            rcorr = r - ((r - 1) ** 2) / (total - 1)
            kcorr = k - ((k - 1) ** 2) / (total - 1)

            if min((kcorr - 1), (rcorr - 1)) == 0:
                val = 0
            else:
                val = np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))

            matrix[i, j] = val
            matrix[j, i] = val

    return pd.DataFrame(matrix, index=cols, columns=cols)


def plot_cramers_v_heatmap(
    df: pd.DataFrame,
    cols: list[str],
    save_path: Path | str | None = None,
) -> pd.DataFrame:
    v_matrix = get_cramers_v_matrix(df, cols)

    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(
        v_matrix,
        annot=True,
        cmap="YlGnBu",
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Cramer's V (0=None, 1=Strong)"},
        ax=ax,
    )
    ax.set_title("Categorical Association Heatmap: Clinical & Demographic Features", fontsize=16, fontweight="bold")

    save_path = _to_path(save_path)
    if save_path:
        save_figure(fig, save_path)

    return v_matrix


def plot_top_outlier_boxplots(
    data: pd.DataFrame,
    columns: list[str],
    save_path: Path | str | None = None,
):
    fig = plt.figure(figsize=(16, 10))
    for i, col in enumerate(columns, 1):
        plt.subplot(2, 2, i)
        sns.boxplot(x=data[col], palette="crest", fliersize=5)
        plt.title(f"Outlier Distribution: {col}", fontweight="bold")
        plt.grid(axis="x", linestyle="--", alpha=0.7)

    save_path = _to_path(save_path)
    if save_path:
        save_figure(fig, save_path)

    return fig



def get_icd9_mapping() -> list[tuple[int, int, str]]:
    """Define lookup table for ICD-9 categories."""
    return [
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


def categorize_icd9_vectorized(series: pd.Series) -> pd.Series:
    """Categorize an ICD-9 series with vectorized logic."""
    val = pd.to_numeric(series, errors="coerce")
    result = pd.Series("Other", index=series.index)
    result[series.isna() | (series == "no_diag")] = "None"

    str_series = series.astype(str).str.upper()
    result[str_series.str.startswith(("E", "V"))] = "External_Supplemental"

    for start, end, label in get_icd9_mapping():
        mask = (val >= start) & (val <= end)
        result[mask] = label
    return result


def refine_diagnosis_categories(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Merge low-frequency diagnosis labels into broader groups."""
    refinement_map = {
        "Circulatory_Cerebrovascular": "Circulatory_Other",
        "Circulatory_Rheumatic": "Circulatory_Other",
        "Circulatory_Pulmonary": "Circulatory_Other",
        "Circulatory_Arterial_Venous": "Circulatory_Other",
        "Endocrine_Non_Diabetes": "Endocrine_Metabolic_Other",
        "Congenital_Anomalies": "Other",
    }
    out = df.copy()
    out[columns] = out[columns].replace(refinement_map)
    return out


def add_comorbidity_features(df: pd.DataFrame, target_diags: list[str]) -> pd.DataFrame:
    """Create high-risk comorbidity flags from diagnosis columns."""
    out = df.copy()

    is_circ = out[target_diags].stack().str.contains("Circulatory").unstack().any(axis=1)
    is_resp = out[target_diags].stack().str.contains("Respiratory").unstack().any(axis=1)
    is_kidney = out[target_diags].stack().str.contains("Genitourinary").unstack().any(axis=1)
    is_diab = out[target_diags].stack().str.contains("Diabetes").unstack().any(axis=1)

    out["has_circulatory"] = is_circ.astype(int)
    out["has_respiratory"] = is_resp.astype(int)
    out["has_kidney"] = is_kidney.astype(int)
    out["diabetes_complications"] = (is_diab & (is_circ | is_kidney)).astype(int)

    return out


def clean_payer_codes(df: pd.DataFrame, column: str = "payer_code") -> pd.DataFrame:
    """Categorize payer codes into broad insurance groups."""
    out = df.copy()

    s = out[column].fillna("Unknown").astype(str).str.upper()
    conditions = [
        s.str.contains("MC|MEDICARE"),
        s.str.contains("MD|MEDICAID"),
        s.str.contains("BC|BLUE|HM|COMMERCIAL|PO|UN"),
        s.str.contains("SP|SELF"),
    ]
    choices = ["Medicare", "Medicaid", "Commercial", "Self_Pay"]

    out[column] = np.select(conditions, choices, default="Other_Insurance")
    out.loc[s == "UNKNOWN", column] = "Unknown"

    return out


def clean_admission_sources(df: pd.DataFrame, column: str = "admission_source_id") -> pd.DataFrame:
    """Group admission source IDs into clinical source buckets."""
    out = df.copy()

    groups = {
        "Physician_Referral": ["1", "2", "3"],
        "Transfer_Hospital": ["4", "5", "6", "10", "18", "22", "25"],
        "Emergency_Room": ["7"],
        "Court_Law": ["8", "9"],
        "Transfer_Facility": ["11", "12", "13", "14"],
    }

    source_map = {idx: cat for cat, ids in groups.items() for idx in ids}
    out[column] = out[column].astype(str).map(source_map).fillna("Other_Source")

    mask_unknown = out[column].isna() | (out[column].isin(["nan", "", "None"]))
    out.loc[mask_unknown, column] = "Unknown"

    return out


def add_medication_adjustment_features(df: pd.DataFrame, med_cols: list[str]) -> pd.DataFrame:
    """Create medication-adjustment indicators used in EDA and modeling."""
    out = df.copy()
    out["med_adjusted_binary"] = out[med_cols].isin(["Up", "Down"]).any(axis=1).astype(int)
    out["count_adjusted_up"] = out[med_cols].apply(
        lambda x: x.astype(str).str.contains("Up").sum(), axis=1
    )
    out["count_adjusted_down"] = out[med_cols].apply(
        lambda x: x.astype(str).str.contains("Down").sum(), axis=1
    )
    out["adj_med_ratio"] = (
        out["count_adjusted_up"] + out["count_adjusted_down"]
    ) / out["num_medications"]
    return out


SPECIALTY_MAP: dict[str, str] = {
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


def add_specialty_logic(
    df: pd.DataFrame,
    source_col: str = "medical_specialty",
    target_col: str = "specialty_logic",
) -> pd.DataFrame:
    """Map raw medical specialties to grouped specialty logic categories."""
    out = df.copy()
    out[target_col] = out[source_col].map(SPECIALTY_MAP).fillna("Other_Unknown")
    return out


def compute_feature_influence_table(
    df: pd.DataFrame,
    target_col: str = "readmitted",
    dropped_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Compute unified feature-target influence table used in EDA cell 41."""
    dropped_cols = dropped_cols or [target_col, "target"]

    feature_df = df.drop(columns=dropped_cols, errors="ignore")
    cat_cols = feature_df.select_dtypes(include=["object", "category"]).columns.tolist()

    df_encoded = pd.get_dummies(feature_df, columns=cat_cols, drop_first=False)
    y_dummy = pd.get_dummies(df[target_col], prefix="Target")

    full_corr_matrix = df_encoded.join(y_dummy).corr()
    target_cols = [col for col in y_dummy.columns]
    all_feature_corr = full_corr_matrix.loc[df_encoded.columns, target_cols]

    all_feature_corr["Influence_Score"] = all_feature_corr.abs().sum(axis=1)
    all_feature_corr = all_feature_corr.sort_values(by="Influence_Score", ascending=False)

    return all_feature_corr


def get_outlier_summary(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Build IQR-based outlier summary for numeric columns."""
    summary_list: list[dict[str, float | str | int]] = []

    for col in columns:
        q1 = data[col].quantile(0.25)
        q3 = data[col].quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
        count = len(outliers)
        perc = (count / len(data)) * 100

        summary_list.append(
            {
                "Column": col,
                "Lower Bound": round(lower_bound, 2),
                "Upper Bound": round(upper_bound, 2),
                "Outlier Count": count,
                "Percentage (%)": round(perc, 2),
            }
        )

    return pd.DataFrame(summary_list)


def analyze_all_categorical_outliers(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Summarize rare-category outliers for all categorical features."""
    cat_columns = df.select_dtypes(include=["object", "category"]).columns
    outlier_data: list[dict[str, object]] = []

    for col in cat_columns:
        counts = df[col].value_counts(normalize=True)
        rare_categories = counts[counts < threshold]

        if not rare_categories.empty:
            outlier_data.append(
                {
                    "Feature": col,
                    "Unique Categories": df[col].nunique(),
                    "Rare Categories Count": len(rare_categories),
                    "Rare Categories List": list(rare_categories.index),
                    "Total Outlier %": round(rare_categories.sum() * 100, 2),
                }
            )

    summary_df = pd.DataFrame(outlier_data)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(by="Rare Categories Count", ascending=False)

    return summary_df
