import os
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent
CSV_PATH = PROJECT_ROOT / "WESAD" / "data" / "m14_merged.csv"
OUT_DIR = Path("eda_outputs_m14")

# From `data_wrangling.py`:
# label_dict = {'baseline': 1, 'stress': 2, 'amusement': 0}
LABEL_MAP = {0: "amusement", 1: "baseline", 2: "stress"}


def _safe_feature_cols(df: pd.DataFrame) -> list[str]:
    """Choose the main window-level sensor features for EDA (not metadata)."""
    include_suffixes = ("_mean", "_std", "_min", "_max")
    include_exact = {"BVP_peak_freq", "TEMP_slope"}

    feature_cols = []
    for c in df.columns:
        if c in include_exact or c.endswith(include_suffixes):
            feature_cols.append(c)

    # Remove any accidental non-sensor columns that could sneak in
    drop_cols = {"label", "subject", "age", "height", "weight", "label_name"}
    feature_cols = [c for c in feature_cols if c not in drop_cols]

    # Keep only columns that actually exist
    return [c for c in feature_cols if c in df.columns]


def _savefig(name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / name
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Could not find `{CSV_PATH}`."
        )

    df = pd.read_csv(CSV_PATH, low_memory=False)

    if "label" not in df.columns or "subject" not in df.columns:
        raise ValueError("Expected columns `label` and `subject` in m14_merged.csv.")

    df["label_name"] = df["label"].map(LABEL_MAP).fillna("unknown")

    print("=== Basic info ===")
    print(f"Rows (windows): {len(df):,}")
    print(f"Subjects: {df['subject'].nunique()}")
    print("Label counts:")
    print(df["label_name"].value_counts(dropna=False))
    print()

    feature_cols = _safe_feature_cols(df)
    if not feature_cols:
        raise ValueError("No feature columns found. Check column names in the CSV.")

    print(f"Feature columns used for correlation/PCA: {len(feature_cols)}")

    sns.set_theme(style="whitegrid", context="notebook")

    # 1) Label balance
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(data=df, x="label_name", order=["baseline", "stress", "amusement", "unknown"])
    ax.set_title("Window count by label")
    ax.set_xlabel("Label")
    ax.set_ylabel("Count (windows)")
    _savefig("01_label_counts.png")

    # 2) Windows per subject, stacked by label
    subj_label_counts = (
        df.groupby(["subject", "label_name"]).size().reset_index(name="count")
    )
    pivot = subj_label_counts.pivot(index="subject", columns="label_name", values="count").fillna(0)
    pivot = pivot[[c for c in ["baseline", "stress", "amusement", "unknown"] if c in pivot.columns]]
    pivot = pivot.sort_index()

    plt.figure(figsize=(12, 5))
    pivot.plot(kind="bar", stacked=True, width=0.9)
    plt.title("Windows per subject (stacked by label)")
    plt.xlabel("Subject")
    plt.ylabel("Count (windows)")
    plt.legend(title="Label", bbox_to_anchor=(1.02, 1), loc="upper left")
    _savefig("02_windows_per_subject_stacked.png")

    # Pick a few high-signal features if present
    preferred = [
        "EDA_mean",
        "EDA_phasic_mean",
        "EDA_tonic_mean",
        "BVP_std",
        "TEMP_mean",
        "Resp_std",
        "net_acc_mean",
        "BVP_peak_freq",
        "TEMP_slope",
    ]
    key_feats = [c for c in preferred if c in df.columns]
    if len(key_feats) < 4:
        # Fall back to any first few features
        key_feats = feature_cols[:6]

    # 3) Boxplot: EDA_mean by label (if available)
    if "EDA_mean" in df.columns:
        plt.figure(figsize=(7, 4))
        ax = sns.boxplot(data=df, x="label_name", y="EDA_mean", order=["baseline", "stress", "amusement"])
        ax.set_title("EDA_mean by label")
        ax.set_xlabel("Label")
        ax.set_ylabel("EDA_mean")
        _savefig("03_box_eda_mean_by_label.png")

    # 4) Boxplot: EDA_phasic_mean by label (if available)
    if "EDA_phasic_mean" in df.columns:
        plt.figure(figsize=(7, 4))
        ax = sns.boxplot(data=df, x="label_name", y="EDA_phasic_mean", order=["baseline", "stress", "amusement"])
        ax.set_title("EDA_phasic_mean by label (cvxEDA phasic component)")
        ax.set_xlabel("Label")
        ax.set_ylabel("EDA_phasic_mean")
        _savefig("04_box_eda_phasic_mean_by_label.png")

    # 5) Motion confounding: net_acc_mean by label (if available)
    if "net_acc_mean" in df.columns:
        plt.figure(figsize=(7, 4))
        ax = sns.boxplot(data=df, x="label_name", y="net_acc_mean", order=["baseline", "stress", "amusement"])
        ax.set_title("net_acc_mean by label (movement confound check)")
        ax.set_xlabel("Label")
        ax.set_ylabel("net_acc_mean")
        _savefig("05_box_net_acc_mean_by_label.png")

    # 6) Correlation heatmap (subset of features for readability)
    corr_cols = []
    for c in key_feats:
        if c in feature_cols:
            corr_cols.append(c)
    if len(corr_cols) < 6:
        corr_cols = feature_cols[:12]

    corr_df = df[corr_cols].copy()
    # Coerce bool->int and drop non-numeric safely
    for c in corr_df.columns:
        if corr_df[c].dtype == bool:
            corr_df[c] = corr_df[c].astype(int)
    corr = corr_df.select_dtypes(include=[np.number]).corr()

    plt.figure(figsize=(10, 7))
    ax = sns.heatmap(corr, cmap="vlag", center=0, linewidths=0.3, square=True)
    ax.set_title("Feature correlation heatmap (subset)")
    _savefig("06_corr_heatmap_subset.png")

    # Prepare data for PCA (use only sensor features)
    X = df[feature_cols].copy()
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(int)
    X = X.apply(pd.to_numeric, errors="coerce")

    valid = X.notna().all(axis=1)
    Xv = X.loc[valid]
    meta = df.loc[valid, ["label_name", "subject"]]

    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xv.values)

    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(Xs)
    pca_df = pd.DataFrame(Z, columns=["PC1", "PC2"])
    pca_df["label_name"] = meta["label_name"].values
    pca_df["subject"] = meta["subject"].values

    explained = pca.explained_variance_ratio_
    title_suffix = f"(explained: {explained[0]:.2%}, {explained[1]:.2%})"

    # 7) PCA scatter colored by label
    plt.figure(figsize=(7, 5))
    ax = sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2",
        hue="label_name",
        hue_order=[c for c in ["baseline", "stress", "amusement", "unknown"] if c in pca_df["label_name"].unique()],
        s=18,
        alpha=0.8,
        linewidth=0,
    )
    ax.set_title(f"PCA of sensor-window features by label {title_suffix}")
    ax.legend(title="Label", bbox_to_anchor=(1.02, 1), loc="upper left")
    _savefig("07_pca_scatter_by_label.png")

    # 8) PCA scatter colored by subject
    plt.figure(figsize=(8, 6))
    ax = sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2",
        hue="subject",
        palette="tab20",
        s=18,
        alpha=0.8,
        linewidth=0,
    )
    ax.set_title(f"PCA of sensor-window features by subject {title_suffix}")
    ax.legend(title="Subject", bbox_to_anchor=(1.02, 1), loc="upper left", ncol=1)
    _savefig("08_pca_scatter_by_subject.png")

    print()
    print("Done. Check the output folder:", OUT_DIR.resolve())


if __name__ == "__main__":
    main()

