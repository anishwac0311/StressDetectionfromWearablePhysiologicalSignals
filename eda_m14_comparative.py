import os
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_selection import mutual_info_classif


PROJECT_ROOT = Path(__file__).resolve().parent
CSV_PATH = PROJECT_ROOT / "WESAD" / "data" / "m14_merged.csv"
OUT_DIR = Path("eda_outputs_m14_comparative")

# From your pipeline:
LABEL_MAP = {0: "amusement", 1: "baseline", 2: "stress"}


def _safe_feature_cols(df: pd.DataFrame) -> list[str]:
    include_suffixes = ("_mean", "_std", "_min", "_max")
    include_exact = {"BVP_peak_freq", "TEMP_slope"}

    feature_cols = [
        c
        for c in df.columns
        if (c in include_exact or c.endswith(include_suffixes))
        and c not in {"label", "subject", "age", "height", "weight", "label_name"}
    ]
    return feature_cols


def _savefig(name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / name
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def _to_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    for c in x.columns:
        if x[c].dtype == bool:
            x[c] = x[c].astype(int)
    x = x.apply(pd.to_numeric, errors="coerce")
    return x


def epsilon_squared_kruskal(feature: pd.Series, y: pd.Series) -> float:
    """
    Multiclass robust effect size based on Kruskal-Wallis H statistic.
    epsilon^2 = (H - k + 1) / (n - k)
    Returns 0 when undefined.
    """
    # Late import so this script still loads even if scipy is missing
    from scipy.stats import kruskal

    x = pd.to_numeric(feature, errors="coerce")
    y = y.astype(int)

    groups = []
    for lab in sorted(y.unique()):
        g = x[y == lab].dropna().values
        if len(g) > 0:
            groups.append(g)

    k = len(groups)
    n = int(np.sum([len(g) for g in groups]))
    if k < 2 or n <= k:
        return 0.0

    H, _p = kruskal(*groups)
    eps2 = (H - k + 1) / (n - k)
    if not np.isfinite(eps2):
        return 0.0
    return float(max(0.0, eps2))


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Could not find `{CSV_PATH}`.")

    df = pd.read_csv(CSV_PATH, low_memory=False)
    if "label" not in df.columns or "subject" not in df.columns:
        raise ValueError("Expected columns `label` and `subject` in m14_merged.csv.")

    df["label_name"] = df["label"].map(LABEL_MAP).fillna("unknown")

    feature_cols = _safe_feature_cols(df)
    X = _to_numeric_frame(df[feature_cols])
    y = df["label"].astype(int)

    # Drop rows with any missing feature
    valid = X.notna().all(axis=1)
    Xv = X.loc[valid]
    yv = y.loc[valid]
    dfv = df.loc[valid].copy()

    sns.set_theme(style="whitegrid", context="notebook")

    # -------------------------
    # 1) Bivariate scatter plots
    # -------------------------
    pairs = [
        ("EDA_phasic_mean", "BVP_std"),
        ("EDA_mean", "Resp_std"),
        ("net_acc_mean", "BVP_std"),
        ("TEMP_mean", "EDA_tonic_mean"),
        ("net_acc_mean", "EDA_std"),
    ]
    pairs = [(a, b) for (a, b) in pairs if a in dfv.columns and b in dfv.columns]

    for i, (a, b) in enumerate(pairs, start=1):
        plt.figure(figsize=(7.2, 5.2))
        ax = sns.scatterplot(
            data=dfv,
            x=a,
            y=b,
            hue="label_name",
            hue_order=[c for c in ["baseline", "stress", "amusement", "unknown"] if c in dfv["label_name"].unique()],
            s=16,
            alpha=0.5,
            linewidth=0,
        )
        ax.set_title(f"Bivariate comparison: {a} vs {b}")
        ax.legend(title="Label", bbox_to_anchor=(1.02, 1), loc="upper left")
        _savefig(f"01_bivariate_{i:02d}_{a}_vs_{b}.png")

    # ----------------------------------------
    # 2) Correlation heatmaps (overall + per label)
    # ----------------------------------------
    # Use a readable subset: key physiology + motion features if present
    corr_candidates = [
        "EDA_mean",
        "EDA_phasic_mean",
        "EDA_tonic_mean",
        "BVP_std",
        "Resp_std",
        "TEMP_mean",
        "net_acc_mean",
        "BVP_peak_freq",
        "TEMP_slope",
    ]
    corr_cols = [c for c in corr_candidates if c in Xv.columns]
    if len(corr_cols) < 6:
        corr_cols = list(Xv.columns[:12])

    def corr_heatmap(data: pd.DataFrame, title: str, fname: str) -> None:
        corr = data[corr_cols].corr(method="spearman")
        plt.figure(figsize=(10, 7))
        ax = sns.heatmap(corr, cmap="vlag", center=0, linewidths=0.3, square=True)
        ax.set_title(title)
        _savefig(fname)

    corr_heatmap(Xv, "Spearman correlation (overall)", "02_corr_overall_spearman.png")
    for lab_val, lab_name in LABEL_MAP.items():
        m = (dfv["label"] == lab_val).values
        if int(m.sum()) < 10:
            continue
        corr_heatmap(
            Xv.loc[m],
            f"Spearman correlation (label = {lab_name})",
            f"03_corr_label_{lab_val}_{lab_name}_spearman.png",
        )

    # ----------------------------------------
    # 3) Rank features by mutual information (nonlinear association)
    # ----------------------------------------
    mi = mutual_info_classif(Xv.values, yv.values, discrete_features=False, random_state=42)
    mi_df = pd.DataFrame({"feature": Xv.columns, "mutual_info": mi}).sort_values("mutual_info", ascending=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mi_df.to_csv(OUT_DIR / "04_feature_ranking_mutual_info.csv", index=False)

    plt.figure(figsize=(8.5, 6))
    top = mi_df.head(20).iloc[::-1]
    ax = sns.barplot(data=top, x="mutual_info", y="feature", color="#4C78A8")
    ax.set_title("Top features by mutual information with label")
    ax.set_xlabel("Mutual information (higher = more predictive)")
    ax.set_ylabel("")
    _savefig("04_feature_ranking_mutual_info_top20.png")

    # ----------------------------------------
    # 4) Rank features by robust multiclass effect size (epsilon^2)
    # ----------------------------------------
    eps2_vals = []
    for c in Xv.columns:
        eps2_vals.append((c, epsilon_squared_kruskal(Xv[c], yv)))
    eps2_df = pd.DataFrame(eps2_vals, columns=["feature", "epsilon_squared"]).sort_values(
        "epsilon_squared", ascending=False
    )
    eps2_df.to_csv(OUT_DIR / "05_feature_ranking_effectsize_eps2.csv", index=False)

    plt.figure(figsize=(8.5, 6))
    top2 = eps2_df.head(20).iloc[::-1]
    ax = sns.barplot(data=top2, x="epsilon_squared", y="feature", color="#E45756")
    ax.set_title("Top features by robust multiclass effect size (Kruskal-Wallis epsilon²)")
    ax.set_xlabel("epsilon² (higher = stronger separation)")
    ax.set_ylabel("")
    _savefig("05_feature_ranking_effectsize_eps2_top20.png")

    print()
    print("Done. Outputs written to:", OUT_DIR.resolve())


if __name__ == "__main__":
    main()

