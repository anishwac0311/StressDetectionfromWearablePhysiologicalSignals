import os
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent
CSV_PATH = PROJECT_ROOT / "WESAD" / "data" / "m14_merged.csv"
OUT_DIR = Path("eda_outputs_m14_correlation")

LABEL_MAP = {0: "amusement", 1: "baseline", 2: "stress"}


def safe_feature_cols(df: pd.DataFrame) -> list[str]:
    include_suffixes = ("_mean", "_std", "_min", "_max")
    include_exact = {"BVP_peak_freq", "TEMP_slope"}

    cols = [
        c
        for c in df.columns
        if (c in include_exact or c.endswith(include_suffixes))
        and c not in {"label", "subject", "age", "height", "weight"}
    ]
    return cols


def to_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    for c in x.columns:
        if x[c].dtype == bool:
            x[c] = x[c].astype(int)
    return x.apply(pd.to_numeric, errors="coerce")


def savefig(name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / name
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def corr_heatmap(X: pd.DataFrame, title: str, fname: str, method: str = "spearman") -> None:
    corr = X.corr(method=method)
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(corr, cmap="vlag", center=0, linewidths=0.2, square=True)
    ax.set_title(title)
    savefig(fname)


def top_corr_pairs(corr: pd.DataFrame, top_k: int = 30) -> pd.DataFrame:
    # remove self-correlation and duplicates (A,B) vs (B,A)
    # Note: we only iterate i<j, so we never include the diagonal or duplicate pairs.
    c = corr
    pairs = []
    cols = c.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append((cols[i], cols[j], c.iloc[i, j]))

    out = pd.DataFrame(pairs, columns=["feature_1", "feature_2", "corr"])
    out["abs_corr"] = out["corr"].abs()
    out = out.sort_values("abs_corr", ascending=False).head(top_k).drop(columns=["abs_corr"])
    return out


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Could not find `{CSV_PATH}`.")

    df = pd.read_csv(CSV_PATH, low_memory=False)
    df["label_name"] = df["label"].map(LABEL_MAP).fillna("unknown")

    feat_cols = safe_feature_cols(df)
    X = to_numeric_frame(df[feat_cols])

    # Keep only fully numeric rows (for clean correlations)
    valid = X.notna().all(axis=1)
    Xv = X.loc[valid]
    dfv = df.loc[valid].copy()

    sns.set_theme(style="whitegrid", context="notebook")

    # Overall correlations
    corr_heatmap(Xv, "Overall correlation (Spearman)", "01_corr_overall_spearman.png", method="spearman")
    corr_heatmap(Xv, "Overall correlation (Pearson)", "02_corr_overall_pearson.png", method="pearson")

    # Save top correlated pairs (Spearman)
    corr_s = Xv.corr(method="spearman")
    pairs_df = top_corr_pairs(corr_s, top_k=40)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs_df.to_csv(OUT_DIR / "03_top_correlated_pairs_spearman.csv", index=False)
    print(f"Saved: {OUT_DIR / '03_top_correlated_pairs_spearman.csv'}")

    # Per-label Spearman correlations
    for lab_val, lab_name in LABEL_MAP.items():
        m = (dfv["label"] == lab_val).values
        if int(m.sum()) < 10:
            continue
        corr_heatmap(
            Xv.loc[m],
            f"Correlation (Spearman) for label: {lab_name}",
            f"04_corr_label_{lab_val}_{lab_name}_spearman.png",
            method="spearman",
        )

    print("Done.")


if __name__ == "__main__":
    main()