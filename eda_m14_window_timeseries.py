import os
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent
CSV_PATH = PROJECT_ROOT / "WESAD" / "data" / "m14_merged.csv"
OUT_DIR = Path("eda_outputs_m14_timeseries")

# From `data_wrangling.py`:
# label_dict = {'baseline': 1, 'stress': 2, 'amusement': 0}
LABEL_MAP = {0: "amusement", 1: "baseline", 2: "stress"}

# Pick a subject to visualize. Change this to any subject id present in your CSV.
SUBJECT_ID = 2

# Features that tend to show “spiky windows” (high variance / max) and movement confounds.
FEATURES_TO_PLOT = [
    "EDA_mean",
    "EDA_std",
    "EDA_max",
    "EDA_phasic_mean",
    "BVP_std",
    "TEMP_mean",
    "net_acc_mean",
]

# Outlier sensitivity: robust z-score threshold per subject (higher = fewer flags).
ROBUST_Z_THRESH = 5.0


def robust_z(x: pd.Series) -> pd.Series:
    """
    Robust z-score using median and MAD.
    z = 0.6745 * (x - median) / MAD
    """
    x = pd.to_numeric(x, errors="coerce")
    med = x.median()
    mad = (x - med).abs().median()
    if mad == 0 or np.isnan(mad):
        return pd.Series(np.zeros(len(x)), index=x.index)
    return 0.6745 * (x - med) / mad


def contiguous_spans(labels: pd.Series) -> list[tuple[int, int, str]]:
    """
    Convert a label sequence into contiguous spans: [(start_idx, end_idx, label_name), ...]
    where end_idx is inclusive.
    """
    spans: list[tuple[int, int, str]] = []
    if labels.empty:
        return spans

    current = labels.iloc[0]
    start = 0
    for i in range(1, len(labels)):
        if labels.iloc[i] != current:
            spans.append((start, i - 1, str(current)))
            start = i
            current = labels.iloc[i]
    spans.append((start, len(labels) - 1, str(current)))
    return spans


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Could not find `{CSV_PATH}`.")

    df = pd.read_csv(CSV_PATH, low_memory=False)
    if "subject" not in df.columns or "label" not in df.columns:
        raise ValueError("Expected columns `subject` and `label` in m14_merged.csv.")

    df["label_name"] = df["label"].map(LABEL_MAP).fillna("unknown")

    if SUBJECT_ID not in set(df["subject"].unique()):
        raise ValueError(
            f"SUBJECT_ID={SUBJECT_ID} not found. Available subjects: {sorted(df['subject'].unique().tolist())}"
        )

    # Keep original row order and build an in-subject window index.
    sdf = df[df["subject"] == SUBJECT_ID].copy()
    sdf["window_idx"] = np.arange(len(sdf))

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Plot 1: label timeline (what segments appear in what order) ---
    sns.set_theme(style="whitegrid", context="notebook")
    plt.figure(figsize=(12, 2.2))
    ax = plt.gca()
    spans = contiguous_spans(sdf["label_name"])
    colors = {"baseline": "#4C78A8", "stress": "#E45756", "amusement": "#72B7B2", "unknown": "#999999"}
    for start, end, lab in spans:
        ax.axvspan(start, end + 1, color=colors.get(lab, "#999999"), alpha=0.35, lw=0)
    ax.set_title(f"Subject {SUBJECT_ID}: label segments across window index")
    ax.set_xlabel("Window index (row order within subject)")
    ax.set_yticks([])
    # Add a legend proxy
    handles = [
        plt.Line2D([0], [0], color=colors[k], lw=8, alpha=0.6, label=k)
        for k in ["baseline", "stress", "amusement"]
        if k in set(sdf["label_name"])
    ]
    if handles:
        ax.legend(handles=handles, title="Label", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"subject_{SUBJECT_ID:02d}_label_timeline.png", dpi=200, bbox_inches="tight")
    plt.close()

    # --- Plot 2: feature traces across windows (window-level “time series”) ---
    feats = [f for f in FEATURES_TO_PLOT if f in sdf.columns]
    if not feats:
        raise ValueError(
            "None of FEATURES_TO_PLOT were found in the CSV. Edit FEATURES_TO_PLOT to match your columns."
        )

    n = len(feats)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, feat in zip(axes, feats):
        y = pd.to_numeric(sdf[feat], errors="coerce")

        # Background shading by label segments to visually link spikes to class blocks
        for start, end, lab in spans:
            ax.axvspan(start, end + 1, color=colors.get(lab, "#999999"), alpha=0.15, lw=0)

        ax.plot(sdf["window_idx"], y, color="black", linewidth=1.0, alpha=0.9)
        ax.set_ylabel(feat)

        # Mark robust outliers for this feature (per subject)
        rz = robust_z(y)
        out_mask = rz.abs() >= ROBUST_Z_THRESH
        if out_mask.any():
            ax.scatter(
                sdf.loc[out_mask, "window_idx"],
                y.loc[out_mask],
                color="#D62728",
                s=18,
                zorder=3,
                label=f"outliers |robust_z|≥{ROBUST_Z_THRESH:g}",
            )
            ax.legend(loc="upper right")

    axes[0].set_title(f"Subject {SUBJECT_ID}: window-feature traces (spikes/outliers show as sudden jumps)")
    axes[-1].set_xlabel("Window index (row order within subject)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"subject_{SUBJECT_ID:02d}_feature_traces.png", dpi=200, bbox_inches="tight")
    plt.close()

    # --- Export: top outlier windows across chosen features ---
    outlier_rows = []
    for feat in feats:
        y = pd.to_numeric(sdf[feat], errors="coerce")
        rz = robust_z(y)
        tmp = sdf[["subject", "label", "label_name", "window_idx"]].copy()
        tmp["feature"] = feat
        tmp["value"] = y
        tmp["robust_z"] = rz
        tmp = tmp.sort_values("robust_z", key=lambda s: s.abs(), ascending=False)
        outlier_rows.append(tmp.head(25))

    out_df = pd.concat(outlier_rows, ignore_index=True)
    out_df.to_csv(OUT_DIR / f"subject_{SUBJECT_ID:02d}_top_outlier_windows.csv", index=False)

    print("Done.")
    print(f"- Plots: {OUT_DIR / f'subject_{SUBJECT_ID:02d}_label_timeline.png'}")
    print(f"- Plots: {OUT_DIR / f'subject_{SUBJECT_ID:02d}_feature_traces.png'}")
    print(f"- CSV:   {OUT_DIR / f'subject_{SUBJECT_ID:02d}_top_outlier_windows.csv'}")
    print()
    print("Note:")
    print("- This is a *window-index* timeline, not raw sensor time. For true within-window spikes, plot raw signals from WESAD .pkl files.")


if __name__ == "__main__":
    main()

