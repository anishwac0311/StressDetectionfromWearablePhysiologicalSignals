import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple


# ---- User settings (edit these) ----
PROJECT_ROOT = Path(__file__).resolve().parent
WESAD_ROOT = PROJECT_ROOT / "WESAD" / "data" / "WESAD"
SUBJECT_ID = 2

# Plot only a slice so figures stay readable. Set END_MIN=None to plot full recording.
START_MIN = 0
END_MIN = 10

OUT_DIR = Path("eda_outputs_wesad_raw")


# Wrist sampling frequencies (WESAD / Empatica E4)
FS_WRIST = {"ACC": 32, "BVP": 64, "EDA": 4, "TEMP": 4}

# WESAD label stream is 700 Hz in your pipeline (`fs_dict['label'] = 700`)
FS_LABEL = 700

# WESAD label meanings (raw dataset convention)
# Your processing uses: baseline=1, stress=2, amusement=3 (see `compute_features()`).
LABEL_NAME = {
    0: "undefined",
    1: "baseline",
    2: "stress",
    3: "amusement",
}

LABEL_COLOR = {
    "baseline": "#4C78A8",
    "stress": "#E45756",
    "amusement": "#72B7B2",
    "undefined": "#999999",
}

# Minimum label span duration (seconds) to annotate with text on the plot
ANNOTATE_MIN_SPAN_S = 15.0


def load_subject_pkl(wesad_root: str, subject_id: int) -> dict:
    name = f"S{subject_id}"
    pkl_path = os.path.join(wesad_root, name, f"{name}.pkl")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Could not find subject file: {pkl_path}")
    with open(pkl_path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def contiguous_spans(arr: np.ndarray) -> List[Tuple[int, int, int]]:
    """
    Convert an integer array into contiguous spans of constant value:
    returns list of (start_idx, end_idx_inclusive, value).
    """
    spans: list[tuple[int, int, int]] = []
    if arr.size == 0:
        return spans
    start = 0
    cur = int(arr[0])
    for i in range(1, arr.size):
        v = int(arr[i])
        if v != cur:
            spans.append((start, i - 1, cur))
            start = i
            cur = v
    spans.append((start, arr.size - 1, cur))
    return spans


def to_seconds(idx: int, fs: int) -> float:
    return idx / float(fs)


def _crop_by_time(
    x: np.ndarray, fs: int, start_s: float, end_s: Optional[float]
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Returns (t, y, start_idx) where:
    - t is time axis in seconds starting at start_s (so t[0]=start_s)
    - y is cropped signal
    - start_idx is the original start index into x
    """
    n = x.shape[0]
    start_idx = int(max(0, round(start_s * fs)))
    end_idx = n if end_s is None else int(min(n, round(end_s * fs)))
    y = x[start_idx:end_idx]
    t = (np.arange(start_idx, end_idx) / float(fs)).astype(float)
    return t, y, start_idx


def shade_labels(
    ax: plt.Axes,
    label_spans: List[Tuple[int, int, int]],
    start_s: float,
    end_s: Optional[float],
) -> None:
    """
    Shade background by label spans (using label time in seconds) and optionally annotate names.
    """
    # Limit shading to requested range
    x0 = start_s
    x1 = end_s if end_s is not None else None

    for s_idx, e_idx, lab in label_spans:
        lab_name = LABEL_NAME.get(int(lab), "unknown")
        c = LABEL_COLOR.get(lab_name, "#999999")
        a = 0.18 if lab_name in {"baseline", "stress", "amusement"} else 0.10

        span_start = to_seconds(s_idx, FS_LABEL)
        span_end = to_seconds(e_idx + 1, FS_LABEL)

        # Skip spans that don't overlap plotting range
        if span_end < x0:
            continue
        if x1 is not None and span_start > x1:
            continue

        vis_start = max(span_start, x0)
        vis_end = span_end if x1 is None else min(span_end, x1)
        ax.axvspan(vis_start, vis_end, color=c, alpha=a, lw=0)

        # Annotate with label name for sufficiently long spans (to avoid clutter)
        if lab_name in {"baseline", "stress", "amusement"} and (vis_end - vis_start) >= ANNOTATE_MIN_SPAN_S:
            x_mid = (vis_start + vis_end) / 2.0
            ax.text(
                x_mid,
                0.98,
                lab_name,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=10,
                color="black",
                alpha=0.85,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.55),
            )


def make_legend_handles() -> list:
    handles = []
    for name in ["baseline", "stress", "amusement"]:
        handles.append(plt.Line2D([0], [0], color=LABEL_COLOR[name], lw=8, alpha=0.4, label=name))
    return handles


def plot_signal(
    *,
    out_path: Path,
    title: str,
    t: np.ndarray,
    y: np.ndarray,
    ylabel: str,
    label_spans: List[Tuple[int, int, int]],
    start_s: float,
    end_s: Optional[float],
) -> None:
    plt.figure(figsize=(14, 4))
    ax = plt.gca()
    shade_labels(ax, label_spans, start_s, end_s)
    ax.plot(t, y, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.legend(handles=make_legend_handles(), title="Label", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_subject_pkl(WESAD_ROOT, SUBJECT_ID)

    # Raw streams
    wrist = data["signal"]["wrist"]
    labels = np.asarray(data["label"]).reshape(-1)

    # Build label spans for shading + export
    spans = contiguous_spans(labels.astype(int))
    spans_df = pd.DataFrame(
        [
            {
                "start_idx": s,
                "end_idx": e,
                "label": lab,
                "label_name": LABEL_NAME.get(lab, "unknown"),
                "start_s": to_seconds(s, FS_LABEL),
                "end_s": to_seconds(e + 1, FS_LABEL),
                "duration_s": to_seconds(e - s + 1, FS_LABEL),
            }
            for (s, e, lab) in spans
        ]
    )
    spans_df.to_csv(OUT_DIR / f"S{SUBJECT_ID:02d}_label_spans.csv", index=False)
    print(f"Saved: {OUT_DIR / f'S{SUBJECT_ID:02d}_label_spans.csv'}")

    start_s = float(START_MIN) * 60.0
    end_s = None if END_MIN is None else float(END_MIN) * 60.0

    # EDA (wrist)
    eda = np.asarray(wrist["EDA"]).reshape(-1)
    t_eda, y_eda, _ = _crop_by_time(eda, FS_WRIST["EDA"], start_s, end_s)
    plot_signal(
        out_path=OUT_DIR / f"S{SUBJECT_ID:02d}_raw_EDA.png",
        title=f"S{SUBJECT_ID:02d} raw EDA (wrist) with label overlay",
        t=t_eda,
        y=y_eda,
        ylabel="EDA (uS)",
        label_spans=spans,
        start_s=start_s,
        end_s=end_s,
    )

    # BVP (wrist)
    bvp = np.asarray(wrist["BVP"]).reshape(-1)
    t_bvp, y_bvp, _ = _crop_by_time(bvp, FS_WRIST["BVP"], start_s, end_s)
    plot_signal(
        out_path=OUT_DIR / f"S{SUBJECT_ID:02d}_raw_BVP.png",
        title=f"S{SUBJECT_ID:02d} raw BVP (wrist) with label overlay",
        t=t_bvp,
        y=y_bvp,
        ylabel="BVP",
        label_spans=spans,
        start_s=start_s,
        end_s=end_s,
    )

    # TEMP (wrist)
    temp = np.asarray(wrist["TEMP"]).reshape(-1)
    t_temp, y_temp, _ = _crop_by_time(temp, FS_WRIST["TEMP"], start_s, end_s)
    plot_signal(
        out_path=OUT_DIR / f"S{SUBJECT_ID:02d}_raw_TEMP.png",
        title=f"S{SUBJECT_ID:02d} raw TEMP (wrist) with label overlay",
        t=t_temp,
        y=y_temp,
        ylabel="TEMP (C)",
        label_spans=spans,
        start_s=start_s,
        end_s=end_s,
    )

    # ACC magnitude (wrist)
    acc = np.asarray(wrist["ACC"])
    if acc.ndim == 2 and acc.shape[1] == 3:
        acc_mag = np.sqrt((acc.astype(float) ** 2).sum(axis=1))
    else:
        acc_mag = np.asarray(acc).reshape(-1).astype(float)
    t_acc, y_acc, _ = _crop_by_time(acc_mag, FS_WRIST["ACC"], start_s, end_s)
    plot_signal(
        out_path=OUT_DIR / f"S{SUBJECT_ID:02d}_raw_ACC_mag.png",
        title=f"S{SUBJECT_ID:02d} raw ACC magnitude (wrist) with label overlay",
        t=t_acc,
        y=y_acc,
        ylabel="|ACC|",
        label_spans=spans,
        start_s=start_s,
        end_s=end_s,
    )

    # Quick combined view (EDA + BVP)
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    for ax in axes:
        shade_labels(ax, spans, start_s, end_s)
        ax.legend(handles=make_legend_handles(), title="Label", bbox_to_anchor=(1.02, 1), loc="upper left")

    axes[0].plot(t_eda, y_eda, color="black", linewidth=0.8)
    axes[0].set_title(f"S{SUBJECT_ID:02d} raw EDA + BVP (wrist) with label overlay")
    axes[0].set_ylabel("EDA (uS)")

    axes[1].plot(t_bvp, y_bvp, color="black", linewidth=0.6)
    axes[1].set_ylabel("BVP")
    axes[1].set_xlabel("Time (s)")

    plt.tight_layout()
    combined_path = OUT_DIR / f"S{SUBJECT_ID:02d}_raw_EDA_BVP_combined.png"
    plt.savefig(combined_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {combined_path}")

    print()
    print("Tip: if you want to zoom into a spike, reduce END_MIN-START_MIN (e.g., plot 2 minutes).")


if __name__ == "__main__":
    main()

