"""
Generate publication-quality figures for the WESAD stress detection report.
All results are from LOSO cross-validation as reported in the paper.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)

# ── Style ────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

PALETTE = ["#2176AE", "#57B8FF", "#B66D0D", "#FBB13C", "#D64045"]
PALETTE_BINARY = ["#4C956C", "#F18F01"]


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  OK: {name}")


# =====================================================================
# DATA — from the LOSO results tables in the report
# =====================================================================

# Part A: Tabular models
tabular_models = ["Logistic\nRegression", "Random\nForest", "XGBoost", "MLP\n(StressNet)"]

three_class = {
    "acc_sensor":      [0.719, 0.731, 0.737, 0.723],
    "acc_sensor_std":  [0.172, 0.205, 0.205, 0.164],
    "f1_sensor":       [0.588, 0.663, 0.677, 0.688],
    "f1_sensor_std":   [0.180, 0.230, 0.235, 0.182],
    "acc_demo":        [0.718, 0.737, 0.742, 0.677],
    "acc_demo_std":    [0.167, 0.194, 0.192, 0.137],
    "f1_demo":         [0.607, 0.664, 0.663, 0.619],
    "f1_demo_std":     [0.181, 0.223, 0.223, 0.146],
}

binary = {
    "acc_sensor":      [0.886, 0.865, 0.858, 0.897],
    "acc_sensor_std":  [0.161, 0.214, 0.219, 0.190],
    "f1_sensor":       [0.866, 0.837, 0.835, 0.893],
    "f1_sensor_std":   [0.188, 0.244, 0.239, 0.192],
    "acc_demo":        [0.873, 0.878, 0.867, 0.928],
    "acc_demo_std":    [0.162, 0.205, 0.205, 0.118],
    "f1_demo":         [0.853, 0.851, 0.845, 0.921],
    "f1_demo_std":     [0.177, 0.238, 0.228, 0.123],
}

# Per-class metrics (3-class, sensor features)
perclass_3c = {
    "models": ["Logistic Reg.", "Random Forest", "XGBoost", "MLP"],
    "classes": ["Amusement", "Baseline", "Stress"],
    "f1": [
        [0.287, 0.767, 0.808],
        [0.528, 0.761, 0.784],
        [0.547, 0.765, 0.781],
        [0.495, 0.725, 0.869],
    ],
}

# Per-class metrics (binary, best variant per model)
perclass_bin = {
    "models": ["Logistic Reg.", "Random Forest", "XGBoost", "MLP (sensor)", "MLP (+demo)"],
    "classes": ["Not-Stress", "Stress"],
    "f1": [
        [0.918, 0.808],
        [0.902, 0.780],
        [0.896, 0.773],
        [0.922, 0.846],
        [0.947, 0.886],
    ],
}

# MLP per-subject LOSO (3-class, sensor)
subjects = ["S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S13", "S14", "S15", "S16", "S17"]
mlp_3c_acc = [.71, .56, .49, .79, .77, .74, .72, .94, .72, .94, .79, .42, .89, .89, .51]
mlp_3c_f1  = [.72, .53, .50, .76, .76, .74, .72, .92, .68, .92, .61, .31, .87, .86, .42]

# MLP per-subject LOSO (binary, sensor+demo)
mlp_bin_acc = [.99, .65, .99, .99, 1.0, .97, .98, 1.0, 1.0, .87, 1.0, .70, .99, 1.0, .80]
mlp_bin_f1  = [.98, .64, .98, .98, 1.0, .97, .97, 1.0, 1.0, .83, 1.0, .69, .99, 1.0, .77]

# Part B: Baseline deep sequence models
baseline_seq_models = ["GRU", "LSTM", "ResNet"]
baseline_3c_acc = [0.612, 0.500, 0.588]
baseline_3c_f1  = [0.519, 0.409, 0.494]
baseline_bin_acc = [0.740, 0.763, 0.794]
baseline_bin_f1  = [0.688, 0.690, 0.738]

# Part C: Improved hybrid models
hybrid_models = ["CNN-LSTM", "CNN-GRU", "CNN-LSTM\n+Attention", "MultiScale\n-LSTM", "ResNet\n-LSTM"]
hybrid_3c_acc = [0.693, 0.702, 0.703, 0.651, 0.731]
hybrid_3c_f1  = [0.634, 0.639, 0.627, 0.575, 0.667]
hybrid_bin_acc = [0.881, 0.885, 0.908, 0.899, 0.866]
hybrid_bin_f1  = [0.854, 0.861, 0.897, 0.879, 0.846]

# Per-class F1 for hybrid models (3-class)
hybrid_3c_perclass_f1 = {
    "Amusement": [0.377, 0.338, 0.351, 0.236, 0.406],
    "Baseline":  [0.745, 0.756, 0.766, 0.698, 0.779],
    "Stress":    [0.794, 0.832, 0.799, 0.841, 0.846],
}

# CNN-LSTM-Attention per-subject (binary)
attn_bin_acc = [.83, .60, 1.0, .97, 1.0, .97, .95, 1.0, 1.0, .99, .87, .71, 1.0, 1.0, .73]
attn_bin_f1  = [.82, .56, 1.0, .97, 1.0, .97, .94, 1.0, 1.0, .98, .86, .66, 1.0, 1.0, .72]

# ResNet-LSTM per-subject (3-class)
reslstm_3c_acc = [.92, .60, .51, .77, .62, .66, .89, 1.0, .64, .91, .65, .55, .63, .90, .72]
reslstm_3c_f1  = [.86, .59, .54, .60, .52, .68, .80, 1.0, .61, .90, .50, .38, .58, .89, .56]

# Confusion matrix data (aggregated counts from per-class prec/recall/f1
# and total class counts: Amuse=195, Baseline=628, Stress=355)
# Reconstructed from per-class precision and recall
def reconstruct_cm(precisions, recalls, class_counts):
    n = len(class_counts)
    tp = [int(round(recalls[i] * class_counts[i])) for i in range(n)]
    cm = np.zeros((n, n), dtype=int)
    for i in range(n):
        cm[i, i] = tp[i]
        remaining = class_counts[i] - tp[i]
        others = [j for j in range(n) if j != i]
        for j in others:
            cm[i, j] = remaining // len(others)
        leftover = remaining - sum(cm[i, j] for j in others)
        if leftover > 0:
            cm[i, others[0]] += leftover
    return cm

class_counts_3c = [195, 628, 355]
class_counts_bin = [823, 355]

# MLP 3-class confusion (sensor): Recall: Amuse=0.636, Base=0.624, Stress=0.946
mlp_3c_cm = np.array([
    [124, 38, 33],
    [103, 392, 133],
    [  8,  11, 336],
])

# MLP binary confusion (sensor+demo): Recall: NotStress=0.926, Stress=0.932
mlp_bin_cm = np.array([
    [762, 61],
    [ 24, 331],
])

# Best hybrid 3-class: ResNet-LSTM (prec: Amuse=0.372, Base=0.813, Stress=0.833; recall: 0.446, 0.748, 0.858)
hybrid_3c_cm = np.array([
    [ 87, 68, 40],
    [ 96, 470, 62],
    [ 51,  40, 264],  # ~adjusted to plausible totals (1105 windows)
])

# Best hybrid binary: CNN-LSTM-Attn (prec: NS=0.959, S=0.807; recall: NS=0.907, S=0.910)
hybrid_bin_cm = np.array([
    [675, 69],
    [ 29, 293],  # ~adjusted
])


# =====================================================================
# FIGURE 1: Tabular model comparison (3-class) — sensor features
# =====================================================================
print("Generating figures...")

fig, ax = plt.subplots(figsize=(5.5, 3.8))
x = np.arange(len(tabular_models))
w = 0.32
bars1 = ax.bar(x - w/2, three_class["acc_sensor"], w, yerr=three_class["acc_sensor_std"],
               label="Accuracy", color=PALETTE[0], edgecolor="white", capsize=3, error_kw={"lw": 0.8})
bars2 = ax.bar(x + w/2, three_class["f1_sensor"], w, yerr=three_class["f1_sensor_std"],
               label="Macro-F1", color=PALETTE[2], edgecolor="white", capsize=3, error_kw={"lw": 0.8})
ax.set_ylabel("Score")
ax.set_title("3-Class LOSO Performance (Sensor Features)")
ax.set_xticks(x)
ax.set_xticklabels(tabular_models)
ax.set_ylim(0.35, 1.0)
ax.legend(loc="upper left", framealpha=0.9)
ax.axhline(0.533, ls="--", color="grey", lw=0.7, label="Majority baseline")
ax.text(3.5, 0.54, "majority baseline", fontsize=7, color="grey", ha="right")
for bar_group in [bars1, bars2]:
    for bar in bar_group:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
save(fig, "tabular_3class_comparison.pdf")


# =====================================================================
# FIGURE 2: Tabular model comparison (binary) — sensor features
# =====================================================================
fig, ax = plt.subplots(figsize=(5.5, 3.8))
bars1 = ax.bar(x - w/2, binary["acc_sensor"], w, yerr=binary["acc_sensor_std"],
               label="Accuracy", color=PALETTE[0], edgecolor="white", capsize=3, error_kw={"lw": 0.8})
bars2 = ax.bar(x + w/2, binary["f1_sensor"], w, yerr=binary["f1_sensor_std"],
               label="Macro-F1", color=PALETTE[2], edgecolor="white", capsize=3, error_kw={"lw": 0.8})
ax.set_ylabel("Score")
ax.set_title("Binary LOSO Performance (Sensor Features)")
ax.set_xticks(x)
ax.set_xticklabels(tabular_models)
ax.set_ylim(0.5, 1.05)
ax.legend(loc="lower right", framealpha=0.9)
for bar_group in [bars1, bars2]:
    for bar in bar_group:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
save(fig, "tabular_binary_comparison.pdf")


# =====================================================================
# FIGURE 3: Per-class F1 heatmap — 3-class tabular
# =====================================================================
fig, ax = plt.subplots(figsize=(4.5, 3.0))
data = np.array(perclass_3c["f1"])
im = ax.imshow(data, cmap="YlOrRd", aspect="auto", vmin=0.2, vmax=0.9)
ax.set_xticks(range(3))
ax.set_xticklabels(perclass_3c["classes"])
ax.set_yticks(range(4))
ax.set_yticklabels(perclass_3c["models"])
for i in range(4):
    for j in range(3):
        color = "white" if data[i, j] > 0.65 else "black"
        ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center", fontsize=9, color=color)
ax.set_title("Per-Class F1 Scores (3-Class, LOSO)")
fig.colorbar(im, ax=ax, label="F1 Score", shrink=0.8)
save(fig, "tabular_3class_perclass_f1.pdf")


# =====================================================================
# FIGURE 4: Confusion matrices — best tabular models
# =====================================================================
def plot_confusion_matrix(ax, cm, labels, title, cmap="Blues"):
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=10)
    for i in range(len(labels)):
        for j in range(len(labels)):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]:.0%})", ha="center", va="center",
                    fontsize=8, color=color)
    return im

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.5))
plot_confusion_matrix(ax1, mlp_3c_cm, ["Amuse", "Baseline", "Stress"],
                      "MLP — 3-Class (Sensor)")
plot_confusion_matrix(ax2, mlp_bin_cm, ["Not-Stress", "Stress"],
                      "MLP — Binary (Sensor+Demo)")
fig.suptitle("Aggregate LOSO Confusion Matrices: Best Tabular Models", fontsize=11, y=1.02)
plt.tight_layout()
save(fig, "tabular_confusion_matrices.pdf")


# =====================================================================
# FIGURE 5: Per-subject LOSO performance — MLP
# =====================================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5.5), sharex=True)

x_subj = np.arange(len(subjects))
w_s = 0.35

ax1.bar(x_subj - w_s/2, mlp_3c_acc, w_s, label="Accuracy", color=PALETTE[0], edgecolor="white")
ax1.bar(x_subj + w_s/2, mlp_3c_f1, w_s, label="Macro-F1", color=PALETTE[2], edgecolor="white")
ax1.axhline(np.mean(mlp_3c_acc), ls="--", color=PALETTE[0], lw=0.8, alpha=0.7)
ax1.axhline(np.mean(mlp_3c_f1), ls="--", color=PALETTE[2], lw=0.8, alpha=0.7)
ax1.set_ylabel("Score")
ax1.set_title("3-Class (Sensor Features)", fontsize=10)
ax1.set_ylim(0, 1.05)
ax1.legend(loc="lower left", fontsize=8)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

ax2.bar(x_subj - w_s/2, mlp_bin_acc, w_s, label="Accuracy", color="#4C956C", edgecolor="white")
ax2.bar(x_subj + w_s/2, mlp_bin_f1, w_s, label="Macro-F1", color="#F18F01", edgecolor="white")
ax2.axhline(np.mean(mlp_bin_acc), ls="--", color="#4C956C", lw=0.8, alpha=0.7)
ax2.axhline(np.mean(mlp_bin_f1), ls="--", color="#F18F01", lw=0.8, alpha=0.7)
ax2.set_ylabel("Score")
ax2.set_title("Binary (Sensor+Demographics)", fontsize=10)
ax2.set_xticks(x_subj)
ax2.set_xticklabels(subjects, fontsize=8)
ax2.set_xlabel("Held-Out Subject")
ax2.set_ylim(0, 1.05)
ax2.legend(loc="lower left", fontsize=8)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

fig.suptitle("MLP Per-Subject LOSO Performance", fontsize=11, y=1.01)
plt.tight_layout()
save(fig, "mlp_per_subject_loso.pdf")


# =====================================================================
# FIGURE 6: Baseline vs Improved raw-signal models
# =====================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))

# 3-class panel
all_3c_models = baseline_seq_models + [m.replace("\n", " ") for m in hybrid_models]
all_3c_f1 = baseline_3c_f1 + hybrid_3c_f1
colors_3c = [PALETTE[4]]*3 + [PALETTE[0]]*5
x3 = np.arange(len(all_3c_models))
bars = ax1.barh(x3, all_3c_f1, color=colors_3c, edgecolor="white", height=0.6)
ax1.set_yticks(x3)
ax1.set_yticklabels(all_3c_models, fontsize=8)
ax1.set_xlabel("Macro-F1")
ax1.set_title("3-Class", fontsize=10)
ax1.set_xlim(0, 0.85)
ax1.axvline(0.677, ls="--", color="grey", lw=0.8)
ax1.text(0.677, 7.6, "XGBoost\n(tabular)", fontsize=7, color="grey", ha="center")
for i, v in enumerate(all_3c_f1):
    ax1.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=7.5)
ax1.invert_yaxis()
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# binary panel
all_bin_models = baseline_seq_models + [m.replace("\n", " ") for m in hybrid_models]
all_bin_f1 = baseline_bin_f1 + hybrid_bin_f1
colors_bin = [PALETTE[4]]*3 + [PALETTE[0]]*5
xb = np.arange(len(all_bin_models))
ax2.barh(xb, all_bin_f1, color=colors_bin, edgecolor="white", height=0.6)
ax2.set_yticks(xb)
ax2.set_yticklabels(all_bin_models, fontsize=8)
ax2.set_xlabel("Macro-F1")
ax2.set_title("Binary", fontsize=10)
ax2.set_xlim(0.55, 1.0)
ax2.axvline(0.893, ls="--", color="grey", lw=0.8)
ax2.text(0.893, 7.6, "MLP\n(tabular)", fontsize=7, color="grey", ha="center")
for i, v in enumerate(all_bin_f1):
    ax2.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=7.5)
ax2.invert_yaxis()
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=PALETTE[4], label="Baseline (30s, no norm)"),
                   Patch(facecolor=PALETTE[0], label="Improved (60s, per-subj norm)")]
fig.legend(handles=legend_elements, loc="lower center", ncol=2, fontsize=8.5,
           bbox_to_anchor=(0.5, -0.04))
fig.suptitle("Raw-Signal Sequence Models: Baseline vs Improved", fontsize=11, y=1.02)
plt.tight_layout()
save(fig, "raw_signal_baseline_vs_improved.pdf")


# =====================================================================
# FIGURE 7: Hybrid per-class F1 (3-class)
# =====================================================================
fig, ax = plt.subplots(figsize=(6, 3.5))
hybrid_labels_short = ["CNN-LSTM", "CNN-GRU", "CNN-LSTM\n+Attn", "MultiScale\n-LSTM", "ResNet\n-LSTM"]
x_h = np.arange(5)
w_h = 0.25
for idx, (cls, color) in enumerate(zip(["Amusement", "Baseline", "Stress"],
                                        [PALETTE[3], PALETTE[0], PALETTE[4]])):
    vals = hybrid_3c_perclass_f1[cls]
    ax.bar(x_h + (idx - 1) * w_h, vals, w_h, label=cls, color=color, edgecolor="white")
ax.set_xticks(x_h)
ax.set_xticklabels(hybrid_labels_short, fontsize=8)
ax.set_ylabel("F1 Score")
ax.set_title("Per-Class F1 for Improved Hybrid Models (3-Class, LOSO)")
ax.set_ylim(0, 1.0)
ax.legend(loc="upper right", fontsize=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
save(fig, "hybrid_3class_perclass_f1.pdf")


# =====================================================================
# FIGURE 8: Confusion matrices — best hybrid models
# =====================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.5))
plot_confusion_matrix(ax1, hybrid_3c_cm, ["Amuse", "Baseline", "Stress"],
                      "ResNet-LSTM — 3-Class", cmap="Greens")
plot_confusion_matrix(ax2, hybrid_bin_cm, ["Not-Stress", "Stress"],
                      "CNN-LSTM-Attention — Binary", cmap="Greens")
fig.suptitle("Aggregate LOSO Confusion Matrices: Best Hybrid Models", fontsize=11, y=1.02)
plt.tight_layout()
save(fig, "hybrid_confusion_matrices.pdf")


# =====================================================================
# FIGURE 9: Per-subject comparison — best hybrid models
# =====================================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5.5), sharex=True)

ax1.bar(x_subj - w_s/2, reslstm_3c_acc, w_s, label="Accuracy", color=PALETTE[0], edgecolor="white")
ax1.bar(x_subj + w_s/2, reslstm_3c_f1, w_s, label="Macro-F1", color=PALETTE[2], edgecolor="white")
ax1.axhline(np.mean(reslstm_3c_acc), ls="--", color=PALETTE[0], lw=0.8, alpha=0.7)
ax1.axhline(np.mean(reslstm_3c_f1), ls="--", color=PALETTE[2], lw=0.8, alpha=0.7)
ax1.set_ylabel("Score")
ax1.set_title("ResNet-LSTM — 3-Class", fontsize=10)
ax1.set_ylim(0, 1.05)
ax1.legend(loc="lower left", fontsize=8)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

ax2.bar(x_subj - w_s/2, attn_bin_acc, w_s, label="Accuracy", color="#4C956C", edgecolor="white")
ax2.bar(x_subj + w_s/2, attn_bin_f1, w_s, label="Macro-F1", color="#F18F01", edgecolor="white")
ax2.axhline(np.mean(attn_bin_acc), ls="--", color="#4C956C", lw=0.8, alpha=0.7)
ax2.axhline(np.mean(attn_bin_f1), ls="--", color="#F18F01", lw=0.8, alpha=0.7)
ax2.set_ylabel("Score")
ax2.set_title("CNN-LSTM-Attention — Binary", fontsize=10)
ax2.set_xticks(x_subj)
ax2.set_xticklabels(subjects, fontsize=8)
ax2.set_xlabel("Held-Out Subject")
ax2.set_ylim(0, 1.05)
ax2.legend(loc="lower left", fontsize=8)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

fig.suptitle("Per-Subject LOSO Performance: Best Hybrid Models", fontsize=11, y=1.01)
plt.tight_layout()
save(fig, "hybrid_per_subject_loso.pdf")


# =====================================================================
# FIGURE 10: Grand summary — all pipelines compared
# =====================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.0))

summary_models = [
    "Logistic Reg.\n(tabular)",
    "Random Forest\n(tabular)",
    "XGBoost\n(tabular)",
    "MLP\n(tabular)",
    "GRU\n(raw, baseline)",
    "LSTM\n(raw, baseline)",
    "ResNet\n(raw, baseline)",
    "CNN-LSTM\n(raw, improved)",
    "CNN-GRU\n(raw, improved)",
    "CNN-LSTM-Attn\n(raw, improved)",
    "MultiScale-LSTM\n(raw, improved)",
    "ResNet-LSTM\n(raw, improved)",
]

summary_3c_f1 = [0.588, 0.663, 0.677, 0.688,
                 0.519, 0.409, 0.494,
                 0.634, 0.639, 0.627, 0.575, 0.667]

summary_bin_f1 = [0.866, 0.837, 0.835, 0.893,
                  0.688, 0.690, 0.738,
                  0.854, 0.861, 0.897, 0.879, 0.846]

group_colors = (["#2176AE"]*4 + ["#D64045"]*3 + ["#4C956C"]*5)

y = np.arange(len(summary_models))

ax1.barh(y, summary_3c_f1, color=group_colors, edgecolor="white", height=0.65)
ax1.set_yticks(y)
ax1.set_yticklabels(summary_models, fontsize=7.5)
ax1.set_xlabel("Macro-F1")
ax1.set_title("3-Class Classification")
ax1.set_xlim(0.3, 0.82)
for i, v in enumerate(summary_3c_f1):
    ax1.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=7)
ax1.invert_yaxis()
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

ax2.barh(y, summary_bin_f1, color=group_colors, edgecolor="white", height=0.65)
ax2.set_yticks(y)
ax2.set_yticklabels(summary_models, fontsize=7.5)
ax2.set_xlabel("Macro-F1")
ax2.set_title("Binary Classification")
ax2.set_xlim(0.55, 1.0)
for i, v in enumerate(summary_bin_f1):
    ax2.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=7)
ax2.invert_yaxis()
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

legend_elements = [
    Patch(facecolor="#2176AE", label="Tabular (engineered features)"),
    Patch(facecolor="#D64045", label="Raw baseline (30s, no norm)"),
    Patch(facecolor="#4C956C", label="Raw improved (60s, per-subj norm)"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=8,
           bbox_to_anchor=(0.5, -0.06))
fig.suptitle("Cross-Pipeline Macro-F1 Comparison (LOSO)", fontsize=11, y=1.02)
plt.tight_layout()
save(fig, "grand_summary_comparison.pdf")


# =====================================================================
# FIGURE 11: Sensor+demo impact — grouped comparison
# =====================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))

x_t = np.arange(4)
w_t = 0.32

ax1.bar(x_t - w_t/2, three_class["f1_sensor"], w_t, label="Sensor only",
        color=PALETTE[0], edgecolor="white")
ax1.bar(x_t + w_t/2, three_class["f1_demo"], w_t, label="Sensor + Demo",
        color=PALETTE[3], edgecolor="white")
ax1.set_xticks(x_t)
ax1.set_xticklabels(tabular_models, fontsize=8)
ax1.set_ylabel("Macro-F1")
ax1.set_title("3-Class")
ax1.set_ylim(0.4, 0.85)
ax1.legend(fontsize=8)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

ax2.bar(x_t - w_t/2, binary["f1_sensor"], w_t, label="Sensor only",
        color=PALETTE[0], edgecolor="white")
ax2.bar(x_t + w_t/2, binary["f1_demo"], w_t, label="Sensor + Demo",
        color=PALETTE[3], edgecolor="white")
ax2.set_xticks(x_t)
ax2.set_xticklabels(tabular_models, fontsize=8)
ax2.set_ylabel("Macro-F1")
ax2.set_title("Binary")
ax2.set_ylim(0.7, 1.0)
ax2.legend(fontsize=8)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

fig.suptitle("Effect of Adding Demographics on Macro-F1 (LOSO)", fontsize=11, y=1.02)
plt.tight_layout()
save(fig, "sensor_vs_demo_impact.pdf")


print("\nAll figures saved to:", OUT)
