"""
slide_figures.py  —  generates presentation figures from corrected results

All methods are run natively at 2D (not sliced from 300D).

Run from the repo root or from src/:
    python src/run_models/slide_figures.py

Output: data/intermediate_data/sentence-transformers/all-MiniLM-L6-v2_results/slide_figures/
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import pairwise_distances

# ── path setup ────────────────────────────────────────────────────────────────
current = os.path.dirname(os.path.abspath(__file__))
while os.path.basename(current) != "src" and current != os.path.dirname(current):
    current = os.path.dirname(current)
src_dir = current
os.chdir(src_dir)
sys.path.insert(0, src_dir)

REDUCTION_2D_DIR = os.path.join(src_dir, "../data/intermediate_data/sentence-transformers/all-MiniLM-L6-v2_reduced_2d")
EMBEDDING_DIR    = os.path.join(src_dir, "../data/intermediate_data/sentence-transformers/all-MiniLM-L6-v2_embeddings")
OUT_DIR          = os.path.join(src_dir, "../data/intermediate_data/sentence-transformers/all-MiniLM-L6-v2_results/slide_figures")
os.makedirs(OUT_DIR, exist_ok=True)

METHODS = ["PCA", "UMAP", "PHATE", "PaCMAP"]
COLORS  = {"PCA": "#4878D0", "UMAP": "#EE854A", "PHATE": "#6ACC65", "PaCMAP": "#D65F5F"}


# ── Slide 1: Trustworthiness & Continuity on benchmark datasets ───────────────
print("Generating slide 1 ...")

benchmark_tc = {
    "RCV1":    {"PCA": (0.7781, 0.8913), "UMAP": (0.9563, 0.9442), "PHATE": (0.7979, 0.9035), "PaCMAP": (0.9360, 0.9333)},
    "arXiv":   {"PCA": (0.7606, 0.8832), "UMAP": (0.9414, 0.9290), "PHATE": (0.7988, 0.8971), "PaCMAP": (0.9051, 0.9274)},
    "Amazon":  {"PCA": (0.7399, 0.8523), "UMAP": (0.9164, 0.9152), "PHATE": (0.7776, 0.8833), "PaCMAP": (0.8645, 0.8992)},
    "DBpedia": {"PCA": (0.7116, 0.8563), "UMAP": (0.9706, 0.9371), "PHATE": (0.8087, 0.9068), "PaCMAP": (0.9353, 0.9325)},
    "WoS":     {"PCA": (0.7866, 0.8850), "UMAP": (0.9504, 0.9434), "PHATE": (0.8505, 0.9207), "PaCMAP": (0.9102, 0.9434)},
}

datasets = list(benchmark_tc.keys())
x = np.arange(len(datasets))
n = len(METHODS)
w = 0.17

fig, ax = plt.subplots(figsize=(11, 5))
for i, method in enumerate(METHODS):
    offset = (i - (n - 1) / 2) * w
    t_vals = [benchmark_tc[d][method][0] for d in datasets]
    c_vals = [benchmark_tc[d][method][1] for d in datasets]
    ax.bar(x + offset - w * 0.26, t_vals, w * 0.5, color=COLORS[method], alpha=1.0)
    ax.bar(x + offset + w * 0.26, c_vals, w * 0.5, color=COLORS[method], alpha=0.45)

ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=12)
ax.set_ylim(0.68, 1.00)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Trustworthiness (solid) & Continuity (faded): Benchmark Datasets", fontsize=13)

legend_handles = [mpatches.Patch(color=COLORS[m], label=m) for m in METHODS]
ax.legend(handles=legend_handles, fontsize=10, loc="lower right")
ax.tick_params(axis="y", labelsize=10)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "slide1_tc_benchmark.png"), dpi=300)
plt.close()
print("  saved slide1_tc_benchmark.png")


# ── Slide 2: DEMaP & Spearman on clean synthetic configs ──────────────────────
print("Generating slide 2 ...")

clean_synth = {
    "Energy\ndepth3": {
        "PCA":    (0.4969, 0.4815),
        "UMAP":   (0.4535, 0.6623),
        "PHATE":  (0.4259, 0.6205),
        "PaCMAP": (0.3846, 0.6068),
    },
    "Offshore\ndepth3": {
        "PCA":    (0.4614, 0.3375),
        "UMAP":   (0.4249, 0.6085),
        "PHATE":  (0.4051, 0.6118),
        "PaCMAP": (0.3638, 0.5334),
    },
    "Energy\ndepth5": {
        "PCA":    (0.5318, 0.4741),
        "UMAP":   (0.5081, 0.6610),
        "PHATE":  (0.4755, 0.5854),
        "PaCMAP": (0.4876, 0.5823),
    },
    "Offshore\ndepth5": {
        "PCA":    (0.5454, 0.4640),
        "UMAP":   (0.4714, 0.7056),
        "PHATE":  (0.5039, 0.7287),
        "PaCMAP": (0.4320, 0.6353),
    },
}

configs = list(clean_synth.keys())
x = np.arange(len(configs))
n = len(METHODS)
w = 0.18

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
metric_labels = ["Spearman Correlation", "DEMaP"]
metric_idx    = [0, 1]

for ax, m_label, m_idx in zip(axes, metric_labels, metric_idx):
    for i, method in enumerate(METHODS):
        offset = (i - (n - 1) / 2) * w
        vals   = [clean_synth[c][method][m_idx] for c in configs]
        ax.bar(x + offset, vals, w * 0.9, color=COLORS[method], label=method, alpha=0.9)

    ax.axhline(0, color="black", linestyle="--", linewidth=1.0, zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=11)
    ax.set_title(m_label, fontsize=13)
    ax.set_ylabel("Score", fontsize=11)
    ax.tick_params(axis="y", labelsize=10)
    ax.axvspan(-0.5, 1.5, color="steelblue", alpha=0.06, zorder=0)
    ax.axvspan( 1.5, 3.5, color="coral",     alpha=0.06, zorder=0)
    ax.set_xlim(-0.5, len(configs) - 0.5)

axes[1].legend(fontsize=10, loc="lower right")
fig.suptitle("DEMaP & Spearman: Clean Synthetic Data  (blue=depth3, red=depth5)", fontsize=13)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "slide2_synth_clean_metrics.png"), dpi=300)
plt.close()
print("  saved slide2_synth_clean_metrics.png")


# ── Slide 3: Noise robustness — DEMaP line plots ──────────────────────────────
print("Generating slide 3 ...")

noise_levels = [0, 25, 50]

noise_demap = {
    "Offshore\ndepth3": {
        "PCA":    [0.3375, 0.3342, 0.4458],
        "UMAP":   [0.6085, 0.4605, 0.4935],
        "PHATE":  [0.6118, 0.4423, 0.4172],
        "PaCMAP": [0.5334, 0.3712, 0.4208],
    },
    "Energy\ndepth3": {
        "PCA":    [0.4815, 0.2899, 0.4418],
        "UMAP":   [0.6623, 0.5851, 0.6430],
        "PHATE":  [0.6205, 0.4569, 0.4546],
        "PaCMAP": [0.6068, 0.3602, 0.6054],
    },
}

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

for ax, (config_name, method_data) in zip(axes, noise_demap.items()):
    for method in METHODS:
        vals = method_data[method]
        ax.plot(noise_levels, vals, marker="o", linewidth=2.2, markersize=7,
                color=COLORS[method], label=method)
    ax.axhline(0, color="black", linestyle="--", linewidth=1.0)
    ax.set_title(config_name.replace("\n", " "), fontsize=13)
    ax.set_xlabel("Noise Level (%)", fontsize=11)
    ax.set_xticks(noise_levels)
    ax.set_xticklabels(["0%", "25%", "50%"], fontsize=11)
    ax.tick_params(axis="y", labelsize=10)

axes[0].set_ylabel("DEMaP", fontsize=12)
axes[1].legend(fontsize=10, loc="lower right")
fig.suptitle("Noise Robustness: DEMaP vs Noise Level", fontsize=13)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "slide3_noise_robustness.png"), dpi=300)
plt.close()
print("  saved slide3_noise_robustness.png")


# ── Slide 3b: Noise robustness — all 4 configs, DEMaP + Spearman ──────────────
print("Generating slide 3b ...")

# (Spearman, DEMaP) per method per config per noise level
noise_all = {
    "Energy depth3": {
        "PCA":    [(0.4969, 0.4815), (0.3044, 0.2899), (0.3919, 0.4418)],
        "UMAP":   [(0.4535, 0.6623), (0.3935, 0.5851), (0.4246, 0.6430)],
        "PHATE":  [(0.4259, 0.6205), (0.2530, 0.4569), (0.2528, 0.4546)],
        "PaCMAP": [(0.3846, 0.6068), (0.2163, 0.3602), (0.4071, 0.6054)],
    },
    "Energy depth5": {
        "PCA":    [(0.5318, 0.4741), (0.3488, 0.2448), (0.3952, 0.2600)],
        "UMAP":   [(0.5081, 0.6610), (0.3038, 0.4220), (0.3187, 0.4419)],
        "PHATE":  [(0.4755, 0.5854), (0.3984, 0.4862), (0.2960, 0.3722)],
        "PaCMAP": [(0.4876, 0.5823), (0.3828, 0.4334), (0.3799, 0.3997)],
    },
    "Offshore depth3": {
        "PCA":    [(0.4614, 0.3375), (0.4575, 0.3342), (0.4575, 0.4458)],
        "UMAP":   [(0.4249, 0.6085), (0.2150, 0.4605), (0.2152, 0.4935)],
        "PHATE":  [(0.4051, 0.6118), (0.1218, 0.4423), (0.0878, 0.4172)],
        "PaCMAP": [(0.3638, 0.5334), (0.1384, 0.3712), (0.2597, 0.4208)],
    },
    "Offshore depth5": {
        "PCA":    [(0.5454, 0.4640), (0.3853, 0.2398), (0.4243, 0.3129)],
        "UMAP":   [(0.4714, 0.7056), (0.3923, 0.5811), (0.3743, 0.5388)],
        "PHATE":  [(0.5039, 0.7287), (0.4438, 0.6609), (0.2941, 0.4851)],
        "PaCMAP": [(0.4320, 0.6353), (0.3103, 0.4558), (0.3592, 0.4889)],
    },
}

configs_all  = list(noise_all.keys())
metric_names = ["Spearman", "DEMaP"]
metric_idx_b = [0, 1]
noise_labels = ["0%", "25%", "50%"]

fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharey="row")

for row, (m_name, m_idx) in enumerate(zip(metric_names, metric_idx_b)):
    for col, config in enumerate(configs_all):
        ax = axes[row][col]
        for method in METHODS:
            vals = [noise_all[config][method][ni][m_idx] for ni in range(3)]
            ax.plot(noise_levels, vals, marker="o", linewidth=2.0, markersize=6,
                    color=COLORS[method], label=method)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.9)
        ax.set_xticks(noise_levels)
        ax.set_xticklabels(noise_labels, fontsize=10)
        ax.tick_params(axis="y", labelsize=9)
        if row == 0:
            ax.set_title(config, fontsize=11, fontweight="bold")
        if col == 0:
            ax.set_ylabel(m_name, fontsize=11)
        if row == 1:
            ax.set_xlabel("Noise Level", fontsize=10)
        if row == 0 and col == 3:
            ax.legend(fontsize=9, loc="lower right")

fig.suptitle("Noise Robustness: Spearman and DEMaP across All Configs", fontsize=13)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "slide3b_noise_all_configs.png"), dpi=300)
plt.close()
print("  saved slide3b_noise_all_configs.png")


# ── Slide 4: Shepard diagrams — PHATE vs PCA on Offshore depth5 clean ─────────
print("Generating slide 4 ...")

STEM       = "Offshore_energy_impacts_on_fisheries_hierarchy_t1.0_maxsub3_depth5_synonyms0_random"
embed_path = os.path.join(EMBEDDING_DIR, f"{STEM}_embed.npy")
pca_path   = os.path.join(REDUCTION_2D_DIR, f"PCA_2d_{STEM}.npy")
phate_path = os.path.join(REDUCTION_2D_DIR, f"PHATE_2d_{STEM}.npy")

x_high = np.load(embed_path)
x_pca  = np.load(pca_path)
x_phate = np.load(phate_path)

np.random.seed(42)
sample_idx = np.random.choice(len(x_high), 500, replace=False)

def shepard_data(x_hi, x_lo, idx):
    dh = pairwise_distances(x_hi[idx]).flatten()
    dl = pairwise_distances(x_lo[idx]).flatten()
    return dh / dh.max(), dl / dl.max()

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

for ax, x_lo, label, color in [
    (axes[0], x_pca,   "PCA",   COLORS["PCA"]),
    (axes[1], x_phate, "PHATE", COLORS["PHATE"]),
]:
    dh, dl = shepard_data(x_high, x_lo, sample_idx)
    ax.scatter(dh, dl, s=2, alpha=0.15, color=color, rasterized=True)
    ax.plot([0, 1], [0, 1], color="red", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.set_title(f"Shepard Diagram: {label}", fontsize=13)
    ax.set_xlabel("High-Dim Distance (Normalized)", fontsize=11)
    ax.set_ylabel("2D Distance (Normalized)", fontsize=11)
    ax.tick_params(labelsize=10)

fig.suptitle("Offshore depth5 (clean): PCA vs PHATE  [PHATE DEMaP=0.73, PCA DEMaP=0.46]", fontsize=13)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "slide4_shepard_pca_vs_phate.png"), dpi=300)
plt.close()
print("  saved slide4_shepard_pca_vs_phate.png")


# ── Slide 5: T/C on synthetic (clean) ─────────────────────────────────────────
print("Generating slide 5 ...")

synth_tc = {
    "Energy\ndepth3":   {"PCA": (0.7813, 0.8738), "UMAP": (0.9490, 0.9307), "PHATE": (0.7938, 0.8965), "PaCMAP": (0.9376, 0.9200)},
    "Offshore\ndepth3": {"PCA": (0.7611, 0.8643), "UMAP": (0.9418, 0.9331), "PHATE": (0.7374, 0.8891), "PaCMAP": (0.9389, 0.9161)},
    "Energy\ndepth5":   {"PCA": (0.7917, 0.8953), "UMAP": (0.9678, 0.9590), "PHATE": (0.7914, 0.9200), "PaCMAP": (0.9567, 0.9435)},
    "Offshore\ndepth5": {"PCA": (0.7815, 0.8887), "UMAP": (0.9684, 0.9605), "PHATE": (0.8314, 0.9382), "PaCMAP": (0.9516, 0.9510)},
}

configs_s = list(synth_tc.keys())
x = np.arange(len(configs_s))
n = len(METHODS)
w = 0.17

fig, ax = plt.subplots(figsize=(11, 5))
for i, method in enumerate(METHODS):
    offset = (i - (n - 1) / 2) * w
    t_vals = [synth_tc[c][method][0] for c in configs_s]
    c_vals = [synth_tc[c][method][1] for c in configs_s]
    ax.bar(x + offset - w * 0.26, t_vals, w * 0.5, color=COLORS[method], alpha=1.0)
    ax.bar(x + offset + w * 0.26, c_vals, w * 0.5, color=COLORS[method], alpha=0.45)

ax.axvspan(-0.5, 1.5, color="steelblue", alpha=0.06, zorder=0)
ax.axvspan( 1.5, 3.5, color="coral",     alpha=0.06, zorder=0)
ax.set_xticks(x)
ax.set_xticklabels(configs_s, fontsize=11)
ax.set_xlim(-0.5, len(configs_s) - 0.5)
ax.set_ylim(0.68, 1.00)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Trustworthiness (solid) & Continuity (faded): Synthetic Data  (blue=depth3, red=depth5)", fontsize=13)

legend_handles = [mpatches.Patch(color=COLORS[m], label=m) for m in METHODS]
ax.legend(handles=legend_handles, fontsize=10, loc="lower right")
ax.tick_params(axis="y", labelsize=10)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "slide5_tc_synthetic.png"), dpi=300)
plt.close()
print("  saved slide5_tc_synthetic.png")


# ── Slide 6: DEMaP & Spearman on benchmark datasets ───────────────────────────
print("Generating slide 6 ...")

benchmark_global = {
    "RCV1":    {"PCA": (0.3965, 0.4687), "UMAP": (0.4863, 0.6847), "PHATE": (0.3725, 0.6453), "PaCMAP": (0.4218, 0.6514)},
    "arXiv":   {"PCA": (0.4858, 0.4665), "UMAP": (0.5430, 0.6406), "PHATE": (0.5796, 0.6243), "PaCMAP": (0.5784, 0.6442)},
    "Amazon":  {"PCA": (0.3708, 0.3605), "UMAP": (0.4400, 0.5482), "PHATE": (0.4184, 0.5123), "PaCMAP": (0.3964, 0.4788)},
    "DBpedia": {"PCA": (0.3095, 0.2374), "UMAP": (0.2967, 0.5000), "PHATE": (0.2244, 0.4385), "PaCMAP": (0.2585, 0.4501)},
    "WoS":     {"PCA": (0.5444, 0.5760), "UMAP": (0.5313, 0.7099), "PHATE": (0.5438, 0.7024), "PaCMAP": (0.5289, 0.6892)},
}

datasets_b = list(benchmark_global.keys())
x = np.arange(len(datasets_b))
n = len(METHODS)
w = 0.18

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)

for ax, (m_label, m_idx) in zip(axes, [("Spearman Correlation", 0), ("DEMaP", 1)]):
    for i, method in enumerate(METHODS):
        offset = (i - (n - 1) / 2) * w
        vals   = [benchmark_global[d][method][m_idx] for d in datasets_b]
        ax.bar(x + offset, vals, w * 0.9, color=COLORS[method], label=method, alpha=0.9)
    ax.axhline(0, color="black", linestyle="--", linewidth=1.0, zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets_b, fontsize=11)
    ax.set_xlim(-0.5, len(datasets_b) - 0.5)
    ax.set_title(m_label, fontsize=13)
    ax.set_ylabel("Score", fontsize=11)
    ax.tick_params(axis="y", labelsize=10)

axes[1].legend(fontsize=10, loc="lower right")
fig.suptitle("DEMaP & Spearman: Benchmark Datasets", fontsize=13)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "slide6_global_benchmark.png"), dpi=300)
plt.close()
print("  saved slide6_global_benchmark.png")


print(f"\nAll figures saved to: {OUT_DIR}")
