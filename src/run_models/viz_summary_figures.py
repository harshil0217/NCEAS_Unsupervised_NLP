"""
viz_summary_figures.py — summary figures for team/Nate presentation

Run from repo root or src/:
    python src/run_models/viz_summary_figures.py

Output: data/intermediate_data/sentence-transformers/all-MiniLM-L6-v2_results/summary_figures/
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── path setup ────────────────────────────────────────────────────────────────
current = os.path.dirname(os.path.abspath(__file__))
while os.path.basename(current) != "src" and current != os.path.dirname(current):
    current = os.path.dirname(current)
src_dir = current
os.chdir(src_dir)
sys.path.insert(0, src_dir)

OUT_DIR = os.path.join(src_dir, "../data/intermediate_data/sentence-transformers/all-MiniLM-L6-v2_results/summary_figures")
os.makedirs(OUT_DIR, exist_ok=True)

BENCHMARK_METHODS = ["PCA", "UMAP", "PHATE", "PaCMAP"]
SYNTHETIC_METHODS = ["PCA", "UMAP", "PHATE", "PaCMAP", "TriMAP", "tSNE"]
COLORS  = {"PCA": "#4878D0", "UMAP": "#EE854A", "PHATE": "#6ACC65", "PaCMAP": "#D65F5F",
           "TriMAP": "#956CB4", "tSNE": "#8C613C"}
DATASETS = ["RCV1", "arXiv", "Amazon", "DBpedia", "WoS"]

# ── Data ──────────────────────────────────────────────────────────────────────

benchmark = {
    # dataset -> method -> (Trust, Cont, Spearman, DEMaP)
    "RCV1":    {"PCA": (0.7781, 0.8913, 0.3965, 0.4687), "UMAP": (0.9563, 0.9442, 0.4863, 0.6847), "PHATE": (0.7979, 0.9035, 0.3725, 0.6453), "PaCMAP": (0.9360, 0.9333, 0.4218, 0.6514)},
    "arXiv":   {"PCA": (0.7606, 0.8832, 0.4858, 0.4665), "UMAP": (0.9414, 0.9290, 0.5430, 0.6406), "PHATE": (0.7988, 0.8971, 0.5796, 0.6243), "PaCMAP": (0.9051, 0.9274, 0.5784, 0.6442)},
    "Amazon":  {"PCA": (0.7399, 0.8523, 0.3708, 0.3605), "UMAP": (0.9164, 0.9152, 0.4400, 0.5482), "PHATE": (0.7776, 0.8833, 0.4184, 0.5123), "PaCMAP": (0.8645, 0.8992, 0.3964, 0.4788)},
    "DBpedia": {"PCA": (0.7116, 0.8563, 0.3095, 0.2374), "UMAP": (0.9706, 0.9371, 0.2967, 0.5000), "PHATE": (0.8087, 0.9068, 0.2244, 0.4385), "PaCMAP": (0.9353, 0.9325, 0.2585, 0.4501)},
    "WoS":     {"PCA": (0.7866, 0.8850, 0.5444, 0.5760), "UMAP": (0.9504, 0.9434, 0.5313, 0.7099), "PHATE": (0.8505, 0.9207, 0.5438, 0.7024), "PaCMAP": (0.9102, 0.9434, 0.5289, 0.6892)},
}

# synthetic clean configs: (Trust, Cont, Spearman, DEMaP) — MiniLM values
synthetic_clean = {
    "Energy\ndepth3":   {"PCA": (0.7813, 0.8738, 0.4969, 0.4815), "UMAP": (0.9443, 0.9285, 0.4169, 0.6364), "PHATE": (0.8350, 0.9112, 0.4328, 0.6872), "PaCMAP": (0.9376, 0.9200, 0.3846, 0.6068), "TriMAP": (0.9317, 0.9190, 0.3053, 0.5569), "tSNE": (0.9593, 0.9278, 0.4455, 0.6152)},
    "Offshore\ndepth3": {"PCA": (0.7611, 0.8643, 0.4614, 0.3375), "UMAP": (0.9424, 0.9242, 0.3387, 0.5340), "PHATE": (0.8135, 0.9015, 0.4023, 0.6195), "PaCMAP": (0.9389, 0.9161, 0.3638, 0.5334), "TriMAP": (0.9307, 0.9240, 0.3647, 0.5122), "tSNE": (0.9531, 0.9236, 0.4524, 0.5291)},
    "Energy\ndepth5":   {"PCA": (0.7917, 0.8953, 0.5318, 0.4164), "UMAP": (0.9700, 0.9582, 0.5100, 0.5999), "PHATE": (0.8292, 0.9115, 0.4434, 0.5282), "PaCMAP": (0.9567, 0.9435, 0.4876, 0.5384), "TriMAP": (0.9519, 0.9529, 0.4730, 0.5186), "tSNE": (0.9644, 0.9592, 0.4811, 0.5753)},
    "Offshore\ndepth5": {"PCA": (0.7815, 0.8887, 0.5454, 0.4169), "UMAP": (0.9691, 0.9604, 0.4734, 0.6769), "PHATE": (0.8702, 0.9524, 0.5101, 0.7283), "PaCMAP": (0.9516, 0.9510, 0.4320, 0.5926), "TriMAP": (0.9475, 0.9501, 0.4041, 0.5829), "tSNE": (0.9651, 0.9547, 0.4761, 0.5451)},
}

# noise data: (Spearman, DEMaP) at noise 0, 25, 50 — MiniLM values
noise_data = {
    "Energy depth3": {
        "PCA":    [(0.4969, 0.4815), (0.3044, 0.2899), (0.3919, 0.4418)],
        "UMAP":   [(0.4169, 0.6364), (0.4014, 0.5895), (0.4312, 0.6729)],
        "PHATE":  [(0.4328, 0.6872), (0.3166, 0.5528), (0.3218, 0.5560)],
        "PaCMAP": [(0.3846, 0.6068), (0.2163, 0.3602), (0.4070, 0.6054)],
        "TriMAP": [(0.3053, 0.5569), (0.2607, 0.4037), (0.2356, 0.4520)],
        "tSNE":   [(0.3479, 0.5188), (0.3898, 0.5030), (0.4338, 0.5774)],
    },
    "Energy depth5": {
        "PCA":    [(0.5318, 0.4164), (0.3488, 0.1903), (0.3952, 0.1927)],
        "UMAP":   [(0.5100, 0.5999), (0.2597, 0.3230), (0.3633, 0.3686)],
        "PHATE":  [(0.4434, 0.5282), (0.4422, 0.5704), (0.3928, 0.4449)],
        "PaCMAP": [(0.4876, 0.5384), (0.3828, 0.3801), (0.3798, 0.3388)],
        "TriMAP": [(0.4730, 0.5186), (0.3367, 0.3263), (0.3495, 0.2709)],
        "tSNE":   [(0.4811, 0.5753), (0.4048, 0.4132), (0.4134, 0.3721)],
    },
    "Offshore depth3": {
        "PCA":    [(0.4614, 0.3375), (0.4575, 0.3342), (0.4575, 0.4458)],
        "UMAP":   [(0.3387, 0.5340), (0.2136, 0.4711), (0.2278, 0.4713)],
        "PHATE":  [(0.4023, 0.6195), (0.2375, 0.5860), (0.1924, 0.5483)],
        "PaCMAP": [(0.3638, 0.5334), (0.1385, 0.3713), (0.2597, 0.4208)],
        "TriMAP": [(0.3647, 0.5122), (0.1527, 0.3311), (0.2027, 0.3940)],
        "tSNE":   [(0.4469, 0.4858), (0.3880, 0.4095), (0.4013, 0.4738)],
    },
    "Offshore depth5": {
        "PCA":    [(0.5454, 0.4169), (0.3853, 0.2013), (0.4243, 0.3017)],
        "UMAP":   [(0.4734, 0.6769), (0.3877, 0.5500), (0.3920, 0.5269)],
        "PHATE":  [(0.5101, 0.7283), (0.4885, 0.6643), (0.4336, 0.6266)],
        "PaCMAP": [(0.4320, 0.5926), (0.3104, 0.4152), (0.3591, 0.4763)],
        "TriMAP": [(0.4041, 0.5829), (0.2916, 0.4234), (0.2941, 0.4271)],
        "tSNE":   [(0.4761, 0.5451), (0.4046, 0.4787), (0.4188, 0.5132)],
    },
}

noise_levels = [0, 25, 50]


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1: Benchmark heatmap — all metrics, all methods, all datasets
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 1: Benchmark heatmap ...")

metric_names = ["Trustworthiness", "Continuity", "Spearman", "DEMaP"]
metric_idx   = [0, 1, 2, 3]

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for ax, mname, midx in zip(axes, metric_names, metric_idx):
    data = np.array([[benchmark[ds][m][midx] for m in BENCHMARK_METHODS] for ds in DATASETS])
    vmin = 0.6 if midx < 2 else 0.0
    im   = ax.imshow(data, aspect="auto", cmap="YlGnBu", vmin=vmin, vmax=1.0 if midx < 2 else 0.8)
    ax.set_xticks(range(len(BENCHMARK_METHODS)))
    ax.set_xticklabels(BENCHMARK_METHODS, fontsize=10, rotation=30, ha="right")
    ax.set_yticks(range(len(DATASETS)))
    ax.set_yticklabels(DATASETS if ax == axes[0] else [], fontsize=11)
    ax.set_title(mname, fontsize=12, fontweight="bold")
    for i in range(len(DATASETS)):
        for j in range(len(BENCHMARK_METHODS)):
            ax.text(j, i, f"{data[i,j]:.2f}", ha="center", va="center",
                    fontsize=9, color="black" if data[i,j] < 0.75 else "white")
    plt.colorbar(im, ax=ax, shrink=0.8)

fig.suptitle("Benchmark Datasets: All Metrics by Method", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig1_benchmark_heatmap.png"), dpi=300, bbox_inches="tight")
plt.close()
print("  saved fig1_benchmark_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2: Trustworthiness vs DEMaP scatter — local vs global tradeoff
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 2: Trust vs DEMaP scatter ...")

markers = {"RCV1": "o", "arXiv": "s", "Amazon": "D", "DBpedia": "^", "WoS": "P"}

fig, ax = plt.subplots(figsize=(8, 6))

for ds in DATASETS:
    for method in BENCHMARK_METHODS:
        t  = benchmark[ds][method][0]
        d  = benchmark[ds][method][3]
        ax.scatter(t, d, color=COLORS[method], marker=markers[ds], s=90,
                   zorder=3, edgecolors="white", linewidths=0.5)

# legend for methods
method_handles = [mpatches.Patch(color=COLORS[m], label=m) for m in BENCHMARK_METHODS]
# legend for datasets
ds_handles = [plt.Line2D([0], [0], marker=markers[ds], color="gray",
              linestyle="None", markersize=8, label=ds) for ds in DATASETS]

leg1 = ax.legend(handles=method_handles, title="Method", fontsize=10,
                 loc="upper left", framealpha=0.9)
ax.add_artist(leg1)
ax.legend(handles=ds_handles, title="Dataset", fontsize=10,
          loc="lower right", framealpha=0.9)

ax.set_xlabel("Trustworthiness  (local structure)", fontsize=12)
ax.set_ylabel("DEMaP  (manifold structure)", fontsize=12)
ax.set_title("Local vs Manifold Structure Preservation: Benchmark", fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.set_xlim(0.68, 1.00)
ax.set_ylim(0.15, 0.80)

# annotate DBpedia UMAP outlier
ax.annotate("DBpedia\nUMAP", xy=(0.9706, 0.50), xytext=(0.93, 0.42),
            fontsize=9, color=COLORS["UMAP"],
            arrowprops=dict(arrowstyle="->", color=COLORS["UMAP"], lw=1.2))

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig2_trust_vs_demap_scatter.png"), dpi=300)
plt.close()
print("  saved fig2_trust_vs_demap_scatter.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3: Benchmark DEMaP grouped bar — clean comparison across datasets
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 3: Benchmark DEMaP bar chart ...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
x = np.arange(len(DATASETS))
n = len(BENCHMARK_METHODS)
w = 0.18

for ax, (mname, midx) in zip(axes, [("Spearman Correlation", 2), ("DEMaP", 3)]):
    for i, method in enumerate(BENCHMARK_METHODS):
        offset = (i - (n - 1) / 2) * w
        vals   = [benchmark[ds][method][midx] for ds in DATASETS]
        ax.bar(x + offset, vals, w * 0.9, color=COLORS[method], label=method, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS, fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(mname, fontsize=13, fontweight="bold")
    ax.set_ylim(0, 0.80)
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(axis="y", alpha=0.3)

axes[0].legend(fontsize=10)
fig.suptitle("Global Structure Preservation: Benchmark Datasets", fontsize=14, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig3_benchmark_spearman_demap.png"), dpi=300)
plt.close()
print("  saved fig3_benchmark_spearman_demap.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4: Synthetic clean — DEMaP grouped bar, benchmark vs synthetic side by side
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 4: Synthetic DEMaP bar chart ...")

synth_configs = list(synthetic_clean.keys())
x = np.arange(len(synth_configs))
n_s = len(SYNTHETIC_METHODS)
w = 0.13

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, (mname, midx) in zip(axes, [("Spearman Correlation", 2), ("DEMaP", 3)]):
    for i, method in enumerate(SYNTHETIC_METHODS):
        offset = (i - (n_s - 1) / 2) * w
        vals   = [synthetic_clean[c][method][midx] for c in synth_configs]
        ax.bar(x + offset, vals, w * 0.9, color=COLORS[method], label=method, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(synth_configs, fontsize=11)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(mname, fontsize=13, fontweight="bold")
    ax.set_ylim(0, 0.80)
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.axvspan(-0.5, 1.5, color="steelblue", alpha=0.06, zorder=0)
    ax.axvspan( 1.5, 3.5, color="coral",     alpha=0.06, zorder=0)
    ax.set_xlim(-0.5, len(synth_configs) - 0.5)

axes[0].legend(fontsize=10)
fig.suptitle("Global Structure Preservation: Synthetic Data (clean)  [blue=depth3, red=depth5]",
             fontsize=14, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig4_synthetic_spearman_demap.png"), dpi=300)
plt.close()
print("  saved fig4_synthetic_spearman_demap.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5: Noise robustness — 2x4 grid, DEMaP line plots for all configs
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 5: Noise robustness grid ...")

configs_noise = list(noise_data.keys())
fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)

for ax, config in zip(axes, configs_noise):
    for method in SYNTHETIC_METHODS:
        vals = [noise_data[config][method][ni][1] for ni in range(3)]  # DEMaP
        ax.plot(noise_levels, vals, marker="o", linewidth=2.2, markersize=7,
                color=COLORS[method], label=method)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_title(config, fontsize=11, fontweight="bold")
    ax.set_xlabel("Noise (%)", fontsize=10)
    ax.set_xticks(noise_levels)
    ax.set_xticklabels(["0%", "25%", "50%"], fontsize=10)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(-0.05, 0.80)

axes[0].set_ylabel("DEMaP", fontsize=12)
axes[-1].legend(fontsize=9, loc="lower right")
fig.suptitle("Noise Robustness: DEMaP across All Synthetic Configs", fontsize=14, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig5_noise_robustness_demap.png"), dpi=300)
plt.close()
print("  saved fig5_noise_robustness_demap.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 6: Method comparison radar — average across benchmark datasets
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 6: Radar chart ...")

metric_labels = ["Trustworthiness", "Continuity", "Spearman", "DEMaP"]
N = len(metric_labels)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

for method in BENCHMARK_METHODS:
    avg_vals = []
    for midx in range(4):
        avg = np.mean([benchmark[ds][method][midx] for ds in DATASETS])
        avg_vals.append(avg)
    avg_vals += avg_vals[:1]
    ax.plot(angles, avg_vals, linewidth=2, color=COLORS[method], label=method)
    ax.fill(angles, avg_vals, alpha=0.08, color=COLORS[method])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(metric_labels, fontsize=12)
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=11)
ax.set_title("Average Performance by Method\n(Benchmark Datasets)", fontsize=13,
             fontweight="bold", pad=20)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig6_radar_benchmark.png"), dpi=300, bbox_inches="tight")
plt.close()
print("  saved fig6_radar_benchmark.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 7: Summary — benchmark vs synthetic DEMaP side by side
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 7: Benchmark vs synthetic DEMaP summary ...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# left: benchmark DEMaP
n_b = len(BENCHMARK_METHODS)
x_b = np.arange(len(DATASETS))
for i, method in enumerate(BENCHMARK_METHODS):
    offset = (i - (n_b - 1) / 2) * w
    vals   = [benchmark[ds][method][3] for ds in DATASETS]
    axes[0].bar(x_b + offset, vals, w * 0.9, color=COLORS[method], label=method, alpha=0.9)
axes[0].set_xticks(x_b)
axes[0].set_xticklabels(DATASETS, fontsize=12)
axes[0].set_ylabel("DEMaP", fontsize=12)
axes[0].set_title("Benchmark Datasets", fontsize=13, fontweight="bold")
axes[0].set_ylim(0, 0.80)
axes[0].grid(axis="y", alpha=0.3)
axes[0].legend(fontsize=10)

# right: synthetic clean DEMaP
x_s = np.arange(len(synth_configs))
for i, method in enumerate(SYNTHETIC_METHODS):
    offset = (i - (n_s - 1) / 2) * w
    vals   = [synthetic_clean[c][method][3] for c in synth_configs]
    axes[1].bar(x_s + offset, vals, w * 0.9, color=COLORS[method], alpha=0.9)
axes[1].set_xticks(x_s)
axes[1].set_xticklabels(synth_configs, fontsize=11)
axes[1].set_ylabel("DEMaP", fontsize=12)
axes[1].set_title("Synthetic Data (clean)", fontsize=13, fontweight="bold")
axes[1].set_ylim(0, 0.80)
axes[1].grid(axis="y", alpha=0.3)
axes[1].axvspan(-0.5, 1.5, color="steelblue", alpha=0.06, zorder=0)
axes[1].axvspan( 1.5, 3.5, color="coral",     alpha=0.06, zorder=0)
axes[1].set_xlim(-0.5, len(synth_configs) - 0.5)

fig.suptitle("DEMaP: Benchmark vs Synthetic Data", fontsize=14, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig7_demap_benchmark_vs_synthetic.png"), dpi=300)
plt.close()
print("  saved fig7_demap_benchmark_vs_synthetic.png")


print(f"\nAll figures saved to:\n  {OUT_DIR}")
