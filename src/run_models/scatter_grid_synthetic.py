"""
scatter_grid_synthetic.py

Produces a Figure-2-style scatter grid for synthetic datasets with both embedding models:
- Rows: 8 (4 configs × 2 models: MiniLM top half, Qwen bottom half)
- Columns: 6 DR methods (PHATE, PCA, UMAP, t-SNE, PaCMAP, TriMAP)
- Points colored by top-level category (category 0)

Run from repo root:
    python src/run_models/scatter_grid_synthetic.py
"""

import os
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# path setup
current = os.path.dirname(os.path.abspath(__file__))
while os.path.basename(current) != "src" and current != os.path.dirname(current):
    current = os.path.dirname(current)
src_dir = current
os.chdir(src_dir)
sys.path.insert(0, src_dir)

EMBEDDING_MODELS = [
    ("MiniLM", "intermediate_data/sentence-transformers/all-MiniLM-L6-v2_reduced_2d"),
    ("Qwen",   "intermediate_data/Qwen/Qwen3-Embedding-0.6B_reduced_2d"),
]

LABEL_DIR = "data_generation/generated_data"
OUT_DIR   = "intermediate_data/summary_figures"
os.makedirs(OUT_DIR, exist_ok=True)

METHODS = ["PHATE", "PCA", "UMAP", "tSNE", "PaCMAP", "TriMAP"]
METHOD_LABELS = {"tSNE": "t-SNE"}

CONFIGS = [
    (
        "Energy_Ecosystems_and_Humans_hierarchy_t1.0_maxsub3_depth5_synonyms0_random",
        "Energy_Ecosystems_and_Humans_hierarchy_t1.0_maxsub3_depth5_synonyms0_noise0.0_random",
        "Ecosystems (d)",
    ),
    (
        "Energy_Ecosystems_and_Humans_hierarchy_t1.0_maxsub5_depth3_synonyms0_random",
        "Energy_Ecosystems_and_Humans_hierarchy_t1.0_maxsub5_depth3_synonyms0_noise0.0_random",
        "Ecosystems (s)",
    ),
    (
        "Offshore_energy_impacts_on_fisheries_hierarchy_t1.0_maxsub3_depth5_synonyms0_random",
        "Offshore_energy_impacts_on_fisheries_hierarchy_t1.0_maxsub3_depth5_synonyms0_noise0.0_random",
        "Fisheries (d)",
    ),
    (
        "Offshore_energy_impacts_on_fisheries_hierarchy_t1.0_maxsub5_depth3_synonyms0_random",
        "Offshore_energy_impacts_on_fisheries_hierarchy_t1.0_maxsub5_depth3_synonyms0_noise0.0_random",
        "Fisheries (s)",
    ),
]

PALETTE = [
    "#E63946", "#457B9D", "#2A9D8F", "#E9C46A",
    "#F4A261", "#6A4C93", "#80B918", "#FF6B6B",
    "#4CC9F0", "#F72585",
]

# Extended palette for category 1 (up to 27 unique values)
PALETTE_EXT = [
    "#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#F4A261",
    "#6A4C93", "#80B918", "#FF6B6B", "#4CC9F0", "#F72585",
    "#264653", "#E76F51", "#A8DADC", "#F1FAEE", "#1D3557",
    "#52B788", "#B7E4C7", "#D62828", "#023E8A", "#F77F00",
    "#FCBF49", "#EAE2B7", "#003049", "#8338EC", "#3A86FF",
    "#FB5607", "#FFBE0B",
]

def load_labels(label_csv_stem, cat_level=0):
    path = os.path.join(LABEL_DIR, f"{label_csv_stem}.csv")
    with open(path) as f:
        rows = list(csv.DictReader(f))
    col = f"category {cat_level}"
    fallback = "category 0"
    return [r[col] if r[col] else r[fallback] for r in rows]

def encode_labels(labels):
    unique = sorted(set(labels))
    mapping = {v: i for i, v in enumerate(unique)}
    return np.array([mapping[l] for l in labels]), unique

n_cols = len(METHODS)

def make_scatter_grid(model_name, reduction_dir, out_filename, title,
                      phate_cat_level=0):
    """
    phate_cat_level: category level used to color PHATE panels (0 or 1).
    All other methods always use category 0.
    """
    n_rows = len(CONFIGS)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 4, n_rows * 3.5),
                             facecolor="white")
    fig.patch.set_facecolor("white")

    for r, (stem, label_stem, row_label) in enumerate(CONFIGS):
        labels_cat0  = load_labels(label_stem, cat_level=0)
        labels_phate = load_labels(label_stem, cat_level=phate_cat_level)
        encoded_cat0,  unique_cat0  = encode_labels(labels_cat0)
        encoded_phate, unique_phate = encode_labels(labels_phate)
        colors_cat0  = [PALETTE[i % len(PALETTE)] for i in encoded_cat0]
        # use extended palette for cat1 so colors don't repeat
        pal_phate    = PALETTE_EXT if phate_cat_level != 0 else PALETTE
        colors_phate = [pal_phate[i % len(pal_phate)] for i in encoded_phate]

        for c, method in enumerate(METHODS):
            ax = axes[r, c]
            ax.set_facecolor("#F7F7F7")
            for spine in ax.spines.values():
                spine.set_edgecolor("#CCCCCC")
                spine.set_linewidth(0.8)

            npy_path = os.path.join(reduction_dir, f"{method}_2d_{stem}.npy")

            use_phate_colors = (method == "PHATE" and phate_cat_level != 0)
            point_colors = colors_phate if use_phate_colors else colors_cat0

            if not os.path.exists(npy_path):
                ax.text(0.5, 0.5, "missing", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, color="gray")
            else:
                x2d = np.load(npy_path)
                ax.scatter(x2d[:, 0], x2d[:, 1], c=point_colors,
                           s=12, alpha=0.85, linewidths=0, rasterized=True)

            ax.set_xticks([])
            ax.set_yticks([])

            if r == 0:
                if method == "PHATE" and phate_cat_level != 0:
                    col_label = f"PHATE\n(level {phate_cat_level})"
                else:
                    col_label = METHOD_LABELS.get(method, method)
                ax.set_title(col_label, fontsize=12, fontweight="bold", pad=6)
            if c == 0:
                ax.set_ylabel(row_label, fontsize=11, fontweight="bold", labelpad=8)

        # legend always uses cat0 (5-6 entries) — keeps layout clean like original
        patches = [
            mpatches.Patch(color=PALETTE[i % len(PALETTE)], label=cat)
            for i, cat in enumerate(unique_cat0)
        ]
        axes[r, -1].legend(
            handles=patches,
            loc="center left",
            bbox_to_anchor=(1.03, 0.5),
            fontsize=8,
            frameon=True,
            framealpha=0.9,
            edgecolor="#CCCCCC",
            title="Category",
            title_fontsize=9,
        )

    plt.suptitle(title, fontsize=14, fontweight="bold", y=1.005)
    plt.tight_layout(h_pad=1.5, w_pad=1.0)

    out_path = os.path.join(OUT_DIR, out_filename)
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {out_path}")

for model_name, reduction_dir in EMBEDDING_MODELS:
    slug = "minilm" if "MiniLM" in model_name else "qwen"
    # category 0 versions (original)
    make_scatter_grid(
        model_name=model_name,
        reduction_dir=reduction_dir,
        out_filename=f"fig2_scatter_grid_{slug}.png",
        title=f"Synthetic Dataset Visualizations ({model_name})",
        phate_cat_level=0,
    )
    # PHATE colored by category 1
    make_scatter_grid(
        model_name=model_name,
        reduction_dir=reduction_dir,
        out_filename=f"fig2_scatter_grid_{slug}_phate_cat1.png",
        title=f"Synthetic Dataset Visualizations ({model_name}): PHATE Colored by Category 1",
        phate_cat_level=1,
    )
