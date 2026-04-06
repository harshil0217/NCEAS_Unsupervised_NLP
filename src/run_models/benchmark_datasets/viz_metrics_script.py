"""
Visualization Quality Metrics: Benchmark Datasets (Batch Script)

Runs Trustworthiness, Continuity, Spearman Correlation, and DEMaP
for all benchmark datasets. Results and Shepard diagrams are saved
to intermediate_data/{embedding_model}_results/.

Usage:
    python src/run_models/benchmark_datasets/viz_metrics_script.py
"""

import os
import sys

# navigate to src/
current_dir = os.path.dirname(os.path.abspath(__file__))
while os.path.basename(current_dir) != 'src':
    parent = os.path.abspath(os.path.join(current_dir, '..'))
    if parent == current_dir:
        raise FileNotFoundError("src/ not found in directory tree.")
    current_dir = parent
os.chdir(current_dir)
sys.path.insert(0, current_dir)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for script use
import matplotlib.pyplot as plt
import phate
import pacmap
import umap as umap_pkg
from sklearn.decomposition import PCA as skPCA
from sklearn.manifold import trustworthiness
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import shortest_path
from scipy.stats import spearmanr

# ========================
# Config
# ========================

embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

datasets = ["rcv1", "arxiv", "amazon", "dbpedia", "wos"]

embedding_dir    = f"{embedding_model}_embeddings"
reduction_2d_dir = f"intermediate_data/{embedding_model}_reduced_2d"
results_dir      = f"intermediate_data/{embedding_model}_results"

os.makedirs(reduction_2d_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# ========================
# Helper functions
# ========================

def load_or_compute_2d(name, path, compute_fn):
    if os.path.exists(path):
        print(f"  Loading cached 2D {name} from {path}...")
        return np.load(path)
    print(f"  Computing 2D {name}...")
    result = compute_fn()
    np.save(path, result)
    print(f"  Saved to {path}")
    return result

def compute_continuity(x_high, x_low, n_neighbors=15):
    n = x_high.shape[0]
    d_high   = pairwise_distances(x_high)
    d_low    = pairwise_distances(x_low)
    rank_low = np.argsort(np.argsort(d_low, axis=1), axis=1)
    continuity = 0.0
    for i in range(n):
        neighbors_high = set(np.argsort(d_high[i])[1:n_neighbors+1])
        neighbors_low  = set(np.argsort(d_low[i])[1:n_neighbors+1])
        missing = neighbors_high - neighbors_low
        for j in missing:
            continuity += rank_low[i, j] - n_neighbors
    norm = 2.0 / (n * n_neighbors * (2 * n - 3 * n_neighbors - 1))
    return 1 - norm * continuity

def compute_demap(x_high, x_low_2d, k_min=3, k_max=15):
    for k in range(k_min, k_max + 1):
        knn = kneighbors_graph(x_high, n_neighbors=k, mode='distance', include_self=False)
        geo = shortest_path(knn, directed=False)
        if not np.any(np.isinf(geo)):
            print(f"    DEMaP using K={k} (min connected)")
            break
    if np.any(np.isinf(geo)):
        max_finite = np.nanmax(geo[np.isfinite(geo)])
        geo[np.isinf(geo)] = 1 + max_finite
    idx      = np.triu_indices(x_high.shape[0], k=1)
    geo_flat = geo[idx]
    euc_flat = pairwise_distances(x_low_2d)[idx]
    return spearmanr(geo_flat, euc_flat)[0]

def plot_shepard(x_high, x_low, name, dataset, sample_size=500):
    indices = np.random.choice(len(x_high), min(sample_size, len(x_high)), replace=False)
    d_high  = pairwise_distances(x_high[indices]).flatten()
    d_low   = pairwise_distances(x_low[indices]).flatten()
    d_high  = d_high / np.max(d_high)
    d_low   = d_low  / np.max(d_low)
    plt.figure(figsize=(6, 6))
    plt.scatter(d_high, d_low, alpha=0.1, s=1, color='teal')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', alpha=0.5)
    plt.title(f"Shepard Diagram: {name}")
    plt.xlabel("High-Dimensional Distance (Normalized)")
    plt.ylabel("Low-Dimensional Distance (Normalized)")
    plt.tight_layout()
    filename = os.path.join(results_dir, f"shepard_{dataset}_{name.lower()}.png")
    plt.savefig(filename, dpi=300)
    plt.close()
    return filename

# ========================
# Main loop
# ========================

for dataset in datasets:
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset}")
    print(f"{'='*60}")

    embed_path = f"{embedding_dir}/{dataset}.npy"
    if not os.path.exists(embed_path):
        print(f"  Embeddings not found at {embed_path}, skipping.")
        continue

    x_high = np.load(embed_path)
    print(f"  Embeddings shape: {x_high.shape}")

    x_high_full = x_high
    print(f"  Using full dataset ({x_high.shape[0]} points) for 2D reductions")

    suffix = f"{dataset}_full{len(x_high_full)}"

    # compute 2D reductions on full dataset
    reductions = {}
    reductions["PCA"] = load_or_compute_2d(
        "PCA", f"{reduction_2d_dir}/PCA_2d_{suffix}.npy",
        lambda: skPCA(n_components=2, random_state=67).fit_transform(x_high_full)
    )
    reductions["UMAP"] = load_or_compute_2d(
        "UMAP", f"{reduction_2d_dir}/UMAP_2d_{suffix}.npy",
        lambda: umap_pkg.UMAP(n_components=2, min_dist=0.05, n_neighbors=10, random_state=67).fit_transform(x_high_full)
    )
    reductions["PHATE"] = load_or_compute_2d(
        "PHATE", f"{reduction_2d_dir}/PHATE_2d_{suffix}.npy",
        lambda: phate.PHATE(n_jobs=-2, random_state=67, n_components=2).fit_transform(x_high_full)
    )
    reductions["PaCMAP"] = load_or_compute_2d(
        "PaCMAP", f"{reduction_2d_dir}/PaCMAP_2d_{suffix}.npy",
        lambda: pacmap.PaCMAP(n_components=2, random_state=67).fit_transform(x_high_full)
    )

    x_high_sub     = x_high_full
    reductions_sub = reductions
    print(f"  Using full dataset for metrics ({x_high_full.shape[0]} points)")

    # compute metrics
    stats = []
    for name, x_low_2d in reductions_sub.items():
        t_score = trustworthiness(x_high_sub, x_low_2d, n_neighbors=15)
        c_score = compute_continuity(x_high_sub, x_low_2d, n_neighbors=15)
        d_high_flat = pairwise_distances(x_high_sub).flatten()
        d_low_flat  = pairwise_distances(x_low_2d).flatten()
        spearman_corr, _ = spearmanr(d_high_flat, d_low_flat)
        print(f"  Computing DEMaP for {name}...")
        demap_score = compute_demap(x_high_sub, x_low_2d)
        stats.append({
            "Method": name,
            "Trustworthiness": round(t_score, 4),
            "Continuity": round(c_score, 4),
            "Spearman Correlation": round(spearman_corr, 4),
            "DEMaP": round(demap_score, 4)
        })
        print(f"  {name}: T={t_score:.4f}, C={c_score:.4f}, Spearman={spearman_corr:.4f}, DEMaP={demap_score:.4f}")

    # save metrics CSV
    output_path = os.path.join(results_dir, f"viz_metrics_{dataset}.csv")
    pd.DataFrame(stats).to_csv(output_path, index=False)
    print(f"  Saved metrics to {output_path}")

    # save Shepard diagrams
    for name, x_low_2d in reductions_sub.items():
        f = plot_shepard(x_high_sub, x_low_2d, name, dataset)
        print(f"  Saved Shepard: {f}")

print(f"\n{'='*60}")
print("All datasets complete.")
print(f"{'='*60}")
