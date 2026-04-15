# NCEAS NLP Reproducibility Guide

Step-by-step instructions to reproduce all figures in the NCEAS Unsupervised NLP project.

---

## Table of Contents

1. [Setup](#1-setup)
2. [Figure 2: Scatter Grids (Synthetic Data)](#2-figure-2-scatter-grids-synthetic-data)
3. [Shepard Diagrams (Synthetic Data)](#3-shepard-diagrams-synthetic-data)
4. [Shepard Diagrams (Benchmark Datasets)](#4-shepard-diagrams-benchmark-datasets)
5. [Phase 3.1 Figures](#5-phase-31-figures)

---

## 1. Setup

### Environment

Follow [INSTALL.md](INSTALL.md) to clone the repo and set up the conda environment.

```bash
conda activate phate-env
```

### Data

This project uses five benchmark datasets and LLM-generated synthetic data. See [INSTALL.md](INSTALL.md) for full download instructions.

| Dataset | Path |
|---------|------|
| RCV1 | `data/rcv1/rcv1.csv` |
| arXiv | `data/arxiv/arxiv_clean.csv` |
| Amazon | `data/amazon/train_40k.csv`, `val_10k.csv` |
| DBpedia | `data/dbpedia/DBPEDIA_test.csv` |
| Web of Science | `data/WebOfScience/Data.xlsx` |
| Synthetic | `data/synthetic/generated_data/` |

### Intermediate Data

Generating embeddings is time-intensive. To skip this step, use the precomputed embeddings and 2D reductions by running the full pipeline once and caching the results (see steps below).

---

## 2. Figure 2: Scatter Grids (Synthetic Data)

**Used in:** Paper (Figure 2), final presentation.

Each figure is a 4×6 grid: rows = 4 synthetic dataset configurations (2 topics × 2 hierarchy shapes), columns = 6 DR methods (PHATE, PCA, UMAP, t-SNE, PaCMAP, TriMAP). Points are colored by category. Two versions are produced per embedding model: top-level category (cat0) and subcategory (cat1).

**Output files** (saved to `results/summary_figures/`):
- `fig2_scatter_grid_minilm.png`
- `fig2_scatter_grid_qwen.png`
- `fig2_scatter_grid_minilm_cat1.png`
- `fig2_scatter_grid_qwen_cat1.png`

---

### Step 1: Generate Synthetic Data, Embeddings, and 2D Reductions

Skip if intermediate data has already been generated.

```bash
cd src/run_models/synthetic_data
bash run_all.sh
```

This generates synthetic CSVs under `data/synthetic/generated_data/`, then embeds each config with both `all-MiniLM-L6-v2` and `Qwen3-Embedding-0.6B` and reduces to 2D using all six DR methods. Results are cached in:

```
src/cache/sentence-transformers/all-MiniLM-L6-v2_reduced_2d/
src/cache/Qwen/Qwen3-Embedding-0.6B_reduced_2d/
```

To generate a single config manually:

```bash
# from repo root
python data/synthetic/generate.py \
  --theme Energy_Ecosystems_and_Humans \
  --t 1.0 --max_sub 3 --depth 5 \
  --synonyms 0 --branching random --add_noise 0
```

Available themes: `Energy_Ecosystems_and_Humans`, `Offshore_energy_impacts_on_fisheries`

---

### Step 2: Generate the Scatter Grid Figures

```bash
# from repo root
python src/run_models/synthetic_data/scatter_grid_synthetic.py
```

This reads the 2D reductions from `src/cache/` and the labels from `data/synthetic/generated_data/`, then saves all four PNGs to `results/summary_figures/`.

---

### Expected Output

**MiniLM: colored by top-level category:**

![Scatter Grid MiniLM Cat0](results/summary_figures/fig2_scatter_grid_minilm.png)

**Qwen: colored by top-level category:**

![Scatter Grid Qwen Cat0](results/summary_figures/fig2_scatter_grid_qwen.png)

**MiniLM: colored by subcategory:**

![Scatter Grid MiniLM Cat1](results/summary_figures/fig2_scatter_grid_minilm_cat1.png)

**Qwen: colored by subcategory:**

![Scatter Grid Qwen Cat1](results/summary_figures/fig2_scatter_grid_qwen_cat1.png)

---

## 3. Shepard Diagrams (Synthetic Data)

**Used in:** Paper (Phase 3.2), final presentation.

Shepard diagrams compare pairwise distances in the original high-dimensional embedding space against distances in the 2D reduced space. Points near the diagonal indicate better global distance preservation. One diagram is produced per DR method per synthetic config.

**Output files** (saved to `results/shepard_diagrams/{embedding_model}/`):
- `shepard_{config}_{method}.png`: one per synthetic config × DR method

---

### Step 1: Generate Synthetic Data and Embeddings

Follow Step 1 from [Figure 2](#2-figure-2-scatter-grids-synthetic-data) above, or skip if intermediate data has already been generated.

---

### Step 2: Run the Visualization Metrics Notebook

Open and run all cells in:

```
src/run_models/synthetic_data/visualization_metrics_synthetic.ipynb
```

Set the `embedding_model` variable at the top of the notebook to switch between models:

```python
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
# or
embedding_model = "Qwen/Qwen3-Embedding-0.6B"
```

The notebook computes Trustworthiness, Continuity, Spearman, and DEMaP for each config and DR method, then saves a Shepard diagram for each combination. Results are cached so the notebook can be safely re-run.

---

## 4. Shepard Diagrams (Benchmark Datasets)

**Used in:** Paper (Phase 3.2), final presentation.

Same as above but for the five real-world benchmark datasets. For large datasets (>10,000 points), metrics are computed over 30 subsamples of 10,000 points each.

**Output files** (saved to `results/shepard_diagrams/{embedding_model}/`):
- `shepard_{dataset}_{method}.png`: one per dataset × DR method

---

### Step 1: Run Metrics on HPCC

Benchmark datasets require GPU and are processed on HPCC (MSU) via SLURM:

```bash
sbatch slurm_viz_{dataset}_{model}.sb
# e.g.: sbatch slurm_viz_rcv1_minilm.sb
```

The script `src/run_models/benchmark_datasets/viz_metrics_script.py` handles embedding, dimensionality reduction, subsampling, and incremental CSV output. It is safe to rerun, results are cached and the script resumes from where it left off.

---

### Step 2: Run the Benchmark Visualization Metrics Notebook

Once all CSVs are synced from HPCC, open and run all cells in:

```
src/run_models/benchmark_datasets/visualization_metrics.ipynb
```

Set the `embedding_model` and `dataset` variables at the top to select which results to visualize. The notebook produces Shepard diagrams saved to `src/cache/{embedding_model}_results/`.

---

## 5. Phase 3.1 Figures

> **Note:** Instructions for Phase 3.1 clustering figures will be added soon.
