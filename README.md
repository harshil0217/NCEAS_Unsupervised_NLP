# NCEAS Unsupervised NLP

NCEAS Project for SS26 CMSE 495 Data Science Capstone, Michigan State University.

**Note:** This project requires a Linux environment with a CUDA-enabled GPU (tested on MSU HPCC).

## Project Video

[![Video Thumbnail](https://img.youtube.com/vi/YqY4ENxIY1E/0.jpg)](https://www.youtube.com/watch?v=YqY4ENxIY1E)

---

## Class Description

CMSE 495 is the MSU Data Science Capstone course where student teams work with real-world community partners to solve data science problems. This project is developed in partnership with Dr. Nathan Brugnone, Maryam Berijanian, and Kuldeep Singh.

---

## Objective

The goal of this project is to benchmark dimensionality reduction methods combined with clustering algorithms on textual embeddings to support unsupervised NLP analysis for US fisheries discourse and narratives. We evaluate how well methods like PHATE, PCA, UMAP, t-SNE, and PaCMAP preserve hierarchical structure in text data when paired with clustering algorithms including Agglomerative, HDBSCAN, Diffusion Condensation, and Hercules. Results are measured using FM index, ARI, AMI, and Rand Index across five benchmark datasets.

---

## Installation

For full installation instructions and environment setup, see:

[INSTALL.md](INSTALL.md)

---

## Key Components

### Text Embeddings
Documents are embedded using two models: `Qwen3-Embedding-0.6B` and `all-MiniLM-L6-v2` via the `sentence-transformers` library.

### Dimensionality Reduction
We compare five reduction methods: PHATE, PCA, UMAP, t-SNE, and PaCMAP. GPU-accelerated implementations (cuML) are used where available.

### Clustering
Four clustering methods are applied at multiple hierarchy levels: Agglomerative Clustering, HDBSCAN, Diffusion Condensation, and Hercules.

### Evaluation
Clustering quality is measured against ground truth labels using FM index, Adjusted Rand Index (ARI), Adjusted Mutual Information (AMI), and Rand Index.


This notebook demonstrates the full pipeline including loading data, generating embeddings, dimensionality reduction, clustering, and visualization.

---
## Hardware Configuration
All experiments in this codebase were conducted in an HPC cluster with 12 CPU cores, 64 GB of RAM, and 2 V100 NVIDIA GPUs

---

## Reproducibility
All reproducibility instructions can be found in [REPRODUCIBILITY.md](./REPRODUCIBILITY.md)


--- 

## Running Benchmark Experiments

Once datasets are in place (see INSTALL.md), run:

```bash
python src/run_models/benchmark_datasets/eval_pipeline.py --dataset arxiv
python src/run_models/benchmark_datasets/eval_pipeline.py --dataset amazon
python src/run_models/benchmark_datasets/eval_pipeline.py --dataset dbpedia
python src/run_models/benchmark_datasets/eval_pipeline.py --dataset rcv1
python src/run_models/benchmark_datasets/eval_pipeline.py --dataset wos
```

### Expected Runtimes
The following are expected runtimes for each dataset pipeline, assuming a hardware configuration similar to ours

-**Arxiv**: 2 Hours
-**Amazon**: 2 Hours
-**Dbpedia**: 5 Hours
-**RCV1**: 30 min
-**WOS**: 4 Hours

---

## Expected Output

Running the benchmark pipeline will generate:
- CSV files with clustering evaluation metrics saved in `results/`
- Scatter grid figures saved in `results/summary_figures/`
- Shepard diagrams saved in `results/shepard_diagrams/`
- Visualization quality metric CSVs saved in `results/viz_metrics/`
- Cached embeddings and reductions in `src/cache/` (not tracked)

---

## Project Structure

```
NCEAS_Unsupervised_NLP/
│
├── data/                                   # All datasets and computed outputs
│   ├── arxiv/                              # arXiv abstracts
│   ├── amazon/                             # Amazon reviews
│   ├── dbpedia/                            # DBpedia articles
│   ├── rcv1/                               # RCV1 news articles
│   ├── WebOfScience/                       # Web of Science papers
│   ├── synthetic/                          # LLM-generated synthetic datasets
│   │   ├── generate.py
│   │   └── generated_data/
│
├── notebooks/
│   ├── milestones/                         # Milestone and demo notebooks
│   │   ├── demo.ipynb
│   │   ├── MVP_demo.ipynb
│   │   └── NCEAS_Reproducibility.ipynb
│   └── evaluations/                        # Analysis and evaluation notebooks
│       ├── metric_tables.ipynb
│       ├── clustering_summary_tables.ipynb
│       ├── visualization_metrics_benchmark.ipynb
│       └── visualization_metrics_synthetic.ipynb
│
├── paper/                                  # ACL-style paper draft
│   └── NCEAS_PHATE_Project_for_ACL/
│       └── latex/
│
├── results/                                # Clustering scores, figures, and Shepard diagrams
│   ├── summary_figures/                    # Scatter grid PNGs
│   ├── shepard_diagrams/                   # Shepard diagram PNGs per model
│   └── viz_metrics/                        # Visualization quality metric CSVs per model
│
├── src/
│   ├── cache/                              # Pipeline cache: embeddings, reductions (not tracked)
│   ├── custom_packages/                    # Custom algorithm implementations
│   │   ├── clusters.py
│   │   ├── diffusion_condensation.py
│   │   ├── fowlkes_mallows.py
│   │   ├── hercules.py
│   │   ├── hierarchical_kmeans_gpu.py
│   │   ├── dendrogram_purity.py
│   │   ├── graph_utils.py
│   │   └── lca_f1.py
│   │
│   └── run_models/
│       ├── benchmark_datasets/             # Benchmark evaluation pipeline
│       │   ├── eval_pipeline.py
│       │   ├── herc_pipeline.py
│       │   ├── viz_metrics_script.py
│       │   ├── visualization_metrics.ipynb
│       │   └── run_eval_pipeline.sh
│       └── synthetic_data/                 # Synthetic data evaluation pipeline
│           ├── scatter_grid_synthetic.py
│           ├── synth_herc_pipeline.py
│           ├── visualization_metrics_synthetic.ipynb
│           ├── viz_metrics_script.py
│           └── run_all.sh
│
├── environment.yml                         # Conda environment (Linux/CUDA)
├── INSTALL.md                              # Installation and data setup instructions
├── LICENSE
└── README.md
```

---

## Report

The project report (ACL-style paper draft) is available in:

[paper/NCEAS_PHATE_Project_for_ACL/latex/](paper/NCEAS_PHATE_Project_for_ACL/latex/)

---

## Authors

[Jisha Goyal](https://github.com/goyaljis)

[Sidharth Rao](https://github.com/CharlieMalick)

[Sukina Alkhalidy](https://github.com/sukaina13)

[Harshil Chidura](https://github.com/harshil0217)
