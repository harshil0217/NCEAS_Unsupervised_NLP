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

---

## Demo

After installing the environment, run:

```bash
jupyter notebook
```

Then open:

```
notebooks/milestones/demo.ipynb
```

This notebook demonstrates the full pipeline including loading data, generating embeddings, dimensionality reduction, clustering, and visualization.

---

## Reproducibility

Full reproducibility instructions including pipeline steps, all figures, and metrics tables are in:

```
REPRODUCIBILITY.md
```

A supplementary Shepard Diagram notebook is also available at:

```
notebooks/milestones/NCEAS_Reproducibility.ipynb
```

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

---

## Expected Output

Running the benchmark pipeline will generate:
- CSV files with clustering evaluation metrics saved in `results/`
- Visualizations produced through the demo notebook
- Additional logs and intermediate outputs depending on the experiment

---

## Project Structure

```
NCEAS_Unsupervised_NLP/
│
├── data/                                   # Benchmark dataset download scripts and instructions
│   ├── arxiv/
│   ├── amazon/
│   ├── dbpedia/
│   ├── rcv1/
│   └── WebOfScience/
│
├── notebooks/
│   ├── milestones/                         # Milestone and demo notebooks
│   │   ├── demo.ipynb
│   │   ├── MVP_demo.ipynb
│   │   └── NCEAS_Reproducibility.ipynb
│   └── analysis/                           # Analysis and evaluation notebooks
│       ├── combine_results.ipynb
│       ├── compare_eval_methods.ipynb
│       ├── embedding_visuals.ipynb
│       ├── final_table.ipynb
│       ├── metric_tables.ipynb
│       ├── ordinal_rankings.ipynb
│       └── parameter_selection.ipynb
│
├── paper/                                  # ACL-style paper draft
│   └── NCEAS_PHATE_Project_for_ACL/
│       └── latex/
│
├── results/                                # Clustering evaluation result tables
│
├── src/
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
│   ├── data_generation/                    # Synthetic data generation using LLMs
│   │   ├── generate.py
│   │   └── generated_data/
│   │
│   ├── intermediate_data/                  # Computed embeddings, reductions, and results
│   │   └── summary_figures/               # Scatter grid figures (Figure 2)
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
├── REPRODUCIBILITY.md                      # Step-by-step reproduction of all figures and results
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
