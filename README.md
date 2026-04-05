# NCEAS Unsupervised NLP

⚠️ **Note:** This project requires a Linux environment with a CUDA-enabled GPU (tested on MSU HPCC).

NCEAS Project for SS26 CMSE 495 Data Science Capstone, Michigan State University

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
Four clustering methods are applied at multiple hierarchy levels: Agglomerative Clustering, HDBSCAN, Diffusion Condensation, and Hercules

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
notebooks/demo.ipynb
```

This notebook demonstrates the full pipeline including loading data, generating embeddings, dimensionality reduction, clustering, and visualization.

---

## Reproducibility

Instructions to reproduce the Shepard Diagram figures from our final report are in:

```
NCEAS_Reproducibility.ipynb
```

This notebook loads precomputed embeddings from the NCEAS Teams Data folder and generates Shepard Diagrams for PCA, UMAP, PHATE, and PaCMAP on the RCV1 dataset.

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
├── notebooks/                    # Demo notebook
│   └── demo.ipynb
│
├── docs/                         # Project documentation and report
│   ├── project_plan.md
│   └── NCEAS_PHATE_Project_for_ACL/
│       └── latex/
│
├── src/
│   ├── custom_packages/          # Custom algorithm implementations
│   │   ├── diffusion_condensation.py
│   │   ├── fowlkes_mallows.py
│   │   ├── hercules.py
│   │   ├── hierarchical_kmeans_gpu.py
│   │   └── clusters.py
│   │
│   ├── data/                     # Benchmark datasets (not included in repo)
│   │   ├── arxiv/
│   │   ├── amazon/
│   │   ├── dbpedia/
│   │   ├── rcv1/
│   │   └── WebOfScience/
│   │
│   ├── data_generation/          # Synthetic data generation using LLMs
│   │   └── generate.py
│   │
│   └── run_models/               # Experiment pipelines
│       ├── benchmark_datasets/
│       │   ├── eval_pipeline.py  # Main benchmark evaluation pipeline
│       │   └── herc_pipeline.py  # HERCULES hierarchical clustering pipeline
│       ├── synthetic_data/
│       │   ├── eval_script.py
│       │   └── synth_herc_pipeline.py
│       ├── visualization_metrics.ipynb           # Visualization quality metrics (benchmark)
│       └── visualization_metrics_synthetic.ipynb # Visualization quality metrics (synthetic)
│
├── environment.yml               # Conda environment (Linux/CUDA)
├── INSTALL.md                    # Installation instructions
├── LICENSE                       # Apache 2.0 License
└── README.md                     # Project overview
```

---

## Report

The project report (ACL-style paper draft) is available in:

docs/NCEAS_PHATE_Project_for_ACL/latex/

## Authors

[Jisha Goyal](https://github.com/goyaljis)

[Sidharth Rao](https://github.com/CharlieMalick)

[Sukina Alkhalidy](https://github.com/sukaina13)

[Harshil Chidura](https://github.com/harshil0217)
