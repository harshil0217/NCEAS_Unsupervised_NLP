# Project Plan – NCEAS Unsupervised NLP

## Overview
This project benchmarks dimensionality reduction and clustering methods on text embeddings to evaluate how well hierarchical structure is preserved in datasets such as arXiv, RCV1, Amazon, and DBpedia.

## What We Built
- Full pipeline for:
  - Text embeddings (MiniLM, Qwen)
  - Dimensionality reduction (PCA, UMAP, PHATE, PaCMAP, TriMAP)
  - Clustering (Agglomerative, HDBSCAN, Diffusion Condensation, HERCULES)
- Evaluation using ARI, AMI, FM, and Rand Index

## How to Run
1. Clone the repository
2. Follow installation instructions in INSTALL.md
3. Run:
   python src/run_models/benchmark_datasets/eval_pipeline.py --dataset arxiv

## Outputs
- Results saved as CSV files in `results/`
- Visualizations available through notebooks

## Data Access
Datasets are not included in the repo. See INSTALL.md for download instructions.

## Repository Guide
- README.md → overview  
- INSTALL.md → setup  
- notebooks/ → demo  
- src/ → code  
- docs/NCEAS_PHATE_Project_for_ACL/latex/ → project report (ACL paper draft)
- 
## Sharing Plan

- Code and pipeline are available in the GitHub repository
- Large datasets and result files are stored in the NCEAS Teams shared workspace
- The project report is located in the docs/ directory
- Setup and execution instructions are provided in README.md and INSTALL.md