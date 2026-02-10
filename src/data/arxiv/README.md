## arXiv Dataset

The arXiv dataset is used as a primary hard benchmark for Phase 1.1.

This benchmark consists of two complementary components:

### 1. arXiv Metadata and Abstracts
Source:
https://www.kaggle.com/datasets/Cornell-University/arxiv

Contains:
- Paper titles
- Abstracts (text used for analysis)
- Category labels

Used for:
- Text embeddings
- Dimensionality reduction (PHATE, UMAP, PCA)
- Clustering

### 2. arXiv Subject Taxonomy
Source:
https://arxiv.org/category_taxonomy

This is a reference hierarchy (not a text dataset) defining:
- Category → subcategory → sub-subcategory

Used for:
- Ground-truth hierarchical structure
- Evaluation of clustering results

---

### Data Storage Policy
Due to size considerations, the full arXiv dataset is **not stored in this repository**.

Team members should download the dataset locally (e.g., via Kaggle) and place the files in this directory for experimentation in later phases.

