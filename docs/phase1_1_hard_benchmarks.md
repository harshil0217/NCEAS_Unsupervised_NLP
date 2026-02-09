# Phase 1.1: Hard Benchmark Datasets

## Objective
The primary goal of Phase 1.1 is to move away from reliance on synthetic datasets used in earlier stages of this project and instead ground evaluation in real, well-established benchmark datasets. This directly addresses reviewer feedback requesting benchmarks that are standard in the NLP and computational linguistics literature.

---

## Reviewer Requirements
Based on feedback from Dr. Nathan and prior reviewer comments, Phase 1.1 must demonstrate that we:

- Identified standard benchmark datasets (RCV1-v2 and arXiv)
- Explained why these datasets qualify as “hard benchmarks”
- Distinguished between primary benchmarks and supporting datasets
- Explicitly checked:
  - Whether the dataset has a hierarchy (multiple levels)
  - Whether the dataset is multi-label
  - Whether the dataset is widely used in prior work
- Addressed concerns regarding the use of synthetic data

Datasets that satisfy these criteria are treated as **hard benchmarks**. Datasets that do not are used only for support or robustness checks.

---

## Primary Hard Benchmarks

### RCV1-v2 (Reuters Corpus Volume 1, Version 2)

RCV1-v2 is a widely used Reuters news dataset designed for hierarchical, multi-label text classification.

**Why it qualifies as a hard benchmark:**
- **Hierarchy:** Yes — topics are organized across multiple levels
- **Multi-label:** Yes — documents may belong to multiple categories
- **Benchmark status:** Yes — extensively used in prior research

The authoritative definition of the dataset, including hierarchy and labeling, is provided in:

> Lewis et al. (2004), *Journal of Machine Learning Research*  
> https://jmlr.csail.mit.edu/papers/volume5/lewis04a/lewis04a.pdf

The official dataset distribution is maintained by the Linguistic Data Consortium (LDC):
> https://catalog.ldc.upenn.edu/LDC2005T04

Access typically requires institutional credentials, which is expected and standard for this benchmark.

#### Alternative Access
A Harvard Dataverse mirror containing both RCV1 and RCV2 is available at:
> https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IEJ2UX

This source is **not required for Phase 1 or Phase 1.1**, but is retained for potential use in later phases.

Based on its structure and widespread use, RCV1-v2 is treated as a **primary hard benchmark**.

---

### arXiv Dataset (Metadata + Subject Taxonomy)

The arXiv benchmark is constructed using two complementary resources that serve different roles.

#### arXiv Metadata and Abstracts
> https://www.kaggle.com/datasets/Cornell-University/arxiv

This dataset contains:
- Paper titles
- Abstracts (text used for analysis)
- Paper category labels

**Used for:**
- Text embeddings
- Dimensionality reduction (PHATE, UMAP, PCA)
- Clustering

#### Official arXiv Subject Taxonomy
> https://arxiv.org/category_taxonomy

This is **not a text dataset**, but a reference hierarchy defining:
- Category → subcategory → sub-subcategory

**Used for:**
- Ground-truth hierarchical structure
- Evaluation of clustering results

Although individual arXiv papers typically have a single primary category, the taxonomy itself provides a clear multi-level hierarchy suitable for benchmarking. We focus on specific, concrete subcategories (e.g., cs.AI, cs.LG, physics subfields) rather than abstract themes, aligning with reviewer requests for measurable structure.

Together, the metadata and taxonomy form a **primary hard benchmark**.

---

## Supporting Datasets

### RCV1/RCV2 Multilingual Dataset (UCI)
> https://archive.ics.uci.edu/dataset/259/reuters+rcv1+rcv2+multilingual+multiview+text+categorization+test+collection

This dataset can be used in Phase 1 but **is not a main dataset for Phase 1.1** because:
- It does not provide a hierarchical structure
- It is not multi-label in the required sense

### PHATE-for-Text Repository Datasets
> https://github.com/sdork/phate-for-text/tree/main/data

Datasets such as Amazon, DBpedia, Web of Science, and EPA data are retained for:
- Validation
- Comparison
- Robustness checks

However, they do not fully meet reviewer criteria for hard benchmarks and are therefore not used as primary datasets in Phase 1.1.

---

## Synthetic Data Concerns
Earlier versions of this project relied on synthetic datasets for exploratory analysis. While useful initially, reviewers noted that synthetic data does not adequately capture the complexity of real-world text.

By replacing synthetic datasets with established benchmarks such as **RCV1-v2** and **arXiv**, Phase 1.1 ensures that evaluation is:
- Realistic
- Reproducible
- Aligned with community standards

---



## Summary
Phase 1.1 establishes RCV1-v2 and arXiv as the two primary hard benchmarks for this project. This satisfies reviewer expectations, clarifies dataset roles, and provides a strong foundation for subsequent experimental phases.
