# Installation Instructions  
NCEAS Unsupervised NLP – PHATE Benchmark Pipeline

These instructions allow instructors, classmates, and community partners to reproduce the project environment and run a working demo.

---

## 1. Clone the Repository

```bash
git clone https://github.com/harshil0217/NCEAS_Unsupervised_NLP.git
cd NCEAS_Unsupervised_NLP
```
## 2. Install Conda (If Not Installed)

Download and install **Miniconda** for your operating system:

https://docs.conda.io/en/latest/miniconda.html

Follow the default installation instructions.

After installation, open a new terminal.

---

## 3. Create the Project Environment

From the root project directory:

```bash
conda env create --prefix ./envs --file environment.yml
conda activate ./envs
```

This creates a fully reproducible environment using the provided `environment.yml` file.

---
## 4. Data Instructions

This repository does **not include benchmark datasets**.  
Datasets must be downloaded separately and placed in the correct folders.

---

### Fastest Option — Use the NCEAS Teams Data Folder

Preprocessed datasets used in this project are available in the **NCEAS Teams Data folder**.

Location:

```bash
Documents  
└── NCEAS  
  └── Team_Management_Files  
    └── Data  
      ├── arxiv  
      ├── amazon  
      ├── dbpedia  
      ├── wos  
      └── rcv1_v2  
```

Example:
src/data/arxiv/
src/data/amazon/
src/data/dbpedia/
src/data/wos/

Download the datasets and place them inside `src/data/` within their respective folders.

---
The benchmark datasets used in this project are publicly available:
## arXiv Dataset

The original arXiv dataset contains over **1.7 million papers**, which is too large to include directly in this repository.  
For our experiments, we use a **30,000 paper subset** generated from the full dataset.

## Option 1 — Use the Preprocessed Dataset (Quick Setup from Teams)

For convenience, the processed dataset used in this project is available in the **NCEAS Teams Data folder**.

### Location

```bash
Documents
└── NCEAS
    └── Team_Management_Files
        └── Data
            └── arxiv
                └── arxiv_30k_clean.csv
```


Download the file and place it in:

src/data/arxiv/

Expected Folder Structure
```bash
src/
└── data/
    └── arxiv/
        └── arxiv_30k_clean.csv
```

Example:

```bash
Documents
└── NCEAS
    └── Team_Management_Files
        └── Data
            └── arxiv
                └── arxiv_30k_clean.csv
```

This file can be used directly by the benchmark scripts and evaluation notebooks.

## Option 2 — Recreate the Dataset (Full Reproducibility)

The dataset can also be recreated from the original arXiv metadata.

### Step 1 — Download the Dataset

Download the dataset from Kaggle:

https://www.kaggle.com/datasets/Cornell-University/arxiv

### Step 2 — Place the File in the Project Folder

After downloading, place the file in:

src/data/arxiv/arxiv-metadata-oai-snapshot.json

### Step 3 — Run the Preprocessing Notebook

Open and run the notebook:

src/data/arxiv/01_download_arxiv_dataset.ipynb

This notebook will:

- Stream the large JSON dataset
- Filter papers to Computer Science and Physics categories
- Combine the **title and abstract** text fields
- Randomly sample **30,000 papers**
- Save the processed dataset as:

src/data/arxiv/arxiv_30k_clean.csv
---

### RCV1 Dataset

- RCV1 dataset (scikit-learn loader):  
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_rcv1.html

The **RCV1 dataset** is automatically downloaded using the scikit-learn dataset loader when running the benchmark scripts.

```python
from sklearn.datasets import fetch_rcv1
rcv1 = fetch_rcv1()
```
The dataset initially loads as a CSR (sparse) matrix containing over 800,000 news documents and 103 topic categories.
The text is then embedded using the Qwen3-Embedding-0.6B model, converting each document into a 1024-dimensional semantic vector for clustering and evaluation.
These embeddings are later reduced using dimensionality reduction methods such as:
PCA (linear projection baseline)
PHATE (trajectory-based manifold mapping)
UMAP (topological manifold learning)
Clustering algorithms such as HDBSCAN and Agglomerative Clustering are applied to the reduced space, and performance is evaluated using the Adjusted Rand Index (ARI) against the ground-truth Reuters topic labels.


Location:

```bash
Documents
└── NCEAS
    └── Team_Management_Files
        └── Data
            ├── arxiv
            ├── amazon
            ├── dbpedia
            ├── wos
            └── rcv1_v2
```

### RCV1 Files in the Teams Folder

Additional documentation and processed files for the RCV1 dataset are available in the NCEAS Teams Data folder:

```bash
Documents
└── NCEAS
    └── Team_Management_Files
        └── Data
            └── rcv1_v2
                ├── Lewis_2004_JMLR_RCV1v2.pdf
                ├── rcv1_qwen_metadata.csv
                └── README.md
```
The rcv1_qwen_metadata.csv file contains the processed metadata and labels used in the evaluation pipeline.

Other Benchmark Dataset Sources:

- Amazon Product Reviews dataset:  
https://nijianmo.github.io/amazon/index.html

Amazon Dataset
Files available in the NCEAS Teams folder:
```bash
Documents
└── NCEAS
    └── Team_Management_Files
        └── Data
            └── Amazon
                ├── train_40k.csv.zip
                └── val_10k.csv.zip
```
Download these files and place them in:

src/data/amazon/

Example structure:
```bash
Documents
src/data/
└── amazon/
    ├── train_40k.csv.zip
    └── val_10k.csv.zip
```


- DBpedia dataset:  
https://github.com/le-scientifique/torchDatasets/tree/master/dbpedia_csv

Files available in the Teams folder:
```bash
Documents
└── NCEAS
    └── Team_Management_Files
        └── Data
            └── DBpedia
                └── DBPEDIA_test.csv
```
Place the file in:
src/data/dbpedia/

Example:
src/data/
```bash
Documents
└── dbpedia/
    └── DBPEDIA_test.csv
```

- Web of Science dataset:  
https://github.com/kk7nc/Text_Classification

Web of Science Dataset
Files available in the Teams folder:
```bash
Documents
└── NCEAS
    └── Team_Management_Files
        └── Data
            └── WebOfScience
                └── Data.xlsx
```
Place the file in:
src/data/wos/

Example:
```bash
src/data/
└── wos/
    └── Data.xlsx
```

- EPA Public Comments dataset (Mirrulations AWS mirror):  
https://registry.opendata.aws/mirrulations/

Download the datasets and place them inside `src/data/` within their respective folders

## 5. Run the Demo

The demo notebook will:

- Load example data  
- Generate embeddings  
- Perform dimensionality reduction (PHATE, PCA, UMAP)  
- Apply clustering  
- Display a visualization figure

This demo notebook is meant to ensure that all libraries needed for the full embedding -> dimensionality reduction -> hierarchical clustering pipeline have been installed and working as intended. 

Start Jupyter:

```bash
jupyter notebook
```

Open:

```
notebooks/demo.ipynb
```

Run all cells from top to bottom.
## 6. Verify Installation

The installation is successful if:

- The notebook runs without errors  
- A clustering visualization is generated  
- No missing package errors occur  

---

## 7. Troubleshooting

If environment creation fails:

```bash
conda deactivate
rm -rf ./envs
conda env create --prefix ./envs --file environment.yml
```

If issues persist, ensure Conda is installed correctly and that you are running the commands from the project root directory.

## Running Benchmark Experiments

In addition to the demo notebook, the repository contains scripts for running
benchmark experiments on real datasets.

Supported benchmark datasets include:

- arXiv
- Amazon Reviews
- DBPedia
- Web of Science
- RCV1

Dataset files should be placed inside:

src/data/

Example structure:

src/data/
    arxiv/
    amazon/
    dbpedia/
    wos/

Once the dataset is placed in the correct folder, the benchmark pipeline
can be executed using the scripts located in:

src/run_models/

Example command:

python src/run_models/arxiv_benchmark.py

All experiments in this project follow the Safe, Portable, Reproducible, and Robust software guidelines described in the CMSE capstone course.