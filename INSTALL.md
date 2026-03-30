# Installation Instructions
NCEAS Unsupervised NLP – PHATE Benchmark Pipeline

These instructions allow instructors, classmates, and community partners to reproduce the project environment and run the benchmark pipeline on a Linux system with CUDA support.

**Note:** This project requires a Linux system with a CUDA-compatible GPU. All experiments are designed to run on an HPC cluster (e.g., MSU HPCC). The pipeline uses GPU-accelerated libraries (cuML, cuPCA, cuUMAP) that are not available on macOS or Windows.

---

## 1. Clone the Repository

```bash
git clone https://github.com/harshil0217/NCEAS_Unsupervised_NLP.git
cd NCEAS_Unsupervised_NLP
```

---

## 2. Install Conda (If Not Installed)

Download and install **Miniconda** for Linux:

https://docs.conda.io/en/latest/miniconda.html

Follow the default installation instructions and open a new terminal after installation.

---

## 3. Create the Project Environment

From the root project directory:

```bash
conda env create -f environment.yml
conda activate phate-env
```

This creates a fully reproducible environment using the provided `environment.yml` file, including all CUDA-accelerated packages.

---

## 4. Data Setup

This repository does **not include benchmark datasets**. Datasets must be downloaded separately and placed in the correct folders.

### Required Folder Structure

```bash
src/data/
├── arxiv/
│   └── arxiv_clean.csv
├── amazon/
│   ├── train_40k.csv
│   └── val_10k.csv
├── dbpedia/
│   └── DBPEDIA_test.csv
├── rcv1/
│   └── rcv1.csv
└── WebOfScience/
    └── Data.xlsx
```

---

### arXiv Dataset

The arXiv dataset used in this project is a **30,000 paper subset** of the full arXiv metadata.

**Option 1: Use the preprocessed dataset (recommended)**

1. Download `arxiv_30k_clean.csv` from the NCEAS Teams Data folder:
   ```
   Documents/NCEAS/Team_Management_Files/Data/arxiv/arxiv_30k_clean.csv
   ```

2. Place it at:
   ```bash
   src/data/arxiv/arxiv_30k_clean.csv
   ```

3. Run the cleaning notebook to generate `arxiv_clean.csv`:
   ```
   src/data/arxiv/clean_arxiv.ipynb
   ```
   This will save the cleaned file to `src/data/arxiv/arxiv_clean.csv`, which is what the pipeline loads.

**Option 2: Download from source**

1. Download the full arXiv metadata from Kaggle:
   https://www.kaggle.com/datasets/Cornell-University/arxiv

2. Subset to 30,000 papers and save as:
   ```bash
   src/data/arxiv/arxiv_30k_clean.csv
   ```

3. Run the cleaning notebook to generate `arxiv_clean.csv`:
   ```
   src/data/arxiv/clean_arxiv.ipynb
   ```

---

### Amazon Dataset

**Option 1: Use the preprocessed dataset (recommended)**

Available in the NCEAS Teams Data folder:

```
Documents/NCEAS/Team_Management_Files/Data/Amazon/
├── train_40k.csv.zip
└── val_10k.csv.zip
```

Download and unzip both files, then place the extracted CSVs at:

```bash
src/data/amazon/train_40k.csv
src/data/amazon/val_10k.csv
```

**Option 2: Download from source**

https://nijianmo.github.io/amazon/index.html

---

### DBpedia Dataset

**Option 1: Use the preprocessed dataset (recommended)**

Available in the NCEAS Teams Data folder:

```
Documents/NCEAS/Team_Management_Files/Data/DBpedia/DBPEDIA_test.csv
```

Place it at:

```bash
src/data/dbpedia/DBPEDIA_test.csv
```

**Option 2: Download from source**

https://github.com/le-scientifique/torchDatasets/tree/master/dbpedia_csv

---

### RCV1 Dataset

**Option 1: Use the preprocessed dataset (recommended)**

Available in the NCEAS Teams Data folder:

```
Documents/NCEAS/Team_Management_Files/Data/rcv1_v2/rcv1.csv
```

Place it at:

```bash
src/data/rcv1/rcv1.csv
```

**Option 2: Download from source**

https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_rcv1.html

---

### Web of Science Dataset

**Option 1: Use the preprocessed dataset (recommended)**

Available in the NCEAS Teams Data folder:

```
Documents/NCEAS/Team_Management_Files/Data/WebOfScience/Data.xlsx
```

Place it at:

```bash
src/data/WebOfScience/Data.xlsx
```

**Option 2: Download from source**

https://github.com/kk7nc/Text_Classification

---

## 5. Run the Demo

Start Jupyter and open the demo notebook:

```bash
jupyter notebook
```

Open:

```
notebooks/demo.ipynb
```

Run all cells from top to bottom. This notebook verifies that the full pipeline (embeddings, dimensionality reduction, clustering, and visualization) is working correctly.

---

## 6. Run Benchmark Experiments

Once datasets are in place, run the evaluation pipeline for any supported dataset:

```bash
python src/run_models/benchmark_datasets/eval_pipeline.py --dataset arxiv
python src/run_models/benchmark_datasets/eval_pipeline.py --dataset amazon
python src/run_models/benchmark_datasets/eval_pipeline.py --dataset dbpedia
python src/run_models/benchmark_datasets/eval_pipeline.py --dataset rcv1
python src/run_models/benchmark_datasets/eval_pipeline.py --dataset wos
```

Results are saved to:

```bash
results/{dataset}_clustering_scores.csv
```

---

## 7. Verify Installation

Installation is successful if:

- The demo notebook runs without errors
- A clustering visualization is generated
- No missing package errors occur

---

## 8. Troubleshooting

If environment creation fails:

```bash
conda deactivate
conda remove -n phate-env --all
conda env create -f environment.yml
```

If issues persist, ensure Conda is installed correctly and that you are running commands from the project root directory.

---

All experiments in this project follow the Safe, Portable, Reproducible, and Robust software guidelines described in the CMSE capstone course.
