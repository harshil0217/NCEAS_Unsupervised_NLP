# Installation Instructions
NCEAS Unsupervised NLP 

These instructions allow instructors, classmates, and community partners to reproduce the project environment and run the benchmark pipeline on a Linux system with CUDA support.


## A few Important Notes
1. This project requires a Linux system with a CUDA-compatible GPU. All experiments are designed to run on an HPC cluster (e.g., MSU HPCC). The pipeline uses GPU-accelerated libraries (cuML, cuPCA, cuUMAP) that are not available on macOS or Windows. For reference, the code in this library was developed and executed with a configuration of 8 CPU cores, 64 GB of RAM, and 2 V100 GPUs.

2. Running the full pipeline for all real and synthetic data sources will take hours with most hardware configurations. If you simply wish to run our pipeline end to end, to ensure our code is reproducible, we recommend only running our pipelines for the **RCV1** dataset, which is the smallest of our data sources.

---

## 1. Clone the Repository

```bash
git clone https://github.com/harshil0217/NCEAS_Unsupervised_NLP.git
cd NCEAS_Unsupervised_NLP
```

---

## 2. Create the Project Environment with Conda

From the root project directory:

```bash
conda env create -f environment.yml
conda activate phate-env
```

This creates a fully reproducible environment using the provided `environment.yml` file, including all CUDA-accelerated packages.

---

## 3. Set Up API Keys

Create a `.env` file in the project root with the following keys:

```bash
# Kaggle (required for arXiv, Amazon, DBpedia downloads)
# Get your token at https://www.kaggle.com/settings > API > Create New Token
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key

# Groq (required for synthetic data generation)
# Get your key at https://console.groq.com
GROQ_API_KEY=your_groq_api_key
```

---

## 4. Data Setup

This repository does **not include benchmark datasets**. Use the provided download script to fetch all datasets automatically.

### Download All Datasets

```bash
python data/download_data.py
```

This downloads and preprocesses all five benchmark datasets automatically:

| Dataset | Source | Notes |
|---------|--------|-------|
| arXiv | Kaggle | ~4 GB download, sampled to 30k papers |
| Amazon | Kaggle | `train_40k.csv` + `val_10k.csv` |
| DBpedia | Kaggle | `DBPEDIA_test.csv` |
| RCV1 | sklearn | No Kaggle needed |
| Web of Science | Mendeley | Downloaded automatically |

To download a specific dataset only:
```bash
python data/download_data.py --datasets rcv1
```

### Required Output Structure

```bash
data/
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




Results are saved to:

```bash
results/{dataset}_clustering_scores.csv
```

---

## 5. Synthetic Data

To generate synthetic data and perform the clustering evaluation pipeline run

`bash src/run_models/synthetic_data/run_all.sh`

Like for benchmark datasets, results will be saved to 

```bash
results/{dataset}_clustering_scores.csv
```

with the synthetic datasets saved to `data/synthetic/generated_data/`


