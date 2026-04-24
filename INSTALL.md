# Installation Instructions
NCEAS Unsupervised NLP 

These instructions allow instructors, classmates, and community partners to reproduce the project environment and run the benchmark pipeline on a Linux system with CUDA support.


## A few Important Notes
1. This project requires a Linux system with a CUDA-compatible GPU. All experiments are designed to run on an HPC cluster (e.g., MSU HPCC). The pipeline uses GPU-accelerated libraries (cuML, cuPCA, cuUMAP) that are not available on macOS or Windows. For reference, the code in this library was developed and executed with a configuration of 8 CPU cores, 64 GB of RAM, and 2 V100 GPUs.

2. Running the full pipeline for all real and synthetic data sources will take hours with most hardware configurations. If you simply wish to run our pipeline end to end, to ensure our code is reproducible, we recommend only running our pipelines for the **RCV1** dataset, which is the smallest of our data sources.

3.  A Paid Groq API key is required to use GPT-OSS-120B, the model used for synthetic data generation. It shouldn't cost you more than a few cents, however



---

## 1. HPCC Setup (MSU HPCC users)

If running on MSU HPCC, run these commands at the start of every new terminal session before anything else:

```bash
module purge
module load Miniforge3
conda activate phate-env
```

If you see import errors like `No module named regex` on a dev node, prefix your commands with `PYTHONPATH=""`:

```bash
PYTHONPATH="" python src/run_models/benchmark_datasets/eval_pipeline.py --dataset rcv1
```

---

## 2. Clone the Repository

```bash
git clone https://github.com/harshil0217/NCEAS_Unsupervised_NLP.git
cd NCEAS_Unsupervised_NLP
```

---

## 3. Create the Project Environment with Conda

Skip this step if you are on MSU HPCC and already ran `conda activate phate-env` above.

From the root project directory:

```bash
conda env create -f environment.yml
conda activate phate-env
```

This creates a fully reproducible environment using the provided `environment.yml` file, including all CUDA-accelerated packages.

---

## 4. Set Up API Keys

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

## 5. Data Setup

This repository does **not include benchmark datasets**. Use the provided download script to fetch all datasets automatically.

### Install Kaggle CLI

Install Kaggle CLI tool with 

`pip install kaggle`

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




**Path convention note:** All scripts must be run from the **repo root** (not from inside `src/`). The scripts internally `cd` into `src/` at startup and use `../` to reference data and results. Running from the wrong directory will cause file-not-found errors.

Results are saved to `results/clustering/benchmark/`.

---

## 6. Quickstart (end-to-end on RCV1)

RCV1 is the smallest dataset and recommended for a quick end-to-end test:

```bash
# HPCC only - run at the start of every session
module purge
module load Miniforge3
conda activate phate-env

# download data (only needed once)
python data/download_data.py --datasets rcv1

# run the pipeline
python src/run_models/benchmark_datasets/eval_pipeline.py --dataset rcv1
```

Results will be saved to `results/clustering/benchmark/rcv1_clustering_scores.csv`.

---

## 7. Synthetic Data

To generate synthetic data run 

`python data/synthetic/generate.py`

**Note**: Ensure that your Groq API key is configured properly in your `.env` file.

The synthetic datasets will be saved to `data/synthetic/generated_data/`


