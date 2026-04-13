# Installation Instructions
NCEAS Unsupervised NLP – PHATE Benchmark Pipeline

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

## 3. Data Setup

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

1. Download `arxiv-metadata-oai-snapshot.json` from [https://www.kaggle.com/datasets/Cornell-University/arxiv](https://www.kaggle.com/datasets/Cornell-University/arxiv)

2. Place it at:
   ```bash
   src/data/arxiv/arxiv-metadata-oai-snapshot.json
   ```

3. Run `clean_arxiv.py` to generate `arxiv_clean.csv`:
   ```
   src/data/arxiv/clean_arxiv.ipynb
   ```

---

### Amazon Dataset

1. Download the files `train_40k.csv` and `val_10k.csv` from the following link

https://www.kaggle.com/datasets/kashnitsky/hierarchical-text-classification

2. Place them both at:
 ```bash
    src/data/amazon/train_40k.csv
    src/data/amazon/val_10k.csv
 ```


---

### DBpedia Dataset


1. Download from `DBPEDIA_test.csv` from kaggle (existing preprocessed dataset incorporated for ease of use)

[https://github.com/le-scientifique/torchDatasets/tree/master/dbpedia_csv](https://www.kaggle.com/code/danofer/dbpedia-preprocessing/input)

2. Place at :
 ```bash
    src/data/dbpedia/DBPEDIA_test.csv
 ```

---

### RCV1 Dataset

1. Execute the python file located at `src/data/rcv1/import_rcv1.py`. This file will load in the RCV1 dataset using the [fetch_rcv1](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_rcv1.html) function, preprocess the data for use, and save the data to `rcv1.csv` 

---

### Web of Science Dataset

1. Download from [https://data.mendeley.com/datasets/9rw3vkcfy4/6](https://data.mendeley.com/datasets/9rw3vkcfy4/6_)
2. Place the `Meta-Data` folder, which contains a file named `Data.csv`, in `src/data/WebOfScience`

---



## 4. Run Clustering Evalution Pipelines

Once datasets are in place, run the clustering evaluation pipeline for any supported dataset:

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

## 5. Synthetic Data

To generate synthetic data and perform the clustering evaluation pipeline run

`bash src/run_models/synthetic_data/run_all.sh`

Like for benchmark datasets, results will be saved to 

```bash
results/{dataset}_clustering_scores.csv
```

with benchmark datasets saved to `src/data_generation/generated_data`


