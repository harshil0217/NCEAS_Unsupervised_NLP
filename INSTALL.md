# Installation Instructions  
NCEAS Unsupervised NLP – PHATE Benchmark Pipeline

These instructions allow instructors, classmates, and community partners to reproduce the project environment and run a working demo.

---

## 1. Clone the Repository

```bash
git clone https://github.com/harshil0217/NCEAS_Unsupervised_NLP.git
cd NCEAS_Unsupervised_NLP
```

---

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

This repository does **NOT** include benchmark datasets or synthetic data. 

Links to each data source can be found here 

### Benchmark Dataset Sources

The benchmark datasets used in this project are publicly available:

- arXiv metadata dataset:  
https://www.kaggle.com/datasets/Cornell-University/arxiv


- RCV1 dataset (scikit-learn loader):  
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_rcv1.html


- Amazon Product Reviews dataset:  
https://nijianmo.github.io/amazon/index.html

- DBpedia dataset:  
https://github.com/le-scientifique/torchDatasets/tree/master/dbpedia_csv

- Web of Science dataset:  
https://github.com/kk7nc/Text_Classification

- EPA Public Comments dataset (Mirrulations AWS mirror):  
https://registry.opendata.aws/mirrulations/

Download the datasets and place them inside `src/data/` within their respective folder

## 5. Run the Demo

Start Jupyter:

```bash
jupyter notebook
```

Open:

```
notebooks/demo.ipynb
```

Run all cells from top to bottom.

The demo notebook will:

- Load example data  
- Generate embeddings  
- Perform dimensionality reduction (PHATE, PCA, UMAP)  
- Apply clustering  
- Display a visualization figure  

---

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


