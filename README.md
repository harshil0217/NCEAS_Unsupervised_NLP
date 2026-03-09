# NCEAS_Unsupervised_NLP
NCEAS Project for SS26 Data Science Capstone

## Class Description
The CMSE 495 data science capstone course is intended to provide students with an opportunity to put together what they have learned across multiple courses to develop a final project that demonstrates their ability to work in a team on real-world problems.

## Objective
The objective of the NCEAS_Unsupervised_NLP project is to support the development of improved methods for unsupervised analysis of unstructured textual data in the context of discourse and narratives surrounding US fisheries, particularly the ecological and social dynamics concerning commercial fishing and government regulation.


## Installation

For full installation instructions and environment setup, see:

[INSTALL.md](INSTALL.md)


## Key Components
### Text Scraping and Data Collection
We will develop a Python API to scrape text from audio transcriptions of fisheries podcasts and related unstructured text from public forums related to US fisheries. The collected data will then be stored in a structured database for analysis.

### NLP and Data Modeling
The project will employ natural language processing (NLP) techniques to analyze the textual data. Advancements in textual embeddings enabled by large language models (LLMs) will be explored

### Exploratory Data Analysis (EDA) & Visualization
Visualizations will play a crucial role in representing the analysis and communicating results to fisheries researchers and expert NLP practitioners

### Analysis and Quantification
The team will apply a few standard and emerging methods for dimensionality reduction and hierarchical clustering to textual embeddings with the goal of advancing the state-of-the-art. The results will be compared through robust statistical analysis. Metrics will include the adjusted Rand index (ARI), mutual information (MI), the Fowlkes–Mallows index (FM), and others as identified by the team.

## Demo
After installing the environment, run:
jupyter notebook

Then open:

notebooks/demo.ipynb

This notebook demonstrates the full PHATE benchmark pipeline including:

- loading example data
- generating embeddings
- dimensionality reduction
- clustering
- visualization

Project Structure
NCEAS_Unsupervised_NLP/
│
├── notebooks/               # Demo and exploratory notebooks
│   └── demo.ipynb
│
├── src/
│   ├── data/                # Benchmark datasets (not included in repo)
│   │   ├── arxiv/
│   │   ├── amazon/
│   │   ├── dbpedia/
│   │   └── wos/
│   │
│   ├── run_models/          # Benchmark experiment scripts
│   │   ├── arxiv_benchmark.py
│   │   ├── amazon_benchmark.py
│   │   └── ...
│   │
│   └── evaluations/         # Evaluation notebooks and analysis
│
├── environment.yml          # Reproducible conda environment
├── INSTALL.md               # Installation instructions
└── README.md                # Project overview

## Authors

[Jisha Goyal](https://github.com/goyaljis)

[Sidharth Rao](https://github.com/CharlieMalick)

[Sukina Alkhalidy](https://github.com/sukaina13)

[Harshil Chidura](https://github.com/harshil0217)
