# Installation Instructions  
NCEAS Unsupervised NLP – PHATE Benchmark Pipeline

---

## 1. Clone the Repository

```bash
git clone https://github.com/harshil0217/NCEAS_Unsupervised_NLP.git
cd NCEAS_Unsupervised_NLP
```

---

## 2. Install Conda (If Not Installed)

Download and install Miniconda:

https://docs.conda.io/en/latest/miniconda.html

Follow the default installation instructions for your operating system.

---

## 3. Create Project Environment

From the root project directory:

```bash
conda env create --prefix ./envs --file environment.yml
conda activate ./envs
```

---

## 4. Data Instructions

This repository does **NOT** include community partner data.

For testing:
- Use the example dataset located in the `data/` directory.
- All paths must remain relative (no absolute paths).

If adding new data, place it inside the `data/` folder.

---

## 5. Run the Demo

Start Jupyter:

```bash
jupyter notebook
```

Open:

```
notebooks/demo.ipynb
```

Run all cells.

If installation was successful, the notebook will:

- Load example data  
- Generate embeddings  
- Run dimensionality reduction (PHATE, PCA, UMAP)  
- Perform clustering  
- Display a visualization figure  

---

## 6. Troubleshooting

If environment creation fails:

```bash
conda deactivate
rm -rf ./envs
conda env create --prefix ./envs --file environment.yml
```