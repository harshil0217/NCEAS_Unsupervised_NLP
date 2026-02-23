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

This repository does **NOT** include community partner data due to privacy and NDA constraints.

For testing purposes:

- Example data is provided in the `data/` directory.
- All file paths must remain **relative**.
- If adding new datasets, place them inside the `data/` folder.

Do **not** use absolute paths.

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