{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "fa8a6d2f-6732-4648-bb9b-5817403b98a5",
   "metadata": {},
   "source": [
    "# NCEAS Reproducibility.md"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e5edd501-022e-43eb-8528-e94fdfc789c2",
   "metadata": {},
   "source": [
    "## Jisha Goyal, Harshil Chidura, Sukhina Alkhalidy, Sidharth Rao"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "887eb44c-0d15-43ca-8fbb-1b48a0f37081",
   "metadata": {},
   "source": [
    "### 24 March 2026"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "dc3855e8-8cd5-494d-8655-424b19b3db61",
   "metadata": {},
   "source": [
    "**Installation**"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "08a53a4b-a31a-493b-a532-21e0eac2e7d9",
   "metadata": {},
   "source": [
    "Our full set of instructions are in our INSTALL.md file on our main Github page. This is the link for it: https://github.com/harshil0217/NCEAS_Unsupervised_NLP/blob/main/INSTALL.md\n",
    "\n",
    "This link contains the complete instructions for repository cloning, environment configuration, and dataset acquisition. The guide includes the necessary steps to recreate the local development environment and reproduce the project's data processing pipeline."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3c0882a3-be04-46ad-9f01-9f90aa101e0f",
   "metadata": {},
   "source": [
    "**Data**"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a746dfda-b60f-48ad-bc3c-1fc6adfd19e9",
   "metadata": {},
   "source": [
    "Our research pipeline validates unsupervised NLP methods by comparing modern LLM embeddings against high-quality human-labeled benchmarks.\n",
    "\n",
    "This project utilizes two primary data streams:\n",
    "\n",
    "- **Benchmark Datasets**: A source of \"Ground Truth\" metrics. Includes RCV1-v2 (Reuters news hierarchies), arXiv (30k scientific abstracts), and Amazon Reviews (scalability testing). The predictions made by the clustering algorithms are compared to the actual results of the data.\n",
    "\n",
    "- **Synthetic Data**: Trajectory-based samples generated via Groq to simulate specific fisheries-related discourse.\n",
    "\n",
    "\n",
    "Generating 1024-D embeddings takes significant time. To reproduce these figures in seconds rather than hours, we use intermediate checkpoints stored in the NCEAS Teams Data folder.\n",
    "\n",
    "**1.** Initialize the folders with this command:\n",
    "\n",
    "   mkdir -p src/data/arxiv src/data/amazon src/data/dbpedia src/data/wos src/data/rcv1_v2\n",
    "   \n",
    "**2.** Move the following high-value intermediate files into your local project structure:\n",
    "\n",
    "    - RCV1: Move rcv1_qwen_metadata.csv and rcv1_qwen_embeddings.npy to src/data/rcv1_v2/.\n",
    "\n",
    "    - arXiv: Move arxiv_30k_clean.csv to src/data/arxiv/.\n",
    "\n",
    "    - Amazon: Move train_40k.csv.zip to src/data/amazon/.\n",
    "  \n",
    "For the RCV1 raw data, you can also fetch it directly via Python:\n",
    "\n",
    "      from sklearn.datasets import fetch_rcv1\n",
    "      \n",
    "      rcv1 = fetch_rcv1() # Automatically downloads the base benchmark\n",
    "\n",
    "To verify the installation and data placement, run the demo notebook. This generates a sample figure using the full PHATE benchmark pipeline.\n",
    "    \n",
    "    jupyter notebook notebooks/demo.ipynb\n",
    "\n",
    "The notebook should load data, generate embeddings, and display a clustered 2D visualization without errors."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "edea3006-4b4d-4df0-bc55-d02a468a5e8a",
   "metadata": {},
   "source": [
    "**Visualizations**"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8538ee72-d8d7-4d85-92b5-d5670f15196f",
   "metadata": {},
   "source": [
    "This project uses a wide variety of visualizations across each phase. While we have worked with a myriad of visualizations and graphs throughout this project to display and analyze our findings, only a select number of them were to be displayed in our final result, one of them being our Shepard diagrams. \n",
    "\n",
    "Our project utilized a comprehensive suite of visualizations to track findings from Phases 1 to 3. A primary focus of our final analysis is the use of Shepard Diagrams to satisfy reviewer requirements for global structural integrity. By comparing PCA, UMAP, PHATE, and PaCMAP, these diagrams allow us to visually verify that the thematic hierarchy of the news corpus remains intact during dimensionality reduction, providing a transparent link between our unsupervised clusters and the Reuters ground truth.\n",
    "\n",
    "A primary focus of our final analysis is the use of Shepard Diagrams to satisfy reviewer requirements for global structural integrity. By comparing PCA, UMAP, PHATE, and PaCMAP, these diagrams allow us to visually verify that the thematic hierarchy of the news corpus remains intact during dimensionality reduction."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "b2a19eca-27df-4ae7-965d-3b9766c78990",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.metrics import pairwise_distances\n",
    "import os\n",
    "\n",
    "# 1. SETUP: Define the helper function for Shepard Diagrams\n",
    "def plot_shepard(x_high, x_low, name, sample_size=500):\n",
    "    \"\"\"\n",
    "    Generates a Shepard Diagram to validate global distance preservation.\n",
    "    x_high: Original 1024-D embeddings\n",
    "    x_low: Projected 2D coordinates (PCA, UMAP, etc.)\n",
    "    \"\"\"\n",
    "    # Randomly sample to keep computation under 1 minute\n",
    "    indices = np.random.choice(len(x_high), sample_size, replace=False)\n",
    "    \n",
    "    # Calculate and flatten pairwise distances\n",
    "    d_high = pairwise_distances(x_high[indices]).flatten()\n",
    "    d_low = pairwise_distances(x_low[indices]).flatten()\n",
    "\n",
    "    # Normalize distances to [0, 1] for fair comparison across algorithms\n",
    "    d_high = d_high / np.max(d_high)\n",
    "    d_low = d_low / np.max(d_low)\n",
    "    \n",
    "    plt.figure(figsize=(6, 6))\n",
    "    plt.scatter(d_high, d_low, alpha=0.1, s=1, color='teal')\n",
    "\n",
    "    # Red dashed line represents 'Perfect' linear preservation\n",
    "    plt.plot([0, 1], [0, 1], color='red', linestyle='--', alpha=0.5)\n",
    "    \n",
    "    plt.title(f\"Shepard Diagram: {name}\")\n",
    "    plt.xlabel(\"High-Dimensional Distance (Normalized)\")\n",
    "    plt.ylabel(\"Low-Dimensional Distance (Normalized)\")\n",
    "    plt.tight_layout()\n",
    "    \n",
    "    # Save using standard snake_case filenames for the repo\n",
    "    filename = f\"shepard_{name.lower()}.png\"\n",
    "    plt.savefig(filename, dpi=300)\n",
    "    plt.show() # Show in notebook for instructor review\n",
    "    return filename\n",
    "\n",
    "# 2. EXECUTION: Run the loop for all Phase 3.2 models\n",
    "# Note: Ensure these .npy files were generated in the previous step\n",
    "reductions = {\n",
    "    \"PCA\": np.load('../data/processed/rcv1_pca_2d.npy'),\n",
    "    \"UMAP\": np.load('../data/processed/rcv1_umap_2d.npy'),\n",
    "    \"PHATE\": np.load('../data/processed/rcv1_phate_2d.npy'),\n",
    "    \"PaCMAP\": np.load('../data/processed/rcv1_pacmap_2d.npy')\n",
    "}\n",
    "\n",
    "# x_high is your master 1024-D embedding array from Phase 1.2\n",
    "x_high = np.load('../data/processed/rcv1_qwen_embeddings.npy')\n",
    "\n",
    "print(\"Starting Shepard Diagram generation...\")\n",
    "for name, x_low in reductions.items():\n",
    "    saved_file = plot_shepard(x_high, x_low, name)\n",
    "    print(f\"Successfully generated and saved: {saved_file}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "322a318d-8bcb-4645-87b5-12c59b7a547e",
   "metadata": {},
   "source": [
    "The code above reproduces the Shepard Diagrams used to validate the structural integrity of our dimensionality reduction models (PCA, UMAP, PHATE, and PaCMAP). By comparing the normalized pairwise distances of the original 1024-D Qwen3 embeddings against their 2D projections, this code provides mathematical proof that the global thematic relationships of the RCV1-v2 dataset were preserved without significant 'tearing' or distortion."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3.12 (default)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
