#!/bin/bash

# Script to run all benchmark dataset experiments
# This script should be run from the benchmark_datasets directory
# Example: cd src/run_models/benchmark_datasets && bash run_benchmarks.sh

set -e  # Exit on error

echo "========================================="
echo "Running Benchmark Dataset Experiments"
echo "========================================="
echo "Repository root: $(pwd)"
echo ""

# Check if conda environment is activated
if [[ -z "$CONDA_DEFAULT_ENV" ]] && [[ -z "$VIRTUAL_ENV" ]]; then
    echo "WARNING: No conda/virtual environment detected."
    echo "Please activate the environment first:"
    echo "  conda activate ./envs"
    echo "  or: source .venv/bin/activate"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Run Amazon benchmark
echo "========================================="
echo "1/3: Running Amazon benchmark..."
echo "========================================="
python amazon.py
echo "✓ Amazon benchmark completed"
echo ""

# Run DBpedia benchmark
echo "========================================="
echo "2/3: Running DBpedia benchmark..."
echo "========================================="
python dbpedia.py
echo "✓ DBpedia benchmark completed"
echo ""

# Run Web of Science benchmark
echo "========================================="
echo "3/3: Running Web of Science benchmark..."
echo "========================================="
python WOS.py
echo "✓ Web of Science benchmark completed"
echo ""

echo "========================================="
echo "All benchmark experiments completed!"
echo "========================================="
