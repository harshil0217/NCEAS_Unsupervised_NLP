#!/bin/bash
#SBATCH --job-name=hdbscan_eval
#SBATCH --partition=general-l
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/hdb_%j.out
#SBATCH --error=logs/hdb_%j.err

mkdir -p logs

echo "======================================"
echo "Job started on $(hostname)"
echo "Time: $(date)"
echo "======================================"

cd /mnt/home/alkhal13/phase3 || exit

source ~/qwen_env/bin/activate

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2

echo "Running HDBSCAN..."

python phase3_HDBSCAN.py

echo "DONE"