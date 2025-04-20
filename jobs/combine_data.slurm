#!/bin/bash
#SBATCH --account=def-mmehride_cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=0-10:00:00
#SBATCH --output=combine_data_%N-%j.out

module load python/3.10.13 gcc/12.3 arrow/18.1.0

virtualenv --no-download $SLURM_TMPDIR/venv
source $SLURM_TMPDIR/venv/bin/activate

pip install --no-index torch transformers datasets protobuf zstandard

export HF_HUB_OFFLINE=1

python data_generation/combine_data.py --datasets datasets/layer_1_chunked

