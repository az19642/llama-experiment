#!/bin/bash
#SBATCH --account=rrg-mmehride
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=30G
#SBATCH --time=0-20:00:00
#SBATCH --output=train_layer_%N-%j.out

module load python/3.10.13 gcc/12.3 arrow/18.1.0

virtualenv --no-download $SLURM_TMPDIR/venv
source $SLURM_TMPDIR/venv/bin/activate

pip install --no-index torch transformers datasets accelerate tensorboard sentencepiece protobuf zstandard

export HF_HUB_OFFLINE=1

python inference/run_inference.py \
    --base-model models/Llama-2-7b-hf \
    --custom-model models/llama_layer_1_best \
    --prompt "The fox" \
    --max-tokens 100 \