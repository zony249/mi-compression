#!/usr/bin/env bash
#SBATCH --job-name=gist
#SBATCH --ntasks=1
#SBATCH --mem=480gb
#SBATCH --time=3-00:00
#SBATCH --output=gist.log
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=h100:4
#SBATCH --account=rrg-lilimou

# This script can either be used interactively or submitted to SLURM with
# sbatch.

# NOTE: LLaMA runs typically take ~7 hours; FLAN-T5-XXL runs typically take ~26
# hours. You can probably get away with training FLAN-T5-XXL less.

# export NCCL_IB_DISABLE=1
# export NCCL_SOCKET_IFNAME=br01 # for me it is 'br0' interface, you should use yours :)
# export NCCL_P2P_DISABLE=1
# export NCCL_DEBUG=INFO

export HYDRA_FULL_ERROR=1

TAG="llama-1tok"

port=$(shuf -i25000-30000 -n1)

env/bin/python3.10 -m debugpy --listen 0.0.0.0:45678 -m accelerate.commands.launch \
    --num_processes 4 \
    --num_machines 1 \
    --main_process_port $port \
    --mixed_precision bf16 \
    --multi_gpu \
    --config_file ds_configs/ds_accel.yaml \
    -m src.train \
        +model=llama-7b wandb.tag=$TAG \
        training.gist.condition=gist \
        training.gist.num_gist_tokens=1
