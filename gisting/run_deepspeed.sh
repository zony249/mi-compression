#!/usr/bin/env bash
#SBATCH --job-name=gist
#SBATCH --ntasks=1
#SBATCH --mem=480gb
#SBATCH --time=3-00:00
#SBATCH --output=gist.log
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4

# This script can either be used interactively or submitted to SLURM with
# sbatch.

# NOTE: LLaMA runs typically take ~7 hours; FLAN-T5-XXL runs typically take ~26
# hours. You can probably get away with training FLAN-T5-XXL less.

# export NCCL_IB_DISABLE=1
# export NCCL_SOCKET_IFNAME=br01 # for me it is 'br0' interface, you should use yours :)
# export NCCL_P2P_DISABLE=1
# export NCCL_DEBUG=INFO

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=5,6

TAG="llama-1tok"

port=$(shuf -i25000-30000 -n1)

deepspeed --master_port $port --num_gpus=2 --no_local_rank \
    --module src.train \
    +model=llama-7b wandb.tag=$TAG \
    training.deepspeed=ds_configs/stage3.json \
    training.gist.condition=gist \
    training.gist.num_gist_tokens=1
