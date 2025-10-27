#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a100
#SBATCH --job-name=s4744760_unet_training
#SBATCH --output slurm_sdout.txt
#SBATCH --error slurm_sderr.txt
#SBATCH --mail-type=end
#SBATCH --mail-user=s4744760.uq@gmail.com

~/python-venv/bin/python train.py
