#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --partition=a100-test
#SBATCH --job-name=3D_unet
#SBATCH --time=00:20:00
#SBATCH --output slurm_sdout.txt
#SBATCH --error slurm_sderr.txt
#SBATCH --mail-type=END
#SBATCH --mail-user=s4744760.uq@gmail.com

~/python-venv/bin/python predict.py
