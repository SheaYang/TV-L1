#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=30:00:00
#SBATCH --output=/scratch/sy1823/HPC/final_project/mydeblur_GPU/BH_0.5/output.txt
#SBATCH --error=/scratch/sy1823/HPC/final_project/mydeblur_GPU/BH_0.5/error.txt

module load slurm
./deblur -f BH_gauss.png
