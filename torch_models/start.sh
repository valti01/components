#!/bin/bash
#SBATCH -A vgila
#SBATCH --job-name=test
#SBATCH --nodelist=deep22
#SBATCH --output=output.log
#SBATCH --error=error.log

source /shared/anaconda/anaconda/bin/activate
conda activate /shared/anaconda/vgila/.conda/envs/env4

python run_eval.py