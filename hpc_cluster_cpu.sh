#!/bin/bash
#SBATCH --job-name=script_inference_test
#SBATCH --output=log.out
#SBATCH --error=log.err
#SBATCH --export=ALL
#SBATCH --time=00:20:00
#SBATCH --nodes=2
module load cuda/11.8
echo "Start at `date`"
echo "---"
source "/home/<BENUTZERNAME>/workspace/venv/bin/activate"
srun --nodes=2 python hpc_train_model.py
deactivate
echo "---"
echo "Done at `date`"