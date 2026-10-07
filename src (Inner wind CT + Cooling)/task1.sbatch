#! /bin/bash
#SBATCH --job-name="Hesperus_task"
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --time=0-02:00
#SBATCH --mail-user=korolkov.msu@mail.ru
#SBATCH --mail-type=ALL
#SBATCH --partition=rocky

# Executable
srun ./program