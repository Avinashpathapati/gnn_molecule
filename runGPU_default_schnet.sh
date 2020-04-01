#!/bin/bash
#SBATCH --time=2-23:00
#SBATCH --mem=128000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=avinashraj316@gmail.com


cd /home/s3754715/gnn_molecule/schnetpack/src/scripts/
python spk_run.py train schnet omdb /home/s3754715/gnn_molecule/schnetpack/dataset/OMDB-GAP1_v1.1.db /home/s3754715/gnn_molecule/schnetpack/model --split 9000 1000 --cuda > out.txt
