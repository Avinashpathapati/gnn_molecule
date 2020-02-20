#!/bin/bash
#SBATCH --time=2-23:00
#SBATCH --mem=128000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=avinashraj316@gmail.com


cd /home/s3754715/gnn/schnetpack/src/scripts/
python spk_omdb_run.py --cuda --mode train --datapath /home/s3754715/gnn/schnetpack/dataset/OMDB-GAP1_v1.1.db --model_path /home/s3754715/gnn/schnetpack/model --property band_gap --dataset omdb --n_epochs 300
