#!/bin/bash
#SBATCH --time=2-23:00
#SBATCH --mem=128000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=avinashraj316@gmail.com


cd /home/s3754715/gnn_molecule/pytorch_geometric/examples/
python omdb_nn_conv.py > out.txt
