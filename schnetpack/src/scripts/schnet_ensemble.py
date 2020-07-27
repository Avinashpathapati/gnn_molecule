import os
import torch
import schnetpack as spk
import logging
import pandas as pd
from schnetpack.datasets import OrganicMaterialsDatabase
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt
import shutil
from ase.units import kcal, mol
from ase.io import read
import schnetpack.train as trn
import spk_ombd_parser as arg_parser
import torch.nn as nn
from schnetpack import AtomsData
from schnetpack.utils.script_utils.settings import get_environment_provider
from torch.utils.data.sampler import RandomSampler
from schnetpack.utils import (
    get_divide_by_atoms,
    get_statistics,
    get_output_module
)

sch_model_1 = torch.load(os.path.join('/home/s3754715/gnn_molecule/schnetpack/model_2020-06-23-18-44-59', 'best_model'))
sch_model_2 = torch.load(os.path.join('/home/s3754715/gnn_molecule/schnetpack/model_2020-06-23-18-44-04', 'best_model'))
sch_model_3 = torch.load(os.path.join('/home/s3754715/gnn_molecule/schnetpack/model_2020-06-23-18-44-00', 'best_model'))
sch_model_4 = torch.load(os.path.join('/home/s3754715/gnn_molecule/schnetpack/model_2020-06-23-18-41-59', 'best_model'))
sch_model_5 = torch.load(os.path.join('/home/s3754715/gnn_molecule/schnetpack/model_2020-06-15-04-47-32', 'best_model'))
omdData = OrganicMaterialsDatabase(args.datapath, download=False, load_only=[args.property], environment_provider=environment_provider)
split_path = os.path.join(args.model_path, "split.npz")
train, val, test = spk.train_test_split(
	data=omdData,
	num_train=9000,
	num_val=1000,
	split_file=split_path
)
print('-----------')
print(len(train))
print(len(val))
print(len(test))
print('-------------')
train_loader = spk.AtomsLoader(train, batch_size=16, sampler=RandomSampler(train), num_workers=4 
	#pin_memory=True
	)
val_loader = spk.AtomsLoader(val, batch_size=16, num_workers=2
	)
test_loader = spk.AtomsLoader(test, batch_size=16, num_workers=2
	)

for count, batch in enumerate(test_loader):
	    # move batch to GPU, if necessary
	    batch = {k: v.to(device) for k, v in batch.items()}

	    # apply model
	    pred_1 = sch_model_1(batch)
	    pred_2 = sch_model_2(batch)
	    pred_3 = sch_model_3(batch)
	    pred_4 = sch_model_4(batch)
	    pred_5 = sch_model_5(batch)

	    # calculate absolute error
	    tmp = torch.sum(torch.abs(torch.mean(torch.stack([pred_1[args.property], pred_2[args.property], pred_3[args.property], pred_4[args.property], pred_5[args.property]],dim=1), dim=1)-batch[args.property]))
	    tmp = tmp.detach().cpu().numpy() # detach from graph & convert to numpy
	    err += tmp

	    # log progress
	    percent = '{:3.2f}'.format(count/len(test_loader)*100)
	    print('Progress:', percent+'%'+' '*(5-len(percent)), end="\r")

err /= len(test)
print('Test MAE', np.round(err, 3), 'eV =',
      np.round(err / (kcal/mol), 3), 'kcal/mol')