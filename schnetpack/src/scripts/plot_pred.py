
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

atom_id_input_arr = []
atom_output = []


def inputExtract(self, input, output):

	print(input[0].shape)
	for data in input[0]:
		for atom in data:
			print(atom)
			atom_id_input_arr.append(atom.numpy())



def outputExtract(self, input, output):
	
	print(input[0].shape)
	print('---------')
	print(output[0].shape)
	# print(output[10].shape)

	for atom_out in output[0].squeeze(1):
		print(atom_out)
		atom_output.append(atom_out)



def main(args):
	print('predictionsss')
	device = torch.device("cuda" if args.cuda else "cpu")
	environment_provider = spk.environment.AseEnvironmentProvider(cutoff=5.0)

	sch_model = torch.load(os.path.join(args.model_path, 'best_model'), map_location=torch.device(device))

	sch_model.representation.embedding.register_forward_hook(inputExtract)
	sch_model.output_modules[0].out_net[1].out_net[1].register_forward_hook(outputExtract)

	# for name, module in sch_model.named_modules():
	# 	print(name)

	#reading test data
	# test_dataset = AtomsData('./cod_predict.db')
	# test_loader = spk.AtomsLoader(test_dataset, batch_size=32)

	#reading stored cod list
	#cod_list = np.load('./cod_id_list_old.npy')
	omdData = OrganicMaterialsDatabase(args.datapath, download=False, load_only=[args.property], environment_provider=environment_provider)
	split_path = os.path.join(args.model_path, "split.npz")
	train, val, test = spk.train_test_split(
		data=omdData,
		num_train=9000,
		num_val=1000,
		split_file=split_path
	)
	print(test[0])
	print(test[1])
	test_loader = spk.AtomsLoader(test, batch_size=1, #num_workers=2
		)
	mean_abs_err = 0
	prediction_list = []
	actual_value_list = []

	print('Started generating predictions')
	for count, batch in enumerate(test_loader):
	    
	    # move batch to GPU, if necessary
	    print('before batch')
	    batch = {k: v.to(device) for k, v in batch.items()}
	    print('after batch')
	    # apply model
	    pred = sch_model(batch)
	    prediction_list.extend(pred['band_gap'].detach().cpu().numpy().flatten().tolist())
	    actual_value_list.extend(batch['band_gap'].detach().cpu().numpy().flatten().tolist())
	    # log progress
	    percent = '{:3.2f}'.format(count/len(test_loader)*100)
	    print('Progress:', percent+'%'+' '*(5-len(percent)), end="\r")



	cod_arr = np.genfromtxt(os.path.join('/home/s3754715/gnn_molecule/schnetpack/dataset/OMDB-GAP1_v1.1', 'CODids.csv'))
	cod_list = cod_arr[10000:].tolist()
	results_df = pd.DataFrame({'cod':cod_list, 'prediction':prediction_list, 'actual': actual_value_list})
	results_df.to_csv('./predictions.csv')



if __name__ == "__main__":

	parser = arg_parser.build_parser()
	(options, args) = parser.parse_args()
	main(options)