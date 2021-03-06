
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
from graphviz import Graph

atom_id_input_arr = []
atom_output_arr = []
cod = []
orange_indices_arr=[]
neighbour_list = []
formula_dict = {}

rec_ct = 0



def constGraph():

	for i in range(0,len(atom_id_input_arr)):

		atom_id_row = atom_id_input_arr[i]
		atom_output_row = atom_output_arr[i]
		orange_indices_row = orange_indices_arr[i]
		neighbour_row = neighbour_list[i]
		print(cod[i])
		g = Graph('G', filename= str(cod[i])+'.gv')
		for j in range(0,len(atom_id_row)):
			# print(j)
			if j in orange_indices_row:
				g.attr('node', style='filled', color='orange')
			else:
				g.attr('node', style='filled', color='white')
			
			g.node(str(j), label=atom_id_row[j])

		
		for j in range(0,len(atom_id_row)):
			# print(j)
			for k in range(0, len(neighbour_row[j])):
				if not neighbour_row[j][k] == -1:
					g.edge(str(j), str(neighbour_row[j][k]))
		
		# g.view()
		g.save(filename='./'+str(cod[i])+'.dot')


	


def outputExtract(self, input, output):
	
	
	# print(input[0].shape)
	# print('---------')
	# print(output[0].shape)
	# print(output[10].shape)]
	atom_output = []
	global rec_ct
	if rec_ct in [0,10,20,50]:
		# print(rec_ct)
		# print(output[0].shape)
		for atom_out in output[0].squeeze(1):
			atom_output.append(atom_out.detach().numpy())

		atom_output_np = np.array(atom_output)
		orange_indices_arr.append(np.where((atom_output_np >=np.mean(atom_output_np)-np.std(atom_output_np)) & 
			(atom_output_np<=np.mean(atom_output_np)+np.std(atom_output_np)))[0].tolist())
		
		# print(orange_indices_arr)
		atom_output_arr.append(atom_output)




	rec_ct += 1



def main(args):
	print('predictionsss')
	device = torch.device("cuda" if args.cuda else "cpu")
	environment_provider = spk.environment.AseEnvironmentProvider(cutoff=5.0)

	sch_model = torch.load(os.path.join(args.model_path, 'best_model'), map_location=torch.device(device))

	# sch_model.representation.embedding.register_forward_hook(inputExtract)
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
	#constructing chemical formula and COD array
	for rec_num in range(0,len(omdData)):
		chem_formula = omdData.get_atoms(idx=rec_num).get_chemical_formula()
		formula_dict[chem_formula]=rec_num

	cod_array = np.genfromtxt(os.path.join('/home/s3754715/gnn_molecule/schnetpack/dataset/OMDB-GAP1_v1.1', 'CODids.csv'), delimiter=',', dtype=(int))


	#to fetch the COD using chemical formula from dictionary at random ids to construct data to print in graph
	for id in [0,10,20,50]:
		atom_id_input_arr.append(test.get_atoms(idx=id).get_chemical_symbols())
		chem_formula = test.get_atoms(idx=id).get_chemical_formula()
		cod.append(cod_array[formula_dict[chem_formula]])
		print(test[id]['_neighbors'].numpy().shape)
		neighbour_list.append(test[id]['_neighbors'].numpy().tolist())

	
	test_loader = spk.AtomsLoader(test, batch_size=1, #num_workers=2
		)
	mean_abs_err = 0
	prediction_list = []
	actual_value_list = []
	print('Started generating predictions')
	#to stop pred after reaching max rec_ct and start constructing graph
	rec_id=0
	for count, batch in enumerate(test_loader):
	    
	    rec_id+=1
	    if(rec_id>51):
	    	break
	    # move batch to GPU, if necessary
	    batch = {k: v.to(device) for k, v in batch.items()}
	    # apply model
	    pred = sch_model(batch)
	    prediction_list.extend(pred['band_gap'].detach().cpu().numpy().flatten().tolist())
	    actual_value_list.extend(batch['band_gap'].detach().cpu().numpy().flatten().tolist())
	    # log progress
	    percent = '{:3.2f}'.format(count/len(test_loader)*100)
	    print('Progress:', percent+'%'+' '*(5-len(percent)), end="\r")



	constGraph()
	# cod_arr = np.genfromtxt(os.path.join('/home/s3754715/gnn_molecule/schnetpack/dataset/OMDB-GAP1_v1.1', 'CODids.csv'))
	# cod_list = cod_arr[10000:].tolist()
	# results_df = pd.DataFrame({'cod':cod_list, 'prediction':prediction_list, 'actual': actual_value_list})
	# results_df.to_csv('./predictions.csv')



if __name__ == "__main__":

	parser = arg_parser.build_parser()
	(options, args) = parser.parse_args()
	main(options)