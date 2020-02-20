
import os
import pandas as pd
import spk_ombd_parser_test as arg_parser
import numpy as np
import shutil
from schnetpack import AtomsData
from ase.io import read


def readTestData(args):
	#read existing omdb predictions csv
	old_omdb_pred = pd.read_csv(os.path.join(args.test_path,'COD_OMDB-GAP1_predictions.csv.gz'),  compression='gzip',error_bad_lines=False)
	old_omdb_file_list = old_omdb_pred['cod_id'].to_numpy('str')

	mol_list = []
	property_list = []
	#needed to link the predictions to the data id
	cod_list = []
	for subdir, dirs, files in os.walk(args.test_path):
	    for file in files:
	    	if file.endswith('.cif'):
	    		fname_without_ext = os.path.splitext(file)[0]
	    		if fname_without_ext in old_omdb_file_list:
	    			file_path = os.path.join(subdir, file)
	    			atoms = read(file_path, index=':')
	    			#putting dummy energy value as data will be only used for predictions
	    			property_list.append({'band_gap': np.array([-97208.40600498248], dtype=np.float32)})
	    			mol_list.extend(atoms)
	    			#cod ids
	    			cod_list.append(fname_without_ext)

	
	print(' Number of records ',str(len(cod_list)))
	np.save('./cod_id_list_old.npy', cod_list)
	return mol_list, property_list

def main(args):

	mol_list, output_list = readTestData(args)
	if os.path.exists("./cod_predict.db"):
  		os.remove("./cod_predict.db")
	new_dataset = AtomsData('./cod_predict.db', available_properties=['band_gap'])
	print('Number of test instances '+str(len(output_list)))
	new_dataset.add_systems(mol_list, output_list)


if __name__ == "__main__":

	parser = arg_parser.build_parser()
	(options, args) = parser.parse_args()
	main(options)