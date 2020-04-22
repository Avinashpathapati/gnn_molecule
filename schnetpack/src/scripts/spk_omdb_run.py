
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

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

def simple_loss_fn(args):
	def loss(batch, result):
		diff = batch[args.property]-result[args.property]
		err_sq = torch.mean(diff ** 2)
		return err_sq
	return loss


def model(args,omdData,atomrefs, means, stddevs):

	schnet = spk.representation.SchNet(
		n_atom_basis=args.features, n_filters=args.features, n_gaussians=50, n_interactions=6,
		cutoff=5.0, cutoff_network=spk.nn.cutoff.CosineCutoff
	)
	output_module = get_output_module(
            args,
            representation=schnet,
            mean=means,
            stddev=stddevs,
            atomref=atomrefs,
        )

	# output_Bgap = spk.atomistic.Atomwise(n_in=args.features, atomref=atomrefs[OrganicMaterialsDatabase.BandGap], property=OrganicMaterialsDatabase.BandGap,
	# 						   mean=means[OrganicMaterialsDatabase.BandGap], stddev=stddevs[OrganicMaterialsDatabase.BandGap])
	model = spk.AtomisticModel(representation=schnet, output_modules=output_module)
	if args.parallel:
		model = nn.DataParallel(model)
	return model

def train_model(args,model,train_loader,val_loader):

	# before setting up the trainer, remove previous training checkpoints and logs
	if os.path.exists(os.path.join(args.model_path,'checkpoints')):
		shutil.rmtree(os.path.join(args.model_path,'checkpoints'))

	if os.path.exists(os.path.join(args.model_path,'log.csv')):
		os.remove(os.path.join(args.model_path,'log.csv'))

	trainable_params = filter(lambda p: p.requires_grad, model.parameters())
	optimizer = Adam(trainable_params, lr=args.lr , weight_decay = 4e-7
		)
	metrics = [
		spk.train.metrics.MeanAbsoluteError(args.property, args.property),
		spk.train.metrics.RootMeanSquaredError(args.property, args.property),
	]
	hooks = [
		trn.CSVHook(log_path=args.model_path, metrics=metrics),
		trn.ReduceLROnPlateauHook(
        optimizer,
        patience=25, factor=0.8, min_lr=1e-6,window_length=1,
        stop_after_min=True
        )
	]

	loss = simple_loss_fn(args)
	trainer = trn.Trainer(model_path=args.model_path,model=model,
	hooks=hooks,
	loss_fn=loss,
	optimizer=optimizer,
	train_loader=train_loader,
	validation_loader=val_loader,
	)
	
	return trainer

def plot_results(args):

	results = np.loadtxt(os.path.join(args.model_path, 'log.csv'), skiprows=1, delimiter=',')
	time = results[:,0]-results[0,0]
	learning_rate = results[:,1]
	train_loss = results[:,2]
	val_loss = results[:,3]
	val_mae = results[:,4]

	print('Final validation MAE:', np.round(val_mae[-1], 2), 'eV =',
		  np.round(val_mae[-1] / (kcal/mol), 2), 'kcal/mol')

	plt.figure(figsize=(14,5))
	plt.subplot(1,2,1)
	plt.plot(time, val_loss, label='Validation')
	plt.plot(time, train_loss, label='Train')
	plt.yscale('log')
	plt.ylabel('Loss [eV]')
	plt.xlabel('Time [s]')
	plt.legend()
	plt.subplot(1,2,2)
	plt.plot(time, val_mae)
	plt.ylabel('mean abs. error [eV]')
	plt.xlabel('Time [s]')
	plt.savefig('./schnet_acc.png')

def main(args):

	#building model and dataset
	device = torch.device("cuda" if args.cuda else "cpu")
	environment_provider = spk.environment.AseEnvironmentProvider(cutoff=5.0)
	omdb = './omdb'
	if args.mode == "train":
		spk.utils.spk_utils.set_random_seed(None)
		if not os.path.exists('omdb'):
			os.makedirs(omdb)

		omdData = OrganicMaterialsDatabase(args.datapath, download=True, load_only=[args.property], environment_provider=environment_provider)
		split_path = os.path.join(args.model_path, "split.npz")
		train, val, test = spk.train_test_split(
			data=omdData,
			num_train=9000,
			num_val=1000,
			split_file=split_path
		)
		train_loader = spk.AtomsLoader(train, batch_size=16, sampler=RandomSampler(train), num_workers=4 
			#pin_memory=True
			)
		val_loader = spk.AtomsLoader(val, batch_size=16, num_workers=2
			)
		test_loader = spk.AtomsLoader(test, batch_size=16, num_workers=2
			)
		atomref = omdData.get_atomref(args.property)
		mean, stddev = get_statistics(
	        args=args,
	        split_path=split_path,
	        train_loader=train_loader,
	        atomref=atomref,
	        divide_by_atoms=get_divide_by_atoms(args),
	        logging=logging
        )
		# means, stddevs = train_loader.get_statistics(
		# 	args.property, get_divide_by_atoms(args),atomref
		# )
		model_train = model(args,omdData,atomref, mean, stddev)
		trainer = train_model(args,model_train,train_loader,val_loader)
		print('started training')
		trainer.train(device=device, n_epochs=args.n_epochs)
		print('training finished')
		sch_model = torch.load(os.path.join(args.model_path, 'best_model'))

		err = 0
		sch_model.eval()
		for count, batch in enumerate(test_loader):
		    # move batch to GPU, if necessary
		    batch = {k: v.to(device) for k, v in batch.items()}

		    # apply model
		    pred = sch_model(batch)

		    # calculate absolute error
		    tmp = torch.sum(torch.abs(pred[args.property]-batch[args.property]))
		    tmp = tmp.detach().cpu().numpy() # detach from graph & convert to numpy
		    err += tmp

		    # log progress
		    percent = '{:3.2f}'.format(count/len(test_loader)*100)
		    print('Progress:', percent+'%'+' '*(5-len(percent)), end="\r")

		err /= len(test)
		print('Test MAE', np.round(err, 3), 'eV =',
		      np.round(err / (kcal/mol), 3), 'kcal/mol')
		
		#plot results
		plot_results(args)

	elif args.mode == "pred":
		print('predictionsss')
		sch_model = torch.load(os.path.join(args.model_path, 'best_model'), map_location=torch.device(device))
		#reading test data
		# test_dataset = AtomsData('./cod_predict.db')
		# test_loader = spk.AtomsLoader(test_dataset, batch_size=32)

		#reading stored cod list
		#cod_list = np.load('./cod_id_list_old.npy')
		omdData = OrganicMaterialsDatabase(args.datapath, download=True, load_only=[args.property], environment_provider=environment_provider)
		split_path = os.path.join(args.model_path, "split.npz")
		train, val, test = spk.train_test_split(
			data=omdData,
			num_train=9000,
			num_val=1000,
			split_file=split_path
		)
		print(len(test))
		test_loader = spk.AtomsLoader(test, batch_size=32, #num_workers=2
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

