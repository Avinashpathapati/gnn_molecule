
import os
import torch
import schnetpack as spk
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
from schnetpack import AtomsData
from schnetpack.utils import (
    get_divide_by_atoms
)


def simple_loss_fn(args):
	def loss(batch, result):
		diff = batch[args.property]-result[args.property]
		err_sq = torch.mean(diff ** 2)
		return err_sq
	return loss


def model(args,omdData,atomrefs, means, stddevs):

	schnet = spk.representation.SchNet(
		n_filters=64, n_gaussians=25, n_interactions=3,
		cutoff=5., cutoff_network=spk.nn.cutoff.CosineCutoff
	)

	output_Bgap = spk.atomistic.Atomwise(n_in=args.features, atomref=atomrefs[OrganicMaterialsDatabase.BandGap], property=OrganicMaterialsDatabase.BandGap,
							   mean=means[OrganicMaterialsDatabase.BandGap], stddev=stddevs[OrganicMaterialsDatabase.BandGap])
	model = spk.AtomisticModel(representation=schnet, output_modules=output_Bgap)
	return model

def train_model(args,model,train_loader,val_loader):

	# before setting up the trainer, remove previous training checkpoints and logs

	if os.path.exists('./omdb/checkpoints'):
		shutil.rmtree('./omdb/checkpoints')

	if os.path.exists('./omdb/log.csv'):
		os.remove('./omdb/log.csv')

	loss = simple_loss_fn(args)
	trainable_params = filter(lambda p: p.requires_grad, model.parameters())
	optimizer = Adam(trainable_params, lr=args.lr)
	metrics = [
		spk.train.metrics.MeanAbsoluteError(args.property, args.property),
		spk.train.metrics.RootMeanSquaredError(args.property, args.property),
	]
	hooks = [
		trn.CSVHook(log_path='./omdb', metrics=metrics),
		trn.ReduceLROnPlateauHook(
			optimizer,
			patience=5, factor=0.6, min_lr=1e-6,
			stop_after_min=True
		)
	]
	trainer = trn.Trainer(model_path='./omdb',model=model,
	hooks=hooks,
	loss_fn=loss,
	optimizer=optimizer,
	train_loader=train_loader,
	validation_loader=val_loader,
	)
	
	return trainer

def plot_results():

	results = np.loadtxt(os.path.join('./omdb', 'log.csv'), skiprows=1, delimiter=',')
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
	plt.show()

def main(args):

	#building model
	device = torch.device("cuda" if args.cuda else "cpu")
	omdb = './omdb'
	if args.mode == "train":
		if not os.path.exists('omdb'):
			os.makedirs(omdb)

		omdData = OrganicMaterialsDatabase(args.datapath, download=False)

		train, val, test = spk.train_test_split(
			data=omdData,
			num_train=10000,
			num_val=2500,
			split_file=os.path.join(args.model_path, "split.npz"),
		)
		train_loader = spk.AtomsLoader(train, batch_size=32, shuffle=True)
		val_loader = spk.AtomsLoader(val, batch_size=32)
		atomref = omdData.get_atomref(args.property)
		means, stddevs = train_loader.get_statistics(
			args.property, get_divide_by_atoms(args),atomref
		)
		model_train = model(args,omdData,atomref, means, stddevs)
		trainer = train_model(args,model_train,train_loader,val_loader)
		trainer.train(device=device, n_epochs=args.n_epochs)
		#plot results
		plot_results()
	elif args.mode == "pred":

		sch_model = torch.load(os.path.join(omdb, 'best_model'))
		#reading test data
		test_dataset = AtomsData('./cod_predict.db')
		test_loader = spk.AtomsLoader(test_dataset, batch_size=100)
		#reading stored cod list
		cod_list = np.load('./cod_id_list_old.npy')
		mean_abs_err = 0
		prediction_list = []
		print('Started generating predictions')
		for count, batch in enumerate(test_loader):
		    
		    # move batch to GPU, if necessary
		    batch = {k: v.to(device) for k, v in batch.items()}
		    # apply model
		    pred = sch_model(batch)
		    prediction_list.extend(pred['band_gap'].detach().numpy().flatten().tolist())

		    # log progress
		    percent = '{:3.2f}'.format(count/len(test_loader)*100)
		    print('Progress:', percent+'%'+' '*(5-len(percent)), end="\r")

		results_df = pd.DataFrame({'cod':cod_list, 'prediction':prediction_list})
		results_df.to_csv('./predictions.csv')



if __name__ == "__main__":

	parser = arg_parser.build_parser()
	(options, args) = parser.parse_args()
	main(options)

