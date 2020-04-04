import os
import os.path as osp
import tarfile
import pickle

import torch
import torch.nn.functional as F
from torch_sparse import coalesce
from torch_geometric.data import (InMemoryDataset, download_url, extract_tar,
								  Data)

from schnetpack.datasets import OrganicMaterialsDatabase
from ase.geometry.analysis import Analysis
import numpy as np
from ase.io import read
from ase.db import connect
from ase.units import eV



class OMDBXYZ(InMemoryDataset):
	r"""Organic Materials Database (OMDB) of bulk organic crystals.

	Registration to the OMDB is free for academic users. This database contains DFT
	(PBE) band gap (OMDB-GAP1 database) for 12500 non-magnetic materials.

	Args:
		path (str): path to directory containing database.
		cutoff (float): cutoff for bulk interactions.
		download (bool, optional): enable downloading if database does not exists.
		subset (list): indices to subset. Set to None for entire database.
		load_only (list, optional): reduced set of properties to be loaded
		collect_triples (bool, optional): Set to True if angular features are needed.
		environment_provider (spk.environment.BaseEnvironmentProvider): define how
			neighborhood is calculated
			(default=spk.environment.SimpleEnvironmentProvider).

	References:
		arXiv: https://arxiv.org/abs/1810.12814 "Band gap prediction for large organic
		crystal structures with machine learning" Bart Olsthoorn, R. Matthias Geilhufe,
		Stanislav S. Borysov, Alexander V. Balatsky (Submitted on 30 Oct 2018)
	"""  

	raw_url = ('https://omdb.mathub.io/dataset/download/'
			   'OMDB-GAP1_v1.1.tar.gz')
	#need to be changed later
	processed_url = 'https://omdb.mathub.io/dataset/download/gap1_v1.1'
	bonds = {"SINGLE": 0, "DOUBLE": 1, "TRIPLE": 2, "AROMATIC": 3}

	def __init__(self, root, transform=None, pre_transform=None,
				 pre_filter=None):
		super(OMDBXYZ, self).__init__(root, transform, pre_transform, pre_filter)
		self.data, self.slices = np.load(self.processed_paths[0])

	@property
	def raw_file_names(self):
		return ['OMDB-GAP1_v1.1.db','bandgaps.csv', 'CODids.csv']

	@property
	def processed_file_names(self):
		return 'omdb_data.npz'

	def download(self):
		url = self.raw_url
		file_path = download_url(url, self.raw_dir)
		print("Converting %s to a .db file.." % file_path)
		tar = tarfile.open(osp.join(self.raw_dir,'OMDB-GAP1_v1.1.tar.gz'), "r:gz")
		names = tar.getnames()
		tar.extractall()
		tar.close()

		structures = read("structures.xyz", index=":")
		Y = np.loadtxt("bandgaps.csv")
		[os.remove(name) for name in names]

		with connect(self.dbpath) as con:
			for i, at in enumerate(structures):
				con.write(at, data={OrganicMaterialsDatabase.BandGap: Y[i]})

	def process(self):

		with open(self.raw_paths[1], 'r') as f:
			target = f.read().split('\n')[0:-1]
			target = [float(i) for i in target]
			target = torch.tensor(target, dtype=torch.float)

		omdData = OrganicMaterialsDatabase(self.raw_paths[0], download=False)
		print(len(omdData))
		data_list = []
		for i in range(len(omdData)):
			mol = omdData[i]
			if mol is None:
				print(str(i),' not a molecule')
				continue

			pos = mol['_positions']
			print('after constructing positions')

			atomic_number = []
			for atom_num in mol['_atomic_numbers']:
				atomic_number.append(atom_num.item())
			
			N = len(atomic_number)	
			x = torch.tensor([
				atomic_number
			], dtype=torch.float).t().contiguous()

			row, col, bond_idx = [], [], []
			print('before fetching the atom properties ',str(i))
			at_obj = omdData.get_atoms(idx=i)
			print('after fetching the atom properties')

			bond_anal = Analysis(at_obj)
			for bond_list in bond_anal.unique_bonds:
				for start, atom_bond_list in enumerate(bond_list):
					for end in atom_bond_list:
						row += [start, end]
						col += [end, start]
						bond_idx += 2 * [self.bonds["SINGLE"]]

			print('after constructing the bonds')
			edge_index = torch.tensor([row, col], dtype=torch.long)
			edge_attr = F.one_hot(torch.tensor(bond_idx),
								   num_classes=len(self.bonds)).to(torch.float)
			edge_index, edge_attr = coalesce(edge_index, edge_attr, N, N)

			y = target[i].unsqueeze(0)
			name = str(at_obj.symbols)
			print('constructing data')
			data = Data(x=x, pos=pos, edge_index=edge_index,edge_attr=edge_attr,
				y=y, name=name)

			print('after constructing data')
			if self.pre_filter is not None and not self.pre_filter(data):
				continue
			if self.pre_transform is not None:
				data = self.pre_transform(data)

			# print('----------------')
			# print(data.x.shape)
			# print(data.edge_index.shape)
			# print(data.x)
			# print(data.edge_index)
			# print('------------------')

			data_list.append(data)
		
		print('saving data')
		#torch.save(self.collate(data_list), self.processed_paths[0])
		np.savez(self.processed_paths[0],np.array(data_list))


