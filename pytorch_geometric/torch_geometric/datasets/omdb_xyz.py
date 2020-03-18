import os
import os.path as osp

import torch
import torch.nn.functional as F
from torch_sparse import coalesce
from torch_geometric.data import (InMemoryDataset, download_url, extract_tar,
								  Data)

from schnetpack.datasets import OrganicMaterialsDatabase
from ase.geometry.analysis import Analysis


class OMDB(InMemoryDataset):
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
			   'gap1_v1.1')
	#need to be changed later
	processed_url = 'https://omdb.mathub.io/dataset/download/gap1_v1.1'
	if rdkit is not None:
		bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

	def __init__(self, root, transform=None, pre_transform=None,
				 pre_filter=None):
		super(OMDB, self).__init__(root, transform, pre_transform, pre_filter)
		self.data, self.slices = torch.load(self.processed_paths[0])

	@property
	def raw_file_names(self):
		return ['structures.xyz','bandgaps.csv', 'CODids.csv']

	@property
	def processed_file_names(self):
		return 'omdb_data.pt'

	def download(self):
		url = self.processed_url if rdkit is None else self.raw_url
		file_path = download_url(url, self.raw_dir)
		extract_tar(file_path, self.raw_dir)
		os.unlink(file_path)

	def process(self):

		with open(self.raw_paths[1], 'r') as f:
			target = f.read().split('\n')[0:-1]
			target = [float(i) for i in target]
			target = torch.tensor(target, dtype=torch.float)

		omdData = OrganicMaterialsDatabase(self.raw_paths[0], download=False)
		data_list = []
		for i, mol in enumerate(omdData):
			if mol is None:
				print(str(i),' not a molecule')
				continue

			pos = mol['_positions']

			atomic_number = []
			for atom_num in mol['_atomic_numbers']:
				atomic_number.append(atom_num.item())
			
			N = len(atomic_number)	
			x = torch.tensor([
				atomic_number
			], dtype=torch.float).t().contiguous()

			row, col, bond_idx = [], [], []
			at_obj = omdData.get_atoms(idx=i)
			bond_anal = Analysis(at_obj)
			for bond_list in bond_anal.unique_bonds:
				for start, atom_bond_list in enumerate(bond_list):
					for end in atom_bond_list:
						row += [start, end]
						col += [end, start]
						bond_idx += 2 * [self.bonds[BT.SINGLE]]

			
			edge_index = torch.tensor([row, col], dtype=torch.long)
			edge_attr = F.one_hot(torch.tensor(bond_idx),
								   num_classes=len(self.bonds)).to(torch.float)
			edge_index, edge_attr = coalesce(edge_index, edge_attr, N, N)

			y = target[i].unsqueeze(0)
			name = mol.GetProp('_Name')

			data = Data(x=x, pos=pos, edge_index=edge_index,edge_attr=edge_attr,
				y=y, name=name)

			if self.pre_filter is not None and not self.pre_filter(data):
				continue
			if self.pre_transform is not None:
				data = self.pre_transform(data)

			data_list.append(data)
	   
		torch.save(self.collate(data_list), self.processed_paths[0])
