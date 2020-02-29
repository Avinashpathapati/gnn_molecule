import os
import os.path as osp

import torch
import torch.nn.functional as F
from torch_sparse import coalesce
from torch_geometric.data import (InMemoryDataset, download_url, extract_tar,
                                  Data)

try:
    import rdkit
    from rdkit import Chem
    from rdkit import rdBase
    from rdkit.Chem.rdchem import HybridizationType
    from rdkit import RDConfig
    from rdkit.Chem import ChemicalFeatures
    from rdkit.Chem.rdchem import BondType as BT
    rdBase.DisableLog('rdApp.error')
except ImportError:
    rdkit = None


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
        return ['structures.sdf','bandgaps.csv', 'CODids.csv']

    @property
    def processed_file_names(self):
        return 'omdb_data.pt'

    def download(self):
        url = self.processed_url if rdkit is None else self.raw_url
        file_path = download_url(url, self.raw_dir)
        print('------------')
        print(self.raw_dir)
        print('--------------')
        extract_tar(file_path, self.raw_dir)
        os.unlink(file_path)

    def process(self):
        if rdkit is None:
            print('Using a pre-processed version of the dataset. Please '
                  'install `rdkit` to alternatively process the raw data.')

            self.data, self.slices = torch.load(self.raw_paths[0])
            data_list = [data for data in self]

            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]

            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])
            return

        with open(self.raw_paths[1], 'r') as f:
            target = f.read().split('\n')[0:-1]
            target = [float(i) for i in target]
            target = torch.tensor(target, dtype=torch.float)

        suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False)
        fdef_name = osp.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
        factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)

        data_list = []
        for i, mol in enumerate(suppl):
            if mol is None:
                continue

            text = suppl.GetItemText(i)
            N = mol.GetNumAtoms()

            pos = text.split('\n')[4:4 + N]
            pos = [[float(x) for x in line.split()[:3]] for line in pos]
            pos = torch.tensor(pos, dtype=torch.float)

            #type_idx = []
            atomic_number = []
            #acceptor = []
            #donor = []
            # aromatic = []
            # sp = []
            # sp2 = []
            # sp3 = []
            # num_hs = []
            for atom in mol.GetAtoms():
                #type_idx.append(self.types[atom.GetSymbol()])
                atomic_number.append(atom.GetAtomicNum())
                #donor.append(0)
                #acceptor.append(0)
                #aromatic.append(1 if atom.GetIsAromatic() else 0)
                # hybridization = atom.GetHybridization()
                # sp.append(1 if hybridization == HybridizationType.SP else 0)
                # sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
                # sp3.append(1 if hybridization == HybridizationType.SP3 else 0)
                # num_hs.append(atom.GetTotalNumHs(includeNeighbors=True))

            # feats = factory.GetFeaturesForMol(mol)
            # for j in range(0, len(feats)):
            #     if feats[j].GetFamily() == 'Donor':
            #         node_list = feats[j].GetAtomIds()
            #         for k in node_list:
            #             donor[k] = 1
            #     elif feats[j].GetFamily() == 'Acceptor':
            #         node_list = feats[j].GetAtomIds()
            #         for k in node_list:
            #             acceptor[k] = 1

            #x1 = F.one_hot(torch.tensor(type_idx), num_classes=len(self.types))
            x2 = torch.tensor([
                atomic_number
                #acceptor, donor, aromatic, sp, sp2, sp3, num_hs
            ], dtype=torch.float).t().contiguous()
            #x = torch.cat([x1.to(torch.float), x2], dim=-1)
            x = x2

            row, col, bond_idx = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
                bond_idx += 2 * [self.bonds[bond.GetBondType()]]

            edge_index = torch.tensor([row, col], dtype=torch.long)
            print(edge_index)
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
