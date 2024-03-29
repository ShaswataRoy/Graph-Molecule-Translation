import torch
import os, random, gc
import pickle

from rdkit import Chem
from torch.utils.data import Dataset
from fuseprop.chemutils import *
from fuseprop.mol_graph import MolGraph

class SubgraphDataset(Dataset):

    def __init__(self, data, avocab, batch_size, num_decode):
        data = [x for smiles in data for x in enum_root(smiles, num_decode)]
        self.batches = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]
        self.avocab = avocab
    
    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        return self.batches[idx]

