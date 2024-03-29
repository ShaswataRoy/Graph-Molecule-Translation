import torch
import random
import rdkit
import rdkit.Chem as Chem
import networkx as nx
from fuseprop.chemutils import *
from fuseprop.nnutils import *
from fuseprop.vocab import common_atom_vocab
from collections import deque

add = lambda x,y : x + y if type(x) is int else (x[0] + y, x[1] + y)

class MolGraph(object):

    BOND_LIST = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC] 
    MAX_POS = 40

    def __init__(self, smiles, init_atoms, root_atoms=None, shuffle_roots=True):
        self.smiles = smiles
        self.mol = get_mol(smiles)
        self.mol_graph = self.build_mol_graph()
        self.init_atoms = set(init_atoms)
        self.root_atoms = self.get_root_atoms() if root_atoms is None else root_atoms
        if len(self.root_atoms) > 0:
            if shuffle_roots: random.shuffle(self.root_atoms)
            self.order = self.get_bfs_order()