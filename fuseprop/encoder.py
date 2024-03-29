import torch
import torch.nn as nn
import rdkit.Chem as Chem
import torch.nn.functional as F
from fuseprop.nnutils import *
from fuseprop.mol_graph import MolGraph
from fuseprop.rnn import LSTM

class MPNEncoder(nn.Module):

    def __init__(self, rnn_type, input_size, node_fdim, hidden_size, depth):
        super(MPNEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.depth = depth
        self.W_o = nn.Sequential( 
                nn.Linear(node_fdim + hidden_size, hidden_size), 
                nn.ReLU()
        )
        if rnn_type == 'LSTM':
            self.rnn = LSTM(input_size, hidden_size, depth) 
        else:
            raise ValueError('unsupported rnn cell type ' + rnn_type)

    def forward(self, fnode, fmess, agraph, bgraph, mask):
        h = self.rnn(fmess, bgraph)
        h = self.rnn.get_hidden_state(h)
        nei_message = index_select_ND(h, 0, agraph)
        nei_message = nei_message.sum(dim=1)
        node_hiddens = torch.cat([fnode, nei_message], dim=1)
        node_hiddens = self.W_o(node_hiddens)

        if mask is None:
            mask = torch.ones(node_hiddens.size(0), 1, device=fnode.device)
            mask[0, 0] = 0 #first node is padding

        return node_hiddens * mask, h 


class GraphEncoder(nn.Module):
    def __init__(self, avocab, rnn_type, embed_size, hidden_size, depth):
        super(GraphEncoder, self).__init__()
        self.avocab = avocab
        self.hidden_size = hidden_size
        self.atom_size = atom_size = avocab.size() + MolGraph.MAX_POS
        self.bond_size = bond_size = len(MolGraph.BOND_LIST) 
        
        self.E_a = torch.eye( avocab.size() ).cpu()
        self.E_b = torch.eye( len(MolGraph.BOND_LIST) ).cpu()
        self.E_pos = torch.eye( MolGraph.MAX_POS ).cpu()

        self.encoder = MPNEncoder(rnn_type, atom_size + bond_size, atom_size, hidden_size, depth)

