import torch
import torch.nn as nn
import rdkit.Chem as Chem
import torch.nn.functional as F
from fuseprop.mol_graph import MolGraph
from fuseprop.encoder import GraphEncoder
from fuseprop.decoder import GraphDecoder
from fuseprop.nnutils import *

def make_cuda(graph_tensors):
    make_tensor = lambda x: x if type(x) is torch.Tensor else torch.tensor(x)
    graph_tensors = [make_tensor(x).cpu().long() for x in graph_tensors[:-1]] + [graph_tensors[-1]]
    return graph_tensors


class AtomVGNN(nn.Module):

    def __init__(self, args):
        super(AtomVGNN, self).__init__()
        self.latent_size = args.latent_size
        self.encoder = GraphEncoder(args.atom_vocab, args.rnn_type, args.embed_size, args.hidden_size, args.depth)
        self.decoder = GraphDecoder(args.atom_vocab, args.rnn_type, args.embed_size, args.hidden_size, args.latent_size, args.diter)

        self.G_mean = nn.Linear(args.hidden_size, args.latent_size)
        self.G_var = nn.Linear(args.hidden_size, args.latent_size)

    def encode(self, graph_tensors):
        graph_vecs = self.encoder(graph_tensors)
        graph_vecs = [graph_vecs[st : st + le].sum(dim=0) for st,le in graph_tensors[-1]]
        return torch.stack(graph_vecs, dim=0)

    def decode(self, init_smiles):
        batch_size = len(init_smiles)
        z_graph_vecs = torch.randn(batch_size, self.latent_size).cpu()
        return self.decoder.decode(z_graph_vecs, init_smiles)