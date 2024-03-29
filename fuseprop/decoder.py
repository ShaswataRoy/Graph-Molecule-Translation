import torch
import torch.nn as nn
import rdkit.Chem as Chem
import torch.nn.functional as F
from fuseprop.nnutils import *
from fuseprop.encoder import GraphEncoder
from fuseprop.mol_graph import MolGraph
from fuseprop.inc_graph import IncGraph
from collections import deque

class HTuple():
    def __init__(self, node=None, mess=None, vmask=None, emask=None):
        self.node, self.mess = node, mess
        self.vmask, self.emask = vmask, emask

class GraphDecoder(nn.Module):

    def __init__(self, avocab, rnn_type, embed_size, hidden_size, latent_size, depth):
        super(GraphDecoder, self).__init__()
        self.avocab = avocab
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.latent_size = latent_size
        self.itensor = torch.LongTensor([]).cpu()

        self.mpn = GraphEncoder(avocab, rnn_type, embed_size, hidden_size, depth)
        self.rnn_cell = self.mpn.encoder.rnn

        self.topoNN = nn.Sequential(
                nn.Linear(hidden_size * 2 + latent_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1)
        )
        self.atomNN = nn.Sequential(
                nn.Linear(hidden_size * 2 + latent_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, avocab.size())
        )
        self.bondNN = nn.Sequential(
                nn.Linear(hidden_size * 3 + latent_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, len(MolGraph.BOND_LIST))
        )

        self.R_bond = nn.Sequential(
                nn.Linear(hidden_size + avocab.size(), hidden_size),
                nn.ReLU()
        )
        self.W_bond = nn.Sequential(
                nn.Linear(hidden_size + len(MolGraph.BOND_LIST), hidden_size),
                nn.ReLU()
        )
        self.E_a = torch.eye( avocab.size() ).cpu()
        self.E_b = torch.eye( len(MolGraph.BOND_LIST) ).cpu()

        self.topo_loss = nn.BCEWithLogitsLoss(reduction='sum')
        self.atom_loss = nn.CrossEntropyLoss(reduction='sum')
        self.bond_loss = nn.CrossEntropyLoss(reduction='sum')

    def get_topo_score(self, src_graph_vecs, batch_idx, topo_vecs):
        topo_cxt = src_graph_vecs.index_select(0, batch_idx)
        return self.topoNN( torch.cat([topo_vecs, topo_cxt], dim=-1) ).squeeze(-1)

    def get_atom_score(self, src_graph_vecs, batch_idx, atom_vecs):
        atom_cxt = src_graph_vecs.index_select(0, batch_idx)
        atom_vecs = torch.cat([atom_vecs, atom_cxt], dim=-1)
        return self.atomNN(atom_vecs)

    def get_bond_score(self, src_graph_vecs, batch_idx, bond_vecs):
        bond_cxt = src_graph_vecs.index_select(0, batch_idx)
        bond_vecs = torch.cat([bond_vecs, bond_cxt], dim=-1)
        return self.bondNN(bond_vecs)


    def decode(self, src_graph_vecs, init_mols, max_decode_step=80):
        assert len(init_mols) == len(src_graph_vecs)
        batch_size = len(src_graph_vecs)
        graph_batch = IncGraph(self.avocab, batch_size, node_fdim=self.mpn.atom_size, edge_fdim=self.mpn.atom_size + self.mpn.bond_size)
        queue = [deque() for _ in range(batch_size)]

        for bid in range(batch_size):    
            root_atoms = graph_batch.add_mol(bid, init_mols[bid])
            queue[bid].extend(root_atoms)
        
        for t in range(max_decode_step):
            batch_list = [ bid for bid in range(batch_size) if len(queue[bid]) > 0 ]
            if len(batch_list) == 0: break

            hgraph = HTuple() 
            graph_tensors = graph_batch.get_tensors()
            hgraph.node, hgraph.mess = self.mpn.encoder(*graph_tensors, mask=None)

            for i,bid in enumerate(batch_list):
                xid = queue[bid][0]
                bid_tensor = self.itensor.new_tensor([bid]) 
                bid_nodes = self.itensor.new_tensor(graph_batch.batch[bid])
                gvec = hgraph.node.index_select(0, bid_nodes).sum(dim=0)
                cxt_vec = torch.cat( (gvec, hgraph.node[xid]), dim=-1).unsqueeze(0)
                stop_score = self.get_topo_score(src_graph_vecs, bid_tensor, cxt_vec)
                stop_score = stop_score.item()

                if stop_score > 0 or graph_batch.can_expand(xid) is False:
                    queue[bid].popleft()
                    continue

                atom_score = self.get_atom_score(src_graph_vecs, bid_tensor, cxt_vec)
                atom_type_id = atom_score.max(dim=-1)[1].item()
                atom_type = self.avocab.get_smiles(atom_type_id)
                yid = graph_batch.add_atom(bid, atom_type)
                #print(atom_score.max(dim=-1))

                cands = list(queue[bid])
                hist = torch.zeros_like(hgraph.node[xid])
                atom_vec = self.E_a[atom_type_id]
                for j,zid in enumerate(cands):
                    cur_hnode = torch.cat([hist, atom_vec], dim=-1)
                    cur_hnode = self.R_bond(cur_hnode)
                    pairs = torch.cat([gvec, cur_hnode, hgraph.node[zid]], dim=-1)
                    pairs = pairs.unsqueeze(0)

                    bid_tensor = self.itensor.new_tensor( [bid] )
                    bond_scores = self.get_bond_score( src_graph_vecs, bid_tensor, pairs )
                    bt = bond_scores.max(dim=-1)[1].item()
                    if bt > 0: 
                        graph_batch.add_bond(yid, zid, bt)
                        bond_vec = self.E_b[bt]
                        z_hnode = torch.cat([hgraph.node[zid], bond_vec], dim=-1)
                        hist += self.W_bond(z_hnode)

                queue[bid].append(yid)
               
        return graph_batch.get_mol()

