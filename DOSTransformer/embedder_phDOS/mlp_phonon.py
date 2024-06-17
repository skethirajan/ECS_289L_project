import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter_sum
from e3nn import o3
from torch_cluster import radius_graph
from e3nn.nn.models.gate_points_2101 import smooth_cutoff


############################################################################################################################
## MLP with Energy embedding for phonon DOS
############################################################################################################################
class mlp_phonon(nn.Module):
    def __init__(self, layers, n_atom_feats, n_bond_feats, n_hidden, r_max, device):
        super(mlp_phonon, self).__init__()

        # Energy embeddings
        self.embeddings = nn.Embedding(51, n_hidden)
        self.GN_encoder = Encoder1(n_atom_feats, n_bond_feats, n_hidden)
        self.GN_decoder = Decoder(n_hidden)

        nnLayer = nn.ModuleList()
        for i in range(layers):
            nnLayer.append(nn.Linear(n_hidden, n_hidden))
            nnLayer.append(nn.PReLU())

        self.out_layer = nn.Sequential(
            nn.Linear(n_hidden * 2, n_hidden),
            nn.LayerNorm(n_hidden),
            nn.PReLU(),
            *nnLayer,
            nn.Linear(n_hidden, 1),
        )
        self.device = device
        self.max_radius = r_max

    def preprocess(self, data):

        if "batch" in data:
            batch = data["batch"]
        else:
            batch = data["pos"].new_zeros(data["pos"].shape[0], dtype=torch.long)

        # if "edge_index" in data:
        #    edge_src = data["edge_index"][0]  # edge source
        #    edge_dst = data["edge_index"][1]  # edge destination
        #    edge_vec = data["edge_vec"]

        edge_index = radius_graph(data["pos"], self.max_radius, batch)
        edge_src = edge_index[0]
        edge_dst = edge_index[1]
        edge_vec = data["pos"][edge_src] - data["pos"][edge_dst]

        return edge_vec

    def forward(self, g):

        input_ids = torch.tensor(np.arange(51)).to(self.device)
        energies = self.embeddings(input_ids)

        edge_vec = self.preprocess(g)
        edge_sh = o3.spherical_harmonics(
            o3.Irreps.spherical_harmonics(1), edge_vec, True, normalization="component"
        )
        edge_length = edge_vec.norm(dim=1)
        edge_attr = smooth_cutoff(edge_length / 4.0)[:, None] * edge_sh

        x, z, edge_attr, energies = self.GN_encoder(
            x=g.x, z=g.z, edge_attr=edge_attr, batch=g.batch, energies=energies
        )

        graph = self.GN_decoder(x, z, g.batch)
        graph = graph.reshape(-1, graph.shape[0], graph.shape[1]).expand(
            51, graph.shape[0], graph.shape[1]
        )
        dos = self.out_layer(torch.cat([energies, graph], dim=2))
        dos = dos.squeeze(2).T
        return dos


############################################################################################################################
## MLP without Energy embedding for phonon DOS
############################################################################################################################
class mlp2_phonon(nn.Module):
    def __init__(self, layers, n_atom_feats, n_bond_feats, n_hidden, r_max, device):
        super(mlp2_phonon, self).__init__()

        self.GN_encoder = Encoder2(n_atom_feats, n_bond_feats, n_hidden)
        nnLayer = nn.ModuleList()
        for i in range(layers):
            nnLayer.append(nn.Linear(n_hidden, n_hidden))
            nnLayer.append(nn.LeakyReLU())

        self.out_layer = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.LeakyReLU(),
            *nnLayer,
            nn.Linear(n_hidden, 51),
        )
        self.device = device
        self.max_radius = r_max

    def preprocess(self, data):

        if "batch" in data:
            batch = data["batch"]
        else:
            batch = data["pos"].new_zeros(data["pos"].shape[0], dtype=torch.long)

        # if:
        # edge_src = data["edge_index"][0]  # edge source
        # edge_dst = data["edge_index"][1]  # edge destination
        # edge_vec = data["edge_vec"]

        edge_index = radius_graph(data["pos"], self.max_radius, batch)
        edge_src = edge_index[0]
        edge_dst = edge_index[1]
        edge_vec = data["pos"][edge_src] - data["pos"][edge_dst]

        return edge_vec

    def forward(self, g):

        edge_vec = self.preprocess(g)
        edge_sh = o3.spherical_harmonics(
            o3.Irreps.spherical_harmonics(1), edge_vec, True, normalization="component"
        )
        edge_length = edge_vec.norm(dim=1)
        edge_attr = smooth_cutoff(edge_length / 4.0)[:, None] * edge_sh
        x, z, edge_attr = self.GN_encoder(
            x=g.x, z=g.z, edge_attr=edge_attr, batch=g.batch
        )

        sum_pooling = scatter_sum(x, g.batch, dim=0)
        dos_vector = self.out_layer(sum_pooling)

        return dos_vector


############################################################################################################################
## Graph Neural Network
############################################################################################################################


class Encoder1(nn.Module):
    def __init__(self, n_atom_feats, n_bond_feats, n_hidden):
        super(Encoder1, self).__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(n_atom_feats, n_hidden), nn.PReLU(), nn.Linear(n_hidden, n_hidden)
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(n_bond_feats, n_hidden), nn.PReLU(), nn.Linear(n_hidden, n_hidden)
        )
        self.reset_parameters()

    def reset_parameters(self):
        for item in [self.node_encoder, self.edge_encoder]:
            if hasattr(item, "reset_parameters"):
                item.reset_parameters()

    def forward(self, x, z, edge_attr, batch, energies):

        x = self.node_encoder(x)
        z = self.node_encoder(z)
        edge_attr = self.edge_encoder(edge_attr)
        energies = energies.reshape(energies.shape[0], 1, energies.shape[1]).expand(
            energies.shape[0], len(batch.unique()), energies.shape[1]
        )
        return x, z, edge_attr, energies


class Encoder2(nn.Module):
    def __init__(self, n_atom_feats, n_bond_feats, n_hidden):
        super(Encoder2, self).__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(n_atom_feats, n_hidden), nn.PReLU(), nn.Linear(n_hidden, n_hidden)
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(n_bond_feats, n_hidden), nn.PReLU(), nn.Linear(n_hidden, n_hidden)
        )
        self.reset_parameters()

    def reset_parameters(self):
        for item in [self.node_encoder, self.edge_encoder]:
            if hasattr(item, "reset_parameters"):
                item.reset_parameters()

    def forward(self, x, z, edge_attr, batch):

        x = self.node_encoder(x)
        z = self.node_encoder(z)
        edge_attr = self.edge_encoder(edge_attr)
        return x, z, edge_attr


class Decoder(nn.Module):
    def __init__(self, n_hidden):
        super(Decoder, self).__init__()
        # self.mlp = nn.Sequential(nn.Linear(n_hidden * 2, n_hidden), nn.LayerNorm(n_hidden), nn.PReLU(), nn.Linear(n_hidden, n_hidden))
        self.mlp = nn.Sequential(nn.Linear(n_hidden * 2, n_hidden))
        # self.mlp = nn.Sequential(nn.Linear(n_hidden, n_hidden))

    def forward(self, x, z, batch):

        a = z
        z = scatter_sum(z, batch, dim=0)
        output = torch.cat([z, scatter_sum(x, batch, dim=0)], dim=1)
        output = self.mlp(output)
        # output = scatter_sum(x, batch, dim = 0)
        # output = self.mlp(output)

        return output
