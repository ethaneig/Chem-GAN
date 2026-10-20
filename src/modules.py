import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import NUM_ATOMS, ATOM_DIM, BOND_DIM, LATENT_DIM


class GraphGenerator(nn.Module):
    def __init__(self, dense_units, dropout_rate, latent_dim, adjacency_shape, feature_shape):
        super(GraphGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.adjacency_shape = adjacency_shape
        self.feature_shape = feature_shape

        layers = []
        in_features = latent_dim
        for units in dense_units:
            layers.append(nn.Linear(in_features, units))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(dropout_rate))
            in_features = units
        self.mlp = nn.Sequential(*layers)

        self.adjacency_out = nn.Linear(in_features, np.prod(adjacency_shape))
        self.feature_out = nn.Linear(in_features, np.prod(feature_shape))

    def forward(self, z):
        x = self.mlp(z)

        x_adj = self.adjacency_out(x)
        x_adj = x_adj.view(-1, *self.adjacency_shape)
        x_adj = (x_adj + x_adj.transpose(-1, -2)) / 2.0
        x_adj = F.softmax(x_adj, dim=1)

        x_feat = self.feature_out(x)
        x_feat = x_feat.view(-1, *self.feature_shape)
        x_feat = F.softmax(x_feat, dim=2)

        return x_adj, x_feat




class RelationalGraphConvLayer(nn.Module):
    def __init__(self, atom_dim, bond_dim, units, activation=F.relu, use_bias=False):
        super().__init__()
        self.units = units
        self.use_bias = use_bias
        self.activation = activation

        self.kernel = nn.Parameter(torch.Tensor(bond_dim, atom_dim, units))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(bond_dim, 1, units))
        else:
            self.register_parameter('bias', None)

        nn.init.xavier_uniform_(self.kernel)
        if self.use_bias:
            nn.init.zeros_(self.bias)

    def forward(self, adjacency, features):
        features_exp = features.unsqueeze(1)
        x = torch.matmul(adjacency, features_exp)
        x = torch.einsum('brnf,rfu->brnu', x, self.kernel)

        if self.use_bias:
            x = x + self.bias 

        x = torch.sum(x, dim=1)
        return self.activation(x)


class GraphDiscriminator(nn.Module):
    def __init__(self, gconv_units, dense_units, dropout_rate,
                 adjacency_shape, feature_shape, use_bias=False):
        super().__init__()
        bond_dim, _, _ = adjacency_shape
        _, atom_dim = feature_shape

        self.gconv_layers = nn.ModuleList()
        in_dim = atom_dim
        for units in gconv_units:
            self.gconv_layers.append(
                RelationalGraphConvLayer(
                    atom_dim=in_dim,
                    bond_dim=bond_dim,
                    units=units,
                    activation=F.relu,
                    use_bias=use_bias
                )
            )
            in_dim = units

        dense_layers = []
        for units in dense_units:
            dense_layers.append(nn.Linear(in_dim, units))
            dense_layers.append(nn.ReLU())
            dense_layers.append(nn.Dropout(dropout_rate))
            in_dim = units
        self.mlp = nn.Sequential(*dense_layers)
        self.output_layer = nn.Linear(in_dim, 1)

    def forward(self, adjacency, features):
        x = features
        for gconv in self.gconv_layers:
            x = gconv(adjacency, x)

        x = torch.mean(x, dim=1)
        x = self.mlp(x)

        return self.output_layer(x)


def build_generator(latent_dim=LATENT_DIM):
    return GraphGenerator(
        dense_units=[128, 256, 512],
        dropout_rate=0.2,
        latent_dim=latent_dim,
        adjacency_shape=(BOND_DIM, NUM_ATOMS, NUM_ATOMS),
        feature_shape=(NUM_ATOMS, ATOM_DIM),
    )


def build_discriminator():
    return GraphDiscriminator(
        gconv_units=[128, 128, 128, 128],
        dense_units=[512, 512],
        dropout_rate=0.2,
        adjacency_shape=(BOND_DIM, NUM_ATOMS, NUM_ATOMS),
        feature_shape=(NUM_ATOMS, ATOM_DIM),
    )
