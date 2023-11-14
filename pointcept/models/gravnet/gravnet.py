"""
Point Transformer V2 Mode (recommend)

Disable Grouped Linear

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from copy import deepcopy
import math
import torch
import torch.nn as nn
import dgl
import dgl.function as fn
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn.pool import voxel_grid
from torch_scatter import segment_csr

import einops
from timm.models.layers import DropPath
import pointops

from pointcept.models.builder import MODELS
from pointcept.models.utils import offset2batch, batch2offset
from pointcept.models.utils import offset2batch, batch2offset
import torch_cmspepr
from pointcept.models.gravnet.gravnet_layer import (
    WeirdBatchNorm,
    GravNetBlock,
    obtain_batch_numbers,
)
from pointcept.models.UD_pulling.UP_pulling import build_graph
from pointcept.models.gravnet.plotting_tools import PlotCoordinates


@MODELS.register_module("gravnet")
class GravnetModel(nn.Module):
    def __init__(self):

        super(GravnetModel, self).__init__()
        input_dim = 6
        output_dim = 13
        n_postgn_dense_blocks = 3

        clust_space_norm = "none"
        activation = "elu"
        weird_batchnom = False
        assert activation in ["relu", "tanh", "sigmoid", "elu"]
        acts = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "elu": nn.ELU(),
        }
        self.act = acts[activation]

        N_NEIGHBOURS = [16, 128, 16, 256]
        TOTAL_ITERATIONS = len(N_NEIGHBOURS)
        self.return_graphs = False
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_gravnet_blocks = TOTAL_ITERATIONS
        self.n_postgn_dense_blocks = n_postgn_dense_blocks
        if weird_batchnom:
            self.ScaledGooeyBatchNorm2_1 = WeirdBatchNorm(self.input_dim)
        else:
            self.ScaledGooeyBatchNorm2_1 = nn.BatchNorm1d(self.input_dim, momentum=0.01)

        self.Dense_1 = nn.Linear(input_dim, 64, bias=False)
        self.Dense_1.weight.data.copy_(torch.eye(64, input_dim))
        print("clust_space_norm", clust_space_norm)
        assert clust_space_norm in ["twonorm", "tanh", "none"]
        self.clust_space_norm = clust_space_norm

        self.d_shape = 32
        self.gravnet_blocks = nn.ModuleList(
            [
                GravNetBlock(
                    64 if i == 0 else (self.d_shape * i + 64),
                    k=N_NEIGHBOURS[i],
                    weird_batchnom=weird_batchnom,
                )
                for i in range(self.n_gravnet_blocks)
            ]
        )

        # Post-GravNet dense layers
        postgn_dense_modules = nn.ModuleList()
        for i in range(self.n_postgn_dense_blocks):
            postgn_dense_modules.extend(
                [
                    nn.Linear(4 * self.d_shape + 64 if i == 0 else 64, 64),
                    self.act,  # ,
                ]
            )
        self.postgn_dense = nn.Sequential(*postgn_dense_modules)

        # Output block
        self.output = nn.Sequential(
            nn.Linear(64, 64),
            self.act,
            nn.Linear(64, 64),
            self.act,
            nn.Linear(64, 64),
        )

        self.post_pid_pool_module = nn.Sequential(  # to project pooled "particle type" embeddings to a common space
            nn.Linear(22, 64),
            self.act,
            nn.Linear(64, 64),
            self.act,
            nn.Linear(64, 22),
            nn.Softmax(dim=-1),
        )
        self.clustering = nn.Linear(64, self.output_dim - 1, bias=False)
        self.beta = nn.Linear(64, 1)

        init_weights_ = True
        if init_weights_:
            # init_weights(self.clustering)
            init_weights(self.beta)
            init_weights(self.postgn_dense)
            init_weights(self.output)

        if weird_batchnom:
            self.ScaledGooeyBatchNorm2_2 = WeirdBatchNorm(64)
        else:
            self.ScaledGooeyBatchNorm2_2 = nn.BatchNorm1d(64, momentum=0.01)

    def forward(self, data_dict):
        step_count = 0
        coord = data_dict["coord"]
        feat = data_dict["feat"]
        offset = data_dict["offset"].int()
        object = data_dict["segment"]
        batch = offset2batch(offset)
        # print("shape", coord.shape)
        g = build_graph(batch, coord)
        # print("graph is built")
        # a batch of point cloud is a list of coord, feat and offset
        g.ndata["h"] = feat
        g.ndata["c"] = coord
        #! this is to have the GT for the loss
        g.ndata["object"] = object
        h = g.ndata["h"]
        c = g.ndata["c"]
        x = feat
        original_coords = coord
        x = self.ScaledGooeyBatchNorm2_1(x)
        x = self.Dense_1(x)

        allfeat = []  # To store intermediate outputs
        allfeat.append(x)
        graphs = []
        loss_regularizing_neig = 0.0
        loss_ll = 0
        if step_count % 10:
            PlotCoordinates(g, path="input_coords")
        for num_layer, gravnet_block in enumerate(self.gravnet_blocks):
            x, graph, loss_regularizing_neig_block, loss_ll_ = gravnet_block(
                g,
                x,
                batch,
                original_coords,
                step_count,
                num_layer,
            )

            allfeat.append(x)
            graphs.append(graph)
            loss_regularizing_neig = (
                loss_regularizing_neig_block + loss_regularizing_neig
            )
            loss_ll = loss_ll_ + loss_ll
            if len(allfeat) > 1:
                x = torch.concatenate(allfeat, dim=1)

        x = torch.cat(allfeat, dim=-1)
        x = self.postgn_dense(x)
        x = self.ScaledGooeyBatchNorm2_2(x)
        x_cluster_coord = self.clustering(x)
        return x_cluster_coord


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.00)
