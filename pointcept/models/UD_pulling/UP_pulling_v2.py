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
from pointcept.models.UD_pulling.tools_v2 import Swin3D, MLPReadout
from pointcept.models.utils import offset2batch, batch2offset
import torch_cmspepr


@MODELS.register_module("FP_v2")
class FancyNet(nn.Module):
    def __init__(self):
        super().__init__()
        #! starting here
        activation = "elu"
        acts = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "elu": nn.ELU(),
        }
        self.act = acts[activation]
        in_dim_node = 6
        num_heads = 8
        hidden_dim = 80
        self.layer_norm = False
        self.batch_norm = True
        self.residual = True
        dropout = 0.05
        self.number_of_layers = 4
        self.num_classes = 13
        num_neigh = [7, 7, 7, 7, 7, 7, 7, 7, 7, 7]
        # self.embedding_h = nn.Linear(in_dim_node, hidden_dim)
        self.embedding_h = nn.Sequential(
            nn.Linear(in_dim_node, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.layers = nn.ModuleList(
            [
                Swin3D(
                    in_dim_node=hidden_dim,
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    layer_norm=self.layer_norm,
                    batch_norm=self.batch_norm,
                    residual=self.residual,
                    dropout=dropout,
                    M=0.5,
                    k_in=num_neigh[ii],
                )
                for ii in range(self.number_of_layers)
            ]
        )
        self.batch_norm1 = nn.BatchNorm1d(in_dim_node, momentum=0.01)
        hidden_dim = hidden_dim
        out_dim = hidden_dim  # * self.number_of_layers
        self.n_postgn_dense_blocks = 3
        postgn_dense_modules = nn.ModuleList()
        for i in range(self.n_postgn_dense_blocks):
            postgn_dense_modules.extend(
                [
                    nn.Linear(out_dim if i == 0 else 64, 64),
                    self.act,  # ,
                ]
            )
        self.postgn_dense = nn.Sequential(*postgn_dense_modules)
        self.clustering = nn.Linear(64, 13, bias=False)
        self.ScaledGooeyBatchNorm2_2 = nn.BatchNorm1d(64, momentum=0.01)

    def forward(self, data_dict):
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
        h = feat
        ##### initial feature embedding
        h = self.batch_norm1(h)
        h = self.embedding_h(h)
        ############################
        full_res_features = []
        losses = 0
        depth_label = 0
        full_up_points = []
        ij_pairs = []
        latest_depth_rep = []
        for l, swin3 in enumerate(self.layers):
            features, up_points, g, i, j, s_l = swin3(g, h, c)
            c = s_l
            up_points = up_points.view(-1)
            ij_pairs.append([i, j])
            full_up_points.append(up_points)
            h = features[up_points]
            c = c[up_points]
            depth_label = depth_label + 1
            # losses = losses + loss_ud
            features_down = features
            for it in range(0, depth_label):
                h_up_down = self.push_info_down(features_down, i, j)
                try:
                    latest_depth_rep[l - it] = h_up_down
                except:
                    latest_depth_rep.append(h_up_down)
                if depth_label > 1 and (l - it - 1) >= 0:
                    # print(l, it)
                    h_up_down_previous = latest_depth_rep[l - it - 1]
                    up_points_down = full_up_points[l - it - 1]
                    h_up_down_previous[up_points_down] = h_up_down
                    features_down = h_up_down_previous
                    i, j = ij_pairs[l - it - 1]
            # h = h_up_down
            # print(h_up_down.shape)
            # full_res_features.append(h_up_down)

        # all_resolutions = torch.concat(h_up_down, dim=1)
        x = self.postgn_dense(h_up_down)
        x = self.ScaledGooeyBatchNorm2_2(x)
        h_out = self.clustering(x)

        return h_out  # , losses / self.number_of_layers

    def push_info_down(self, features, i, j):
        # feed information back down averaging the information of the upcoming uppoints
        g_connected_down = dgl.graph((j, i), num_nodes=features.shape[0])
        g_connected_down.ndata["features"] = features
        g_connected_down.update_all(
            fn.copy_u("features", "m"), fn.sum("m", "h")
        )  #! full resolution graph
        h_up_down = g_connected_down.ndata["h"]
        # g connected down is the highest resolution graph with mean features of the up nodes
        return h_up_down


def build_graph(batch, coord):
    import time

    unique_instances = torch.unique(batch).view(-1)
    list_graphs = []
    for instance in unique_instances:
        mask = batch == instance
        # print("nimner of hits per graph", instance, torch.sum(mask))
        x = coord[mask]
        # knn_g = dgl.knn_graph(x, 3)
        # tic = time.time()
        edge_index = torch_cmspepr.knn_graph(
            x, k=3
        )  # no need to split by batch as we are looping through instances
        # toc = time.time()
        # print("time to build the graph", toc-tic)
        knn_g = dgl.graph((edge_index[0], edge_index[1]), num_nodes=x.shape[0])
        list_graphs.append(knn_g)
    g = dgl.batch(list_graphs)
    return g
