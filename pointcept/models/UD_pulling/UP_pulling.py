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
from pointcept.models.UD_pulling.tools import Swin3D, MLPReadout
from pointcept.models.utils import offset2batch, batch2offset



@MODELS.register_module("FP")
class FancyNet(nn.Module):
    def __init__(self):
        super().__init__()
        #! starting here
        in_dim_node = 6
        num_heads = 8
        hidden_dim = 80
        self.layer_norm = False
        self.batch_norm = True
        self.residual = True
        dropout = 0.05
        self.number_of_layers = 3
        self.num_classes = 13
        self.embedding_h = nn.Linear(in_dim_node, hidden_dim)
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
                )
                for _ in range(self.number_of_layers)
            ]
        )
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        out_dim = hidden_dim * self.number_of_layers
        self.MLP_layer = MLPReadout(out_dim, self.num_classes)
        #! end here

    def forward(self, data_dict):
        coord = data_dict["coord"]
        feat = data_dict["feat"]
        offset = data_dict["offset"].int()
        object = data_dict["segment"]
        batch = offset2batch(offset)
        g = build_graph(batch, coord)
        print("graph is built")
        # a batch of point cloud is a list of coord, feat and offset
        g.ndata["h"] = feat
        g.ndata["c"] = coord
        #! this is to have the GT for the loss
        g.ndata["object"] = object
        h = g.ndata["h"]
        c = g.ndata["c"]
        h = feat
        h = self.embedding_h(h)
        h = self.batch_norm1(h)
        full_res_features = []
        losses = 0
        depth_label = 0
        full_up_points = []
        ij_pairs = []
        latest_depth_rep = []
        for l, swin3 in enumerate(self.layers):

            features, up_points, g, loss_ud, i, j = swin3(g, h, c)
            up_points = up_points.view(-1)
            ij_pairs.append([i, j])
            full_up_points.append(up_points)
            h = features[up_points]
            # print(up_points)
            # print("h_level+1", h.shape)
            c = c[up_points]
            losses = losses + loss_ud
            depth_label = depth_label + 1
            features_down = features
            for it in range(0, depth_label):
                # print("it", it)
                # print("features_down", features_down.shape)
                h_up_down = self.push_info_down(features_down, i, j)
                # print("h_up_down", h_up_down.shape)
                try:
                    # print("updating rep")
                    latest_depth_rep[l - it] = h_up_down
                except:
                    # print("new rep at level", l, it, depth_label)
                    latest_depth_rep.append(h_up_down)
                if depth_label > 1 and (l - it - 1) >= 0:
                    # print(l, it)
                    h_up_down_previous = latest_depth_rep[l - it - 1]
                    up_points_down = full_up_points[l - it - 1]
                    h_up_down_previous[up_points_down] = h_up_down
                    features_down = h_up_down_previous
                    i, j = ij_pairs[l - it - 1]
            # print(h_up_down.shape)
            full_res_features.append(h_up_down)

        all_resolutions = torch.concat(full_res_features, dim=1)
        h_out = self.MLP_layer(all_resolutions)

        return h_out

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
    unique_instances = torch.unique(batch).view(-1)
    list_graphs = []
    for instance in unique_instances:
        mask = batch == instance
        x = coord[mask]
        knn_g = dgl.knn_graph(x, 3)
        list_graphs.append(knn_g)
    g = dgl.batch(list_graphs)
    return g