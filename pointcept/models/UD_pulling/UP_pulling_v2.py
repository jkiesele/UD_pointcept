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
from pointcept.models.UD_pulling.tools_v2 import MP, MP_up, Downsample_block, MLPReadout
from pointcept.models.UD_pulling.up_down_MP import Push_info_up
from pointcept.models.UD_pulling.plotting_tools import PlotCoordinates
from pointcept.models.utils import offset2batch, batch2offset
import torch_cmspepr
import pointcept.utils.comm as comm


@MODELS.register_module("FP_v2")
class UNet(nn.Module):
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
        hidden_dim = 32
        self.layer_norm = False
        self.batch_norm = True
        self.residual = True
        dropout = 0.05
        self.number_of_layers = 5
        planes = [32, 64, 128, 256, 512]
        self.num_classes = 13
        num_neigh = [
            16,
            16,
            16,
        ]
        n_layers = [2, 2, 2, 2, 2, 2]
        self.embedding_h = nn.Sequential(
            nn.Linear(in_dim_node, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.message_passing = nn.ModuleList(
            [
                MP(
                    in_dim_node=planes[ii],
                    hidden_dim=planes[ii],
                    num_heads=num_heads,
                    layer_norm=self.layer_norm,
                    batch_norm=self.batch_norm,
                    residual=self.residual,
                    dropout=dropout,
                    M=0.25,
                    k_in=16,
                    n_layers=n_layers[ii],
                )
                for ii in range(self.number_of_layers - 1)
            ]
        )
        self.bottelneck = MP(
            in_dim_node=planes[-1],
            hidden_dim=planes[-1],
            num_heads=num_heads,
            layer_norm=self.layer_norm,
            batch_norm=self.batch_norm,
            residual=self.residual,
            dropout=dropout,
            M=0.25,
            k_in=16,
            n_layers=n_layers[self.number_of_layers - 1],
        )
        self.contract_blocks = nn.ModuleList(
            [
                Downsample_block(
                    in_planes=planes[ii], out_planes=planes[ii + 1], M=0.25
                )
                for ii in range(self.number_of_layers - 1)
            ]
        )

        self.extend_blocks = nn.ModuleList(
            [
                Push_info_up(planes[len(planes) - ii - 1], planes[len(planes) - ii - 2])
                for ii in range(self.number_of_layers - 1)
            ]
        )
        n_layers = [2, 2, 2, 2, 2, 2]
        self.message_passing_up = nn.ModuleList(
            [
                MP_up(
                    in_dim_node=planes[len(planes) - ii - 1],
                    hidden_dim=planes[len(planes) - ii - 1],
                    num_heads=num_heads,
                    layer_norm=self.layer_norm,
                    batch_norm=self.batch_norm,
                    residual=self.residual,
                    dropout=dropout,
                    M=0.25,
                    k_in=16,
                    n_layers=n_layers[ii],
                )
                for ii in range(self.number_of_layers - 1)
            ]
        )

        out_dim = hidden_dim
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
        self.ScaledGooeyBatchNorm2_2 = nn.BatchNorm1d(64, momentum=0.1)
        self.step_count = 0

    def forward(self, data_dict):
        coord = data_dict["coord"]
        feat = data_dict["feat"]
        offset = data_dict["offset"].int()
        object = data_dict["segment"]
        batch = offset2batch(offset)

        g = build_graph(batch, coord)

        g.ndata["h"] = feat
        g.ndata["c"] = coord
        g.ndata["object"] = object
        h = g.ndata["h"]
        c = g.ndata["c"]
        h = feat
        if comm.get_local_rank() == 0 and self.step_count % 100 == 0:
            PlotCoordinates(
                g,
                path="input_coords",
                features_type="ones",
                predict=False,
                epoch=str(0),
                step_count=self.step_count,
            )
        g1 = g
        h = self.embedding_h(h)
        ############################
        hs = []
        losses = 0
        depth_label = 0
        down_outs = []
        adj_m = []
        ij_pairs = []
        latest_depth_rep = []

        for l, (mp, down) in enumerate(zip(self.message_passing, self.contract_blocks)):
            # Do message passing flat and store features for skipped connections
            print("h MP in ", l, h.shape)
            g, h = mp(g, h, c)
            print("h MP out", l, h.shape)
            adj_m.append([g.edges()[0], g.edges()[1]])
            s_l = g.ndata["s_l"]
            h_store = h
            hs.append(h_store)

            # Go down one level
            print("down in ", l, g.ndata["h"].shape)
            features, down_points, g, i, j = down(g)
            print("down out ", l, features.shape)
            c = s_l
            down_points = down_points.view(-1)
            ij_pairs.append([i, j])
            down_outs.append(down_points)
            h = features[down_points]
            c = c[down_points]
            depth_label = depth_label + 1

        print("bottelneck in ", h.shape)
        g, h = self.bottelneck(g, h, c)
        print("bottelneck out ", h.shape)
        h_store = h
        hs.append(h_store)
        for layer_idx in range(self.number_of_layers - 1):
            up_idx = self.number_of_layers - layer_idx - 1
            i, j = ij_pairs[up_idx - 1]
            # h = hs[up_idx]
            h_above = hs[up_idx - 1]
            idx = down_outs[up_idx - 1]
            print("extend blocks in", layer_idx, h.shape)
            h = self.extend_blocks[layer_idx](h, h_above, idx, i, j)
            print("extend blocks out", layer_idx, h.shape)
            i, j = adj_m[up_idx - 1]
            g = dgl.graph((i, j), num_nodes=h.shape[0])
            g.ndata["h"] = h
            g, h = self.message_passing_up[layer_idx](g, h)
            # skipped connection
            h = h + h_above

        x = self.postgn_dense(h)
        x = self.ScaledGooeyBatchNorm2_2(x)
        h_out = self.clustering(x)
        g1.ndata["final_clustering"] = torch.argmax(h_out, dim=1)
        if comm.get_local_rank() == 0 and self.step_count % 100 == 0:
            PlotCoordinates(
                g1,
                path="final_clustering",
                features_type="ones",
                predict=False,
                epoch=str(0),
                step_count=self.step_count,
            )
        self.step_count = self.step_count + 1
        return h_out  # , losses / self.number_of_layers


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

    # if l ==0:
    #     g1.ndata["up_points"] = down_points + 1
    # losses = losses + loss_ud
    # features_up = features

    # for it in range(0, depth_label):
    #     h_up_down = self.push_info_down(features_up, i, j)
    #     try:
    #         latest_depth_rep[l - it] = h_up_down
    #     except:
    #         latest_depth_rep.append(h_up_down)
    #     if depth_label > 1 and (l - it - 1) >= 0:
    #         # print(l, it)
    #         h_up_down_previous = latest_depth_rep[l - it - 1]
    #         up_points_down = full_down_points[l - it - 1]
    #         h_up_down_previous[up_points_down] = h_up_down
    #         features_up = h_up_down_previous
    #         i, j = ij_pairs[l - it - 1]

    # full_res_features.append(h_up_down)
