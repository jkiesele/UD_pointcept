import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import sys
from functools import partial
import os.path as osp
import time
import numpy as np
import torch_cmspepr
from torch import Tensor
from torch.nn import Linear
from dgl.nn import EdgeWeightNorm, GraphConv


class Push_info_up(nn.Module):
    """
    Every down node is connected to 5 up nodes

    """

    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.in_planes = in_planes
        self.Concat_MLP_Aggregation = Concat_MLP_Aggregation(in_planes, out_planes)

    def forward(self, h, h_above, idx, i, j):
        # feed information back down averaging the information of the upcoming uppoints
        new_h = torch.zeros((h_above.shape[0], h.shape[1])).to(h.device)
        new_h[idx] = h
        g_connected_up = dgl.graph((j, i), num_nodes=new_h.shape[0])
        g_connected_up.ndata["features"] = new_h
        g_connected_up.update_all(
            fn.copy_u("features", "m"), self.Concat_MLP_Aggregation
        )
        h_up_down = g_connected_up.ndata["h_up_down"]
        return h_up_down


class Concat_MLP_Aggregation(nn.Module):
    """
    Feature aggregation in a DGL graph
    """

    def __init__(self, in_planes, out_planes):
        super(Concat_MLP_Aggregation, self).__init__()
        self.out_dim = in_planes
        self.FFC1 = nn.Linear(in_planes * 5, out_planes)
        self.FFC2 = nn.Linear(out_planes, out_planes)

    def forward(self, nodes):
        concat_neigh_h = nodes.mailbox["m"].view(-1, self.out_dim * 5)
        h = self.FFC1(concat_neigh_h)
        h = F.relu(h)
        h = self.FFC2(h)
        return {"h_up_down": h}


class MLP_difs_softmax(nn.Module):
    """
    Param:
    """

    def __init__(
        self,
        in_dim,
        out_dim,
    ):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.edgedistancespassing_softmax = EdgeDistancesPassing_softmax(
            self.in_channels, self.out_channels
        )
        self.edgedistancespassing_1 = EdgeDistancesPassing_1()

        self.norm = EdgeWeightNorm(norm="right")
        self.meanmaxaggregation = MeanMax_aggregation(self.out_channels)
        # self.batch_norm = nn.BatchNorm1d(out_dim)
        # self.FFN_layer1 = nn.Linear(out_dim, out_dim * 2)
        # self.FFN_layer2 = nn.Linear(out_dim * 2, out_dim)
        # self.batch_norm2 = nn.BatchNorm1d(out_dim)

    def forward(self, g, h):
        h_in = h
        g.ndata["features"] = h
        g.apply_edges(self.edgedistancespassing_softmax)
        print("edge data shape", g.edata["att_weight"].shape)
        norm_edge_weight = self.norm(
            dgl.reverse(g), g.edata["att_weight"].view(-1)
        )  # this allows to spread the info of one node into upnodes
        g.edata["att_weight"] = norm_edge_weight
        g.update_all(self.edgedistancespassing_1, fn.sum("feature_n", "h_updated"))
        h = g.ndata["h_updated"]
        # h = self.batch_norm(h)
        # h_in2 = h
        # h = self.FFN_layer1(h)
        # h = F.relu(h)
        # h = self.FFN_layer2(h)
        # h = h_in2 + h  # residual connection
        # h = self.batch_norm2(h)
        return h


class EdgeDistancesPassing_softmax(nn.Module):
    """
    Compute the input feature from neighbors
    """

    def __init__(self, in_dim, out_dim):
        super(EdgeDistancesPassing_softmax, self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(in_dim, out_dim),  #! Dense 3
            nn.ReLU(),
            nn.Linear(out_dim, 1),  #! Dense 4
            nn.ReLU(),
        )

    def forward(self, edges):
        dif = edges.src["features"] - edges.dst["features"]
        att_weight = self.MLP(dif)
        att_weight = torch.sigmoid(att_weight)  #! try sigmoid
        return {"att_weight": att_weight}


class MLP_difs(nn.Module):
    """
    Param:
    """

    def __init__(
        self,
        in_dim,
        out_dim,
    ):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.edgedistancespassing = EdgeDistancesPassing(
            self.in_channels, self.out_channels
        )
        self.meanmaxaggregation = MeanMax_aggregation(self.out_channels)
        self.batch_norm = nn.BatchNorm1d(out_dim)
        self.FFN_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_layer2 = nn.Linear(out_dim * 2, out_dim)
        self.batch_norm2 = nn.BatchNorm1d(out_dim)

    def forward(self, g, h):
        h_in = h
        g.ndata["features"] = h
        g.update_all(self.edgedistancespassing, self.meanmaxaggregation)
        h = g.ndata["h_updated"]
        h = h_in + h
        h = self.batch_norm(h)
        h_in2 = h
        h = self.FFN_layer1(h)
        h = F.relu(h)
        h = self.FFN_layer2(h)
        h = h_in2 + h  # residual connection
        h = self.batch_norm2(h)
        return h


class EdgeDistancesPassing_1(nn.Module):
    """
    Compute the input feature from neighbors
    """

    def __init__(self):
        super(EdgeDistancesPassing_1, self).__init__()

    def forward(self, edges):
        feature = edges.data["att_weight"] * edges.src["features"]
        return {"feature_n": feature}


class EdgeDistancesPassing(nn.Module):
    """
    Compute the input feature from neighbors
    """

    def __init__(self, in_dim, out_dim):
        super(EdgeDistancesPassing, self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(in_dim, out_dim),  #! Dense 3
            nn.ReLU(),
            nn.Linear(out_dim, 1),  #! Dense 4
            nn.ReLU(),
        )

    def forward(self, edges):
        dif = edges.src["features"] - edges.dst["features"]
        att_weight = self.MLP(dif)
        att_weight = torch.sigmoid(att_weight)  #! try sigmoid
        feature = att_weight * edges.src["features"]
        return {"feature_n": feature}


class MeanMax_aggregation(nn.Module):
    """
    Feature aggregation in a DGL graph
    """

    def __init__(self, out_dim):
        super(MeanMax_aggregation, self).__init__()
        self.O = nn.Linear(out_dim * 2, out_dim)

    def forward(self, nodes):
        mean_agg = torch.mean(nodes.mailbox["feature_n"], dim=1)
        max_agg = torch.max(nodes.mailbox["feature_n"], dim=1)[0]
        out = torch.cat([mean_agg, max_agg], dim=-1)
        out = self.O(out)
        return {"h_updated": out}


class MeanMax_aggregation_2(nn.Module):
    """
    Feature aggregation in a DGL graph
    """

    def __init__(self, out_dim):
        super(MeanMax_aggregation_2, self).__init__()
        self.O = nn.Linear(out_dim * 2, out_dim)

    def forward(self, nodes):
        mean_agg = torch.mean(nodes.mailbox["feature_n"], dim=1)
        max_agg = torch.max(nodes.mailbox["feature_n"], dim=1)[0]
        out = torch.cat([mean_agg, max_agg], dim=-1)
        out = self.O(out)
        return {"h_updated": out}


class MLP_difs_maxpool(nn.Module):
    """
    Param:
    """

    def __init__(
        self,
        in_dim,
        out_dim,
    ):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.edgedistancespassing = EdgePassing(self.in_channels, self.out_channels)
        self.meanmaxaggregation = Max_aggregation(self.out_channels)
        # self.batch_norm = nn.BatchNorm1d(out_dim)
        # self.FFN_layer1 = nn.Linear(out_dim, out_dim * 2)
        # self.FFN_layer2 = nn.Linear(out_dim * 2, out_dim)
        # self.batch_norm2 = nn.BatchNorm1d(out_dim)

    def forward(self, g, h):
        g.ndata["features"] = h
        g.update_all(self.edgedistancespassing, self.meanmaxaggregation)
        h = g.ndata["h_updated"]
        # h = h_in + h
        # h = self.batch_norm(h)
        # h_in2 = h
        # h = self.FFN_layer1(h)
        # h = F.relu(h)
        # h = self.FFN_layer2(h)
        # h = h_in2 + h  # residual connection
        # h = self.batch_norm2(h)
        return h


class EdgePassing(nn.Module):
    """
    Compute the input feature from neighbors
    """

    def __init__(self, in_dim, out_dim):
        super(EdgePassing, self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(in_dim, out_dim),  #! Dense 3
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),  #! Dense 4
            nn.ReLU(),
        )

    def forward(self, edges):
        dif = edges.src["features"]
        dif = self.MLP(dif)
        # att_weight = torch.sigmoid(att_weight)  #! try sigmoid
        # feature = att_weight * edges.src["features"]
        return {"feature_n": dif}


class Max_aggregation(nn.Module):
    """
    Feature aggregation in a DGL graph
    """

    def __init__(self, out_dim):
        super(Max_aggregation, self).__init__()

    def forward(self, nodes):
        max_agg = torch.max(nodes.mailbox["feature_n"], dim=1)[0]

        return {"h_updated": max_agg}
