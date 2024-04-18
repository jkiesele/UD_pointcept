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
from pointcept.models.UD_pulling.attention_layers_point_transformer import (
    GraphTransformerLayer,
)
from pointcept.models.UD_pulling.up_down_MP import (
    MLP_difs,
    MLP_difs_softmax,
    MLP_difs_maxpool,
)
import pointops


class MP(nn.Module):
    """MAIN block
    1) Find coordinates and score for the graph
    2) Do knn down graph
    3) Message passing on the down graph SWIN3D_Blocks

    """

    def __init__(
        self,
        in_dim_node,
        hidden_dim,
        num_heads,
        layer_norm,
        batch_norm,
        residual,
        dropout,
        M,
        k_in,
        n_layers,
    ):
        super().__init__()
        self.k = k_in
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.residual = residual

        # self.send_scores = SendScoresMessage()
        # self.find_up = FindUpPoints()
        self.sigmoid_scores = nn.Sigmoid()
        self.funky_coordinate_space = False
        if self.funky_coordinate_space:
            self.embedding_coordinates = nn.Linear(
                in_dim_node, 3
            )  # node feat is an integer
        self.M = M  # number of points up to connect to
        self.embedding_h = nn.Linear(in_dim_node, hidden_dim)
        self.SWIN3D_Blocks = SWIN3D_Blocks(
            n_layers,
            hidden_dim,
            num_heads,
            dropout,
            self.layer_norm,
            self.batch_norm,
            self.residual,
            possible_empty=True,
        )

    def forward(self, g, h, c):
        object = g.ndata["object"]
        # 1) Find coordinates and score for the graph
        # embedding to calculate the coordinates in the embedding space #! this could also be kept to the original coordinates
        if self.funky_coordinate_space:
            s_l = self.embedding_coordinates(h)
        else:
            s_l = c
        h = self.embedding_h(h)
        scores = torch.rand(h.shape[0]).to(h.device)

        # 2) Do knn down graph
        g.ndata["s_l"] = s_l
        g = knn_per_graph(
            g, s_l, 16
        )  #! if these are learnt then they should be added to the gradients, they are not at the moment

        # 3) Message passing on the down graph SWIN3D_Blocks
        h = self.SWIN3D_Blocks(g, h)

        g.ndata["scores"] = scores
        g.ndata["object"] = object
        g.ndata["s_l"] = s_l
        g.ndata["h"] = h

        # calculate loss of score
        # g.update_all(self.send_scores, self.find_up)
        # loss_ud = self.find_up.loss_ud
        return g, h


class MP_up(nn.Module):
    """MAIN block
    3) Message passing on flat graph SWIN3D_Blocks

    """

    def __init__(
        self,
        in_dim_node,
        hidden_dim,
        num_heads,
        layer_norm,
        batch_norm,
        residual,
        dropout,
        M,
        k_in,
        n_layers,
    ):
        super().__init__()
        self.k = k_in
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.residual = residual

        self.SWIN3D_Blocks = SWIN3D_Blocks(
            n_layers,
            hidden_dim,
            num_heads,
            dropout,
            self.layer_norm,
            self.batch_norm,
            self.residual,
            possible_empty=True,
        )

    def forward(self, g, h):
        # 3) Message passing on the down graph SWIN3D_Blocks
        h = self.SWIN3D_Blocks(g, h)
        g.ndata["h"] = h
        return g, h


class Downsample_block(nn.Module):
    """
    4) Downsample:
            - find down points (currently random score)
            - find neigh of from up to down
    """

    def __init__(
        self,
        hidden_dim,
        M,
    ):
        super().__init__()

        self.Downsample = Downsample_maxpull(hidden_dim, M)

    def forward(self, g):
        # 4) Downsample:
        features, up_points, new_graphs_up, i, j = self.Downsample(g)
        return features, up_points, new_graphs_up, i, j


class SWIN3D_Blocks(nn.Module):
    """Point 3)
    Just multiple blocks of sparse attention over the down graph
    """

    def __init__(
        self,
        n_layers,
        hidden_dim,
        num_heads,
        layer_norm,
        batch_norm,
        residual,
        dropout,
        possible_empty=True,
    ):
        super().__init__()
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.residual = residual
        self.layers_message_passing = nn.ModuleList(
            [
                GraphTransformerLayer(
                    hidden_dim,
                    hidden_dim,
                    num_heads,
                    dropout,
                    self.layer_norm,
                    self.batch_norm,
                    self.residual,
                    possible_empty=True,
                )
                for zz in range(n_layers)
            ]
        )

    def forward(self, g, h):
        g.ndata["h"] = h
        for ii, conv in enumerate(self.layers_message_passing):
            h = conv(g, h)
        return h


class Downsample(nn.Module):
    """Point 4)
    - find up points
    - find neigh of from down to up
    """

    def __init__(self, hidden_dim, M):
        super().__init__()
        self.M = M
        self.embedding_features_to_att = nn.Linear(hidden_dim + 3, hidden_dim)
        self.MLP_difs = MLP_difs(hidden_dim, hidden_dim)

    def forward(self, g):
        features = g.ndata["h"]
        list_graphs = dgl.unbatch(g)
        s_l = g.ndata["s_l"]
        graphs_UD = []
        graphs_U = []
        up_points = []
        for i in range(0, len(list_graphs)):
            graph_i = list_graphs[i]
            number_nodes_graph = graph_i.number_of_nodes()

            # find up nodes
            s_l_i = graph_i.ndata["s_l"]
            scores_i = graph_i.ndata["scores"].view(-1)
            device = scores_i.device
            number_up = np.floor(number_nodes_graph * 0.25).astype(int)
            up_points_i_index = torch.flip(torch.sort(scores_i, dim=0)[1], [0])[
                0:number_up
            ]
            up_points_i = torch.zeros_like(scores_i)
            up_points_i[up_points_i_index.long()] = 1
            up_points_i = up_points_i.bool()

            up_points.append(up_points_i)

            # connect down to up
            number_up_points_i = torch.sum(up_points_i)
            if number_up_points_i > 5:
                M_i = 5
            else:
                M_i = number_up_points_i
            nodes = torch.range(start=0, end=number_nodes_graph - 1, step=1).to(device)
            nodes_up = nodes[up_points_i]
            nodes_down = nodes[~up_points_i]

            neigh_indices, neigh_dist_sq = torch_cmspepr.select_knn_directional(
                s_l_i[~up_points_i], s_l_i[up_points_i], M_i
            )
            j = nodes_up[neigh_indices]
            j = j.view(-1)
            i = torch.tile(nodes_down.view(-1, 1), (1, M_i)).reshape(-1)

            g_i = dgl.graph((i.long(), j.long()), num_nodes=number_nodes_graph).to(
                device
            )
            g_i.ndata["h"] = graph_i.ndata["h"]
            g_i.ndata["s_l"] = graph_i.ndata["s_l"]
            g_i.ndata["object"] = graph_i.ndata["object"]
            # find index in original numbering
            graphs_UD.append(g_i)
            # use this way if no message passing between nodes
            # edge_index = torch_cmspepr.knn_graph(s_l_i[up_points_i], k=7)
            # graph_up = dgl.graph(
            #     (edge_index[0], edge_index[1]), num_nodes=len(nodes_up)
            # ).to(device)
            graph_up = dgl.DGLGraph().to(device)
            graph_up.add_nodes(len(nodes_up))
            graph_up.ndata["object"] = g_i.ndata["object"][up_points_i]
            graph_up.ndata["s_l"] = g_i.ndata["s_l"][up_points_i]
            graphs_U.append(graph_up)

        graphs_UD = dgl.batch(graphs_UD)
        i, j = graphs_UD.edges()
        graphs_U = dgl.batch(graphs_U)
        # naive way of giving the coordinates gradients
        features = torch.cat((features, s_l), dim=1)
        features = self.embedding_features_to_att(features)

        # do attention in g connected to up, this features have only been updated for points that have neighbourgs pointing to them: up-points
        features = self.MLP_difs(graphs_UD, features)

        up_points = torch.concat(up_points, dim=0).view(-1)

        return features, up_points, graphs_U, i, j


def knn_per_graph(g, sl, k):
    """Build knn for each graph in the batch

    Args:
        g (_type_): original batch of dgl graphs
        sl (_type_): coordinates
        k (_type_): number of neighbours

    Returns:
        _type_: updates batch of dgl graphs with edges
    """
    graphs_list = dgl.unbatch(g)
    node_counter = 0
    new_graphs = []
    for graph in graphs_list:
        non = graph.number_of_nodes()
        sls_graph = sl[node_counter : node_counter + non]
        edge_index = torch_cmspepr.knn_graph(sls_graph, k=k)
        new_graph = dgl.graph(
            (edge_index[0], edge_index[1]), num_nodes=sls_graph.shape[0]
        )
        new_graphs.append(new_graph)
        node_counter = node_counter + non
    return dgl.batch(new_graphs)


##############
# some less important stuff
##############
class MLPReadout(nn.Module):
    """Final FCC layer for node classification"""

    def __init__(self, input_dim, output_dim, L=2):  # L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [
            nn.Linear(input_dim // 2**l, input_dim // 2 ** (l + 1), bias=True)
            for l in range(L)
        ]
        list_FC_layers.append(nn.Linear(input_dim // 2**L, output_dim, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.drop_out = nn.Dropout(0.1)
        self.L = L

    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
            y = self.drop_out(y)
        y = self.FC_layers[self.L](y)
        return y


class SendScoresMessage(nn.Module):
    """
    Compute the input feature from neighbors
    """

    def __init__(self):
        super(SendScoresMessage, self).__init__()

    def forward(self, edges):
        score_neigh = edges.src["scores"]
        same_object = edges.dst["object"] == edges.src["object"]
        return {"score_neigh": score_neigh.view(-1), "same_object": same_object}


class FindUpPoints(nn.Module):
    """
    Feature aggregation in a DGL graph
    """

    def __init__(self):
        super(FindUpPoints, self).__init__()
        self.loss_ud = 0

    def forward(self, nodes):
        same_object = nodes.mailbox["same_object"]
        scores_neigh = nodes.mailbox["score_neigh"]
        # loss per neighbourhood of same object as src node
        values_max, index = torch.max(scores_neigh * same_object, dim=1)
        number_points_same_object = torch.sum(same_object, dim=1)
        # print("number_points_same_object", number_points_same_object)
        # print("values_max", values_max)
        loss_u = 1 - values_max
        # loss_d = (
        #     1 / number_points_same_object * torch.sum(scores_neigh * same_object, dim=1)
        # )
        sum_same_object = torch.sum(scores_neigh * same_object, dim=1) - values_max
        # print("sum_same_object", sum_same_object)
        mask_ = number_points_same_object > 0
        if torch.sum(mask_) > 0:
            loss_d = 1 / number_points_same_object[mask_] * sum_same_object[mask_]
            # per neigh measure
            # print("loss_u", loss_u)
            # print("loss_d", torch.mean(loss_d))
            loss_total = loss_u.clone()
            # this takes into account some points not having neigh of the same class
            loss_total[mask_] = loss_u[mask_] + loss_d
            total_loss_ud = torch.mean(loss_total)
            # print("loss ud normal", total_loss_ud)
        else:
            total_loss_ud = torch.mean(loss_u)
            # print("loss ud no neigh", total_loss_ud)
        # print("total_loss_ud", total_loss_ud)
        self.loss_ud = total_loss_ud
        fake_feature = torch.sum(scores_neigh, dim=1)
        return {"new_feat": fake_feature}


class Downsample_maxpull(nn.Module):
    """Point 4)
    - find up points
    - find neigh of from down to up
    Maxpulling as in the Point transformer as well
    """

    def __init__(self, hidden_dim, M):
        super().__init__()
        self.M = M
        self.embedding_features_to_att = nn.Linear(hidden_dim + 3, hidden_dim)
        self.MLP_difs = MLP_difs_maxpool(hidden_dim, hidden_dim)

    def forward(self, g):
        features = g.ndata["h"]
        list_graphs = dgl.unbatch(g)
        s_l = g.ndata["s_l"]
        graphs_UD = []
        graphs_U = []
        up_points = []
        for i in range(0, len(list_graphs)):
            graph_i = list_graphs[i]
            number_nodes_graph = graph_i.number_of_nodes()

            # find up nodes
            s_l_i = graph_i.ndata["s_l"]
            scores_i = graph_i.ndata["scores"].view(-1)
            device = scores_i.device
            number_up = np.floor(number_nodes_graph * 0.25).astype(int)
            # up_points_i_index = torch.flip(torch.sort(scores_i, dim=0)[1], [0])[
            #     0:number_up
            # ]
            # Use farthest point sampling
            n_o = torch.cuda.IntTensor([number_up])
            o = torch.cuda.IntTensor([number_nodes_graph])
            print("shapes are", s_l_i.shape, n_o, o)
            up_points_i_index = pointops.farthest_point_sampling(s_l_i, o, n_o)
            print(up_points_i_index)
            up_points_i = torch.zeros_like(scores_i)
            up_points_i[up_points_i_index.long()] = 1
            up_points_i = up_points_i.bool()

            up_points.append(up_points_i)

            # connect down to up
            number_up_points_i = torch.sum(up_points_i)
            if number_up_points_i > 5:
                M_i = 5
            else:
                M_i = number_up_points_i
            nodes = torch.range(start=0, end=number_nodes_graph - 1, step=1).to(device)
            nodes_up = nodes[up_points_i]
            nodes_down = nodes[~up_points_i]

            neigh_indices, neigh_dist_sq = torch_cmspepr.select_knn_directional(
                s_l_i[~up_points_i], s_l_i[up_points_i], M_i
            )
            j = nodes_up[neigh_indices]
            j = j.view(-1)
            i = torch.tile(nodes_down.view(-1, 1), (1, M_i)).reshape(-1)

            g_i = dgl.graph((i.long(), j.long()), num_nodes=number_nodes_graph).to(
                device
            )
            g_i.ndata["h"] = graph_i.ndata["h"]
            g_i.ndata["s_l"] = graph_i.ndata["s_l"]
            g_i.ndata["object"] = graph_i.ndata["object"]
            # find index in original numbering
            graphs_UD.append(g_i)
            # use this way if no message passing between nodes
            # edge_index = torch_cmspepr.knn_graph(s_l_i[up_points_i], k=7)
            # graph_up = dgl.graph(
            #     (edge_index[0], edge_index[1]), num_nodes=len(nodes_up)
            # ).to(device)
            graph_up = dgl.DGLGraph().to(device)
            graph_up.add_nodes(len(nodes_up))
            graph_up.ndata["object"] = g_i.ndata["object"][up_points_i]
            graph_up.ndata["s_l"] = g_i.ndata["s_l"][up_points_i]
            graphs_U.append(graph_up)

        graphs_UD = dgl.batch(graphs_UD)
        i, j = graphs_UD.edges()
        graphs_U = dgl.batch(graphs_U)
        # naive way of giving the coordinates gradients
        features = torch.cat((features, s_l), dim=1)
        features = self.embedding_features_to_att(features)

        # do attention in g connected to up, this features have only been updated for points that have neighbourgs pointing to them: up-points
        features = self.MLP_difs(graphs_UD, features)

        up_points = torch.concat(up_points, dim=0).view(-1)

        return features, up_points, graphs_U, i, j
