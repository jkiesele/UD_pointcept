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

"""
    Fancy Pulling
    
"""


def knn_per_graph(g, sl, k):
    graphs_list = dgl.unbatch(g)
    node_counter = 0
    new_graphs = []
    for graph in graphs_list:
        non = graph.number_of_nodes()
        sls_graph = sl[node_counter : node_counter + non]
        # new_graph = dgl.knn_graph(sls_graph, k, exclude_self=True)
        # tic = time.time()
        edge_index = torch_cmspepr.knn_graph(
            sls_graph, k=k
        )  # no need to split by batch as we are looping through instances
        # toc = time.time()
        # print("time to build the graph inside attention", toc-tic)
        new_graph = dgl.graph(
            (edge_index[0], edge_index[1]), num_nodes=sls_graph.shape[0]
        )
        new_graphs.append(new_graph)
        node_counter = node_counter + non
    return dgl.batch(new_graphs)


class MLPReadout(nn.Module):
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


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, use_bias, possible_empty):
        super().__init__()
        self.possible_empty = possible_empty
        self.out_dim = out_dim
        self.num_heads = num_heads

        if use_bias:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        else:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=False)

    def propagate_attention(self, g):
        # Compute attention score
        g.apply_edges(src_dot_dst("K_h", "Q_h", "score"))  # , edges)
        g.apply_edges(scaled_exp("score", np.sqrt(self.out_dim)))
        # Send weighted values to target nodes
        eids = g.edges()
        g.send_and_recv(
            eids,
            fn.u_mul_e("V_h", "score", "V_h"),
            fn.sum("V_h", "wV"),  # deprecated in dgl 1.0.1
        )
        # print(g.edata["score"].shape)
        g.send_and_recv(
            eids, fn.copy_e("score", "score"), fn.sum("score", "z")
        )  # copy_e deprecated in dgl 1.0.1
        # print("wV ", g.ndata["wV"])

    def forward(self, g, h):

        Q_h = self.Q(h)
        K_h = self.K(h)
        V_h = self.V(h)
        # print("Q_h", Q_h)
        # print("K_h", Q_h)
        # print("V_h", Q_h)
        # Reshaping into [num_nodes, num_heads, feat_dim] to
        # get projections for multi-head attention
        g.ndata["Q_h"] = Q_h.view(-1, self.num_heads, self.out_dim)
        g.ndata["K_h"] = K_h.view(-1, self.num_heads, self.out_dim)
        g.ndata["V_h"] = V_h.view(-1, self.num_heads, self.out_dim)
        self.propagate_attention(g)
        if self.possible_empty:
            # print(g.ndata["wV"].shape, g.ndata["z"].shape, g.ndata["z"].device)
            g.ndata["z"] = g.ndata["z"].tile((1, 1, self.out_dim))
            mask_empty = g.ndata["z"] > 0
            head_out = g.ndata["wV"]
            head_out[mask_empty] = head_out[mask_empty] / (g.ndata["z"][mask_empty])
            g.ndata["z"] = g.ndata["z"][:, :, 0].view(
                g.ndata["wV"].shape[0], self.num_heads, 1
            )
        else:
            head_out = g.ndata["wV"] / g.ndata["z"]
        return head_out


def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        return {
            out_field: (edges.src[src_field] * edges.dst[dst_field]).sum(
                -1, keepdim=True
            )
        }

    return func


def scaled_exp(field, scale_constant):
    def func(edges):
        # clamp for softmax numerical stability
        return {field: torch.exp((edges.data[field] / scale_constant).clamp(-5, 5))}

    return func


class GraphTransformerLayer(nn.Module):
    """
    Param:
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        num_heads,
        dropout=0.0,
        layer_norm=False,
        batch_norm=True,
        residual=True,
        use_bias=False,
        possible_empty=False,
    ):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.possible_empty = possible_empty

        self.attention = MultiHeadAttentionLayer(
            in_dim, out_dim // num_heads, num_heads, use_bias, possible_empty
        )

        self.O = nn.Linear(out_dim, out_dim)

        if self.layer_norm:
            self.layer_norm1 = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm1 = nn.BatchNorm1d(out_dim)

        # FFN
        self.FFN_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_layer2 = nn.Linear(out_dim * 2, out_dim)

        if self.layer_norm:
            self.layer_norm2 = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm2 = nn.BatchNorm1d(out_dim)

    def forward(self, g, h):
        h_in1 = h  # for first residual connection
        # print("h in attention", h)
        # multi-head attention out
        attn_out = self.attention(g, h)

        h = attn_out.view(-1, self.out_channels)
        # print("h attention", h)
        h = F.dropout(h, self.dropout, training=self.training)

        h = self.O(h)
        # print("h attention 1", h)
        if self.residual:
            h = h_in1 + h  # residual connection

        if self.layer_norm:
            h = self.layer_norm1(h)

        if self.batch_norm:
            h = self.batch_norm1(h)

        h_in2 = h  # for second residual connection

        # FFN
        h = self.FFN_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_layer2(h)

        if self.residual:
            h = h_in2 + h  # residual connection

        if self.layer_norm:
            h = self.layer_norm2(h)

        if self.batch_norm:
            h = self.batch_norm2(h)
        # print("h attention final", h)
        return h


class Swin3D(nn.Module):
    """
    Param:
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
        self.Downsample = Downsample(hidden_dim, M)
        # self.layers_message_passing = nn.ModuleList(
        #     [
        #         GraphTransformerLayer(
        #             hidden_dim,
        #             hidden_dim,
        #             2,
        #             0.05,
        #             False,
        #             True,
        #             True,
        #             possible_empty=True,
        #         )
        #         for zz in range(n_layers)
        #     ]
        # )

    def forward(self, g, h, c):
        object = g.ndata["object"]
        # embedding to calculate the coordinates in the embedding space #! this could also be kept to the original coordinates
        if self.funky_coordinate_space:
            s_l = self.embedding_coordinates(h)
        else:
            s_l = c
        # do knn and graph attention blocks
        g.ndata["s_l"] = s_l
        g = knn_per_graph(g, s_l, 16)
        h = self.SWIN3D_Blocks(g)
        features = h
        scores = torch.rand(h.shape[0]).to(h.device)

        # do an update to calculate the loss of the neighbourhoods
        g.ndata["features"] = features
        g.ndata["scores"] = scores
        g.ndata["object"] = object
        g.ndata["s_l"] = s_l
        g.ndata["h"] = h
        # g.update_all(self.send_scores, self.find_up)
        # loss_ud = self.find_up.loss_ud

        up_points, new_graphs_up, i, j = self.Downsample(g)
        # list_new = []
        # for zzz in range(0, len(new_graphs_up)):
        #     g_i = new_graphs_up[zzz]
        #     s_li = g_i.ndata["s_l"]
        #     edge_index = torch_cmspepr.knn_graph(
        #         s_li, k=3
        #     )  # no need to split by batch as we are looping through instances
        #     g_i_ = dgl.graph((edge_index[0], edge_index[1]), num_nodes=s_li.shape[0])
        #     g_i_ = dgl.remove_self_loop(g_i_)
        #     list_new.append(g_i_)
        # new_graphs_up = dgl.batch(list_new)
        #
        #
        # features_up = features[up_points]
        # for ii, conv in enumerate(self.layers_message_passing):
        #     features_up = conv(new_graphs_up, features_up)
        # features[up_points] = features_up
        return features, up_points, new_graphs_up, i, j, s_l


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
        #! maybe this should be the distance to simplify at first ?
        #! also this could be a softmax as Jan was saying
        dif = edges.src["features"] - edges.dst["features"]
        att_weight = self.MLP(dif)
        att_weight = torch.exp(-att_weight)
        feature = att_weight * edges.dst["features"]
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


class SWIN3D_Blocks(nn.Module):
    """
    Param:
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

    def forward(self, g):
        for ii, conv in enumerate(self.layers_message_passing):
            h = conv(g, h)

        return h


class Downsample(nn.Module):
    """
    Param:
    """

    def __init__(self, hidden_dim, M):
        super().__init__()
        self.M = M  # number of points up to connect to
        self.embedding_features_to_att = nn.Linear(hidden_dim + 3, hidden_dim)

        # self.attention_layer = GraphTransformerLayer(
        #     hidden_dim,
        #     hidden_dim,
        #     num_heads,
        #     dropout,
        #     self.layer_norm,
        #     self.batch_norm,
        #     self.residual,
        #     possible_empty=True,
        # )
        self.MLP_difs = MLP_difs(hidden_dim, hidden_dim)

    def forward(self, g):
        list_graphs = dgl.unbatch(g)
        s_l = g.ndata["s_l"]
        new_graphs = []
        new_graphs_up = []
        up_points = []
        for i in range(0, len(list_graphs)):
            graph_i = list_graphs[i]
            number_nodes_graph = graph_i.number_of_nodes()
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
            number_up_points_i = torch.sum(up_points_i)
            if number_up_points_i > 5:
                M_i = 5
            else:
                M_i = number_up_points_i
            nodes = torch.range(start=0, end=number_nodes_graph - 1, step=1).to(device)
            nodes_up = nodes[up_points_i]
            nodes_down = nodes[~up_points_i]
            dist_to_up = torch.cdist(s_l_i[~up_points_i], s_l_i[up_points_i])
            # take the smallest distance and take first M
            indices_connect = torch.topk(-dist_to_up, k=M_i, dim=1)[1]
            j = nodes_up[indices_connect]
            j = j.view(-1)
            i = torch.tile(nodes_down.view(-1, 1), (1, M_i)).reshape(-1)

            g_i = dgl.graph((i.long(), j.long()), num_nodes=number_nodes_graph).to(
                device
            )
            g_i.ndata["h"] = graph_i.ndata["h"]
            g_i.ndata["s_l"] = graph_i.ndata["s_l"]
            g_i.ndata["object"] = graph_i.ndata["object"]
            # find index in original numbering
            new_graphs.append(g_i)
            edge_index = torch_cmspepr.knn_graph(s_l_i[up_points_i], k=7)
            graph_up = dgl.graph(
                (edge_index[0], edge_index[1]), num_nodes=len(nodes_up)
            ).to(device)
            # use this way if no message passing between nodes
            # graph_up = dgl.DGLGraph().to(device)
            # graph_up.add_nodes(len(nodes_up))
            graph_up.ndata["object"] = g_i.ndata["object"][up_points_i]
            graph_up.ndata["s_l"] = g_i.ndata["s_l"][up_points_i]

            # add knn in graphs up to do message passing after
            new_graphs_up.append(graph_up)

        g_connected_to_up = dgl.batch(new_graphs)
        i, j = g_connected_to_up.edges()
        new_graphs_up = dgl.batch(new_graphs_up)
        # naive way of giving the coordinates gradients
        features = torch.cat((features, s_l), dim=1)
        features = self.embedding_features_to_att(features)
        # do attention in g connected to up, this features have only been updated for points that have neighbourgs pointing to them: up-points
        features = self.MLP_difs(g_connected_to_up, features)
        # features = self.attention_layer(g_connected_to_up, features)
        up_points = torch.concat(up_points, dim=0).view(-1)

        return up_points, new_graphs_up, i, j
