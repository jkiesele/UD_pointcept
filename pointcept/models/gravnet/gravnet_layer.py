import torch
from torch import Tensor
from torch.nn import Linear
from torch_scatter import scatter
from torch_geometric.nn.conv import MessagePassing
import torch.nn as nn
import dgl
import dgl.function as fn
import numpy as np
from dgl.nn import EdgeWeightNorm
import torch_cmspepr
from torch_scatter import scatter_min, scatter_max, scatter_mean, scatter_add
import dgl
from typing import Optional, Union
from torch_geometric.typing import OptTensor, PairTensor, PairOptTensor
from pointcept.models.gravnet.plotting_tools import PlotCoordinates


class GravNetConv_dgl(nn.Module):
    """Implementation of gravnet in dgl (the obsolete version had a mix of dgl and pytorch geometric)"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        space_dimensions: int,
        propagate_dimensions: int,
        k: int,
        num_workers: int = 1,
        weird_batchnom=False,
    ):
        super(GravNetConv_dgl, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k
        self.num_workers = num_workers
        self.lin_s = Linear(in_channels, space_dimensions, bias=False)
        self.lin_h = Linear(in_channels, propagate_dimensions)
        self.lin = Linear(in_channels + 2 * propagate_dimensions, out_channels)
        self.norm = EdgeWeightNorm(norm="right", eps=0.0)
        self.edgedistancespassing = EdgeDistancesPassing()
        self.meanmaxaggregation = MeanMax_aggregation()

    def reset_parameters(self):
        self.lin_s.reset_parameters()
        self.lin_h.reset_parameters()

    def forward(self, g, x: Tensor, original_coord: Tensor, batch: OptTensor = None):

        h_l: Tensor = self.lin_h(x)  #! input_feature_transform
        s_l: Tensor = self.lin_s(x)
        graph = knn_per_graph(g, s_l, self.k)
        graph.ndata["s_l"] = s_l
        graph.ndata["h_l"] = h_l
        row = graph.edges()[0]
        col = graph.edges()[1]
        edge_index = torch.stack([row, col], dim=0)

        edge_weight = torch.sqrt(
            (s_l[edge_index[0]] - s_l[edge_index[1]]).pow(2).sum(-1)
        )
        graph.edata["potential"] = torch.exp(-torch.square(edge_weight))

        #! AverageDistanceRegularizer (currently not used)
        # dist = edge_weight
        # dist = torch.sqrt(dist + 1e-6)
        # graph.edata["dist"] = dist
        # graph.ndata["ones"] = torch.ones_like(s_l)
        # # average dist per node and divide by the number of neighbourgs
        # graph.update_all(fn.u_mul_e("ones", "dist", "m"), fn.mean("m", "dist"))
        # avdist = graph.ndata["dist"]
        # loss_regularizing_neig = 1e-2 * torch.mean(torch.square(avdist - 0.5))
        # # propagate_type: (x: OptPairTensor, edge_weight: OptTensor)

        # #! LLRegulariseGravNetSpace
        # graph.edata["_edge_w"] = dist
        # graph.update_all(fn.copy_e("_edge_w", "m"), fn.sum("m", "in_weight"))
        # degs = graph.dstdata["in_weight"] + 1e-4
        # graph.dstdata["_dst_in_w"] = 1 / degs
        # graph.apply_edges(
        #     lambda e: {"_norm_edge_weights": e.dst["_dst_in_w"] * e.data["_edge_w"]}
        # )
        # dist = graph.edata["_norm_edge_weights"]

        # gndist = (
        #     (original_coord[edge_index[0]] - original_coord[edge_index[1]])
        #     .pow(2)
        #     .sum(-1)
        # )

        # gndist = torch.sqrt(gndist + 1e-6)
        # graph.edata["_edge_w_gndist"] = dist
        # graph.update_all(fn.copy_e("_edge_w_gndist", "m"), fn.sum("m", "in_weight"))
        # degs = graph.dstdata["in_weight"] + 1e-4
        # graph.dstdata["_dst_in_w"] = 1 / degs
        # graph.apply_edges(
        #     lambda e: {"_norm_edge_weights_gn": e.dst["_dst_in_w"] * e.data["_edge_w"]}
        # )
        # gndist = graph.edata["_norm_edge_weights_gn"]
        # loss_llregulariser = 0.1 * torch.mean(torch.square(dist - gndist))
        # print(torch.square(dist - gndist))
        #! this is the output_feature_transform
        graph.update_all(self.edgedistancespassing, self.meanmaxaggregation)
        out = graph.ndata["h_updated"]
        #! not sure this cat is exactly the same that is happening in the RaggedGravNet but they also cat
        out = self.lin(torch.cat([out, x], dim=-1))
        return (out, graph, s_l)


class EdgeDistancesPassing(nn.Module):
    """
    Compute the input feature from neighbors
    """

    def __init__(self):
        super(EdgeDistancesPassing, self).__init__()

    def forward(self, edges):
        info_to_agg = edges.data["potential"].view(-1, 1) * edges.src["h_l"]

        return {"feat_dist": info_to_agg}


class MeanMax_aggregation(nn.Module):
    """
    Feature aggregation in a DGL graph
    """

    def __init__(self):
        super(MeanMax_aggregation, self).__init__()

    def forward(self, nodes):
        mean_agg = torch.mean(nodes.mailbox["feat_dist"], dim=1)
        max_agg = torch.max(nodes.mailbox["feat_dist"], dim=1)[0]
        out = torch.cat([mean_agg, max_agg], dim=-1)
        return {"h_updated": out}


def knn_per_graph(g, sl, k):
    graphs_list = dgl.unbatch(g)
    node_counter = 0
    new_graphs = []
    for graph in graphs_list:
        non = graph.number_of_nodes()
        sls_graph = sl[node_counter : node_counter + non]
        # new_graph = dgl.knn_graph(sls_graph, k, exclude_self=True)
        edge_index = torch_cmspepr.knn_graph(sls_graph, k=k)
        new_graph = dgl.graph(
            (edge_index[0], edge_index[1]), num_nodes=sls_graph.shape[0]
        )
        new_graph = dgl.remove_self_loop(new_graph)
        new_graphs.append(new_graph)
        node_counter = node_counter + non
    return dgl.batch(new_graphs)


class WeirdBatchNorm(nn.Module):
    def __init__(self, n_neurons, eps=1e-5):

        super(WeirdBatchNorm, self).__init__()

        # stores number of neuros
        self.n_neurons = n_neurons

        # initinalize batch normalization parameters
        self.gamma = nn.Parameter(torch.ones(self.n_neurons))
        self.beta = nn.Parameter(torch.zeros(self.n_neurons))
        self.mean = torch.zeros(self.n_neurons)
        self.den = torch.ones(self.n_neurons)
        self.viscosity = 0.999999
        self.epsilon = eps
        self.fluidity_decay = 0.01
        self.max_viscosity = 1

    def forward(self, input):
        x = input.detach()
        mu = x.mean(dim=0)
        var = x.var(dim=0, unbiased=False)

        mu_update = self._calc_update(self.mean, mu)
        self.mean = mu_update
        var_update = self._calc_update(self.den, var)
        self.den = var_update

        # normalization
        center_input = x - self.mean
        denominator = self.den + self.epsilon
        denominator = denominator.sqrt()

        in_hat = center_input / denominator

        self._update_viscosity()

        # scale and shift
        out = self.gamma * in_hat + self.beta

        return out

    def _calc_update(self, old, new):
        delta = new - old.to(new.device)
        update = old.to(new.device) + (1 - self.viscosity) * delta.to(new.device)
        update = update.to(new.device)
        return update

    def _update_viscosity(self):
        if self.fluidity_decay > 0:
            newvisc = (
                self.viscosity
                + (self.max_viscosity - self.viscosity) * self.fluidity_decay
            )
            self.viscosity = newvisc


def scatter_count(input: torch.Tensor):
    """
    Returns ordered counts over an index array
    Example:
    >>> scatter_count(torch.Tensor([0, 0, 0, 1, 1, 2, 2])) # input
    >>> [3, 2, 2]
    Index assumptions work like in torch_scatter, so:
    >>> scatter_count(torch.Tensor([1, 1, 1, 2, 2, 4, 4]))
    >>> tensor([0, 3, 2, 0, 2])
    """
    return scatter_add(torch.ones_like(input, dtype=torch.long), input.long())


def obtain_batch_numbers(x, g):
    dev = x.device
    graphs_eval = dgl.unbatch(g)
    number_graphs = len(graphs_eval)
    batch_numbers = []
    for index in range(0, number_graphs):
        gj = graphs_eval[index]
        num_nodes = gj.number_of_nodes()
        batch_numbers.append(index * torch.ones(num_nodes).to(dev))
        # num_nodes = gj.number_of_nodes()

    batch = torch.cat(batch_numbers, dim=0)
    return batch


class GravNetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 96,
        space_dimensions: int = 3,
        propagate_dimensions: int = 22,
        k: int = 40,
        # batchnorm: bool = True
        weird_batchnom=False,
    ):
        super(GravNetBlock, self).__init__()
        self.d_shape = 32
        out_channels = self.d_shape
        if weird_batchnom:
            self.batchnorm_gravnet1 = WeirdBatchNorm(self.d_shape)
        else:
            self.batchnorm_gravnet1 = nn.BatchNorm1d(self.d_shape, momentum=0.01)
        propagate_dimensions = self.d_shape
        self.gravnet_layer = GravNetConv_dgl(
            self.d_shape,
            out_channels,
            space_dimensions,
            propagate_dimensions,
            k,
            weird_batchnom,
        )

        self.post_gravnet = nn.Sequential(
            nn.Linear(
                out_channels + space_dimensions + self.d_shape, self.d_shape
            ),  #! Dense 3
            nn.ELU(),
            nn.Linear(self.d_shape, self.d_shape),  #! Dense 4
            nn.ELU(),
        )
        self.pre_gravnet = nn.Sequential(
            nn.Linear(in_channels, self.d_shape),  #! Dense 1
            nn.ELU(),
            nn.Linear(self.d_shape, self.d_shape),  #! Dense 2
            nn.ELU(),
        )
        # self.output = nn.Sequential(nn.Linear(self.d_shape, self.d_shape), nn.ELU())

        # init_weights(self.output)
        init_weights(self.post_gravnet)
        init_weights(self.pre_gravnet)

        if weird_batchnom:
            self.batchnorm_gravnet2 = WeirdBatchNorm(self.d_shape)
        else:
            self.batchnorm_gravnet2 = nn.BatchNorm1d(self.d_shape, momentum=0.01)

        self.step = 0

    def forward(
        self,
        g,
        x: Tensor,
        batch: Tensor,
        original_coords: Tensor,
        step_count,
        num_layer,
    ) -> Tensor:
        x = self.pre_gravnet(x)
        x = self.batchnorm_gravnet1(x)
        x_input = x
        xgn, graph, gncoords = self.gravnet_layer(
            g, x, original_coords, batch
        ) # loss_regularizing_neig, ll_r
        g.ndata["gncoords"] = gncoords
        if (step_count % 50) == 0 and self.training:
            PlotCoordinates(g, path="gravnet_coord", num_layer=str(num_layer))
            self.step += 1
        # gncoords = gncoords.detach()
        x = torch.cat((xgn, gncoords, x_input), dim=1)
        x = self.post_gravnet(x)
        x = self.batchnorm_gravnet2(x)  #! batchnorm 2
        # x = global_exchange(x, batch)
        # x = self.output(x)
        return x, graph #, loss_regularizing_neig, ll_r


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.00)
