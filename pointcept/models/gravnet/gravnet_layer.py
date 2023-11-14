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


class GravNetConv(MessagePassing):
    """The GravNet operator from the `"Learning Representations of Irregular
    Particle-detector Geometry with Distance-weighted Graph
    Networks" <https://arxiv.org/abs/1902.07987>`_ paper, where the graph is
    dynamically constructed using nearest neighbors.
    The neighbors are constructed in a learnable low-dimensional projection of
    the feature space.
    A second projection of the input feature space is then propagated from the
    neighbors to each vertex using distance weights that are derived by
    applying a Gaussian function to the distances.
    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        space_dimensions (int): The dimensionality of the space used to
           construct the neighbors; referred to as :math:`S` in the paper.
        propagate_dimensions (int): The number of features to be propagated
           between the vertices; referred to as :math:`F_{\textrm{LR}}` in the
           paper.
        k (int): The number of nearest neighbors.
        num_workers (int): Number of workers to use for k-NN computation.
            Has no effect in case :obj:`batch` is not :obj:`None`, or the input
            lies on the GPU. (default: :obj:`1`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        space_dimensions: int,
        propagate_dimensions: int,
        k: int,
        num_workers: int = 1,
        weird_batchnom=False,
        **kwargs
    ):
        super(GravNetConv, self).__init__(flow="target_to_source", **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k
        self.num_workers = num_workers
        if weird_batchnom:
            self.batchnorm_gravconv = WeirdBatchNorm(out_channels)
        else:
            self.batchnorm_gravconv = nn.BatchNorm1d(out_channels)
        self.lin_s = Linear(in_channels, space_dimensions, bias=False)
        # self.lin_s.weight.data.copy_(torch.eye(space_dimensions, in_channels))
        # torch.nn.init.xavier_uniform_(self.lin_s.weight, gain=0.001)
        self.lin_h = Linear(in_channels, propagate_dimensions)
        self.lin = Linear(in_channels + 2 * propagate_dimensions, out_channels)
        self.norm = EdgeWeightNorm(norm="right", eps=0.0)
        # self.reset_parameters()

    def reset_parameters(self):
        self.lin_s.reset_parameters()
        self.lin_h.reset_parameters()
        # self.lin.reset_parameters()

    def forward(
        self, g, x: Tensor, original_coord: Tensor, batch: OptTensor = None
    ) -> Tensor:
        """"""

        assert x.dim() == 2, "Static graphs not supported in `GravNetConv`."

        b: OptTensor = None
        if isinstance(batch, Tensor):
            b = batch
        h_l: Tensor = self.lin_h(x)  #! input_feature_transform

        # print("weights input_feature_transform", self.lin_h.weight.data)
        # print("bias input_feature_transform", self.lin_h.bias.data)

        s_l: Tensor = self.lin_s(x)
        # print("weights input_spatial_transform", self.lin_s.weight.data)
        # print("bias input_spatial_transform", self.lin_s.bias.data)
        s_l = s_l  ##+ original_coords
        # print("coordinates INPUTS TO FIRST LAYER")
        # print(s_l)

        graph = knn_per_graph(g, s_l, self.k)
        graph.ndata["s_l"] = s_l
        row = graph.edges()[0]
        col = graph.edges()[1]
        edge_index = torch.stack([row, col], dim=0)

        edge_weight = (s_l[edge_index[0]] - s_l[edge_index[1]]).pow(2).sum(-1)

        # print("distancesq distancesq distancesq")
        # print(edge_weight)
        # edge_weight = edge_weight + 1e-5
        #! normalized edge weight
        # print("edge weight", edge_weight)
        # edge_weight = self.norm(graph, edge_weight)
        # print("normalized edge weight", edge_weight)
        # edge_weight = torch.exp(-10.0 * edge_weight)  # 10 gives a better spread

        #! AverageDistanceRegularizer
        dist = edge_weight
        dist = torch.sqrt(dist + 1e-6)
        graph.edata["dist"] = dist
        graph.ndata["ones"] = torch.ones_like(s_l)
        # average dist per node and divide by the number of neighbourgs
        graph.update_all(fn.u_mul_e("ones", "dist", "m"), fn.mean("m", "dist"))
        avdist = graph.ndata["dist"]
        loss_regularizing_neig = 1e-2 * torch.mean(torch.square(avdist - 0.5))
        # propagate_type: (x: OptPairTensor, edge_weight: OptTensor)

        #! LLRegulariseGravNetSpace
        graph.edata["_edge_w"] = dist
        graph.update_all(fn.copy_e("_edge_w", "m"), fn.sum("m", "in_weight"))
        degs = graph.dstdata["in_weight"] + 1e-4
        graph.dstdata["_dst_in_w"] = 1 / degs
        graph.apply_edges(
            lambda e: {"_norm_edge_weights": e.dst["_dst_in_w"] * e.data["_edge_w"]}
        )
        dist = graph.edata["_norm_edge_weights"]

        gndist = (
            (original_coord[edge_index[0]] - original_coord[edge_index[1]])
            .pow(2)
            .sum(-1)
        )

        gndist = torch.sqrt(gndist + 1e-6)
        graph.edata["_edge_w_gndist"] = dist
        graph.update_all(fn.copy_e("_edge_w_gndist", "m"), fn.sum("m", "in_weight"))
        degs = graph.dstdata["in_weight"] + 1e-4
        graph.dstdata["_dst_in_w"] = 1 / degs
        graph.apply_edges(
            lambda e: {"_norm_edge_weights_gn": e.dst["_dst_in_w"] * e.data["_edge_w"]}
        )
        gndist = graph.edata["_norm_edge_weights_gn"]
        loss_llregulariser = 0.1 * torch.mean(torch.square(dist - gndist))
        # print(torch.square(dist - gndist))
        #! this is the output_feature_transform

        out = self.propagate(
            edge_index,
            x=[h_l, None],
            edge_weight=edge_weight,
            size=(s_l.size(0), s_l.size(0)),
        )
        # print("outfeats", out)
        #! not sure this cat is exactly the same that is happening in the RaggedGravNet but they also cat
        out = self.lin(torch.cat([out, x], dim=-1))
        # out = self.batchnorm_gravconv(out)
        return (
            out,
            graph,
            s_l,
            loss_regularizing_neig,
            loss_llregulariser,
        )

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return x_j * edge_weight.unsqueeze(1)

    def aggregate(
        self, inputs: Tensor, index: Tensor, dim_size: Optional[int] = None
    ) -> Tensor:

        out_mean = scatter(
            inputs, index, dim=self.node_dim, dim_size=dim_size, reduce="mean"
        )

        out_max = scatter(
            inputs, index, dim=self.node_dim, dim_size=dim_size, reduce="max"
        )
        return torch.cat([out_mean, out_max], dim=-1)

    def __repr__(self):
        return "{}({}, {}, k={})".format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.k
        )


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
        self.gravnet_layer = GravNetConv(
            self.d_shape,
            out_channels,
            space_dimensions,
            propagate_dimensions,
            k,
            weird_batchnom,
        ).jittable()

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
        self.output = nn.Sequential(nn.Linear(self.d_shape, self.d_shape), nn.ELU())

        init_weights(self.output)
        init_weights(self.post_gravnet)
        init_weights(self.pre_gravnet)

        if weird_batchnom:
            self.batchnorm_gravnet2 = WeirdBatchNorm(self.d_shape)
        else:
            self.batchnorm_gravnet2 = nn.BatchNorm1d(self.d_shape, momentum=0.01)

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
        xgn, graph, gncoords, loss_regularizing_neig, ll_r = self.gravnet_layer(
            g, x, original_coords, batch
        )
        g.ndata["gncoords"] = gncoords
        if step_count % 50:
            PlotCoordinates(g, path="gravnet_coord", num_layer=str(num_layer))
        # gncoords = gncoords.detach()
        x = torch.cat((xgn, gncoords, x_input), dim=1)
        x = self.post_gravnet(x)
        x = self.batchnorm_gravnet2(x)  #! batchnorm 2
        # x = global_exchange(x, batch)
        # x = self.output(x)
        return x, graph, loss_regularizing_neig, ll_r


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.00)
