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


class GraphTransformerLayer(nn.Module):
    """
    Attention as in the point transformer: https://arxiv.org/pdf/2012.09164.pdf
    Subtraction plus position encoding
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

        # if self.layer_norm:
        #     self.layer_norm1 = nn.LayerNorm(out_dim)

        # if self.batch_norm:
        #     self.batch_norm1 = nn.BatchNorm1d(out_dim)

        # # FFN
        # self.FFN_layer1 = nn.Linear(out_dim, out_dim * 2)
        # self.FFN_layer2 = nn.Linear(out_dim * 2, out_dim)

        # if self.layer_norm:
        #     self.layer_norm2 = nn.LayerNorm(out_dim)

        # if self.batch_norm:
        #     self.batch_norm2 = nn.BatchNorm1d(out_dim)

    def forward(self, g, h, c):
        h_in1 = h  # for first residual connection
        attn_out = self.attention(g, h, c)
        h = attn_out.view(-1, self.out_channels)
        print("h attention", h)
        # h = F.dropout(h, self.dropout, training=self.training)

        h = self.O(h)
        # print("h attention 1", h)
        # if self.residual:
        #     h = h_in1 + h  # residual connection

        # if self.layer_norm:
        #     h = self.layer_norm1(h)

        # if self.batch_norm:
        #     h = self.batch_norm1(h)

        # h_in2 = h  # for second residual connection

        # # FFN
        # h = self.FFN_layer1(h)
        # h = F.relu(h)
        # h = F.dropout(h, self.dropout, training=self.training)
        # h = self.FFN_layer2(h)

        # if self.residual:
        #     h = h_in2 + h  # residual connection

        # if self.layer_norm:
        #     h = self.layer_norm2(h)

        # if self.batch_norm:
        #     h = self.batch_norm2(h)
        # # print("h attention final", h)
        return h


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

        # self.FFN_layer1 = nn.Linear(in_dim, out_dim * 2 * num_heads)
        # self.FFN_layer2 = nn.Linear(out_dim * 2 * num_heads, out_dim * num_heads)

        self.linear_p = nn.Sequential(
            nn.Linear(3, 3),
            nn.ReLU(inplace=True),
            nn.Linear(3, out_dim),
        )

        self.MLP_edge = MLP_edge(out_dim)

    def propagate_attention(self, g):
        # Compute attention score
        g.apply_edges(self.MLP_edge)
        g.apply_edges(scaled_exp("score", np.sqrt(self.out_dim)))
        # Send weighted values to target nodes
        eids = g.edges()
        g.ndata["V_h_P_e"] = g.ndata["V_h"] + g.ndata["P_e"]
        g.send_and_recv(
            eids,
            fn.u_mul_e("V_h_P_e", "score", "V_h_P_e"),
            fn.sum("V_h_P_e", "wV"),
        )

        g.send_and_recv(eids, fn.copy_e("score", "score"), fn.sum("score", "z"))

    def forward(self, g, h, c):

        Q_h = self.Q(h)
        K_h = self.K(h)
        V_h = self.V(h)
        position_encoding = self.linear_p(c)

        # Reshaping into [num_nodes, num_heads, feat_dim] to
        # get projections for multi-head attention
        g.ndata["Q_h"] = Q_h.view(-1, self.num_heads, self.out_dim)
        g.ndata["K_h"] = K_h.view(-1, self.num_heads, self.out_dim)
        g.ndata["V_h"] = V_h.view(-1, self.num_heads, self.out_dim)
        g.ndata["P_e"] = position_encoding.view(-1, self.num_heads, self.out_dim)

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
            out_field: (edges.src[src_field] - edges.dst[dst_field] + edges.src["P_e"])
        }

    return func


def scaled_exp(field, scale_constant):
    def func(edges):
        # clamp for softmax numerical stability
        return {field: torch.exp((edges.data[field] / scale_constant).clamp(-5, 5))}

    return func


class MLP_edge(nn.Module):
    """
    Compute the input feature from neighbors
    """

    def __init__(self, out_dim):
        super(MLP_edge, self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, 1),
        )

    def forward(self, edges):
        dif = edges.src["K_h"] - edges.dst["Q_h"] + edges.src["P_e"]
        att_weight = self.MLP(dif)

        return {"score": att_weight}
