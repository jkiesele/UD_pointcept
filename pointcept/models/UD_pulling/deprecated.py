
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



class GravNetConv(nn.Module):
    """
    Param: [in_dim, out_dim]
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
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k
        self.num_workers = num_workers
        self.lin_s = Linear(in_channels, space_dimensions, bias=False)
        self.lin_h = Linear(in_channels, propagate_dimensions)
        self.lin = Linear(2 * propagate_dimensions, out_channels)

    def forward(self, g, x, original_coords, batch):
        h_l: Tensor = self.lin_h(x)  #! input_feature_transform
        s_l: Tensor = self.lin_s(x)
        graph = knn_per_graph(g, s_l, self.k)
        graph.ndata["s_l"] = s_l
        row = graph.edges()[0]
        col = graph.edges()[1]
        edge_index = torch.stack([row, col], dim=0)

        edge_weight = (s_l[edge_index[0]] - s_l[edge_index[1]]).pow(2).sum(-1)
        edge_weight = torch.sqrt(edge_weight + 1e-6)
        edge_weight = torch.exp(-torch.square(edge_weight))
        graph.edata["edge_weight"] = edge_weight.view(-1, 1)
        graph.ndata["h"] = h_l
        graph.update_all(self.message_func, self.reduce_func)
        out = graph.ndata["h"]

        out = self.lin(out)

        return (out, s_l)

    def message_func(self, edges):
        e_ij = edges.data["edge_weight"] * edges.src["h"]
        return {"e": e_ij}

    def reduce_func(self, nodes):
        mean_ = torch.mean(nodes.mailbox["e"], dim=-2)
        max_ = torch.max(nodes.mailbox["e"], dim=-2)[0]
        h = torch.cat((mean_, max_), dim=-1)
        return {"h": h}

