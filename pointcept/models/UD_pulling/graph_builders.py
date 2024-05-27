'''
Differernt ways to build the graphs.
No learnable parameters here
'''

import torch
import torch_cmspepr
from torch import Tensor
from torch.nn import Linear
import dgl
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import numpy as np


'''
Vertical graph builders (up/down).
U-net like naming convention, i.e. "down" means less points
'''
class ContractGraphBase(torch.nn.Module):
    '''
    class only defines the interface,
    the functionality is defined in select_points, the rest is boiler plate.

    The node features are 's_l' (the coordinates for the kNN) and 'h' (generic features)
    '''
    def __init__(self, vertical_k):
        self.vertical_k = vertical_k
        super().__init__()

    def select_points(self, g):
        '''
        this should return a boolean selection tensor 
        '''
        # for testing simply select every 3rd point
        return torch.arange(g.number_of_nodes()) % 3 == 0

    def forward(self,g):
        '''
        g is a dgl graph.
        the graph needs node features 's_l' which are the coordinates to build neighbours from
        Returns 
         - vertical graph
         - new (lower dimensional) horizontal 'graph' (just points for now)
        '''

        list_graphs = dgl.unbatch(g)

        out_vertical = []
        out_horizontal = []
        for gi in list_graphs:
            vert, hor = self.forward_per_graph(gi)
            out_vertical.append(vert)
            out_horizontal.append(hor)

        #concat each graph in outs
        return dgl.batch(out_vertical), dgl.batch(out_horizontal)

    def create_edges(self, p, s_l, n_nodes, M):

        neigh_indices, _ = torch_cmspepr.select_knn_directional( #NOTICE THE SWAP HERE
            (s_l[~p]).to('cuda'), # from #DEBUG
            (s_l[p]).to('cuda'), # to
            
            M)
        neigh_indices = neigh_indices.to(s_l.device)#DEBUG
        
        nodes = torch.range(start=0, end=n_nodes - 1, step=1).to(s_l.device)
        nodes_up = nodes[~p]
        nodes_down = nodes[p]

        j = nodes_down[neigh_indices]
        j = j.view(-1)
        i = torch.tile(nodes_up.view(-1, 1), (1, M)).reshape(-1)

        return (i.long(), j.long())

    def forward_per_graph(self, g):
        '''
        implements forward per graph
        Returns
        - vertical graph
        - horizontal graph (points)
        '''
        device = g.device
        p = self.select_points(g) #this gives a boolean to select
        n_nodes = g.number_of_nodes()

        number_down_points = torch.sum(p)
        M = self.vertical_k
        if number_down_points > self.vertical_k:
            M = self.vertical_k
        else:
            M = number_down_points

        # create directed edges from 
        s_l = g.ndata['s_l']
        edges = self.create_edges(p, s_l, n_nodes, M)

        print(edges)
        # this is the vertical graph
        g_vert = dgl.graph(edges, num_nodes=n_nodes).to(device)
        #copy node data
        for k in g.ndata.keys():
            g_vert.ndata[k] = g.ndata[k]
        
        #build reduced graph (selected points only)
        g_hor = dgl.DGLGraph().to(device)
        g_hor.add_nodes(number_down_points)
        for k in g.ndata.keys():
            g_hor.ndata[k] = g.ndata[k][p]

        return g_vert, g_hor


# test the crontract class on a simple example


def visualise_graph(g, outname, edge_label_name='weight'):
    #use networkx here and visualise it to a pdf
    import networkx as nx
    import matplotlib.pyplot as plt

    G = nx.DiGraph()
    G.add_nodes_from(range(g.number_of_nodes()))
    edges = g.edges()
    ekeys = g.edata.keys()
    for i in range(edges[0].shape[0]):
        # also add the edge features as attributes
        G.add_edge(edges[0][i].item(), edges[1][i].item(), **{k: g.edata[k][i].tolist() for k in ekeys})

    #also pass the features s_l
    s_l = g.ndata['s_l']
    pos = {i: s_l[i].tolist() for i in range(g.number_of_nodes())}
    nx.set_node_attributes(G, pos, 'pos')

    #if nodes have property 'p' then color them
    if 'p' in g.ndata.keys():
        p = g.ndata['p']
        color_map = []
        for i in range(g.number_of_nodes()):
            if p[i]:
                color_map.append('red')
            else:
                color_map.append('blue')
    else:
        color_map = ['blue' for i in range(g.number_of_nodes())]

    # get the full list of edge features keys from the dgl graph
    
    
    #create an entry to be drawn next to the edge for each feature
    edge_labels = {}
    for k in ekeys:
        edge_labels[k] = {i: str(g.edata[k][i].tolist()) for i in range(g.number_of_edges())}

    #draw the position according to pos, add all other node labels as text per node
    # also draw with edge attributes
    nx.draw(G, with_labels=True, pos=pos, node_color=color_map)
    edge_labels = nx.get_edge_attributes(G, edge_label_name)
    # round the edge attributes to 2 significant digits. these can be arrays so use numpy here
    edge_labels = {k: np.round(v, 2)  for k, v in edge_labels.items()}
    #draw them with very small font size
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,font_size=4)
    plt.savefig(outname)
    plt.close()
    #clear all
    plt.clf()

def test():
    #create a test graph with no edges as input, only nodes; but with coordinates and make it a batch of two
    n_nodes = 13
    gs =[]
    for _ in range(1):
        g = dgl.DGLGraph()
        g.add_nodes(n_nodes)
        g.ndata['s_l'] = torch.rand((n_nodes, 2))
        g.ndata['h'] = torch.rand((n_nodes, 3))
        gs.append(g)
    g = dgl.batch(gs)



    #put it on the gpu
    #g.to('cuda')

    #create the contract graph
    cg = ContractGraphBase(3)

    #add node feature if it is a selected point
    p = cg.select_points(g)
    g.ndata['p'] = p

    #print the graphs
    visualise_graph(g, "initial.pdf")

    #run the forward
    g_vert, g_hor = cg(g)

    #take it from the gpu
    g_vert = g_vert.to('cpu')
    g_hor = g_hor.to('cpu')

    #print the graphs
    visualise_graph(g_vert, "vertical.pdf")
    visualise_graph(g_hor, "horizontal.pdf")

#test()






'''
Helpers
'''
def invert_directed_graph(g):
    '''
    g is a dgl graph
    copy all data and invert edges
    '''

    src, dst = g.edges()
    # Create a new graph with the same number of nodes and inverted edges
    g_inv = dgl.graph((dst, src), num_nodes=g.num_nodes())
    for k in g.ndata.keys():
        g_inv.ndata[k] = g.ndata[k]
    for k in g.edata.keys():
        g_inv.edata[k] = g.edata[k]
    return g_inv



class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, 
                 in_dim, 
                 out_dim, 
                 num_heads, 
                 use_bias, 
                 possible_empty,
                 normalize_sending=False,
                 equivariant=False):
        super().__init__()
        self.possible_empty = possible_empty
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.normalize_sending = normalize_sending
        self.equivariant = equivariant

        self.sqrt_d = np.sqrt(self.out_dim)

        if self.equivariant:
            self.A = nn.Linear(in_dim, num_heads, bias=use_bias) 
        else:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)

        self.V = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
       
        if self.equivariant:
            self.apply_linears = self.apply_linears_eq
        else:
            self.apply_linears = self.apply_linears_noteq

    def propagate_attention(self, g, score=None):
        # Compute attention score
        
        if score is None:
            g.apply_edges(src_dot_dst("K_h", "Q_h", "score"))  # , edges)
            g.apply_edges(scaled_exp("score", self.sqrt_d))
        else:
            g.edata['score'] = score

        visualise_graph(g, "attention_prop.pdf", edge_label_name='score')
        # normalisation for score
        
        # invert the edges explicitly
        if self.normalize_sending:
            g = invert_directed_graph(g)
 
        #set edges of g to the inverted edges
        g.send_and_recv(
            g.edges(), fn.copy_e("score", "score"), fn.sum("score", "z")
        ) 
        
        #normalise the score of each edge by the sum z for the respective node
        g.apply_edges(fn.e_div_v("score", "z", "nscore"))

        # Send weighted values to target nodes, invert back
        if self.normalize_sending:
            g = invert_directed_graph(g)

        visualise_graph(g, "attention_nprop.pdf", edge_label_name='nscore')
        
        g.send_and_recv(
            g.edges(),
            fn.u_mul_e("V_h", "score", "V_h"),
            fn.sum("V_h", "wV"),  # deprecated in dgl 1.0.1
        )

        return g
    
    def apply_linears_noteq(self, g, h):

        Q_h = self.Q(h)
        K_h = self.K(h)
        V_h = self.V(h)
        
        g.ndata["Q_h"] = Q_h.view(-1, self.num_heads, self.out_dim)
        g.ndata["K_h"] = K_h.view(-1, self.num_heads, self.out_dim)
        g.ndata["V_h"] = V_h.view(-1, self.num_heads, self.out_dim)
        g = self.propagate_attention(g)

        return g
        
    def apply_linears_eq(self, g, h):
        # create edges that correspond to the difference in h
        g.ndata['_h_temp'] = h

        V_h = self.V(h)
        g.ndata["V_h"] = V_h.view(-1, self.num_heads, self.out_dim)

        # subtract the source node features from the target node features to create edge features
        # in the end the sign is irrelevant, but this is closer to the original idea
        g.apply_edges(fn.v_sub_u('_h_temp', '_h_temp', '_diff'))
        
        # no need to run the 'diff' property through the query and key layers
        score = self.A(g.edata['_diff']) #that is already the score

        #add another dimension to the score
        score = score.view(-1, self.num_heads, 1)

        #remove h_temp from the graph, diff too
        g.ndata.pop('_h_temp')
        g.edata.pop('_diff')

        g = self.propagate_attention(g, score)
        return g


    def forward(self, g, h):

        g = self.apply_linears(g, h)

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



def test_MultiHeadAttentionLayer():
    #create a test graph with no edges as input, only nodes; but with coordinates and make it a batch of two
    n_nodes = 13
    gs =[]
    for _ in range(1):
        g = dgl.DGLGraph()
        g.add_nodes(n_nodes)
        g.ndata['s_l'] = torch.rand((n_nodes, 2))
        g.ndata['h'] = torch.rand((n_nodes, 3))
        gs.append(g)
    g = dgl.batch(gs)

    #create the contract graph
    cg = ContractGraphBase(3)

    #add node feature if it is a selected point
    p = cg.select_points(g)
    g.ndata['p'] = p

    #run the forward
    g_vert, g_hor = cg(g)

    #make the graph bi-directional, just for robustness tests
    # g_vert = dgl.add_reverse_edges(g_vert)

    #create the attention layer
    mha = MultiHeadAttentionLayer(3, 3, 2, True, True, normalize_sending=True, equivariant=True)
    '''
    (in_dim, 
                 out_dim, 
                 num_heads, 
                 use_bias, 
                 possible_empty,
                 normalize_sending=False)
    '''

    #run the attention layer
    g.ndata['parsed'] = mha(g_vert, g_vert.ndata['h'])
    # visualise the graph
    visualise_graph(g, "attention.pdf")

test_MultiHeadAttentionLayer()

'''
Horizontal graph builders (if needed)
'''