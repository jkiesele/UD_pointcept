'''
Differernt ways to build the graphs.
No learnable parameters here
'''

import torch
import torch_cmspepr
from torch import Tensor
from torch.nn import Linear
import dgl


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
        # for testing simply select every other point
        return torch.arange(g.number_of_nodes()) % 2 == 0

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
            (s_l[~p]).to('cuda'), #DEBUG
            (s_l[p]).to('cuda'), 
            
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


def visualise_graph(g, outname):
    #use networkx here and visualise it to a pdf
    import networkx as nx
    import matplotlib.pyplot as plt

    G = nx.DiGraph()
    G.add_nodes_from(range(g.number_of_nodes()))
    edges = g.edges()
    for i in range(edges[0].shape[0]):
        G.add_edge(edges[0][i].item(), edges[1][i].item())

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

    #draw the position according to pos
    nx.draw(G, with_labels=True, pos=pos, node_color=color_map)
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
    cg = ContractGraphBase(4)

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

test()






'''
Helpers
'''
# invert_directed_graph


'''
Horizontal graph builders (if needed)
'''