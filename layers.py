import torch 
from utils import *
import torch
import torch_scatter
from torch.nn import Linear, Sequential, LayerNorm, ReLU
from torch_geometric.nn.conv import MessagePassing
from torch_sparse import SparseTensor

class  Thickness_ProcessorLayer(MessagePassing):
    def __init__(self, in_channels, out_channels,  **kwargs):
        super(Thickness_ProcessorLayer, self).__init__(  **kwargs )

        self.edge_mlp = Sequential(Linear( 3* in_channels , out_channels),
                                   ReLU(),
                                   Linear( out_channels, out_channels),
                                   LayerNorm(out_channels)
                                   )

        self.node_mlp = Sequential(Linear( 2* in_channels , out_channels),
                                   ReLU(),
                                   Linear( out_channels, out_channels),
                                   LayerNorm(out_channels)
                                   )

        self.reset_parameters()

    def reset_parameters(self):
        """
        reset parameters for stacked MLP layers
        """
        self.edge_mlp[0].reset_parameters()
        self.edge_mlp[2].reset_parameters()

        self.node_mlp[0].reset_parameters()
        self.node_mlp[2].reset_parameters()

    def forward(self, x, edge_index, edge_attr, edge_weight = None, size = None):
        """
        Handle the pre and post-processing of node features/embeddings,
        as well as initiates message passing by calling the propagate function.

        Note that message passing and aggregation are handled by the propagate
        function, and the update

        x has shpae [node_num , in_channels] (node embeddings)
        edge_index: [2, edge_num]
        edge_attr: [E, in_channels]

        """
        out, updated_edges = self.propagate(edge_index, x = x, edge_attr = edge_attr, size = (x.shape[0], x.shape[0]), dim_size = x.shape[0], edge_weight=edge_weight)

        updated_nodes = torch.cat([x,out],dim=1)

        updated_nodes = x + self.node_mlp(updated_nodes)

        return updated_nodes, updated_edges

    def message(self, x_i, x_j, edge_attr):
        """
        source_node: x_i has the shape of [E, in_channels]
        target_node: x_j has the shape of [E, in_channels]
        target_edge: edge_attr has the shape of [E, out_channels]

        The messages that are passed are the raw embeddings. These are not processed.
        """

        updated_edges=torch.cat([x_i, x_j, edge_attr], dim = 1)
        updated_edges=self.edge_mlp(updated_edges)+edge_attr

        return updated_edges

    def aggregate(self, updated_edges, edge_index, dim_size = None, edge_weight=None):
        """
        First we aggregate from neighbors (i.e., adjacent nodes) through concatenation,
        then we aggregate self message (from the edge itself). This is streamlined
        into one operation here.
        """

        if edge_weight is None:
            edge_weight = torch.ones(updated_edges.shape[0], device=updated_edges.device)

        # The axis along which to index number of nodes.
        node_dim = 0

        # Apply edge weight to updated edges
        updated_edges = updated_edges * edge_weight.view(-1, 1)  # Broadcasting for multidimensional features

        if type(edge_index) == SparseTensor:
            row_idx, col_idx, _ = edge_index.coo()
            out = torch_scatter.scatter(updated_edges, row_idx, dim=node_dim, reduce = 'sum', dim_size=dim_size)
        else:
            out = torch_scatter.scatter(updated_edges, edge_index[0, :], dim=node_dim, reduce = 'sum', dim_size=dim_size)

        return out, updated_edges