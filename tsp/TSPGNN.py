import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool as gap


class TSPGNN(nn.Module):
    """
    Built off of the model found at https://github.com/flixpar/AlphaTSP/blob/master/notebooks/policynet.ipynb
    (posted on Piazza by Professor Drori)
    """
    def __init__(self):
        super(TSPGNN, self).__init__()
        self.conv1 = GCNConv(1, 16)
        self.conv2 = GCNConv(16, 16)
        self.conv3 = GCNConv(16, 1)
        self.fc = nn.Linear(16, 1)

    def forward(self, graph):
        x, edges, choices = graph.edge_attr, graph.edge_index, graph.x # I am using the edge weights as the main feature
        x = self.conv1(x.float(), edges)
        x = F.relu(x)
        x = self.conv2(x, edges)
        x = F.relu(x)

        c = self.conv3(x, edges)
        c = nn.Linear(graph.num_edges, graph.num_nodes)(c.view((1, -1)))
        c = F.relu(c.squeeze())
        choice = F.softmax(c, dim=0)

        v = gap(x, torch.zeros(graph.num_edges, dtype=torch.long))
        value = self.fc(v)

        return choice, value.item()  # output is a pi vector and a reward value
