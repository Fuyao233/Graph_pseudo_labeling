import torch.nn as nn 
from Non_Homophily_Large_Scale.models import *
import torch
from torch_geometric.nn import GINConv


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,
                 dropout):
        
        super(GCN, self).__init__()
        
        self.convs = torch.nn.ModuleList(
            [GINConv(in_channels=input_dim, out_channels=hidden_dim)] +
            [GINConv(in_channels=hidden_dim, out_channels=hidden_dim)                             
                for i in range(num_layers-2)] + 
            [GINConv(in_channels=hidden_dim, out_channels=output_dim)]    
        )
        self.bns = torch.nn.ModuleList([
            torch.nn.BatchNorm1d(num_features=hidden_dim) 
                for i in range(num_layers-1)
        ])
        self.dropout = dropout
    
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
    
    def forward(self, data):
        x, adj_t =data.x, data.edge_index
        for conv, bn in zip(self.convs[:-1], self.bns):
            x1 = F.relu(bn(conv(x, adj_t)))
            if self.training:
                x1 = F.dropout(x1, p=self.dropout)
            x = x1
        x = self.convs[-1](x, adj_t)
        return x

