from typing import Optional
from torch import Tensor
from torch_geometric.typing import OptTensor
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from models.myConv import myConv
from models.GCN import GCN 
from models.mlp import MLP
import torch


class ourModel(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hidden_dim, 
                 output_dim, 
                 num_layers,
                 dropout,
                 soft_flag = True):
        
        super(ourModel, self).__init__()
        
        self.soft_flag = soft_flag
        
        self.convs = torch.nn.ModuleList(
            [myConv(in_channels=input_dim, out_channels=hidden_dim, n_class=output_dim, soft_flag=soft_flag)] +
            [myConv(in_channels=hidden_dim, out_channels=hidden_dim, n_class=output_dim, soft_flag=soft_flag)                             
                for i in range(num_layers-2)] + 
            [myConv(in_channels=hidden_dim, out_channels=output_dim, n_class=output_dim, soft_flag=soft_flag)]    
        )
        
        self.lins = torch.nn.ModuleList(
            [nn.Linear(input_dim, output_dim)] +
            [nn.Linear(hidden_dim, output_dim)                             
                for i in range(num_layers-1)]
        )

        self.bns = torch.nn.ModuleList([
            torch.nn.BatchNorm1d(num_features=hidden_dim) 
                for i in range(num_layers-1)
        ])
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
    
    def forward(self,data,auto_encoder_loss_flag=False):
        auto_encoder_loss = None
        x = deepcopy(data)
        res = []

        for n, conv in enumerate(self.convs[:-1]):
            bn = self.bns[n]
            
            
            layer_node_logits = self.lins[n](x.x)
            conv.set_node_confidence(layer_node_logits)
            res.append(layer_node_logits)
            
            x1 = None
            if auto_encoder_loss_flag:
                conv.set_auto_encoder_loss_flag()
                x1, sub_auto_encoder_loss = conv(x)
                auto_encoder_loss = sub_auto_encoder_loss if auto_encoder_loss is None else auto_encoder_loss + sub_auto_encoder_loss

            else:
                x1 = conv(x)
            x1 = bn(x1) 
            x1 = F.relu(x1) 
            if self.training:
                x1 = F.dropout(x1, p=self.dropout)

            x.x = x1
        
        layer_node_logits = self.lins[-1](x.x)
        self.convs[-1].set_node_confidence(layer_node_logits)
        res.append(layer_node_logits)
        
            
            # if auto_encoder_loss_flag:
            #     auto_encoder_loss_flag = auto_encoder_loss + conv.cal_autoencoder_loss(x)
        
        if auto_encoder_loss_flag:   
            self.convs[-1].set_auto_encoder_loss_flag()
            x, sub_auto_encoder_loss = self.convs[-1](x)
            auto_encoder_loss = auto_encoder_loss + sub_auto_encoder_loss
        else:
            x = self.convs[-1](x)
        res.append(x)
        res = torch.stack(res)
        
        if not self.training:
            return res[-1]
        
        if auto_encoder_loss_flag:
            if self.soft_flag:
                return res, auto_encoder_loss
            else:
                return res[-1], auto_encoder_loss
        else:
            if self.soft_flag:
                return res
            else:
                return res[-1]
        
        # x = data.x
        # for i, lin in enumerate(self.convs[:-1]):
        #     x = lin(x)
        #     x = F.relu(x, inplace=True)
        #     x = self.bns[i](x)
        #     x = F.dropout(x, p=self.dropout, training=self.training)
        # x = self.convs[-1](x)
        # return x 
    
    def restart(self):
        for convs in self.convs:
            convs.restart()

class ourModel_basis(nn.Module):
        def __init__(self, 
                 input_dim, 
                 hidden_dim, 
                 output_dim, 
                 num_layers,
                 dropout,
                 soft_flag = True):
        
            super(ourModel, self).__init__()
        
            self.soft_flag = soft_flag
            
            self.convs = torch.nn.ModuleList(
                [myConv(in_channels=input_dim, out_channels=hidden_dim, n_class=output_dim, soft_flag=soft_flag)] +
                [myConv(in_channels=hidden_dim, out_channels=hidden_dim, n_class=output_dim, soft_flag=soft_flag)                             
                    for i in range(num_layers-2)] + 
                [myConv(in_channels=hidden_dim, out_channels=output_dim, n_class=output_dim, soft_flag=soft_flag)]    
            )
            
            self.lins = torch.nn.ModuleList(
                [nn.Linear(input_dim, output_dim)] +
                [nn.Linear(hidden_dim, output_dim)                             
                    for i in range(num_layers-1)]
            )

            self.bns = torch.nn.ModuleList([
                torch.nn.BatchNorm1d(num_features=hidden_dim) 
                    for i in range(num_layers-1)
            ])
            self.dropout = dropout
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
        
