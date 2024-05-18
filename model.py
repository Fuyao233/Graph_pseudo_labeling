from typing import Optional
from torch import Tensor
from torch_geometric.typing import OptTensor
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from models.myConv import myConv
from models.myConv_basis import myConv_basis
from models.GCN import GCN 
from models.mlp import MLP
import torch
from basis_method.basis_process_dim32_class2 import basis_process_dim32_class2

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
                 basis_num=4,
                 soft_flag = True,
                 h=0.075):
        
            super(ourModel_basis, self).__init__()
        
            self.soft_flag = soft_flag
            self.num_class = output_dim
            self.basis_num = basis_num
            self.basis_dim = hidden_dim
            basis_matrix = basis_process_dim32_class2(basis_dim=hidden_dim, basis_num=basis_num, h=h)
            self.cross_basis_matrix = self.produce_cross_basis(basis_matrix)
            
            self.convs = torch.nn.ModuleList(
                [myConv_basis(in_channels=input_dim, out_channels=hidden_dim, n_class=output_dim, cross_basis_matrix=self.cross_basis_matrix, basis_num=basis_num, soft_flag=soft_flag)] +
                [myConv_basis(in_channels=hidden_dim, out_channels=hidden_dim, n_class=output_dim, cross_basis_matrix=self.cross_basis_matrix, basis_num=basis_num, soft_flag=soft_flag)                             
                    for i in range(num_layers-2)] + 
                [myConv_basis(in_channels=hidden_dim, out_channels=hidden_dim, n_class=output_dim, cross_basis_matrix=self.cross_basis_matrix, basis_num=basis_num, soft_flag=soft_flag)]    
            )
            
            self.res_lin = nn.Linear(hidden_dim, output_dim)
            
            # self.lins = torch.nn.ModuleList(
            #     [nn.Linear(input_dim, output_dim)] +
            #     [nn.Linear(hidden_dim, output_dim)                             
            #         for i in range(num_layers-1)]
            # )

            self.bns = torch.nn.ModuleList([
                torch.nn.BatchNorm1d(num_features=hidden_dim) 
                    for i in range(num_layers)
            ])
            self.dropout = dropout
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
        
        def produce_cross_basis(self, basis_matrix):
            basis_matrix = torch.tensor(basis_matrix)
            # get inverse
            inv_basis_matrix = torch.inverse(basis_matrix)
            
            basis_matrix = basis_matrix / torch.linalg.eigvals(basis_matrix).real.max()
            inv_basis_matrix = inv_basis_matrix / torch.linalg.eigvals(inv_basis_matrix).real.max()
            
            # get combination_matrix c*c*N*d*d
            c = self.num_class 
            N = self.basis_num 
            d = self.basis_dim
            combine_matrix = torch.empty((c, c, N, d, d))
            for i in range(c):
                for j in range(c):
                    if i==j:
                        combine_matrix[i][j] = torch.eye(d).unsqueeze(0).expand(N, -1, -1)
                    else:                        
                        combine_matrix[i][j] = torch.einsum('bij,bjk->bik', basis_matrix[i], inv_basis_matrix[j])
            
            # cross 
            cross_basis = torch.empty((c, c, 2**(N//2), d, d))
            scale_weight = torch.empty((c, c, 2**(N//2), d, d))
            for i in range(c):
                for j in range(c):
                    if i == j:
                        cross_basis[i][j] = torch.eye(d).unsqueeze(0).expand(2**(N//2), -1, -1)
                    else:    
                        for idx in range(2**(N//2)):
                            index_sequence = [(idx >> k) & 1 for k in range(N//2)]
                            combine_matrix_sequence = combine_matrix[i][j]
                            
                            select_matrix_sequence = None 
                            if i<j:
                                select_matrix_sequence = [combine_matrix_sequence[2*k+bit, :, :] for k, bit in enumerate(index_sequence)]
                            else:
                                index_sequence.reverse()
                                select_matrix_sequence = [combine_matrix_sequence[2*(N//2-k)-bit-1, :, :] for k, bit in enumerate(index_sequence)]
                            
                            result_matrix = select_matrix_sequence[0]
                            for matrix in select_matrix_sequence[1:]:
                                result_matrix = torch.matmul(result_matrix, matrix)
                            
                            cross_basis[i][j][idx] = result_matrix
                            
            return cross_basis
        
        def produce_message_passing_function(self):
            for conv in self.convs:
                conv.produce_message_passing_function()
                
        def forward(self,data):
            self.produce_message_passing_function()
            x = deepcopy(data)
            # res = []

            for n, conv in enumerate(self.convs):
                bn = self.bns[n]
                
                # layer_node_logits = self.lins[n](x.x)
                # conv.set_node_confidence(layer_node_logits)
                # res.append(layer_node_logits)
                
                x1 = conv(x)
                x1 = bn(x1) 
                x1 = F.relu(x1) 
                if self.training:
                    x1 = F.dropout(x1, p=self.dropout)

                x.x = x1
            
            # layer_node_logits = self.lins[-1](x.x)
            # self.convs[-1].set_node_confidence(layer_node_logits)
            # res.append(layer_node_logits)
            
                
                # if auto_encoder_loss_flag:
                #     auto_encoder_loss_flag = auto_encoder_loss + conv.cal_autoencoder_loss(x)
            
            # if auto_encoder_loss_flag:   
            #     self.convs[-1].set_auto_encoder_loss_flag()
            #     x, sub_auto_encoder_loss = self.convs[-1](x)
            #     auto_encoder_loss = auto_encoder_loss + sub_auto_encoder_loss
            # else:
            
            # x = self.convs[-1](x)
            logits = self.res_lin(x1)
            return logits
        
            # res.append(x)
            # res = torch.stack(res)
            
            # if not self.training:
            #     return res[-1]
            
            # if auto_encoder_loss_flag:
            #     if self.soft_flag:
            #         return res, auto_encoder_loss
            #     else:
            #         return res[-1], auto_encoder_loss
            # else:
            #     if self.soft_flag:
            #         return res
            #     else:
            #         return res[-1]
            
            return x
            
            # x = data.x
            # for i, lin in enumerate(self.convs[:-1]):
            #     x = lin(x)
            #     x = F.relu(x, inplace=True)
            #     x = self.bns[i](x)
            #     x = F.dropout(x, p=self.dropout, training=self.training)
            # x = self.convs[-1](x)
            # return x 
    