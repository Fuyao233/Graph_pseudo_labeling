from torch import Tensor
from torch_geometric.typing import OptTensor
from Non_Homophily_Large_Scale.models import *
from torch_scatter import scatter_mean

class Encoder(nn.Module):
    def __init__(self, in_channel, out_channal=4, hidden_channel=32) -> None:
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(in_features=in_channel, out_features=hidden_channel)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=hidden_channel, out_features=out_channal)
    
    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)

class Decoder(nn.Module):
    def __init__(self, out_channel, in_channal=4, hidden_channel=32) -> None:
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(in_features=in_channal, out_features=hidden_channel)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=hidden_channel, out_features=out_channel)
    
    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)


class myGCNconv(GCNConv):
    def __init__(self, in_channels, out_channels, n_class):
        super(myGCNconv, self).__init__(in_channels, out_channels)
        self.encoder_group = nn.ModuleList([Encoder(out_channels) for _ in range(n_class)])
        self.decoder_group = nn.ModuleList([Decoder(out_channels) for _ in range(n_class)])
        
        self.auto_encoder_loss_flag = False
    
    def set_antoencoder_index(self, antoencoder_indices):
        self.antoencoder_indices = antoencoder_indices # pseudo node labels
    
    def encoder_forward(self, x):
        mlp_group = self.encoder_group
        indices = self.antoencoder_indices
        # group
        sorted_indices = torch.argsort(indices)
        sorted_antoencoder_indices = self.antoencoder_indices[sorted_indices]
        sorted_x = x[sorted_indices]
        
        # forward
        outputs = []
        start_idx = (sorted_antoencoder_indices == -1).nonzero(as_tuple=True)[0][0]
        end_idx = (sorted_antoencoder_indices == -1).nonzero(as_tuple=True)[0][-1] + 1
        self.unlabeled_features = sorted_x[start_idx:end_idx] # save for reconstruction
        self.unlabeled_indices = sorted_indices[start_idx:end_idx]
        outputs.append(torch.zeros(((end_idx-start_idx), mlp_group[0].fc2.out_features)).to(self.unlabeled_indices.device)) # add features of unlabeled nodes
        for mlp_idx in range(len(mlp_group)):
            start_idx = (sorted_antoencoder_indices == mlp_idx).nonzero(as_tuple=True)[0][0]
            end_idx = (sorted_antoencoder_indices == mlp_idx).nonzero(as_tuple=True)[0][-1] + 1
            
            outputs.append(mlp_group[mlp_idx](sorted_x[start_idx:end_idx]))
        
        # resort the outputs
        processed = torch.cat(outputs, dim=0)
        inverse_indices = torch.argsort(sorted_indices)
        result = processed[inverse_indices]

        return result         
    
    def decoder_forward(self, x, index):
        mlp_group = self.decoder_group
        indices = self.antoencoder_indices[index]
        
        
        sorted_indices = torch.argsort(indices)
        sorted_antoencoder_indices = indices[sorted_indices]
        sorted_x = x[sorted_indices]
        
        # forward
        outputs = []
        start_idx = (sorted_antoencoder_indices == -1).nonzero(as_tuple=True)[0][0]
        end_idx = (sorted_antoencoder_indices == -1).nonzero(as_tuple=True)[0][-1] + 1
        outputs.append(torch.zeros(((end_idx-start_idx), mlp_group[0].fc2.out_features)).to(self.unlabeled_indices.device)) # add features of unlabeled nodes
        for mlp_idx in range(len(mlp_group)):
            start_idx = (sorted_antoencoder_indices == mlp_idx).nonzero(as_tuple=True)[0][0]
            end_idx = (sorted_antoencoder_indices == mlp_idx).nonzero(as_tuple=True)[0][-1] + 1
            
            outputs.append(mlp_group[mlp_idx](sorted_x[start_idx:end_idx]))
        
        # resort the outputs
        processed = torch.cat(outputs, dim=0)
        inverse_indices = torch.argsort(sorted_indices)
        result = processed[inverse_indices]

        return result         
    
    def set_auto_encoder_loss_flag(self):
        self.auto_encoder_loss_flag = True
    
    def forward(self, x: Tensor, edge_index,
        edge_weight: OptTensor = None) -> Tensor:

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, self.flow, x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, self.flow, x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache
                    
        x = self.lin(x)
        
        loss = None
        if self.auto_encoder_loss_flag:
            loss = self.cal_autoencoder_loss(x)
        
        x = self.encoder_forward(x)
        # x = torch.stack([self.encoder_group[i](x[i]) for i in self.antoencoder_indices]).squeeze()
        
        # propagate_type: (x: Tensor, edge_weight: OptTensor)

        # print(x.size())
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)

        out[self.unlabeled_indices] = self.unlabeled_features
        
        if self.bias is not None:
            out = out + self.bias

        if self.auto_encoder_loss_flag:
            self.auto_encoder_loss_flag = False
            return out, loss
        else:
            return out
    
    # def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
    #     x_j = super().message(x_j, edge_weight)
    #     return x_j
        # return torch.stack([self.encoder_group[i](x_j[i]) for i in self.antoencoder_index])
    
    def aggregate(self, inputs, index, ptr, dim_size):
        # inputs = torch.stack([self.decoder_group[self.antoencoder_index_node[input_node_num]](inputs[i]) for i,input_node_num in enumerate(index)])
        inputs = self.decoder_forward(inputs, index)
        return scatter_mean(inputs, index, dim=0, dim_size=dim_size)

    def cal_autoencoder_loss(self, features):
        x = features.clone()
        loss = 0
        for mlp_idx in range(len(self.encoder_group)):
            x_idx = x[self.antoencoder_indices==mlp_idx]
            x_embedding = self.encoder_group[mlp_idx](x_idx)
            x_tilda = self.decoder_group[mlp_idx](x_embedding)
            # loss = F.mse_loss(x_idx, x_tilda, reduction='sum')/x_idx.size()[0] + loss
            loss = F.mse_loss(x_idx, x_tilda)
        return loss
    

class ourModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,
                 dropout):
        
        super(ourModel, self).__init__()
        
        self.convs = torch.nn.ModuleList(
            [myGCNconv(in_channels=input_dim, out_channels=hidden_dim, n_class=output_dim)] +
            [myGCNconv(in_channels=hidden_dim, out_channels=hidden_dim, n_class=output_dim)                             
                for i in range(num_layers-2)] + 
            [myGCNconv(in_channels=hidden_dim, out_channels=output_dim, n_class=output_dim)]    
        )
        self.bns = torch.nn.ModuleList([
            torch.nn.BatchNorm1d(num_features=hidden_dim) 
                for i in range(num_layers-1)
        ])
        self.dropout = dropout
        
        # self.probability
    
    def forward(self,data,auto_encoder_loss_flag=False):
        # select autoencoder index
        autoencoder_indices = data.training_labels
        for conv in self.convs:
            conv.set_antoencoder_index(autoencoder_indices)
        
        x, adj_t =data.x, data.edge_index
        auto_encoder_loss = None
        for conv, bn in zip(self.convs[:-1], self.bns):
            x1 = None
            if auto_encoder_loss_flag:
                conv.set_auto_encoder_loss_flag()
                x1, auto_encoder_loss = conv(x, adj_t)
            else:
                x1 = conv(x, adj_t)
            x1 = F.relu(bn(x1))
            if self.training:
                x1 = F.dropout(x1, p=self.dropout)
            x = x1
            
            # if auto_encoder_loss_flag:
            #     auto_encoder_loss_flag = auto_encoder_loss + conv.cal_autoencoder_loss(x)
            
        x = self.convs[-1](x, adj_t)
        
        if auto_encoder_loss_flag:
            return torch.softmax(x,dim=1), auto_encoder_loss
        else:
            return torch.softmax(x,dim=1)
        
    
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,
                 dropout):
        
        super(GCN, self).__init__()
        
        self.convs = torch.nn.ModuleList(
            [GCNConv(in_channels=input_dim, out_channels=hidden_dim)] +
            [GCNConv(in_channels=hidden_dim, out_channels=hidden_dim)                             
                for i in range(num_layers-2)] + 
            [GCNConv(in_channels=hidden_dim, out_channels=output_dim)]    
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
        return torch.softmax(x,dim=1)


# model based on probability
# class myGCNconv(GCNConv):
#     def __init__(self, in_channels, out_channels, n_class):
#         super(myGCNconv, self).__init__(in_channels, out_channels)
#         self.encoder_group = nn.ModuleList([Encoder(out_channels) for _ in range(n_class)])
#         self.decoder_group = nn.ModuleList([Decoder(out_channels) for _ in range(n_class)])
    
    
#     def set_antoencoder_index(self, antoencoder_index_node, antoencoder_index_edge):
#         self.antoencoder_index_node = antoencoder_index_node
#         self.antoencoder_index_edge = antoencoder_index_edge
    
#     def forward(self, x: Tensor, edge_index,
#         edge_weight: OptTensor = None) -> Tensor:

#         if self.normalize:
#             if isinstance(edge_index, Tensor):
#                 cache = self._cached_edge_index
#                 if cache is None:
#                     edge_index, edge_weight = gcn_norm(  # yapf: disable
#                         edge_index, edge_weight, x.size(self.node_dim),
#                         self.improved, self.add_self_loops, self.flow, x.dtype)
#                     if self.cached:
#                         self._cached_edge_index = (edge_index, edge_weight)
#                 else:
#                     edge_index, edge_weight = cache[0], cache[1]

#             elif isinstance(edge_index, SparseTensor):
#                 cache = self._cached_adj_t
#                 if cache is None:
#                     edge_index = gcn_norm(  # yapf: disable
#                         edge_index, edge_weight, x.size(self.node_dim),
#                         self.improved, self.add_self_loops, self.flow, x.dtype)
#                     if self.cached:
#                         self._cached_adj_t = edge_index
#                 else:
#                     edge_index = cache

#         x = self.lin(x)
#         x = torch.stack([self.encoder_group[i](x[i]) for i in self.antoencoder_index_node]).squeeze()
#         # propagate_type: (x: Tensor, edge_weight: OptTensor)

#         print(x.size())
#         out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
#                              size=None)

#         if self.bias is not None:
#             out = out + self.bias

#         return out
        
    # def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
    #     x_j = super().message(x_j, edge_weight)
        
        
    #     return x_j
    #     # return torch.stack([self.encoder_group[i](x_j[i]) for i in self.antoencoder_index])
    
    # def aggregate(self, inputs, index, ptr, dim_size):
    #     inputs = torch.stack([self.decoder_group[self.antoencoder_index_node[input_node_num]](inputs[i]) for i,input_node_num in enumerate(index)])
    #     return scatter_mean(inputs, index, dim=0, dim_size=dim_size)
    
