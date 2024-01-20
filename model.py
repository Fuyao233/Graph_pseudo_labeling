from torch import Tensor
from torch_geometric.typing import OptTensor
from Non_Homophily_Large_Scale.models import *
from torch_scatter import scatter_mean
import torch.nn.init as init

class basisAutoencoder(nn.Module):
    def __init__(self, basis) -> None:
        # a group of basis for a class 
        self.basis = basis # n * matrix_size * matrix_size
        self.coefficients = nn.Parameter(torch.randn(len(basis)))
        
        # if inverse_basis is None:
        #     # it is better if the reverse basis is offered to avoid redundant calculations
        #     self.inverse_basis = torch.zeros_like(basis)
        #     for i, individual in enumerate(basis):
        #         self.inverse_basis[i] = torch.inverse(individual).clone()
        # else:
        #     self.inverse_basis = inverse_basis
        # self.coefficients_inverse = nn.Parameter(torch.randn(len(self.reverse_basis)))
    
    def forward(self, input):
        input = F.linear(input, self.basis @ self.coefficients)
        return F.relu(input) # TODO: relu可能会影响可逆

class Encoder(nn.Module):
    def __init__(self, in_channel, out_channel=4, hidden_channel=32) -> None:
        super(Encoder, self).__init__()
        self.fc = nn.Linear(in_features=in_channel, out_features=out_channel)
        
        self.fc1 = nn.Linear(in_features=in_channel, out_features=hidden_channel)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=hidden_channel, out_features=out_channel)
        
        self.in_channel = in_channel
        self.out_channel = out_channel
        
    def forward(self,x):
        x = self.fc(x)
        x = self.relu(x)
        # x = self.fc1(x)
        # x = self.relu(x)
        # x = self.fc2(x)
        return x
    
    def restart(self):
        init.xavier_uniform_(self.fc.weight)
        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc2.weight)
        
        if self.fc.bias is not None:
            init.zeros_(self.fc.bias)
        if self.fc1.bias is not None:
            init.zeros_(self.fc1.bias)
        if self.fc2.bias is not None:
            init.zeros_(self.fc2.bias)
            
class Decoder(nn.Module):
    def __init__(self, out_channel, in_channal=4, hidden_channel=32) -> None:
        super(Decoder, self).__init__()
        self.fc = nn.Linear(in_features=in_channal, out_features=out_channel)
        
        self.fc1 = nn.Linear(in_features=in_channal, out_features=hidden_channel)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=hidden_channel, out_features=out_channel)
        self.in_channel = in_channal
        self.out_channel = out_channel
    
    def forward(self,x):
        x = self.fc(x)
        x = self.relu(x)
        # x = self.fc1(x)
        # x = self.relu(x)
        # x = self.fc2(x)
        return x 

    def restart(self):
        init.xavier_uniform_(self.fc.weight)
        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc2.weight)
        
        if self.fc.bias is not None:
            init.zeros_(self.fc.bias)
        if self.fc1.bias is not None:
            init.zeros_(self.fc1.bias)
        if self.fc2.bias is not None:
            init.zeros_(self.fc2.bias)

class myGCNconv_basis(GCNConv):
    def __init__(self, in_channels, out_channels, n_class, basis):
        super(myGCNconv, self).__init__(in_channels, out_channels)
        print("GCN_basis")

        # size of basis: (n_class+1) * 2 * n_basis(for each class) * matrix_size
        # self.combination_group[0][i]: message passing function from class i to common class
        # self.combination_group[1][i]: message passing function from common class to class i
        self.message_function_group = [[], []]
        for i in range(n_class):
            self.message_function_group[0].append(self.produce_composite_basis(basis, i+1, 0))
            self.message_function_group[1].append(self.produce_composite_basis(basis, 0, i+1))
        
        self.auto_encoder_loss_flag = False
        self.n_class = n_class 
        
    def produce_composite_basis(self, basis, i, j):
        # from class i to class j
        source_basis = np.array(basis[i][0])
        inv_target_basis = np.array(basis[j][1])
        return basisAutoencoder(basis=np.matmul(source_basis, inv_target_basis))
            
    
    def set_antoencoder_index(self, antoencoder_indices):
        self.antoencoder_indices = antoencoder_indices # pseudo node labels

    def encoder_forward(self, x):
        basis_group = self.message_function_group[0]
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
        outputs.append(torch.zeros(((end_idx-start_idx), basis_group[0].out_channel)).to(self.unlabeled_indices.device)) # add features of unlabeled nodes
        
        for basis_id in range(self.n_class):
            start_idx = None 
            end_idx = None 
            match_index = (sorted_antoencoder_indices == basis_id).nonzero(as_tuple=True)[0]
            if len(match_index)==0:
                continue 
            else:
                start_idx = match_index[0]
                end_idx = match_index[-1] + 1
            
            outputs.append(basis_group[basis_id](sorted_x[start_idx:end_idx]))
        
        # resort the outputs
        processed = torch.cat(outputs, dim=0)
        inverse_indices = torch.argsort(sorted_indices)
        result = processed[inverse_indices]

        # self.add_self_loops = False 
        
        return result         

    def decoder_forward(self, x, index):
        
        basis_group = self.message_function_group[1]
        indices = self.antoencoder_indices[index]
        
        sorted_indices = torch.argsort(indices)
        sorted_antoencoder_indices = indices[sorted_indices]
        sorted_x = x[sorted_indices]
        
        # forward
        outputs = []

        for basis_id in range(len(self.n_class)):
            match_indices = (sorted_antoencoder_indices == basis_id).nonzero(as_tuple=True)[0]
            start_idx = None 
            end_idx = None 
            if len(match_indices) == 0:
                continue 
            else:
                start_idx = (sorted_antoencoder_indices == basis_id).nonzero(as_tuple=True)[0][0]
                end_idx = (sorted_antoencoder_indices == basis_id).nonzero(as_tuple=True)[0][-1] + 1
            
            outputs.append(basis_group[basis_id](sorted_x[start_idx:end_idx]))
        
        # resort the outputs
        processed = torch.cat(outputs, dim=0)
        inverse_indices = torch.argsort(sorted_indices)
        result = processed[inverse_indices]

        return result         
    
    def set_auto_encoder_loss_flag(self):
        self.auto_encoder_loss_flag = True
    
    def forward(self, x: Tensor, edge_index,
        edge_weight: OptTensor = None) -> Tensor:
        self.add_self_loops = False
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
        
        x_clone = x.clone()
        
        loss = None
        if self.auto_encoder_loss_flag:
            loss = self.cal_autoencoder_loss(x)

        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,size=None)

        # out[self.unlabeled_indices] = self.unlabeled_features
        
        if self.bias is not None:
            out = out + self.bias if out is not None else self.bias
        
        out = self.update_node(x_clone, out)

        if self.auto_encoder_loss_flag:
            self.auto_encoder_loss_flag = False
            return out, loss
        else:
            return out
    
    def update_node(self, x, neighbor_feature):
        relu = nn.ReLU()
        return relu(x+neighbor_feature)
    
    def aggregate(self, inputs, index, ptr, dim_size):
        inputs = torch.stack([self.decoder_group[self.antoencoder_index_node[input_node_num]](inputs[i]) for i,input_node_num in enumerate(index)])
        
        if len(inputs) == 0:
            return 
        else:
            inputs = self.decoder_forward(inputs, index)
            return scatter_mean(inputs, index, dim=0, dim_size=dim_size)

    def cal_autoencoder_loss(self, features):
        # |A x \tilede{A}-1|_2
        # loss = 0
        # for mlp_idx in range(len(self.encoder_group)):
        #     A = self.encoder_group[mlp_idx].fc.weight
        #     B = self.decoder_group[mlp_idx].fc.weight
        #     loss = loss + torch.norm(torch.mm(B,A) - torch.eye(B.size()[0]).to(A.device), p='fro')
        
        # reconstruction loss
        x = features.clone()
        loss = 0
        for mlp_idx in range(len(self.encoder_group)):
            x_idx = x[self.antoencoder_indices==mlp_idx]
            x_embedding = self.encoder_group[mlp_idx](x_idx)
            x_tilda = self.decoder_group[mlp_idx](x_embedding)
            # loss = F.mse_loss(x_idx, x_tilda, reduction='sum')/x_idx.size()[0] + loss
            loss = F.mse_loss(x_idx, x_tilda)
        return loss
    
    def restart(self):
        for encoder in self.encoder_group:
            encoder.restart()
        for decoder in self.decoder_group:
            decoder.restart()
  
# model with encoder&decoder
class myGCNconv(GCNConv):
    def __init__(self, in_channels, out_channels, n_class):
        super(myGCNconv, self).__init__(in_channels, out_channels)
        print("GCN_autoencoder")
        # linear autoencoder
        self.lin = nn.Linear(in_channels, out_channels)
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
        outputs.append(torch.zeros(((end_idx-start_idx), mlp_group[0].out_channel)).to(self.unlabeled_indices.device)) # add features of unlabeled nodes
        for mlp_idx in range(len(mlp_group)):
            start_idx = None 
            end_idx = None 
            match_index = (sorted_antoencoder_indices == mlp_idx).nonzero(as_tuple=True)[0]
            if len(match_index)==0:
                continue 
            else:
                start_idx = match_index[0]
                end_idx = match_index[-1] + 1
            
            outputs.append(mlp_group[mlp_idx](sorted_x[start_idx:end_idx]))
        
        # resort the outputs
        processed = torch.cat(outputs, dim=0)
        inverse_indices = torch.argsort(sorted_indices)
        result = processed[inverse_indices]

        # self.add_self_loops = False 
        
        return result         
    
    def decoder_forward(self, x, index):
        
        mlp_group = self.decoder_group
        indices = self.antoencoder_indices[index]
        
        sorted_indices = torch.argsort(indices)
        sorted_antoencoder_indices = indices[sorted_indices]
        sorted_x = x[sorted_indices]
        
        # forward
        outputs = []
        # start_idx = (sorted_antoencoder_indices == -1 ).nonzero(as_tuple=True)[0][0] 
        # end_idx = (sorted_antoencoder_indices == -1).nonzero(as_tuple=True)[0][-1] + 1
        # outputs.append(torch.zeros(((end_idx-start_idx), mlp_group[0].out_channel)).to(self.unlabeled_indices.device)) # add features of unlabeled nodes
        for mlp_idx in range(len(mlp_group)):
            match_indices = (sorted_antoencoder_indices == mlp_idx).nonzero(as_tuple=True)[0]
            start_idx = None 
            end_idx = None 
            if len(match_indices) == 0:
                continue 
            else:
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
        self.add_self_loops = False
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
        
        x_clone = x.clone()
        
        loss = None
        if self.auto_encoder_loss_flag:
            loss = self.cal_autoencoder_loss(x)
        
        x = self.encoder_forward(x)
        # x = torch.stack([self.encoder_group[i](x[i]) for i in self.antoencoder_indices]).squeeze()
        

        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
        
        # if self.bias is not None:
        #     out = out + self.bias if out is not None else self.bias
        
        out = self.update_node(x_clone, out)

        if self.auto_encoder_loss_flag:
            self.auto_encoder_loss_flag = False
            return out, loss
        else:
            return out
    
    def update_node(self, x, neighbor_feature):
        if neighbor_feature is None:
            return F.relu(x)
        else: 
            return F.relu(x+neighbor_feature)
    
    def aggregate(self, inputs, index, ptr, dim_size):
        
        if len(inputs) == 0:
            return 
        else:
            inputs = self.decoder_forward(inputs, index)
            return scatter_mean(inputs, index, dim=0, dim_size=dim_size)

    def cal_autoencoder_loss(self, features):
        # |A x \tilede{A}-1|_2
        # loss = 0
        # for mlp_idx in range(len(self.encoder_group)):
        #     A = self.encoder_group[mlp_idx].fc.weight
        #     B = self.decoder_group[mlp_idx].fc.weight
        #     loss = loss + torch.norm(torch.mm(B,A) - torch.eye(B.size()[0]).to(A.device), p='fro')
        
        # reconstruction loss
        x = features.clone()
        loss = 0
        for mlp_idx in range(len(self.encoder_group)):
            x_idx = x[self.antoencoder_indices==mlp_idx]
            x_embedding = self.encoder_group[mlp_idx](x_idx)
            x_tilda = self.decoder_group[mlp_idx](x_embedding)
            # loss = F.mse_loss(x_idx, x_tilda, reduction='sum')/x_idx.size()[0] + loss
            loss = F.mse_loss(x_idx, x_tilda)
        return loss
    
    def restart(self):
        for encoder in self.encoder_group:
            encoder.restart()
        for decoder in self.decoder_group:
            decoder.restart()
  
class ourModel(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hidden_dim, 
                 output_dim, 
                 num_layers,
                 dropout):
        
        super(ourModel, self).__init__()
        
        self.convs = torch.nn.ModuleList(
            [myGCNconv(in_channels=input_dim, out_channels=hidden_dim, n_class=output_dim)] +
            [myGCNconv(in_channels=hidden_dim, out_channels=hidden_dim, n_class=output_dim)                             
                for i in range(num_layers-2)] + 
            [myGCNconv(in_channels=hidden_dim, out_channels=output_dim, n_class=output_dim)]    
        )
        # self.convs = torch.nn.ModuleList(
        #     [nn.Linear(input_dim, hidden_dim)] +
        #     [nn.Linear(hidden_dim, hidden_dim)                             
        #         for i in range(num_layers-2)] + 
        #     [nn.Linear(hidden_dim, output_dim)]    
        # )
        self.bns = torch.nn.ModuleList([
            torch.nn.BatchNorm1d(num_features=hidden_dim) 
                for i in range(num_layers-1)
        ])
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
    
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
                x1, sub_auto_encoder_loss = conv(x, adj_t)
                auto_encoder_loss = sub_auto_encoder_loss if auto_encoder_loss is None else auto_encoder_loss + sub_auto_encoder_loss
                    
            else:
                x1 = conv(x, adj_t)
            # x1 = F.relu(bn(x1))
            x1 = bn(x1)
            if self.training:
                x1 = F.dropout(x1, p=self.dropout)
            x = x1
            
        
            
            # if auto_encoder_loss_flag:
            #     auto_encoder_loss_flag = auto_encoder_loss + conv.cal_autoencoder_loss(x)
        
        if auto_encoder_loss_flag:   
            self.convs[-1].set_auto_encoder_loss_flag()
            x, sub_auto_encoder_loss = self.convs[-1](x, adj_t)
            auto_encoder_loss = auto_encoder_loss + sub_auto_encoder_loss
        else:
            x = self.convs[-1](x, adj_t)
        
        if auto_encoder_loss_flag:
            return x, auto_encoder_loss
        else:
            return x
        
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
        return x

class MLP(nn.Module):
    """ adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=.5):
        super(MLP, self).__init__()
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()
        if num_layers == 1:
            # just linear layer i.e. logistic regression
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout
        self.hidden_dim = hidden_channels
        self.num_layers = num_layers

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data, input_tensor=False):
        if not input_tensor:
            x = data.x
        else:
            x = data
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x

# model with only encoder
# class myGCNconv(GCNConv):
#     def __init__(self, in_channels, out_channels, n_class):
#         super(myGCNconv, self).__init__(in_channels, out_channels)
#         self.encoder_group = nn.ModuleList([Encoder(in_channel=out_channels, out_channel=out_channels) for _ in range(n_class)]) # 线性层(self.lin)已调整维度
#         self.decoder_group = nn.ModuleList([Decoder(in_channal=out_channels, out_channel=out_channels) for _ in range(n_class)])
        
#         self.auto_encoder_loss_flag = False
    
#     def set_antoencoder_index(self, antoencoder_indices):
#         self.antoencoder_indices = antoencoder_indices # pseudo node labels
    
#     def encoder_forward(self, x):
#         mlp_group = self.encoder_group
#         indices = self.antoencoder_indices
#         # group
#         sorted_indices = torch.argsort(indices) # indices[sorted_indices] = [-1,-1,-1,...0,0,0,...,1,1,1...]
#         sorted_antoencoder_indices = self.antoencoder_indices[sorted_indices]
#         sorted_x = x[sorted_indices] # x -> [-1,-1,...0,0,0,...1,1,1...](labels)
         
#         # forward
#         outputs = []
#         start_idx = (sorted_antoencoder_indices == -1).nonzero(as_tuple=True)[0][0]
#         end_idx = (sorted_antoencoder_indices == -1).nonzero(as_tuple=True)[0][-1] + 1
#         # outputs.append(sorted_x[start_idx:end_idx]) # add features of unlabeled nodes
#         outputs.append(torch.zeros_like(sorted_x[start_idx:end_idx])) # add features of unlabeled nodes
        
#         for mlp_idx in range(len(mlp_group)):
#             start_idx = (sorted_antoencoder_indices == mlp_idx).nonzero(as_tuple=True)[0][0]
#             end_idx = (sorted_antoencoder_indices == mlp_idx).nonzero(as_tuple=True)[0][-1] + 1
            
#             outputs.append(mlp_group[mlp_idx](sorted_x[start_idx:end_idx]))
        
#         # resort the outputs
#         processed = torch.cat(outputs, dim=0)
#         inverse_indices = torch.argsort(sorted_indices)
#         result = processed[inverse_indices]

#         # self.add_self_loops = False 
        
#         return result         
    
#     def decoder_forward(self, x, index):
#         mlp_group = self.decoder_group
#         indices = self.antoencoder_indices[index]
        
        
#         sorted_indices = torch.argsort(indices)
#         sorted_antoencoder_indices = indices[sorted_indices]
#         sorted_x = x[sorted_indices]
        
#         # forward
#         outputs = []
#         start_idx = (sorted_antoencoder_indices == -1 ).nonzero(as_tuple=True)[0][0]
#         end_idx = (sorted_antoencoder_indices == -1).nonzero(as_tuple=True)[0][-1] + 1
#         outputs.append(torch.zeros(((end_idx-start_idx), mlp_group[0].out_channel)).to(self.unlabeled_indices.device)) # add features of unlabeled nodes
#         for mlp_idx in range(len(mlp_group)):
#             start_idx = (sorted_antoencoder_indices == mlp_idx).nonzero(as_tuple=True)[0][0]
#             end_idx = (sorted_antoencoder_indices == mlp_idx).nonzero(as_tuple=True)[0][-1] + 1
            
#             outputs.append(mlp_group[mlp_idx](sorted_x[start_idx:end_idx]))
        
#         # resort the outputs
#         processed = torch.cat(outputs, dim=0)
#         inverse_indices = torch.argsort(sorted_indices)
#         result = processed[inverse_indices]

#         return result         
    
#     def set_auto_encoder_loss_flag(self):
#         self.auto_encoder_loss_flag = True
    
#     def forward(self, x: Tensor, edge_index,
#         edge_weight: OptTensor = None) -> Tensor:
#         self.add_self_loops = False
        
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
        
#         loss = None
#         if self.auto_encoder_loss_flag:
#             loss = self.cal_autoencoder_loss(x)
        
#         x = self.encoder_forward(x)
#         # x = torch.stack([self.encoder_group[i](x[i]) for i in self.antoencoder_indices]).squeeze()
        
#         # propagate_type: (x: Tensor, edge_weight: OptTensor)

#         # print(x.size())
#         out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
#                              size=None)

#         # out[self.unlabeled_indices] = self.unlabeled_features
        
#         if self.bias is not None:
#             out = out + self.bias

#         if self.auto_encoder_loss_flag:
#             self.auto_encoder_loss_flag = False
#             return out, loss
#         else:
#             return out
    
#     # def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
#     #     x_j = super().message(x_j, edge_weight)
#     #     return x_j
#         # return torch.stack([self.encoder_group[i](x_j[i]) for i in self.antoencoder_index])
    
#     def aggregate(self, inputs, index, ptr, dim_size):
#         # inputs = torch.stack([self.decoder_group[self.antoencoder_index_node[input_node_num]](inputs[i]) for i,input_node_num in enumerate(index)])
#         inputs = self.decoder_forward(inputs, index)
#         # return scatter_mean(inputs, index, dim=0, dim_size=dim_size)
        
#         return scatter_mean(inputs, index, dim=0, dim_size=dim_size)

#     def cal_autoencoder_loss(self, features):
#         # |A x \tilede{A}-1|_2
#         # loss = 0
#         # for mlp_idx in range(len(self.encoder_group)):
#         #     A = self.encoder_group[mlp_idx].fc.weight
#         #     B = self.decoder_group[mlp_idx].fc.weight
#         #     loss = loss + torch.norm(torch.mm(B,A) - torch.eye(B.size()[0]).to(A.device), p='fro')
        
#         # reconstruction loss
#         x = features.clone()
#         loss = 0
#         for mlp_idx in range(len(self.encoder_group)):
#             x_idx = x[self.antoencoder_indices==mlp_idx]
#             x_embedding = self.encoder_group[mlp_idx](x_idx)
#             x_tilda = self.decoder_group[mlp_idx](x_embedding)
#             # loss = F.mse_loss(x_idx, x_tilda, reduction='sum')/x_idx.size()[0] + loss
#             loss = F.mse_loss(x_idx, x_tilda)
#         return loss
    
#     def restart(self):
#         for encoder in self.encoder_group:
#             encoder.restart()
#         for decoder in self.decoder_group:
#             decoder.restart()
    




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
    
