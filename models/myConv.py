from Non_Homophily_Large_Scale.models import *
from models.autoencoder import Encoder, Decoder
from torch_scatter import scatter_mean

    
class myConv(SAGEConv):
    def __init__(self, in_channels, out_channels, n_class, soft_flag):
        super().__init__(in_channels, out_channels)
        # self.encoder_group = nn.ModuleList([Encoder(in_channels) for _ in range(n_class)])
        # self.decoder_group = nn.ModuleList([Decoder(in_channels) for _ in range(n_class)])
        # self.soft_flag = soft_flag

    def forward(self, graph):
        self.graph = graph
        x = graph.x.clone() 
        
        edge_index = graph.edge_index 
        
        autoencoder_loss = None
        if self.auto_encoder_loss_flag:
            autoencoder_loss = self.cal_autoencoder_loss(x)
        
        x_clone = x.clone() 
        out = self.propagate(edge_index, x=x, size=None)
        
        out = self.lin_l(out)
        out = out + self.lin_r(x_clone) 

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        self.graph = None 
        if self.auto_encoder_loss_flag:
            self.auto_encoder_loss_flag = False
            return out, autoencoder_loss
        else:
            return out

    def decoder_forward(self, x, index, select_edge_indicies):
        # head <-- tail
        head_node_confidence = self.node_confidence[index]
        tail_node_confidence = self.node_confidence[self.graph.edge_index[0,:]][select_edge_indicies]
        weights = torch.einsum('ab,ab->a',head_node_confidence, tail_node_confidence)
        inputs = torch.einsum('ab,a->ab', x, weights)
        return inputs
        
    def aggregate(self, inputs, index, ptr, dim_size):
        if len(inputs) == 0:
            return 
        else:
            inputs, index, select_edge_indicies = self.select_edge(inputs, index)
            inputs = self.decoder_forward(inputs, index, select_edge_indicies)
            return scatter_mean(inputs, index, dim=0, dim_size=dim_size)
    
    def cal_autoencoder_loss(self, features):
        
        return torch.tensor(0) 

    
    def set_auto_encoder_loss_flag(self):
        self.auto_encoder_loss_flag = True
    
    def set_node_confidence(self, logits):
        self.node_confidence = torch.softmax(logits, dim=1) 
        
    def select_edge(self, inputs, index):
        
        edge_label = self.graph.edge_pseudolabel
        propogated_confidence_from = self.graph.propogated_confidence_from
        threshold = self.graph.edge_threshold
        
        select_edge_indicies = propogated_confidence_from>threshold
        
        new_inputs = inputs[select_edge_indicies]
        new_index = index[select_edge_indicies]
        
        return new_inputs, new_index, select_edge_indicies
    


# class myConv(SAGEConv):
#     def __init__(self, in_channels, out_channels, n_class, soft_flag):
#         super().__init__(in_channels, out_channels)
#         self.encoder_group = nn.ModuleList([Encoder(in_channels) for _ in range(n_class)])
#         self.decoder_group = nn.ModuleList([Decoder(in_channels) for _ in range(n_class)])
#         self.soft_flag = soft_flag

#     def forward(self, graph):
#         self.graph = graph
#         x = graph.x.clone() 
        
#         edge_index = graph.edge_index 
        
#         autoencoder_loss = None
#         if self.auto_encoder_loss_flag:
#             autoencoder_loss = self.cal_autoencoder_loss(x)
        
#         x_clone = x.clone() 
#         x = self.encoder_forward(x)
#         out = self.propagate(edge_index, x=x, size=None)
        
#         out = self.lin_l(out)
#         out = out + self.lin_r(x_clone) 

#         if self.normalize:
#             out = F.normalize(out, p=2., dim=-1)

#         self.graph = None 
#         if self.auto_encoder_loss_flag:
#             self.auto_encoder_loss_flag = False
#             return out, autoencoder_loss
#         else:
#             return out
    
#     def encoder_forward(self, x):
#         if self.soft_flag:
#             # TODO:based on combination
#             combination_weight = self.node_confidence.unsqueeze(-1)
#             output = [self.encoder_group[i](x) for i in range(len(self.encoder_group))]
#             output = torch.stack(output) # c * N * d
#             output = output.permute(1,0,2) # N * c * dr
#             res = output * combination_weight
#             return res.sum(dim=1) # N * d
        
#         else:
#             # based on edge_pseudolabel
#             mlp_group = self.encoder_group
#             indices = self.graph.edge_pseudolabel
#             # group
#             sorted_indices = torch.argsort(indices)
#             sorted_antoencoder_indices = indices[sorted_indices]
#             sorted_x = x[sorted_indices]
            
#             # forward
#             outputs = []
#             if -1 in sorted_antoencoder_indices:
#                 start_idx = (sorted_antoencoder_indices == -1).nonzero(as_tuple=True)[0][0]
#                 end_idx = (sorted_antoencoder_indices == -1).nonzero(as_tuple=True)[0][-1] + 1
#                 self.unlabeled_features = sorted_x[start_idx:end_idx] # save for reconstruction
#                 self.unlabeled_indices = sorted_indices[start_idx:end_idx]
#                 outputs.append(torch.zeros(((end_idx-start_idx), mlp_group[0].out_channel)).to(self.unlabeled_indices.device)) # add features of unlabeled nodes
#             for mlp_idx in range(len(mlp_group)):
#                 start_idx = None 
#                 end_idx = None 
#                 match_index = (sorted_antoencoder_indices == mlp_idx).nonzero(as_tuple=True)[0]
#                 if len(match_index)==0:
#                     continue 
#                 else:
#                     start_idx = match_index[0]
#                     end_idx = match_index[-1] + 1
                
#                 outputs.append(mlp_group[mlp_idx](sorted_x[start_idx:end_idx]))
            
#             # resort the outputs
#             processed = torch.cat(outputs, dim=0)
#             inverse_indices = torch.argsort(sorted_indices)
#             result = processed[inverse_indices]

#             # self.add_self_loops = False 
            
#             return result         
    
#     def decoder_forward(self, x, index):
#         if self.soft_flag:
#             # TODO:based on combination
#             combination_weight = self.node_confidence[index].unsqueeze(-1)
#             output = [self.decoder_group[i](x) for i in range(len(self.decoder_group))]
#             output = torch.stack(output) # c * N * d
#             output = output.permute(1,0,2) # N * c * d
#             res = output * combination_weight 
#             return res.sum(dim=1)

#         else:
#             # base on edge_pseudolabel
#             mlp_group = self.decoder_group
#             indices = self.graph.edge_pseudolabel[index]
            
#             sorted_indices = torch.argsort(indices)
#             sorted_antoencoder_indices = indices[sorted_indices]
#             sorted_x = x[sorted_indices]
            
#             # forward
#             outputs = []
#             if -1 in sorted_antoencoder_indices:
#                 start_idx = (sorted_antoencoder_indices == -1 ).nonzero(as_tuple=True)[0][0] 
#                 end_idx = (sorted_antoencoder_indices == -1).nonzero(as_tuple=True)[0][-1] + 1
#                 outputs.append(torch.zeros(((end_idx-start_idx), mlp_group[0].out_channel)).to(self.unlabeled_indices.device)) # add features of unlabeled nodes
            
            
#             for mlp_idx in range(len(mlp_group)):
#                 match_indices = (sorted_antoencoder_indices == mlp_idx).nonzero(as_tuple=True)[0]
#                 start_idx = None 
#                 end_idx = None 
#                 if len(match_indices) == 0:
#                     continue 
#                 else:
#                     start_idx = (sorted_antoencoder_indices == mlp_idx).nonzero(as_tuple=True)[0][0]
#                     end_idx = (sorted_antoencoder_indices == mlp_idx).nonzero(as_tuple=True)[0][-1] + 1
                
#                 outputs.append(mlp_group[mlp_idx](sorted_x[start_idx:end_idx]))
            
#             # resort the outputs
#             processed = torch.cat(outputs, dim=0)
#             inverse_indices = torch.argsort(sorted_indices)
#             result = processed[inverse_indices]

#             return result         
    
#     def aggregate(self, inputs, index, ptr, dim_size):
#         if len(inputs) == 0:
#             return 
#         else:
#             inputs, index = self.select_edge(inputs, index)
#             inputs = self.decoder_forward(inputs, index)
#             return scatter_mean(inputs, index, dim=0, dim_size=dim_size)
    
#     def cal_autoencoder_loss(self, features):
        
#         # return torch.tensor(0) # TODO:暂时废弃，后续如需算则考虑替换training_labels
#         # |A x \tilede{A}-1|_2
#         loss = 0
#         for mlp_idx in range(len(self.encoder_group)):
#             A = self.encoder_group[mlp_idx].fc.weight
#             B = self.decoder_group[mlp_idx].fc.weight
#             loss = loss + torch.norm(torch.mm(B,A) - torch.eye(B.size()[0]).to(A.device), p='fro')
        
#         # reconstruction loss
#         label_for_autoencoder_loss = torch.zeros_like(self.graph.y)-1
#         label_for_autoencoder_loss[self.graph.train_index_A] = self.graph.edge_pseudolabel[self.graph.train_index_A]
        
#         x = features.clone()
#         loss = 0
#         for mlp_idx in range(len(self.encoder_group)):
#             x_idx = x[label_for_autoencoder_loss==mlp_idx]
#             x_embedding = self.encoder_group[mlp_idx](x_idx)
#             x_tilda = self.decoder_group[mlp_idx](x_embedding)
#             # loss = F.mse_loss(x_idx, x_tilda, reduction='sum')/x_idx.size()[0] + loss
#             loss = F.mse_loss(x_idx, x_tilda)
#         return loss
    
#     def set_auto_encoder_loss_flag(self):
#         self.auto_encoder_loss_flag = True
    
#     def set_node_confidence(self, logits):
#         self.node_confidence = torch.softmax(logits, dim=1) 
        
#     def select_edge(self, inputs, index):
        
#         edge_label = self.graph.edge_pseudolabel
#         propogated_confidence_from = self.graph.propogated_confidence_from
#         threshold = self.graph.edge_threshold
        
#         select_edge_indicies = propogated_confidence_from>threshold
        
#         new_inputs = inputs[select_edge_indicies]
#         new_index = index[select_edge_indicies]
        
#         return new_inputs, new_index
    
