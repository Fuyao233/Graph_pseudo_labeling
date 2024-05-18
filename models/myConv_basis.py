from Non_Homophily_Large_Scale.models import *
from torch_scatter import scatter_mean

class myConv_basis(SAGEConv):
    def __init__(self, in_channels, out_channels, n_class, cross_basis_matrix, basis_num, soft_flag):
        super().__init__(out_channels, out_channels)
        self.cross_basis_matrix = cross_basis_matrix # c * c * (2**{N//2}) * d * d
        self.lin = nn.Linear(in_channels, out_channels)
        self.N = basis_num
        self.c = n_class
        
        c = n_class
        N = basis_num
        par_x = torch.zeros((c,c,N//2))
        par_y = torch.zeros((c,c,N//2))
        for i in range(c):
            for j in range(i+1, c):
                par_x[i][j] = torch.randn(N//2)
                par_y[i][j] = torch.randn(N//2)
                
        self.par_x = nn.Parameter(par_x)
        self.par_y = nn.Parameter(par_y)


    def produce_message_passing_function(self):
        c = self.c 
        N = self.N 
        
        par_x = self.par_x 
        par_y = self.par_y
        device = par_x.device
        par_n = torch.zeros((c,c,N//2), device=device)
        par_m = torch.zeros((c,c,N//2), device=device)
        for i in range(c):
            for j in range(i+1, c):

                # TODO: 处理可能的除零
                x_q = par_x[i][j]**2
                y_q = par_y[i][j]**2
                
                par_n[j][i] = (par_y[i][j] / (y_q-x_q)).flip(0)
                par_m[j][i] = (par_x[i][j] / (x_q-y_q)).flip(0)
                
                x = par_x[i][j]
                y = par_y[i][j]
                n = par_n[j][i]
                m = par_m[j][i]
                
                m = m

        # par_m = nn.Parameter(par_m)
        # par_n = nn.Parameter(par_n)
        
        # cross weight
        cross_weight = torch.empty((c, c, 2**(N//2)), device=device)
        for i in range(c):
            for j in range(c):
                if i == j:
                    cross_weight[i][j] = 0
                else:
                    for idx in range(2**(N//2)):
                        index_sequence = [(idx >> k) & 1 for k in range(N//2)]

                        select_sequence = None 
                        if i<j:
                            x_sequence = par_x[i][j]
                            y_sequence = par_y[i][j]
                            # bit==0 -> x; bit==1 -> y
                            select_sequence = [(x_sequence if bit==0 else y_sequence)[k] for k, bit in enumerate(index_sequence)]
                        else:
                            n_sequence = par_n[i][j]
                            m_sequence = par_m[i][j]
                            # bit==0 -> n; bit==1 -> m
                            select_sequence = [(n_sequence if bit==0 else m_sequence)[k] for k, bit in enumerate(index_sequence)]
                        
                        result = torch.prod(torch.stack(select_sequence),dim=0)
                        
                        cross_weight[i][j][idx] = result
        # cross_weight = nn.Parameter(cross_weight)   
        # passing_function
        self.cross_basis_matrix = self.cross_basis_matrix.to(device)
        passing_function = torch.einsum('...i,...ijk->...jk', cross_weight, self.cross_basis_matrix)
        # passing_fucntion = nn.Parameter(passing_fucntion)
        
        self.cross_weight = cross_weight
        self.passing_function = passing_function
        self.par_n = par_n
        self.par_m = par_m 
        
    def forward(self, graph):
        self.graph = graph
        x = graph.x.clone() 
        x = self.lin(x)
        
        edge_index = graph.edge_index 
        
        x_clone = x.clone() 
        
        out = self.propagate(edge_index, x=x, size=None)
        
        out = self.lin_l(out)
        out = out + self.lin_r(x_clone) 

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        self.graph = None 
        return out

    def passing(self, inputs, src_index, dst_index):
        node_label = self.graph.edge_pseudolabel 
        c = self.c 
        
        src_label = node_label[src_index]
        dst_label = node_label[dst_index]
        indices_list = [[[torch.where(torch.logical_and(src_label==i, dst_label==j))[0]] if not i==j else [] for j in range(c)] for i in range(c)]
        
        for i in range(c):
            for j in range(c):
                if i != j:
                    indices = indices_list[i][j]
                    inputs[indices] = inputs[indices] @ self.passing_function[i][j]
        
        return inputs
    
    def aggregate(self, inputs, index, ptr, dim_size):
        if len(inputs) == 0:
            return 
        else:
            inputs, src_index, dst_index = self.select_edge(inputs, index)
            inputs = self.passing(inputs, src_index, dst_index)
            return scatter_mean(inputs, dst_index, dim=0, dim_size=dim_size)

    # def set_node_confidence(self, logits):
    #     self.node_confidence = torch.softmax(logits, dim=1) 
        
    def select_edge(self, inputs, index):

        edge_label = self.graph.edge_pseudolabel
        propogated_confidence_from = self.graph.propogated_confidence_from
        threshold = self.graph.edge_threshold
        
        select_edge_indicies = propogated_confidence_from>threshold
        
        new_inputs = inputs[select_edge_indicies]
        dst_index = index[select_edge_indicies]
        scr_index = self.graph.edge_index[0,:][select_edge_indicies]
        
        return new_inputs, scr_index, dst_index
    