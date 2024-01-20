import torch 

class Flexmatch:
    def __init__(self, graph, prediction, tau, batch_size):
        self.graph = graph
        self.prediction = prediction
        self.tau = tau # fixed threshold
        self.batch_size = batch_size

    def is_warm_up(self):
        warm_up_flags = torch.zeros(self.graph.num_class)
        criterion = torch.sum(self.graph.pseudolabel==-1)
        for i, _ in enumerate(warm_up_flags):
            warm_up_flags[i] = torch.sum(self.graph.pseudolabel==i) < criterion # 'True' means it needs warm up
        return warm_up_flags 

    def get_sigma_c(self):
        unlabel_prediction = self.prediction[self.graph.pseudolabel==-1]
        maximum, indices = unlabel_prediction.max(dim=1)
        sigma_c = torch.zeros(self.graph.num_class)
        for i, _ in enumerate(sigma_c):
            sigma_c[i] = torch.sum((indices==i)*(maximum>self.tau))
        return sigma_c

    def get_batch(self):
        unlabeled_indices = torch.where(self.graph.pseudolabel==-1)[0]
        selected_indices = None
        if len(unlabeled_indices) <= self.batch_size:
            selected_indices = unlabeled_indices
        else:
            shuffled_indices = torch.randperm(len(unlabeled_indices))[:self.batch_size]
            selected_indices = unlabeled_indices[shuffled_indices]
        res = torch.zeros_like(self.graph.pseudolabel)
        res[selected_indices] = 1
        return res==1 # 'True' means selected sample 
    
    def select(self):
        print(f'Threshold: {self.tau}')
        # naive, fixed threshold
        confidence, y_hat = self.prediction.max(dim=1)
        # unlabeled_index = self.graph.pseudolabel == -1 # potentially add validation nodes
        unlabeled_index = self.graph.test_index * (self.graph.pseudolabel == -1)
        
        indices = torch.zeros_like(self.graph.y)==1
        
        for c in range(self.graph.num_class):
            selected_indices = torch.where(unlabeled_index*(y_hat==c)*(confidence>self.tau))[0]
            # selected_indices = initial_indices
            self.graph.pseudolabel[selected_indices] = c
            
            self.graph.val_index[selected_indices] = False # dynamically adjust validation set
            
            indices[selected_indices] = True
        
        pseudo_label_acc = torch.mean((y_hat[indices] == self.graph.y[indices])*1.)
        if torch.sum(indices)==0:
            print(f'\nAdd no pseudo labels.\n')
        else:
            print(f'\nNumber of new labels: {torch.sum(indices)}; Accuracy of pseudo labels when selecting: {pseudo_label_acc.item()}\n')
        
        
        # old method
        # warm_up_flags = self.is_warm_up()
        # tau_c = torch.zeros(self.graph.num_class)
        # for i in range(len(tau_c)):
        #     sigma_c = self.get_sigma_c()
        #     beta_c = None
        #     if warm_up_flags[i]:
        #         # warm up
        #         beta_c = sigma_c/(max(sigma_c.max(), self.graph.num_nodes-torch.sum(sigma_c)))
        #     else:
        #         beta_c = sigma_c/sigma_c.max()
        #     tau_c[i] = (beta_c*self.tau)[i]
        
        # confidence, y_hat = self.prediction.max(dim=1)
        
        # unlabeled_index_batch = self.get_batch()
        # # unlabeled_index = self.graph.pseudolabel == -1
        # for c in range(self.graph.num_class):
        #     self.graph.pseudolabel[unlabeled_index_batch*(y_hat==c)*(confidence>tau_c[c])] = c 
            # print(f'class {c} add {torch.sum(unlabeled_index*(y_hat==c)*(confidence>tau_c[c]))} samples')
        

        
    
    

# def uncertainty_aware(graph, prediction, tau):
#     sigma = prediction> 


# def select_train_data(graph, prediction, strategy):
#     """
#     add unlabelled samples with high confidence to training data
#     """
#     if strategy == 'Flexmatch':
            
#     else:
#         raise ValueError('Invalid selecting method!')