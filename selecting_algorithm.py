import torch 
from copy import deepcopy
import random
import time
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils import enable_dropout, accuracy

class Flexmatch:
    def __init__(self, graph, prediction, node_threshold, edge_threshold, batch_size, warm_up):
        self.graph = graph
        self.prediction = prediction
        self.node_threshold = node_threshold 
        self.edge_threshold = edge_threshold 
        self.batch_size = batch_size
        self.warm_up = warm_up

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
        # print(f'Node_threshold: {self.node_threshold}; Edge_threshold: {self.edge_threshold}')
        print(f'Node_threshold: {self.node_threshold}')
        self.graph.edge_threshold = self.edge_threshold
        # naive, fixed xthreshold
        confidence, y_hat = self.prediction.max(dim=1)
        self.graph.full_confidence = torch.softmax(self.prediction, dim=1).detach()
        
        self.graph.label_confidence = confidence.detach().clone()
        self.graph.label_confidence[self.graph.train_index_A] = 1.
        self.graph.edge_pseudolabel = y_hat.detach().clone()
        self.graph.edge_pseudolabel[self.graph.train_index_A] = self.graph.y[self.graph.train_index_A].clone() # edge pseudolabel = y_hat + train_index_A
        self.graph.propogated_confidence_from = self.graph.label_confidence[self.graph.edge_index[0]]
        self.graph.propogated_confidence_to = self.graph.label_confidence[self.graph.edge_index[1]]
        
        node_unlabeled_index = (self.graph.unlabeled_index) * (self.graph.node_pseudolabel == -1)
        
        node_indices = torch.zeros_like(self.graph.y)==1
        edge_indices = torch.zeros_like(self.graph.y)==1
        
        for c in range(self.graph.num_class):
            selected_node_indices = torch.where(node_unlabeled_index*(y_hat==c)*(confidence>self.node_threshold))[0]
            self.graph.node_pseudolabel[selected_node_indices] = c
            node_indices[selected_node_indices] = True
        
        # set all node_pseudolabel as part of train_index_A
        # self.graph.label_confidence[node_indices] = 1.
        
        # split node_pseudolabel (based on the last iteration)
        if 'node_pseudolabel_indices_A' not in self.graph:
            self.graph.node_pseudolabel_indices_A = torch.zeros_like(self.graph.node_pseudolabel, dtype=bool)
            self.graph.node_pseudolabel_indices_B = torch.zeros_like(self.graph.node_pseudolabel, dtype=bool)
        node_pseudolabel_indices = torch.where(self.graph.node_pseudolabel>=0)[0]
        # random_indices = torch.randperm(len(node_pseudolabel_indices))
        # self.graph.node_pseudolabel_indices_A[node_pseudolabel_indices[random_indices[:len(node_pseudolabel_indices)//2]]] = True 
        # self.graph.label_confidence[self.graph.node_pseudolabel_indices_A] = 1.
        # self.graph.node_pseudolabel_indices_B[node_pseudolabel_indices[random_indices[len(node_pseudolabel_indices)//2:]]] = True 
        
        self.graph.label_confidence[node_pseudolabel_indices] = 1.
        self.graph.edge_pseudolabel[node_pseudolabel_indices] = self.graph.node_pseudolabel[node_pseudolabel_indices]
        
        node_pseudo_label_acc = torch.mean((y_hat[node_indices] == self.graph.y[node_indices])*1.)
        edge_pseudo_label_acc = torch.mean((y_hat[edge_indices] == self.graph.y[edge_indices])*1.)
        
        if torch.sum(node_indices)==0:
            print(f'\nAdd no pseudo labels.\n')
        else:
            print(f'\nNumber of new node labels: {torch.sum(node_indices)}; Accuracy of pseudo node labels when selecting: {node_pseudo_label_acc.item()}\n')
            # print(f'\nNumber of new edge labels: {torch.sum(edge_indices)}; Accuracy of pseudo edge labels when selecting: {edge_pseudo_label_acc.item()}\n')
        
        
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
        

class UPS:
    def __init__(self):
        pass
    
    def select(self, args, graph, model, itr):
        pseudo_idx = []
        pseudo_target = []
        pseudo_maxstd = []
        gt_target = []
        idx_list = []
        gt_list = []
        target_list = []
        nl_mask = []
        model.eval()
        if not args.no_uncertainty:
            f_pass = 10
            enable_dropout(model)
        else:
            f_pass = 1

        indexs = torch.arange(graph.num_nodes).to(args.gpu)
        indexs = indexs[graph.unlabeled_index]
        inputs = graph.to(args.gpu)
        targets = graph.y.to(args.gpu)[graph.unlabeled_index]
        
        with torch.no_grad():

            out_prob = []
            out_prob_nl = []
            for _ in range(f_pass):
                outputs = model(inputs)[graph.unlabeled_index]
                out_prob.append(F.softmax(outputs, dim=1)) #for selecting positive pseudo-labels
                out_prob_nl.append(F.softmax(outputs/args.temp_nl, dim=1)) #for selecting negative pseudo-labels
            out_prob = torch.stack(out_prob)
            out_prob_nl = torch.stack(out_prob_nl)
            out_std = torch.std(out_prob, dim=0)
            out_std_nl = torch.std(out_prob_nl, dim=0)
            out_prob = torch.mean(out_prob, dim=0)
            out_prob_nl = torch.mean(out_prob_nl, dim=0)
            max_value, max_idx = torch.max(out_prob, dim=1)
            max_std = out_std.gather(1, max_idx.view(-1,1))
            out_std_nl = out_std_nl.cpu().numpy()
            
            #selecting negative pseudo-labels
            interm_nl_mask = ((out_std_nl < args.kappa_n) * (out_prob_nl.cpu().numpy() < args.tau_n)) *1

            #manually setting the argmax value to zero
            for enum, item in enumerate(max_idx.cpu().numpy()):
                interm_nl_mask[enum, item] = 0
            nl_mask.extend(interm_nl_mask)

            idx_list.extend(indexs.cpu().numpy().tolist())
            gt_list.extend(targets.cpu().numpy().tolist())
            target_list.extend(max_idx.cpu().numpy().tolist())

            #selecting positive pseudo-labels
            if not args.no_uncertainty:
                selected_idx = ((max_value>=args.tau_p) * (max_std.squeeze(1) < args.kappa_p))
            else:
                selected_idx = max_value>=args.tau_p

            pseudo_maxstd.extend(max_std.squeeze(1)[selected_idx].cpu().numpy().tolist())
            pseudo_target.extend(max_idx[selected_idx].cpu().numpy().tolist())
            pseudo_idx.extend(indexs[selected_idx].cpu().numpy().tolist())
            gt_target.extend(targets[selected_idx].cpu().numpy().tolist())

            loss = F.cross_entropy(outputs, targets.to(dtype=torch.long))
            prec1 = accuracy(outputs[selected_idx], targets[selected_idx])


        pseudo_target = np.array(pseudo_target)
        gt_target = np.array(gt_target)
        pseudo_maxstd = np.array(pseudo_maxstd)
        pseudo_idx = np.array(pseudo_idx)

        #class balance the selected pseudo-labels
        if itr < args.class_blnc-1:
            min_count = 5000000 #arbitary large value
            for class_idx in range(graph.num_class):
                class_len = len(np.where(pseudo_target==class_idx)[0])
                if class_len < min_count:
                    min_count = class_len
            min_count = max(25, min_count) #this 25 is used to avoid degenarate cases when the minimum count for a certain class is very low

            blnc_idx_list = []
            for class_idx in range(graph.num_class):
                current_class_idx = np.where(pseudo_target==class_idx)
                if len(np.where(pseudo_target==class_idx)[0]) > 0:
                    current_class_maxstd = pseudo_maxstd[current_class_idx]
                    sorted_maxstd_idx = np.argsort(current_class_maxstd)
                    current_class_idx = current_class_idx[0][sorted_maxstd_idx[:min_count]] #select the samples with lowest uncertainty 
                    blnc_idx_list.extend(current_class_idx)

            blnc_idx_list = np.array(blnc_idx_list)
            pseudo_target = pseudo_target[blnc_idx_list]
            pseudo_idx = pseudo_idx[blnc_idx_list]
            gt_target = gt_target[blnc_idx_list]

        pseudo_labeling_acc = (pseudo_target == gt_target)*1
        pseudo_labeling_acc = (sum(pseudo_labeling_acc)/len(pseudo_labeling_acc))*100
        print(f'Pseudo-Labeling Accuracy (positive): {pseudo_labeling_acc}, Total Selected: {len(pseudo_idx)}')

        pseudo_nl_mask = []
        pseudo_nl_idx = []
        nl_gt_list = []

        for i in range(len(idx_list)):
            if idx_list[i] not in pseudo_idx and sum(nl_mask[i]) > 0:
                pseudo_nl_mask.append(nl_mask[i])
                pseudo_nl_idx.append(idx_list[i])
                nl_gt_list.append(gt_list[i])

        nl_gt_list = np.array(nl_gt_list)
        pseudo_nl_mask = np.array(pseudo_nl_mask)
        one_hot_targets = np.eye(graph.num_class)[nl_gt_list]
        one_hot_targets = one_hot_targets - 1
        one_hot_targets = np.abs(one_hot_targets)
        flat_pseudo_nl_mask = pseudo_nl_mask.reshape(1,-1)[0]
        flat_one_hot_targets = one_hot_targets.reshape(1,-1)[0]
        flat_one_hot_targets = flat_one_hot_targets[np.where(flat_pseudo_nl_mask == 1)]
        flat_pseudo_nl_mask = flat_pseudo_nl_mask[np.where(flat_pseudo_nl_mask == 1)]

        nl_accuracy = (flat_pseudo_nl_mask == flat_one_hot_targets)*1
        nl_accuracy_final = (sum(nl_accuracy)/len(nl_accuracy))*100
        print(f'Pseudo-Labeling Accuracy (negative): {nl_accuracy_final}, Total Selected: {len(nl_accuracy)}, Unique Samples: {len(pseudo_nl_mask)}')
        pseudo_label_dict = {'pseudo_idx': pseudo_idx.tolist(), 'pseudo_target':pseudo_target.tolist(), 'nl_idx': pseudo_nl_idx, 'nl_mask': pseudo_nl_mask.tolist()}
        
        graph.pseudo_label_dict = pseudo_label_dict
        pseudolabel_ts = torch.tensor(pseudo_target).to(args.gpu)
        graph.edge_pseudolabel[torch.tensor(pseudo_idx)] = pseudolabel_ts
        graph.node_pseudolabel[torch.tensor(pseudo_idx)] = pseudolabel_ts
        
        return prec1, pseudo_labeling_acc, len(pseudo_idx), nl_accuracy_final, len(nl_accuracy), len(pseudo_nl_mask), pseudo_label_dict   
        

# def uncertainty_aware(graph, prediction, tau):
#     sigma = prediction> 


# def select_train_data(graph, prediction, strategy):
#     """
#     add unlabelled samples with high confidence to training data
#     """
#     if strategy == 'Flexmatch':
            
#     else:
#         raise ValueError('Invalid selecting method!')