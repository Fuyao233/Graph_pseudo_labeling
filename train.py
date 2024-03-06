import torch
from torch_geometric.data import Data
from utils import *
from model import GCN, ourModel, MLP
from tqdm import tqdm
from torch.optim import Adam, SGD, AdamW
import torch.nn.functional as F
from selecting_algorithm import Flexmatch, UPS
import os 
import pandas as pd 
import argparse
from copy import deepcopy
import numpy as np
import torch.nn as nn
import yaml
from copy import deepcopy
import pickle
import random

class Trainer:
    def __init__(self, graph, model, device_num, model_name,
                 lr, momentum, weight_decay,  
                 flexmatch_weight,
                 flex_batch,
                 autoencoder_weight,
                 metric,
                 args,
                 select_method='flexmatch') :
        self.graph = graph 
        self.model = model 
        self.model_name = model_name
        self.args = args 
        
        self.lr = lr
        self.momentum = momentum 
        self.weight_decay = weight_decay 
        
        self.metric = metric
        # self.weight_decay = weight_decay
        self.edge_threshold = args.edge_threshold # to select edge label
        self.node_threshold = args.node_threshold # to select node label
        self.method = 'flexmatch'
        self.flexmatch_weight = flexmatch_weight # weight for flexmatch loss
        # self.flex_batch = int(self.graph.num_nodes*flex_batch_ratio) # select from a random batch of unlabeled samples each time updating training samples 
        self.flex_batch = flex_batch # select from a random batch of unlabeled samples each time updating training samples 
        
        self.edge_status_list = []
        # self.update_stage_threshold = 0.005
        
        self.autoencoder_weight = autoencoder_weight
        
        self.warm_up = args.warm_up
        
        torch.cuda.set_device(device_num)
        device = torch.cuda.current_device()
        self.graph.to(device)
        self.model.to(device)
            
        self.graph.node_pseudolabel = torch.zeros_like(graph.y)-2 # '-2' means unavailable data
        self.graph.node_pseudolabel[self.graph.unlabeled_index] = -1 # '-1' means unlabeled data
        self.graph.edge_pseudolabel = torch.zeros_like(graph.y) - 1 # '-1' means unlabeled
        

        # record the best model during the whole training process 
        self.best_model = None # best model among all best_model_iter
        self.best_graph = None # graph according to the best model(some edges may be omitted)
        self.best_val_metric = -torch.inf
        self.best_training_num_record = 0 # record the num of training samples of the best model
        self.global_iteration = 0 # number of adding samples 
        
        # self.update_autoencoder_prob()
        # print(self.graph.autoencoder_prob)
        # self.update_pipeline_flag = 0
        self.initialize_training_labels()
        self.update_training_graph()
        
        self.stopper = EarlyStopper(max_iter=300)
    
    def eval(self, model, graph, keyword='test', metric='auc'):
        model.eval()
        key = None
        if keyword == 'test':
            key = 'test_index'
        elif keyword == 'val':
            key = 'val_index'
        else:
            raise ValueError('Wrong keyword!')
        
        with torch.no_grad():
            logits = model(graph)
            y_hat = torch.argmax(logits, dim=1)
             
            if metric == 'accuracy':
                metric = torch.mean((y_hat[self.graph[key]]==self.graph.y[self.graph[key]]).float()).item()
            elif metric == 'auc':
                # metric = roc_auc_score(self.graph.y[self.graph[key]].cpu().numpy(),torch.softmax(logits[self.graph[key]], dim=1)[:,1].cpu().detach().numpy()) 
                metric = eval_rocauc(graph.y[graph[key]].reshape((-1,1)), logits[graph[key]])
            else:
                assert metric in ['accuracy', 'auc']
            
            # if keyword == 'test':
            #     print(f'{key} {metric} is {metric:.3f}')

        # auc_score = roc_auc_score(graph.y[graph[key]].cpu().numpy(),torch.softmax(logits[graph[key]],dim=1)[:,1].cpu().detach().numpy()) 
        return y_hat, metric
    
    def if_edge_change(self):
        if len(self.edge_status_list)==0 or self.update_pipeline_flag == self.edge_status_list[-1]:
            pass 
        else:
            # status stage changes
            # use the graph achieving the best metric
            self.training_graph.train_index = self.best_graph.train_index.clone()
            self.training_graph.test_index = self.best_graph.test_index.clone()
            self.training_graph.val_index = self.best_graph.val_index.clone()
            self.training_graph.training_labels = self.best_graph.training_labels.clone()
            self.training_graph.node_pseudolabel = self.best_graph.node_pseudolabel.clone()
            self.training_graph.edge_pseudolabel = self.best_graph.edge_pseudolabel.clone()
            self.training_graph.homo_edge_flags = self.best_graph.homo_edge_flags.clone()
            self.training_graph.detected_edge_flags = self.best_graph.detected_edge_flags.clone()
    
    def update_edge_flags(self):
        # True denotes homo edge (under groudtruth and pseudo labels)
        # False denotes unknow, they may be homo(undetected) or heterophily
        
        in_node_labels = self.training_graph.training_labels[[self.training_graph.edge_index[0,:]]]
        out_node_labels = self.training_graph.training_labels[[self.training_graph.edge_index[1,:]]]
        self.training_graph.homo_edge_flags = torch.logical_and(in_node_labels == out_node_labels, torch.logical_and((in_node_labels>=0), (out_node_labels>=0)))
        self.training_graph.detected_edge_flags = torch.logical_and((in_node_labels>=0), (out_node_labels>=0))

    def update_train_data(self, model, itr):
        prediction = torch.softmax(model(self.graph), dim=1)
        if self.method == 'flexmatch':
            # dynamic threshold
            # Flexmatch(self.graph, prediction, self.fixed_threshold if self.update_pipeline_flag-1==0 else self.fixed_threshold*self.fixed_threshold, self.flex_batch).select()
            
            # fixed threshold
            Flexmatch(self.graph, prediction, self.node_threshold, self.edge_threshold, self.flex_batch).select()
            # UPS().select(self.args, self.graph, model, itr)
        
        self.update_training_labels()
        self.update_training_graph()
        
        self.global_iteration = self.global_iteration+1    
            
        self.graph.edge_pseudolabel[self.graph.train_index] = self.graph.y[self.graph.train_index].clone()
        
    
    def initialize_training_labels(self):
        self.graph.training_labels = self.graph.node_pseudolabel.clone()
        self.graph.training_labels[self.graph['train_index']] = self.graph.y[self.graph['train_index']]
        self.graph.label_confidence = torch.zeros_like(self.graph.training_labels)
        self.graph.label_confidence[self.graph['train_index']] = 1.
        

        # # add noise
        # self.graph.edge_pseudolabel[self.graph.train_index] = self.graph.y[self.graph.train_index].clone()
        # tmp = self.graph.edge_pseudolabel[torch.logical_not(self.graph.train_index)]
        # indices = torch.randperm(len(tmp))
        # zero_indices = (self.graph.y[torch.logical_not(self.graph.train_index)] == 0).nonzero(as_tuple=True)[0]
        # one_indices = (self.graph.y[torch.logical_not(self.graph.train_index)] == 1).nonzero(as_tuple=True)[0]

        # # 随机选择1101个0的索引和2110个1的索引
        # add_one_num = 1000
        # # add
        # selected_zero_indices = zero_indices[torch.randperm(len(zero_indices))]
        # selected_one_indices = one_indices[torch.randperm(len(one_indices))]
        # selected_zero_indices = zero_indices[torch.randperm(len(zero_indices))][:1101]
        # selected_one_indices = one_indices[torch.randperm(len(one_indices))][:2110]
        # tmp[selected_zero_indices] = 0
        # tmp[selected_one_indices] = 1
        
        # # 0 -> 1 
        # zero_indices = (tmp == 0).nonzero(as_tuple=True)[0]
        # rand_zero_indices = zero_indices[torch.randperm(len(zero_indices))][:685]
        # tmp[rand_zero_indices] = 1

        # # 1 -> 0
        # one_indices = (tmp == 1).nonzero(as_tuple=True)[0]
        # rand_one_indices = one_indices[torch.randperm(len(one_indices))][:285]
        # tmp[rand_one_indices] = 0
        
        # self.graph.edge_pseudolabel[torch.logical_not(self.graph.train_index)] = tmp
        
        # noise_indices = self.graph.val_index+self.graph.test_index+self.graph.unlabeled_index
        # noise_indices = torch.nonzero(noise_indices).squeeze()
        # noise_indices = torch.tensor(random.sample(noise_indices.tolist(), round(self.args.noise_rate*len(noise_indices))))
        # self.graph.pseudolabels_for_message_passing[noise_indices] = 1 - self.graph.pseudolabels_for_message_passing[noise_indices]
        # print(self.cal_edge_node_accuracy())
        
    def update_training_labels(self):
        # prediction is probability in [0,1]
        
        self.graph.training_labels = self.graph.node_pseudolabel.clone()
        self.graph.training_labels[self.graph['train_index']] = self.graph.y[self.graph['train_index']]
        
        # label_confidence, pseudolabels_for_message_passing = torch.max(prediction, dim=1)
        # label_confidence[self.graph['train_index']] = 1.
        # self.graph.label_confidence = label_confidence.clone().detach()
        
        # self.graph.pseudolabels_for_message_passing = pseudolabels_for_message_passing.clone().detach()
        # self.graph.pseudolabels_for_message_passing[self.graph['train_index']] = self.graph.y[self.graph['train_index']]

        
    def update_training_graph(self):
        """
            produce training graph according to self.graph 
            training graph only contains nodes with labels or pseudolabels and edges with two labeled ends 
            
            Attributes of the training graph:
                - y: groud truth for all nodes
                - x: node features
                - edge_index: edge
                - train_index: train indices
                - val_index: validation data indices
                - test_index: test data indices
                - num_nodes: number of nodes 
                - num_class: number of class
                
                - node_pseudolabel(size=num_nodes): '-2' means unavailable data, '-1' means unlabeled data
                - edge_pseudolabel(size=num_nodes):  '-1' means unlabeled data
                - training_labels(size=num_nodes): -1 denotes unlabeled, x(x>=0) denotes pseudolabels (training_labels=pseudolabel+train_index)
                - label_confidence(size=num_nodes): confidence of model 
                - pseudolabels_for_message_passing(size=num_nodes): prediction of all nodes(just for message passing but not for pseudolabel loss)（abandoned）
                - ground_truth_homo_edge_flags(size=edge_index): True denotes homo edge, False denotes hetero edge
                - homo_edge_flags(size=edge_index): 检测出来的homo edge（TODO:待细分，homo_edge有1->1也有0->0）
                - detected_edge_flags(size=edge_index): 检测出来的edge, True denotes two ends have been pseudolabels or training labels
            
        """
        self.training_graph = deepcopy(self.graph)
        self.update_edge_flags()
        
        # if self.global_iteration >= 3:
        #     self.args.mask_edge_flag = False 
        
        # print(f'pipeline_flag:{self.update_pipeline_flag}')
        
        # if self.update_pipeline_flag==0:
        #     # edge mask
        #     self.training_graph.edge_index = torch.zeros((2,0)).to(torch.cuda.current_device(), dtype=graph.edge_index.dtype)
            
        # elif self.update_pipeline_flag==1:
        #     # save homo edge
        #     change_flag = self.if_edge_change()
        #     self.training_graph.edge_index = self.graph.edge_index[:, self.training_graph.homo_edge_flags]
        
        # elif self.update_pipeline_flag==2:
        #     # save all edge
        #     self.if_edge_change()
        #     self.training_graph.edge_index = self.graph.edge_index[:, self.training_graph.detected_edge_flags]

        # else:
        #     assert self.update_pipeline_flag <= 2
        
        
        
        # delete_node_indices = torch.where(self.graph.training_labels==-1)[0]
        # delete_edge_indices = find_edges(self.training_graph,delete_node_indices)
        # self.training_graph = remove_edges(self.training_graph,delete_edge_indices)

        # self.args.mask_edge_flag = self.training_graph.edge_index.size()[1] <= 0 # calculate autoencoder loss after adding edges
        
        # if self.warm_up:
        #     self.training_graph.edge_index = torch.zeros((2,0)).to(torch.cuda.current_device(), dtype=graph.edge_index.dtype)
        
    def cal_loss(self, model_name, model, graph, criterion, record):
        loss = None 
        out = None 
        autoencoder_loss = None
        if model_name == 'ourModel':
            if self.warm_up:
                out = model(graph)
            else:
                out, autoencoder_loss = model(graph, auto_encoder_loss_flag=True)    
                
            groundtruth_loss = criterion(out[graph.train_index], graph.y[graph.train_index])
            loss = (groundtruth_loss + self.autoencoder_weight * autoencoder_loss) if autoencoder_loss is not None else groundtruth_loss
            
            if self.method == 'flexmatch' and torch.sum(graph.node_pseudolabel>=0) > 0: 
                pseudolabel_index = graph.node_pseudolabel >= 0
                pseudolabel_loss = criterion(out[pseudolabel_index], graph.node_pseudolabel[pseudolabel_index])
                record[1].append(pseudolabel_loss.item())
                loss = loss + self.flexmatch_weight*pseudolabel_loss       
            
            # if 'pseudo_label_dict' in graph:
            #     pass 
            # else:

            
            record[0].append([groundtruth_loss.item(), autoencoder_loss.item() if not self.warm_up else 0]) 

        elif model_name == 'mlp':
            out = model(graph)
            loss = criterion(out[graph.train_index], graph.y[graph.train_index])
            
            if self.method == 'flexmatch' and torch.sum(graph.node_pseudolabel>=0) > 0: 
                pseudolabel_index = graph.node_pseudolabel >= 0
                pseudolabel_loss = criterion(out[pseudolabel_index], graph.node_pseudolabel[pseudolabel_index])
                record[1].append(pseudolabel_loss.item())
                loss = loss + self.flexmatch_weight*pseudolabel_loss     
        
        elif model_name == 'GCN':
            out = model(graph)
            loss = criterion(out[graph.train_index], graph.y[graph.train_index])
        
        else:
            assert False
        
        return out, loss

    def cal_edge_node_accuracy(self):
        node_pseudolabel_indices = self.graph.node_pseudolabel >= 0
        node_labels_pseudo_acc = torch.sum(self.graph.node_pseudolabel[node_pseudolabel_indices]==self.graph.y[node_pseudolabel_indices]) / torch.sum(node_pseudolabel_indices)
        
        edge_pseudolabel_indices = self.graph.propogated_confidence_from>self.graph.edge_threshold
        edge_pseudolabel = self.graph.edge_pseudolabel.clone()
        edge_pseudolabel_indices[edge_pseudolabel_indices] = -1
        n_edge_pseudolabel = torch.sum(edge_pseudolabel_indices)
        # edge_pseudolabel_acc = torch.sum(self.graph.edge_pseudolabel==self.graph.y) / torch.sum(self.graph.edge_pseudolabel>=0)
        
        edge_label_pred = torch.cat((edge_pseudolabel[self.graph.edge_index[0,:]].unsqueeze(0), edge_pseudolabel[self.graph.edge_index[1,:]].unsqueeze(0)), dim=0)
        mask = (edge_label_pred != -1).all(dim=0)
        edge_label_pred = edge_label_pred[:, mask]
        edge_label_y = torch.cat((self.graph.y[self.graph.edge_index[0,:]].unsqueeze(0), self.graph.y[self.graph.edge_index[1,:]].unsqueeze(0)), dim=0)
        edge_label_y = edge_label_y[:, mask]
        edge_label_flag = edge_label_pred==edge_label_y
        edge_acc = torch.mean(torch.logical_and(edge_label_flag[0,:], edge_label_flag[1,:])*1.)
        
        return node_labels_pseudo_acc, n_edge_pseudolabel, edge_acc
        
    def train(self):
        model_iter = self.model 
        graph_iter = self.training_graph 
        optimizer = SGD(model_iter.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        # scheduler = 
        criterion = nn.CrossEntropyLoss()
        
        epoch = 0
        progress_bar = tqdm(total=torch.sum(self.graph['unlabeled_index']).item())
        
        counter = 0 # if we cann't include new unlabeled nodes for some iterations, break
        iteration_count = 0
        N = None
        
        loss_list = []
        metric_list = []
        val_metric_list = []
        test_metric_list = []
        pseudo_loss_list = []
        threshold_accuracy_list = []
        add_num_list = []
        pseudolabel_num = []
        best_val_metric_list = []
        homo_edge_acc = []
        node_labels_pseudo_acc_list = []
        n_edge_pseudolabel_list = []
        edge_acc_list = []
        edge_label_list = []
        
        # record the best model between two labels additions
        best_model_iter = None 
        best_val_metric_iter = -torch.inf
        best_test_metric_iter = -torch.inf # corresponding test_metric to best_val_metric
        
        # while torch.sum(self.graph.pseudolabel==-1) > torch.sum(self.graph['val_index']):
        while torch.sum(self.graph.node_pseudolabel>=0) <= torch.sum(self.graph['unlabeled_index']) and iteration_count<=5:
            if epoch==1:
                print(type(model_iter))
            
            if iteration_count>0 and epoch==1:
                
                node_labels_pseudo_acc, n_edge_pseudolabel, edge_acc = self.cal_edge_node_accuracy()
                
                node_labels_pseudo_acc_list.append(node_labels_pseudo_acc.item())
                n_edge_pseudolabel_list.append(n_edge_pseudolabel.item())
                edge_acc_list.append(edge_acc.item())
                
                print('================================================================================')
                print(f'Number of used edge: [{n_edge_pseudolabel}/{self.graph.edge_index.size()[1]}]')
                print(f'Accuracy of used edge: {edge_acc}')
                print(f'node_labels_pseudo_acc: {node_labels_pseudo_acc}')
                print('================================================================================')
            
            model_iter.train()
            
            # iteration untill all nodes included 
            graph_iter=self.training_graph
            
            # calculate the loss
            optimizer.zero_grad()
            out, loss = self.cal_loss(model_name=self.model_name, model=model_iter, graph=graph_iter, criterion=criterion, record=[loss_list, pseudo_loss_list])

            loss.backward()
            optimizer.step()
            # loss_list.append([groundtruth_loss.item(), 0])
            
            # calculate metric on validation data and training data
            metric = None 
            self.metric = 'accuracy' if (self.metric=='accuracy' or self.graph.num_class>2) else 'auc'
            if self.metric == 'accuracy':
                metric = cal_accuracy(graph_iter.y[graph_iter.train_index], out[graph_iter.train_index])
            elif self.metric == 'auc':
                pred, metric = cal_auc_score(graph_iter.y[graph_iter.train_index], out[graph_iter.train_index])
            
            metric_list.append(metric)
            pseudolabel_num.append(torch.sum(graph_iter.node_pseudolabel >= 0).item())

            val_pred_logits, val_metric = self.eval(model_iter, graph_iter, keyword='val', metric=self.metric)
            test_pred_logits, test_metric = self.eval(model_iter, graph_iter, keyword='test', metric=self.metric)
            val_metric_list.append(val_metric)
            test_metric_list.append(test_metric)

            
            # record model 
            if val_metric > best_val_metric_iter:
                best_val_metric_iter = val_metric
                best_test_metric_iter = test_metric
                best_model_iter = model_iter
                best_test_pred_logits = test_pred_logits
                
            
            
            if self.stopper.early_stop(epoch, val_metric):
                
                iteration_count = iteration_count + 1
                # self.edge_status_list.append(self.update_pipeline_flag)
                # if self.update_pipeline_flag>=1:
                #     in_node_labels = graph_iter.pseudolabel[graph_iter.edge_index[0,:]]
                #     out_node_labels = graph_iter.pseudolabel[graph_iter.edge_index[1,:]]
                #     detected_homo_flags = torch.logical_and(in_node_labels>=0, out_node_labels>=0)
                #     homo_edge_acc.append(torch.mean((graph_iter.homo_edge_flags[detected_homo_flags]==graph_iter.ground_truth_homo_edge_flags[detected_homo_flags])==1).item())
                #     print(f'Homo edge acc:{homo_edge_acc}')
                print(f'\nBest val metric:{best_val_metric_iter}')
                print(f'Best test metric:{best_test_metric_iter}')
                print(f'Is warm up? {self.warm_up}')
                
                # record the global sbest model
                if best_val_metric_iter > self.best_val_metric:
                    self.best_val_metric = best_val_metric_iter
                    self.best_test_metric = best_test_metric_iter
                    self.best_model = deepcopy(best_model_iter)
                    self.best_graph = deepcopy(graph_iter)
                    self.best_training_num_record = N 
                # else:
                #     # update stage
                #     if np.sum(np.array(self.edge_status_list)==self.update_pipeline_flag)>2 and best_val_metric_iter-best_val_metric_list[-1] < self.update_stage_threshold and best_val_metric_iter-best_val_metric_list[-2] < self.update_stage_threshold:
                #         self.update_pipeline_flag = self.update_pipeline_flag + 1
                        
                #     if self.update_pipeline_flag>2:
                #         break
                    
                best_val_metric_list.append(best_val_metric_iter)    
                best_val_metric_iter = -torch.inf
                    
               

                    
                # break 
                # out = best_model_iter(graph_iter)
                
                # just for observation
                threshold_accuracy, add_num_obs = accuracy_threshold(out, graph_iter, 0.9)
                # threshold_accuracy_list.append(threshold_accuracy.item())
                

                
                
                if self.warm_up:
                    # 如果涉及多轮重启则需要改动，此处默认使用最后一次的图
                    self.warm_up = False
                else:
                    # with open('performance_noise.txt', '+a') as f:
                    #     node_labels_pseudo_acc, node_labels_for_message_passing_acc, edge_acc = self.cal_edge_node_accuracy()
                    #     f.write(f'{self.args.noise_rate}\t{node_labels_pseudo_acc}\t{node_labels_for_message_passing_acc}\t{edge_acc}\t{best_test_metric_iter}\n')
                    
                    # with open(f'utils_data/noise_test/n_{self.args.noise_rate}_graph.pkl', 'wb') as f:
                    #     pickle.dump(self.best_graph, f)
                        
                    # with open(f'utils_data/noise_test/n_{self.args.noise_rate}_pred.pkl', 'wb') as f:
                    #     pickle.dump(best_test_pred_logits, f)
                    # break
                    pass
                
                if self.model_name == 'mlp' or self.model_name == 'GCN':
                    break    
                
                # pseudolabeling
                self.update_train_data(best_model_iter, iteration_count)
                edge_label_list.append(self.graph.edge_pseudolabel.detach().cpu().numpy())
                
                # restart
                epoch = 0
                model_iter = load_model(self.model_name, graph_iter, self.args)
                model_iter.to(torch.cuda.current_device())
                optimizer = SGD(model_iter.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.args.weight_decay)
                self.stopper.reset()
                N = torch.sum(self.training_graph.node_pseudolabel>=0).item()
                if N == progress_bar.n:
                    counter = counter + 1
                    if counter>5:
                        break
                else:
                    counter = 0
                # add_num_list.append(add_num_obs[0])
                
                 
                # node_labels_pseudo_acc, node_labels_for_message_passing_acc, edge_acc = self.cal_edge_node_accuracy()
                # print(f'node_labels_pseudo_acc: {node_labels_pseudo_acc}, node_labels_for_message_passing_acc: {node_labels_for_message_passing_acc}, edge_acc: {edge_acc}\n')
                

                    

            
            progress_bar.set_description(f'Train accuracy: {metric}, Loss:{loss.item()}, AUC:{metric}')
            progress_bar.n = torch.sum(graph_iter.node_pseudolabel>=0).item()
            progress_bar.refresh()
            epoch = epoch + 1
        
        
        
        if not os.path.exists(utils_data_pt):
            os.mkdir(utils_data_pt)
        
        np.save(os.path.join(utils_data_pt, 'y.npy'), graph.y.detach().cpu().numpy())
        np.save(os.path.join(utils_data_pt, 'loss.npy'), loss_list)
        np.save(os.path.join(utils_data_pt, 'metric.npy'), metric_list)
        np.save(os.path.join(utils_data_pt, 'val_metric.npy'), val_metric_list)
        np.save(os.path.join(utils_data_pt, 'test_metric.npy'), test_metric_list)
        np.save(os.path.join(utils_data_pt, 'threshold_accuracy.npy'), threshold_accuracy_list)
        np.save(os.path.join(utils_data_pt, 'add_num.npy'), add_num_list)
        np.save(os.path.join(utils_data_pt, 'pseudo_loss.npy'), pseudo_loss_list)
        np.save(os.path.join(utils_data_pt, 'edge_status.npy'), self.edge_status_list)
        np.save(os.path.join(utils_data_pt, 'best_val_metric_list.npy'), best_val_metric_list)
        np.save(os.path.join(utils_data_pt, 'final_accuracy.npy'), np.array([self.best_test_metric]))
        print(f'Final metric:{self.best_test_metric}')
        np.save(os.path.join(utils_data_pt, 'pseudolabel_num.npy'), pseudolabel_num)
        np.save(os.path.join(utils_data_pt, 'homo_edge_acc.npy'), homo_edge_acc)
        np.save(os.path.join(utils_data_pt, 'node_labels_pseudo_acc_list.npy'), node_labels_pseudo_acc_list)
        np.save(os.path.join(utils_data_pt, 'n_edge_pseudolabel_list.npy'), n_edge_pseudolabel_list)
        np.save(os.path.join(utils_data_pt, 'edge_acc_list.npy'), edge_acc_list)
        np.save(os.path.join(utils_data_pt, 'edge_label_list.npy'), edge_label_list)
        
        with open(os.path.join(utils_data_pt, 'setting.yaml'), 'w') as f:
            yaml.dump(vars(args), f)
    
        return self.best_test_metric
            
def save_res(acc, data_name, root='res/baselines.csv'):
    df = None
    if os.path.exists(root):
        df = pd.read_csv(root, index_col=0)
        
    else:
        df = pd.DataFrame()
    dic = df.to_dict()
    dic[data_name] = {}
    dic[data_name]['GCN'] = acc
    df = pd.DataFrame(dic)
    df.to_csv(root)
    df.to_excel('res/baselines.xlsx')

def load_model(model_name, graph, args):
    if model_name == 'mlp':
        return MLP(in_channels=graph.num_features,
                   out_channels=graph.num_class,
                   hidden_channels=args.hidden_dim,
                   num_layers=args.num_layers,
                   dropout=args.dropout
                   )
        
    elif model_name == 'ourModel':
        return ourModel(input_dim=graph.num_features, 
                    output_dim=graph.num_class, 
                    hidden_dim=args.hidden_dim, 
                    num_layers=args.num_layers,
                    dropout=args.dropout)
    
    elif model_name == 'GCN':
        return GCN(input_dim=graph.num_features,
                   output_dim=graph.num_class,
                   hidden_dim=args.hidden_dim,
                   num_layers=args.num_layers,
                   dropout=args.dropout)
    
    else:
        assert model_name in ['mlp', 'ourModel', 'GCN']

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='yelp-chi')
    parser.add_argument('--gpu', type=int, default=4)
    
    # hyper-parameter
    # for autoencoder
    parser.add_argument('--autoencoder_weight', type=float, default=0.01)
    # parser.add_argument('--embedding_dim', type=float, default=10)
    # for train
    parser.add_argument('--warm_up', type=bool, default=True)
    # parser.add_argument('--noise_rate', type=float, default=0.35)
    # for flexmatch
    parser.add_argument('--flex_batch', type=float, default=64)
    parser.add_argument('--flexmatch_weight', type=float, default=0.8)
    parser.add_argument('--node_threshold', type=float, default=0.95)
    parser.add_argument('--edge_threshold', type=float, default=0.8)
    # for ups
    parser.add_argument('--no_uncertainty', type=bool, default=False)
    parser.add_argument('--temp_nl', type=float, default=2)
    parser.add_argument('--kappa_p', type=float, default=0.05)
    parser.add_argument('--kappa_n', type=float, default=0.005)
    parser.add_argument('--tau_p', type=float, default=0.8)
    parser.add_argument('--tau_n', type=float, default=0.05)
    parser.add_argument('--class_blnc', type=int, default=3)
    # for dataset 
    parser.add_argument('--train_ratio', type=float, default=0.2)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--unlabel_ratio', type=float, default=0.4)
    parser.add_argument('--metric', type=str, default='auc')
    # for model
    parser.add_argument('--model_name', type=str, default='ourModel') # also the embedding dimension of encoder
    # parser.add_argument('--mask_edge_flag', action='store_true', default=True) # mask the edges         (deprecated)
    parser.add_argument('--hidden_dim', type=int, default=32) # also the embedding dimension of encoder
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.3)
    # for optimizer
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    # parser.add_argument('--weight_decay', type=float, default=0.0005)
    
    # utils_data_pt = './utils_data/new_model_DE'
    
    seed = 42
    random.seed(seed)  
    torch.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed) 
    
    args = parser.parse_args()
    dataset = load_dataset(args.dataset)
    utils_data_pt = f'./utils_data/{args.model_name}_{args.edge_threshold}_{args.dataset}'
    
    
    split_dataset_balanced(dataset, args.train_ratio, args.test_ratio, args.val_ratio, args.unlabel_ratio)
    graph = prepocessing(dataset)
    
    if directed_check(graph):
        print("Undirected!")
    else:
        print('Directed!')
    
    model = load_model(args.model_name, graph, args)
    
    if args.model_name == 'ourModel':
        model = load_model('mlp', graph, args)
    
    
    # model = GCN(input_dim=graph.num_features
    #             output_dim=graph.num_class,
    #             hidden_dim=args.hidden_dim, 
    #              num_layers=args.num_layers,
    #              dropout=args.dropout)
    
    trainer = Trainer(graph, model, device_num=args.gpu, model_name=args.model_name,
                      lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
                      metric=args.metric,
                      flex_batch=args.flex_batch,
                      flexmatch_weight=args.flexmatch_weight,
                      autoencoder_weight=args.autoencoder_weight,
                      args=args)
    
    # print(f'Noise rate: {args.noise_rate}')
    
    acc = trainer.train()
    
    