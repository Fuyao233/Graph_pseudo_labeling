import torch
from torch_geometric.data import Data
from utils import *
from model import GCN, ourModel, MLP
from tqdm import tqdm
from torch.optim import Adam, SGD, AdamW
import torch.nn.functional as F
from selecting_algorithm import Flexmatch
import os 
import pandas as pd 
import argparse
from copy import deepcopy
import numpy as np
import torch.nn as nn
import yaml
from copy import deepcopy

import random

class Trainer:
    def __init__(self, graph, model, device_num, model_name,
                 lr, momentum, weight_decay,
                 fixed_threshold,  
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
        self.fixed_threshold = fixed_threshold
        self.method = 'flexmatch'
        self.flexmatch_weight = flexmatch_weight # weight for flexmatch loss
        # self.flex_batch = int(self.graph.num_nodes*flex_batch_ratio) # select from a random batch of unlabeled samples each time updating training samples 
        self.flex_batch = flex_batch # select from a random batch of unlabeled samples each time updating training samples 
        
        self.autoencoder_weight = autoencoder_weight
        
        torch.cuda.set_device(device_num)
        device = torch.cuda.current_device()
        self.graph.to(device)
        self.model.to(device)
            
        self.graph.pseudolabel = torch.zeros_like(graph.y)-2 # '-2' means labeled data
        self.graph.pseudolabel[graph['test_index']] = -1 # '-1' means unlabeled data
        if 'val_index' in graph:
            self.graph.pseudolabel[graph['val_index']] = -1 # validation data is treated equally as unlabeled data

        # record the best model during the whole training process 
        self.best_model = None # best model among all best_model_iter
        self.best_graph = None # graph according to the best model(some edges may be omitted)
        self.best_val_metric = -torch.inf
        self.best_training_num_record = 0 # record the num of training samples of the best model
        self.global_iteration = 0 # number of adding samples 
        
        # self.update_autoencoder_prob()
        # print(self.graph.autoencoder_prob)
        self.update_pipeline_flag = 0
        self.update_training_labels()
        self.update_training_graph()
        
        self.stopper = EarlyStopper(patience=100, max_iter=100)
    
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
                metric = torch.mean((y_hat[self.graph[key]]==self.graph.y[self.graph[key]]).float())
            elif metric == 'auc':
                # metric = roc_auc_score(self.graph.y[self.graph[key]].cpu().numpy(),torch.softmax(logits[self.graph[key]], dim=1)[:,1].cpu().detach().numpy()) 
                metric = eval_rocauc(graph.y[graph[key]].reshape((-1,1)), logits[graph[key]])
            else:
                assert metric in ['accuracy', 'auc']
            
            # if keyword == 'test':
            #     print(f'{key} {metric} is {metric:.3f}')

        # auc_score = roc_auc_score(graph.y[graph[key]].cpu().numpy(),torch.softmax(logits[graph[key]],dim=1)[:,1].cpu().detach().numpy()) 
        return metric
        
    def update_training_graph(self):
        """
            produce training graph according to self.graph 
            training graph only contains nodes with labels or pseudolabels and edges with two labeled ends 
        """
        training_graph = deepcopy(self.graph)

        # if self.global_iteration >= 3:
        #     self.args.mask_edge_flag = False 
        
        print(f'pipeline_flag:{self.update_pipeline_flag}')
        if self.update_pipeline_flag==0:
            # edge mask
            training_graph.edge_index = torch.zeros((2,0)).to(torch.cuda.current_device(), dtype=graph.edge_index.dtype)
            
        elif self.update_pipeline_flag==1:
            # save homo edge
            training_graph.edge_index = self.graph.edge_index[:,training_graph.homo_edge_flags]
            # self.update_pipeline_flag = self.update_pipeline_flag + 1
            # self.update_pipeline_flag = 0
        
        elif self.update_pipeline_flag==2:
            # save all edge
            training_graph.edge_index = self.graph.edge_index
            # self.update_pipeline_flag = 0

        else:
            assert self.update_pipeline_flag <= 2
        
        delete_node_indices = torch.where(self.graph.training_labels==-1)[0]
        delete_edge_indices = find_edges(training_graph,delete_node_indices)
        training_graph = remove_edges(training_graph,delete_edge_indices)
        
        self.training_graph = training_graph
        self.args.mask_edge_flag = self.training_graph.edge_index.size()[1] <= 0 # calculate autoencoder loss after adding edges
        
    
    def update_train_data(self, prediction):
        prediction = torch.softmax(prediction, dim=1)
        if self.method == 'flexmatch':
            # dynamic threshold
            # Flexmatch(self.graph, prediction, self.fixed_threshold if self.update_pipeline_flag-1==0 else self.fixed_threshold*self.fixed_threshold, self.flex_batch).select()
            
            # fixed threshold
            Flexmatch(self.graph, prediction, self.fixed_threshold, self.flex_batch).select()
        
        self.update_training_labels()
        self.update_training_graph()
        
        self.global_iteration = self.global_iteration+1
        
    def update_training_labels(self):
        self.graph.training_labels = self.graph.pseudolabel.clone()
        self.graph.training_labels[self.graph['train_index']] = self.graph.y[self.graph['train_index']]
    
    def cal_loss(self, model_name, model, graph, criterion, record):
        loss = None 
        out = None 
        if model_name == 'ourModel':
            if self.args.mask_edge_flag:
                out = model(graph)
            else:
                out, autoencoder_loss = model(graph, auto_encoder_loss_flag=True)    
                
            groundtruth_loss = criterion(out[graph.train_index], graph.y[graph.train_index])
            loss = groundtruth_loss + self.autoencoder_weight * autoencoder_loss if not self.args.mask_edge_flag else groundtruth_loss
            
            if self.method == 'flexmatch' and torch.sum(graph.pseudolabel>=0) > 0: 
                pseudolabel_index = graph.pseudolabel >= 0
                pseudolabel_loss = criterion(out[pseudolabel_index], graph.pseudolabel[pseudolabel_index])
                record[1].append(pseudolabel_loss.item())
                loss = loss + self.flexmatch_weight*pseudolabel_loss            
            
            record[0].append([groundtruth_loss.item(), autoencoder_loss.item() if not self.args.mask_edge_flag else 0]) 

        elif model_name == 'mlp':
            out = model(graph)
            loss = criterion(out[graph.train_index], graph.y[graph.train_index])
            
            if self.method == 'flexmatch' and torch.sum(graph.pseudolabel>=0) > 0: 
                pseudolabel_index = graph.pseudolabel >= 0
                pseudolabel_loss = criterion(out[pseudolabel_index], graph.pseudolabel[pseudolabel_index])
                record[1].append(pseudolabel_loss.item())
                loss = loss + self.flexmatch_weight*pseudolabel_loss     
        
        else:
            assert False
        
        return out, loss
            
    def train(self):
        model_iter = self.model 
        graph_iter = self.training_graph 
        optimizer = SGD(model_iter.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        # scheduler = 
        criterion = nn.CrossEntropyLoss()
        
        epoch = 0
        progress_bar = tqdm(total=torch.sum(self.graph['test_index']).item())
        
        counter = 0 # if we cann't include new unlabeled nodes for some iterations, break
        N = None
        
        loss_list = []
        metric_list = []
        val_metric_list = []
        test_metric_list = []
        pseudo_loss_list = []
        threshold_accuracy_list = []
        add_num_list = []
        pseudolabel_num = []
        
        
        # record the best model between two labels additions
        best_model_iter = None 
        best_val_metric_iter = -torch.inf
        best_test_metric_iter = -torch.inf # corresponding test_metric to best_val_metric
        
        # while torch.sum(self.graph.pseudolabel==-1) > torch.sum(self.graph['val_index']):
        while torch.sum(self.graph.pseudolabel>=0) <= torch.sum(self.graph['val_index']+self.graph['test_index']):
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
            if self.metric == 'accuracy':
                metric = cal_accuracy(graph_iter.y[graph_iter.train_index], out[graph_iter.train_index])
            elif self.metric == 'auc':
                metric = cal_auc_score(graph_iter.y[graph_iter.train_index], out[graph_iter.train_index])
            
            metric_list.append(metric.item())
            pseudolabel_num.append(torch.sum(graph_iter.pseudolabel >= 0).item())

            val_metric = self.eval(model_iter, graph_iter, keyword='val', metric=self.metric)
            test_metric = self.eval(model_iter, graph_iter, keyword='test', metric=self.metric)
            val_metric_list.append(val_metric.item())
            test_metric_list.append(test_metric.item())
            
            # record model 
            if val_metric > best_val_metric_iter:
                best_val_metric_iter = val_metric
                best_test_metric_iter = test_metric
                best_model_iter = model_iter
            
            
            if self.stopper.early_stop(epoch, val_metric):
                
                # record the global best model
                if best_val_metric_iter > self.best_val_metric:
                    self.best_val_metric = best_val_metric_iter
                    self.best_test_metric = best_test_metric_iter
                    self.best_model = deepcopy(best_model_iter)
                    self.best_graph = deepcopy(graph_iter)
                    self.best_training_num_record = N 
               
                if self.model_name == 'mlp':
                    break    
                    
                # break 
                out = best_model_iter(graph_iter)
                
                # just for observation
                threshold_accuracy, add_num_obs = accuracy_threshold(out, graph_iter, self.fixed_threshold if self.update_pipeline_flag-1==0 else self.fixed_threshold*self.fixed_threshold)
                threshold_accuracy_list.append(threshold_accuracy.item())
                
                epoch = 0

                # reinitialize model
                # if N<torch.sum(graph_iter['test_index']+graph_iter['val_index']).item():
                    
                # model.restart()
                model_iter = load_model(self.model_name, graph_iter, self.args)
                model_iter.to(torch.cuda.current_device())
                
                # if self.update_pipeline_flag == 2:
                # # inherit encoder&decoder after training on graph with only homophily edges 
                #     for src_layer, tgt_layer in zip(best_model_iter.convs, model_iter.convs):
                #         for src_encoder, tgt_encoder in zip(src_layer.encoder_group, tgt_layer.encoder_group):
                #             tgt_encoder.load_state_dict(src_encoder.state_dict())

                #         for src_decoder, tgt_decoder in zip(src_layer.decoder_group, tgt_layer.decoder_group):
                #             tgt_decoder.load_state_dict(src_decoder.state_dict())
                
                optimizer = SGD(model_iter.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.args.weight_decay)
                
                self.stopper.reset()
                self.update_train_data(out)
                N = torch.sum(self.training_graph.pseudolabel>=0).item()
                if N == progress_bar.n:
                    counter = counter + 1
                    if counter>5:
                        break
                else:
                    counter = 0
                add_num_list.append(add_num_obs[0])
                

            
            progress_bar.set_description(f'Train accuracy: {metric.item()}, Loss:{loss.item()}, AUC:{metric}')
            progress_bar.n = torch.sum(graph_iter.pseudolabel>=0).item()
            progress_bar.refresh()
            epoch = epoch + 1
        
        
        
        if not os.path.exists(utils_data_pt):
            os.mkdir(utils_data_pt)
            
        np.save(os.path.join(utils_data_pt, 'loss.npy'), loss_list)
        np.save(os.path.join(utils_data_pt, 'metric.npy'), metric_list)
        np.save(os.path.join(utils_data_pt, 'val_metric.npy'), val_metric_list)
        np.save(os.path.join(utils_data_pt, 'test_metric.npy'), test_metric_list)
        np.save(os.path.join(utils_data_pt, 'threshold_accuracy.npy'), threshold_accuracy_list)
        np.save(os.path.join(utils_data_pt, 'add_num.npy'), add_num_list)
        np.save(os.path.join(utils_data_pt, 'pseudo_loss.npy'), pseudo_loss_list)
        np.save(os.path.join(utils_data_pt, 'final_accuracy.npy'), np.array([self.best_test_metric.item()]))
        print(f'Final metric:{self.best_test_metric.item()}')
        np.save(os.path.join(utils_data_pt, 'pseudolabel_num.npy'), pseudolabel_num)
        
        
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
    
    else:
        assert model_name in ['mlp', 'ourModel']

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='twitch-e')
    parser.add_argument('--gpu', type=int, default=4)
    
    # hyper-parameter
    # for autoencoder
    parser.add_argument('--autoencoder_weight', type=float, default=0.1)
    # parser.add_argument('--embedding_dim', type=float, default=10)
    # for flexmatch
    parser.add_argument('--flex_batch', type=float, default=64)
    parser.add_argument('--flexmatch_weight', type=float, default=0.8)
    parser.add_argument('--fixed_threshold', type=float, default=0.9)
    # for dataset 
    parser.add_argument('--train_ratio', type=float, default=0.5)
    parser.add_argument('--test_ratio', type=float, default=0.25)
    parser.add_argument('--val_ratio', type=float, default=0.25)
    parser.add_argument('--metric', type=str, default='auc')
    # for model
    parser.add_argument('--model_name', type=str, default='ourModel') # also the embedding dimension of encoder
    parser.add_argument('--mask_edge_flag', action='store_true', default=True) # mask the edges
    parser.add_argument('--hidden_dim', type=int, default=32) # also the embedding dimension of encoder
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.3)
    # for optimizer
    parser.add_argument('--lr', type=float, default=0.01)
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
    utils_data_pt = f'./utils_data/1_stage_{args.model_name}_{args.dataset}_{args.mask_edge_flag if args.model_name == "ourModel" else ""}' 
    
    
    split_dataset(dataset, args.train_ratio, args.test_ratio, args.val_ratio)
    graph = prepocessing(dataset)
    
    model = load_model(args.model_name, graph, args)
    
    # model = GCN(input_dim=graph.num_features
    #             output_dim=graph.num_class,
    #             hidden_dim=args.hidden_dim, 
    #              num_layers=args.num_layers,
    #              dropout=args.dropout)
    
    trainer = Trainer(graph, model, device_num=args.gpu, model_name=args.model_name,
                      lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
                      metric=args.metric,
                      fixed_threshold=args.fixed_threshold,
                      flex_batch=args.flex_batch,
                      flexmatch_weight=args.flexmatch_weight,
                      autoencoder_weight=args.autoencoder_weight,
                      args=args)
    acc = trainer.train()
    
    