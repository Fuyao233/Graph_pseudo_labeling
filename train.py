import torch
from torch_geometric.data import Data
from utils import load_dataset, split_dataset, prepocessing, EarlyStopper, find_edges, remove_edges
from model import GCN, ourModel
from tqdm import tqdm
from torch.optim import Adam
import torch.nn.functional as F
from selecting_algorithm import Flexmatch
import os 
import pandas as pd 
import argparse
from copy import deepcopy
import numpy as np
import torch.nn as nn

class Trainer:
    def __init__(self, graph, model, device_num, lr, weight_decay, 
                 fixed_threshold,  
                 flexmatch_weight,
                 flex_batch,
                 autoencoder_weight,
                 select_method='flexmatch',) :
        self.graph = graph 
        self.model = model 
        self.lr = lr
        self.weight_decay = weight_decay
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
            self.graph.pseudolabel[graph['val_index']] = -1

        # self.update_autoencoder_prob()
        # print(self.graph.autoencoder_prob)
        self.update_training_labels()
        self.update_training_graph()
        
        self.stopper = EarlyStopper()
    
    # def get_batch_indices(self):
    #     true_indices = torch.where(self.graph.['train_index'])[0]
    
    # def update_autoencoder_prob(self):
    #     self.graph.autoencoder_prob = torch.zeros((self.graph.num_nodes,self.graph.num_class))
    #     # dynamically calculate the portion of each class (taking pseudo-labels into account) 
    #     # batch_indices = self.graph['train_index'] * 
    #     class_portion = torch.tensor([((torch.sum(self.graph.y[self.graph['train_index']]==c))+torch.sum(self.graph.pseudolabel==c))/((torch.sum(self.graph['train_index'])) + torch.sum(self.graph.pseudolabel>=0) ) for c in range(self.graph.num_class)])
    #     print(class_portion)
    #     for i,_ in enumerate(self.graph.autoencoder_prob):
    #         if self.graph.pseudolabel[i] == -2:
    #             self.graph.autoencoder_prob[i][self.graph.y[i]] = 1.
    #         elif  self.graph.pseudolabel[i] == -1:
    #             self.graph.autoencoder_prob[i] = class_portion
    #         elif self.graph.pseudolabel[i] >= 0:
    #             self.graph.autoencoder_prob[i][self.graph.pseudolabel[i]] = 1.
    #         else:
    #             raise ValueError("What's up??")
    
    def eval(self):
        with torch.no_grad():
            y_hat = torch.argmax(self.model(self.graph), dim=1)
            # print(y_hat.size())
            # print(y_hat['test_index'].size())
            # print(self.graph.y['test_index'].size())
            acc1 = torch.mean((y_hat[self.graph['test_index']]==self.graph.y[self.graph['test_index']]).float())
            # acc2 = torch.mean((self.graph.pseudolabel[self.graph['test_index']]==self.graph.y[self.graph['test_index']]).float())
            
            # print(f'Reforward accuracy is {acc1:.3f} \n Pseudolabel accuracy is {acc2:.3f}')
            print(f'Reforward accuracy is {acc1:.3f}')
        
        return acc1
        
    def update_training_graph(self):
        """
            produce training graph according to self.graph 
            training graph only contains nodes with labels or pseudolabels and edges with two labeled ends 
        """
        training_graph = deepcopy(self.graph)
        
        delete_node_indices = self.graph.training_labels==-1 
        delete_edge_indices = find_edges(training_graph,delete_node_indices)
        training_graph = remove_edges(training_graph,delete_edge_indices)
        
        self.training_graph = training_graph
        
    
    def update_train_data(self, prediction):
        if self.method == 'flexmatch':
            Flexmatch(self.graph, prediction, self.fixed_threshold, self.flex_batch).select()
        
        self.update_training_graph()
        
    def update_training_labels(self):
        self.graph.training_labels = self.graph.pseudolabel.clone()
        self.graph.training_labels[self.graph['train_index']] = self.graph.y[self.graph['train_index']]
        
    def train(self):
        model = self.model 
        graph = self.training_graph 
        optimizer = Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        epoch = 0
        progress_bar = tqdm(total=torch.sum(self.graph['test_index']).item())
        
        counter = 0 # if we cann't include new unlabeled nodes, break
        N = None
        
        loss_list = []
        acc_list = []
        pseudo_loss_list = []
        
        while torch.sum(self.graph.pseudolabel==-1) > 0:
            # iteration untill all nodes are 
            graph=self.training_graph
            
            optimizer.zero_grad()
            
            out, autoencoder_loss = model(graph, auto_encoder_loss_flag=True)    
            groundtruth_loss = criterion(out[graph.train_index], graph.y[graph.train_index].to(torch.int64))
            loss = groundtruth_loss + self.autoencoder_weight * autoencoder_loss
            acc = torch.argmax(out[graph.training_labels>=0],dim=1)==graph.y[graph.train_index]
            acc = torch.sum(acc) / torch.sum(graph.train_index)
            
            loss_list.append([groundtruth_loss.item(), autoencoder_loss.item()])
            acc_list.append(acc.item())
            
            if self.method == 'flexmatch' and torch.sum(graph.pseudolabel>=0) > 0: 
                pseudolabel_index = graph.pseudolabel >= 0
                pseudolabel_loss = criterion(out[pseudolabel_index], graph.pseudolabel[pseudolabel_index].to(torch.int64))
                pseudo_loss_list.append(pseudolabel_loss.item())
                loss = loss + self.flexmatch_weight*pseudolabel_loss
                
            loss.backward()
            optimizer.step()
            
            
            if self.stopper.early_stop(epoch, acc):
                self.update_train_data(out)
                epoch = 0
                self.stopper.reset()
                N = torch.sum(self.graph.pseudolabel>=0).item()
                if N == progress_bar.n:
                    counter = counter + 1
                    if counter>5:
                        break
                else:
                    counter = 0
                
            progress_bar.set_description(f'Train accuracy: {acc.item()}, Loss:{loss.item()}')
            progress_bar.n = torch.sum(self.graph.pseudolabel>=0).item()
            progress_bar.refresh()
            epoch = epoch + 1
            
        acc = self.eval()
        
        np.save('loss.npy', loss_list)
        np.save('acc.npy', acc_list)
        np.save('pseudo_loss.npy', pseudo_loss_list)
        
        return acc
            
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



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='twitch-e')
    parser.add_argument('--gpu', type=int, default=4)
    
    # hyper-parameter
    # for autoencoder
    parser.add_argument('--autoencoder_weight', type=float, default=0.6)
    # for flexmatch
    parser.add_argument('--flex_batch', type=float, default=256)
    parser.add_argument('--flexmatch_weight', type=float, default=0.8)
    parser.add_argument('--fixed_threshold', type=float, default=0.9)
    # for dataset 
    parser.add_argument('--train_ratio', type=float, default=0.2)
    parser.add_argument('--test_ratio', type=float, default=None)
    parser.add_argument('--val_ratio', type=float, default=None)
    # for model
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.3)
    # for optimizer
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    
    
    args = parser.parse_args()
    dataset = load_dataset(args.dataset)
    
    split_dataset(dataset, args.train_ratio)
    graph = prepocessing(dataset)
    
    model = ourModel(input_dim=graph.num_features, output_dim=graph.num_class, hidden_dim=args.hidden_dim, num_layers=args.num_layers,dropout=args.dropout)

    trainer = Trainer(graph, model, device_num=0, lr=args.lr,
                      weight_decay=args.weight_decay,
                      fixed_threshold=args.fixed_threshold,
                      flex_batch=args.flex_batch,
                      flexmatch_weight=args.flexmatch_weight,
                      autoencoder_weight=args.autoencoder_weight)
    acc = trainer.train()
    # trainer = Trainer(graph, model, device_num=args.gpu, lr=args.lr,
    #                   weight_decay=args.weight_decay,
    #                   fixed_threshold=args.fixed_threshold,
    #                   flex_batch=args.flex_batch,
    #                   flexmatch_weight=args.flexmatch_weight,
    #                   autoencoder_weight=args.autoencoder_weight)
    # acc = trainer.train()

    