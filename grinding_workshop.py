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
from sklearn.metrics import roc_auc_score 
import random 
import copy 

utils_data_pt = './utils_data/MLP'

class MLP(nn.Module):
    def __init__(self, in_channel, out_channel, hidden_channel) -> None:
        super(MLP, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(in_channel, hidden_channel),
            nn.ReLU(),
            nn.Linear(hidden_channel, hidden_channel),
            nn.ReLU(),
            nn.Linear(hidden_channel, out_channel),
        )
    
    def forward(self, x):
        return self.model(x)


def test(model, criterion, data, labels):
    model.eval()
    loss = None
    acc = None
    auc = None 
    with torch.no_grad():
        
        logits = model(data)
        loss = criterion(logits, labels)
        
        predict_y = torch.argmax(logits, dim=1)
        acc = torch.mean((predict_y==labels)*1.)
        auc = roc_auc_score(labels.cpu().numpy(),torch.softmax(logits, dim=1)[:,1].cpu().detach().numpy()) 
        
    return loss, acc, auc
        
        
def train(model, graph, device):
    
    torch.cuda.set_device(device)
    device = torch.cuda.current_device()
    
    model.to(device)
    graph.to(device)
    
    train_index = graph['train_index']
    val_index = graph['val_index']
    test_index = graph['test_index']
    epoch = 200
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9)
    # scheduler = torch.lr_scheduler.StepLR(optimizer, step_size = 100, gamma = 0.1)
    
    best_model = None 
    best_val_auc = -torch.inf
    
    earlystopper = EarlyStopper(patience=30)
    for e in range(epoch):
        optimizer.zero_grad()
        logits = model(graph.x[train_index])
        loss = criterion(logits, graph.y[train_index])
        loss.backward()
        optimizer.step()
        
        predict_y = torch.argmax(logits, dim=1)
        acc = torch.mean((predict_y==graph.y[train_index])*1.)
        
        print(f'train accuracy:{acc}')
        val_loss, val_acc , val_auc= test(model, criterion, graph.x[val_index], graph.y[val_index])
        if val_auc>best_val_auc:
            best_model = copy.deepcopy(model)
            best_val_auc = val_auc
            
        print(f'val accuracy: {val_acc}; val auc score: {val_auc}')
        
        # if earlystopper.early_stop(e, val_loss):
        #     break
        
        


    test_loss, test_acc, auc_score = test(best_model, criterion, graph.x[test_index], graph.y[test_index])
    print(f'test accurracy: {test_acc}; test auc score: {auc_score}')
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='twitch-e')
    parser.add_argument('--gpu', type=int, default=4)
    
    # hyper-parameter
    # for autoencoder
    parser.add_argument('--autoencoder_weight', type=float, default=0.01)
    # for flexmatch
    parser.add_argument('--flex_batch', type=float, default=64)
    parser.add_argument('--flexmatch_weight', type=float, default=0.8)
    parser.add_argument('--fixed_threshold', type=float, default=0.99)
    # for dataset 
    parser.add_argument('--train_ratio', type=float, default=0.5)
    parser.add_argument('--test_ratio', type=float, default=0.25)
    parser.add_argument('--val_ratio', type=float, default=0.25)
    # for model
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.3)
    # for optimizer
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    

    seed = 42
    random.seed(seed)  
    torch.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed) 
    
    args = parser.parse_args()
    dataset = load_dataset(args.dataset)
    
    split_dataset(dataset, args.train_ratio, args.test_ratio, args.val_ratio)
    graph = prepocessing(dataset)
    model = MLP(graph.x.size()[1], 2, 32)
    train(model, graph,1)
    
    