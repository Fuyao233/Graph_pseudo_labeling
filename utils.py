from Non_Homophily_Large_Scale.dataset import *
from torch_geometric.data import Data
import random 
from sklearn.metrics import roc_auc_score 

# datanames = ['twitch-e', 'fb100', 'ogbn-proteins', 'deezer-europe', 'arxiv-year', 'pokec', 'snap-patents',
#              'yelp-chi', 'ogbn-arxiv', 'ogbn-products', 'Cora', 'CiteSeer', 'PubMed', 'chameleon', 'cornell',
#              'film', 'squirrel', 'texas', 'wisconsin', 'genius', 'twitch-gamer', 'wiki']
datanames = ['fb100']

def load_dataset(name):
    return load_nc_dataset(name)


def directed_check(graph):
    for i in range(10):
        if not torch.sum(graph.edge_index[0]==i) == torch.sum(graph.edge_index[0]==i):
            return False
    return True

def split_dataset(dataset, train_ratio, test_ratio=None, val_ratio=None, unlabel_ratio=None):
    """
        split the graph dataset into three parts for node classification
    Args:
        dataset
        train_ratio
        test_ratio 
        val_ratio 
        unlabel_ratio
    """
    if test_ratio is None:
        test_ratio = 1 - train_ratio

    if (train_ratio+test_ratio+(0 if val_ratio is None else val_ratio))+((0 if unlabel_ratio is None else unlabel_ratio)) != 1:
        raise ValueError("The sum of train_ratio, test_ratio, unlabel_ratio and val_ratio must be equal to 1.")
        
    for graph in dataset:
        N_nodes = graph[0]['num_nodes']
        train_index, test_index, val_index, unlabeled_index = None, None, None, None
        if val_ratio is None:
            num_train = int(train_ratio*N_nodes)
            index = torch.zeros(N_nodes)
            indices = random.sample(range(N_nodes), num_train)
            index.view(-1)[indices] = 1
            train_index = (index==1)
            test_index = (index!=1)
        else:
            num_train = int(train_ratio*N_nodes)
            num_val = int(val_ratio*N_nodes)
            num_unlabeled = int(unlabel_ratio*N_nodes)
            index = torch.zeros(N_nodes)
            train_indices = random.sample(range(N_nodes), num_train)
            val_indices = random.sample(set(range(N_nodes)) - set(train_indices), num_val)
            unlabeled_indices = random.sample(set(range(N_nodes)) - set(train_indices) - set(val_indices), num_unlabeled)
            index.view(-1)[train_indices] = 1
            index.view(-1)[val_indices] = 2
            index.view(-1)[unlabeled_indices] = 3
            test_index = (index==0)
            train_index = (index==1)
            val_index = (index==2)
            unlabeled_index = (index==3)
        graph[0]['train_index'] = train_index
        graph[0]['test_index'] = test_index
        graph[0]['val_index'] = val_index
        graph[0]['unlabeled_index'] = unlabeled_index

        if len(dataset) == 1:
            break
            
        # graph['train_index']

import torch
import random
from collections import defaultdict

def split_dataset_balanced(dataset, train_ratio, test_ratio=None, val_ratio=None, unlabel_ratio=None):
    """
    Split the graph dataset into parts for node classification in a balanced manner.
    Args:
        dataset: Graph dataset.
        train_ratio: Fraction of nodes to be used for training.
        test_ratio: Fraction of nodes to be used for testing.
        val_ratio: Fraction of nodes to be used for validation.
        unlabel_ratio: Fraction of nodes to be used as unlabeled.
    """
    if test_ratio is None:
        test_ratio = 1 - train_ratio - (0 if val_ratio is None else val_ratio) - (0 if unlabel_ratio is None else unlabel_ratio)
    if val_ratio is None:
        val_ratio = 0
    if unlabel_ratio is None:
        unlabel_ratio = 0

    if not (0 < train_ratio < 1) or not (0 <= test_ratio <= 1) or not (0 <= val_ratio <= 1) or not (0 <= unlabel_ratio <= 1):
        raise ValueError("All ratios must be between 0 and 1.")

    if abs(train_ratio + test_ratio + val_ratio + unlabel_ratio - 1) > 1e-6:
        raise ValueError("The sum of train_ratio, test_ratio, val_ratio, and unlabel_ratio must be equal to 1.")
    
    for graph in dataset:
        N_nodes = graph[0]['num_nodes']
        labels = dataset.label  
        
        # Create a dictionary to hold indices for each label
        label_indices = defaultdict(list)
        exclude_indices = []
        for i, label in enumerate(labels):
            if label.item() < 0:
                exclude_indices.append(i)
            label_indices[label.item()].append(i)
        
        train_index, test_index, val_index, unlabeled_index = [], [], [], []
        
        # Split indices for each label according to the specified ratios
        for label, indices in label_indices.items():
            random.shuffle(indices)
            n_train = int(train_ratio * len(indices))
            n_val = int(val_ratio * len(indices))
            n_unlabeled = int(unlabel_ratio * len(indices))

            train_index.extend(indices[:n_train])
            val_index.extend(indices[n_train:n_train + n_val])
            unlabeled_index.extend(indices[n_train + n_val:n_train + n_val + n_unlabeled])
            test_index.extend(indices[n_train + n_val + n_unlabeled:])
        
        # Convert indices lists to boolean masks
        total_index = torch.zeros(N_nodes, dtype=torch.bool)
        total_index[train_index] = True
        graph[0]['train_index'] = total_index.clone()
        total_index.fill_(False)
        total_index[test_index] = True
        graph[0]['test_index'] = total_index.clone()
        if val_ratio > 0:
            total_index.fill_(False)
            total_index[val_index] = True
            graph[0]['val_index'] = total_index.clone()
        if unlabel_ratio > 0:
            total_index.fill_(False)
            total_index[unlabeled_index] = True
            graph[0]['unlabeled_index'] = total_index.clone()

        graph[0]['train_index'][exclude_indices] = False
        graph[0]['test_index'][exclude_indices] = False
        graph[0]['val_index'][exclude_indices] = False
        graph[0]['unlabeled_index'][exclude_indices] = False
        
        if len(dataset) == 1:
            break



def prepocessing(dataset):
    data = Data()
    graph = dataset[0][0]
    label = dataset.label
    data.x = graph['node_feat']
    data.y = label
    
    data.num_class = len(torch.unique(label[label>=0]))
    
    # if 'pseudolabel' not in graph:

    for key in graph.keys():
        if key == 'node_feat':
            continue
        data[key] = graph[key]
    
    # identify homo and hetero edges
    # data.y
    in_node_labels = data.y[data.edge_index[0,:]]
    out_node_labels = data.y[data.edge_index[1,:]]
    data.ground_truth_homo_edge_flags = in_node_labels == out_node_labels
    
    return data

def find_edges(data, node_indices):
    # node_indices = torch.tensor(node_indices)
    mask = (data.edge_index[0].unsqueeze(1) == node_indices) | (data.edge_index[1].unsqueeze(1) == node_indices)
    mask = mask.any(dim=1)
    edge_indices = mask.nonzero(as_tuple=False).view(-1)
    
    return edge_indices

def remove_edges(data, edge_indices):
    mask = torch.ones(data.edge_index.size(1), dtype=torch.bool)
    mask[edge_indices] = False
    data.edge_index = data.edge_index[:, mask]
    if data.edge_attr is not None:
        data.edge_attr = data.edge_attr[mask]
    return data

# def identify_homo_indices(graph):
#     for i in range(graph.num_class):
#         homo_edge_indices = torch.logical_and(torch.graph.training_labels == i
#     pass 

# def identify_hetero_indices(graph):
#     pass 

def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        try:
            res.append(correct_k.mul_(100.0 / batch_size))
        except:
            res = (torch.tensor(0.0), torch.tensor(0.0))
    return res

def eval_rocauc(y_true, y_pred):
    """ adapted from ogb
    https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/evaluate.py"""
    rocauc_list = []
    y_true = y_true.detach().cpu().numpy()
    if y_true.shape[1] == 1:
        # use the predicted class for single-class classification
        y_pred = F.softmax(y_pred, dim=-1)[:,1].unsqueeze(1).detach().cpu().numpy()
    else:
        y_pred = y_pred.detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            is_labeled = y_true[:, i] == y_true[:, i]
            score = roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i])
                                
            rocauc_list.append(score)

    if len(rocauc_list) == 0:
        raise RuntimeError(
            'No positively labeled data available. Cannot compute ROC-AUC.')

    return sum(rocauc_list)/len(rocauc_list)



def cal_accuracy(labels, logits):
    y_hat = torch.argmax(logits, dim=1)
    return torch.mean((y_hat==labels).float()).item()

def cal_auc_score(labels, logits):
    return logits, roc_auc_score(labels.cpu().numpy(),torch.softmax(logits, dim=1)[:,1].cpu().detach().numpy()) 

def accuracy_threshold(logits, graph, threshold):
    out_observe = torch.softmax(logits, dim=1)
    pred_prob, pred_y = torch.max(out_observe, dim=1)
    pred_prob = pred_prob[graph.node_pseudolabel == -1]
    pred_y = pred_y[graph.node_pseudolabel == -1]
    labels = graph.y[graph.node_pseudolabel == -1][pred_prob>threshold]
    pred_y = pred_y[pred_prob>threshold]
    threshold_accuracy = torch.mean((pred_y==labels)*1.)
    return threshold_accuracy, pred_y.size()

class EarlyStopper:
    def __init__(self, patience=20, min_delta=0.01, max_iter=200):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.acc_record = -np.inf
        self.epoch_counter =0
        self.max_iter = max_iter
        self.loss = np.inf

    def reset(self):
        self.counter = 0
        self.acc_record = -np.inf
        self.loss = np.inf
        
    def early_stop(self, epoch_num, acc_record):
        """
            When epoch>max_iter or the number of times when acc_record>min_acc accumulate to self.patience

            Returns:
                True: need to stop
                False: continue
        """
        if acc_record > self.acc_record:
            self.epoch_counter=epoch_num
            self.acc_record = acc_record
            self.counter = 0

        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        
        
        
        if epoch_num >= self.max_iter:
            return True
        
        return False



if __name__ == "__main__":
    for data in datanames:
        dataset = load_nc_dataset(data)
        split_dataset(dataset,0.2,0.8)