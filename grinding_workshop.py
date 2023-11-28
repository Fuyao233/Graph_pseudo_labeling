import torch
from torch_geometric.data import Data
from utils import load_dataset, split_dataset, prepocessing
from model import GCN
datanames = ['twitch-e', 'fb100', 'ogbn-proteins', 'deezer-europe', 'arxiv-year', 'pokec', 'snap-patents',
             'yelp-chi', 'ogbn-arxiv', 'ogbn-products', 'Cora', 'CiteSeer', 'PubMed', 'chameleon', 'cornell',
             'film', 'squirrel', 'texas', 'wisconsin', 'genius', 'twitch-gamer', 'wiki']
# datanames = ['fb100']

# for data in datanames:
#     dataset = load_dataset(data)
#     print(data, len(dataset))
data = 'yelp-chi'
dataset = load_dataset(data)
split_dataset(dataset, 0.2, 0.8)

graph = prepocessing(dataset)

model = GCN(input_dim=dataset[0][0]['node_feat'].size()[1], hidden_dim=32, output_dim=graph.num_class,num_layers=3,dropout=0.7)
output = model(graph)
print(output)

