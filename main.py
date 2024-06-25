import argparse
from train import * 
from torch_geometric.utils import subgraph
import wandb
from wandb_config import *


sweep_configuration = {
    "method": "bayes",
    "name": "sweep",
    "metric": {"goal": "maximize", "name": "test_acc_of_early_stop"},
    "parameters": {
        # "epochs": {"values": [ 3 ]},
        "main_loss_weight": {"values": [0.4, 0.6, 0.8, 1]},
        # "num_layers": {"values": [3,4,]},
        # "dropout": {"values": [0.5, 0, 0.3, 0.6 ]},
        # "weight_decay":{"values":[0.000,0.0001,0.001, 0.005,0.0005,0.01]},
        # "h":{"values":[ 0.5, 0.2,0.1]},
    },
}

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='wisconsin')
parser.add_argument('--gpu', type=int, default=4)
parser.add_argument('--save_root', type=str, default='record')

# hyper-parameter
parser.add_argument('--max_iteration', type=int, default=8)
# for basis
parser.add_argument('--h', type=float, default=0.2)

# for autoencoder
parser.add_argument('--autoencoder_weight', type=float, default=0.01)
# for train
parser.add_argument('--warm_up', type=bool, default=True)
parser.add_argument('--upper_bound', type=bool, default=False)
parser.add_argument('--main_loss_weight', type=int, default=0.7)
parser.add_argument('--A_B_ratio', type=int, default=0.5)
# for flexmatch
parser.add_argument('--flexmatch_weight', type=float, default=0.6)
parser.add_argument('--node_threshold', type=float, default=0.75)
parser.add_argument('--edge_threshold', type=float, default=0.5)

# for dataset 
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--train_ratio', type=float, default=0.2)
parser.add_argument('--test_ratio', type=float, default=0.2)
parser.add_argument('--val_ratio', type=float, default=0.2)
parser.add_argument('--unlabel_ratio', type=float, default=0.4)
parser.add_argument('--metric', type=str, default='auc')
parser.add_argument('--dataset_balanced', type=bool, default=False)
# for model
parser.add_argument('--model_name', type=str, default='mlp') # also the embedding dimension of encoder
parser.add_argument('--hidden_dim', type=int, default=32) # also the embedding dimension of encoder
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--soft_flag', type=bool, default=False)
# for optimizer
parser.add_argument('--lr', type=float, default=1)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=0.001)

args = parser.parse_args()

seed_all(args.seed)
dataset = load_dataset(args.dataset)
sweep_id = wandb.sweep(sweep=sweep_configuration, project=f'layer_loss_')

def main():
    
    wandb.init()
    
    dataset = load_dataset(args.dataset)
    split_dataset_balanced(dataset, args)
    graph = preprocessing(dataset)
    model = load_model(args.model_name, graph, args)
    
    if args.model_name == 'ourModel':
        model = load_model('mlp', graph, args)
    
    print(args.soft_flag)
    trainer = Trainer(graph, model, args=args)
    
    acc = trainer.train()

if __name__ == "__main__":
    wandb.agent(sweep_id, function=main, count=50)
    wandb.finish()