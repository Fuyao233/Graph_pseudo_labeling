import argparse
from train import * 
from torch_geometric.utils import subgraph
import wandb
from wandb_config import *

def main():
    wandb.init(config=wandb.config)
    args = wandb.config
    dataset = load_dataset(args.dataset)
    
    # if 'test_ratio' in args:
    #     args.test_ratio = None 
    # if 'val_ratio' in args:
    #     args.val_ratio = None
    split_dataset(dataset, args.train_ratio)
    graph = prepocessing(dataset)
    
    # model = ourModel(input_dim=graph.num_features, output_dim=graph.num_class, hidden_dim=64, num_layers=3,dropout=0.3)
    model = ourModel(input_dim=graph.num_features, output_dim=graph.num_class, hidden_dim=args.hidden_dim, num_layers=args.num_layers,dropout=args.dropout)
    
    trainer = Trainer(graph, model, device_num=args.gpu, lr=args.lr, 
                      weight_decay=args.weight_decay, 
                      fixed_threshold=args.fixed_threshold,
                      flex_batch=args.flex_batch,
                      flexmatch_weight=args.flexmatch_weight,
                      autoencoder_weight=args.autoencoder_weight)
    acc = trainer.train()
    wandb.log({"accuracy": acc})
    # save_res(acc.item(), args.dataset)

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
    wandb_config['parameters'].update({
    'dataset': {'value':args.dataset},
    'gpu': {'value':args.gpu},
    'train_ratio': {'value':args.train_ratio}
    })
    for item in range(5):
        sweep_id = wandb.sweep(wandb_config, project=f"{args.dataset}_{args.train_ratio}")
        
        wandb.agent(sweep_id,function=main,count=50)