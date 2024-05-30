from train import *
import wandb

def main():

    wandb.init()

    config = wandb.config
    args.lr = config.lr 
    args.weight_decay = config.weight_decay
    # args.node_threshold = config.node_threshold 
    args.edge_threshold = config.edge_threshold 
    args.num_layers = config.num_layers
    args.dropout = config.dropout 
    
    if args.model_name == 'ourModel_basis':
        args.h = config.h 
    if args.model_name == 'ourModel':
        args.autoencoder_weight = config.autoencoder_weight
    
    seed = args.seed
    random.seed(seed)  
    torch.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed) 


    dataset = load_dataset(args.dataset)


    split_dataset_balanced(dataset, args)
    graph = preprocessing(dataset)

    print("====================================================")
    if directed_check(graph):
        print("Undirected!")
    else:
        print('Directed!')
    print(f'Dataset: {args.dataset}')
    print(f'To test upper bound? {"Yes" if args.upper_bound else "No"}')
    print("====================================================")

    model = load_model(args.model_name, graph, args)

    if args.model_name == 'ourModel':
        model = load_model('mlp', graph, args)

    # print(args.soft_flag)
    trainer = Trainer(graph, model, args=args)

    # acc = trainer.train(wandb_record=True)
    acc = trainer.train(wandb_record=True)


if __name__ == "__main__":
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='yelp-chi')
    parser.add_argument('--gpu', type=int, default=5)

    # hyper-parameter
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
    parser.add_argument('--edge_threshold', type=float, default=0.7)
    
    # for dataset 
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--train_ratio', type=float, default=0.2)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--unlabel_ratio', type=float, default=0.4)
    parser.add_argument('--metric', type=str, default='auc')
    parser.add_argument('--dataset_balanced', type=bool, default=False)

    # for model
    parser.add_argument('--model_name', type=str, default='ourModel') # also the embedding dimension of encoder
    parser.add_argument('--hidden_dim', type=int, default=32) # also the embedding dimension of encoder
    parser.add_argument('--soft_flag', type=bool, default=True)
    parser.add_argument('--basis_flag', type=bool, default=False) 
    # for optimizer
    parser.add_argument('--momentum', type=float, default=0.9)
    
    args = parser.parse_args()
    
    sweep_configuration = {
        "method": "bayes",
        "name": "sweep",
        "metric": {"goal": "maximize", "name": "test_acc_of_early_stop"},
        "parameters": {
            "lr": {"values": [0.01,0.005, 0.001,0.0005, 0.05, 0.0002,0.002,0.0001,0.0005,]},
            "num_layers": {"values": [3,4,]},
            "dropout": {"values": [0.5, 0, 0.3,0.6 ]},
            "weight_decay":{"values":[0.000,0.0001,0.001, 0.005,0.0005,0.01]},
            # "node_threshold": {"values": [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]},
            "edge_threshold": {"values": [0.3, 0.4, 0.5, 0.7, 0.75, 0.8, 0.9]},
            "h": {"values": [0.05, 0.075, 0.1, 0.15, 0.175, 0.2, 0.25]},
            "autoencoder_weight": {"values": [0.01, 0.05, 0.1]},
        },
    }
    
    
    project_name = f'{args.model_name}_{args.dataset}'
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=project_name)
    
    wandb.agent(sweep_id, function=main, count=60)
    os.environ["WANDB_SILENT"] = "true"
    wandb.finish()