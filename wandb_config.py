wandb_config = {
    "method": 'random',
} 

metric = {
    'name': 'pseudolabel_acc',
    'goal': 'maximize'
}

wandb_config['metric'] = metric

wandb_config['parameters'] = {}

# fixed

# discrete
wandb_config['parameters'].update({
    'hidden_dim' : {
        'values' : [16,32,64,128]
    },
    'num_layers' : {
        'values': [2,3]
    },
    'flex_batch': {
        'values': [128,256,512,1024]
    },
})

# continuous
wandb_config['parameters'].update({
    'autoencoder_weight': {
        'distribution': 'q_uniform',
        'q': 0.1,
        'min': 0.2,
        'max': 0.8
    },
    'flexmatch_weight': {
        'distribution': 'q_uniform',
        'q': 0.1,
        'min': 0.2,
        'max': 0.8
    },
    'fixed_threshold': {
        'distribution': 'uniform',
        'min': 0.6,
        'max': 0.9
    },
    'dropout': {
        'distribution': 'uniform',
        'min': 0,
        'max': 0.6
    },
    'lr': {
        'distribution': 'log_uniform_values',
        'min': 1e-6,
        'max': 0.1
    },
    'weight_decay': {
        'distribution': 'log_uniform_values',
        'min': 1e-10,
        'max': 1e-6
    }
})


# if __name__ == "__main__":
#     print(wandb_config.weight_decay)
