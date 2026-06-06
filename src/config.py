import torch
# Data parameters
dataset_name = '' # BM25_1300_K_16_TENS_DIM_F_64-20240226T140712Z-001 / 'BM25_1300_K_8_TENS_DIM_F_64-20240226T133020Z-001'

project_name = 'Graph Neural Re-Ranking via Corpus Graph'
entity_name = 'difra00'

# System Setting
device = 'cuda' if torch.cuda.is_available() else 'cpu'

n_feats = 768

# Training parameters
epochs = 100
batch_size = 128

seed = 789
seed_list = [789, 234, 567, 890, 123, 456, 42, 321, 654, 987]
loss_type = 'lambdarank'

# Hyperparameters
embedding_name = 'tctcolbert2' #'castorini/tct_colbert-v2-hnp-msmarco' / 'castorini/tct_colbert-msmarco'
K_cg = 8
lr = 0.01
wd = 0
hidden_dim = 128
dropout_prob = 0.6
n_layers = 2
n_layers_mlp = 1
aggr = 'hadamard'
conv_type = 'gcn'
modality = 'multistage'
score = True

if conv_type == 'gat':
    heads = 1

if conv_type == 'signed':
    negatives = 1

negatives = 1
if modality == 'global':
    structure_learning = False
    lamb = 0.5
    pooling = 'hierarchical'
    pooling_ratio = 0.5





# Constants

WORKERS = 1
MAX_DOCS = 1000
MAX_EDGES = 50000
IGNORE_INDEX = -1


sweep_config = {
    'method': 'grid'
}

sweep_config['metric'] = {'name': 'nDCG@10 on test (Mean)',
                          'goal': 'maximize'
                         }


parameters_dict_local = {
    'lr': {
        'values': [1e-2]
    },
    'hidden_dim': {
        'values': [128]
    },
    'wd': {
        'values': [0]
    },
    'n_layers': {
        'values': [1, 2, 3]
    },
    'aggr': {
        'values': ['hadamard']
    },
    'dropout_prob': {
        'values': [0, 0.1, 0.3, 0.5]
    },
    'conv_type': {
        'values': ['gcn', 'gat']
    },
    'loss_type': {
        'values': ['lambdarank']
    },
    'modality': {
        'values': ['multistage']
    }, 
    'n_layers_mlp': {
        'values': [1, 2]
    }, 
    'K_cg': {
        'values': [8]
    },
    'score': {
        'values': [True]
    },
    'heads': {
        'values': [1]
    }, 
    'embedding_name': {
        'values': ['tctcolbert']
    }, 
    'K_multistage': {
        'values': [100, 200]
    },
}

parameters_dict_global = {
    'lr': {
        'values': [1e-2]
    },
    'hidden_dim': {
        'values': [64]
    },
    'wd': {
        'values': [0]
    },
    'n_layers': {
        'values': [1]
    },
    'aggr': {
        'values': ['hadamard']
    },
    'dropout_prob': {
        'values': [0.1, 0.3]
    },
    'conv_type': {
        'values': ['gcn', 'gat', 'gin', 'sage', 'signed']
    },
    'loss_type': {
        'values': ['lambdarank']
    },
    'modality': {
        'values': ['global']
    }, 
    'n_layers_mlp': {
        'values': [1, 2]
    }, 
    'K_cg': {
        'values': [8]
    },
    'score': {
        'values': [True]
    },
    'pooling_ratio': {
        'values': [0.8]
    },
    'pooling': {
        'values': ['hierarchical', 'maxmean']
    },
    'structure_learning': {
        'values': [True, False]
    },
    'lamb': {
        'values': [0.5, 0.8]
    }, 
    'negatives': {
        'values': [0]
    },
    'heads': {
        'values': [0]
    },
    'aggr_sage': {
        'values': ['empty']
    }
    
}

