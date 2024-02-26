import torch
# Data parameters
dataset_name = 'trial' # 'Oracolar_Prefer_Minus_Ones_BM25_1000_K_64_TENS_DIM_F_64' 'bm25_1000_k_64_np32' / 'Oracolar_BM25_1000_K_64_TENS_DIM_F_64'

project_name = 'Graph Neural Re-Ranking via Corpus Graph'
entity_name = 'difra00'

# System Setting
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# IR parameters
K_bm25 = 1000
K_cg = 16

n_docs = K_bm25
n_feats = 768
n_edges = K_bm25*K_cg

n_edges_q_id1 = 1300
n_edges_q_id2 = 1700
n_edges_q_id3 = 2000

# Training parameters
epochs = 200
batch_size = 1
train_perc = 0.6
val_perc = 0.2
test_perc = 1 - train_perc - val_perc
mode = 'hp'
seed = 42
seed_list = [789, 234, 567, 890, 123]#, 456, 789, 321, 654, 987]
loss_type = 'listmle'
ndcgk = [10, 20]

# Hyperparameters
lr = 0.01
wd = 0.01
hidden_dim = 64
dropout_prob = 0.3
n_layers = 1
aggr = 'hadamart'
conv_type = 'gat'
modality = 'local'
neighbors = True
heads = 1

sweep_config = {
    'method': 'grid'
}

sweep_config['metric'] = {'name': 'AUROC on test (Mean)',
                          'goal': 'maximize'
                         }

parameters_dict_gcn = {
    'lr': {
        'values': [1e-2, 1e-3]
    },
    'hidden_dim': {
        'values': [64, 128, 256]
    },
    'wd': {
        'values': [0, 1e-2]
    },
    'n_layers': {
        'values': [1, 2, 3]
    },
    'aggr': {
        'values': ['hadamart']
    },
    'dropout_prob': {
        'values': [0, 0.3]
    },
    'neighbors': {
        'values': [True]
    },
    'conv_type': {
        'values': ['gcn']
    },
    'loss_type': {
        'values': ['mse', 'listnet', 'listmle']
    }
}

parameters_dict_gat = { # 648
    'lr': {
        'values': [1e-2, 1e-3]
    },
    'hidden_dim': {
        'values': [64, 128, 256]
    },
    'wd': {
        'values': [0, 1e-2]
    },
    'n_layers': {
        'values': [1, 2, 3]
    },
    'aggr': {
        'values': ['hadamart']
    },
    'dropout_prob': {
        'values': [0, 0.3]
    },
    'neighbors': {
        'values': [True]
    },
    'conv_type': {
        'values': ['gat']
    },
    'heads': {
        'values': [1, 2, 4]
    },
    'loss_type': {
        'values': ['mse', 'listnet', 'listmle']
    }
}

parameters_dict_MLP = {
    'lr': {
        'values': [1e-2, 1e-3]
    },
    'hidden_dim': {
        'values': [64, 128, 256]
    },
    'wd': {
        'values': [0, 1e-2]
    },
    'n_layers': {
        'values': [1, 2, 3]
    },
    'aggr': {
        'values': ['hadamart']
    },
    'dropout_prob': {
        'values': [0, 0.3]
    },
    'conv_type': {
        'values': ['mlp']
    },
    'loss_type': {
        'values': ['mse', 'listnet', 'listmle']
    }
}

parameters_dict_local = {
    'lr': {
        'values': [1e-2, 1e-3]
    },
    'hidden_dim': {
        'values': [64, 128, 256]
    },
    'wd': {
        'values': [0, 1e-2]
    },
    'n_layers': {
        'values': [1, 2, 3, 4]
    },
    'aggr': {
        'values': ['concat', 'sum', 'hadamart']
    },
    'dropout_prob': {
        'values': [0, 0.3]
    },
    'conv_type': {
        'values': ['gcn', 'gat', 'mlp']
    },
    'loss_type': {
        'values': ['mse', 'listnet', 'listmle']
    },
    'modality': {
        'values': ['local', 'single']
    },
    'neighbors': {
        'values': [True]
    },
    'heads': {
        'values': [1, 2, 4]
    }
}