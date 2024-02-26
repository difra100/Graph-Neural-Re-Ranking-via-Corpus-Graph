import numpy as np
import random
import torch
import torch_geometric
import pytorch_lightning as pl
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

def set_seed(seed_value):
    # Set seed for NumPy
    # np.random.seed(seed_value)

    # # Set seed for Python's random module
    # random.seed(seed_value)

    # # Set seed for PyTorch
    # torch.manual_seed(seed_value)

    # # Set seed for GPU (if available)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(seed_value)
    #     torch.cuda.manual_seed_all(seed_value)

    #     # Set the deterministic behavior for cudNN
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False

    # # Set seed for PyTorch Geometric
    # torch_geometric.seed_everything(seed_value)

    # Set seed for PyTorch Lightning
    pl.seed_everything(seed_value)
    print(f"{seed_value} have been correctly set!")
    # if torch.cuda.is_available():
    # #     # torch.cuda.manual_seed(seed_value)
    # #     # torch.cuda.manual_seed_all(seed_value)

    # #     # Set the deterministic behavior for cudNN
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False
def set_determinism_the_old_way(deterministic: bool):
    # determinism for cudnn
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        # fixing non-deterministic part of horovod
        # https://github.com/PyTorchLightning/pytorch-lightning/pull/1572/files#r420279383
        os.environ["HOROVOD_FUSION_THRESHOLD"] = str(0)


def get_n_params(model):
    '''
    Count the NN parameters in a nn.Module object of pytorch.
    '''
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def coo_to_adjacency_matrix(coo_matrix):
    # Extract row, col, and data from the COO matrix
    row_indices, col_indices = coo_matrix

    # Determine the size of the sparse tensor based on the maximum node index
    size = (max(row_indices.max(), col_indices.max()) + 1, max(row_indices.max(), col_indices.max()) + 1)

    # Create a sparse COO tensor
    sparse_coo = torch.sparse_coo_tensor(indices=torch.stack([row_indices, col_indices]), values=torch.ones_like(row_indices).float(), size=size)

    # Convert the sparse COO tensor to a dense adjacency matrix
    adjacency_matrix = sparse_coo.to_dense()

    return adjacency_matrix


def adj_matrix_to_coo(adj):
    return torch.nonzero(adj).T


def retrieve_dataset_from_file(dataset_name):
    dataset = []
    path = f'./data/{dataset_name}/tensors/'
    list_of_files = os.listdir(path)
    for file in list_of_files:
        diz = {}
        adj_matr = torch.load(path + file + '/adjacency_matrix.pt')
        doc_feat = torch.load(path + file + '/doc_feat_tensor.pt').float()
        qrels_tensor = torch.load(path + file + '/qrels_tensor.pt').float()
        query_tensor = torch.load(path + file + '/query_tensor.pt').float()
        qid = int(file[4])
        diz['doc_feat'] = doc_feat
        diz['adj_matrix'] = adj_matr
        diz['q_ids'] = qid
        diz['q_rels'] = qrels_tensor
        diz['query_feat'] = query_tensor

        dataset.append(diz)
    return dataset



def split_train_val(elements, train_percentage=0.8, val_percentage=0.2, random_seed=None):
   
    # Ensure the percentages sum to 1.0
    total_percentage = train_percentage + val_percentage
    assert total_percentage <= 1.0, "The sum of train and validation percentages must be less than or equal to 1.0."

    # Split the data
    train_set, val_set = train_test_split(elements, test_size=val_percentage, random_state=random_seed)

    return train_set, val_set

def k_fold_cross_validation(elements, k=5, random_seed=None):
  
    # Ensure k is less than or equal to the length of the data
    assert k <= len(elements), "The number of folds (k) should be less than or equal to the length of the data."

    # Initialize k-fold splitter
    kfold = KFold(n_splits=k, shuffle=True, random_state=random_seed)

    # Create an empty list to store fold sets
    fold_sets = []

    # Iterate over k folds
    for train_indices, val_indices in kfold.split(elements):
        train_set = [elements[i] for i in train_indices]
        val_set = [elements[i] for i in val_indices]
        fold_sets.append((train_set, val_set))

    return fold_sets