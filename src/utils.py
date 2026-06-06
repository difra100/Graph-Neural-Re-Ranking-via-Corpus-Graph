import numpy as np
import random
import torch
import torch_geometric
import pytorch_lightning as pl
import os
import re
import json
import pandas as pd
import ir_measures
from typing import Any, Dict, List, Set, Tuple
import scipy
import scipy.sparse
import warnings
from torch_geometric.utils import from_scipy_sparse_matrix
import ir_measures
from ir_measures import *



def set_seed(seed_value):

    # Set seed for PyTorch Lightning
    pl.seed_everything(seed_value)
    print(f"{seed_value} have been correctly set!")


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
    pattern = r'\d+'


    for file in list_of_files:
        
        diz = {}
        adj_matr = torch.load(path + file + '/adjacency_matrix.pt')
        doc_feat = torch.load(path + file + '/doc_feat_tensor_new.pt').float()
        qrels_tensor = torch.load(path + file + '/qrels_tensor_new.pt').float()
        query_tensor = torch.load(path + file + '/query_tensor_new.pt').float()
        
        match = re.search(pattern, file)
        qid = int(match.group())
   
        diz['doc_feat'] = doc_feat
        diz['adj_matrix'] = adj_matr
        diz['q_ids'] = qid
        diz['q_rels'] = qrels_tensor
        diz['query_feat'] = query_tensor

        dataset.append(diz)

    return dataset

def get_doc_df(qid, scores, index_to_docno = None):

    '''
    Returns a dataframe with the following columns: query_id, doc_id, score, rank.
    '''
    qid_column = np.full(len(index_to_docno), qid,dtype=int)

    rank = torch.arange(0, len(index_to_docno)) 

    df = pd.DataFrame({'query_id': qid_column, 'doc_id': [index_to_docno[i] for i in range(len(scores))], 'score': scores.squeeze().numpy()})
    
    # They are sorted from the highest score to the lowest, than the rank column is applied.
    df.sort_values(by=['score'], ascending=False, inplace=True)
    df['rank'] = rank.numpy()
    df.query_id = df.query_id.astype(int)
    return df


def get_new_repr_docs(init_df):
    diz_docs = {}

    for _, row in init_df.iterrows():
        query_id = str(row['query_id'])
        doc_id = row['doc_id']
        score = float(row['score'])

        if query_id not in diz_docs:
            diz_docs[query_id] = {}

        diz_docs[query_id][doc_id] = score

    return diz_docs


def get_key_from_value(dictionary, value):
    for key, val in dictionary.items():
        if val == value:
            return key
    return None  # Value not found


def generate_corpus_subgraph_induced_by_query(                                          
    topk_documents_df: pd.DataFrame,
    complete_corpus_graph
) -> Dict[str, List[str]]:
    """
    Constructs and refines a corpus subgraph, focusing on relationships within a document subset.

    Conceptual Steps:
    1. Define Nodes: Identify and set the documents of interest as nodes in our subgraph. This step
    uses the 'documents_df' to extract document numbers, which will serve as nodes.

    2. Draw Edges: For each node, retrieve potential connections (edges) from the complete corpus graph.
    This involves fetching neighbors for each document from the comprehensive graph structure.

    3. Filter Edges: Refine the connections by ensuring each node (document) only connects to other nodes
    (documents) within our subset. This filtering process removes edges that lead outside the
    specified subset, maintaining the subgraph's integrity.

    4. Construct Subgraph: Populate the subgraph with nodes and their valid, filtered connections. This
    results in a dictionary where each key is a document number, and its value is a list of neighbor
    document numbers—all within the subset (i.e., valid neighbours).

    Args:
    - documents_df (pd.DataFrame): DataFrame containing documents of interest, identified by 'docno'.
    - graph_reference (NpTopKCorpusGraph): The complete corpus graph for neighbor retrieval.

    Returns:
    - Dict[str, List[str]]: Represents the corpus subgraph. Keys are document numbers ('docno'),
    and values are lists of neighbor document numbers, ensuring all are within the specified subset.
    """

    # Step 1: Define Nodes
    # Extract a set of document numbers to serve as valid nodes within our subgraph.
    valid_docnos = set(topk_documents_df['docno'])

    # Initialize the subgraph
    corpus_subgraph = {}
    found = 0
    for docno in valid_docnos:
        # Step 2: Draw Edges
        # Retrieve neighbors for the current document from the complete corpus graph.
        # CHANGE
        try:
            all_neighbors = complete_corpus_graph.neighbours(docno)
            found += 1
        except LookupError:
            warnings.warn(f"Document {docno} not found in the corpus graph.")
            continue
        # Step 3: Filter Edges
        # Filter these neighbors to include only those also present in our subset (valid_docnos).
        valid_neighbors = [neighbor for neighbor in all_neighbors if neighbor in valid_docnos]

        # Step 4: Construct Subgraph
        # Update our subgraph to include the current document and its filtered neighbors.
        corpus_subgraph[docno] = valid_neighbors  # Populate subgraph
    
    return corpus_subgraph
    
def build_adjacency_matrix(subgraph: Dict[str, list], docno_to_index: Dict[str, int]) -> np.ndarray:
    """
    Generates an adjacency matrix from a subgraph and a mapping of document numbers to indices.

    Parameters:
    subgraph (Dict[str, list]): A dictionary representing the subgraph with document numbers as keys.
    docno_to_index (Dict[str, int]): A dictionary mapping document numbers to their respective indices.

    Returns:
    np.ndarray: A symmetric adjacency matrix representing the graph.
    """
    # Error handling: Check if inputs are dictionaries
    if not isinstance(subgraph, dict) or not isinstance(docno_to_index, dict):
        raise ValueError("Both subgraph and docno_to_index must be dictionaries.")

    # Determine the size of the adjacency matrix
    # CHANGE
    matrix_size = len(docno_to_index)
    
    adjacency_matrix = np.zeros((matrix_size, matrix_size), dtype=int)

    # Iterate over each document and its neighbors in the subgraph
    for doc, neighbors in subgraph.items():
        if doc not in docno_to_index:
            raise KeyError(f"Document number {doc} not found in docno_to_index mapping.")
        doc_index = docno_to_index[doc]

        for neighbor in neighbors:
            if neighbor not in docno_to_index:
                raise KeyError(f"Neighbor {neighbor} of document {doc} not found in docno_to_index mapping.")
            
            neighbor_index = docno_to_index[neighbor]

            # Mark the connection in the matrix, ensuring symmetry
            adjacency_matrix[doc_index, neighbor_index] = adjacency_matrix[neighbor_index, doc_index] = 1

    return adjacency_matrix
    
    
def adjacency_matrix_to_coo(adjacency_matrix: np.ndarray) -> torch.Tensor:
    """
    Converts an adjacency matrix to COO format using PyTorch Geometric.

    Parameters:
    adjacency_matrix (np.ndarray): The adjacency matrix to be converted.

    Returns:
    torch.Tensor: Edge index tensor in COO format.
    """
    # Convert the numpy adjacency matrix to a SciPy sparse matrix (COO format)
    scipy_sparse_matrix = scipy.sparse.coo_matrix(adjacency_matrix)

    # Convert the SciPy sparse matrix to PyTorch Geometric COO format
    edge_index, edge_weight = from_scipy_sparse_matrix(scipy_sparse_matrix)

    return edge_index


def compute_output(x, A, query_feat, model, aggr, conv_type):
                
    rep_query = torch.repeat_interleave(query_feat, repeats=x.shape[1], dim=1)

    if aggr == 'concat':

        x = torch.cat((x, rep_query), dim = -1)

    elif aggr == 'sum':

        x = x + rep_query

    elif aggr == 'hadamard':
    
        x = x * rep_query


    if conv_type != 'mlp':

        out = model(x[0], A[0])
        
    else:

        out = model(x[0])

    out = out.squeeze()

    return out


def get_performance_metrics(path_qrels, val_indices, doc_df):
    
    '''
    Get Performance Metrics computed through the ir_measures library...
    '''
    qrels_test = list(ir_measures.read_trec_qrels(f'{path_qrels}'))
    qrels_test = [qrels_test[i] for i in range(len(qrels_test)) if qrels_test[i].query_id in list(map(str, val_indices))]
    judged_indices = [qrels_test[i].query_id for i in range(len(qrels_test)) if qrels_test[i].query_id in list(map(str, val_indices))]


    doc_df.doc_id = doc_df.doc_id.astype(str)
    doc_df.query_id = doc_df.query_id.astype(str)
    doc_df.score = doc_df.score.astype(float)
    doc_df = doc_df[doc_df.query_id.isin(judged_indices)]
    doc_test = doc_df

    output_diz = ir_measures.calc_aggregate([nDCG@10, P(rel = 2)@3, AP(rel=2), RR(rel = 2), R(rel=2)@1000], qrels_test, doc_test)

    return output_diz


# Get the qrels file for the validation set.
# qrels = qrels[['qid', 'iteration', 'docno', 'label']]

# qrels.qid = qrels.qid.astype(str)
# qrels.docno = qrels.docno.astype(str)
# qrels.label = qrels.label.astype(int)
# qrels.iteration = qrels.iteration.astype(int)

# qrels.to_csv('data/msmarco_data/msmarco_data_qrels/qrels_val.txt', sep = ' ', index = False, header = False)


# Get pre-computed bm25 scores.
# pipeline = bm25 #>> pt.text.get_text(pt.get_dataset('irds:msmarco-passage'), 'text')
# data_query = dataset.get_topics()

# for i in range(0, len(data_query)):
#     print(f"Currently processing query {i+1}/{len(data_query)}")
#     query_id = data_query.loc[i].qid
#     path_to_save = f'data/msmarco_data/msmarco_pre-computed_bm25/{mode}/{data_query.loc[i].qid}.json'
    
#     if os.path.exists(path_to_save):
#         print("skipped")
#         continue

#     output = pipeline(data_query.iloc[i:i+1, :])
    
#     with open(path_to_save, 'w') as f:
#         json.dump(output.to_dict(), f)


