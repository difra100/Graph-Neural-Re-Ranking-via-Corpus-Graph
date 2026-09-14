
# from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import pyterrier as pt
from pyterrier_dr import TctColBert, FlexIndex
from pyterrier_t5 import MonoT5ReRanker
from pyterrier_adaptive.corpus_graph import NpTopKCorpusGraph
from npids import Lookup

import scipy
import time
if not pt.started():
    pt.init(no_download=True)  # skip Maven DNS lookup; use cached JARs
import torch

import os
from pathlib import Path

from src.config import *
from src.utils import *
import json


class LegacyNpTopKCorpusGraph:
    def __init__(self, graph_path, docnos_path, k=None):
        graph_path = Path(graph_path)
        with (graph_path / 'pt_meta.json').open('rt') as fin:
            self.meta = json.load(fin)

        self._data_k = self.meta['k']
        if k is not None:
            assert k <= self._data_k
        self._k = self._data_k if k is None else k
        self._edges_path = graph_path / 'edges.u32.np'
        self._weights_path = graph_path / 'weights.f16.np'
        self._docnos = Lookup(docnos_path)

    @property
    def edges_data(self):
        edges = np.memmap(self._edges_path, mode='r', dtype=np.uint32).reshape(-1, self._data_k)
        return edges[:, :self._k] if self._k != self._data_k else edges

    @property
    def weights_data(self):
        weights = np.memmap(self._weights_path, mode='r', dtype=np.float16).reshape(-1, self._data_k)
        return weights[:, :self._k] if self._k != self._data_k else weights

    def neighbours(self, docid, weights=False):
        as_str = isinstance(docid, str)
        if as_str:
            docid = self._docnos.inv[docid]
        neigh = self.edges_data[docid]
        if as_str:
            neigh = self._docnos.fwd[neigh]
        if weights:
            return neigh, self.weights_data[docid]
        return neigh


def get_corpus_graph(graph_type, corpus_name, K_cg, corpusgraph_name='corpusgraph_bm25_k16',
                     flex_index=None):
    """Return the corpus graph used to induce query subgraphs.

    graph_type:
      'semantic' -> the local TCT-ColBERT kNN graph under data/msmarco-index_*/corpusgraph_k16
                    (the original GNRR graph).
      'lexical'  -> a BM25 kNN corpus graph downloaded from the pyterrier dataset,
                    i.e. NpTopKCorpusGraph.from_dataset('msmarco_passage', <name>).to_limit_k(K_cg).
    """
    if graph_type == 'lexical':
        g = NpTopKCorpusGraph.from_dataset('msmarco_passage', corpusgraph_name)
        return g.to_limit_k(K_cg)
    if flex_index is None:
        flex_index = FlexIndex(str(corpus_name))
    return load_corpus_graph(flex_index, Path(corpus_name), K_cg)


def load_corpus_graph(flex_index, index_path, k):
    try:
        return flex_index.corpus_graph(k=k)
    except TypeError as exc:
        if 'abstract class CorpusGraph' not in str(exc):
            raise

    graph_path = index_path / f'corpusgraph_k{k}'
    if not (graph_path / 'pt_meta.json').exists():
        candidates = []
        for path in index_path.glob('corpusgraph_k*/pt_meta.json'):
            graph_k = int(path.parent.name.replace('corpusgraph_k', ''))
            if graph_k >= k:
                candidates.append((graph_k, path.parent))
        if not candidates:
            raise FileNotFoundError(f'No corpus graph with k >= {k} found under {index_path}')
        graph_path = sorted(candidates)[0][1]

    docnos_path = graph_path / 'docnos.npids'
    if docnos_path.exists():
        return NpTopKCorpusGraph(graph_path, k=k)

    docnos_path = index_path / 'docnos.npids'
    return LegacyNpTopKCorpusGraph(graph_path, docnos_path, k=k)



class Dataset_terrierlike(Dataset):

    def __init__(self, data_name, K_cg = 8, fast_train = False, fast = True, train = None, indices = None, bm25_prec = None, embedding_name = None, graph_type = 'semantic', corpusgraph_name = 'corpusgraph_bm25_k16'):

        # Train Data
        self.dataset = pt.get_dataset(data_name)
        
        
        # Full Corpus
        self.dataset_retr = pt.get_dataset('irds:msmarco-passage')
        self.add_text = pt.text.get_text(self.dataset_retr, 'text') # This is the text field for the documents

        self.K_cg = K_cg # Corpus Graph n° of neighbors
        
        self.embedding_name = embedding_name

        if self.embedding_name.startswith('tctcolbert'):
            self.emb_dim = 768
            if not fast_train:
                self.encoder = TctColBert('castorini/tct_colbert-msmarco' if self.embedding_name == 'tctcolbert' else 'castorini/tct_colbert-v2-hnp-msmarco', device='cuda' if WORKERS == 0 else 'cpu')
        elif self.embedding_name.startswith('contriever'):
            self.emb_dim = 768
            if not fast_train:
                from src.contriever_encoder import ContrieverEncoder
                _hf_name = ('facebook/contriever-msmarco' if self.embedding_name == 'contriever-msmarco'
                            else 'facebook/contriever')
                self.encoder = ContrieverEncoder(_hf_name, device='cuda' if WORKERS == 0 else 'cpu')
        
        # Other initial embeddings can be added here
        self.corpus_name = Path('data') / f'msmarco-index_{self.embedding_name}'

        flex_index = FlexIndex(str(self.corpus_name))
        self.indices = indices
        self.graph_type = graph_type
        # Build the corpus graph (semantic TCT kNN, or lexical BM25 kNN from pyterrier).
        self.graph = get_corpus_graph(graph_type, self.corpus_name, self.K_cg,
                                      corpusgraph_name=corpusgraph_name, flex_index=flex_index)

        self.bm25_prec_path = bm25_prec
        
        self.fast = fast
        self.fast_train = fast_train

        if self.fast and not self.fast_train:
            self.payload = flex_index.payload()

        self.queries = self.dataset.get_topics()
        self.qrels = self.dataset.get_qrels()

        self.indices = list(map(str, self.indices))
        self.queries = self.queries[self.queries['qid'].isin(self.indices)]
        self.qrels = self.qrels[self.qrels['qid'].isin(self.indices)]
        
        self.train = train

        _suffix = '' if self.embedding_name in ('tctcolbert', 'tctcolbert2') else f'_{self.embedding_name}'
        if self.train:
            self.path = f'data/msmarco_data/train_data_fast{_suffix}/'
        else:
            self.path = f'data/msmarco_data/val_data_fast{_suffix}/'
        
    def gat_sample_fast(self, idx):
        
        q = self.queries.iloc[idx:idx+1, :]
        
        qid = str(q['qid'].values[0])
        
        # If needed to use other datasets change the path here

        try:
            x = torch.load(self.path + 'tensors/' + 'qid_' + qid + '_tensors/doc_feat_tensor_new.pt')
        
        except FileNotFoundError: # Some file do not have any bm25 output
            
            return torch.randn((MAX_DOCS, self.emb_dim)), torch.randn((1, self.emb_dim)), torch.zeros((2, MAX_EDGES), dtype=torch.int64), torch.randn((MAX_DOCS)), torch.tensor([IGNORE_INDEX]), torch.tensor([0, 0]), {k: '0' for k in range(MAX_DOCS)}, torch.zeros((MAX_EDGES,), dtype=torch.float)

        original_dim_x = x.shape[0]
        
        x = torch.nn.functional.pad(x, (0, 0, 0, MAX_DOCS - original_dim_x), 'constant', 0)
    
        A = torch.load(self.path + 'tensors/' + 'qid_' + qid + '_tensors/adjacency_matrix.pt')
        original_dim_A = A.shape[1]
        padded_edges = torch.zeros((2, MAX_EDGES), dtype=torch.int64)
        padded_edges[:, :A.shape[1]] = A
        A = padded_edges
        # fast tensors store a binary adjacency (no weights) -> unit edge weights
        edge_weight_fast = torch.zeros((MAX_EDGES,), dtype=torch.float)
        edge_weight_fast[:original_dim_A] = 1.0

        y = torch.load(self.path + 'tensors/' + 'qid_' + qid + '_tensors/qrels_tensor_new.pt')
        y = torch.nn.functional.pad(y, (0, MAX_DOCS - original_dim_x), 'constant', 0)

        q_ids = qid
        query_feat = torch.load(self.path + 'tensors/' + 'qid_' + qid + '_tensors/query_tensor_new.pt')

        # Escape condition, whether we have only one document or 0, or also if the query has no relevant documents.
        if original_dim_x <= 1 or len(self.qrels[self.qrels['qid'] == qid]) == 0:
            # print(f'Query {qid} has no relevant documents')
            return torch.randn((MAX_DOCS, self.emb_dim)), torch.randn((1, self.emb_dim)), torch.zeros((2, MAX_EDGES), dtype=torch.int64), torch.randn((MAX_DOCS)), torch.tensor([IGNORE_INDEX]), torch.tensor([0, 0]), {k: '0' for k in range(MAX_DOCS)}, torch.zeros((MAX_EDGES,), dtype=torch.float)

        # print(x.shape, query_feat.shape, A.shape, y[:MAX_DOCS].shape, torch.tensor([original_dim_x, original_dim_A]))


        return x, query_feat, A, y[:MAX_DOCS], torch.tensor([int(q_ids)]), torch.tensor([original_dim_x, original_dim_A]), {k: '0' for k in range(MAX_DOCS)}, edge_weight_fast

    def get_sample_slow(self, idx):

        # Initialize Queries
        q = self.queries.iloc[idx:idx+1, :]
        
        q.reset_index(drop=True, inplace=True)

        # Initialize Qrels
        q_rels = self.qrels[self.qrels['qid'] == q['qid'][0]]
        
        # print("Remaining qrels: ", len(q_rels))
        
        # Get pre-computed BM25 scores and candidates
        with open(self.bm25_prec_path + q['qid'][0] + '.json', 'r') as f:
            candidates = json.load(f)


        candidates = pd.DataFrame(candidates)

        candidates = self.add_text(candidates)

        # Drop duplicates and manage simple exceptions like documents for which one or zero docs are retrieved
        topk_documents_df = candidates.drop_duplicates(subset='docno')

        docs = topk_documents_df['docno']

        max_num = len(topk_documents_df)
        
        #queries = q.query[0]

        if max_num <= 1 or len(q_rels) == 0:
            # print(f'Query {q["qid"][0]} has no relevant documents')
            return torch.randn((MAX_DOCS, self.emb_dim)), torch.randn((1, self.emb_dim)), torch.zeros((2, MAX_EDGES), dtype=torch.int64), torch.randn((MAX_DOCS)), torch.tensor([IGNORE_INDEX]), torch.tensor([0, 0]), {k: '0' for k in range(MAX_DOCS)}, torch.zeros((MAX_EDGES,), dtype=torch.float)
        
        # Encode Queries
        query_enc = self.encoder.encode_queries(q['query'])

        # Keep Track of the doc ordering, and followingly build the adjacency matrix
        docno_to_index = {docno: idx for idx, docno in enumerate(topk_documents_df['docno'])}
        
        index_to_docno = {docno_to_index[k]: k for k in docno_to_index}

        # Fill the remaining places with 'zeros'
        if len(index_to_docno) < MAX_DOCS:
            u = 0
            for i in range(max_num, MAX_DOCS):
                index_to_docno[max_num + u] = '0'
                u+=1

        # Fast is used whether we want to use pre-computed embeddings
        if not self.fast:
        
            doc_encs = self.encoder.encode_docs(docs)
        
        else:
            doc_encs = np.empty((len(topk_documents_df), query_enc.shape[-1]), dtype=query_enc.dtype)
            
            try:

                for doc in range(len(topk_documents_df)):
                    docno = index_to_docno[doc]
                    real_id = self.payload[0][docno]
                    doc_encs[doc] = self.payload[1][real_id]

            except IndexError:
                doc_encs = self.encoder.encode_docs(docs)
     

        # weighted query-induced subgraph (edge weights = cosine/BM25 similarity)
        A, edge_weight = induced_weighted_edges(topk_documents_df, self.graph, docno_to_index)

        query_feat = (torch.from_numpy(query_enc).clone().unsqueeze(0)).detach()

        x = (torch.from_numpy(doc_encs).clone()).detach() 
        
        
        merged_df = topk_documents_df.merge(q_rels.loc[:, ['qid', 'docno', 'label']], on=['qid', 'docno'], how='left')
        
        merged_df = merged_df.fillna(-1)
        

        original_dim_x = x.shape[0]
        x = torch.nn.functional.pad(x, (0, 0, 0, MAX_DOCS - original_dim_x), 'constant', 0)
        y = torch.from_numpy(np.array(merged_df['label']))
        y = torch.nn.functional.pad(y, (0, MAX_DOCS - original_dim_x), 'constant', 0)
        
        original_dim_A = A.shape[1]
        padded_edges = torch.zeros((2, MAX_EDGES), dtype=torch.int64)
        padded_edges[:, :A.shape[1]] = A
        A = padded_edges
        padded_w = torch.zeros((MAX_EDGES,), dtype=torch.float)
        padded_w[:edge_weight.shape[0]] = edge_weight
        edge_weight = padded_w

        doc_mapping = index_to_docno

        #print(f'Time taken for one batch: {end_ - start_} seconds')
        return x, query_feat.squeeze(0), A, y[:MAX_DOCS], torch.tensor([int(q['qid'][0])]), torch.tensor([original_dim_x, original_dim_A]), doc_mapping, edge_weight

        
        
    def __len__(self):
        return len(self.queries)
    
    def __getitem__(self, idx):

        if self.fast_train: # Always is the training data datamodule.
            return self.gat_sample_fast(idx)
        else:
            return self.get_sample_slow(idx)
        

class DataModule_terrier(pl.LightningDataModule):

    def __init__(self, train_path, val_path, batch_size, K_cg = 8, fast_train = False, fast = True, mode = 'hp', embedding_name = None, train_indices = None, val_indices = None, graph_type = 'semantic', corpusgraph_name = 'corpusgraph_bm25_k16'):
        super().__init__()

        self.mode = mode  # "hp" or "test"
        self.train_path, self.val_path = train_path, val_path
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.embedding_name = embedding_name
        self.batch_size = batch_size
        self.fast_train = fast_train
        self.graph_type = graph_type
        self.corpusgraph_name = corpusgraph_name

        # Full Corpus
        self.K_cg = K_cg
        self.fast = fast

    def setup(self, stage=None):
        if stage == 'fit':
            return
        
    def train_dataloader(self):
        
        self.bm25_prec_path = f'./data/msmarco_data/msmarco_pre-computed_bm25/train/'
        
        self.train_set = Dataset_terrierlike(self.train_path, self.K_cg, fast_train = self.fast_train, fast = self.fast, train = True, bm25_prec = self.bm25_prec_path, embedding_name = self.embedding_name, indices = self.train_indices, graph_type = self.graph_type, corpusgraph_name = self.corpusgraph_name)
        
        return DataLoader(self.train_set, shuffle=True, batch_size=self.batch_size, num_workers = WORKERS, pin_memory = True)#, collate_fn = lambda batch: collate_fn(batch, queries=self.train_set.queries, qrels=self.train_set.qrels, bm25=self.bm25, payload=self.train_set.payload, corpus_graph=self.train_set.graph, fast=self.fast, encoder=self.encoder, add_text=self.add_text))

    def val_dataloader(self):
        
        self.bm25_prec_path = f'./data/msmarco_data/msmarco_pre-computed_bm25/val/'

        self.val_set = Dataset_terrierlike(self.val_path, self.K_cg, fast = self.fast, fast_train = self.fast_train, train = False, bm25_prec=self.bm25_prec_path, embedding_name = self.embedding_name, indices = self.val_indices, graph_type = self.graph_type, corpusgraph_name = self.corpusgraph_name)
        
        return DataLoader(self.val_set, shuffle=False, batch_size=self.batch_size, num_workers = WORKERS, pin_memory = True)
        
    
