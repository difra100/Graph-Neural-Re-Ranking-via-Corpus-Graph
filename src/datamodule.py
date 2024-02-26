from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset
import pytorch_lightning as pl
import os

# num_cpu_cores = os.cpu_count()

from src.config import *
from src.utils import *



# Sample raw data example
# sample_data = {'q_id1': {'doc_feat': torch.randn((n_docs, n_feats)),
#                          'adj_matr': torch.zeros((2, n_edges_q_id1)),
#                          'q_rels': torch.randn(n_docs, 1)},
#                 'q_id2': {'doc_feat': torch.randn((n_docs, n_feats)),
#                          'adj_matr': torch.zeros((2, n_edges_q_id2)),
#                          'q_rels': torch.randn(n_docs, 1)},
#                 'q_id3': {'doc_feat': torch.randn((n_docs, n_feats)),
#                                         'adj_matr': torch.zeros((2, n_edges_q_id3)),
#                                         'q_rels': torch.randn(n_docs, 1)}
# }

class Dataset_QD(Dataset):
    def __init__(self, data): #, max_edges):

        self.data = data # Dict form example
        # self.max_edges = max_edges

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):

        
        if isinstance(self.data[idx], tuple):

            x = self.data[idx][0]
            A = self.data[idx][2]
            y = self.data[idx][3]
            query_feat = self.data[idx][1]

            q_ids = self.data[idx][4]

        # doc, query, adj, qrels

        else:

            x = self.data[idx]['doc_feat']
        
            A = self.data[idx]['adj_matrix']
            # padded_edges = torch.zeros((2, self.max_edges))
            # padded_edges[:, ]
            y = self.data[int(idx)]['q_rels']
            q_ids = self.data[int(idx)]['q_ids']
            query_feat = self.data[int(idx)]['query_feat']


        return x, query_feat, A, y, q_ids



        


class DataModule(pl.LightningDataModule):

    def __init__(self, train_set, val_set, test_set, mode, batch_size):

        self.mode = mode  # "hp" or "test"
        self.train_set, self.val_set, self.test_set = train_set, val_set, test_set
        self.batch_size = batch_size

    def setup(self, stage=None):
        if stage == 'fit':
            return
        
    def train_dataloader(self, *args, **kwargs):
        self.train_set = Dataset_QD(self.train_set)
        return DataLoader(self.train_set, shuffle=True, batch_size=self.batch_size)#, num_workers = num_cpu_cores)

    def val_dataloader(self, *args, **kwargs):
        if self.mode == 'hp':
            self.val_set = Dataset_QD(self.val_set)
            return DataLoader(self.val_set, shuffle=False, batch_size=self.batch_size)#, num_workers = num_cpu_cores)
        elif self.mode == 'test':
            self.test_set = Dataset_QD(self.test_set)
            return DataLoader(self.test_set, shuffle=False, batch_size=self.batch_size)#, num_workers = num_cpu_cores)