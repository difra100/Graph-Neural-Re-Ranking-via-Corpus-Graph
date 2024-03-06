from torch_geometric.nn import SAGEConv, GCNConv, GATConv
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn import metrics

import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torchmetrics.functional.retrieval import retrieval_normalized_dcg
from  torchmetrics.functional import retrieval_recall, retrieval_precision


class GNN_NR(nn.Module):
    def __init__(self, input_features, config, output_dim = 1, device='cpu'):
        super().__init__()

        self.conf = config

        # Encoder MLP with batch normalization and dropout
        self.enc = nn.Sequential(
            nn.Linear(input_features, config.hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(config.hidden_dim),
            nn.Dropout(config.dropout_prob)
        )

        # Decoder MLP with batch normalization and dropout
        self.dec = nn.Sequential(
            nn.Linear(config.hidden_dim if config.conv_type != 'gat' else config.heads*config.hidden_dim,  output_dim),
            nn.ReLU(),  # You can change the activation function here if needed
            nn.BatchNorm1d( output_dim),
            nn.Dropout(config.dropout_prob)
        )

        
        if config.conv_type == 'gcn':
            self.gnnconv = GCNConv
            self.layers = [self.gnnconv(config.hidden_dim, config.hidden_dim).to(device) for _ in range(config.n_layers)]

        if config.conv_type == 'gat':
            self.gnnconv = GATConv
            self.layers = [self.gnnconv(config.hidden_dim, config.hidden_dim, heads = config.heads).to(device)]
            for _ in range(config.n_layers-1):
                self.layers.append(self.gnnconv(config.heads * config.hidden_dim, config.hidden_dim, heads = config.heads).to(device)) 

        self.neighbors = config.neighbors
        # Graph convolution layers
        

        self.reset_parameters()
        self.to(device)

    def reset_parameters(self):
        for module in [self.enc, self.dec]:
            if isinstance(module, nn.Sequential):
                for layer in module:
                    if isinstance(layer, nn.Linear):
                        layer.reset_parameters()
        
        for el in self.layers:
            el.reset_parameters()


    def forward(self, x, edge_index):
    
        x = self.enc(x)

        if not self.neighbors:
            edge_index = (torch.zeros((edge_index.shape[0], edge_index.shape[1])).to(torch.int64)).to(x.device.type)
        
        for layer in self.layers:

            if self.conf.conv_type != 'gat' or self.conf.heads == 1:
                x = x + F.relu(layer(x, edge_index))
            else:
                x = F.relu(layer(x, edge_index))


        output = self.dec(x)

        return output
    

class MLP(nn.Module):
    def __init__(self, input_features, hidden_dim, output_dim = 1, n_layers=1, device='cpu', dropout_prob=0.5):
        super(MLP, self).__init__()

        layers = []
        # Input layer
        layers.append(nn.Linear(input_features, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.Dropout(dropout_prob))

        # Hidden layers
        for _ in range(n_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(dropout_prob))

        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(nn.ReLU())  # You can change the activation function here if needed
        layers.append(nn.BatchNorm1d(output_dim))
        layers.append(nn.Dropout(dropout_prob))

        self.mlp = nn.Sequential(*layers)

        self.reset_parameters()
        self.to(device)

    def reset_parameters(self):
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()

    def forward(self, x):
        return self.mlp(x)



class GNN_LG(nn.Module):

    def __init__(self, input_features, config, modality = 'local', conv_type = 'gcn', device = 'cpu'):
        super(GNN_LG, self).__init__()

        self.modality = modality
        self.conv_type = conv_type

        if self.modality == 'local':
            
            
            self.GNN = GNN_NR(input_features, config, output_dim = config.hidden_dim, device = device)
            self.mlp = MLP(input_features, config.hidden_dim, output_dim = config.hidden_dim, n_layers=config.n_layers, device = device, dropout_prob=config.dropout_prob)

            self.mlp_final = MLP(2*config.hidden_dim, config.hidden_dim, output_dim = 1, n_layers=config.n_layers, device = device, dropout_prob=config.dropout_prob)
        
        
        ### Work in progress Global features
            
    def forward(self, x, edge_index):

        if self.modality == 'local':
            x_local, x_individual = x.clone(), x.clone()

            z_local = self.GNN(x_local, edge_index)
            z_individual = self.mlp(x_individual)
         
            # print(z_individual.shape)

            z_tot = torch.cat((z_local, z_individual), dim = -1)


            y_pred = self.mlp_final(z_tot)


            return y_pred



        



class ListNetLoss(nn.Module):

    def __init__(self):
        super(ListNetLoss, self).__init__()

    def forward(self, y_pred, y_true, eps=1e-15):
        
        """
        ListNet loss introduced in "Learning to Rank: From Pairwise Approach to Listwise Approach".
        :param y_pred: predictions from the model, shape [batch_size, slate_length]
        :param y_true: ground truth labels, shape [batch_size, slate_length]
        :return: loss value, a torch.Tensor
        """

        y_pred = y_pred.clone().squeeze()
        y_true = y_true.clone().squeeze()
        
        preds_smax = F.softmax(y_pred)
        true_smax = F.softmax(y_true)

        preds_smax = preds_smax + eps
        preds_log = torch.log(preds_smax)

        return torch.mean(-torch.sum(true_smax * preds_log))
    

class ListMLELoss(nn.Module):

    def __init__(self):
        super(ListMLELoss, self).__init__()

    def forward(self, y_pred, y_true, eps=1e-15, padded_value_indicator=-1.):
        
        """
        ListMLE loss introduced in "Listwise Approach to Learning to Rank - Theory and Algorithm".
        :param y_pred: predictions from the model, shape [batch_size, slate_length]
        :param y_true: ground truth labels, shape [batch_size, slate_length]
        :return: loss value, a torch.Tensor
        """

        y_pred = y_pred.clone().squeeze(-1).unsqueeze(0)
        y_true = y_true.clone().squeeze(-1).unsqueeze(0)


        # shuffle for randomised tie resolution
        random_indices = torch.randperm(y_pred.shape[-1])
        y_pred_shuffled = y_pred[:, random_indices]
        y_true_shuffled = y_true[:, random_indices]

        y_true_sorted, indices = y_true_shuffled.sort(descending=True, dim=-1)

        mask = y_true_sorted == padded_value_indicator

        preds_sorted_by_true = torch.gather(y_pred_shuffled, dim=1, index=indices)
        preds_sorted_by_true[mask] = float("-inf")

        max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)

        preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values

        cumsums = torch.cumsum(preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1).flip(dims=[1])

        observation_loss = torch.log(cumsums + eps) - preds_sorted_by_true_minus_max

        observation_loss[mask] = 0.0

        return torch.mean(torch.sum(observation_loss, dim=1))



class TrainingModule(pl.LightningModule):

    def __init__(self, model, lr, wd, aggr, model_family, loss_type = 'mse', ndcgk = [10, 20], recall = [10, 20], precision = [10, 20]):
        super().__init__()
        self.model = model
        self.lr = lr
        self.wd = wd
        self.aggr = aggr
        self.model_family = model_family
        self.ndcgk = ndcgk
        self.recall = recall
        self.prec = precision

        if loss_type == 'mse':
            self.loss = nn.MSELoss()

        elif loss_type == 'listnet':
            self.loss = ListNetLoss()
        
        elif loss_type == 'listmle':
            self.loss = ListMLELoss()

        self.train_prop = {'loss': []}
        self.test_prop = {'loss': []}

        for i in self.ndcgk:
            self.train_prop[f'nDCG@{str(i)}'] = [] 
            self.test_prop[f'nDCG@{str(i)}'] = []
        
        for i in self.recall:
            self.train_prop[f'recall@{str(i)}'] = [] 
            self.test_prop[f'recall@{str(i)}'] = []

        for i in self.prec:
            self.train_prop[f'precision@{str(i)}'] = [] 
            self.test_prop[f'precision@{str(i)}'] = []

    def training_step(self, batch, batch_idx):

        x, query_feat, A, y, _ = batch
        

        # print(x.dtype)
        # print(query_feat.dtype)
        # print(A.dtype)
        # print(y.dtype)

        target = y[0].squeeze()


        mask = torch.nonzero(target!=-1)

        
        if self.aggr == 'concat':
            rep_query = torch.repeat_interleave(query_feat, repeats=x.shape[1], dim=1)

            x = torch.cat((x, rep_query), dim = -1)

        elif self.aggr == 'sum':
        

            rep_query = torch.repeat_interleave(query_feat, repeats=x.shape[1], dim=1)
        
            # print(rep_query.shape)
            x = x + rep_query

        elif self.aggr == 'hadamart':
        

            rep_query = torch.repeat_interleave(query_feat, repeats=x.shape[1], dim=1)
        
            # print(rep_query.shape)
            x = x * rep_query
 
            

  
        if self.model_family == 'gcn' or self.model_family == 'gat':
            out = self.model(x[0], A[0])
        else:
            out = self.model(x[0])


        out = out.squeeze()      
        if out[mask].shape[0] == 0:
            # print("Skip training step")
            return
        loss = self.loss(out[mask], target[mask])


        self.train_prop['loss'].append(loss)

        known_target = target[mask]
        known_out = out[mask]

        for metric in self.ndcgk:
            ndcg = retrieval_normalized_dcg(known_out, known_target, top_k = metric)
            self.train_prop['nDCG@'+str(metric)].append(ndcg)
 
        relevant_out = known_out.squeeze()
        relevant_target = (known_target.squeeze() > 0)
      
        
        
        for metric in self.recall:
            
            rec = retrieval_recall(relevant_out, relevant_target, top_k = metric)

            self.train_prop['recall@'+str(metric)].append(rec)

        for metric in self.prec:
            prec = retrieval_precision(relevant_out, relevant_target, top_k=metric)
            self.train_prop['precision@'+str(metric)].append(prec)

        return loss

    def validation_step(self, batch, batch_idx):
        
        if len(self.train_prop['loss']) == 0:
            print("Skip validation check....")
            return
        
        x, query_feat, A, y, _ = batch


        # print(x.dtype)
        # print(query_feat.dtype)
        # print(A.dtype)
        # print(y.dtype)
        target = y[0].squeeze()

        mask = torch.nonzero(target!=-1)
        
        if self.aggr == 'concat':
            rep_query = torch.repeat_interleave(query_feat, repeats=x.shape[1], dim=1)

            x = torch.cat((x, rep_query), dim = -1)

        elif self.aggr == 'sum':

            # print(query_feat.shape)
            rep_query = torch.repeat_interleave(query_feat, repeats=x.shape[1], dim=1)
            # print(rep_query.shape)
            x = x + rep_query
       
        elif self.aggr == 'hadamart':
        

            rep_query = torch.repeat_interleave(query_feat, repeats=x.shape[1], dim=1)
        
            # print(rep_query.shape)
            x = x * rep_query

        if self.model_family == 'gcn' or self.model_family == 'gat':

            out = self.model(x[0], A[0])
        else:
            out = self.model(x[0])


        out = out.squeeze()
        if out[mask].shape[0] == 0:
            # print("Skip training step")
            return      
        loss = self.loss(out[mask], target[mask])
        self.test_prop['loss'].append(loss)
        
        known_target = target[mask]
        known_out = out[mask]

        for metric in self.ndcgk:
            ndcg = retrieval_normalized_dcg(known_out, known_target, top_k = metric)
            self.test_prop['nDCG@'+str(metric)].append(ndcg)
 
        relevant_out = known_out.squeeze()
        relevant_target = (known_target.squeeze() > 0)
      
        
        
        for metric in self.recall:
            
            rec = retrieval_recall(relevant_out, relevant_target, top_k = metric)

            self.test_prop['recall@'+str(metric)].append(rec)

        for metric in self.prec:
            prec = retrieval_precision(relevant_out, relevant_target, top_k=metric)
            self.test_prop['precision@'+str(metric)].append(prec)

        return loss
    
    def test_step(self, batch, batch_idx):

        x, query_feat, A, y, _ = batch

        target = y[0].squeeze()

        mask = torch.nonzero(target!=-1)
        
        if self.aggr == 'concat':
            rep_query = torch.repeat_interleave(query_feat, repeats=x.shape[1], dim=1)

            x = torch.cat((x, rep_query), dim = -1)

        elif self.aggr == 'sum':
            # print(query_feat.shape)
            rep_query = torch.repeat_interleave(query_feat, repeats=x.shape[1], dim=1)

            # print(rep_query.shape)
            x = x + rep_query

        elif self.aggr == 'hadamart':
        
            rep_query = torch.repeat_interleave(query_feat, repeats=x.shape[1], dim=1)
        
            # print(rep_query.shape)
            x = x * rep_query
       

        if self.model_family == 'gcn' or self.model_family == 'gat':
            out = self.model(x[0], A[0])
        else:
            out = self.model(x[0])

        out = out.squeeze()  
        if out[mask].shape[0] == 0:
            # print("Skip training step")
            return    
        loss = self.loss(out[mask], target[mask])
        self.test_prop['loss'].append(loss)

        known_target = target[mask]
        known_out = out[mask]

        for metric in self.ndcgk:
            ndcg = retrieval_normalized_dcg(known_out, known_target, top_k = metric)
            self.test_prop['nDCG@'+str(metric)].append(ndcg)
 
        relevant_out = known_out.squeeze()
        relevant_target = (known_target.squeeze() > 0)
        
        
        for metric in self.recall:
            
            rec = retrieval_recall(relevant_out, relevant_target, top_k = metric)

            self.test_prop['recall@'+str(metric)].append(rec)

        for metric in self.prec:
            prec = retrieval_precision(relevant_out, relevant_target, top_k=metric)
            self.test_prop['precision@'+str(metric)].append(prec)

        return loss

    def configure_optimizers(self):
    
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        
        return self.optimizer
    