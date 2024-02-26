import torch 
import torch.nn as nn
from sklearn import metrics

import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torchmetrics.functional.retrieval import retrieval_normalized_dcg

from src.GNN import ListNetLoss, ListMLELoss


class Get_Metrics(Callback):

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
       
        # Compute the metrics
        train_loss = sum(
            pl_module.train_prop['loss']) / len(pl_module.train_prop['loss'])
            
        for metric in pl_module.ndcgk:
            train_dcg = sum(
                pl_module.train_prop['nDCG@'+str(metric)])/len(pl_module.train_prop['nDCG@'+str(metric)])
            pl_module.log(name=f'nDCG@{str(metric)} on train', value=train_dcg,
                    on_epoch=True, prog_bar=True, logger=True)
            pl_module.train_prop['nDCG@'+str(metric)] = []
        pl_module.log(name='Loss on train', value=train_loss,
                    on_epoch=True, prog_bar=True, logger=True)
            
        
        test_loss = sum(
            pl_module.test_prop['loss']) / len(pl_module.test_prop['loss'])
        
        pl_module.last_metrics = []

        for metric in pl_module.ndcgk:    
            test_dcg = sum(
                pl_module.test_prop['nDCG@'+str(metric)])/len(pl_module.test_prop['nDCG@'+str(metric)])
 
            pl_module.log(name=f'nDCG@{str(metric)} on test', value=test_dcg,
                    on_epoch=True, prog_bar=True, logger=True)
            pl_module.test_prop['nDCG@'+str(metric)] = []

            pl_module.last_metrics.append(test_dcg)
            

        # Log the metrics
        pl_module.log(name='Loss on test', value=test_loss,
                on_epoch=True, prog_bar=True, logger=True)
        


        # Re-initialize the metrics
        pl_module.train_prop['loss'] = []

        pl_module.test_prop['loss'] = []

 


class TrainingModule(pl.LightningModule):

    def __init__(self, model, lr, wd, aggr, model_family, loss_type = 'mse', ndcgk = [10]):
        super().__init__()
        self.model = model
        self.lr = lr
        self.wd = wd
        self.aggr = aggr
        self.model_family = model_family
        self.ndcgk = ndcgk

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
        loss = self.loss(out[mask], target[mask])
  

        self.train_prop['loss'].append(loss)

        for metric in self.ndcgk:
            ndcg = retrieval_normalized_dcg(out[mask], target[mask], k = metric)
            self.train_prop['nDCG@'+str(metric)].append(ndcg)

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
        loss = self.loss(out[mask], target[mask])
        self.test_prop['loss'].append(loss)
        for metric in self.ndcgk:

            
            ndcg = retrieval_normalized_dcg(out[mask], target[mask], k = metric)
            
            self.test_prop['nDCG@'+str(metric)].append(ndcg)


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
        loss = self.loss(out[mask], target[mask])
        self.test_prop['loss'].append(loss)
        for metric in self.ndcgk:
            # print(out[mask].shape)
            # print(out[mask][:20], target[mask][:20])
            ndcg = retrieval_normalized_dcg(out[mask], target[mask], k = metric)
            # print(ndcg)
            # print("DONE")
            self.test_prop['nDCG@'+str(metric)].append(ndcg)


        return loss


    def configure_optimizers(self):
    
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        
        return self.optimizer
    

# def nDCG(pred, target, K):
#     topk = torch.topk(pred, K)
#     sort_preds = pred[topk.indices]
#     sort_targets = target[topk.indices]
#     indexes = torch.zeros(sort_preds.shape[0]).long()



#     ndcg = RetrievalNormalizedDCG()
#     ndcgK = ndcg(sort_preds, sort_targets, indexes=indexes)

#     return ndcgK