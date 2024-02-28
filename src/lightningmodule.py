import torch 
import torch.nn as nn
from sklearn import metrics

import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torchmetrics.functional.retrieval import retrieval_normalized_dcg
from  torchmetrics.functional import retrieval_recall, retrieval_precision


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
        
        for metric in pl_module.recall:
            test_recall = sum(
                pl_module.test_prop['recall@'+str(metric)])/len(pl_module.test_prop['recall@'+str(metric)])
            pl_module.log(name=f'recall@{str(metric)} on test', value=test_recall,
                    on_epoch=True, prog_bar=True, logger=True)
            pl_module.test_prop['recall@'+str(metric)] = []

        for metric in pl_module.prec:
            test_precision = sum(
                pl_module.test_prop['precision@'+str(metric)])/len(pl_module.test_prop['precision@'+str(metric)])
            pl_module.log(name=f'precision@{str(metric)} on test', value=test_precision,
                    on_epoch=True, prog_bar=True, logger=True)
            pl_module.test_prop['precision@'+str(metric)] = []
            

        # Log the metrics
        pl_module.log(name='Loss on test', value=test_loss,
                on_epoch=True, prog_bar=True, logger=True)
        


        # Re-initialize the metrics
        pl_module.train_prop['loss'] = []

        pl_module.test_prop['loss'] = []

 


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
    

# def nDCG(pred, target, K):
#     topk = torch.topk(pred, K)
#     sort_preds = pred[topk.indices]
#     sort_targets = target[topk.indices]
#     indexes = torch.zeros(sort_preds.shape[0]).long()



#     ndcg = RetrievalNormalizedDCG()
#     ndcgK = ndcg(sort_preds, sort_targets, indexes=indexes)

#     return ndcgK