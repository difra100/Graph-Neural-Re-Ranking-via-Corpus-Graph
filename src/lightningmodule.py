import torch 
import torch.nn as nn
from sklearn import metrics

import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

try:
    from allrank.models.losses.lambdaLoss import lambdaLoss
    from allrank.data.dataset_loading import PADDED_Y_VALUE
except ModuleNotFoundError:
    PADDED_Y_VALUE = -1

    def _discounts(size, device):
        ranks = torch.arange(size, device=device, dtype=torch.float32)
        return 1.0 / torch.log2(ranks + 2.0)

    def lambdaLoss(
        y_pred,
        y_true,
        weighing_scheme="lambdaRank_scheme",
        reduction_log="natural",
        padded_value_indicator=PADDED_Y_VALUE,
        **_,
    ):
        losses = []
        for pred_row, true_row in zip(y_pred, y_true):
            valid = true_row != padded_value_indicator
            pred = pred_row[valid]
            target = true_row[valid].float()
            if pred.numel() < 2:
                losses.append(pred.sum() * 0.0)
                continue

            target_diff = target.unsqueeze(1) - target.unsqueeze(0)
            preferred = target_diff > 0
            if not preferred.any():
                losses.append(pred.sum() * 0.0)
                continue

            pred_diff = pred.unsqueeze(1) - pred.unsqueeze(0)
            pair_loss = F.softplus(-pred_diff)

            if weighing_scheme == "lambdaRank_scheme":
                _, pred_order = pred.sort(descending=True)
                inv_rank = torch.empty_like(pred_order)
                inv_rank[pred_order] = torch.arange(pred.numel(), device=pred.device)
                discount = _discounts(pred.numel(), pred.device)[inv_rank]
                ideal_target, _ = target.sort(descending=True)
                ideal_dcg = ((2.0**ideal_target - 1.0) * _discounts(pred.numel(), pred.device)).sum()
                gains = 2.0**target - 1.0
                weights = (
                    (gains.unsqueeze(1) - gains.unsqueeze(0)).abs()
                    * (discount.unsqueeze(1) - discount.unsqueeze(0)).abs()
                )
                pair_loss = pair_loss * (weights / ideal_dcg.clamp_min(1e-8))

            losses.append(pair_loss[preferred].mean())

        if not losses:
            return y_pred.sum() * 0.0

        loss = torch.stack(losses).mean()
        if reduction_log == "binary":
            loss = loss / torch.log(torch.tensor(2.0, device=loss.device))
        return loss

from src.GNN import ListNetLoss, ListMLELoss
from src.utils import *
from src.config import IGNORE_INDEX, MAX_DOCS, MAX_EDGES 



class Get_Metrics(Callback):

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
       

        diz_test = get_performance_metrics(pl_module.qrels_folder, pl_module.eval_indices, pl_module.doc_test_df)

        pl_module.doc_test_df = pd.DataFrame(columns=['query_id', 'doc_id', 'score', 'rank'])

        
        pl_module.last_metrics = []

        test_dcg = diz_test[nDCG@10]
 
        pl_module.log(name=f'nDCG@{str(10)} on test', value=test_dcg,
                on_epoch=True, prog_bar=True, logger=True)

        pl_module.last_metrics.append(test_dcg)
    
        test_p3 = diz_test[P(rel = 2)@3]   

        pl_module.log(name=f'P(rel = 2)@{str(3)} on test', value=test_p3,
                on_epoch=True, prog_bar=True, logger=True)

        test_ap = diz_test[AP(rel = 2)]   
        
        pl_module.log(name=f'AP(rel = 2) on test', value=test_ap,
                on_epoch=True, prog_bar=True, logger=True)        

        test_rr = diz_test[RR(rel = 2)]   
        
        pl_module.log(name=f'RR(rel = 2) on test', value=test_rr,
                on_epoch=True, prog_bar=True, logger=True)        

        
        

        pl_module.train_prop['loss'] = []

        pl_module.test_prop['loss'] = []


class TrainingModule(pl.LightningModule):

    def __init__(self, model, lr, wd, aggr, model_family, dataset_name, K_cg = 8, fast_train = False, qrels_folder = '', loss_type = 'mse', eval_indices= '', exp_name = ''):
        super().__init__()
        self.model = model
        self.lr = lr
        self.wd = wd
        self.aggr = aggr
        self.model_family = model_family
        self.dataset_name = dataset_name
        self.K_cg = K_cg
        self.exp_name = exp_name


        self.qrels_folder = qrels_folder
            

        self.best_metric = 0
        self.loss_type = loss_type
        self.eval_indices = eval_indices
        self.fast_train = fast_train

        if loss_type == 'mse':
            self.loss = nn.MSELoss()

        elif loss_type == 'listnet':
            self.loss = ListNetLoss()
        
        elif loss_type == 'listmle':
            self.loss = ListMLELoss()
        
        elif loss_type == 'ranknet' or loss_type == 'lambdarank':
            self.out = torch.tensor([], device = 'cuda' if torch.cuda.is_available() else 'cpu')
            self.target = torch.tensor([], device = 'cuda' if torch.cuda.is_available() else 'cpu')

        

        self.train_prop = {'loss': []}
        self.test_prop = {'loss': []}


        self.doc_train_df = pd.DataFrame(columns=['query_id', 'doc_id', 'score', 'rank'])
        self.doc_test_df = pd.DataFrame(columns=['query_id', 'doc_id', 'score', 'rank'])
        self.doc_test_df = pd.DataFrame(columns=['query_id', 'doc_id', 'score', 'rank'])


    def training_step(self, batch, batch_idx):

        X, Query_feat, Adj, Y, Qid, original_dims, index_to_docno_batch = batch

        
        loss = 0
        
        for sample_idx in range(X.shape[0]):

            x = X[sample_idx].unsqueeze(0)
            query_feat = Query_feat[sample_idx].unsqueeze(0)
            A = Adj[sample_idx].unsqueeze(0)
            y = Y[sample_idx].unsqueeze(0)
            qid = Qid[sample_idx].item()
  
            
            
            if qid == IGNORE_INDEX:    
                continue

            x = x[:, :original_dims[sample_idx][0], :]
            
            A = A[:, :, :original_dims[sample_idx][1]]

            y = y[:, :original_dims[sample_idx][0]]
            
            # print(x.shape, A.shape, y.shape, query_feat.shape)
        
            
            target = y[0].squeeze(-1)

            mask = torch.nonzero(target!=IGNORE_INDEX)
    

            out = compute_output(x, A, query_feat, self.model, self.aggr, self.model_family)
            
            

            if self.loss_type == 'mse':
                
                if out[mask].shape[0] == 0:
                    continue

                loss_ = self.loss(out[mask], target[mask].type(torch.float32))

                loss += loss_

                
            elif self.loss_type == 'listnet' or self.loss_type == 'listmle':
                target = torch.where(target == IGNORE_INDEX, torch.tensor(0, device = self.device), target)
                
                loss_ = self.loss(out, target)

                loss += loss_


            elif self.loss_type == 'ranknet' or self.loss_type == 'lambdarank':
                
                target = torch.where(target == IGNORE_INDEX, torch.tensor(0, device = self.device), target)
                
                target = F.pad(target, (0, MAX_DOCS - target.shape[0]), value=PADDED_Y_VALUE)
                out_ = F.pad(out.clone(), (0, MAX_DOCS - out.shape[0]), value=PADDED_Y_VALUE)
                
                self.out = torch.cat([self.out, out_.unsqueeze(0)], dim=0)
                self.target = torch.cat([self.target, target.unsqueeze(0).type(torch.float32)], dim=0)


            # inter_df = get_doc_df(qid.cpu(), out.cpu().detach(), path)
    
            # self.doc_train_df = pd.concat([self.doc_train_df, inter_df], axis=0, ignore_index=True)
    
        
        if self.loss_type == 'ranknet':
            loss = lambdaLoss(self.out, self.target, weighing_scheme="rankNet_scheme", reduction_log="natural", padded_value_indicator=PADDED_Y_VALUE)
            self.out = torch.tensor([], device = self.device)
            self.target = torch.tensor([], device = self.device)

        elif self.loss_type == 'lambdarank':
            loss = lambdaLoss(self.out, self.target, weighing_scheme="lambdaRank_scheme", reduction_log="natural", padded_value_indicator=PADDED_Y_VALUE)
            self.out = torch.tensor([], device = self.device)
            self.target = torch.tensor([], device = self.device)
        
        

        if loss == 0:
            print("Skip training check....")
            return

        self.train_prop['loss'].append(loss/X.shape[0])

        return loss/X.shape[0]


    def validation_step(self, batch, batch_idx):
        
        if len(self.train_prop['loss']) == 0:
            print("Skip validation check....")
            return
        
        X, Query_feat, Adj, Y, Qid, original_dims, index_to_docno_batch = batch
    

        loss = 0
        
        for sample_idx in range(X.shape[0]):

            x = X[sample_idx].unsqueeze(0)
            query_feat = Query_feat[sample_idx].unsqueeze(0)
            A = Adj[sample_idx].unsqueeze(0)
            y = Y[sample_idx].unsqueeze(0)
            qid = Qid[sample_idx]
            # print(x.shape, A.shape, y.shape, query_feat.shape)

            if qid == IGNORE_INDEX:    
                continue

            x = x[:, :original_dims[sample_idx][0], :]
            
            A = A[:, :, :original_dims[sample_idx][1]]

            y = y[:, :original_dims[sample_idx][0]]
            
            # print(x.shape, A.shape, y.shape, query_feat.shape)
            

            target = y[0].squeeze(-1)

            if not self.fast_train:
                index_to_docno = {k: index_to_docno_batch[k][sample_idx] for k in range(original_dims[sample_idx][0])}
            else:
                with open ('data/msmarco_data/val_data_fast/' + 'metrics_directory/' + str(qid.item()) + '.json', 'r') as f:
                    index_to_docno = json.load(f)
            
                
                index_to_docno = {int(k): v for k, v in index_to_docno.items()}
            # for k in index_to_docno:
            #     print("VALIDATION: ", index_to_docno[k])
            #     break
            mask = torch.nonzero(target!=IGNORE_INDEX)

            
            out = compute_output(x, A, query_feat, self.model, self.aggr, self.model_family)


            if self.loss_type == 'mse':
                
                if out[mask].shape[0] == 0:
                    continue

                loss_ = self.loss(out[mask], target[mask].type(torch.float32))

                loss += loss_

                
            elif self.loss_type == 'listnet' or self.loss_type == 'listmle':
                target = torch.where(target == IGNORE_INDEX, torch.tensor(0, device = self.device), target)
                
                loss_ = self.loss(out, target)

                loss += loss_

            elif self.loss_type == 'ranknet' or self.loss_type == 'lambdarank':
                
                target = torch.where(target == IGNORE_INDEX, torch.tensor(0, device = self.device), target)
                
                target = F.pad(target, (0, MAX_DOCS - target.shape[0]), value=PADDED_Y_VALUE)
                out_ = F.pad(out.clone(), (0, MAX_DOCS - out.shape[0]), value=PADDED_Y_VALUE)

                
                self.out = torch.cat([self.out, out_.unsqueeze(0)], dim=0)
                self.target = torch.cat([self.target, target.unsqueeze(0).type(torch.float32)], dim=0)


            inter_df = get_doc_df(qid.cpu(), out.cpu().detach(), index_to_docno = index_to_docno)

            self.doc_test_df = pd.concat([self.doc_test_df, inter_df], axis=0, ignore_index=True)
 
            
        
        if self.loss_type == 'ranknet':
            loss = lambdaLoss(self.out, self.target, weighing_scheme="rankNet_scheme", reduction_log="natural", padded_value_indicator=PADDED_Y_VALUE)
            self.out = torch.tensor([], device = self.device)
            self.target = torch.tensor([], device = self.device)

        elif self.loss_type == 'lambdarank':
            loss = lambdaLoss(self.out, self.target, weighing_scheme="lambdaRank_scheme", reduction_log="natural", padded_value_indicator=PADDED_Y_VALUE)
            self.out = torch.tensor([], device = self.device)
            self.target = torch.tensor([], device = self.device)
        
        

        self.test_prop['loss'].append(loss)


        return loss
        
    
    def test_step(self, batch, batch_idx):
        
        
        X, Query_feat, Adj, Y, Qid, original_dims, index_to_docno_batch = batch
    

        loss = 0
        
        for sample_idx in range(X.shape[0]):

            x = X[sample_idx].unsqueeze(0)
            query_feat = Query_feat[sample_idx].unsqueeze(0)
            A = Adj[sample_idx].unsqueeze(0)
            y = Y[sample_idx].unsqueeze(0)
            qid = Qid[sample_idx]
            # print(x.shape, A.shape, y.shape, query_feat.shape)

            if qid == IGNORE_INDEX:    
                continue

            x = x[:, :original_dims[sample_idx][0], :]
            
            A = A[:, :, :original_dims[sample_idx][1]]

            y = y[:, :original_dims[sample_idx][0]]
            
            # print(x.shape, A.shape, y.shape, query_feat.shape)
            
            target = y[0].squeeze(-1)

            if not self.fast_train:
                index_to_docno = {k: index_to_docno_batch[k][sample_idx] for k in range(original_dims[sample_idx][0])}
            else:
                with open ('data/msmarco_data/val_data_fast/' + 'metrics_directory/' + str(qid.item()) + '.json', 'r') as f:
                    index_to_docno = json.load(f)
                
                index_to_docno = {int(k): v for k, v in index_to_docno.items()}
            # for k in index_to_docno:
            #     print("TEST: ", index_to_docno[k])
            #     break

            mask = torch.nonzero(target!=IGNORE_INDEX)

            
            out = compute_output(x, A, query_feat, self.model, self.aggr, self.model_family)    


            if self.loss_type == 'mse':
                
                if out[mask].shape[0] == 0:
                    continue

                loss_ = self.loss(out[mask], target[mask].type(torch.float32))

                loss += loss_

                
            elif self.loss_type == 'listnet' or self.loss_type == 'listmle':
                target = torch.where(target == IGNORE_INDEX, torch.tensor(0, device = self.device), target)
                
                loss_ = self.loss(out, target)

                loss += loss_

            elif self.loss_type == 'ranknet' or self.loss_type == 'neuralndcg' or self.loss_type == 'lambdarank':
                
                target = torch.where(target == IGNORE_INDEX, torch.tensor(0, device = self.device), target)
                
                target = F.pad(target, (0, MAX_DOCS - target.shape[0]), value=PADDED_Y_VALUE)
                out_ = F.pad(out.clone(), (0, MAX_DOCS - out.shape[0]), value=PADDED_Y_VALUE)

                
                self.out = torch.cat([self.out, out_.unsqueeze(0)], dim=0)
                self.target = torch.cat([self.target, target.unsqueeze(0).type(torch.float32)], dim=0)


            inter_df = get_doc_df(qid.cpu(), out.cpu().detach(), index_to_docno = index_to_docno)

            self.doc_test_df = pd.concat([self.doc_test_df, inter_df], axis=0, ignore_index=True)
            
        
        if self.loss_type == 'ranknet':
            loss = lambdaLoss(self.out, self.target, weighing_scheme="rankNet_scheme", reduction_log="natural", padded_value_indicator=PADDED_Y_VALUE)
            self.out = torch.tensor([], device = self.device)
            self.target = torch.tensor([], device = self.device)

        elif self.loss_type == 'lambdarank':
            loss = lambdaLoss(self.out, self.target, weighing_scheme="lambdaRank_scheme", reduction_log="natural", padded_value_indicator=PADDED_Y_VALUE)
            self.out = torch.tensor([], device = self.device)
            self.target = torch.tensor([], device = self.device)
        
        

        self.test_prop['loss'].append(loss)


        return loss
        
    def configure_optimizers(self):
    
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        
        return self.optimizer
    
