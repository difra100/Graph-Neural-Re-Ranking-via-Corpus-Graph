import torch
import numpy as np
import random
import argparse
from torch_geometric.nn import SAGEConv, GCNConv, GATConv

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import gc
import shutil
import wandb
import pprint

from src.config import *
from src.datamodule import *
from src.lightningmodule import * 
from src.GNN import *

import json


parser = argparse.ArgumentParser()

# Training Inputs
parser.add_argument("--dataset_name", type = str, default=dataset_name)
parser.add_argument("--mode", type = str, default = mode)
parser.add_argument("--eval", type = bool, default = False)
parser.add_argument("--sweep", type = bool, default = False)
parser.add_argument("--epochs", type = int, default = 200)
parser.add_argument("--seed", type = int, default = seed)
parser.add_argument("--wb", type = bool, default = False)
parser.add_argument("--resume_sweep", type = str, default = '')
parser.add_argument("--count", type = int, default = 100)
parser.add_argument("--loss_type", type = str, default = loss_type)




# Data Inputs
parser.add_argument("--batch_size", type = int, default = batch_size)


# Hyperparameters
parser.add_argument("--hidden_dim", type = int, default = hidden_dim)
parser.add_argument("--n_layers", type = int, default = n_layers)
parser.add_argument("--dropout_prob", type = float, default = dropout_prob)
parser.add_argument("--lr", type = float, default = lr)
parser.add_argument("--wd", type = float, default = wd)
parser.add_argument("--aggr", type = str, default = aggr)
parser.add_argument("--conv_type", type = str, default = conv_type)
parser.add_argument("--modality", type = str, default = modality)

parser.add_argument("--neighbors", type = bool, default = neighbors)
parser.add_argument("--heads", type = bool, default = heads)









# System Settings
parser.add_argument("--device", type = str, default = device)





set_determinism_the_old_way(deterministic = True)


args = parser.parse_args()

dataset_name = args.dataset_name
mode = args.mode
batch_size = args.batch_size

hidden_dim = args.hidden_dim
n_layers = args.n_layers
dropout_prob = args.dropout_prob

device = args.device
eval = args.eval
sweep = args.sweep
lr = args.lr
wd = args.wd
aggr = args.aggr
seed = args.seed
resume_sweep = args.resume_sweep
wb = args.wb
count = args.count
neighbors = args.neighbors
conv_type = args.conv_type
heads = args.heads
loss_type = args.loss_type
modality = args.modality


# if modality == 'single':
#     if conv_type == 'gcn':
#         parameters_dict = parameters_dict_gcn
#     elif conv_type == 'gat': 
#         parameters_dict = parameters_dict_gat
#     elif conv_type == 'mlp':
#         parameters_dict = parameters_dict_MLP
# elif modality == 'local':
parameters_dict = parameters_dict_local



    


if dataset_name == 'trial':

    dataset = sample_data = [{'doc_feat': torch.randn((n_docs, n_feats)).to(device),
                    'adj_matrix': torch.zeros((2, random.randint(10, 300)), device = device).to(torch.int64),
                    'q_rels': torch.randint(low=-1, high=4, size=(n_docs,), dtype=torch.float),
                    'q_ids': i,
                    'query_feat': torch.randn((1, n_feats)).to(device)} for i in range(30)
                ]

else:
    dataset = retrieve_dataset_from_file(dataset_name=dataset_name)




dataset_length = len(dataset)

train_indices = json.load(open(f'./data/{dataset_name}/train_indices.json'))

val_indices = json.load(open(f'./data/{dataset_name}/val_indices.json'))

test_indices = json.load(open(f'./data/{dataset_name}/test_indices.json'))

if not sweep and not eval:

    set_seed(seed)
    

    train_data = [dataset[x] for x in range(dataset_length) for i in train_indices if i == dataset[x]['q_ids']]

    val_data = [dataset[x] for x in range(dataset_length) for i in val_indices if i == dataset[x]['q_ids']]

    test_data = [dataset[x] for x in range(dataset_length) for i in test_indices if i == dataset[x]['q_ids']]




    pl_dataset = DataModule(train_data, val_data, test_data, mode, batch_size)

    if modality == 'local' and conv_type == 'mlp':
        print("No possible")
        exit()
    
    if modality == 'single':
        if conv_type == 'gcn' or conv_type == 'gat':
            model = GNN_NR(n_feats if aggr != 'concat' else 2*n_feats, args, device = device)
        
        elif conv_type == 'mlp':
            model = MLP(n_feats if aggr != 'concat' else 2*n_feats, hidden_dim, output_dim = 1, device = device, dropout_prob=dropout_prob)
    else:
        model = GNN_LG(n_feats if aggr != 'concat' else 2*n_feats, args, modality = modality, conv_type=conv_type, device = device)
    

    prefix = "Sweeps/"
    exp_name = f"{aggr}_{n_layers}_{seed}_{lr}_{wd}_{hidden_dim}_{dropout_prob}"

    tot_dir = prefix + exp_name + '/'

    shutil.rmtree(prefix, ignore_errors=True)
    os.makedirs(prefix, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(dirpath = tot_dir, save_top_k=1, monitor="nDCG@10 on test", mode="max")
    early_stop = EarlyStopping(monitor='nDCG@10 on test', patience=10, mode="max")
    compute_metrics = Get_Metrics()
    if device == 'cuda':
        num_gpus = 1
    else: 
        num_gpus = 0
            # wandb_logger = WandbLogger(project=project_name, name=exp_name, config=hyperparameters)
    print("NUM_GPUS: ", num_gpus)
    trainer = pl.Trainer(
        max_epochs=epochs if aggr != 'tctcolbert' else 1,  # maximum number of epochs.
        gpus=num_gpus,  # the number of gpus we have at our disposal.
        default_root_dir=tot_dir, callbacks=[compute_metrics, early_stop, checkpoint_callback],
    )

    pl_training_module = TrainingModule(model, lr, wd, aggr, model_family = conv_type, loss_type = loss_type,  ndcgk = ndcgk, recall = recall, precision = precision)
    trainer.fit(model=pl_training_module, datamodule=pl_dataset)

    print("Best model score is:\n", checkpoint_callback.best_model_score.item())


def compute_runs(config):
    
    print(f"Hyperparameters tuning initialized....{config.conv_type}.type..............")
    
    train_data = [dataset[x] for x in range(dataset_length) for i in train_indices if i == dataset[x]['q_ids']]

    val_data = [dataset[x] for x in range(dataset_length) for i in val_indices if i == dataset[x]['q_ids']]
    
    fold_split = train_data + val_data
    folds = k_fold_cross_validation(fold_split, k = 5, random_seed=42)
    metrics = {f'nDCG@{str(i)}': [] for i in ndcgk}
    if config.modality == 'local' and config.conv_type == 'mlp':
        print("No possible")
        return {f'nDCG@{str(i)}': [0] for i in ndcgk}
            
    for fold_n in range(len(folds)):
        train_data, val_data = folds[fold_n][0], folds[fold_n][1]
        test_data = None
        for seed_n in range(len(seed_list)//2):
            print(f"Fold n° {fold_n} and seed: {seed_n}")
            set_seed(seed_list[seed_n])

            pl_dataset = DataModule(train_data, val_data, test_data, mode, batch_size)
            pl_dataset.prepare_data()
            pl_dataset.setup()

            prefix = f"Sweeps_{config.dataset_name}_{config.conv_type}/"
            
            if config.modality == 'single':
                if config.conv_type == 'gcn' or config.conv_type == 'gat':
                    model = GNN_NR(n_feats if config.aggr != 'concat' else 2*n_feats, config, device = device)
                    exp_name = f"{config.conv_type}_{config.aggr}_{config.n_layers}_{seed_n}_{config.lr}_{config.wd}_{config.hidden_dim}_{config.dropout_prob}_{config.neighbors}"
                    
                elif config.conv_type == 'mlp':
                    model = MLP(n_feats if config.aggr != 'concat' else 2*n_feats, config.hidden_dim, output_dim = 1, device = device, dropout_prob=config.dropout_prob)
                    exp_name = f"{config.conv_type}_{config.aggr}_{config.n_layers}_{seed_n}_{config.lr}_{config.wd}_{config.hidden_dim}_{config.dropout_prob}"
            else:
                model = GNN_LG(n_feats if config.aggr != 'concat' else 2*n_feats, config, modality = config.modality, conv_type=config.conv_type, device = device)
                exp_name = f"{config.modality}_{config.conv_type}_{config.aggr}_{config.n_layers}_{seed_n}_{config.lr}_{config.wd}_{config.hidden_dim}_{config.dropout_prob}"
            
            tot_dir = prefix + exp_name + '/'

            shutil.rmtree(prefix, ignore_errors=True)
            os.makedirs(prefix, exist_ok=True)

            checkpoint_callback = ModelCheckpoint(dirpath = tot_dir, save_top_k=1, monitor="nDCG@10 on test", mode="max")
            early_stop = EarlyStopping(monitor='nDCG@10 on test', patience=10, mode="max")
            compute_metrics = Get_Metrics()
            if device == 'cuda':
                num_gpus = 1
            else: 
                num_gpus = 0
                    # wandb_logger = WandbLogger(project=project_name, name=exp_name, config=hyperparameters)
            # print("NUM_GPUS: ", num_gpus)
            trainer = pl.Trainer(
                max_epochs=epochs if aggr != 'tctcolbert' else 1,  # maximum number of epochs.
                gpus=num_gpus,  # the number of gpus we have at our disposal.
                default_root_dir= tot_dir, callbacks=[compute_metrics, early_stop, checkpoint_callback],
            # logger=wandb_logger,
                enable_checkpointing=True
            )

            pl_training_module = TrainingModule(model, config.lr, config.wd, config.aggr, model_family = config.conv_type, loss_type = config.loss_type, ndcgk = ndcgk, recall = recall, precision = precision)
            trainer.fit(model=pl_training_module, datamodule=pl_dataset)

            print("Best model path is:", checkpoint_callback.best_model_path)
            # and prints it score
            print("Best model score is:\n", checkpoint_callback.best_model_score)
           
            # if model_family == 'gnn':
            #     model = GNN_NR(n_feats if config.aggr != 'concat' else 2*n_feats, config, device = device)
    
            # elif model_family == 'mlp':
            #     model = MLP(n_feats if config.aggr != 'concat' else 2*n_feats, config.hidden_dim, device = device, dropout_prob=config.dropout_prob)

   

            test_model = TrainingModule.load_from_checkpoint(checkpoint_callback.best_model_path, model = model, lr = config.lr, wd = config.wd, aggr = config.aggr, model_family = config.conv_type, loss_type = config.loss_type, ndcgk = ndcgk, recall = recall, precision = precision)

       

            trainer.test(test_model, dataloaders=pl_dataset.val_dataloader())

           
            ndcg = sum(test_model.test_prop['nDCG@'+str(10)])/len(test_model.test_prop['nDCG@'+str(10)])

            print("The score is: ", ndcg)


            for metric in ndcgk:
        
                ndcg = sum(test_model.test_prop['nDCG@'+str(metric)])/len(test_model.test_prop['nDCG@'+str(metric)])

                metrics[f'nDCG@{str(metric)}'].append(ndcg.detach().cpu().item())
            
           

            del model
        
            del pl_training_module
            del trainer
            del test_model
            del checkpoint_callback
            del early_stop
            del compute_metrics

        

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return metrics

def compute_runs_for_eval():
    print("Test initialized..........................")
    metrics = {f'nDCG@{str(i)}': [] for i in ndcgk}

    for metric in recall:
        metrics[f'recall@{str(metric)}'] = []
    
    for metric in precision:    
        metrics[f'precision@{str(metric)}'] = []

    print("METRICS: ", metrics)
    
    train_data = [dataset[x] for x in range(dataset_length) for i in train_indices if i == dataset[x]['q_ids']]

    val_data = [dataset[x] for x in range(dataset_length) for i in val_indices if i == dataset[x]['q_ids']]

    test_data = [dataset[x] for x in range(dataset_length) for i in test_indices if i == dataset[x]['q_ids']]


    best_seed = 0
    best_ndcg = 0

    if modality == 'local' and conv_type == 'mlp':
        print("No possible")
        return {f'nDCG@{str(i)}': [0] for i in ndcgk}, 0, 0
    
    for seed_n in range(len(seed_list)):
        set_seed(seed_list[seed_n])
        
        pl_dataset = DataModule(train_data, val_data, test_data, mode, batch_size)
        pl_dataset.prepare_data()
        pl_dataset.setup()

        if modality == 'single':
            if conv_type == 'gcn' or conv_type == 'gat':
                model = GNN_NR(n_feats if aggr != 'concat' else 2*n_feats, args, device = device)

            elif conv_type == 'mlp':
                model = MLP(n_feats if aggr != 'concat' else 2*n_feats, hidden_dim, output_dim = 1, device = device, dropout_prob=dropout_prob)
        else:
            model = GNN_LG(n_feats if aggr != 'concat' else 2*n_feats, args, modality = modality, conv_type=conv_type, device = device)


        prefix = f"Sweeps_{loss_type}_{modality}_{conv_type}/"
        exp_name = f"{modality}_{conv_type}_{aggr}_{n_layers}_{seed_n}_{lr}_{wd}_{hidden_dim}_{dropout_prob}_{loss_type}"

        tot_dir = prefix + exp_name + '/'

        shutil.rmtree(prefix, ignore_errors=True)
        os.makedirs(prefix, exist_ok=True)

        checkpoint_callback = ModelCheckpoint(dirpath = tot_dir, save_top_k=1, monitor="nDCG@10 on test", mode="max")
        early_stop = EarlyStopping(monitor='nDCG@10 on test', patience=10, mode="max")
        compute_metrics = Get_Metrics()
        if device == 'cuda':
            num_gpus = 1
        else: 
            num_gpus = 0
                # wandb_logger = WandbLogger(project=project_name, name=exp_name, config=hyperparameters)
        # print("NUM_GPUS: ", num_gpus)

        trainer = pl.Trainer(
            max_epochs=epochs if aggr != 'tctcolbert' else 1,  # maximum number of epochs.
            gpus=num_gpus,  # the number of gpus we have at our disposal.
            default_root_dir= tot_dir, callbacks=[compute_metrics, early_stop, checkpoint_callback],
        # logger=wandb_logger,
            enable_checkpointing=True
        )

        pl_training_module = TrainingModule(model, lr, wd, aggr, model_family = conv_type, loss_type = loss_type, ndcgk = ndcgk, recall = recall, precision = precision)
        trainer.fit(model=pl_training_module, datamodule=pl_dataset)

        print("Best model path is:", checkpoint_callback.best_model_path)
            # and prints it score
        print("Best model score is:\n", checkpoint_callback.best_model_score)
        
        # if model_family == 'gnn':
        #     model = GNN_NR(n_feats if aggr != 'concat' else 2*n_feats, args, device = device)

        # elif model_family == 'mlp':
        #     model = MLP(n_feats if aggr != 'concat' else 2*n_feats, hidden_dim, device = device, dropout_prob=dropout_prob)



        test_model = TrainingModule.load_from_checkpoint(checkpoint_callback.best_model_path, model = model, lr = lr, wd = wd, aggr = aggr, model_family = conv_type, loss_type = loss_type, ndcgk = ndcgk, recall = recall, precision = precision)

        trainer.test(test_model, dataloaders=pl_dataset.val_dataloader())


        ndcg = sum(test_model.test_prop['nDCG@'+str(10)])/len(test_model.test_prop['nDCG@'+str(10)])

        print("The score is: ", ndcg)

        assert ndcg == checkpoint_callback.best_model_score

        best_ndcg = max(ndcg, best_ndcg)

        if ndcg == best_ndcg:
            best_seed = seed_n
            
        for metric in ndcgk:
    
            ndcg = sum(test_model.test_prop['nDCG@'+str(metric)])/len(test_model.test_prop['nDCG@'+str(metric)])

            metrics[f'nDCG@{str(metric)}'].append(ndcg.detach().cpu().item())
        for metric in recall:
            rec = sum(test_model.test_prop['recall@'+str(metric)])/len(test_model.test_prop['recall@'+str(metric)])
            metrics[f'recall@{str(metric)}'].append(rec.detach().cpu().item())

        for metric in precision:
            prec = sum(test_model.test_prop['precision@'+str(metric)])/len(test_model.test_prop['precision@'+str(metric)])
            metrics[f'precision@{str(metric)}'].append(prec.detach().cpu().item())

        del model
        del pl_training_module
        del trainer
        del test_model
        del checkpoint_callback
        del early_stop
        del compute_metrics

    

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

   

    return metrics, best_seed, best_ndcg
    


def sweep_train(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config, resume = True if resume_sweep != '' else False):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config

        metrics = compute_runs(config = config)
 
        for metric in ndcgk:

            metric_current = metrics[f'nDCG@{str(metric)}']
            metrics_array = np.array(metric_current)
            mean, std = np.mean(metrics_array), np.std(metrics_array)
            print("MEAN AND STANDARD DEVIATION: {} +- {}".format(mean, std))
            wandb.log({f"nDCG@{str(metric)} on test (Mean)": mean, f"nDCG@{str(metric)} on test (Std.)": std})



if mode == 'hp' and sweep:

    sweep_config['parameters'] = parameters_dict
    sweep_config['parameters']['dataset_name'] =  {
        'values': [dataset_name]
    }

    if resume_sweep != '':
        sweep_id = resume_sweep
        print("RESUMED PAST SWEEP....")
    else:

        sweep_id = wandb.sweep(sweep_config, project=project_name, entity=entity_name)
    pprint.pprint(sweep_config)

    
    wandb.agent(sweep_id, sweep_train, count = count, project = project_name, entity= entity_name)

    wandb.finish()

elif eval and not sweep:
    print("Testing modalities")
    metrics, best_seed, best_ndcg = compute_runs_for_eval()
    
    print("nDCG metrics: \n")
    for metric in ndcgk:

        metric_current = metrics[f'nDCG@{str(metric)}']
        metrics_array = np.array(metric_current)
        mean, std = np.mean(metrics_array), np.std(metrics_array)
        print("MEAN AND STANDARD DEVIATION: {} +- {}".format(mean, std))
        print("Best Seed and best nDCG@{} are: {} ; {}".format(str(metric), best_seed, best_ndcg))
    
    print("RECALL metrics: \n")
    for metric in recall:
        metric_current = metrics[f'recall@{str(metric)}']
        metrics_array = np.array(metric_current)
        mean, std = np.mean(metrics_array), np.std(metrics_array)
        print("MEAN AND STANDARD DEVIATION: {} +- {}".format(mean, std))
    
    print("PRECISION metrics: \n")
    for metric in precision:
        metric_current = metrics[f'precision@{str(metric)}']
        metrics_array = np.array(metric_current)
        mean, std = np.mean(metrics_array), np.std(metrics_array)
        print("MEAN AND STANDARD DEVIATION: {} +- {}".format(mean, std))

    


