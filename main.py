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
from src.attention import AttnReranker
from src.edge_gat_reranker import EdgeGATReranker
from src.learned_graph import LearnedEdgeGATReranker
import copy
import time
import json


def str2bool(value):
    """Strict string->bool parser. `--flag False` must mean False.

    argparse's `type=bool` makes any non-empty string truthy (so `--score False`
    silently became True), which broke every ablation that needs a flag disabled.
    """
    if isinstance(value, bool):
        return value
    v = str(value).strip().lower()
    if v in ("true", "t", "1", "yes", "y"):
        return True
    if v in ("false", "f", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {value!r}")


parser = argparse.ArgumentParser()

# python main.py --dataset_name 'msmarco_data' --mode 'hp' --sweep True --n_seeds 1 --fast_train True --wb True --save_best_model True --count 2000 --length_train 1000 --length_val 200 --device 'cuda'
# python main.py --dataset_name 'msmarco_data' --mode 'hp' --n_seeds 1 --fast_train True --save_best_model True --length_train 2000 --length_val 200 --device 'cuda'
# Experiments Inputs
parser.add_argument("--dataset_name", type = str, default=dataset_name, help = 'Name of the dataset, e.g. msmarco_data')
parser.add_argument("--K_cg", type = int, default=K_cg, help = 'Number of maximum neighbors in the corpus graph.')
parser.add_argument("--mode", type = str, default = 'hp', help = "Validation Set to use during training. e.g. 'hp' for Hyperparameter Tuning")
parser.add_argument("--sweep", type = str2bool, default = False, help = 'Whether to run a sweep or not, in that case choose the hyperparameter space properly.')
parser.add_argument("--epochs", type = int, default = 200, help = 'Number of epochs to train the model.')
parser.add_argument("--seed", type = int, default = seed, help = 'Seed for reproducibility.')
parser.add_argument("--wb", type = str2bool, default = False, help = 'Whether to log data into wandb.')
parser.add_argument("--save_best_model", type = str2bool, default = False, help = 'Whether to save the best model after training.')
parser.add_argument("--resume_sweep", type = str, default = '', help = 'If a sweep have been interrupted, you can resume it by passing the sweep id.')
parser.add_argument("--count", type = int, default = 2000, help = 'Number of runs to perform in the sweep.')
parser.add_argument("--length_train", type = int, default = 10000000, help = 'Maximum number of training samples to use.')
parser.add_argument("--length_val", type = int, default = 10000000, help = 'Maximum number of validation samples to use.')
parser.add_argument("--loss_type", type = str, default = loss_type, help = 'Loss type to use during training. e.g. "mse", "ranknet", "lambdarank", "listnet", "listmle"')
parser.add_argument("--embedding_name", type = str, default = embedding_name, help = 'Source of the initial embedding. e.g. "tctcolbert"')
parser.add_argument("--fast_train", type = str2bool, default = False, help = 'Whether to use pre-compued data to speed up the datamodule. This is bounded to about 2000 qids.')
parser.add_argument("--n_seeds", type = int, default = 1, help = 'Select the number of seeds to use in the sweep.')
parser.add_argument("--patience", type = int, default = 5, help = 'Patience value to use for the early stopping.')


# Data Inputs
parser.add_argument("--batch_size", type = int, default = batch_size)

# Hyperparameters
parser.add_argument("--hidden_dim", type = int, default = hidden_dim, help = 'Hidden dimension of the GNN.')
parser.add_argument("--n_layers", type = int, default = n_layers, help = 'Number of layers of the GNN.')
parser.add_argument("--n_layers_mlp", type = int, default = n_layers_mlp, help = 'Number of layers of the MLP that concatenates GNN (local) and Individual representations.')
parser.add_argument("--score", type = str2bool, default = score, help = "Whether to use the GNN with the initial bm25 as features.")
parser.add_argument("--dropout_prob", type = float, default = dropout_prob, help = 'Dropout level associated to each layer.')
parser.add_argument("--lr", type = float, default = lr, help = 'Learning rate of the model.')
parser.add_argument("--wd", type = float, default = wd, help = 'Weight decay of the model.')
parser.add_argument("--aggr", type = str, default = aggr, help = 'Type of aggregation to use. e.g. "hadamard".')
parser.add_argument("--pooling_ratio", type = float, default = 0.5, help = 'Ratio of the pooling layer. This must be enabled in the global mode.')
parser.add_argument("--conv_type", type = str, default = conv_type, help = 'Type of convolution to use. e.g. "gat", "gcn".')
parser.add_argument("--modality", type = str, default = modality, help = 'Type of modality to use. e.g. "single", "local", "global".')
parser.add_argument("--heads", type = int, default = 1, help = 'Number of heads to use in the GAT convolution (>=1).')
parser.add_argument("--negatives", type = int, default = 0, help = 'Number of negative samples to use in the signed convolution.')
parser.add_argument("--pooling", type = str, default = 'hierarchical', help = 'Pooling type to use in the global mode.')
parser.add_argument("--structure_learning", type = str2bool, default = False, help = 'If enabling or not the structure learning mode of HGPSL.')
parser.add_argument("--lamb", type = float, default = 0, help = 'This is the trade-off factor between using or not structure learning.')
parser.add_argument("--aggr_sage", type = str, default = 'max', help = "GraphSAGE's aggregation modality.")
parser.add_argument("--K_multistage", type = int, default = 100, help = 'If modality == "multistage" this is the subgraph dimension to employ.')
parser.add_argument("--disable_gnn", type = str2bool, default = False, help = 'No-GNN control: zero the GNN branch (local modality) to isolate scorer capacity.')
parser.add_argument("--graph_type", type = str, default = 'semantic', help = "Corpus graph: 'semantic' (TCT kNN) or 'lexical' (BM25 kNN from pyterrier).")
parser.add_argument("--corpusgraph_name", type = str, default = 'corpusgraph_bm25_k16', help = "pyterrier dataset variant for the lexical corpus graph.")
parser.add_argument("--norm", type = str2bool, default = False, help = "edgegat: use BatchNorm (True) vs LayerNorm (False).")
parser.add_argument("--gumbel_temp", type = float, default = 1.0, help = "Binary Concrete temperature for learned_edgegat edge selector (lower = sharper).")
parser.add_argument("--sparsity_reg", type = float, default = 0.0, help = "λ_s: weight of the edge sparsity regulariser (learned_edgegat only).")
parser.add_argument("--model_path", type = str, default = "", help = "Optional checkpoint to warm-start from (loaded with strict=False; use to init learned_edgegat from a trained edgegat backbone).")
parser.add_argument("--feat_recon_reg", type = float, default = 0.0, help = "λ_dae: SLAPS-style auxiliary reconstruction loss weight (learned_edgegat only). Provides dense gradient to all edges, fixing supervision starvation from sparse ranking labels.")
parser.add_argument("--mask_ratio", type = float, default = 0.15, help = "Fraction of input feature dims to mask for reconstruction auxiliary task (learned_edgegat only).")


# System Settings
parser.add_argument("--device", type = str, default = device, help = 'Device to use. e.g. "cuda", "cpu".')


set_determinism_the_old_way(deterministic = True)


args = parser.parse_args()

dataset_name = args.dataset_name
mode = args.mode
batch_size = args.batch_size
hidden_dim = args.hidden_dim
n_layers = args.n_layers
n_layers_mlp = args.n_layers_mlp
dropout_prob = args.dropout_prob
device = args.device
sweep = args.sweep
lr = args.lr
wd = args.wd
aggr = args.aggr
seed = args.seed
resume_sweep = args.resume_sweep
wb = args.wb
count = args.count
conv_type = args.conv_type
heads = max(1, args.heads)  # GAT needs >=1; old scripts pass --heads 0 (was truthy==1)
args.heads = heads
pooling_ratio = args.pooling_ratio
loss_type = args.loss_type
modality = args.modality
K_cg = args.K_cg
length_train = args.length_train
length_val = args.length_val
save_best_model = args.save_best_model
score = args.score
epochs = args.epochs
fast_train = args.fast_train
n_seeds = args.n_seeds
negatives = args.negatives
pooling = args.pooling
structure_learning = args.structure_learning
lamb = args.lamb
patience = args.patience
aggr_sage = args.aggr_sage
K_multistage = args.K_multistage
graph_type = args.graph_type
corpusgraph_name = args.corpusgraph_name
embedding_name = args.embedding_name   # override config default ('tctcolbert2')




if sweep: # If we are running a sweep, we need to set the hyperparameters space
    parameters_dict = parameters_dict_local

if dataset_name == 'msmarco_data':

    train_path = "irds:msmarco-passage/train/split200-train"

    val_path = "irds:msmarco-passage/train/split200-valid"  # TODO In fast_train = False modality, it works only if we limit the number of length_val == 200.

    train_set = pt.get_dataset(train_path)
    val_set = pt.get_dataset(val_path)

    train_indices = list(train_set.get_topics().qid)

    # When fast_train=True the DataLoader iterates over ALL train_indices but only
    # those with precomputed tensors produce real training samples — the rest hit
    # FileNotFoundError and return IGNORE_INDEX, wasting ~6000 batches/epoch.
    # Filter upfront (before applying length_train) so --length_train N gives
    # exactly N queries from the precomputed fast tensor set.
    if fast_train:
        import os as _os
        _fast_dir = f'data/{dataset_name}/train_data_fast/tensors/'
        if _os.path.isdir(_fast_dir):
            _avail = set(
                d.replace('qid_', '').replace('_tensors', '')
                for d in _os.listdir(_fast_dir)
                if d.startswith('qid_')
            )
            train_indices = [q for q in train_indices if str(q) in _avail]

    train_indices = train_indices[:length_train]

    val_list_indices = list(val_set.get_topics().qid) + list(train_set.get_topics().qid)[15500:16000]
    val_indices = val_list_indices[:length_val]


    print(f"Length of the train set is: {len(train_indices)}")

    print(f"Length of the val set is: {len(val_indices)}")

    qrels_path = 'data/msmarco_data/msmarco_data_qrels/qrels_val.txt'
        
  
if not sweep:

    best_metric = 0
    best_seed = seed_list[0]
    ndcg_list = []
    for seed_n in range(n_seeds): #len(seed_list)//2):

        print(f"Seed: {seed_n}")
        set_seed(seed_list[seed_n])
    
        if dataset_name == 'msmarco_data':
            
            pl_dataset = DataModule_terrier(train_path, val_path, batch_size, K_cg = K_cg, fast_train = fast_train, fast = True, mode = mode, embedding_name = embedding_name, train_indices = train_indices, val_indices = val_indices, graph_type = graph_type, corpusgraph_name = corpusgraph_name)
            
        if conv_type == 'transformer':
            model = AttnReranker(n_feats if aggr != 'concat' else 2*n_feats, args, device = device, multistage = (modality == 'multistage'))
        elif conv_type == 'edgegat':
            model = EdgeGATReranker(n_feats, args, device = device)
        elif conv_type == 'learned_edgegat':
            model = LearnedEdgeGATReranker(n_feats, args, device = device)
            if args.model_path:
                ckpt = torch.load(args.model_path, map_location=device)
                if any(k.startswith("model.") for k in ckpt):
                    ckpt = {k[len("model."):]: v for k, v in ckpt.items() if k.startswith("model.")}
                # strict=False only skips missing/unexpected keys — size mismatches
                # still raise RuntimeError. Filter to matching-shape keys only.
                model_sd = model.state_dict()
                ckpt_compat = {k: v for k, v in ckpt.items()
                               if k in model_sd and v.shape == model_sd[k].shape}
                missing, unexpected = model.load_state_dict(ckpt_compat, strict=False)
                print(f"[warm-start] loaded {len(ckpt_compat)}/{len(ckpt)} shape-matched keys from {args.model_path}")
                if len(ckpt_compat) == 0:
                    print("[warm-start] WARNING: no keys matched — checkpoint architecture differs too much; training from scratch.")
        elif modality == 'single':
            if conv_type != 'mlp':
                model = GNN_NR(n_feats if aggr != 'concat' else 2*n_feats, args, device = device)

            elif conv_type == 'mlp':
                model = MLP(n_feats if aggr != 'concat' else 2*n_feats, hidden_dim, output_dim = 1, n_layers = n_layers, device = device, dropout_prob=dropout_prob)
        else:
            model = GNN_LG(n_feats if aggr != 'concat' else 2*n_feats, args, modality = modality, conv_type=conv_type, device = device)
    
        early_stop = EarlyStopping(monitor='nDCG@10 on test', patience=patience, mode="max")
        compute_metrics = Get_Metrics()

        if device == 'cuda':
            num_gpus = 1
        else:
            num_gpus = 0

        prefix = "./logs/"

        if modality != 'global':
            exp_name = f"{aggr}_{n_layers}_{seed}_{lr}_{wd}_{hidden_dim}_{dropout_prob}_{conv_type}_{modality}_{n_layers_mlp}_{embedding_name}"

        else:
            exp_name = f"{conv_type}_{pooling}_{structure_learning}_{n_layers_mlp}_{seed}"

        if modality == 'multistage':
            exp_name += f"_{K_multistage}"

        if conv_type == 'signed':
            exp_name += f"_{negatives}"

        if conv_type == 'gat':
            exp_name += f"_{heads}"

        # disambiguate lexical vs semantic corpus-graph checkpoints
        if graph_type != 'semantic':
            exp_name += f"_{graph_type}"

        tot_dir = prefix + exp_name + '/'

        checkpoint_callback = ModelCheckpoint(dirpath = tot_dir, save_top_k=1, monitor="nDCG@10 on test", mode="max")

        # Save model weights to models/ immediately on each validation improvement,
        # so the file exists even if training is interrupted before completing all seeds.
        class _LiveSave(Callback):
            def __init__(self, path, save_flag, best_ref):
                self._path = path
                self._save = save_flag
                self._best_ref = best_ref  # mutable list [best_metric]
                self._seed_best = 0.0
            def on_validation_epoch_end(self, trainer, pl_module):
                if not self._save:
                    return
                val = trainer.callback_metrics.get('nDCG@10 on test', 0.0)
                val = val.item() if hasattr(val, 'item') else float(val)
                if val > self._seed_best:
                    self._seed_best = val
                    if val > self._best_ref[0]:
                        self._best_ref[0] = val
                        import os as _os
                        _os.makedirs(_os.path.dirname(self._path), exist_ok=True)
                        torch.save(pl_module.model.state_dict(), self._path)
                        print(f'\n[checkpoint] model saved (nDCG@10={val:.4f}) → {self._path}')

        _model_path = f'./models/{dataset_name}/{exp_name}.pt'
        _best_ref = [best_metric]   # shared mutable so later seeds can compare
        live_save = _LiveSave(_model_path, save_best_model, _best_ref)

        wandb_logger = WandbLogger(project=project_name, name=exp_name) if wb else None

        print("NUM_GPUS: ", num_gpus)
        trainer = pl.Trainer(
            max_epochs=epochs,  # maximum number of epochs.
            gpus=num_gpus,  # the number of gpus we have at our disposal.
            default_root_dir=tot_dir, callbacks=[compute_metrics, early_stop, checkpoint_callback, live_save], deterministic = True if device == 'cpu' else False, logger = wandb_logger
        )

        pl_training_module = TrainingModule(model, lr, wd, aggr, model_family = conv_type, dataset_name=dataset_name, qrels_folder = qrels_path, fast_train = fast_train, K_cg = K_cg, eval_indices = val_indices, loss_type = loss_type, exp_name = exp_name, sparsity_reg = args.sparsity_reg, feat_recon_reg = args.feat_recon_reg)
        trainer.fit(model=pl_training_module, datamodule=pl_dataset)

        print("Best model score is:\n", checkpoint_callback.best_model_score.item())

        test_model = TrainingModule.load_from_checkpoint(checkpoint_callback.best_model_path, model = model, lr = lr, wd = wd, qrels_folder = qrels_path, fast_train = fast_train, aggr = aggr, model_family = conv_type, dataset_name=dataset_name, K_cg = K_cg, eval_indices = val_indices, loss_type = loss_type)
            
        trainer.test(test_model, dataloaders=pl_dataset.val_dataloader())      
        
        diz_test = get_performance_metrics(qrels_path, val_indices, test_model.doc_test_df)

        print(diz_test)

        # Use the score already tracked by ModelCheckpoint (nDCG@10 on val) instead of
        # re-indexing diz_test with an ir_measures key, which can silently miss.
        current_score = checkpoint_callback.best_model_score.item() \
            if checkpoint_callback.best_model_score is not None else 0.0
        if save_best_model and current_score > best_metric:
            best_metric = current_score
            best_seed = seed_list[seed_n]
            torch.save(test_model.model.state_dict(), f'./models/{dataset_name}/{exp_name}.pt')
            print("Model saved!")
        try:
            ndcg_val = diz_test[nDCG@10]
        except Exception:
            ndcg_val = current_score
        ndcg_list.append(ndcg_val)
        if wb:
            wandb.log({f"nDCG@{str(10)} on test ({seed_list[seed_n]})": ndcg_val})
    
    ndcg_array = np.array(ndcg_list)

    mean, std = np.mean(ndcg_array), np.std(ndcg_array)
    
    if wb:
        wandb.log({f"nDCG@{str(10)} mean": mean, f"nDCG@{str(10)} std": std})


def compute_runs(config):
    
    metrics = {f'nDCG@{str(10)}': []}

    # Constraints

    if config.n_layers_mlp > 1 and config.modality == 'single':
        return {f'nDCG@{str(10)}': [0]}
    
    try:
        if config.conv_type != 'signed' and config.negatives == 1.5:
            return {f'nDCG@{str(10)}': [0]}
        
        if config.pooling != 'hierarchical' and (config.pooling_ratio != 0.7 or config.lamb != 0.2 or config.structure_learning != False):
            return {f'nDCG@{str(10)}': [0]}
    except AttributeError:
        pass

    
    
    
    best_metric = 0
    best_seed = seed_list[0]

    for seed_n in range(n_seeds): #len(seed_list)//2):

        print(f"Seed: {seed_n}")
        set_seed(seed_list[seed_n])
        if dataset_name == 'msmarco_data':
        
            pl_dataset = DataModule_terrier(train_path, val_path, batch_size, K_cg = config.K_cg, fast_train = fast_train, fast = True, mode = mode, embedding_name = embedding_name, train_indices = train_indices, val_indices = val_indices, graph_type = graph_type, corpusgraph_name = corpusgraph_name)
            
        if config.conv_type == 'transformer':
            model = AttnReranker(n_feats if config.aggr != 'concat' else 2*n_feats, config, device = device, multistage = (config.modality == 'multistage'))
        elif config.conv_type == 'edgegat':
            model = EdgeGATReranker(n_feats, config, device = device)
        elif config.conv_type == 'learned_edgegat':
            model = LearnedEdgeGATReranker(n_feats, config, device = device)
        elif modality == 'single':
            if conv_type != 'mlp':
                model = GNN_NR(n_feats if config.aggr != 'concat' else 2*n_feats, config, device = device)

            elif conv_type == 'mlp':
                model = MLP(n_feats if config.aggr != 'concat' else 2*n_feats, config.hidden_dim, output_dim = 1, n_layers = config.n_layers, device = device, dropout_prob=config.dropout_prob)
        else:
            model = GNN_LG(n_feats if config.aggr != 'concat' else 2*n_feats, config, modality = config.modality, conv_type=config.conv_type, device = device)


        prefix = f"Sweeps/Sweeps_{config.dataset_name}/"
        
        exp_name = f"{config.conv_type}_{config.aggr}_{config.n_layers}_{config.n_layers_mlp}_{seed_n}_{config.lr}_{config.wd}_{config.hidden_dim}_{config.dropout_prob}_{config.modality}_{config.K_cg}_{config.score}_{config.embedding_name}"

        if config.conv_type == 'signed':
            exp_name += f"_{config.negatives}"
        
        if config.conv_type == 'gat':
            exp_name += f"_{config.heads}"

        if config.conv_type == 'sage':
            exp_name += f"_{config.aggr_sage}"


        tot_dir = prefix + exp_name + '/'

        shutil.rmtree(prefix, ignore_errors=True)

        os.makedirs(prefix, exist_ok=True)

        checkpoint_callback = ModelCheckpoint(dirpath = tot_dir, save_top_k=1, monitor="nDCG@10 on test", mode="max")
        early_stop = EarlyStopping(monitor='nDCG@10 on test', patience= patience, mode="max")
        compute_metrics = Get_Metrics()
        if device == 'cuda':
            num_gpus = 1
        else: 
            num_gpus = 0
                # wandb_logger = WandbLogger(project=project_name, name=exp_name, config=hyperparameters)
        # print("NUM_GPUS: ", num_gpus)

        wandb_logger = WandbLogger(project=project_name, name=exp_name) if wb else None
        
        trainer = pl.Trainer(
            max_epochs=epochs if aggr != 'tctcolbert' else 1,  # maximum number of epochs.
            gpus=num_gpus,  # the number of gpus we have at our disposal.
            default_root_dir= tot_dir, callbacks=[compute_metrics, early_stop, checkpoint_callback],
            enable_checkpointing=True, deterministic = True if device == 'cpu' else False, logger = wandb_logger
        )
        
        pl_training_module = TrainingModule(model, lr = config.lr, wd = config.wd, aggr = config.aggr, model_family = config.conv_type, dataset_name=dataset_name, fast_train = fast_train, qrels_folder = qrels_path, K_cg = config.K_cg, eval_indices = val_indices, loss_type = config.loss_type, exp_name = exp_name, sparsity_reg = getattr(config, 'sparsity_reg', 0.0))
        trainer.fit(model=pl_training_module, datamodule=pl_dataset)

        print("Best model path is:", checkpoint_callback.best_model_path)
        # and prints it score
        print("Best model score is:\n", checkpoint_callback.best_model_score)

        test_model = TrainingModule.load_from_checkpoint(checkpoint_callback.best_model_path, model = model, fast_train = fast_train, lr = config.lr, wd = config.wd, aggr = config.aggr, model_family = config.conv_type, dataset_name=dataset_name, qrels_folder = qrels_path, K_cg = config.K_cg, eval_indices = val_indices, loss_type = config.loss_type, exp_name = exp_name)


        ##### Test on Test Set #####
        trainer.test(test_model, dataloaders=pl_dataset.val_dataloader())
        
        diz_test = get_performance_metrics(qrels_path, val_indices, test_model.doc_test_df)


        ndcg = diz_test[nDCG@10]

        print("The score is: ", ndcg)


        ndcg = diz_test[nDCG@10]
        metrics[f'nDCG@{str(10)}'].append(ndcg)
        
        if diz_test[nDCG@10] > best_metric and save_best_model == True:
            best_metric = diz_test[nDCG@10]
            best_seed = seed_list[seed_n]
            torch.save(test_model.model.state_dict(), f'./models/{dataset_name}/{exp_name}.pt')
            print("Model saved!")

        # Log the nDCG for each seed 
        wandb.log({f"nDCG@{str(10)} on test ({seed_list[seed_n]})": ndcg})

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
    
    wandb.log({"Best Seed": best_seed})

    return metrics



def sweep_train(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config, resume = True if resume_sweep != '' else False):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config

        metrics = compute_runs(config = config)
 
        metric_current = metrics[f'nDCG@{str(10)}']

        metrics_array = np.array(metric_current)
        mean, std = np.mean(metrics_array), np.std(metrics_array)
        print("MEAN AND STANDARD DEVIATION: {} +- {}".format(mean, std))
        wandb.log({f"nDCG@{str(10)} on test (Mean)": mean, f"nDCG@{str(10)} on test (Std.)": std})


if sweep:

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


    