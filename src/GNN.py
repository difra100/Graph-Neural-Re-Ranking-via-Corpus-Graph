import json
from torch_geometric.nn import SAGEConv, GCNConv, GATConv, GIN, GraphNorm, GATv2Conv, LayerNorm, InstanceNorm, SignedConv
from torch_geometric.utils import negative_sampling, subgraph
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch
from collections import namedtuple
from sklearn.manifold import TSNE
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import copy
from src.GNN_pool import *

activation = nn.LeakyReLU

class Config:
    def __init__(self, conv_type, hidden_dim, dropout_prob, n_layers):
   
        self.conv_type = conv_type
        self.hidden_dim = hidden_dim
        self.dropout_prob = dropout_prob
        self.n_layers = n_layers


class GNN_NR(nn.Module):
    def __init__(self, input_features, config, output_dim = 1, device='cpu', enc_mode = False):
        super().__init__()

        self.conf = config
        self.enc_mode = enc_mode
        self.dev = device

        if self.conf.score:
            input_features += 1

        self.input_features = input_features

        if self.conf.conv_type == 'gcn':
            layers = []
            layers.append(GCNConv(input_features, self.conf.hidden_dim).to(device))

            for _ in range(self.conf.n_layers - 1):
                layers.append(GCNConv(self.conf.hidden_dim, self.conf.hidden_dim).to(device))
        
        
        if self.conf.conv_type == 'gin':
            layers = [GIN(input_features, self.conf.hidden_dim, self.conf.n_layers, act='leakyrelu', dropout=self.conf.dropout_prob).to(device)]
        
        if self.conf.conv_type == 'gat':
            layers = []
            layers.append(GATConv(input_features, self.conf.hidden_dim, heads = self.conf.heads).to(device))

            for _ in range(self.conf.n_layers - 1):
                layers.append(GATConv(self.conf.hidden_dim*self.conf.heads, self.conf.hidden_dim, heads = self.conf.heads).to(device))
        
        if self.conf.conv_type == 'gatv2':
            layers = []
            layers.append(GATv2Conv(input_features, self.conf.hidden_dim, heads = self.conf.heads).to(device))

            for _ in range(self.conf.n_layers - 1):
                layers.append(GATv2Conv(self.conf.hidden_dim*self.conf.heads, self.conf.hidden_dim, heads = self.conf.heads).to(device))
        
        if self.conf.conv_type == 'sage':
            layers = []
            layers.append(SAGEConv(input_features, self.conf.hidden_dim, aggr = self.conf.aggr_sage).to(device))

            for _ in range(self.conf.n_layers - 1):
                layers.append(SAGEConv(self.conf.hidden_dim, self.conf.hidden_dim, aggr = self.conf.aggr_sage).to(device))

        if self.conf.conv_type == 'signed':
            layers = []
            layers.append(SignedConv(input_features, self.conf.hidden_dim, first_aggr = True).to(device))

            for _ in range(self.conf.n_layers - 1):
                layers.append(SignedConv(self.conf.hidden_dim*2, self.conf.hidden_dim, first_aggr = True).to(device))

        
        
        if self.conf.conv_type != 'gin':
            dropout_layers = [nn.Dropout(self.conf.dropout_prob) for _ in range(self.conf.n_layers)]
            activation_layers = [activation() for _ in range(self.conf.n_layers)]
            
            #batch_layers = [InstanceNorm(self.conf.hidden_dim) for _ in range(self.conf.n_layers)] if not self.conf.conv_type.startswith('gat') else [InstanceNorm(self.conf.heads*self.conf.hidden_dim) for _ in range(self.conf.n_layers)]
            #self.batch_layers = nn.Sequential(*batch_layers)        
            self.activation_layers = nn.Sequential(*activation_layers)
            self.dropout_layers = nn.Sequential(*dropout_layers)
        
        # Decoder
        if self.conf.conv_type.startswith('gat'):
            self.dec = nn.Sequential(
                nn.Linear(self.conf.hidden_dim*self.conf.heads,  output_dim)
            )
        elif self.conf.conv_type == 'signed':
            self.dec = nn.Sequential(
                nn.Linear(self.conf.hidden_dim*2,  output_dim)
            )
        else:
            self.dec = nn.Sequential(
                nn.Linear(self.conf.hidden_dim,  output_dim)
            )

        self.layers = nn.Sequential(*layers)
        
        self.reset_parameters()
        self.to(device)
    
    def reset_parameters(self):
        for module in [self.dec]:
            if isinstance(module, nn.Sequential):
                for layer in module:
                    if isinstance(layer, nn.Linear):
                        layer.reset_parameters()

        for el in self.layers:
            if isinstance(el, activation) or isinstance(el, nn.Dropout):
                continue
            else:
                el.reset_parameters()

    def forward(self, x, edge_index):

        # for el in self.layers:
        #     for k in el.parameters():
        #         if len(k.shape) == 2:
        #             print(k.shape)
        #             print(k)
        # plot_data(x, -1)
        
        
        # print("Edge Indices are: ", edge_index)
        # print("X after encoding (mean): \n", torch.mean(x))
        
        # plot_data(x, 0)
        
        if self.conf.conv_type != 'gin' and self.conf.conv_type != 'signed':
            for i, layer in enumerate(self.layers):

                # batch_layer = self.batch_layers[i]
                drop_layer = self.dropout_layers[i]
                activation_layer = self.activation_layers[i]

                if i == 0 or self.conf.conv_type.startswith('gat'):
                    x = drop_layer(activation_layer(layer(x, edge_index))) # The first layer does not have a residual connection / Gat do not have residuals
                else:
                    x = x + drop_layer(activation_layer(layer(x, edge_index)))
        
        elif self.conf.conv_type == 'signed':
            
            pos_edge_index = edge_index # Sampling positive edges
            # print("Pos edge index: ", pos_edge_index.shape)
            # Sampling negative edges
            neg_edge_index = negative_sampling(pos_edge_index, num_nodes=x.shape[0], num_neg_samples=int(pos_edge_index.shape[1]*self.conf.negatives))
            # print("Neg edge index: ", neg_edge_index.shape)
            
            for i, layer in enumerate(self.layers):

                # batch_layer = self.batch_layers[i]
                drop_layer = self.dropout_layers[i]
                activation_layer = self.activation_layers[i]

                x = drop_layer(activation_layer(layer(x, pos_edge_index, neg_edge_index)))

                # print("Docs: ", x.shape)
                
        elif self.conf.conv_type == 'gin':

            x = self.layers[0](x, edge_index)
        # plot_data(x, i+1)
        
        if self.enc_mode:
            return x
        else:
            output = self.dec(x)

            return output
    

class MLP(nn.Module):
    def __init__(self, input_features, hidden_dim, output_dim = 1, n_layers=1, device='cpu', dropout_prob=0.5):
        super(MLP, self).__init__()

        layers = []
        # Input layer
        layers.append(nn.Linear(input_features, hidden_dim))
        #layers.append(nn.ReLU())
        layers.append(activation())

        layers.append(nn.Dropout(dropout_prob))

        # Hidden layers
        for _ in range(n_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            #layers.append(nn.ReLU())
            layers.append(activation())
            layers.append(nn.Dropout(dropout_prob))

        # Output layer
        self.dec = nn.Sequential(nn.Linear(hidden_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

        self.reset_parameters()
        self.to(device)

    def reset_parameters(self):
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()

    def forward(self, x, enc = False):

        encodings = self.mlp(x)

        if enc:
            return encodings
        else:
            return self.dec(encodings)


class GNN_LG(nn.Module):

    def __init__(self, input_features, config, modality = 'local', conv_type = 'gcn', device = 'cpu'):
        super(GNN_LG, self).__init__()

        self.modality = modality
        self.conv_type = conv_type
        self.config = config
        self.loaded = False
        self.dev = device

        if self.modality == 'local':

            self.GNN = GNN_NR(input_features, self.config, output_dim = self.config.hidden_dim, device = device)
            self.mlp_final = MLP(self.config.hidden_dim + input_features, self.config.hidden_dim, output_dim = 1, n_layers=self.config.n_layers_mlp, device = device, dropout_prob=self.config.dropout_prob)
        
        elif self.modality == 'multistage':
            self.GNN = GNN_NR(input_features, self.config, output_dim = self.config.hidden_dim, device = device)
            self.mlp_final = MLP(self.config.hidden_dim, self.config.hidden_dim, output_dim = 1, n_layers=self.config.n_layers_mlp, device = device, dropout_prob=self.config.dropout_prob)
        
        elif self.modality == 'global':

            with open("models/msmarco_data/config_models.json", 'r') as f:
                config_dict = json.load(f)
                model_dict = config_dict[self.conv_type] # Get dict with best parameters
            
            mod_dict = namedtuple('ObjectDict', model_dict.keys())
            object_dict = mod_dict(*model_dict.values())
            
            self.GNN = GNN_NR(input_features, object_dict, output_dim = object_dict.hidden_dim, device = device)

            load_path = object_dict.model_path

            weight_dict = torch.load(load_path, map_location=device)

            weight_dict = {k[4:]: v for k, v in weight_dict.items() if k.startswith('GNN')}
            
            self.GNN.load_state_dict(weight_dict)

            if self.config.pooling == 'hierarchical':
                self.pool = GNN_Glob(self.config, input_features, device = device)
                self.mlp_final = MLP(self.config.hidden_dim + object_dict.hidden_dim + input_features, self.config.hidden_dim, output_dim = 1, n_layers=self.config.n_layers_mlp, device = device, dropout_prob=self.config.dropout_prob)
            
            elif self.config.pooling == 'maxmean':
                
                self.project = MLP(input_features*2, self.config.hidden_dim, output_dim = self.config.hidden_dim, n_layers=self.config.n_layers_mlp, device = device, dropout_prob=self.config.dropout_prob)
                self.mlp_final = MLP(self.config.hidden_dim + object_dict.hidden_dim + input_features, self.config.hidden_dim, output_dim = 1, n_layers=self.config.n_layers_mlp, device = device, dropout_prob=self.config.dropout_prob)
            
            else:
            
                self.mlp_final = MLP(object_dict.hidden_dim + input_features*2, self.config.hidden_dim, output_dim = 1, n_layers=self.config.n_layers_mlp, device = device, dropout_prob=self.config.dropout_prob)
            

        
        ### Work in progress Global features
            
    def forward(self, x, edge_index):

        if self.modality == 'local':
            
            

            x_local, x_individual = x.clone(), x.clone()
            
            if self.config.score:
                scores =  torch.arange(0, x_local.shape[0], 1, device = self.dev)
                x_local = torch.cat((x_local, scores.unsqueeze(-1)), dim=-1)

            # GNN branch. By default it is ACTIVE (this is the intended GNRR-local
            # behavior). Set config.disable_gnn=True to zero it out, which gives the
            # "no-GNN scorer" control (same capacity, no graph propagation) used in
            # the ablation study.
            if getattr(self.config, 'disable_gnn', False):
                z_local = torch.zeros(x_local.shape[0], self.config.hidden_dim, device = self.dev)
            else:
                z_local = self.GNN(x_local, edge_index)

            z_tot = torch.cat((z_local, x_individual), dim = -1)
            y_pred = self.mlp_final(z_tot)

            return y_pred
        
        elif self.modality == 'multistage':
            
            x_individual = x.clone()

            scores = torch.sum(x_individual, dim = -1)
            # print("SCORES: ", scores.shape)

            # Append scores to the input (Achieved with TCTColBert)
            x_individual = torch.cat((x_individual, scores.unsqueeze(-1)), dim=-1)


            # Get The TOP-K indices and define the new subgraph
            scores_stats = torch.topk(scores, k = self.config.K_multistage if x_individual.shape[0] > self.config.K_multistage else x_individual.shape[0])
            top_indices = scores_stats.indices

            # Get small subgraph

            edge_index, _ = subgraph(top_indices, edge_index, num_nodes = x_individual.shape[0])
            z_local = self.GNN(x_individual.detach(), edge_index)
            z_indices = z_local[top_indices]
            head = self.mlp_final(z_indices).squeeze(-1)   # stage-2 score for top-K

            # Cascade: rank the top-K purely by the GNN head (no TCT residual, which
            # would otherwise be swamped by the large-magnitude TCT score), placing
            # them above the remaining candidates which keep their TCT order.
            from src.attention import cascade_place
            return cascade_place(scores, top_indices, head).unsqueeze(-1)

        elif self.modality == 'global':
            
            x_local, x_individual, x_global = x.clone(), x.clone(), x.clone()
            
            # Global Pooling methods
            if self.config.pooling == 'hierarchical':
                z_global = self.pool(x_global, edge_index)
            elif self.config.pooling == 'mean':
                z_global = gap(x_global, None)
            elif self.config.pooling == 'max':
                z_global = gmp(x_global, None)
            elif self.config.pooling == 'maxmean':
                z_global = self.project(torch.cat((gap(x_global, None), gmp(x_global, None)), dim = -1))

            z_global = torch.repeat_interleave(z_global, repeats=x_local.shape[0], dim=0)
            self.GNN.eval()
            z_local = self.GNN(x_local, edge_index)
            
            z_tot = torch.cat((z_global, z_local.detach(), x_individual), dim = -1)
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


def plot_data(x, it, n_dims=2):

    # Plot t-SNE at each iteration
    tsne = TSNE(n_components=n_dims)
    x_tsne = tsne.fit_transform(x.detach().cpu().numpy())
    plt.scatter(x_tsne[:, 0], x_tsne[:, 1])
    plt.title(f"t-SNE at iteration {it+1}")
    plt.show()
