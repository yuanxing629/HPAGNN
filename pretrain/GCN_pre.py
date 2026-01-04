import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GCNConv
from torch_geometric.nn.glob import global_mean_pool, global_max_pool, global_add_pool


def get_readout_layers(readout):
    readout_func_dict = {
        "mean": global_mean_pool,
        "sum": global_add_pool,
        "max": global_max_pool
    }
    readout_func_dict = {k.lower(): v for k, v in readout_func_dict.items()}
    ret_readout = []
    for k, v in readout_func_dict.items():
        if k in readout.lower():
            ret_readout.append(v)
    return ret_readout


class GCNClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, model_args):
        super(GCNClassifier, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = model_args.latent_dim
        self.emb_normalize = model_args.emb_normalize
        self.device = torch.device('cuda:' + str(model_args.device))
        self.num_gnn_layers = len(self.latent_dim)
        self.readout_layers = get_readout_layers(model_args.readout)

        self.gnn_layers = nn.ModuleList()
        self.gnn_layers.append(GCNConv(self.input_dim, self.latent_dim[0]))
        for i in range(1, self.num_gnn_layers):
            self.gnn_layers.append(
                GCNConv(self.latent_dim[i - 1], self.latent_dim[i], normalize=model_args.adj_normalize))
        self.gnn_non_linear = nn.ReLU()

        self.mlp = nn.Linear(self.latent_dim[-1], output_dim)
        self.Softmax = nn.Softmax(dim=-1)

    def update_state_dict(self, state_dict):
        original_state_dict = self.state_dict()
        loaded_state_dict = dict()
        for k, v in state_dict.items():
            if k in original_state_dict.keys():
                loaded_state_dict[k] = v
        self.load_state_dict(loaded_state_dict)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for i in range(self.num_gnn_layers):
            if not self.gnn_layers[i].normalize:
                edge_weight = torch.ones(edge_index.shape[1]).to(self.device)
                x = self.gnn_layers[i](x, edge_index, edge_weight)
            else:
                x = self.gnn_layers[i](x, edge_index)
            if self.emb_normalize:  # the l2 normalization after gnn layer
                x = F.normalize(x, p=2, dim=-1)
            x = self.gnn_non_linear(x)

        node_emb = x
        pooled = []
        for readout in self.readout_layers:
            pooled.append(readout(x, batch))
        x = torch.cat(pooled, dim=1)
        graph_emb = x

        logits = self.mlp(graph_emb)
        probs = self.Softmax(logits)
        return logits, probs, node_emb, graph_emb
