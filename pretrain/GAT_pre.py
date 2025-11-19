import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GATConv
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


class GATClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, model_args):
        super(GATClassifier, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = model_args.latent_dim
        self.emb_normalize = model_args.emb_normalize
        self.device = torch.device('cuda:' + str(model_args.device))
        self.num_gnn_layers = len(self.latent_dim)
        self.dense_dim = model_args.gat_hidden * model_args.gat_heads
        self.readout_layers = get_readout_layers(model_args.readout)

        self.gnn_layers = nn.ModuleList()
        self.gnn_layers.append(GATConv(input_dim, model_args.gat_hidden, heads=model_args.gat_heads,
                                       dropout=model_args.gat_dropout, concat=model_args.gat_concate))
        for i in range(1, self.num_gnn_layers):
            self.gnn_layers.append(GATConv(self.dense_dim, model_args.gat_hidden, heads=model_args.gat_heads,
                                           dropout=model_args.gat_dropout, concat=model_args.gat_concate))
        self.gnn_non_linear = nn.ReLU()

        self.bn = nn.BatchNorm1d(self.dense_dim)

        self.mlp = nn.Linear(self.dense_dim * len(self.readout_layers), output_dim)
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
            x = self.gnn_layers[i](x, edge_index)
            if self.emb_normalize:
                x = F.normalize(x, p=2, dim=-1)
            x = self.gnn_non_linear(x)

        node_emb = x
        pooled = []
        for readout in self.readout_layers:
            pooled.append(readout(x, batch))
        x = torch.cat(pooled, dim=1)
        graph_emb = x
        graph_emb = self.bn(graph_emb)

        graph_emb = F.dropout(graph_emb, p=0.5, training=self.training)
        logits = self.mlp(graph_emb)
        probs = self.Softmax(logits)
        return logits, probs, node_emb, graph_emb
