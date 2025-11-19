import torch.nn as nn
from models.GCN import GCNNet
from models.GAT import GATNet
from models.GIN import GINNet

__all__ = ['GnnNets']


def get_model(input_dim, output_dim, model_args, data_args):
    if model_args.model_name.lower() == 'gcn':
        return GCNNet(input_dim, output_dim, model_args, data_args)
    elif model_args.model_name.lower() == 'gat':
        return GATNet(input_dim, output_dim, model_args, data_args)
    elif model_args.model_name.lower() == 'gin':
        return GINNet(input_dim, output_dim, model_args, data_args)
    else:
        raise NotImplementedError


class GnnBase(nn.Module):
    def __init__(self):
        super(GnnBase, self).__init__()

    def forward(self, data):
        data = data.to(self.device)
        logtis, probs, node_emb, graph_emb, min_distances = self.model(data)
        return logtis, probs, node_emb, graph_emb, min_distances

    def update_state_dict(self, state_dict):
        original_state_dict = self.state_dict()
        loaded_state_dict = dict()
        for k, v in state_dict.items():
            if k in original_state_dict.keys():
                loaded_state_dict[k] = v
        self.load_state_dict(loaded_state_dict)

    def to_device(self):
        self.to(self.device)

    def save_state_dict(self):
        pass


class GnnNets(GnnBase):
    def __init__(self, input_dim, output_dim, model_args,data_args):
        super(GnnNets, self).__init__()
        self.model = get_model(input_dim, output_dim, model_args,data_args)
        self.device = model_args.device

    def forward(self, data):
        data = data.to(self.device)
        logtis, probs, node_emb, graph_emb, min_distances = self.model(data)
        return logtis, probs, node_emb, graph_emb, min_distances
