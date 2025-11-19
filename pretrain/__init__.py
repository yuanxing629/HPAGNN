import torch.nn as nn
from pretrain.GCN_pre import GCNClassifier
from pretrain.GIN_pre import GINClassifier
from pretrain.GAT_pre import GATClassifier

__all__ = ['GnnClassifier']


def get_model(input_dim, output_dim, model_args):
    if model_args.model_name.lower() == 'gcn':
        return GCNClassifier(input_dim, output_dim, model_args)
    elif model_args.model_name.lower() == 'gat':
        return GATClassifier(input_dim, output_dim, model_args)
    elif model_args.model_name.lower() == 'gin':
        return GINClassifier(input_dim, output_dim, model_args)
    else:
        raise NotImplementedError


class GnnBase(nn.Module):
    def __init__(self):
        super(GnnBase, self).__init__()

    def forward(self, data):
        data = data.to(self.device)
        logtis, probs, node_emb, graph_emb = self.model(data)
        return logtis, probs, node_emb, graph_emb

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


class GnnClassifier(GnnBase):
    def __init__(self, input_dim, output_dim, model_args):
        super(GnnClassifier, self).__init__()
        self.model = get_model(input_dim, output_dim, model_args)
        self.device = model_args.device

    def forward(self, data):
        data = data.to(self.device)
        logtis, probs, node_emb, graph_emb = self.model(data)
        return logtis, probs, node_emb, graph_emb
