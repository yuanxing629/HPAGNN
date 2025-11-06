import os
from typing import List
import torch
import random
import numpy as np

class DataParser():
    def __init__(self):
        super(DataParser, self).__init__()
        self.dataset_dir = './datasets'
        self.dataset_name = 'MUTAG'
        self.task = None
        self.random_split: bool = True
        self.data_split_ratio: List = [0.8, 0.1, 0.1]

class ModelParser():
    def __init__(self):
        super(ModelParser, self).__init__()
        self.device: int = 0
        self.model_name: str  = 'gcn'
        self.checkpoint: str = './checkpoint'
        self.concate: bool = False # whether to concate the gnn features before mlp
        self.latent_dim: List[int] = [128, 128, 128]
        self.readout: str = 'max'
        self.gnn_dropout: float = 0.0 # the dropout after gnn layers
        self.dropout: float = 0.5 # the dropout after mlp layers
        self.adj_normalize: bool = True # the edge_weight normalization for gcn conv
        self.emb_normalize: bool = False  # the l2 normalization after gnn layer

        self.enable_prot = True
        self.num_prototypes_per_class = 5

    def process_args(self) -> None:
        if torch.cuda.is_available():
            self.device = torch.device('cuda', self.device)
        else:
            pass

class TrainParser():
    def __init__(self):
        super(TrainParser, self).__init__()
        self.learning_rate: float = 5e-3 # 0.005
        self.batch_size: int = 24
        self.weight_decay: float = 0.0
        self.max_epochs: int = 500
        self.save_epoch = 20
        self.early_stopping = 10000

        self.last_layer_optimizer_lr = 1e-4            # the learning rate of the last layer
        self.joint_optimizer_lrs = {'features': 1e-4,
                       'add_on_layers': 3e-3,
                       'prototype_vectors': 3e-3}      # the learning rates of the joint training optimizer
        self.warm_epochs = 10                          # the number of warm epochs
        self.proj_epochs = 50                         # the epoch to start mcts
        self.sampling_epochs = 100                     # the epoch to start sampling edges
        self.nearest_graphs = 10                       # number of graphs in projection

data_args = DataParser()
model_args = ModelParser()
train_args = TrainParser()