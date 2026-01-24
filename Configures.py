from typing import List
import torch
import os


class DataParser:
    def __init__(self):
        super(DataParser, self).__init__()
        self.dataset_dir = './datasets'
        self.dataset_name = 'BA_2Motifs'
        self.task = None
        self.random_split: bool = True
        self.data_split_ratio: List = [0.8, 0.1, 0.1]


class ModelParser:
    def __init__(self):
        super(ModelParser, self).__init__()
        self.device: int = 0
        self.model_name: str = 'gin'
        self.checkpoint: str = './checkpoint'
        self.concate: bool = False  # whether to concate the gnn features before mlp
        self.latent_dim: List[int] = [128, 128, 128]
        self.readout: str = 'max'
        self.mlp_hidden: List[int] = []  # the hidden units for mlp classifier
        self.gnn_dropout: float = 0.0  # the dropout after gnn layers
        self.dropout: float = 0.5  # the dropout after mlp layers

        self.adj_normalize: bool = True  # the edge_weight normalization for gcn conv
        self.emb_normalize: bool = True  # the l2 normalization after gnn layer

        self.num_prototypes_per_class = 5

        # 搜索的图节点控制
        self.min_nodes: int = 5
        self.max_nodes: int = 9

        self.max_gen_nodes = 9

        # GAT
        self.gat_dropout = 0.6  # dropout in gat layer
        self.gat_heads = 10  # multi-head
        self.gat_hidden = 10  # the hidden units for each head
        self.gat_concate = True  # the concatenation of the multi-head feature
        self.num_gat_layer = 3

    def process_args(self) -> None:
        if torch.cuda.is_available():
            self.device = torch.device('cuda', self.device)
        else:
            pass


class TrainParser:
    def __init__(self):
        super(TrainParser, self).__init__()
        self.learning_rate: float = 5e-3  # 0.005
        self.batch_size: int = 32
        self.weight_decay: float = 1e-3
        self.max_epochs: int = 300
        self.save_epoch = 20
        self.early_stopping = 10000

        self.last_layer_optimizer_lr = 1e-4  # the learning rate of the last layer
        self.joint_optimizer_lrs = {'features': 1e-4,
                                    'add_on_layers': 3e-3,
                                    'prototype_vectors': 3e-3}  # the learning rates of the joint training optimizer
        self.warm_epochs = 10  # the number of warm epochs
        self.proj_epochs = 100
        self.search_epochs = 50  # the epoch to start mcts
        self.sampling_epochs = 100  # the epoch to start sampling edges

        self.random_seed = 42

        # 损失控制
        self.lambda_proto = 0.5  # 约束原型学习过程，使其靠近初始化的原型
        self.lambda_rec = 0.005  # 重构损失
        self.lambda_align = 0.1  # 约束生成的原型靠近学到的原型
        self.lambda_div = 0.01  # 防止同类原型过于接近


data_args = DataParser()
model_args = ModelParser()
train_args = TrainParser()
