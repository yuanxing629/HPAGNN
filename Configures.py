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

        if self.dataset_name.lower() == 'mutag'.lower():
            self.num_node_classes = 7
        elif self.dataset_name.lower() == 'mutagenicity'.lower():
            self.num_node_classes = 14
        elif self.dataset_name.lower() == 'proteins'.lower():
            self.num_node_classes = 3
        elif self.dataset_name.lower() == 'nci1'.lower():
            self.num_node_classes = 37
        else:
            self.num_node_classes = 1


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
        self.emb_normalize: bool = False  # the l2 normalization after gnn layer

        self.num_prototypes_per_class = 5

        # 生成相关
        self.graph_size = 5



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
        self.batch_size: int = 24
        self.weight_decay: float = 0.0
        self.max_epochs: int = 300
        self.save_epoch = 20
        self.early_stopping = 100

        self.last_layer_optimizer_lr = 1e-4  # the learning rate of the last layer
        self.joint_optimizer_lrs = {'features': 1e-4,
                                    'add_on_layers': 3e-3,
                                    'prototype_vectors': 3e-3}  # the learning rates of the joint training optimizer
        self.warm_epochs = 10  # the number of warm epochs
        self.search_epochs = 50  # the epoch to start mcts
        self.sampling_epochs = 100  # the epoch to start sampling edges

        self.random_seed = 1


        # 损失控制
        self.lambda_proto = 0.1  # 约束原型学习过程，使其靠近初始化的原型
        self.lambda_rec = 0.05 # 重构损失
        self.lambda_align = 0.01 # 约束生成的原型靠近学到的原型
        self.lambda_div = 0.01  # 防止同类原型过于接近



class MCTSParser(DataParser, ModelParser):
    rollout: int = 10                         # the rollout number
    high2low: bool = False                    # expand children with different node degree ranking method
    c_puct: float = 5                         # the exploration hyper-parameter
    min_atoms: int = 5                        # for the synthetic dataset, change the minimal atoms to 5.
    max_atoms: int = 10
    expand_atoms: int = 10                     # # of atoms to expand children

    def process_args(self) -> None:
        self.explain_model_path = os.path.join(self.checkpoint,
                                               self.dataset_name,
                                               f"{self.model_name}_best.pth")


class RewardParser():
    def __init__(self):
        super().__init__()
        self.reward_method: str = 'mc_l_shapley'                         # Liberal, gnn_score, mc_shapley, l_shapley， mc_l_shapley
        self.local_raduis: int = 4                                       # (n-1) hops neighbors for l_shapley
        self.subgraph_building_method: str = 'zero_filling'
        self.sample_num: int = 100                                       # sample time for monte carlo approximation



data_args = DataParser()
model_args = ModelParser()
train_args = TrainParser()

mcts_args = MCTSParser()
reward_args = RewardParser()
