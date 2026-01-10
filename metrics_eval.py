import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, silhouette_score
from torch_geometric.utils import subgraph, to_networkx
from torch_geometric.data import Data, Batch
import networkx as nx

from my_search import differentiable_search_subgraph
from models import GnnNets
from load_dataset import get_dataset, get_dataloader
from Configures import data_args, train_args, model_args
import os
import random
import re


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 关键：让 CuDNN 和算子都尽量确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


random_seed = train_args.random_seed
set_seed(random_seed)


def load_model_with_dynamic_prototypes(gnnNets, ckpt_path):
    print(f"Loading model from {ckpt_path} with dynamic prototype handling...")
    checkpoint = torch.load(ckpt_path, weights_only=False)
    state_dict = checkpoint['net']

    # 1. 分析 state_dict，找出原型相关的参数
    # 假设 key 的格式是 "model.prototype_node_emb.0", "model.prototype_node_emb.1" ...
    # 或者 "prototype_node_emb.0" (取决于 GnnNets 是否 wrap 了 model)
    proto_keys = [k for k in state_dict.keys() if "prototype_node_emb" in k]

    if not proto_keys:
        print("Warning: No prototype embeddings found in checkpoint!")
        return gnnNets

    # 2. 获取原型的数量和每个原型的形状
    # 我们需要找到最大的索引
    max_idx = -1
    # 用字典存每个索引对应的 tensor shape
    shapes = {}

    pattern = re.compile(r"prototype_node_emb\.(\d+)")

    for k in proto_keys:
        match = pattern.search(k)
        if match:
            idx = int(match.group(1))
            max_idx = max(max_idx, idx)
            shapes[idx] = state_dict[k].shape

    print(f"Found {max_idx + 1} prototypes in checkpoint.")

    # 3. 在模型中预先创建占位 Parameter
    # 必须按顺序创建，否则加载时索引会乱
    # gnnNets.model 是 GINNet 实例

    # 先清空（以防万一）
    gnnNets.model.prototype_node_emb = torch.nn.ParameterList()

    device = next(gnnNets.parameters()).device

    for i in range(max_idx + 1):
        if i in shapes:
            # 创建一个形状匹配的 Parameter (值无所谓，会被 overwrite)
            shape = shapes[i]
            dummy_param = torch.nn.Parameter(torch.zeros(shape, device=device))
            gnnNets.model.prototype_node_emb.append(dummy_param)
        else:
            print(f"Error: Prototype {i} missing in state_dict!")

    # 4. 正常加载权重
    gnnNets.load_state_dict(state_dict)
    print("Model loaded successfully.")

    return gnnNets


class ExplanationEvaluator:
    def __init__(self, model, model_args):
        """
        :param model: 训练好的 GINNet 模型
        :param model_args: 配置参数 (包含 num_prototypes_per_class 等)
        """
        self.model = model
        self.model.eval()
        self.model_args = model_args
        self.device = model_args.device


    def evaluate_silhouette(self):
        """
        计算原型的 Silhouette Score。
        评估原型在潜在空间中的聚类效果。
        """
        with torch.no_grad():
            # [C*m, d]
            proto_embeddings = self.model.proto_node_emb_2_graph_emb()

            # 计算所有原型之间的平方欧式距离矩阵
            # ||Pi - Pj||^2 = ||Pi||^2 + ||Pj||^2 - 2 * Pi * Pj^T
            p_sq = torch.sum(proto_embeddings ** 2, dim=1, keepdim=True)  # [N, 1]
            dist_matrix = p_sq + p_sq.t() - 2 * torch.mm(proto_embeddings, proto_embeddings.t())

            dist_matrix = dist_matrix.cpu().numpy()
            dist_matrix = np.maximum(dist_matrix, 0)
            np.fill_diagonal(dist_matrix, 0)


        # 生成标签: [0,0,0,0,0, 1,1,1,1,1, ...]
        proto_class_identity = self.model.prototype_class_identity.detach().cpu().numpy()
        labels = np.argmax(proto_class_identity, axis=1)

        # Silhouette Score 需要至少 2 个簇
        if len(set(labels)) < 2:
            return 0.0

        # 使用'precomputed'模式，直接传入距离矩阵
        score = silhouette_score(dist_matrix, labels, metric='precomputed')
        return score


if __name__ == '__main__':
    dataset = get_dataset(data_args.dataset_dir, data_args.dataset_name, task=data_args.task)
    input_dim = dataset.num_node_features
    output_dim = int(dataset.num_classes)

    dataloader = get_dataloader(dataset, train_args.batch_size, random_seed,
                                data_split_ratio=data_args.data_split_ratio)
    gnnNets = GnnNets(input_dim, output_dim, model_args)
    gnnNets.to_device()

    # 使用新的加载方式
    ckpt_path = f"./checkpoint/{data_args.dataset_name}/{model_args.model_name}_{model_args.readout}_best.pth"
    # 确保 ckpt_path 存在

    if os.path.exists(ckpt_path):
        load_model_with_dynamic_prototypes(gnnNets, ckpt_path)
    else:
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    print("\n--- HPAGNN Interpretability Evaluation ---")

    evaluator = ExplanationEvaluator(gnnNets.model, model_args)

    # Silhouette
    sil = evaluator.evaluate_silhouette()
    print(f"Silhouette Score: {sil:.4f}")

