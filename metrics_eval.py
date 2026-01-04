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

    def get_prediction_prob(self, data, target_class):
        """获取模型对特定类别的预测概率"""
        with torch.no_grad():
            # forward 返回: logits, probs, node_emb, graph_emb, min_distances
            _, probs, _, _, _ = self.model(data)
            return probs[0, target_class].item()

    def generate_explanation_mask(self, data, target_class):
        """
        核心逻辑：
        1. 找到该 target_class 下，与当前 data 最相似的那个原型 (Prototype)。
        2. 使用 differentiable_search_subgraph 在 data 上搜索最像该原型的子图。
        3. 返回该子图对应的节点掩码 (Node Mask)。
        """
        self.model.eval()

        # 1. 获取当前图的 Embedding
        with torch.no_grad():
            _, _, _, graph_emb, _ = self.model(data)  # [1, d]

        # 2. 获取所有原型的 Embedding
        proto_graph_emb = self.model.proto_node_emb_2_graph_emb()  # [Total_Protos, d]

        # 3. 筛选属于 target_class 的原型
        m = self.model_args.num_prototypes_per_class
        start_idx = target_class * m
        end_idx = (target_class + 1) * m
        class_protos = proto_graph_emb[start_idx:end_idx]  # [m, d]

        # 4. 找到最近的原型索引
        # 计算当前图与该类所有原型的相似度 (使用模型中定义的逻辑或简单的 Cosine/Euclidean)
        # 这里为了简化，直接用 Cosine Similarity
        sims = F.cosine_similarity(graph_emb, class_protos)
        best_local_idx = torch.argmax(sims).item()
        best_proto_emb = class_protos[best_local_idx]  # [d]

        # 5. 在当前图上搜索解释子图
        # 注意：differentiable_search_subgraph 需要传入单个 Data 对象
        # 它返回的是 set(final_node_indices)
        best_nodes_set, _ = differentiable_search_subgraph(
            proto_graph_emb=best_proto_emb,
            source_data=data,
            model=self.model,
            min_nodes=self.model_args.min_nodes,
            max_nodes=self.model_args.max_nodes,
            iterations=50,  # 推理时迭代次数可以适当减少
            lr=0.05
        )

        # 6. 构建二值 Mask (Tensor)
        node_mask = torch.zeros(data.num_nodes, dtype=torch.float, device=self.device)
        if best_nodes_set:
            indices = torch.tensor(list(best_nodes_set), device=self.device)
            node_mask[indices] = 1.0

        return node_mask

    def evaluate_fidelity(self, dataset, num_samples=100):
        """
        计算 Fidelity+ (Occlusion) 和 Fidelity- (Sparsity)
        """
        fidelity_plus_scores = []
        fidelity_minus_scores = []

        # 随机采样一部分测试集进行评估
        indices = np.random.choice(len(dataset), min(len(dataset), num_samples), replace=False)

        print(f"Evaluating Fidelity on {len(indices)} samples...")

        for idx in indices:
            data = dataset[idx].to(self.device)
            if data.batch is None:
                data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=self.device)

            # 1. 原始预测
            target_class = data.y.item()
            orig_prob = self.get_prediction_prob(data, target_class)

            # 2. 获取解释掩码 (Important Nodes = 1)
            node_mask = self.generate_explanation_mask(data, target_class)

            # 如果解释为空 (搜索失败)，跳过
            if node_mask.sum() == 0:
                continue

            important_nodes = node_mask.bool()

            # 3. 计算 Fidelity+ (Occlusion): 移除重要节点
            # 构造补图 (保留不重要的节点)
            unimportant_nodes = ~important_nodes
            if unimportant_nodes.sum() > 0:
                sub_edge_index, _ = subgraph(unimportant_nodes, data.edge_index, relabel_nodes=True)
                sub_data = Data(x=data.x[unimportant_nodes], edge_index=sub_edge_index)
                sub_data.batch = torch.zeros(sub_data.num_nodes, dtype=torch.long, device=self.device)

                occluded_prob = self.get_prediction_prob(sub_data, target_class)
                fidelity_plus_scores.append(orig_prob - occluded_prob)
            else:
                # 如果把图全删了，概率视为 0 (或者 1/num_classes)
                fidelity_plus_scores.append(orig_prob)

            # 4. 计算 Fidelity- (Sparsity): 仅保留重要节点
            sub_edge_index_imp, _ = subgraph(important_nodes, data.edge_index, relabel_nodes=True)
            sub_data_imp = Data(x=data.x[important_nodes], edge_index=sub_edge_index_imp)
            sub_data_imp.batch = torch.zeros(sub_data_imp.num_nodes, dtype=torch.long, device=self.device)

            kept_prob = self.get_prediction_prob(sub_data_imp, target_class)
            # Fidelity- 定义为 P_orig - P_kept (越接近0越好)
            fidelity_minus_scores.append(orig_prob - kept_prob)

        return {
            "fidelity_plus": np.mean(fidelity_plus_scores),  # 越高越好
            "fidelity_minus": np.mean(fidelity_minus_scores)  # 越低越好
        }

    def evaluate_auc(self, dataset):
        """
        计算 AUC-ROC。需要 Dataset 中包含 Ground Truth (data.GT_mask 或类似字段)
        通常用于 BA-Shapes 等合成数据集。
        """
        y_true = []
        y_scores = []

        print("Evaluating AUC...")

        for i in range(len(dataset)):
            data = dataset[i].to(self.device)

            # 检查是否有 GT
            # 假设 GT 存储在 data.edge_gt (边掩码) 或 data.node_gt (节点掩码)
            # 你的 differentiable_search 返回的是节点集合，所以我们基于节点计算
            if not hasattr(data, 'train_mask'):  # 这里用 train_mask 举例，实际上你要看 load_dataset.py 里怎么存 GT
                # 如果没有显式的 GT 字段，跳过 (MUTAG 等真实数据集通常没有)
                return None

                # 假设合成数据集的 GT 存在 data.z 或类似字段，这里需要根据 load_dataset.py 调整
            # 比如 BA-Shapes，通常 GT mask 是最后 5 个节点
            # 为了通用，这里假设 data.node_labels 是 GT (如果是 Motif 数据集)
            # 如果没有，你需要在 load_dataset.py 里把 motif 的 mask 加进去

            # 这里写一个占位逻辑，你需要根据 BA-Shapes 的实际 GT 调整：
            # 假设 data.y 是图标签，但在 Node Classification 任务中 data.y 是节点标签
            # 如果是 Graph Classification 的 Motif 检测，通常需要手动标记 Motif 节点
            pass
            # (由于你提供的 load_dataset.py 没有明确的 GT mask 字段用于解释，
            #  AUC 代码仅提供逻辑框架)

            # target_class = data.y.item()
            # node_mask = self.generate_explanation_mask(data, target_class)
            # gt_mask = data.node_gt # 需在 dataset 中实现

            # y_true.extend(gt_mask.cpu().numpy())
            # y_scores.extend(node_mask.cpu().numpy())

        if not y_true:
            print("Warning: No Ground Truth found for AUC evaluation.")
            return 0.0

        return roc_auc_score(y_true, y_scores)

    def evaluate_silhouette(self):
        """
        计算原型的 Silhouette Score。
        评估原型在潜在空间中的聚类效果。
        """
        with torch.no_grad():
            # [C*m, d]
            proto_embeddings = self.model.proto_node_emb_2_graph_emb().cpu().numpy()


        # 生成标签: [0,0,0,0,0, 1,1,1,1,1, ...]
        proto_class_identity = self.model.prototype_class_identity.detach().cpu().numpy()
        labels = np.argmax(proto_class_identity, axis=1)

        # Silhouette Score 需要至少 2 个簇
        if len(set(labels)) < 2:
            return 0.0

        score = silhouette_score(proto_embeddings, labels)
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

    # Fidelity
    # 使用测试集子集
    # test_dataset = dataloader["test"].dataset
    # fids = evaluator.evaluate_fidelity(test_dataset, num_samples=len(test_dataset))
    # print(f"Fidelity+ ↑: {fids['fidelity_plus']:.4f}")
    # print(f"Fidelity- ↓:  {fids['fidelity_minus']:.4f}")

    # AUC (如果有 GT)
    # auc = evaluator.evaluate_auc(test_dataset)
    # print(f"AUC: {auc:.4f}")
