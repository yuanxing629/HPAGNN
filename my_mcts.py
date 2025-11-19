import os
import math
import torch
import torch.nn as nn
import networkx as nx
from Configures import mcts_args, train_args, model_args, data_args
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_networkx
from functools import partial
from collections import Counter
from utils2 import PlotUtils
import random
import numpy as np
from load_dataset import get_dataset, get_dataloader
from models import GnnNets


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 关键：让 CuDNN 和算子都尽量确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


random_seed = train_args.random_seed
set_seed(random_seed)


class MCTSNode():

    def __init__(self, coalition: list, data: Data,
                 ori_graph: nx.Graph, c_puct: float = 10.0,
                 W: float = 0, N: int = 0, P: float = 0):
        self.data = data
        self.coalition = coalition  # 列表
        self.ori_graph = ori_graph  # nx.Graph
        self.c_puct = c_puct
        self.children = []
        self.W = W  # sum of node value
        self.N = N  # times of arrival
        self.P = P  # property score (reward)

    def Q(self):
        return self.W / self.N if self.N > 0 else 0

    def U(self, n):
        return self.c_puct * self.P * math.sqrt(n) / (1 + self.N)


def mcts_rollout(tree_node, state_map, data, graph, score_func):
    cur_graph_coalition = tree_node.coalition
    if len(cur_graph_coalition) <= mcts_args.min_atoms:  # 若联盟大小小于最小原子阈值，直接返回得分
        return tree_node.P

    # Expand if this node has never been visited
    if len(tree_node.children) == 0:  # 若节点未被访问，则通过移除节点来创造孩子
        node_degree_list = list(graph.subgraph(cur_graph_coalition).degree)  # 从当前子图获得节点度数
        node_degree_list = sorted(node_degree_list, key=lambda x: x[1], reverse=mcts_args.high2low)  # 由度数对节点排序，False即升序
        all_nodes = [x[0] for x in node_degree_list]

        if len(all_nodes) < mcts_args.expand_atoms:  # 配置为10。这里要选择top10个节点用于扩展
            expand_nodes = all_nodes
        else:
            expand_nodes = all_nodes[:mcts_args.expand_atoms]

        for each_node in expand_nodes:  # 对于可扩展节点，通过移除节点来创造新的联盟、找到连通分量并保持最大子图（确保连通性）
            # for each node, pruning it and get the remaining sub-graph
            # here we check the resulting sub-graphs and only keep the largest one
            subgraph_coalition = [node for node in all_nodes if node != each_node]  # Pruned

            subgraphs = [graph.subgraph(c)
                         for c in nx.connected_components(graph.subgraph(subgraph_coalition))]
            main_sub = subgraphs[0]
            for sub in subgraphs:
                if sub.number_of_nodes() > main_sub.number_of_nodes():
                    main_sub = sub

            new_graph_coalition = sorted(list(main_sub.nodes()))

            # check the state map and merge the same sub-graph 状态重复数据删除，检查状态图中是否已经存在新子图以避免冗余计算
            Find_same = False
            for old_graph_node in state_map.values():
                if Counter(old_graph_node.coalition) == Counter(new_graph_coalition):
                    new_node = old_graph_node
                    Find_same = True

            if Find_same == False:
                new_node = MCTSNode(new_graph_coalition, data=data, ori_graph=graph)
                state_map[str(new_graph_coalition)] = new_node

            Find_same_child = False
            for cur_child in tree_node.children:
                if Counter(cur_child.coalition) == Counter(new_graph_coalition):
                    Find_same_child = True

            if Find_same_child == False:
                tree_node.children.append(new_node)

        scores = compute_scores(score_func, tree_node.children)  # 使用评分函数对所有新创建的子项进行评分
        for child, score in zip(tree_node.children, scores):
            child.P = score
    # 选择和反向传播
    sum_count = sum([c.N for c in tree_node.children])
    selected_node = max(tree_node.children, key=lambda x: x.Q() + x.U(sum_count))  # 选择具有最高Q+U值的子项(UCB公式)
    v = mcts_rollout(selected_node, state_map, data, graph, score_func)  # 在选定子项上递归调用转出
    selected_node.W += v  # 更新总奖励
    selected_node.N += 1  # 更新访问计数
    return v


def mcts(data, gnnNet, prototype_node_emb):
    data = Data(x=data.x, edge_index=data.edge_index)
    graph = to_networkx(data, to_undirected=True)
    data = Batch.from_data_list([data])
    num_nodes = graph.number_of_nodes()
    root_coalition = sorted([i for i in range(num_nodes)])  # All nodes
    root = MCTSNode(root_coalition, data=data, ori_graph=graph)  # 创建根节点，将所有节点作为初始联盟
    state_map = {str(root.coalition): root}  # 初始化状态图
    score_func = partial(gnn_prot_score, data=data, gnnNet=gnnNet, prototype_node_emb=prototype_node_emb)
    for rollout_id in range(mcts_args.rollout):  # 配置为10，执行多次（10次）MCTS部署
        mcts_rollout(root, state_map, data, graph, score_func)

    explanations = [node for _, node in state_map.items()]
    explanations = sorted(explanations, key=lambda x: x.P, reverse=True)  # 按分数和大小对解释进行排序，
    explanations = sorted(explanations, key=lambda x: len(x.coalition))

    result_node = explanations[0]
    for result_idx in range(len(explanations)):  # 在约束内找到最佳解释
        x = explanations[result_idx]
        if len(x.coalition) <= mcts_args.max_atoms and x.P > result_node.P:
            result_node = x

    # compute the projected prototype to return
    mask = torch.zeros(data.num_nodes).type(torch.float32)
    mask[result_node.coalition] = 1.0  # 为选定节点创建mask
    device = gnnNets.device if hasattr(gnnNets, "device") else next(gnnNets.parameters()).device
    mask = mask.to(device)
    data = data.to(device)
    ret_x = data.x * mask.unsqueeze(1)
    ret_edge_index = data.edge_index
    mask_data = Data(x=ret_x, edge_index=ret_edge_index)  # 对节点特征应用mask
    mask_data = Batch.from_data_list([mask_data])
    _, _, _, emb, _ = gnnNet(mask_data)  # 通过GNN获得嵌入
    return result_node.coalition, result_node.P, emb  # 返回联盟，分数和嵌入


def compute_scores(score_func, children):
    results = []
    for child in children:
        if child.P == 0:
            score = score_func(child.coalition)
        else:
            score = child.P
        results.append(score)
    return results


def gnn_prot_score(coalition, data, gnnNet, prototype_node_emb):
    """ the similarity value of subgraph with selected nodes """
    epsilon = 1e-4
    mask = torch.zeros(data.num_nodes).type(torch.float32)  # 为联盟节点创建mask，并应用于节点特征（屏蔽非联盟节点）
    mask[coalition] = 1.0
    device = gnnNets.device if hasattr(gnnNets, "device") else next(gnnNets.parameters()).device
    mask = mask.to(device)
    data = data.to(device)
    ret_x = data.x * mask.unsqueeze(1)
    ret_edge_index = data.edge_index
    # row, col = data.edge_index
    # edge_mask = (mask[row] == 1) & (mask[col] == 1)
    # ret_edge_index = data.edge_index[:, edge_mask]

    mask_data = Data(x=ret_x, edge_index=ret_edge_index)
    mask_data = Batch.from_data_list([mask_data])
    _, _, _, emb, _ = gnnNet(mask_data)  # 将mask_data传递给GNN，获得嵌入
    proto_graph_emb = prototype_node_emb.max(dim=0)[0]
    distance = torch.norm(emb - proto_graph_emb) ** 2
    similarity = torch.log((distance + 1) / (distance + epsilon))
    return similarity.item()

def calculate_similarity(graph_emb1, graph_emb2):
    epsilon = 1e-4

    # 如果是 1D 向量，先扩展成 [1, d]，避免 torch.mm 报 "self must be a matrix"
    if graph_emb1.dim() == 1:
        graph_emb1 = graph_emb1.unsqueeze(0)  # [1, d]
    if graph_emb2.dim() == 1:
        graph_emb2 = graph_emb2.unsqueeze(0)  # [1, d]

    xp = torch.mm(graph_emb1, torch.t(graph_emb2))
    distance = -2 * xp + torch.sum(graph_emb1 ** 2, dim=1, keepdim=True) + torch.t(
        torch.sum(graph_emb2 ** 2, dim=1, keepdim=True))
    similarity = torch.log((distance + 1) / (distance + epsilon))
    return similarity, distance

if __name__ == '__main__':

    print('start loading data====================')
    dataset = get_dataset(data_args.dataset_dir, data_args.dataset_name)
    input_dim = dataset.num_node_features
    output_dim = int(dataset.num_classes)
    dataloader = get_dataloader(dataset, train_args.batch_size, random_seed,
                                data_split_ratio=data_args.data_split_ratio)
    data_indices = dataloader['train'].dataset.indices
    print('start training model==================')
    gnnNets = GnnNets(input_dim, output_dim, model_args, data_args)

    ckpt_dir = f"./checkpoint/{data_args.dataset_name}/"
    checkpoint = torch.load(os.path.join(ckpt_dir, f"{model_args.model_name}_{model_args.readout}_best.pth"),
                            weights_only=False)
    gnnNets.update_state_dict(checkpoint['net'])
    gnnNets.to_device()
    gnnNets.eval()

    save_dir = os.path.join('./results',
                            f"{mcts_args.dataset_name}_"
                            f"{model_args.model_name}_")
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    plotutils = PlotUtils(dataset_name=data_args.dataset_name)

    device = gnnNets.device if hasattr(gnnNets, "device") else next(gnnNets.parameters()).device
    P = int(gnnNets.model.prototype_node_emb.size(0))
    for p_idx in range(P):
        best = {"sim": float("-inf"), "data_idx": None, "sub_nodes": None}
        proto_label = p_idx // model_args.num_prototypes_per_class
        prototype_node_emb = gnnNets.model.prototype_node_emb[p_idx].detach().to(device)

        for data_idx in range(len(dataset)):
            data = dataset[data_idx].to(device)

            label_val = int(data.y.item() if data.y.numel() == 1 else int(data.y))
            if label_val != int(proto_label):
                continue

            _, _, node_emb, _, _ = gnnNets(data)

            coalition, sim,_ = mcts(data, gnnNets, prototype_node_emb)

            if sim is not None and sim > best['sim']:
                best.update(sim=float(sim), data_idx=data_idx, sub_nodes=coalition)

        if best['data_idx'] is None:
            continue

        # 保存可视化：整图 + 黑色加粗的子图边（utils 已按数据集处理节点配色）
        data = dataset[best["data_idx"]]
        graph = to_networkx(data, to_undirected=True)
        plotutils.plot(graph, best['sub_nodes'], x=data.x,
                       figname=os.path.join(save_dir, f"search_prot_{proto_label}_{p_idx:03d}.png"))
