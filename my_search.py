import numpy as np
import torch
import torch.nn.functional as F
from Configures import model_args
from torch_geometric.utils import subgraph, to_networkx
from torch_geometric.data import Data
import torch.optim as optim
import networkx as nx


def calculate_similarity(graph_emb1, graph_emb2):
    epsilon = 1e-4

    # 如果是 1D 向量，先扩展成 [1, d]，避免 torch.mm 报 "self must be a matrix"
    if graph_emb1.dim() == 1:
        graph_emb1 = graph_emb1.unsqueeze(0)  # [1, d]
    if graph_emb2.dim() == 1:
        graph_emb2 = graph_emb2.unsqueeze(0)  # [1, d]

    xp = torch.mm(graph_emb1, torch.t(graph_emb2))
    distance = (-2 * xp + torch.sum(graph_emb1 ** 2, dim=1, keepdim=True) +
                torch.t(torch.sum(graph_emb2 ** 2, dim=1, keepdim=True)))
    distance = torch.clamp(distance, min=0.0)
    similarity = torch.log((distance + 1) / (distance + epsilon))
    return similarity, distance


def post_process_topk_lcc(mask_values, source_data, min_nodes, max_nodes):
    """
    后处理策略：选取 Mask 最高的节点，提取诱导子图，保留其中的最大连通分量 (LCC)
    """
    mask_np = mask_values.detach().cpu().numpy()
    edge_index = source_data.edge_index.detach().cpu().numpy()

    # 选取前 K 个节点
    k = max_nodes
    topk_indices = np.argsort(mask_np)[-k:]
    topk_set = set(topk_indices.tolist())

    # 构建 NetworkX 图
    G = nx.Graph()
    G.add_nodes_from(topk_indices)
    for u, v in edge_index.T:
        if u in topk_set and v in topk_set:
            G.add_edge(u, v)

    # 选取最大连通分量
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    if len(components) == 0:
        return None
    return list(components[0])


def differentiable_search_subgraph(
        proto_graph_emb,
        source_data,
        model,
        min_nodes=3,
        max_nodes=10,
        iterations=50,
        lr=0.05,
        l1_reg=0.01,
        lambda_entropy=0.01,
        lambda_conn=0.1  # 连通性正则权重
):
    """
    使用可微掩码搜索与原型最相似的连通子图
    """
    device = proto_graph_emb.device
    num_nodes = source_data.num_nodes

    # 将目标向量从模型的计算图中分离，使其成为一个纯粹的“常量锚点”
    # 这样 backward 只能影响 mask_logits，而不会尝试回传到模型参数
    proto_graph_emb = proto_graph_emb.detach()

    # 1. 初始化掩码参数 (使用 logits 形式)
    mask_logits = torch.nn.Parameter(torch.randn(num_nodes, device=device) * 0.1)
    optimizer = torch.optim.Adam([mask_logits], lr=lr)

    model.eval()
    with torch.enable_grad():
        for i in range(iterations):
            optimizer.zero_grad()

            # 2. 计算软掩码
            mask = torch.sigmoid(mask_logits)

            # 3. 构造加权特征 (Soft Masking)
            # 这种方式让模型在优化时能感知到哪些节点是不需要的
            x_weighted = source_data.x.to(device) * mask.unsqueeze(-1)
            edge_index = source_data.edge_index.to(device)
            batch = torch.zeros(num_nodes, dtype=torch.long, device=device)

            # 4. 通过模型 Encoder 获取当前掩码下的图嵌入
            # 模拟 forward 过程，但使用加权后的特征
            curr_x = x_weighted
            for layer in model.gnn_layers:
                curr_x = layer(curr_x, edge_index)
                if model.emb_normalize:
                    curr_x = F.normalize(curr_x, p=2, dim=-1)
                curr_x = model.gnn_non_linear(curr_x)

            # Readout
            pooled = []
            for readout in model.readout_layers:
                pooled.append(readout(curr_x, batch))
            graph_emb = torch.cat(pooled, dim=-1)

            if model.emb_normalize:
                graph_emb = F.normalize(graph_emb, p=2, dim=-1)

            # 5. 计算 Loss
            # 相似度损失 (假设使用余弦相似度，若模型用欧氏距离则改用 -dist)
            similarity_val, _ = calculate_similarity(graph_emb, proto_graph_emb)
            sim_loss = -similarity_val.mean()

            # 稀疏性与熵正则
            l1_loss = l1_reg * torch.sum(mask)
            ent_loss = lambda_entropy * (
                -torch.mean(mask * torch.log(mask + 1e-8) + (1 - mask) * torch.log(1 - mask + 1e-8)))

            # 连通性约束
            row, col = source_data.edge_index
            conn_loss = lambda_conn * torch.mean((mask[row] - mask[col]) ** 2)

            total_loss = sim_loss + l1_loss + ent_loss + conn_loss

            total_loss.backward()
            optimizer.step()


    # 6. 后处理：执行 Top-K + LCC
    final_mask = torch.sigmoid(mask_logits)
    best_sub_nodes = post_process_topk_lcc(final_mask, source_data, min_nodes, max_nodes)

    # 7. 计算最终 Hard 子图的相似度作为反馈
    final_sim = 0.0
    if best_sub_nodes is not None:
        node_idx = torch.tensor(best_sub_nodes, dtype=torch.long, device=device)
        sub_edge_index, _ = subgraph(node_idx, source_data.edge_index.to(device), relabel_nodes=True)
        sub_data = Data(x=source_data.x.to(device)[node_idx], edge_index=sub_edge_index)
        sub_data.batch = torch.zeros(sub_data.num_nodes, dtype=torch.long, device=device)

        with torch.no_grad():
            _, _, _, final_graph_emb, _ = model.forward(sub_data)
            sim_val, _ = calculate_similarity(final_graph_emb, proto_graph_emb)
            final_sim = sim_val.item()

    return best_sub_nodes, final_sim
