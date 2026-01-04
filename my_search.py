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
    distance = -2 * xp + torch.sum(graph_emb1 ** 2, dim=1, keepdim=True) + torch.t(
        torch.sum(graph_emb2 ** 2, dim=1, keepdim=True))
    similarity = torch.log((distance + 1) / (distance + epsilon))
    return similarity, distance


def differentiable_search_subgraph(
        proto_graph_emb,
        source_data,  # 传入完整的 Data 对象 (包含 x, edge_index)
        model,
        min_nodes: int = 5,
        max_nodes: int = 12,
        iterations: int = 100,  # 优化步数
        lr: float = 0.05,  # 学习率
        l1_reg: float = 0.1,  # 稀疏性惩罚权重
        lambda_entropy: float = 0.1,  # [新增] 熵正则权重
):
    """
    使用梯度下降优化节点掩码，寻找与原型最相似的连通子图。
    """
    device = source_data.x.device
    num_nodes = source_data.num_nodes

    # 初始化：给一点随机噪声防止陷入鞍点
    mask_logits = torch.nn.Parameter(torch.randn(num_nodes, device=device) * 0.1)

    # 只优化 mask，冻结模型
    optimizer = optim.Adam([mask_logits], lr=lr)
    model.eval()

    # 目标原型嵌入 [1, d]
    if proto_graph_emb.dim() == 1:
        proto_graph_emb = proto_graph_emb.unsqueeze(0)
    target = proto_graph_emb.detach()

    with torch.enable_grad():
        for i in range(iterations):
            optimizer.zero_grad()

            mask = torch.sigmoid(mask_logits)

            # 将 mask 应用到特征
            # 注意：source_data.x 不需要梯度，但 mask 需要，乘积 masked_x 会有梯度
            masked_x = source_data.x * mask.view(-1, 1)

            # 构造临时数据
            masked_data = source_data.clone()
            masked_data.x = masked_x
            if not hasattr(masked_data, "batch") or masked_data.batch is None:
                masked_data.batch = torch.zeros(num_nodes, dtype=torch.long, device=device)

            # 前向传播
            # 即使 model 在 eval 模式，只要输入 masked_x 有梯度，输出就会有梯度
            _, _, _, current_graph_emb, _ = model(masked_data)

            # 计算 Loss
            sim_loss = torch.nn.functional.mse_loss(current_graph_emb, target)

            # L1 Loss: 推动 mask -> 0
            size_loss = torch.mean(mask)

            # Entropy Loss: 推动 mask -> 0 或 1 (二值化)
            # H = -p*log(p) - (1-p)*log(1-p)
            entropy = -mask * torch.log(mask + 1e-8) - (1 - mask) * torch.log(1 - mask + 1e-8)
            entropy_loss = entropy.mean()

            # 组合 Loss
            loss = sim_loss + l1_reg * size_loss + lambda_entropy * entropy_loss

            # 反向传播
            loss.backward()
            optimizer.step()

    # 3. 严格的连通性后处理
    with torch.no_grad():
        final_mask = torch.sigmoid(mask_logits)

        # 1. 确定“种子”：Mask 值最大的那个节点，一定得选
        best_node_idx = torch.argmax(final_mask).item()

        # 2. 构建 NetworkX 图用于拓扑分析
        G = to_networkx(source_data, to_undirected=True)

        # 3. 广度优先搜索 (BFS) 扩展
        # 从种子出发，优先访问 Mask 值高的邻居
        # 只要节点数没到 max_nodes，且邻居的 Mask 值超过阈值，就加入

        selected_nodes = {best_node_idx}
        candidates = []  # (mask_value, node_idx)

        # 将种子的邻居加入候选队列
        for neighbor in G.neighbors(best_node_idx):
            candidates.append((final_mask[neighbor].item(), neighbor))

        # 排序候选：Mask 值大的优先
        candidates.sort(key=lambda x: x[0], reverse=True)

        # 扩展循环
        while len(selected_nodes) < max_nodes and candidates:
            score, node = candidates.pop(0)  # 取出分数最高的

            if node in selected_nodes:
                continue

            # 动态阈值：如果还没达到 min_nodes，门槛低一点；否则门槛高一点
            threshold = 0.3 if len(selected_nodes) < min_nodes else 0.5

            if score > threshold:
                selected_nodes.add(node)
                # 将新节点的邻居加入候选
                new_neighbors = []
                for nb in G.neighbors(node):
                    if nb not in selected_nodes:
                        new_neighbors.append((final_mask[nb].item(), nb))
                # 重新排序 (简单起见，这里可以优化为优先队列)
                candidates.extend(new_neighbors)
                candidates.sort(key=lambda x: x[0], reverse=True)
            else:
                # 如果最高分的候选都不满足阈值，且已经满足最小节点数，提前退出
                if len(selected_nodes) >= min_nodes:
                    break

        # 最终确认
        final_node_indices = torch.tensor(list(selected_nodes), dtype=torch.long, device=device)
        final_node_indices = torch.sort(final_node_indices)[0]

        # 计算相似度返回
        sub_edge_index, _ = subgraph(final_node_indices, source_data.edge_index, relabel_nodes=True)
        final_data = Data(x=source_data.x[final_node_indices], edge_index=sub_edge_index)
        final_data.batch = torch.zeros(final_data.num_nodes, dtype=torch.long, device=device)
        _, _, _, final_emb, _ = model(final_data)
        sim, _ = calculate_similarity(target, final_emb)

        return set(final_node_indices.tolist()), float(sim.item())
