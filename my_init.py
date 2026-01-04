import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, to_networkx,to_dense_adj
import networkx as nx


@torch.no_grad()
def extract_connected_subgraph(
        graph: Data,
        model,  # 预训练的classifier
        target_class: int = None,
        min_nodes: int = 6,
        max_nodes: int = 12,
        use_occlusion: bool = True,
        conf_keep_ratio: float = 0.95,
        auto_expand: bool = True,
        expand_step: int = 5,
        return_info: bool = False,
        verbose_cc: bool = False,  # 是否打印 CC 尺寸变化
):
    model.eval()
    device = next(model.parameters()).device
    g = graph.to(device)

    if not hasattr(g, "batch") or g.batch is None:
        g.batch = torch.zeros(g.num_nodes, dtype=torch.long, device=device)

    # 原始预测
    logits, probs, node_emb, graph_emb = model(g)
    if target_class is None:
        target_class = int(logits.argmax(dim=-1).item())
    original_conf = float(probs[0, target_class].item())
    threshold = conf_keep_ratio * original_conf

    # 重要性：cos + occlusion
    node_importance = F.cosine_similarity(node_emb, graph_emb, dim=-1)

    if use_occlusion:
        occlusion_drop = torch.zeros(g.num_nodes, device=device)
        for i in range(g.num_nodes):
            mask = torch.ones(g.num_nodes, dtype=torch.bool, device=device)
            mask[i] = False
            sub_edge_index, _ = subgraph(mask, g.edge_index, relabel_nodes=True)
            if mask.sum().item() == 0:
                continue
            temp_batch = torch.zeros(mask.sum().item(), dtype=torch.long, device=device)
            temp_data = Data(x=g.x[mask], edge_index=sub_edge_index, batch=temp_batch).to(device)
            _, temp_probs, _, _ = model(temp_data)
            conf = float(temp_probs[0, target_class].item())
            occlusion_drop[i] = original_conf - conf

        occlusion_drop = torch.clamp(occlusion_drop, min=0.0)
        final_importance = node_importance + 0.8 * occlusion_drop
    else:
        final_importance = node_importance

    sorted_nodes = torch.argsort(final_importance.detach().cpu(), descending=True).tolist()
    N = len(sorted_nodes)

    min_nodes = max(1, min_nodes)
    max_nodes = min(max_nodes, N)

    expanded_times = 0
    current_max = max_nodes

    # 全局兜底（记录“连通子图”上的最好 conf）
    best_cc_data = None
    best_cc_topk_nodes = None
    best_cc_conf = -1.0

    def build_topk_sub(topk_nodes):
        sub_edge_index, _ = subgraph(topk_nodes, g.edge_index, relabel_nodes=True)
        sub_batch = torch.zeros(topk_nodes.size(0), dtype=torch.long, device=device)
        sub_data = Data(x=g.x[topk_nodes], edge_index=sub_edge_index, batch=sub_batch).to(device)
        sub_data.num_nodes = topk_nodes.size(0)
        return sub_data

    def take_largest_cc(sub_data):
        """在 sub_data 上取最大连通分量，返回 cc_data 以及 cc 内部索引 final_nodes"""
        # sub_data 为空的兜底
        if sub_data.num_nodes == 0:
            final_nodes = torch.tensor([0], dtype=torch.long, device=device)
            cc_edge_index, _ = subgraph(final_nodes, sub_data.edge_index, relabel_nodes=True)
            cc_data = Data(x=sub_data.x[final_nodes], edge_index=cc_edge_index, batch=None).to(device)
            cc_data.num_nodes = cc_data.x.size(0)
            return cc_data, final_nodes

        G_nx = to_networkx(sub_data, to_undirected=True)
        ccs = list(nx.connected_components(G_nx))
        if len(ccs) == 0:
            largest_cc = {0}
        else:
            largest_cc = max(ccs, key=len)

        final_nodes = torch.tensor(sorted(largest_cc), dtype=torch.long, device=device)
        if final_nodes.numel() == 0:
            final_nodes = torch.tensor([0], dtype=torch.long, device=device)

        cc_edge_index, _ = subgraph(final_nodes, sub_data.edge_index, relabel_nodes=True)
        cc_data = Data(x=sub_data.x[final_nodes], edge_index=cc_edge_index, batch=None).to(device)
        cc_data.num_nodes = cc_data.x.size(0)
        return cc_data, final_nodes

    while True:
        valid = []  # (k, conf_cc, cc_data, topk_nodes, final_nodes)

        # 在[min_nodes, current_max]内扫描所有 k
        for k in range(current_max, min_nodes - 1, -1):
            topk_nodes = torch.tensor(sorted_nodes[:k], dtype=torch.long, device=device)
            topk_sub = build_topk_sub(topk_nodes)

            # ===== 改法2关键：先取 CC，再用 CC 算 conf =====
            cc_data, final_nodes = take_largest_cc(topk_sub)

            # cc_data 可能 < min_nodes，这里先照样算 conf（用于兜底）
            cc_data.batch = torch.zeros(cc_data.num_nodes, dtype=torch.long, device=device)
            _, cc_probs, _, _ = model(cc_data)
            conf_cc = float(cc_probs[0, target_class].item())

            if verbose_cc:
                print(f"[k={k}] topk_nodes={k}, cc_nodes={cc_data.num_nodes}, conf_cc={conf_cc:.4f}")

            # 全局最好兜底（在 CC 上比较）
            if conf_cc > best_cc_conf:
                best_cc_conf = conf_cc
                best_cc_data = cc_data
                best_cc_topk_nodes = topk_nodes
                best_cc_final_nodes = final_nodes

            # 只有“CC节点数也>=min_nodes 且 conf_cc 达标” 才算可行
            if cc_data.num_nodes >= min_nodes and conf_cc >= threshold:
                valid.append((k, conf_cc, cc_data, topk_nodes, final_nodes))

        if valid:
            # 选最小可行 k（valid 里 k 是降序扫的，所以最后一个最小）
            k_min, conf_min, cc_min, topk_min, final_nodes_min = valid[-1]
            best_cc_conf = conf_min
            best_cc_data = cc_min
            best_cc_topk_nodes = topk_min
            best_cc_final_nodes = final_nodes_min
            break

        # 没可行解：放宽 max_nodes 或退出
        if auto_expand and current_max < N:
            current_max = min(N, current_max + expand_step)
            expanded_times += 1
            continue
        else:
            break

    # ===== best_cc_data 此时就是“最终返回的连通子图” =====
    connected_subgraph = Data(
        x=best_cc_data.x.detach().cpu(),
        edge_index=best_cc_data.edge_index.detach().cpu()
    )
    connected_subgraph.num_nodes = connected_subgraph.x.size(0)

    if hasattr(graph, "y") and graph.y is not None:
        connected_subgraph.y = graph.y.detach().cpu()

    # orig_node_idx：topk_nodes 中对应 CC 节点的原图索引
    connected_subgraph.orig_node_idx = best_cc_topk_nodes[best_cc_final_nodes].detach().cpu()

    print(f"原始节点: {graph.num_nodes} → 子图节点: {connected_subgraph.num_nodes}")
    print(f"原始置信度: {original_conf:.4f} → 子图置信度: {best_cc_conf:.4f} "
          f"(保留 {best_cc_conf / (original_conf + 1e-12):.2%})")
    if expanded_times > 0:
        print(f"[AutoExpand] 放宽了 {expanded_times} 次 max_nodes，最终 max_nodes={current_max}")

    if return_info:
        info = dict(
            original_nodes=g.num_nodes,
            sub_nodes=connected_subgraph.num_nodes,
            original_conf=original_conf,
            sub_conf=best_cc_conf,
            keep_ratio=best_cc_conf / (original_conf + 1e-12),
            target_class=target_class,
            min_nodes=min_nodes,
            init_max_nodes=max_nodes,
            final_max_nodes=current_max,
            expanded_times=expanded_times
        )
        return connected_subgraph, info

    return connected_subgraph



def extract_connected_subgraph2(
        graph: Data,
        model,  # 预训练的classifier
        target_class: int = None,
        min_nodes: int = 6,
        max_nodes: int = 12,
        return_info: bool = False,
):
    """
    三阶段高效子图提取：
    1) 节点贡献度快速评估
    2) 结构感知邻域扩展
    3) 子图质量快速筛选
    """
    model.eval()
    device = next(model.parameters()).device
    g = graph.to(device)

    if not hasattr(g, "batch") or g.batch is None:
        g.batch = torch.zeros(g.num_nodes, dtype=torch.long, device=device)

    # 原始预测
    # ===== 启用梯度计算 =====
    g.x.requires_grad = True
    # 确保模型参数的梯度追踪是启用的
    for param in model.parameters():
        param.requires_grad = True
    logits, probs, node_emb, graph_emb = model(g)
    if target_class is None:
        target_class = int(logits.argmax(dim=-1).item())
    original_conf = float(probs[0, target_class].item())

    # ===== 阶段1: 节点贡献度快速评估 =====
    # 使用梯度法计算节点贡献度
    # 创建新的计算图用于梯度计算
    loss = logits[0, target_class]
    loss.backward(retain_graph=True)

    # if g.x.grad is None:
    #     # 降级方案：如果无法获得梯度，使用嵌入范数
    #     contribution = torch.norm(node_emb, dim=1)
    # else:
    node_grad = torch.norm(g.x.grad, dim=1)
    contribution = node_grad * torch.norm(node_emb, dim=1)

    # 选择top-k种子节点
    k = min(5, g.num_nodes)  # 种子节点数
    # 确保贡献度是有效的
    if torch.isnan(contribution).any() or torch.isinf(contribution).any():
        contribution = torch.norm(node_emb, dim=1)

    topk_values, topk_indices = torch.topk(contribution, min(k, g.num_nodes))
    topk_indices = topk_indices.cpu().tolist()

    # ===== 阶段2: 结构感知邻域扩展 =====
    candidate_subgraphs = []
    candidate_node_sets = []

    for seed_idx in topk_indices:
        # 计算节点亲和度：嵌入相似度 + 拓扑接近性
        seed_emb = node_emb[seed_idx].unsqueeze(0)
        emb_sim = F.cosine_similarity(node_emb, seed_emb, dim=1)

        # 计算Jaccard邻居重叠度
        adj = to_dense_adj(g.edge_index, max_num_nodes=g.num_nodes)[0]
        # 确保邻接矩阵是二值的
        adj = (adj > 0).float()

        # 高效计算所有节点与种子节点的Jaccard相似度
        seed_neighbors = adj[seed_idx]  # 形状: (N,)
        node_degrees = adj.sum(dim=1)  # 每个节点的度数，形状: (N,)

        # 计算交集大小 (种子节点与每个节点共有的邻居数量)
        intersection = torch.mm(adj, seed_neighbors.unsqueeze(1)).squeeze(1)  # 形状: (N,)

        # 计算并集大小
        seed_degree = seed_neighbors.sum()
        union = seed_degree + node_degrees - intersection

        # 避免除零错误
        union = torch.clamp(union, min=1e-8)

        # 计算Jaccard相似度
        jaccard_sim = intersection / union

        # 确保jaccard_sim是一维张量
        jaccard_sim = jaccard_sim.squeeze()

        # 计算嵌入相似度
        seed_emb = node_emb[seed_idx].unsqueeze(0)  # 形状: (1, d)
        emb_sim = F.cosine_similarity(node_emb, seed_emb, dim=1)  # 形状: (N,)

        # 融合两种相似度
        lambda_val = 0.5
        node_affinity = lambda_val * emb_sim + (1 - lambda_val) * jaccard_sim  # 形状: (N,)

        # BFS扩展，优先选择高亲和度节点
        visited = set([seed_idx])
        queue = [seed_idx]
        affinity_threshold = 0.3  # 亲和度阈值

        while queue and len(visited) < max_nodes:
            current = queue.pop(0)
            # 获取邻居节点
            neighbors_src = g.edge_index[1, g.edge_index[0] == current].cpu().tolist()
            neighbors_dst = g.edge_index[0, g.edge_index[1] == current].cpu().tolist()
            neighbors = list(set(neighbors_src + neighbors_dst) - visited)

            # 按亲和度排序
            neighbor_affinities = [(n, float(node_affinity[n])) for n in neighbors]
            neighbor_affinities.sort(key=lambda x: x[1], reverse=True)

            # 扩展高亲和度邻居
            for n, affinity in neighbor_affinities:
                if affinity > affinity_threshold and len(visited) < max_nodes:
                    visited.add(n)
                    queue.append(n)
                if len(visited) >= max_nodes:
                    break

        # 确保子图连通性
        node_list = list(visited)
        sub_edge_index, _ = subgraph(node_list, g.edge_index, relabel_nodes=True)

        # 构建子图
        sub_data = Data(
            x=g.x[node_list],
            edge_index=sub_edge_index,
            batch=torch.zeros(len(node_list), dtype=torch.long, device=device)
        ).to(device)
        sub_data.num_nodes = len(node_list)

        candidate_subgraphs.append(sub_data)
        candidate_node_sets.append(node_list)

    # ===== 阶段3: 子图质量快速筛选 =====
    best_subgraph = None
    best_node_set = None
    best_conf = -1
    best_score = -1
    alpha = 0.7  # 语义保留权重

    for sub_data, node_set in zip(candidate_subgraphs, candidate_node_sets):
        # 1) 计算子图置信度保留率
        _, sub_probs, _, _ = model(sub_data)
        sub_conf = float(sub_probs[0, target_class].item())
        conf_retention = sub_conf / (original_conf + 1e-8)

        # 2) 计算结构紧凑性得分
        if sub_data.num_nodes > 1:
            # 平均度
            avg_degree = 2 * sub_data.edge_index.size(1) / sub_data.num_nodes

            # 直径估计 (BFS从任意节点)
            adj_sub = to_dense_adj(sub_data.edge_index, max_num_nodes=sub_data.num_nodes)[0]
            diameter = 0
            for start in range(min(3, sub_data.num_nodes)):  # 采样3个起点
                dist = {i: float('inf') for i in range(sub_data.num_nodes)}
                dist[start] = 0
                queue = [start]
                while queue:
                    u = queue.pop(0)
                    neighbors = torch.where(adj_sub[u] > 0)[0].tolist()
                    for v in neighbors:
                        if dist[v] == float('inf'):
                            dist[v] = dist[u] + 1
                            queue.append(v)
                max_dist = max(dist.values())
                diameter = max(diameter, max_dist)
            diameter = max(1, diameter)  # 避免除零
        else:
            avg_degree = 0
            diameter = 1

        compactness = avg_degree / (diameter + 1e-8)

        # 3) 综合得分
        score = alpha * conf_retention + (1 - alpha) * compactness

        # 选择最佳子图
        if score > best_score:
            best_score = score
            best_subgraph = sub_data
            best_node_set = node_set
            best_conf = sub_conf

    # 构建最终子图 (CPU版本)
    final_subgraph = Data(
        x=best_subgraph.x.detach().cpu(),
        edge_index=best_subgraph.edge_index.detach().cpu()
    )
    final_subgraph.num_nodes = final_subgraph.x.size(0)
    final_subgraph.orig_node_idx = torch.tensor(best_node_set, dtype=torch.long).cpu()

    if hasattr(graph, "y") and graph.y is not None:
        final_subgraph.y = graph.y.detach().cpu()

    # 打印信息
    print(f"原始节点: {graph.num_nodes} → 子图节点: {final_subgraph.num_nodes}")
    print(f"原始置信度: {original_conf:.4f} → 子图置信度: {best_conf:.4f} "
          f"(保留 {best_conf / (original_conf + 1e-12):.2%})")

    if return_info:
        info = {
            'original_nodes': graph.num_nodes,
            'sub_nodes': final_subgraph.num_nodes,
            'original_conf': original_conf,
            'sub_conf': best_conf,
            'conf_retention': best_conf / (original_conf + 1e-12),
            'target_class': target_class,
            'selection_score': best_score
        }
        return final_subgraph, info

    return final_subgraph
