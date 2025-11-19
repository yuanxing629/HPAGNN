# --- search_strategies.py 或你现有放置 match_* 的文件中 ---

import torch
import torch.nn.functional as F

@torch.no_grad()
def conditional_subgraph_sampling(
    node_emb: torch.Tensor,           # [N, d]  每个节点的嵌入（来自 GNN 编码器最后一层 or 你用于原型的层）
    edge_index: torch.Tensor,         # [2, E]  原图边（建议无向/对称）
    prototype_node_emb: torch.Tensor, # [d]     原型向量（与你的“节点原型”同维）
    budget: int = 10,                 # B：子图的最大节点数（参照论文中的约束）
    pool: str = "mean",               # 子图嵌入的聚合方式：'mean' 或 'sum'
    eps: float = 1e-4                 # 论文相似度中的 ε（Eq.(2)）
):
    """
    ProtGNN+ 风格的“条件子图采样”近似版（贪心连接式）。
    目标：在连通约束与预算 B 下，选择一个子图 S，使得 sim(p, f(G_S)) 最大。
    - 子图嵌入 f(G_S) 用节点嵌入的 mean/sum 聚合（你的原型是节点嵌入维度，因此可直接比较）。
    - 相似度采用论文 eq.(2)： sim(p,h) = log((||p-h||^2 + 1)/(||p-h||^2 + eps))

    返回:
      sub_nodes: List[int]     # 子图节点
      sub_edges: List[Tuple]   # 子图边（诱导边）
      sim: float               # 最终子图与原型的相似度
    """
    N, d = node_emb.size()
    p = prototype_node_emb.view(1, -1)  # [1, d]

    # ---- helpers ----
    def proto_sim(h: torch.Tensor) -> float:
        # h: [d]
        dist2 = torch.sum((h - prototype_node_emb) ** 2)
        return torch.log((dist2 + 1.0) / (dist2 + eps)).item()

    def pool_emb(idx: torch.Tensor) -> torch.Tensor:
        # idx: [k] 节点索引
        if idx.numel() == 0:
            return torch.zeros(d, device=node_emb.device)
        sub = node_emb.index_select(0, idx)  # [k, d]
        if pool == "sum":
            h = sub.sum(dim=0)
        else:
            h = sub.mean(dim=0)
        return h

    # 构建无向邻接表
    ei = edge_index
    if ei.numel() == 0:
        # 无边图：退化为选一个与原型最相似的单点
        sims = F.cosine_similarity(node_emb, p.expand_as(node_emb), dim=1)
        seed = int(torch.argmax(sims).item())
        hS = node_emb[seed]
        return [seed], [], proto_sim(hS)

    # 若边未对称，做对称
    src, dst = ei[0], ei[1]
    undirected = torch.cat([ei, ei.flip(0)], dim=1) if not torch.equal(src.sort()[0], dst.sort()[0]) else ei
    adj = [[] for _ in range(N)]
    for u, v in undirected.t().tolist():
        if v not in adj[u]:
            adj[u].append(v)
        if u not in adj[v]:
            adj[v].append(u)

    # 以与原型最相似的节点作为 seed
    seed_scores = F.cosine_similarity(node_emb, p.expand_as(node_emb), dim=1)  # [N]
    seed = int(torch.argmax(seed_scores).item())

    S = [seed]             # 当前子图节点（list 便于记录顺序）
    S_set = set(S)
    E_S = []               # 已选边（连接顺序）
    hS = node_emb[seed].clone()  # 当前子图嵌入
    if pool == "mean":
        # 维护累计向量与计数，便于 O(1) 更新
        sum_vec = node_emb[seed].clone()
        count = 1

    # 迭代扩展直到达到预算或没有可扩展邻居
    while len(S) < budget:
        # 候选：与 S 邻接的所有未入选节点
        cand = []
        for u in S:
            for v in adj[u]:
                if v not in S_set:
                    cand.append((u, v))
        if not cand:
            break

        best_gain = -1e9
        best_pair = None
        best_h_new = None

        # 评估每个 (u in S, v not in S) 的“加入 v”后相似度增益
        base_sim = proto_sim(hS)
        for (u, v) in cand:
            if pool == "mean":
                h_new = (sum_vec + node_emb[v]) / (count + 1)
            else:
                h_new = hS + node_emb[v]

            sim_new = proto_sim(h_new)
            gain = sim_new - base_sim
            if gain > best_gain:
                best_gain = gain
                best_pair = (u, v)
                best_h_new = h_new

        # 若所有增益都很差，也允许加入“提升最小但保持连通”的一个节点，保证连通采样
        if best_pair is None:
            break

        u, v = best_pair
        # 收纳 v 与 (u, v) 这条连接边
        S.append(v)
        S_set.add(v)
        E_S.append((u, v))
        hS = best_h_new
        if pool == "mean":
            sum_vec = sum_vec + node_emb[v]
            count += 1

    # 用诱导边作为最终高亮边（去重 & 无向）
    E_set = set()
    for (u, v) in undirected.t().tolist():
        if u in S_set and v in S_set:
            a, b = (u, v) if u <= v else (v, u)
            E_set.add((a, b))
    sub_edges = sorted(E_set)
    final_sim = proto_sim(hS)

    return S, sub_edges, final_sim
