import torch
import torch.nn.functional as F
from Configures import model_args


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


# def greedy_match_prototype_on_graph(
#         node_emb,  # [N, d] 该真实图的节点嵌入
#         edge_index,  # [2, E]
#         prototype_node_emb: torch.Tensor,  # [M, d] 这个原型的节点嵌入 (graph_size = M)
#         ensure_connected: bool = True,
# ):
#     """
#     在一张真实图上，用贪心方法搜索与某个“节点级原型”最相似的子图。
#     返回:
#       - best_sub_nodes: List[int]      最相似子图的节点索引集合
#       - best_sim: float                原型图嵌入 vs 子图图嵌入 的相似度
#       - best_sub_node_emb: [K, d]      该子图的节点嵌入 (K <= graph_size)
#       - best_sub_graph_emb: [d]        该子图的图嵌入 (简单 mean-pool 或 readout)
#     """
#
#     N, d = node_emb.size()  # N 为该真实图的节点个数
#     M = prototype_node_emb.size(0)  # M 为该原型的节点个数
#     graph_size = model_args.graph_size
#     if graph_size is None:
#         graph_size = M
#     graph_size = min(graph_size, N)  # 子图节点数不可能超过整图
#
#     # 1) 原型图嵌入: 对原型节点嵌入做 READOUT
#     if model_args.readout == 'max':
#         proto_graph_emb = prototype_node_emb.max(dim=0)[0]
#     elif model_args.readout == 'sum':
#         proto_graph_emb = prototype_node_emb.sum(dim=0)
#     else:
#         proto_graph_emb = prototype_node_emb.mean(dim=0)
#
#     proto_graph_emb = F.normalize(proto_graph_emb, dim=0)  # 归一化
#
#     # 2) 归一化真实图节点嵌入 & 预先算好每个节点与原型的单点相似度
#     H = node_emb  # [N, d]
#     H_norm = F.normalize(H, dim=1)  # [N, d]
#     node_scores = torch.mv(H_norm, proto_graph_emb)  # [N]
#
#     # 3) 建邻接表 (list 形式)，用于贪心扩展保持连通性
#     src, dst = edge_index
#     src = src.detach().cpu().tolist()
#     dst = dst.detach().cpu().tolist()
#     adj_list = [[] for _ in range(N)]
#     for u, v in zip(src, dst):
#         adj_list[u].append(v)
#         adj_list[v].append(u)
#
#     best_sim = -1e9
#     best_sub_nodes = None
#     best_sub_node_emb = None
#     best_sub_graph_emb = None
#
#     # 4) 以每个节点为起点，贪心扩展一个大小 ≤ graph_size 的连通子图
#     for v0 in range(N):
#         sub_nodes = {v0}
#         frontier = {v0}
#
#         while len(sub_nodes) < graph_size and len(frontier) > 0:
#             # 收集当前 frontier 的一阶邻居作为候选
#             candidates = set()
#             for u in frontier:
#                 for w in adj_list[u]:
#                     if w not in sub_nodes:
#                         candidates.add(w)
#
#             if not candidates:
#                 # fallback：连通区域扩不动了，但 sub_nodes 还没到 graph_size
#                 # 从整个图剩余的节点里，挑相似度最高的若干个补齐
#                 remaining = [idx for idx in range(N) if idx not in sub_nodes]
#                 if not remaining:
#                     break  # 图里已经没有可用节点了
#
#                 needed = graph_size - len(sub_nodes)
#                 remaining_sorted = sorted(
#                     remaining, key=lambda idx: float(node_scores[idx]), reverse=True
#                 )
#                 for idx in remaining_sorted[:needed]:
#                     sub_nodes.add(idx)
#                 # 补齐之后就可以退出 while
#                 break
#
#             # 在候选中选出 node_scores 最高的一个点加入子图
#             best_cand = max(candidates, key=lambda idx: float(node_scores[idx]))
#             sub_nodes.add(best_cand)
#
#             if ensure_connected:
#                 frontier = {best_cand}
#             else:
#                 frontier.add(best_cand)
#
#         # 得到当前起点对应的候选子图节点集合
#         sub_nodes_list = sorted(list(sub_nodes))
#         H_sub = H[sub_nodes_list]  # [K, d]
#
#         # 5) 对子图节点嵌入做 READOUT 得到子图图嵌入 [d]
#         if model_args.readout == 'max':
#             sub_graph_emb = H_sub.max(dim=0)[0]
#         elif model_args.readout == 'sum':
#             sub_graph_emb = H_sub.sum(dim=0)
#         else:
#             sub_graph_emb = H_sub.mean(dim=0)
#
#         sub_graph_emb = F.normalize(sub_graph_emb, dim=0)
#
#         # 6) 和原型图嵌入做相似度（cos）
#         sim_tensor, _ = calculate_similarity(proto_graph_emb, sub_graph_emb)
#         sim = float(sim_tensor.squeeze().item())
#
#         # 7) 记录最优子图
#         if sim > best_sim:
#             best_sim = sim
#             best_sub_nodes = sub_nodes_list
#             best_sub_node_emb = H_sub
#             best_sub_graph_emb = sub_graph_emb
#
#     return best_sub_nodes, best_sim, best_sub_node_emb, best_sub_graph_emb

@torch.no_grad()
def match_connected_subgraph(
    node_emb: torch.Tensor,            # [N, d] 真实图节点嵌入
    edge_index: torch.Tensor,          # [2, E]
    prototype_node_emb: torch.Tensor,  # [M, d]（或 [graph_size, d]）
):
    """
    连通约束的贪心匹配（单源多边界法）：
    - 初始化：选全图与原型图最相似的节点作为起点；
    - 迭代：每步只从当前子图的“边界邻居”中选一个分数最高的节点加入；
    - 若边界为空则停止（保持连通），选到 <= graph_size 个节点。
    返回：
      sub_nodes: List[int]（连通）
      sub_edges: List[(u,v)]（诱导边）
      sim: float  （原型图嵌入 vs 匹配子图嵌入 的余弦相似）
    """
    N, d = node_emb.size()
    device = node_emb.device
    graph_size = max(1, min(int(model_args.graph_size), N))
    readout = model_args.readout

    # 1) 原型图嵌入
    if readout == 'max':
        proto_graph_emb = prototype_node_emb.max(dim=0)[0]
    elif readout == 'sum':
        proto_graph_emb = prototype_node_emb.sum(dim=0)
    else:
        proto_graph_emb = prototype_node_emb.mean(dim=0)
    proto_graph_emb = F.normalize(proto_graph_emb, dim=0)  # [d]

    # 2) 节点与原型图的相似得分（余弦）
    H = node_emb
    H_norm = F.normalize(H, dim=1)        # [N, d]
    node_scores = torch.mv(H_norm, proto_graph_emb)  # [N]

    # 3) 邻接表
    src, dst = edge_index
    src = src.detach().cpu().tolist()
    dst = dst.detach().cpu().tolist()
    adj_list = [[] for _ in range(N)]
    for u, v in zip(src, dst):
        if u == v:
            continue
        adj_list[u].append(v)
        adj_list[v].append(u)

    # 4) 初始化：选分数最高的节点作为起点
    start = int(torch.argmax(node_scores).item())
    sub_nodes = {start}

    # 5) 连通贪心生长：每次只从“子图的所有边界邻居”里挑最高分的一个
    while len(sub_nodes) < graph_size:
        candidates = set()
        for u in list(sub_nodes):
            for w in adj_list[u]:
                if w not in sub_nodes:
                    candidates.add(w)
        if not candidates:
            break
        best_cand = max(candidates, key=lambda idx: float(node_scores[idx]))
        sub_nodes.add(best_cand)

    sub_nodes_list = sorted(list(sub_nodes))
    # 诱导边
    edges_set = set()
    for u in sub_nodes_list:
        for v in adj_list[u]:
            if v in sub_nodes and u < v:
                edges_set.add((u, v))
    sub_edges_list = sorted(list(edges_set))

    # 6) 匹配子图的图嵌入 & 与原型图相似
    H_sub = H[sub_nodes_list]  # [K', d]
    if readout == 'max':
        sub_graph_emb = H_sub.max(dim=0)[0]
    elif readout == 'sum':
        sub_graph_emb = H_sub.sum(dim=0)
    else:
        sub_graph_emb = H_sub.mean(dim=0)
    sub_graph_emb = F.normalize(sub_graph_emb, dim=0)
    sim,_ = calculate_similarity(proto_graph_emb,sub_graph_emb)
    sim = float(sim.squeeze().item())

    return sub_nodes_list, sub_edges_list, sim
