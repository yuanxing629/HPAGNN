import os
from typing import Iterable, List, Tuple, Dict, Optional

import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils.num_nodes import maybe_num_nodes
from textwrap import wrap
from torch_geometric.utils import to_networkx



# -----------------------------
# Helpers
# -----------------------------
def pyg_data_to_nx(data, is_undirected: bool = True) -> nx.Graph:
    """
    将 PyG 的 Data 转为 NetworkX Graph，节点索引为 0..N-1 的 Python int。
    边来自 edge_index，is_undirected=True 时去重并视为无向。
    """
    if hasattr(data, "x") and data.x is not None:
        N = int(data.x.size(0))
    else:
        N = int(maybe_num_nodes(data.edge_index))

    G = nx.Graph() if is_undirected else nx.DiGraph()
    G.add_nodes_from(range(N))

    ei = data.edge_index
    src = ei[0].detach().cpu().tolist()
    dst = ei[1].detach().cpu().tolist()
    edges = list(zip(src, dst))
    if is_undirected:
        # 去重 + 无向
        edges = list({tuple(sorted(e)) for e in edges})
    G.add_edges_from(edges)

    # 附加属性
    if hasattr(data, "x") and data.x is not None:
        x_cpu = data.x.detach().cpu()
        for i in range(N):
            G.nodes[i]["x"] = x_cpu[i]
    if hasattr(data, "y") and data.y is not None:
        G.graph["y"] = (
            int(data.y.detach().cpu().item()) if data.y.numel() == 1 else data.y.detach().cpu().tolist()
        )
    return G


def coerce_nodes_to_graph(graph: nx.Graph, nodes: Iterable) -> List:
    """将外部节点 id 强制映射/过滤到 graph 的真实节点标签集合。"""
    gnodes = set(graph.nodes())
    out = []
    if not gnodes:
        return out
    sample_type = type(next(iter(gnodes)))
    for n in nodes:
        try:
            nn = sample_type(n)
        except Exception:
            nn = n
        if nn in gnodes:
            out.append(nn)
    return out


def coerce_edges_to_graph(graph: nx.Graph, edges: Iterable[Tuple]) -> List[Tuple]:
    """将外部边列表强制映射/过滤到 graph 的节点集合。"""
    gnodes = set(graph.nodes())
    out = []
    if not gnodes:
        return out
    sample_type = type(next(iter(gnodes)))
    for (u, v) in edges:
        try:
            uu, vv = sample_type(u), sample_type(v)
        except Exception:
            uu, vv = u, v
        if uu in gnodes and vv in gnodes:
            out.append((uu, vv))
    return out


def compute_layout(graph: nx.Graph, pos: Optional[Dict] = None) -> Dict:
    """若未给布局，默认使用 kamada_kawai 布局。"""
    if pos is not None:
        return pos
    return nx.kamada_kawai_layout(graph)


def build_molecule_color_and_labels(dataset_name: str, x_tensor: Optional[torch.Tensor]):
    """
    为分子数据集构建节点颜色和标签映射（根据元素 one-hot）。
    返回两个 dict: node_id->color, node_id->label；若非分子数据集返回 (None, None)。
    """
    if x_tensor is None:
        return None, None

    dataset = dataset_name.lower()
    if dataset not in ("mutag", "mutagenicity", "bbbp", "clintox"):
        return None, None

    # 这里按 MUTAG / Mutagenicity 常见 one-hot 编码做映射；其它数据集不冲突也可用
    if dataset == "mutag":
        node_dict = {0: "C", 1: "N", 2: "O", 3: "F", 4: "I", 5: "Cl", 6: "Br"}
        palette = ["#F39C12", "#1ABC9C", "#E74C3C", "#9B59B6", "#F7DC6F", "#2ECC71", "#D35400"]
    elif dataset == "mutagenicity":
        node_dict = {
            0: "C",
            1: "O",
            2: "Cl",
            3: "H",
            4: "N",
            5: "F",
            6: "Br",
            7: "S",
            8: "P",
            9: "I",
            10: "Na",
            11: "K",
            12: "Li",
            13: "Ca",
        }
        palette = [
            "#F39C12",
            "#E74C3C",
            "#2ECC71",
            "#58D68D",
            "#1ABC9C",
            "#9B59B6",
            "#D35400",
            "#566573",
            "#F4D03F",
            "#FAD7A0",
            "#5DADE2",
            "#2E86C1",
            "#154360",
            "#1F618D",
        ]

    atom_types = np.where(x_tensor.detach().cpu().numpy() == 1)[1]
    node_labels = {i: node_dict.get(int(t), str(int(t))) for i, t in enumerate(atom_types)}
    node_colors = {i: palette[int(t) % len(palette)] for i, t in enumerate(atom_types)}
    return node_colors, node_labels

def _edge_logits_to_graph(edge_logits: torch.Tensor,
                          thresh: float = 0.30,
                          topk_edges: int | None = None,
                          symmetric: bool = True):
    """
    将 edge_logits ([N,N] 或 [1,N,N]) 转为一个 NetworkX 无向图（按概率取边）。
    - 若提供 topk_edges，则忽略 thresh，仅取概率最高的 K 条边（按上三角计数）。
    - symmetric=True 时对称化 (A + A^T)/2 并清零对角。
    返回: (G:nx.Graph, weights:dict[(u,v)->float])  其中权重为边概率
    """
    if edge_logits.dim() == 3:
        edge_logits = edge_logits[0]
    with torch.no_grad():
        A = torch.sigmoid(edge_logits.detach().cpu())  # [N,N] in [0,1]
    if symmetric:
        A = 0.5 * (A + A.t())
    N = A.size(0)
    A = A * (1 - torch.eye(N))  # 去掉对角

    # 取边
    edges = []
    weights = {}
    if topk_edges is not None and topk_edges > 0:
        # 仅取上三角，避免重复
        iu, iv = torch.triu_indices(N, N, offset=1)
        scores = A[iu, iv]
        k = min(topk_edges, iu.numel())
        vals, idx = torch.topk(scores, k)
        for s, u, v in zip(vals.tolist(), iu[idx].tolist(), iv[idx].tolist()):
            if s <= 0:
                continue
            e = (int(u), int(v))
            edges.append(e)
            weights[e] = float(s)
    else:
        # 阈值筛选（上三角）
        iu, iv = torch.triu_indices(N, N, offset=1)
        mask = (A[iu, iv] >= thresh)
        for u, v, s in zip(iu[mask].tolist(), iv[mask].tolist(), A[iu, iv][mask].tolist()):
            e = (int(u), int(v))
            edges.append(e)
            weights[e] = float(s)

    # 构图（仅包含出现在边中的节点）
    nodes = sorted(set([u for e in edges for u in e]))
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    return G, weights

# -----------------------------
# Plot Utils
# -----------------------------
class PlotUtils:
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name

    def save_graph(self, data, out_path):
        """直接绘制一个 PyG Data 对象"""
        G = to_networkx(data, to_undirected=True)

        # 获取节点标签用于着色 (假设是 One-hot x)
        labels = None
        if data.x is not None:
            labels = data.x.argmax(dim=1).tolist()

        # 绘图逻辑 (复用你现有的 style)
        plt.figure(figsize=(6, 6))
        # ... (你的绘图代码, nx.draw 等)
        # 例如:
        pos = nx.kamada_kawai_layout(G)
        nx.draw(G, pos, node_color=labels, cmap=plt.cm.Set2, with_labels=False)
        plt.savefig(out_path)
        plt.close()

    def plot_molecule(
            self,
            graph: nx.Graph,
            nodelist: Iterable,
            x: Optional[torch.Tensor],
            edgelist: Optional[Iterable[Tuple]] = None,
            figname: Optional[str] = None,
            title: Optional[str] = None,
            *,
            # —— 想把非高亮边变浅：把 base_edge_alpha 调低、或 base_edge_color 改成浅灰
            base_edge_color="#B0BEC5",
            base_edge_width: float = 2.0,
            base_edge_alpha: float = 0.5,
            highlight_edge_color: str = "#000000",
            highlight_edge_width: float = 4.5,
            highlight_edge_alpha: float = 0.95,
            node_size: int = 420,
            node_edgecolor: str = "#2C3E50",
            node_linewidth: float = 1.2,
            highlight_node_edgecolor: str = "#000000",
            highlight_node_linewidth: float = 1.8,
    ):
        """分子图：整图按元素配色，只把原型子图边高亮加粗；子图节点描边更粗。"""
        # 颜色与标签（元素）
        node_colors_map, node_labels_map = build_molecule_color_and_labels(self.dataset_name, x)
        # 进入统一绘制
        self.draw_full_graph_with_highlight(
            graph=graph,
            highlight_nodes=nodelist,
            highlight_edges=edgelist,
            node_color_map=node_colors_map,
            node_labels_map=node_labels_map,
            figname=figname,
            title_sentence=title,
            base_edge_color=base_edge_color,
            base_edge_width=base_edge_width,
            base_edge_alpha=base_edge_alpha,
            highlight_edge_color=highlight_edge_color,
            highlight_edge_width=highlight_edge_width,
            highlight_edge_alpha=highlight_edge_alpha,
            node_size=node_size,
            node_edgecolor=node_edgecolor,
            node_linewidth=node_linewidth,
            highlight_node_edgecolor=highlight_node_edgecolor,
            highlight_node_linewidth=highlight_node_linewidth,
        )

    def plot_subgraph(
            self,
            graph: nx.Graph,
            nodelist: Iterable,
            edgelist: Optional[Iterable[Tuple]] = None,
            figname: Optional[str] = None,
            title: Optional[str] = None,
            *,
            base_edge_color: str = "#B0BEC5",
            base_edge_width: float = 2.0,
            base_edge_alpha: float = 0.6,
            highlight_edge_color: str = "#37474F",
            highlight_edge_width: float = 4.0,
            highlight_edge_alpha: float = 0.95,
            node_size: int = 360,
            node_edgecolor: str = "#263238",
            node_linewidth: float = 1.2,
            highlight_node_edgecolor: str = "#000000",
            highlight_node_linewidth: float = 1.8,
    ):
        """通用图：整图基线样式 + 高亮子图边；子图节点描边更粗。
           对于合成数据集（如 BA_2Motifs），所有节点统一为 #FF8C00，仅高亮边黑色加粗。
        """
        SYN_ONE_TYPE = {"ba_2motifs", "ba_3motifs", "ba_shapes", "ba_grid", "ba", "synthetic"}

        is_synth = self.dataset_name.lower() in SYN_ONE_TYPE
        if is_synth:
            # 合成数据集：所有节点一种颜色
            node_color_map = {n: "#FF8C00" for n in graph.nodes()}
            highlight_edge_color = "#000000"  # 边用黑色加粗
        else:
            # 非合成：保留默认的统一淡蓝基色
            node_color_map = {n: "#90CAF9" for n in graph.nodes()}

        node_labels_map = None  # 合成数据集一般无多类型标签

        self.draw_full_graph_with_highlight(
            graph=graph,
            highlight_nodes=nodelist,
            highlight_edges=edgelist,
            node_color_map=node_color_map,
            node_labels_map=node_labels_map,
            figname=figname,
            title_sentence=title,
            base_edge_color=base_edge_color,
            base_edge_width=base_edge_width,
            base_edge_alpha=base_edge_alpha,
            highlight_edge_color=highlight_edge_color,
            highlight_edge_width=highlight_edge_width,
            highlight_edge_alpha=highlight_edge_alpha,
            node_size=node_size,
            node_edgecolor=node_edgecolor,
            node_linewidth=node_linewidth,
            highlight_node_edgecolor=highlight_node_edgecolor,
            highlight_node_linewidth=highlight_node_linewidth,
        )

    # ------ Core drawer ------
    def draw_full_graph_with_highlight(
            self,
            graph: nx.Graph,
            highlight_nodes: Iterable,
            highlight_edges: Optional[Iterable[Tuple]] = None,
            *,
            node_color_map: Optional[Dict] = None,  # node_id -> color
            node_labels_map: Optional[Dict] = None,  # node_id -> label
            title_sentence: Optional[str] = None,
            figname: Optional[str] = None,
            dpi: int = 220,
            pos: Optional[Dict] = None,
            # 边样式：整图（基础）与子图（高亮）
            base_edge_color: str = "#2F4F4F",
            base_edge_width: float = 2.0,
            base_edge_alpha: float = 0.65,
            highlight_edge_color: str = "#FF6A00",
            highlight_edge_width: float = 4.0,
            highlight_edge_alpha: float = 0.95,
            # 节点样式
            node_size: int = 400,
            node_edgecolor: str = "#2C3E50",
            node_linewidth: float = 1.2,
            highlight_node_edgecolor: str = "#000000",
            highlight_node_linewidth: float = 1.8,
    ):
        """
        核心绘制：先画整图（节点保持原色、边为基础样式），再覆盖绘制“高亮边”，
        最后对子图节点加粗描边（颜色可不同）。不对整图进行“降灰”，符合“只强调原型边”的需求。
        """
        # --- 校正 & 过滤 高亮节点/边 ---
        H_nodes = set(coerce_nodes_to_graph(graph, highlight_nodes))
        if highlight_edges is None:
            # 诱导边
            H_edges = [(u, v) for (u, v) in graph.edges() if u in H_nodes and v in H_nodes]
        else:
            H_edges = coerce_edges_to_graph(graph, highlight_edges)

        # --- 布局 ---
        pos = compute_layout(graph, pos)

        # --- 画整图边（基础样式） ---
        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=list(graph.edges()),
            edge_color=base_edge_color,
            width=base_edge_width,
            alpha=base_edge_alpha,
            arrows=False,
        )

        # --- 画整图节点（保持原始配色） ---
        if node_color_map is None:
            # 未给颜色映射：统一淡色；如果想保持之前颜色，可自行传入 node_color_map
            node_colors = ["#F2F4F4"] * graph.number_of_nodes()
            node_list = list(graph.nodes())
        else:
            node_list = list(graph.nodes())
            node_colors = [node_color_map.get(n, "#F2F4F4") for n in node_list]

        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=node_list,
            node_color=node_colors,
            node_size=node_size,
            linewidths=node_linewidth,
            edgecolors=node_edgecolor,
        )

        # --- 覆盖绘制“高亮边”（只把原型子图边加深/加粗） ---
        if H_edges:
            nx.draw_networkx_edges(
                graph,
                pos,
                edgelist=H_edges,
                edge_color=highlight_edge_color,
                width=highlight_edge_width,
                alpha=highlight_edge_alpha,
                arrows=False,
            )

        # --- 子图节点：更粗描边以提示被匹配 ---
        if len(H_nodes) > 0:
            if node_color_map:
                hl_colors = [node_color_map.get(n, "#F2F4F4") for n in H_nodes]
            else:
                hl_colors = "#F2F4F4"

            nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=list(H_nodes),
                node_color=hl_colors,
                node_size=node_size,
                linewidths=highlight_node_linewidth,
                edgecolors=highlight_node_edgecolor,
            )

        # --- 全图标签（显示所有节点类型，例如 C、O、N） ---
        if node_labels_map:
            nx.draw_networkx_labels(
                graph,
                pos,
                labels=node_labels_map,
                font_size=9,
                font_color="#1C1C1C",
                font_weight="bold"
            )

        # --- 题注 & 保存 ---
        plt.axis("off")
        if title_sentence:
            plt.title("\n".join(wrap(title_sentence, width=60)))

        if figname is not None:
            os.makedirs(os.path.dirname(figname), exist_ok=True)
            plt.savefig(figname, dpi=dpi, bbox_inches="tight")
        plt.close("all")

    def save_full_graph_highlight(self, data, highlight_nodes, highlight_edges, out_png, title=None):
        """将整图画出来，仅对子图的边加深，节点描边加粗；支持分子数据集标签/着色。"""
        G = pyg_data_to_nx(data, is_undirected=True)
        x = data.x if hasattr(data, 'x') else None

        # 复用已有分子绘图，传入 edgelist 即只强调这些边
        if self.dataset_name.lower() in ['bbbp', 'mutag', 'clintox', 'mutagenicity']:
            self.plot_molecule(G, nodelist=list(highlight_nodes), x=x, edgelist=list(highlight_edges), figname=out_png)
        elif self.dataset_name.lower() in ['ba_2motifs']:
            self.plot_subgraph(G,nodelist=list(highlight_nodes),edgelist=list(highlight_edges),figname=out_png)
        else:
            # 通用图：整图背景 + 子图高亮
            self.draw_full_graph_with_highlight(
                graph=G,
                highlight_nodes=list(highlight_nodes),
                highlight_edges=list(highlight_edges),
                figname=out_png,
                title_sentence=title
            )
        return out_png

    def save_generated_prototype_png(self,
                                     edge_logits: torch.Tensor,
                                     out_png: str,
                                     *,
                                     title: str | None = None,
                                     thresh: float = 0.30,
                                     topk_edges: int | None = None,
                                     symmetric: bool = True):
        """
        将“生成的原型”的 edge_logits 渲染为独立的小图并保存。
        注意：这不是在真实图上高亮，而是把生成原型本身画出来。
        """
        G, weights = _edge_logits_to_graph(edge_logits, thresh=thresh,
                                           topk_edges=topk_edges, symmetric=symmetric)

        plt.figure(figsize=(4.8, 4.2), dpi=220)

        # 布局
        if G.number_of_nodes() == 0:
            # 空图兜底：给一张空白图
            plt.axis('off')
            if title:
                plt.title("\n".join(wrap(title, width=60)))
            os.makedirs(os.path.dirname(out_png), exist_ok=True)
            plt.savefig(out_png, dpi=220, bbox_inches='tight')
            plt.close()
            return out_png

        pos = nx.kamada_kawai_layout(G)

        # 节点颜色：合成数据集（如 BA_2Motifs）统一 #FF8C00；其他统一浅蓝
        ds = self.dataset_name.lower()
        if ds in ("ba_2motifs", "ba_3motifs", "ba_shapes", "ba_grid"):
            node_color = "#FF8C00"
        else:
            node_color = "#90CAF9"

        nx.draw_networkx_nodes(
            G, pos,
            nodelist=list(G.nodes()),
            node_color=node_color,
            node_size=420,
            linewidths=1.6,
            edgecolors="#000000",
        )

        # 边：黑色，线宽随概率变化
        if G.number_of_edges() > 0:
            widths = []
            for e in G.edges():
                w = weights.get(tuple(sorted(e)), 0.5)
                widths.append(1.0 + 4.0 * float(w))
            nx.draw_networkx_edges(
                G, pos,
                edgelist=list(G.edges()),
                edge_color="#000000",
                width=widths,
                alpha=0.95
            )

        # 不显示节点标签（生成原型没有元素类型）
        plt.axis('off')
        if title:
            plt.title("\n".join(wrap(title, width=60)))
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        plt.savefig(out_png, dpi=220, bbox_inches='tight')
        plt.close()
        return out_png

    def save_generated_prototype_from_adj(self,
                                          adj_hard: torch.Tensor,
                                          adj_soft: torch.Tensor | None,
                                          out_png: str,
                                          *,
                                          title: str | None = None):
        """
        基于生成原型的邻接矩阵(硬/软)绘制独立小图。
        - adj_hard: [N, N] in {0,1}
        - adj_soft: [N, N] in [0,1] (可选，用作边权重)
        """



        A = adj_hard.detach().cpu()
        N = A.size(0)

        # 构建 graph
        G = nx.Graph()
        G.add_nodes_from(range(N))

        weights = {}
        for i in range(N):
            for j in range(i + 1, N):
                if A[i, j] > 0:
                    G.add_edge(i, j)
                    if adj_soft is not None:
                        w = float(adj_soft[i, j].item())
                    else:
                        w = 1.0
                    weights[(i, j)] = w

        plt.figure(figsize=(4.8, 4.2), dpi=220)

        if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
            plt.axis('off')
            if title:
                plt.title("\n".join(wrap(title, width=60)))
            os.makedirs(os.path.dirname(out_png), exist_ok=True)
            plt.savefig(out_png, dpi=220, bbox_inches='tight')
            plt.close()
            return out_png

        pos = nx.kamada_kawai_layout(G)

        # 节点颜色策略
        ds = self.dataset_name.lower()
        if ds in ("ba_2motifs", "ba_3motifs", "ba_shapes", "ba_grid"):
            node_color = "#FF8C00"
        else:
            node_color = "#90CAF9"

        nx.draw_networkx_nodes(
            G, pos,
            nodelist=list(G.nodes()),
            node_color=node_color,
            node_size=420,
            linewidths=1.6,
            edgecolors="#000000",
        )

        # 边：线宽随权重变化
        widths = []
        for e in G.edges():
            w = weights.get(tuple(sorted(e)), 0.5)
            widths.append(1.0 + 4.0 * float(w))
        nx.draw_networkx_edges(
            G, pos,
            edgelist=list(G.edges()),
            edge_color="#000000",
            width=widths,
            alpha=0.95,
        )

        plt.axis('off')
        if title:
            plt.title("\n".join(wrap(title, width=60)))
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        plt.savefig(out_png, dpi=220, bbox_inches='tight')
        plt.close()
        return out_png

    def save_standalone_graph(self, data, out_path):
        """
        专门用于绘制生成的 Data 对象（不依赖背景大图）
        """


        # 转换为 networkx
        G = to_networkx(data, to_undirected=True)

        # 移除自环 (可选，生成图可能带有自环)
        G.remove_edges_from(nx.selfloop_edges(G))

        plt.figure(figsize=(6, 6))

        # 节点颜色
        ds = self.dataset_name.lower()
        if ds in ("ba_2motifs", "ba_3motifs", "ba_shapes", "ba_grid"):
            node_colors = "#FF8C00"
        else:
            node_colors = "#90CAF9"

        # 布局
        pos = nx.kamada_kawai_layout(G)
        if len(G.nodes) == 0:
            # 防止空图报错
            plt.close()
            return

        # 绘图
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap=plt.cm.Set2, node_size=300)
        nx.draw_networkx_edges(G, pos, width=2.0, alpha=0.8)

        # 如果需要显示 labels (原子类型等)，可以在这里添加 logic
        # labels = {i: str(i) for i in G.nodes()}
        # nx.draw_networkx_labels(G, pos, labels=labels)

        plt.axis('off')
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
