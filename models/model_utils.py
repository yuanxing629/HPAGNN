import torch
import torch.nn as nn

from torch_geometric.nn.glob import global_mean_pool, global_add_pool, global_max_pool

def get_readout_layers(readout):
    readout_func_dict = {
        "mean": global_mean_pool,
        "sum": global_add_pool,
        "max": global_max_pool
    }
    readout_func_dict = {k.lower(): v for k, v in readout_func_dict.items()}
    ret_readout = []
    for k, v in readout_func_dict.items():
        if k in readout.lower():
            ret_readout.append(v)
    return ret_readout


def build_subgraph_adj(node_indices, edge_index, num_nodes):
    """
    给定一个图的 edge_index 和一个子图节点列表 node_indices，
    构造该子图的邻接矩阵（K×K），节点顺序按 node_indices 的顺序。
    """
    device = edge_index.device
    node_indices = list(node_indices)
    K = len(node_indices)
    idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(node_indices)}

    A = torch.zeros(K, K, device=device)
    src, dst = edge_index

    for u, v in zip(src.tolist(), dst.tolist()):
        if u in idx_map and v in idx_map:
            i = idx_map[u]
            j = idx_map[v]
            A[i, j] = 1.0
            A[j, i] = 1.0  # 无向图

    # 去掉自环
    eye = torch.eye(K, device=device)
    A = A * (1.0 - eye)
    return A


class GraphDecoder(nn.Module):
    """
    Prototype node embeddings -> graph:
      - inner-product edge decoder
      - node-type classifier head
      - optional graph embedding via external graph encoder, else READOUT

    Inputs
    ------
    prototype_node_emb: [N,d] or [B,N,d]
    tau: float or None         # Gumbel temperature; None -> plain sigmoid
    hard: bool                 # return STE hard adjacency
    use_soft_adj_for_encoder: bool  # True: 用 adj_soft 进 encoder；False: 用 adj_hard（若可用）
    graph_encoder: callable or None # (adj:[B,N,N], x:[B,N,F_enc]) -> [B,D]
    readout: str               # 'mean' | 'sum'  (当 graph_encoder is None 时使用)
    Returns (dict)
      node_feats:   [B,N,F_edge]   # 用于边解码/编码器的节点隐向量
      node_logits:  [B,N,C]        # 节点类型logits
      node_type_probs: [B,N,C]
      node_type_ids:   [B,N]
      edge_logits:  [B,N,N]
      edge_probs:   [B,N,N]
      adj_soft:     [B,N,N]
      adj_hard:     [B,N,N] or None
      graph_emb:    [B,D]          # 经 graph_encoder 或 READOUT 得到
    """

    def __init__(
            self,
            proto_dim: int,  # 原型节点嵌入维度 d
            hidden_dim: int = 128,  # 中间隐层维度
            edge_feat_dim: int = 128,  # 用于边解码/读出的节点特征维度
            num_node_classes: int = 7,  # 节点类型类别数（如 MUTAG 7类）
            use_bias: bool = True,
    ):
        super().__init__()
        self.proto_dim = proto_dim
        self.hidden_dim = hidden_dim
        self.edge_feat_dim = edge_feat_dim
        self.num_node_classes = num_node_classes

        # 基础 MLP: prototype_node_emb -> node_hidden
        self.backbone = nn.Sequential(
            nn.Linear(proto_dim, hidden_dim, bias=use_bias),
            nn.ReLU(inplace=True),
        )
        # 到用于边解码/编码器的节点隐向量
        self.edge_feat_head = nn.Linear(hidden_dim, edge_feat_dim, bias=use_bias)
        # 节点类型分类头（原子类别 logits）
        self.node_cls_head = nn.Linear(hidden_dim, num_node_classes, bias=use_bias)

        # 内积解码的可学习缩放
        self.logit_scale = nn.Parameter(torch.tensor(1.0))  # 训练过程中，logit_scale 会自动学习一个合适的温度

    @staticmethod
    def _ensure_batch(x):
        # 如果传进来的原型节点嵌入是[n, d](单图)，会把它变成[1, N, d]
        if x.dim() == 2:
            x = x.unsqueeze(0)
        return x

    @staticmethod
    def _sym_clean(A, symmetric=True, remove_self_loops=True):
        if symmetric:
            A = 0.5 * (A + A.transpose(-1, -2))  # 强制对称(无向图)
        if remove_self_loops:
            idx = torch.arange(A.size(-1), device=A.device)
            A[..., idx, idx] = 0.0  # 对角线全置0，去掉自环
        return A

    @staticmethod
    def _gumbel_sigmoid(logits, tau=1.0, hard=False, eps=1e-10):
        u = torch.rand_like(logits)  # u ~ U(0, 1)
        g = -torch.log(-torch.log(u + eps) + eps)
        y = torch.sigmoid((logits + g) / max(tau, 1e-6))
        if hard:  # straight-Through Estimator。正向是hard二值，反向梯度从soft y 传
            y_hard = (y > 0.5).float()
            y = y_hard.detach() - y.detach() + y
        return y

    def forward(
            self,
            prototype_node_emb: torch.Tensor,
            tau: float = None,
            hard: bool = False,
            symmetric: bool = True,
            remove_self_loops: bool = True,
            readout_layers=None,
            batch=None
    ):
        # 1) batch 统一
        H = self._ensure_batch(prototype_node_emb)  # [B,N,d]
        B, N, _ = H.shape

        # 2) 节点隐向量 + 两个头：edge特征 & 节点类型
        node_hidden = self.backbone(H)  # [B,N,H]
        node_feats = self.edge_feat_head(node_hidden)  # [B,N,F_edge]  用于边解码/编码器
        node_logits = self.node_cls_head(node_hidden)  # [B,N,C]
        node_type_probs = torch.softmax(node_logits, dim=-1)  # [B,N,C]
        node_type_ids = node_type_probs.argmax(dim=-1)  # [B,N]
        num_nodes = node_feats.size(1)

        # 3) 内积解码
        # 用内积表示节点之间建边的倾向
        edge_logits = torch.bmm(node_feats, node_feats.transpose(1, 2)) * self.logit_scale  # [B,N,N]
        if tau is None:
            edge_probs = torch.sigmoid(edge_logits)
        else:
            edge_probs = self._gumbel_sigmoid(edge_logits, tau=tau, hard=False)

        adj_soft = self._sym_clean(edge_probs, symmetric=symmetric, remove_self_loops=remove_self_loops)

        # 4) 硬邻接（可选）
        adj_hard = None
        if hard:
            if tau is None:
                y_hard = (adj_soft > 0.5).float()
                adj_hard = y_hard.detach() - adj_soft.detach() + adj_soft
            else:
                y = self._gumbel_sigmoid(edge_logits, tau=tau, hard=True)
                adj_hard = self._sym_clean(y, symmetric=symmetric, remove_self_loops=remove_self_loops)

        # 5) graph_emb：默认就做 READOUT；如果传入 readout_layers 就用它们并投到 proto_dim
        if readout_layers is None:
            # 默认：对 node_feats 做 mean 读出
            graph_emb = node_feats.mean(dim=1)  # [B, edge_feat_dim]
            # 如需强制等维到 proto_dim，可投影
            if graph_emb.size(-1) != self.proto_dim:
                if not hasattr(self, 'projection'):
                    self.projection = nn.Linear(graph_emb.size(-1), self.proto_dim).to(prototype_node_emb.device)
                graph_emb = self.projection(graph_emb)
        else:
            # 对 batch 里每个图做 pooling
            node_for_pool = node_feats  # 也可用 node_hidden
            node_feats_flat = node_for_pool.reshape(-1, node_for_pool.size(-1))  # [B, N, F] -> [B * N, F]
            if batch is None:
                batch_indices = torch.arange(B, device=prototype_node_emb.device).unsqueeze(1).repeat(1, N).reshape(-1)
            pooled = [readout(node_feats_flat, batch_indices) for readout in readout_layers]
            graph_emb = torch.cat(pooled, dim=-1)
            if graph_emb.size(-1) != self.proto_dim:
                if not hasattr(self, 'projection'):
                    self.projection = nn.Linear(graph_emb.size(-1), self.proto_dim).to(prototype_node_emb.device)
                graph_emb = self.projection(graph_emb)

        return {
            "node_feats": node_feats,
            "node_logits": node_logits,
            "node_type_probs": node_type_probs,
            "node_type_ids": node_type_ids,
            "edge_logits": edge_logits, # 重构损失L_rec对它做BCEWithLogits
            "edge_probs": edge_probs,
            "adj_soft": adj_soft,
            "adj_hard": adj_hard, # 用来构造PyG的edge_index（通过nonzero）；作为生成图的真实离散结构
            "graph_emb": graph_emb,  # 始终 [B, 128] 在alignment_loss中用它和原型的图级嵌入做MSE对齐
        }

