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




class GraphDecoder(nn.Module):
    def __init__(
            self,
            proto_dim: int,              # 原型节点嵌入维度 d
            num_node_features: int,      # 数据集原始节点特征维度
            hidden_dim: int = 128,       # 中间隐层维度
            use_bias: bool = True,
    ):
        super().__init__()
        self.proto_dim = proto_dim
        self.hidden_dim = hidden_dim
        self.num_node_features = num_node_features

        # 两层 MLP：H̃ -> 节点属性 X̃
        self.node_mlp = nn.Sequential(
            nn.Linear(proto_dim, hidden_dim, bias=use_bias),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_node_features, bias=use_bias),
        )

        # 内积解码的可学习缩放
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

        # graph_emb 维度对齐到 proto_dim 时用
        self.projection = None


    @staticmethod
    def _sym_clean(A, symmetric=True, remove_self_loops=True):
        if symmetric:
            A = 0.5 * (A + A.transpose(-1, -2))
        if remove_self_loops:
            idx = torch.arange(A.size(-1), device=A.device)
            A[..., idx, idx] = 0.0
        return A

    def forward(
        self,
        prototype_node_emb: torch.Tensor,
        hard: bool = False,      # 这里的 hard 只决定是否直接二值化；Tl/Th 在外面做
        readout_layers=None,
        batch=None,
    ):
        # 1) batch 统一: [B, N, d]
        # 若输入 [N, d] -> [1, N, d]
        if prototype_node_emb.dim() == 2:
            prototype_node_emb = prototype_node_emb.unsqueeze(0)
        H = prototype_node_emb
        B, N, _ = H.shape

        # 2) 两层 MLP 生成节点属性（重构 X）
        node_feats = self.node_mlp(H)            # [B, N, F_x], F_x = num_node_features

        # 3) 内积解码得到边 logits / 概率
        edge_logits = torch.bmm(node_feats, node_feats.transpose(1, 2)) * self.logit_scale  # [B, N, N]
        edge_probs = torch.sigmoid(edge_logits)

        adj_soft = self._sym_clean(edge_probs) # 对称化&移除自环

        # 4) 可选：粗硬阈值（真正的 Tl/Th 过滤我们在外面用 init_graph 再做一次）
        adj_hard = None
        if hard:
            y_hard = (adj_soft > 0.5).float()
            adj_hard = y_hard.detach() - adj_soft.detach() + adj_soft

        # 5) graph_emb：对 node_feats 做 READOUT，再投到 proto_dim
        if readout_layers is None:
            graph_emb = node_feats.mean(dim=1)   # [B, F_x]
        else:
            node_feats_flat = node_feats.reshape(-1, node_feats.size(-1))  # [B*N, F_x]
            if batch is None:
                batch_indices = torch.arange(B, device=H.device).unsqueeze(1).repeat(1, N).reshape(-1)
            pooled = [readout(node_feats_flat, batch_indices) for readout in readout_layers]
            graph_emb = torch.cat(pooled, dim=-1)

        if graph_emb.size(-1) != self.proto_dim:
            if self.projection is None:
                self.projection = nn.Linear(graph_emb.size(-1), self.proto_dim).to(H.device)
            graph_emb = self.projection(graph_emb)

        return {
            "node_feats": node_feats,   # 用于节点属性重构 + 生成
            "edge_logits": edge_logits,
            "edge_probs": edge_probs,
            "adj_soft": adj_soft,
            "adj_hard": adj_hard,       # 粗硬邻接（可不用）
            "graph_emb": graph_emb,     # 对齐到 proto_dim 的图级嵌入
        }

