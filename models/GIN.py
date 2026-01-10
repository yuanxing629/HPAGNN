import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GINConv
from torch_geometric.data import Data, Batch
from my_search import differentiable_search_subgraph
from models.model_utils import get_readout_layers
from my_init import extract_connected_subgraph, extract_connected_subgraph2
import copy
from torch_geometric.utils import subgraph
import networkx as nx
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min


# GIN
class GINNet(nn.Module):
    def __init__(self, input_dim, output_dim, model_args):
        super(GINNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.latent_dim = model_args.latent_dim
        self.mlp_hidden = model_args.mlp_hidden
        self.emb_normalize = model_args.emb_normalize
        self.device = torch.device('cuda:' + str(model_args.device))
        self.num_gnn_layers = len(self.latent_dim)
        self.num_mlp_layers = len(self.mlp_hidden) + 1
        self.dense_dim = self.latent_dim[-1]
        self.readout_layers = get_readout_layers(model_args.readout)
        self.readout_name = model_args.readout.lower()

        self.gnn_layers = nn.ModuleList()
        self.gnn_layers.append(GINConv(nn.Sequential(
            nn.Linear(input_dim, self.latent_dim[0], bias=False),
            nn.BatchNorm1d(self.latent_dim[0]),
            nn.ReLU(),
            nn.Linear(self.latent_dim[0], self.latent_dim[0], bias=False),
            nn.BatchNorm1d(self.latent_dim[0])),
            train_eps=True))

        for i in range(1, self.num_gnn_layers):
            self.gnn_layers.append(GINConv(nn.Sequential(
                nn.Linear(self.latent_dim[i - 1], self.latent_dim[i], bias=False),
                nn.BatchNorm1d(self.latent_dim[i]),
                nn.ReLU(),
                nn.Linear(self.latent_dim[i], self.latent_dim[i], bias=False),
                nn.BatchNorm1d(self.latent_dim[i])),
                train_eps=True)
            )

        self.gnn_non_linear = nn.ReLU()

        self.Softmax = nn.Softmax(dim=-1)
        self.mlp_non_linear = nn.ELU()

        # prototype layers
        self.epsilon = 1e-4
        self.proto_dim = self.dense_dim * len(self.readout_layers)  # graph_data embedding dim
        self.prototype_shape = (output_dim * model_args.num_prototypes_per_class, 128)

        self.num_prototypes_per_class = model_args.num_prototypes_per_class

        # 先初始化为随机，后续用 initialize_prototypes_based_on_confidence来进行初始化
        # 只指定prototype_node_emb，图嵌入通过READOUT来算出
        self.num_prototypes = self.prototype_shape[0]

        self.last_layer = nn.Linear(self.num_prototypes, output_dim,
                                    bias=False)  # do not use bias

        assert (self.num_prototypes % output_dim == 0)

        # an onehot indication matrix for each prototype's class identity
        self.prototype_class_identity = torch.zeros(self.num_prototypes,
                                                    output_dim)
        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // model_args.num_prototypes_per_class] = 1

        # used for graph_decoder
        self.node_mlp = nn.Sequential(
            nn.Linear(self.proto_dim, self.dense_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(self.dense_dim, self.input_dim, bias=True),
        )
        # 内积解码的可学习缩放
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

        # 生成候选的存储
        self.generated_candidates = None  # 生成候选
        self.max_gen_nodes = model_args.max_gen_nodes

        # initialize the last layer
        self.set_last_layer_incorrect_connection(incorrect_strength=-0.5)

        # 原型节点嵌入用 ParameterList 存
        self.prototype_node_emb = nn.ParameterList()  # 先空着，初始化时再填

        # [新增] 存储每个原型对应的“源图”，用于后续的动态搜索
        self.prototype_source_graphs = []  # List[Data]

        # [修改] init_proto_graphs 仅用于记录最开始的状态，实际运算使用 source graphs
        self.init_proto_graphs = None

        # self.init_proto_graphs = None
        # self.init_proto_selection_info = None
        self.init_proto_node_list = None
        self.init_proto_edge_list = None
        # self.init_proto_in_which_graph = None
        self.init_info_dict = None
        self.anchor_graph_emb = None
        self.init_proto_graph_emb = None

        self.min_nodes = model_args.min_nodes
        self.max_nodes = model_args.max_nodes

    def proto_node_emb_2_graph_emb(self):
        """
        将每个原型的节点嵌入 [N_p, d] 通过 mean/sum/max 聚合成图级嵌入 [C*m, d]。
        注意：self.prototype_node_emb 是 ParameterList
        """
        proto_graph_emb_list = []

        for proto_nodes in self.prototype_node_emb:  # [C*m,_,d]
            # proto_nodes: [N_p, d]
            if 'max' in self.readout_name:
                g = proto_nodes.max(dim=0)[0]
            elif 'sum' in self.readout_name:
                g = proto_nodes.sum(dim=0)
            else:
                g = proto_nodes.mean(dim=0)
            proto_graph_emb_list.append(g)

        proto_graph_emb = torch.stack(proto_graph_emb_list, dim=0)  # [C*m, d]

        # 确保原型向量与图嵌入处于同一个单位球面空间
        if self.emb_normalize:
            proto_graph_emb = F.normalize(proto_graph_emb, p=2, dim=-1)

        return proto_graph_emb

    # 初始化原型
    @torch.no_grad()
    def initialize_prototypes(self, trainloader, classifier):
        """
        按类收集样本 -> 聚类 -> 选取最靠近簇中心的样本
        """
        device = self.device
        classifier.to(device)
        classifier.eval()

        m = self.num_prototypes_per_class
        num_classes = self.output_dim

        # 分别存储预测正确和所有样本
        correct_candidates = {c: [] for c in range(num_classes)}
        all_candidates = {c: [] for c in range(num_classes)}

        for batch in trainloader:
            batch = batch.to(device)
            logits, probs, node_emb, graph_emb = classifier(batch)
            print(probs)

            # graph_emb: [B, d], node_emb: [total_nodes_in_batch, d]
            data_list = batch.to_data_list()

            probs_cpu = probs.detach().cpu()
            node_emb_cpu = node_emb.detach().cpu()
            graph_emb_cpu = graph_emb.detach().cpu()
            batch_indices = batch.batch.detach().cpu()

            for i, data_i in enumerate(data_list):
                y_true = int(batch.y[i].item())
                pred = int(probs_cpu[i].argmax().item())
                conf = float(probs_cpu[i][y_true].item())

                # 提取当前图的节点嵌入
                batch_mask = (batch_indices == i)
                current_node_emb = node_emb_cpu[batch_mask]
                current_graph_emb = graph_emb_cpu[i]

                tup = {
                    'graph_emb': current_graph_emb.clone(),
                    'node_emb': current_node_emb.clone(),
                    'graph_data': data_i.clone(),
                    'conf': conf
                }

                all_candidates[y_true].append(tup)
                if pred == y_true:
                    correct_candidates[y_true].append(tup)

        selected_items = []

        # 对每个类进行K-Means聚类
        for c in range(num_classes):
            # 策略：优先用预测正确的；若不够 m 个，退化到使用该类所有样本
            pool = correct_candidates[c]
            if len(pool) < m:
                print(f"Warning: Class {c} has only {len(pool)} correct samples. Falling back to all samples.")
                pool = all_candidates[c]

            # 极端情况检查
            if len(pool) == 0:
                raise ValueError(f"Class {c} has NO samples in the training set!")

            # 聚类条件检查
            if len(pool) >= m:
                embeddings = np.array([item['graph_emb'] for item in pool])
                if self.emb_normalize:
                    # L2 归一化，确保 K-Means 是在“角度”上聚类
                    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                    embeddings = embeddings / (norms + 1e-9)
                # 显式重塑为 2D 数组防止 sklearn 报错
                if embeddings.ndim == 1:
                    embeddings = embeddings.reshape(-1, 1)

                kmeans = KMeans(n_clusters=m, random_state=42, n_init='auto').fit(embeddings)
                closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, embeddings)
                for idx in closest:
                    selected_items.append((c, pool[idx]))
            else:
                # 样本数极少，无法聚类，直接循环选取
                print(f"Warning: Class {c} sample size {len(pool)} < {m}. Using repetition.")
                for k in range(m):
                    idx = k % len(pool)
                    selected_items.append((c, pool[idx]))

        # ---------- 对每个整图提连通子图，并重算 emb ----------
        init_proto_node_emb = []
        init_proto_graph_emb_list = []
        init_proto_node_list = []
        init_proto_edge_list = []
        init_proto_in_which_graph = []

        #  存储源图
        self.prototype_source_graphs = []
        init_subgraphs = []  # 用于存储子图，对应 prototype embedding 的结构

        for c, item in selected_items:  # 这里的 item 是字典
            g_emb_full = item['graph_emb']
            n_emb_full = item['node_emb']
            g_data_full = item['graph_data']
            conf_full = item['conf']

            #  保存源图，供后续 Search 使用
            self.prototype_source_graphs.append(g_data_full)

            # 临时启用梯度计算
            # 初始子图提取（Gradient-based)
            with torch.enable_grad():
                sub_data, sub_info = extract_connected_subgraph2(
                    g_data_full,
                    classifier,
                    target_class=c,
                    min_nodes=self.min_nodes,
                    max_nodes=self.max_nodes,
                    return_info=True,
                )

            # 保存子图结构，供 Generate 阶段计算 A_init 使用
            # sub_data 的 edge_index 已经是 relabel 过的 [0, N_sub-1]
            init_subgraphs.append(sub_data.detach().cpu())

            node_idx = sub_data.orig_node_idx.cpu()  # 原图中的节点编号 (N_sub,)
            edge_index_sub = sub_data.edge_index.cpu()  # 子图内部编号 (2, E)
            orig_edge_index = node_idx[edge_index_sub]  # (2, E), 原图编号

            init_proto_node_list.append(node_idx)
            init_proto_edge_list.append(orig_edge_index.t().tolist())

            # 在子图上重算 embedding
            sub_data_gpu = sub_data.clone().to(device)
            if not hasattr(sub_data_gpu, "batch") or sub_data_gpu.batch is None:
                sub_data_gpu.batch = torch.zeros(sub_data_gpu.num_nodes, dtype=torch.long, device=device)

            _, sub_probs, sub_node_emb, sub_graph_emb = classifier(sub_data_gpu)

            init_proto_node_emb.append(sub_node_emb.detach().cpu())
            init_proto_graph_emb_list.append(sub_graph_emb.squeeze(0).detach().cpu())  # [d]
            # 这里依然保存全图，用于 visualize_init_prototypes 能够画出全图背景
            init_proto_in_which_graph.append(g_data_full)

        # ---------- 堆叠 / 搬到 device ----------
        init_proto_graph_emb = torch.stack(init_proto_graph_emb_list, dim=0).to(device)
        init_proto_node_emb = [t.to(device) for t in init_proto_node_emb]

        # ---------- 注册成可学习参数 ----------
        self.prototype_node_emb = nn.ParameterList([
            nn.Parameter(t.clone().detach(), requires_grad=True)
            for t in init_proto_node_emb
        ])

        #  使用 anchor_graph_emb 作为动态目标，而不是固定的 init
        if hasattr(self, "anchor_graph_emb"):
            self.anchor_graph_emb = init_proto_graph_emb.clone().detach()
        else:
            self.register_buffer("anchor_graph_emb", init_proto_graph_emb.clone().detach())

        if hasattr(self, "init_proto_graph_emb"):
            self.init_proto_graph_emb = init_proto_graph_emb.clone().detach()
        else:
            self.register_buffer("init_proto_graph_emb", init_proto_graph_emb.clone().detach())

        self.init_proto_graphs = init_subgraphs

        self.init_proto_node_list = init_proto_node_list
        self.init_proto_edge_list = init_proto_edge_list

        return init_proto_node_emb, init_proto_graph_emb, init_proto_node_list, init_proto_edge_list, init_proto_in_which_graph

    #  动态锚点更新函数
    def update_prototype_anchors(self, momentum=0.6):
        """
        混合机制的核心：使用可微掩码优化 (Differentiable Mask Optimization) 更新 anchor
        """
        was_training = self.training
        self.eval()

        # 当前学习到的原型图向量
        with torch.no_grad():
            current_proto_graph_embs = self.proto_node_emb_2_graph_emb()  # [P, d]

        new_anchor_embs = []

        # 遍历每个原型
        for i in range(self.num_prototypes):
            # 1. 获取目标向量
            target_vec = current_proto_graph_embs[i]  # [d]

            # 2. 获取源图
            source_graph = self.prototype_source_graphs[i].to(self.device)
            if not hasattr(source_graph, "batch") or source_graph.batch is None:
                source_graph.batch = torch.zeros(source_graph.num_nodes, dtype=torch.long, device=self.device)

            best_nodes, best_sim = differentiable_search_subgraph(
                proto_graph_emb=target_vec,
                source_data=source_graph,  # 直接传 data，因为需要 x 和 edge_index
                model=self,
                min_nodes=self.min_nodes,
                max_nodes=self.max_nodes,
                iterations=50,  # 迭代次数，50次通常足够收敛
                lr=0.05,  # 学习率
                l1_reg=0.01,  # 稀疏正则权重，可根据效果微调
                lambda_entropy=0.01,  # 熵正则权重
                lambda_conn=0.1
            )

            if best_nodes is not None and len(best_nodes) > 0:
                node_idx = torch.tensor(sorted(list(best_nodes)), dtype=torch.long, device=self.device)

                # 再次 forward 最终确定的子图以获取 Embedding
                sub_edge_index, _ = subgraph(node_idx, source_graph.edge_index, relabel_nodes=True)

                sub_data = Data(x=source_graph.x[node_idx], edge_index=sub_edge_index)
                sub_data.batch = torch.zeros(sub_data.num_nodes, dtype=torch.long, device=self.device)

                _, _, _, sub_graph_emb, _ = self.forward(sub_data)
                new_anchor_embs.append(sub_graph_emb.detach())
            else:
                # 兜底：如果优化失败，保持旧 anchor
                new_anchor_embs.append(self.anchor_graph_emb[i].unsqueeze(0))

        new_anchors_tensor = torch.cat(new_anchor_embs, dim=0).detach()
        # 如果是第一次更新（或 anchor 还没初始化好），直接赋值
        # 否则：Old * momentum + New * (1 - momentum)
        if not hasattr(self, "anchor_graph_emb"):
            self.register_buffer("anchor_graph_emb", new_anchors_tensor)
        else:
            self.anchor_graph_emb = momentum * self.anchor_graph_emb + (1 - momentum) * new_anchors_tensor

        if was_training:
            self.train()

    @torch.no_grad()
    def refresh_and_update_anchors(self, trainloader, top_k_ratio=0.2):
        """
        动态源图更新与锚点投影
        """
        was_training = self.training
        self.eval()
        device = self.device

        m = self.num_prototypes_per_class
        num_classes = self.output_dim

        # 1. 收集当前模型预测正确的样本及其嵌入
        class_samples = {c: [] for c in range(num_classes)}
        for batch in trainloader:
            batch = batch.to(device)
            _, probs, _, graph_emb, _ = self.forward(batch)

            probs_cpu = probs.detach().cpu()
            emb_cpu = graph_emb.detach().cpu().numpy()
            data_list = batch.to_data_list()

            for i, data in enumerate(data_list):
                y_true = int(data.y.item())
                pred = int(probs_cpu[i].argmax().item())
                if pred == y_true:
                    class_samples[y_true].append({
                        'data': data.clone().cpu(),
                        'emb': emb_cpu[i]
                    })

        # 2. 聚类更新源图池
        ordered_source_graphs = [None] * (num_classes * m)
        for c in range(num_classes):
            pool = class_samples[c]
            # 如果当前模型对该类全预测错了，保留旧的源图，不进行更新
            if len(pool) < m:
                # 降级处理：若正确样本不足，保留旧源图
                start_idx = c * m
                for k in range(m):
                    ordered_source_graphs[start_idx + k] = self.prototype_source_graphs[start_idx + k]
                continue

            # 对预测正确的样本进行聚类
            embeddings = np.array([s['emb'] for s in pool])
            kmeans = KMeans(n_clusters=m, random_state=42, n_init='auto').fit(embeddings)

            # 获取最靠近 m 个簇中心的真实样本
            closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, embeddings)

            for k, sample_idx in enumerate(closest):
                global_proto_idx = c * m + k
                ordered_source_graphs[global_proto_idx] = pool[sample_idx]['data']

        # 3.更新成员变量并执行锚点投影
        self.prototype_source_graphs = ordered_source_graphs
        self.update_prototype_anchors(momentum=0.5)

        if was_training:
            self.train()

    def set_last_layer_incorrect_connection(self, incorrect_strength):
        """
        the incorrect strength will be actual strength if -0.5 then input -0.5
        """
        positive_one_weights_locations = torch.t(self.prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        self.last_layer.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations)

    def prototype_distances(self, x):
        prototype_graph_emb = self.proto_node_emb_2_graph_emb()
        xp = torch.mm(x, torch.t(prototype_graph_emb))
        distance = -2 * xp + torch.sum(x ** 2, dim=1, keepdim=True) + torch.t(
            torch.sum(prototype_graph_emb ** 2, dim=1, keepdim=True))
        similarity = torch.log((distance + 1) / (distance + self.epsilon))
        return similarity, distance

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for i in range(self.num_gnn_layers):
            x = self.gnn_layers[i](x, edge_index)
            if self.emb_normalize:
                x = F.normalize(x, p=2, dim=-1)
            x = self.gnn_non_linear(x)

        node_emb = x
        pooled = []
        for readout in self.readout_layers:
            pooled.append(readout(x, batch))
        x = torch.cat(pooled, dim=-1)

        # 如果开启了归一化，确保图级嵌入也是单位模长
        if self.emb_normalize:
            x = F.normalize(x, p=2, dim=-1)

        graph_emb = x

        similarity, min_distances = self.prototype_distances(x)
        logits = self.last_layer(similarity)
        probs = self.Softmax(logits)
        return logits, probs, node_emb, graph_emb, min_distances

    @torch.no_grad()
    def search_prototype_subgraphs(self, trainloader, min_nodes: int, max_nodes: int):
        """
        对当前模型学到的每个原型， 在训练集的真实图上搜索与其图级嵌入最相似的真实子图
        (min_nodes <= 子图节点数 <= max_nodes)。

        返回：
          best_node_list: List[P]，best_node_list[p] 是 Tensor[ num_nodes_sub_p ]
                          表示该原型对应子图在那张图里的节点编号（相对那张 Data，从 0 开始）
          best_edge_list: List[P]，best_edge_list[p] 是 LongTensor[2, E_sub_p]，子图的边列表
          best_sim_list:  List[P]，best_sim_list[p] 是 float，相似度
          best_graph_idx_list
        """
        # 保持不变，用于最终可视化
        device = self.device
        self.eval()

        # 当前原型的图级嵌入 [P, d]
        proto_graph_emb = self.proto_node_emb_2_graph_emb()  # 你已有的函数
        num_prototypes = proto_graph_emb.size(0)

        # 初始化最优解
        best_node_list = [None for _ in range(num_prototypes)]
        best_edge_list = [None for _ in range(num_prototypes)]
        best_sim_list = [float("-inf") for _ in range(num_prototypes)]
        best_graph_idx_list = [None for _ in range(num_prototypes)]
        global_graph_idx = 0

        for batch in trainloader:
            batch = batch.to(device)
            logits, probs, node_emb, graph_emb, _ = self.forward(batch)

            data_list = batch.to_data_list()
            # node_emb: [sum_i N_i, d]，data_list 里每个 Data 的节点是 0..N_i-1
            cursor = 0
            for data in data_list:
                N = data.num_nodes
                node_emb_g = node_emb[cursor:cursor + N]  # [N, d]
                cursor += N

                edge_index_g = data.edge_index.to(device)  # [2, E]，节点编号 0..N-1
                x_g = data.x.to(device)

                # 对每个原型，在这张图上跑一次搜索
                for p_idx in range(num_prototypes):
                    proto_emb_p = proto_graph_emb[p_idx]  # [d]

                    temp_data = Data(x=data.x.to(device), edge_index=data.edge_index.to(device))
                    temp_data.batch = torch.zeros(temp_data.num_nodes, dtype=torch.long, device=device)

                    sub_nodes, sub_sim = differentiable_search_subgraph(
                        proto_graph_emb=proto_emb_p,
                        source_data=temp_data,
                        model=self,
                        min_nodes=min_nodes,
                        max_nodes=max_nodes,
                        iterations=100,  # 推理阶段可以用更多迭代
                        lr=0.05
                    )
                    if sub_nodes is None:
                        continue

                    if sub_sim > best_sim_list[p_idx]:
                        best_sim_list[p_idx] = sub_sim
                        # sub_nodes 是 set[int]，转 Tensor 并排序一下
                        node_idx = torch.tensor(sorted(list(sub_nodes)), dtype=torch.long)

                        # 构造子图边：选两端都在 node_idx 的边
                        node_mask = torch.zeros(N, dtype=torch.bool, device=device)
                        node_mask[node_idx] = True
                        mask_edge = node_mask[edge_index_g[0]] & node_mask[edge_index_g[1]]
                        sub_edge_index = edge_index_g[:, mask_edge].detach().cpu()  # 保留原图节点编号

                        best_node_list[p_idx] = node_idx.detach().cpu()
                        best_edge_list[p_idx] = sub_edge_index
                        best_graph_idx_list[p_idx] = global_graph_idx
                global_graph_idx += 1

        return best_node_list, best_edge_list, best_sim_list, best_graph_idx_list

    @staticmethod
    def _sym_clean(A, symmetric=True, remove_self_loops=True):
        if symmetric:
            A = 0.5 * (A + A.transpose(-1, -2))
        if remove_self_loops:
            idx = torch.arange(A.size(-1), device=A.device)
            A[..., idx, idx] = 0.0
        return A

    def graph_decoder(self, prototype_node_emb):
        # 若输入 [N, d] -> [1, N, d]
        if prototype_node_emb.dim() == 2:
            prototype_node_emb = prototype_node_emb.unsqueeze(0)
        H = prototype_node_emb
        B, N, _ = H.shape

        # 两层 MLP 生成节点属性（重构 X）
        node_feats = self.node_mlp(H)

        # 内积解码得到边 logits / 概率
        edge_logits = torch.bmm(node_feats, node_feats.transpose(1, 2)) * self.logit_scale  # [B, N, N]
        edge_probs = torch.sigmoid(edge_logits)

        # 对称并去自环
        adj_soft = self._sym_clean(edge_probs)

        return {
            "node_feats": node_feats,  # 用于节点属性重构 + 生成
            "edge_logits": edge_logits,
            "edge_probs": edge_probs,
            "adj_soft": adj_soft,
        }

    def _largest_connected_component(self, adj_hard: torch.Tensor):
        """
        adj_hard: [N, N] in {0,1}
        返回 largest_cc_nodes: List[int] 以及子图边列表 List[(u,v)]
        """
        N = adj_hard.size(0)
        # 无边的情况
        if adj_hard.sum() == 0:
            nodes = list(range(N))
            edges = []
            return nodes, edges

        visited = [False] * N
        comps = []

        for v in range(N):
            if not visited[v]:
                stack = [v]
                comp = []
                visited[v] = True
                while stack:
                    u = stack.pop()
                    comp.append(u)
                    neighbors = (adj_hard[u] > 0).nonzero(as_tuple=False).view(-1).tolist()
                    for w in neighbors:
                        if not visited[w]:
                            visited[w] = True
                            stack.append(w)
                comps.append(comp)

        # 取最大连通分量
        comps.sort(key=len, reverse=True)
        largest_cc = comps[0]
        largest_cc_set = set(largest_cc)

        # 生成边列表
        edges = []
        for i in largest_cc:
            for j in range(i + 1, N):
                if j in largest_cc_set and adj_hard[i, j] > 0:
                    edges.append((i, j))

        return largest_cc, edges

    # 在训练后生成原型图+可视化
    def generate_candidate_prototypes(self, threshold: float = 0.9):
        """
        [策略升级] 基于锚点的生成式细化 (Anchor-based Generative Refinement)
        不再凭空生成，而是利用 Search Branch 找到的源图作为骨架，
        生成分支学习一个 Attention/Mask 来“提纯”这个骨架。
        """
        device = self.device
        candidate_embeddings = []
        candidate_node_features = []
        candidate_graph_data = []

        was_training = self.training
        self.eval()

        with torch.no_grad():
            for p_idx in range(self.num_prototypes):
                # 1. 获取对应的源图 (来自 Search Branch 的成果)
                # 注意：self.prototype_source_graphs 是 Data 对象列表
                source_data = self.prototype_source_graphs[p_idx].to(device)

                # 2. 获取当前原型的 Embedding
                proto_vec = self.prototype_node_emb[p_idx]  # [N_k, d] (这是以前的可学习参数)
                # 或者使用图级嵌入
                # proto_vec = self.proto_node_emb_2_graph_emb()[p_idx] # [d]

                # 3. 这里的关键逻辑：
                # 我们不再用 proto_vec 解码出 N*N 矩阵。
                # 而是计算 proto_vec 与 source_data 节点的"相关性"，作为保留边的依据。

                # 获取源图的节点特征
                #  【新增修正】先通过 GNN 提取节点嵌入
                # 不能直接用 source_data.x，因为维度不对
                x_src, edge_index_src = source_data.x, source_data.edge_index
                # 临时通过 GNN 层 (不经过最后一层分类器)
                for i in range(self.num_gnn_layers):
                    x_src = self.gnn_layers[i](x_src, edge_index_src)
                    if self.emb_normalize:
                        x_src = F.normalize(x_src, p=2, dim=-1)
                    x_src = self.gnn_non_linear(x_src)

                # 此时 x_src 是 [N_src, 128]，可以喂给 decoder 了
                node_emb_src = x_src

                # 如果 Search Branch 已经找到过一个最优子图索引，我们应该基于那个子图做细化
                # 如果没有记录，就基于全图做 (这里假设基于全图，或者你可以存下 best_node_list)

                # 计算节点重要性分数 (Refinement Score)
                # 简单做法：计算源图节点与原型嵌入的相似度
                # 假设 proto_vec 是图级嵌入 [d]
                # scores = torch.matmul(x_src, proto_vec.unsqueeze(1)).squeeze() # [N_src]
                # scores = torch.sigmoid(scores)

                # 高级做法 (使用你现有的 decoder MLP):
                # 我们把源图的节点特征视为 p_i，输入 node_mlp 得到重构特征
                # 然后计算边概率，但只保留 edge_index 中存在的边！

                # 使用源图特征进行解码
                decoder_out = self.graph_decoder(node_emb_src)
                edge_probs_full = decoder_out["edge_probs"][0]  # [N_src, N_src] 全连接概率

                # 4. [核心创新] 结构掩码 (Structural Masking)
                # 只保留源图真实存在的边！
                # 这样生成的图，结构必然是源图的子集，绝对不可能是完全图。

                # 构建源图的邻接矩阵 Mask
                src_adj_mask = torch.zeros_like(edge_probs_full)
                src_adj_mask[edge_index_src[0], edge_index_src[1]] = 1.0

                # 融合：Refined Adjacency = Generator Probability * Structural Prior
                refined_adj = edge_probs_full * src_adj_mask

                # 5. 阈值截断与清理
                # 现在 refined_adj 非常稀疏，且结构合理
                # 过滤掉低概率边
                final_adj = (refined_adj > threshold).float()

                # 6. 取最大连通分量 (保证连通性)
                # 复用你之前的逻辑
                sub_nodes, sub_edges = self._largest_connected_component(final_adj)

                # 排序与提取
                sub_nodes = torch.tensor(sorted(list(sub_nodes)), dtype=torch.long, device=device)

                # 限制大小 (避免图太大)
                max_gen_nodes = self.max_gen_nodes
                if sub_nodes.numel() > max_gen_nodes:
                    # 按节点度数/重要性剪枝
                    pass  # (保留你原有的剪枝逻辑)

                # 提取最终子图数据
                # 注意：这里生成的 node_feats 应该是 generator 重构的特征，还是原图特征？
                # 建议：使用原图特征 x_src，这样解释性更强（"这是从真实数据中提炼的"）
                # 或者使用 generator 的输出 node_feats (更加抽象、平滑)
                # 为了 Hybrid 的故事，使用 generator 的输出 node_feats 更好，代表"理想化"
                node_feats_sub = decoder_out["node_feats"][0][sub_nodes].detach().cpu()

                # 构建边列表
                A_new_sub = final_adj[sub_nodes][:, sub_nodes]
                edge_index_new = (A_new_sub > 0).nonzero(as_tuple=False).t().contiguous().detach().cpu()

                # 封装结果
                graph_data = Data(
                    x=node_feats_sub,
                    edge_index=edge_index_new,
                    y=torch.tensor([p_idx // self.num_prototypes_per_class], dtype=torch.long),
                    prototype_idx=torch.tensor([p_idx], dtype=torch.long),
                )

                # 存入列表
                graph_data = graph_data.to(device)
                batch_data = Batch.from_data_list([graph_data])
                _, _, _, graph_emb, _ = self.forward(batch_data)  # 获取生成图的嵌入用于对齐 loss

                candidate_embeddings.append(graph_emb)
                candidate_graph_data.append(graph_data)
                candidate_node_features.append(node_feats_sub)

        if was_training:
            self.train()

        if candidate_embeddings:
            self.generated_candidates = {
                "embeddings": torch.cat(candidate_embeddings, dim=0),
                "node_features": candidate_node_features,
                "graph_data": candidate_graph_data,
            }
        else:
            self.generated_candidates = None

        return self.generated_candidates

    # 计算重构损失
    def reconstruction_loss(self, batch_data=None):
        """
        [修正] 引入自编码任务 (Auto-Encoder Task)
        使用当前的 Graph Decoder 重构输入的真实 Batch 数据。
        这教会生成器什么是“合理的图结构”。
        """
        if batch_data is None:
            # 如果没传 batch，退化为 0 (或者你可以保留旧逻辑作为正则，但不推荐)
            return torch.tensor(0.0, device=self.device)

        # 1. 获取 Batch 中真实节点的嵌入 (来自 GNN Encoder 的中间层输出)
        # 注意：我们需要 node_emb，这通常在 forward 的时候已经算过了。
        # 为了避免重复计算，我们可以让 train loop 传进来，或者在这里重新 forward 一次
        # 这里为了代码解耦，假设 batch_data 是原始数据，我们重新编码获取 node_emb

        # 使用 self.gnn_layers 提取特征 (不经过 last_layer)
        x, edge_index = batch_data.x, batch_data.edge_index
        for i in range(self.num_gnn_layers):
            x = self.gnn_layers[i](x, edge_index)
            if self.emb_normalize:
                x = F.normalize(x, p=2, dim=-1)
            x = self.gnn_non_linear(x)
        real_node_emb = x  # [Total_Nodes, d]

        # 2. 使用 Decoder 重构邻接关系
        # 注意：这里是对 Batch 里所有图的所有节点进行解码
        # 为了效率，我们只采样一部分边，或者只对每个子图内部做解码
        # 简单起见，利用 PyG 的 batch 属性，只计算同一图内部的边概率

        decoder_out = self.graph_decoder(real_node_emb)
        edge_logits = decoder_out['edge_logits']  # 注意：这里 graph_decoder 假设输入是 [B, N, d]

        # !重要修正!：你的 graph_decoder 实现是针对 [B, N, d] 的 (原型是这样的)。
        # 但这里的 real_node_emb 是 [Total_Nodes, d] (PyG 格式)。
        # 我们需要适配一下 graph_decoder 或者在这里手写一个内积解码

        # --- 手写适配 PyG Batch 的解码 ---
        node_feats = self.node_mlp(real_node_emb)  # [Total_Nodes, d_out]

        # 为了计算 BCE，我们需要正样本（存在的边）和负样本（不存在的边）

        # 正样本：batch_data.edge_index
        src, dst = batch_data.edge_index
        pos_score = (node_feats[src] * node_feats[dst]).sum(dim=-1) * self.logit_scale
        pos_loss = -F.logsigmoid(pos_score).mean()

        # 负样本：随机采样
        neg_src = torch.randint(0, batch_data.num_nodes, (src.size(0),), device=self.device)
        neg_dst = torch.randint(0, batch_data.num_nodes, (src.size(0),), device=self.device)
        # (简单采样，不严格排除正样本，在大图中影响不大；严格做法需用 negative_sampling)
        from torch_geometric.utils import negative_sampling
        neg_edge_index = negative_sampling(edge_index, num_nodes=batch_data.num_nodes)

        neg_src, neg_dst = neg_edge_index
        neg_score = (node_feats[neg_src] * node_feats[neg_dst]).sum(dim=-1) * self.logit_scale
        neg_loss = -F.logsigmoid(-neg_score).mean()

        rec_loss = pos_loss + neg_loss
        return rec_loss

    def alignment_loss(self):
        """
            计算对齐约束 - 生成候选应与原型嵌入靠近
            L_align = (1/C×m) ∑∑ ||g_{ck} ^G - {p}_{ck} ^G||_2^2
        """
        # 1. 只有当缓存为空时才生成 (通常在 Epoch 开始时会被清空一次)
        if self.generated_candidates is None:
            self.generate_candidate_prototypes(threshold=0.9)

        if self.generated_candidates is None:
            return torch.tensor(0.0).to(self.device)

        generated_embs = self.generated_candidates['embeddings']
        prototype_graph_emb = self.proto_node_emb_2_graph_emb()

        # 2. [关键修复] 强制归一化
        # 确保两个向量都在单位球面上，MSE 范围被限制在 [0, 4]
        # 避免几万的 Loss 冲垮模型
        gen_norm = F.normalize(generated_embs, p=2, dim=-1)
        proto_norm = F.normalize(prototype_graph_emb, p=2, dim=-1)

        alignment_loss = F.mse_loss(gen_norm, proto_norm)

        return alignment_loss

    def proto_loss(self):
        # [修改] 使用动态 anchor_graph_emb 作为 target
        if not hasattr(self, "anchor_graph_emb"):
            return torch.tensor(0.0, device=self.device)

        # 当前学到的抽象原型
        proto_graph_emb = self.proto_node_emb_2_graph_emb()
        # 动态更新的真实子图锚点
        target_emb = self.anchor_graph_emb.to(self.device)

        if target_emb.dim() == 3 and target_emb.size(1) == 1:
            target_emb = target_emb.squeeze(1)

        return F.mse_loss(proto_graph_emb, target_emb)

    def diversity_loss(self):
        diversity_loss = torch.tensor(0.0, device=self.device)
        prototype_graph_emb = self.proto_node_emb_2_graph_emb()
        for c in range(self.output_dim):
            p = prototype_graph_emb[c * self.num_prototypes_per_class:(c + 1) * self.num_prototypes_per_class]
            p = F.normalize(p, p=2, dim=1)  # L2归一化(余弦相似度等于内积)
            matrix1 = torch.mm(p, torch.t(p)) - torch.eye(p.shape[0]).to(
                self.device) - 0.3  # 内积减单位矩阵(去掉自相似)，再减去阈值\theta 0.3
            matrix2 = torch.zeros(matrix1.shape).to(self.device)
            diversity_loss = diversity_loss + torch.sum(torch.where(matrix1 > 0, matrix1, matrix2))

        return diversity_loss
