import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GCNConv
from torch_geometric.data import Data, Batch
from my_search import match_connected_subgraph
from models.model_utils import get_readout_layers, build_subgraph_adj, GraphDecoder


# GCN
class GCNNet(nn.Module):
    def __init__(self, input_dim, output_dim, model_args, data_args):
        super(GCNNet, self).__init__()
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
        self.gnn_layers.append(GCNConv(input_dim, self.latent_dim[0], normalize=model_args.adj_normalize))
        for i in range(1, self.num_gnn_layers):
            self.gnn_layers.append(
                GCNConv(self.latent_dim[i - 1], self.latent_dim[i], normalize=model_args.adj_normalize))
        self.gnn_non_linear = nn.ReLU()

        self.dropout = nn.Dropout(model_args.dropout)
        self.Softmax = nn.Softmax(dim=-1)
        self.mlp_non_linear = nn.ELU()

        self.bn = nn.BatchNorm1d(self.latent_dim[-1])

        # prototype layers
        self.epsilon = 1e-4
        self.proto_dim = self.dense_dim * len(self.readout_layers)  # graph_data embedding dim
        self.prototype_shape = (output_dim * model_args.num_prototypes_per_class, 128)
        self.graph_size = model_args.graph_size
        self.prototype_node_shape = (output_dim * model_args.num_prototypes_per_class, self.graph_size, 128)
        self.num_prototypes_per_class = model_args.num_prototypes_per_class

        # 先初始化为随机，后续用 initialize_prototypes_based_on_confidence来进行初始化
        # 只指定prototype_node_emb，图嵌入通过READOUT来算出
        self.num_prototypes = self.prototype_shape[0]
        self.prototype_node_emb = nn.Parameter(torch.empty(self.num_prototypes, self.graph_size, self.proto_dim),
                                               requires_grad=True)
        nn.init.xavier_uniform_(self.prototype_node_emb)

        self.last_layer = nn.Linear(self.num_prototypes, output_dim,
                                    bias=False)  # do not use bias

        assert (self.num_prototypes % output_dim == 0)

        # an onehot indication matrix for each prototype's class identity
        self.prototype_class_identity = torch.zeros(self.num_prototypes,
                                                    output_dim)
        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // model_args.num_prototypes_per_class] = 1

        self.graph_decoder = GraphDecoder(proto_dim=self.proto_dim, hidden_dim=128, edge_feat_dim=128,
                                          num_node_classes=data_args.num_node_classes).to(self.device)

        # 生成候选的存储
        self.generated_candidates = None  # 生成候选

        # initialize the last layer
        self.set_last_layer_incorrect_connection(incorrect_strength=-0.5)

    def proto_node_emb_2_graph_emb(self):
        if 'max' in self.readout_name:
            proto_graph_emb = self.prototype_node_emb.max(dim=1)[0]  # [P,128]
        elif 'sum' in self.readout_name:
            proto_graph_emb = self.prototype_node_emb.sum(dim=1)
        else:
            proto_graph_emb = self.prototype_node_emb.mean(dim=1)
        return proto_graph_emb

    # 初始化原型
    def initialize_prototypes(self, trainloader, classifier):
        classifier.to(self.device)
        classifier.eval()

        m = self.num_prototypes_per_class
        num_classes = self.output_dim
        # 记录每个类别的所有正确预测样本
        correct_predictions = {c: [] for c in range(num_classes)}
        all_samples = {c: [] for c in range(num_classes)}
        global_idx = 0

        with torch.no_grad():
            for batch in trainloader:
                batch = batch.to(self.device)
                logits, probs, node_emb, graph_emb = classifier(batch)

                # graph_emb 应为 [batch_size, embed_size]
                if graph_emb is None:
                    raise RuntimeError("Graph embedding is None")

                # 因为 batch 是合并的 Data (Batch)，使用 to_data_list() 获取原始 Data 列表
                data_list = batch.to_data_list()
                probs_cpu = probs.detach().cpu()
                node_emb_cpu = node_emb.detach().cpu()
                graph_emb_cpu = graph_emb.detach().cpu()

                for i in range(graph_emb_cpu.size(0)):
                    y_true = int(batch.y[i].item())  # 第i个图的真实标签，例如0
                    pred = int(probs_cpu[i].argmax().item())  # 第i个图的预测结果，例如0
                    conf = float(probs_cpu[i][y_true].item())  # 第i个图的置信度得分，即被正确分类的概率，例如0.996

                    # 获取第i个图对应的图数据
                    graph_data = data_list[i].clone().to('cpu')  # 确保图数据在CPU上

                    # 获取第i个图的节点嵌入
                    # 注意：node_emb_cpu 包含整个batch的节点嵌入，需要根据batch划分提取当前图的节点嵌入
                    batch_mask = (batch.batch == i)  # 创建当前图的掩码
                    current_node_emb = node_emb_cpu[batch_mask.cpu()]  # 提取当前图的节点嵌入

                    # (embedding, graph_data, node_emb, global_idx, confidence) 五元组
                    all_samples[y_true].append(
                        (graph_emb_cpu[i].clone(), graph_data, current_node_emb, global_idx, conf))
                    # 只存储预测正确的样本:
                    if pred == y_true:
                        correct_predictions[y_true].append(
                            (graph_emb_cpu[i].clone(), graph_data, current_node_emb, global_idx, conf))
                    global_idx = global_idx + 1

        # 现在对每个类做选择，若类 c 的正确预测样本长度 L >= m ，则从中取 m 个最高的；否则全取并从 all_samples 填充
        selected_embeddings = []  # 原型嵌入列表
        selected_graphs = []  # 对应的图数据列表
        selected_node_embeddings = []  # 对应的节点嵌入列表
        selection_info = []  # 选择信息，用于调试

        info_dict = {
            'correct_predictions_counts': {c: len(correct_predictions[c]) for c in range(num_classes)},
            'total_samples_counts': {c: len(all_samples[c]) for c in range(num_classes)},
            'final_selection': {c: 0 for c in range(num_classes)},
            'selected_confidences': {c: [] for c in range(num_classes)}
        }

        for c in range(num_classes):
            pool = correct_predictions[c]
            L = len(pool)
            chosen_embeddings = []
            chosen_graphs = []
            chosen_node_embeddings = []
            chosen_info = []
            chosen_confidences = []

            if L >= m:
                # 按置信度从高到低排序，选择前m个
                pool_sorted = sorted(pool, key=lambda x: x[4], reverse=True)  # 按置信度降序排序
                selected = pool_sorted[:m]
                chosen_embeddings = [item[0] for item in selected]
                chosen_graphs = [item[1] for item in selected]
                chosen_node_embeddings = [item[2] for item in selected]  # 提取节点嵌入
                chosen_confidences = [item[4] for item in selected]  # 置信度
                chosen_info = [f"class_{c}_high_conf_idx_{item[3]}_conf_{item[4]:.4f}" for item in selected]
                info_dict['final_selection'][c] = m
            else:
                # 如果正确预测样本不足 m 个，则使用所有正确预测样本
                pool_sorted = sorted(pool, key=lambda x: x[4], reverse=True)
                chosen_embeddings = [item[0] for item in pool_sorted]
                chosen_graphs = [item[1] for item in pool_sorted]
                chosen_node_embeddings = [item[2] for item in pool_sorted]  # 提取节点嵌入
                chosen_confidences = [item[4] for item in pool_sorted]  # 置信度
                chosen_info = [f"class_{c}_high_conf_idx_{item[3]}_conf_{item[4]:.4f}" for item in pool_sorted]

                # 如果仍然不足，重复使用最高置信度的样本
                need = m - len(pool)
                # 重复使用置信度最高的样本
                highest_conf_sample = pool_sorted[0] if pool_sorted else None

                for _ in range(need):
                    if highest_conf_sample:
                        chosen_embeddings.append(highest_conf_sample[0].clone())
                        chosen_graphs.append(highest_conf_sample[1])
                        chosen_node_embeddings.append(highest_conf_sample[2])
                        chosen_confidences.append(highest_conf_sample[4])
                        chosen_info.append(
                            f"class_{c}_repeated_idx_{highest_conf_sample[3]}_conf_{highest_conf_sample[4]:.4f}")

            selected_embeddings.extend(chosen_embeddings[:m])  # (C*m, 128)
            selected_graphs.extend(chosen_graphs[:m])
            selected_node_embeddings.extend(chosen_node_embeddings[:m])  # (C*m, N_i, 128)
            selection_info.extend(chosen_info[:m])
            info_dict['selected_confidences'][c] = chosen_confidences[:m]

        # 用选出的 C*m 个图的节点嵌入+图嵌入来初始化可学习的 proto_node_emb
        # prototype_node_emb 形状: [num_prototypes, graph_size, proto_dim]
        P, G, d = self.prototype_node_shape  # P=C*m, G=self.graph_size, d=self.protodim
        if len(selected_embeddings) < P:
            print(f"Warning: only {len(selected_embeddings)} candidate graphs for {P} prototypes, "
                  f"will repeat some high-confidence samples.")
            # 简单粗暴：循环填满
            while len(selected_embeddings) < P:
                selected_embeddings.append(selected_embeddings[-1].clone())
                selected_node_embeddings.append(selected_node_embeddings[-1].clone())
                selected_graphs.append(selected_graphs[-1])
        # assert len(selected_embeddings) == P, \
        #     f"selected_embeddings 数量 {len(selected_embeddings)} 与 num_prototypes {P} 不一致"

        init_proto_node_list = []
        init_proto_edge_list = []
        for k in range(P):  # 第k个原型
            # 第k个代表性图(全图)的图嵌入 & 节点嵌入
            graph_emb_k = selected_embeddings[k].to(self.device)  # [d]
            node_emb_k = selected_node_embeddings[k].to(self.device)  # [N_k, d]
            nk = node_emb_k.size(0)  # 该图的节点数 N_k

            # 1) 计算节点重要性：与图嵌入的余弦相似度
            g_norm = F.normalize(graph_emb_k, dim=0)  # [d]
            nodes_norm = F.normalize(node_emb_k, dim=1)  # [N_k, d]
            scores = torch.mv(nodes_norm, g_norm)  # [N_k]

            # 2) 按重要性排序，取前 min(N_k, graph_size) 个节点(降序)
            idx_sorted = torch.argsort(scores, descending=True)  # [N_k]
            used = min(nk, self.graph_size)

            # 3) 先用均值 + 小噪声填满整块 [graph_size, d]
            mean_node = node_emb_k.mean(dim=0, keepdim=True)  # [1, d]
            proto_block = mean_node + 1e-2 * torch.randn(self.graph_size, d, device=self.device)

            # 4) 用最重要的节点替换前 used 个位置
            proto_block[:used] = node_emb_k[idx_sorted[:used]]
            # 这样使得，若N_k >= graph_size，得到了排名前graph_size的节点嵌入
            # 若N_k < graph_size, 取到了N_k个节点嵌入和均值+噪声

            # 记录该原型对应的子图节点和边（保持原图索引）
            # 这些节点的索引相对于 selected_graphs[k] 那个原图（Data）的节点顺序
            node_idx = idx_sorted[:used].cpu()  # 原图中的节点索引，长度 = used

            graph_data_k = selected_graphs[k]  # 原始图 Data，在 CPU 上
            edge_index = graph_data_k.edge_index  # [2, E]，使用的是原图的节点编号
            num_nodes = graph_data_k.num_nodes

            # 节点掩码：标记原图中哪些节点属于这个子图
            node_mask = torch.zeros(num_nodes, dtype=torch.bool)
            node_mask[node_idx] = True

            # 只保留两端都在子图节点集合内的边（仍然使用原图的节点编号）
            edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
            sub_edge_index = edge_index[:, edge_mask].clone()  # [2, E_sub]，仍然是原图索引

            # 记录：
            # - node_idx: 该原型子图在原图中的节点索引
            # - sub_edge_index: 该子图在原图中的边(端点也是原图节点编号)
            init_proto_node_list.append(node_idx)  # List[Tensor(num_nodes_sub)]
            edges_for_vis = sub_edge_index.t().tolist()  # 形状 [E_sub, 2] -> List[List[int]]
            init_proto_edge_list.append(edges_for_vis)

            # 5) 写入到可学习原型参数中
            self.prototype_node_emb[k].data.copy_(proto_block)

        init_proto_node_emb = self.prototype_node_emb.detach().clone()
        init_proto_graph_emb = self.proto_node_emb_2_graph_emb().detach().clone()
        init_proto_in_which_graph = selected_graphs

        # [C*m,graph_size,d]  [C*m,d]  [num_nodes_sub_k]  [2, E_sub_k]  [C*m] info
        return init_proto_node_emb, init_proto_graph_emb, init_proto_node_list, init_proto_edge_list, init_proto_in_which_graph, info_dict

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
            if not self.gnn_layers[i].normalize:
                edge_weight = torch.ones(edge_index.shape[1]).to(self.device)
                x = self.gnn_layers[i](x, edge_index, edge_weight)
            else:
                x = self.gnn_layers[i](x, edge_index)
            if self.emb_normalize:  # the l2 normalization after gnn layer
                x = F.normalize(x, p=2, dim=-1)
            x = self.gnn_non_linear(x)

        node_emb = x
        pooled = []
        for readout in self.readout_layers:
            pooled.append(readout(x, batch))
        x = torch.cat(pooled, dim=-1)
        graph_emb = x
        graph_emb = self.bn(graph_emb)

        graph_emb = F.dropout(graph_emb, p=0.5, training=self.training)

        similarity, min_distances = self.prototype_distances(x)
        logits = self.last_layer(similarity)
        probs = self.Softmax(logits)
        return logits, probs, node_emb, graph_emb, min_distances

    # 生成候选
    def generate_candidate_prototypes(self):
        """
        生成候选原型 - 核心生成函数
        为每个原型生成多个候选图，然后提取它们的嵌入作为候选原型
        """

        candidate_embeddings = []
        candidate_adj_hard = []
        candidate_adj_soft = []
        candidate_node_features = []
        candidate_node_types = []
        candidate_edge_logits = []  # 存储edge_logits
        candidate_edge_probs = []  # 存储edge_probs
        candidate_graph_data = []  # 存储图数据对象
        candidate_node_lists = []  # 存储节点列表

        with torch.no_grad():
            # 为每个原型生成候选
            for i, node_emb in enumerate(self.prototype_node_emb):
                # node_emb: [self.graph_size, 128], 一共num_classes * num_prototype_per_class 个
                # node_emb = node_emb.to(self.device)

                # 使用解码器生成图结构，同时获取图嵌入
                decoder_out = self.graph_decoder(node_emb, tau=None, hard=True, readout_layers=self.readout_layers)
                # # 训练期：软邻接 + 节点类型 logits
                # out = decoder(proto, tau=1.0, hard=False)  # 用 Gumbel 训练更平滑
                # node_logits = out["node_logits"]  # 监督 CrossEntropy
                # adj_soft = out["adj_soft"]  # 监督 BCE/负采样
                # graph_emb = out["graph_emb"]  # 可与类别原型/标签对齐
                #
                # # 推理期：离散邻接 + 节点类型
                # out_inf = decoder(proto, tau=None, hard=True)
                # node_type = out_inf["node_type_ids"]  # [N]
                # A_bin = out_inf["adj_hard"]  # [N,N]
                # g_emb = out_inf["graph_emb"]  # [F]

                node_features = decoder_out['node_feats']
                node_type_ids = decoder_out['node_type_ids']
                edge_logits = decoder_out['edge_logits']
                edge_probs = decoder_out['edge_probs']
                adj_soft = decoder_out['adj_soft']
                adj_hard = decoder_out['adj_hard']
                graph_emb = decoder_out['graph_emb']

                single_adj_soft = adj_soft[0]
                if adj_hard is None:
                    # 如果意外没有硬邻接，就用 soft > 0.5 二值化一个
                    single_adj_hard = (single_adj_soft > 0.5).float()
                else:
                    single_adj_hard = adj_hard[0]
                single_node_features = node_features[0]
                single_node_types = node_type_ids[0]

                # 创建图数据对象
                graph_data, node_list = self._create_graph_data_from_generated(
                    single_node_features, single_adj_hard, single_node_types, prototype_idx=i
                )

                # 现在graph_embedding是通过读出函数计算得到的，维度为[1, proto_dim]
                candidate_embeddings.append(graph_emb)
                candidate_adj_hard.append(single_adj_hard)
                candidate_adj_soft.append(single_adj_soft)
                candidate_node_features.append(single_node_features)
                candidate_node_types.append(single_node_types)
                candidate_edge_logits.append(edge_logits)  # 存储边logits
                candidate_edge_probs.append(edge_probs)
                candidate_graph_data.append(graph_data)  # 存储图数据
                candidate_node_lists.append(node_list)  # 存储节点列表

            # 合并所有候选
            if candidate_embeddings:
                self.generated_candidates = {
                    'embeddings': torch.cat(candidate_embeddings, dim=0),
                    'adj_matrices_hard': candidate_adj_hard,
                    'adj_matrices_soft': candidate_adj_soft,
                    'node_features': candidate_node_features,
                    'node_types': candidate_node_types,
                    'edge_logits': candidate_edge_logits,
                    'edge_probs': candidate_edge_probs,
                    'graph_data': candidate_graph_data,
                    'node_lists': candidate_node_lists
                }

        return self.generated_candidates

    # 从生成创建图数据
    def _create_graph_data_from_generated(self, node_features, adj_hard, node_types, prototype_idx):
        # 确保节点特征在CPU上
        node_features = node_features.detach().cpu()
        adj_matrix = adj_hard.detach().cpu()
        node_types = node_types.detach().cpu()

        # 获取非零元素的索引（边）
        edge_index = torch.nonzero(adj_hard, as_tuple=False).t()

        # 创建节点列表
        num_nodes = node_features.shape[0]
        node_list = [j for j in range(num_nodes)]

        # 创建PyG Data对象
        graph_data = Data(
            x=node_features,
            edge_index=edge_index,
            y=torch.tensor([prototype_idx // self.num_prototypes_per_class]),  # 类别标签
            prototype_idx=torch.tensor([prototype_idx]),  # 原型索引
            node_types=node_types  # 节点类型
        )

        return graph_data, node_list

    # 计算重构损失
    def compute_reconstruction_loss(self):
        """
        纯自监督版本的重构损失：
        - 使用 decoder 对候选原型图生成 edge_logits
        - 将 edge_logits.sigmoid() 二值化/或用 soft label 得到 target_adj (detach)
        - 对两个之间做 BCEWithLogitsLoss
        """
        # 确保已经有生成结果
        if self.generated_candidates is None:
            self.generate_candidate_prototypes()

        if self.generated_candidates is None:
            return torch.tensor(0.0, device=self.device)

        edge_logits_list = self.generated_candidates['edge_logits']
        bce_logits = torch.nn.BCEWithLogitsLoss()
        total_rec_loss = torch.tensor(0.0, device=self.device)
        num_used = 0

        for p, edge_logits in enumerate(edge_logits_list):
            if edge_logits is None:
                continue

            edge_logits = edge_logits.to(self.device)
            if edge_logits.dim() == 3:
                edge_logits = edge_logits[0]  # [N,N]
            N = edge_logits.size(0)

            # 使用自身的 sigmoid 作为 target（或者二值化）
            with torch.no_grad():
                # soft label 版本
                # target_adj = torch.sigmoid(edge_logits).detach()
                # hard label 版本：
                target_adj = (torch.sigmoid(edge_logits) > 0.5).float().detach()

            # 只对非对角线元素算 BCE
            mask = ~torch.eye(N, dtype=torch.bool, device=self.device)
            logits_flat = edge_logits[mask]
            target_flat = target_adj[mask]

            rec_loss_p = bce_logits(logits_flat, target_flat)
            total_rec_loss = total_rec_loss + rec_loss_p
            num_used += 1

        if num_used == 0:
            return torch.tensor(0.0, device=self.device)

        reconstruction_loss = total_rec_loss / num_used
        return reconstruction_loss

    def compute_alignment_loss(self):
        """
            计算对齐约束 - 生成候选应与原型嵌入靠近
            L_align = (1/C×m) ∑∑ ||g_{ck} ^G - {p}_{ck} ^G||_2^2
        """
        if self.generated_candidates is None:
            return torch.tensor(0.0).to(self.device)

        generated_embs = self.generated_candidates['embeddings']

        prototype_graph_emb = self.proto_node_emb_2_graph_emb()

        alignment_loss = F.mse_loss(generated_embs, prototype_graph_emb)
        return alignment_loss

    def compute_proto_loss(self, init_proto_graph_emb=None):
        if init_proto_graph_emb is None:
            return torch.tensor(0.0).to(self.device)
        prototype_graph_emb = self.proto_node_emb_2_graph_emb()
        proto_loss = F.mse_loss(init_proto_graph_emb, prototype_graph_emb)

        return proto_loss

    def compute_diversity_loss(self):
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
