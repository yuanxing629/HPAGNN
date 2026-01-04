# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn.conv import GCNConv
# from torch_geometric.data import Data, Batch
#
# from models.model_utils import get_readout_layers, GraphDecoder
#
#
# # GCN
# class GCNNet(nn.Module):
#     def __init__(self, input_dim, output_dim, model_args):
#         super(GCNNet, self).__init__()
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.latent_dim = model_args.latent_dim
#         self.mlp_hidden = model_args.mlp_hidden
#         self.emb_normalize = model_args.emb_normalize
#         self.device = torch.device('cuda:' + str(model_args.device))
#         self.num_gnn_layers = len(self.latent_dim)
#         self.num_mlp_layers = len(self.mlp_hidden) + 1
#         self.dense_dim = self.latent_dim[-1]
#         self.readout_layers = get_readout_layers(model_args.readout)
#         self.readout_name = model_args.readout.lower()
#
#         self.gnn_layers = nn.ModuleList()
#         self.gnn_layers.append(GCNConv(input_dim, self.latent_dim[0], normalize=model_args.adj_normalize))
#         for i in range(1, self.num_gnn_layers):
#             self.gnn_layers.append(
#                 GCNConv(self.latent_dim[i - 1], self.latent_dim[i], normalize=model_args.adj_normalize))
#         self.gnn_non_linear = nn.ReLU()
#
#         self.dropout = nn.Dropout(model_args.dropout)
#         self.Softmax = nn.Softmax(dim=-1)
#         self.mlp_non_linear = nn.ELU()
#
#         self.bn = nn.BatchNorm1d(self.latent_dim[-1])
#
#         # prototype layers
#         self.epsilon = 1e-4
#         self.proto_dim = self.dense_dim * len(self.readout_layers)  # graph_data embedding dim
#         self.prototype_shape = (output_dim * model_args.num_prototypes_per_class, 128)
#         self.graph_size = model_args.graph_size
#         self.prototype_node_shape = (output_dim * model_args.num_prototypes_per_class, self.graph_size, 128)
#         self.num_prototypes_per_class = model_args.num_prototypes_per_class
#
#         # 先初始化为随机，后续用 initialize_prototypes_based_on_confidence来进行初始化
#         # 只指定prototype_node_emb，图嵌入通过READOUT来算出
#         self.num_prototypes = self.prototype_shape[0]
#
#         self.last_layer = nn.Linear(self.num_prototypes, output_dim,
#                                     bias=False)  # do not use bias
#
#         assert (self.num_prototypes % output_dim == 0)
#
#         # an onehot indication matrix for each prototype's class identity
#         self.prototype_class_identity = torch.zeros(self.num_prototypes,
#                                                     output_dim)
#         for j in range(self.num_prototypes):
#             self.prototype_class_identity[j, j // model_args.num_prototypes_per_class] = 1
#
#         self.graph_decoder = GraphDecoder(proto_dim=self.proto_dim, num_node_features=self.input_dim, hidden_dim=128,
#                                           use_bias=True).to(self.device)
#
#         # 生成候选的存储
#         self.generated_candidates = None  # 生成候选
#
#         # initialize the last layer
#         self.set_last_layer_incorrect_connection(incorrect_strength=-0.5)
#
#         # 原型节点嵌入现在用 ParameterList 存，每个原型大小可以不一样
#         self.prototype_node_emb = nn.ParameterList()  # 先空着，初始化时再填
#
#         self.init_proto_graphs = None
#         self.init_proto_selection_info = None
#
#     def proto_node_emb_2_graph_emb(self):
#         if 'max' in self.readout_name:
#             proto_graph_emb = self.prototype_node_emb.max(dim=1)[0]  # [P,128]
#         elif 'sum' in self.readout_name:
#             proto_graph_emb = self.prototype_node_emb.sum(dim=1)
#         else:
#             proto_graph_emb = self.prototype_node_emb.mean(dim=1)
#         return proto_graph_emb
#
#     # 初始化原型
#     @torch.no_grad()
#     def initialize_prototypes(self, trainloader, classifier):
#         """
#         使用预训练分类器，从训练集中为每一类选择 top-m 个高置信度整图，
#         作为初始化的原型。
#
#         返回：
#           - init_proto_node_emb: List[Tensor (N_k, d)], 每个原型对应一个整图的所有节点嵌入
#           - init_proto_graph_emb: Tensor (P, d), P = C * m
#           - init_proto_graph_data: List[Data]，每个原型对应的原始图（PyG Data）
#           - info_dict: 一些计数 / 置信度信息
#         """
#         device = self.device
#
#         classifier.to(device)
#         classifier.eval()
#
#         m = self.num_prototypes_per_class
#         num_classes = self.output_dim
#
#         # 每类样本： (graph_emb, node_emb, graph_data, conf)
#         correct_predictions = {c: [] for c in range(num_classes)}
#         all_samples = {c: [] for c in range(num_classes)}
#
#         for batch in trainloader:
#             batch = batch.to(device)
#             # 假设 classifier 返回: logits, probs, node_emb, graph_emb
#             logits, probs, node_emb, graph_emb = classifier(batch)
#
#             # graph_emb: [B, d], node_emb: [total_nodes_in_batch, d]
#             data_list = batch.to_data_list()
#
#             probs_cpu = probs.detach().cpu()
#             node_emb_cpu = node_emb.detach().cpu()
#             graph_emb_cpu = graph_emb.detach().cpu()
#             batch_cpu = batch.batch.detach().cpu()
#
#             for i, data_i in enumerate(data_list):
#                 y_true = int(batch.y[i].item())
#                 pred = int(probs_cpu[i].argmax().item())
#                 conf = float(probs_cpu[i][y_true].item())
#
#                 # 当前图的节点 mask
#                 batch_mask = (batch_cpu == i)
#                 current_node_emb = node_emb_cpu[batch_mask]  # [N_i, d]
#                 current_graph_emb = graph_emb_cpu[i]  # [d]
#
#                 # 这里记录的是当前图的 Data（在 CPU 上）
#                 graph_data_i = data_i.clone()  # PyG Data.clone() 即可
#
#                 tup = (
#                     current_graph_emb.clone(),  # graph_emb
#                     current_node_emb.clone(),  # node_emb
#                     graph_data_i,  # graph_data
#                     conf
#                 )
#                 all_samples[y_true].append(tup)
#                 if pred == y_true:
#                     correct_predictions[y_true].append(tup)
#
#         # ---------- 选出每类的 top-m 图 ----------
#         selected_node_emb_list = []  # List[P] of [N_k, d]
#         selected_graph_emb_list = []  # 将来 stack 成 [P, d]
#         selected_graph_data_list = []  # List[P] of Data
#         selection_info = []
#
#         info_dict = {
#             'correct_predictions_counts': {c: len(correct_predictions[c]) for c in range(num_classes)},
#             'total_samples_counts': {c: len(all_samples[c]) for c in range(num_classes)},
#             'final_selection': {c: 0 for c in range(num_classes)},
#             'selected_confidences': {c: [] for c in range(num_classes)},
#         }
#
#         for c in range(num_classes):
#             pool = correct_predictions[c]
#             if len(pool) == 0:
#                 # 如果这一类一个预测正确的都没有，就退化到 all_samples
#                 pool = all_samples[c]
#
#             if len(pool) == 0:
#                 raise RuntimeError(f"Class {c} has no samples in training data!")
#
#             # 按置信度从高到低排序
#             pool_sorted = sorted(pool, key=lambda x: x[3], reverse=True)
#
#             chosen = []
#             while len(chosen) < m:
#                 need = m - len(chosen)
#                 if len(pool_sorted) >= need:
#                     chosen.extend(pool_sorted[:need])
#                 else:
#                     # 样本数不足 m，就重复最高置信度的样本
#                     if len(pool_sorted) == 0:
#                         break
#                     chosen.extend(pool_sorted)
#             chosen = chosen[:m]
#
#             info_dict['final_selection'][c] = len(chosen)
#             info_dict['selected_confidences'][c] = [item[3] for item in chosen]
#
#             for item in chosen:
#                 g_emb, n_emb, g_data, conf = item
#                 selected_graph_emb_list.append(g_emb)
#                 selected_node_emb_list.append(n_emb)
#                 selected_graph_data_list.append(g_data)
#                 selection_info.append(f"class_{c}_conf_{conf:.4f}")
#
#         # ---------- 堆叠 / 搬到 device ----------
#         init_proto_graph_emb = torch.stack(selected_graph_emb_list, dim=0).to(device)  # [P, d]
#         init_proto_node_emb = [t.to(device) for t in selected_node_emb_list]  # List[P] of [N_k, d]
#         init_proto_graph_data = selected_graph_data_list  # List[Data]（留在 CPU）
#
#         # ---------- 注册成可学习参数 ----------
#         self.prototype_node_emb = nn.ParameterList([
#             nn.Parameter(t.clone().detach(), requires_grad=True)
#             for t in init_proto_node_emb
#         ])
#
#         # 图级嵌入的“初始版本”作为 buffer（做 proto MSE 约束时用）
#         self.register_buffer(
#             "init_proto_graph_emb",
#             init_proto_graph_emb.clone().detach()
#         )
#
#         # 存一份 selection_info 方便以后调试 / 打印
#         self.init_proto_selection_info = selection_info
#
#         return init_proto_node_emb, init_proto_graph_emb, init_proto_graph_data, info_dict
#
#     def set_last_layer_incorrect_connection(self, incorrect_strength):
#         """
#         the incorrect strength will be actual strength if -0.5 then input -0.5
#         """
#         positive_one_weights_locations = torch.t(self.prototype_class_identity)
#         negative_one_weights_locations = 1 - positive_one_weights_locations
#
#         correct_class_connection = 1
#         incorrect_class_connection = incorrect_strength
#         self.last_layer.weight.data.copy_(
#             correct_class_connection * positive_one_weights_locations
#             + incorrect_class_connection * negative_one_weights_locations)
#
#     def prototype_distances(self, x):
#         prototype_graph_emb = self.proto_node_emb_2_graph_emb()
#         xp = torch.mm(x, torch.t(prototype_graph_emb))
#         distance = -2 * xp + torch.sum(x ** 2, dim=1, keepdim=True) + torch.t(
#             torch.sum(prototype_graph_emb ** 2, dim=1, keepdim=True))
#         similarity = torch.log((distance + 1) / (distance + self.epsilon))
#         return similarity, distance
#
#     def forward(self, data):
#         x, edge_index, batch = data.x, data.edge_index, data.batch
#         for i in range(self.num_gnn_layers):
#             if not self.gnn_layers[i].normalize:
#                 edge_weight = torch.ones(edge_index.shape[1]).to(self.device)
#                 x = self.gnn_layers[i](x, edge_index, edge_weight)
#             else:
#                 x = self.gnn_layers[i](x, edge_index)
#             if self.emb_normalize:  # the l2 normalization after gnn layer
#                 x = F.normalize(x, p=2, dim=-1)
#             x = self.gnn_non_linear(x)
#
#         node_emb = x
#         pooled = []
#         for readout in self.readout_layers:
#             pooled.append(readout(x, batch))
#         x = torch.cat(pooled, dim=-1)
#         graph_emb = x
#         graph_emb = self.bn(graph_emb)
#
#         graph_emb = F.dropout(graph_emb, p=0.5, training=self.training)
#
#         similarity, min_distances = self.prototype_distances(x)
#         logits = self.last_layer(similarity)
#         probs = self.Softmax(logits)
#         return logits, probs, node_emb, graph_emb, min_distances
#
#     # 生成候选
#     def generate_candidate_prototypes(self):
#         """
#         生成候选原型 - 核心生成函数
#         为每个原型生成多个候选图，然后提取它们的嵌入作为候选原型
#         """
#
#         candidate_embeddings = []
#         candidate_adj_hard = []
#         candidate_adj_soft = []
#         candidate_node_features = []
#         candidate_node_types = []
#         candidate_edge_logits = []  # 存储edge_logits
#         candidate_edge_probs = []  # 存储edge_probs
#         candidate_graph_data = []  # 存储图数据对象
#         candidate_node_lists = []  # 存储节点列表
#
#         with torch.no_grad():
#             # 为每个原型生成候选
#             for i, node_emb in enumerate(self.prototype_node_emb):
#                 # node_emb: [self.graph_size, 128], 一共num_classes * num_prototype_per_class 个
#                 # node_emb = node_emb.to(self.device)
#
#                 # 使用解码器生成图结构，同时获取图嵌入
#                 decoder_out = self.graph_decoder(node_emb, tau=None, hard=True, readout_layers=self.readout_layers)
#                 # # 训练期：软邻接 + 节点类型 logits
#                 # out = decoder(proto, tau=1.0, hard=False)  # 用 Gumbel 训练更平滑
#                 # node_logits = out["node_logits"]  # 监督 CrossEntropy
#                 # adj_soft = out["adj_soft"]  # 监督 BCE/负采样
#                 # graph_emb = out["graph_emb"]  # 可与类别原型/标签对齐
#                 #
#                 # # 推理期：离散邻接 + 节点类型
#                 # out_inf = decoder(proto, tau=None, hard=True)
#                 # node_type = out_inf["node_type_ids"]  # [N]
#                 # A_bin = out_inf["adj_hard"]  # [N,N]
#                 # g_emb = out_inf["graph_emb"]  # [F]
#
#                 node_features = decoder_out['node_feats']
#                 node_type_ids = decoder_out['node_type_ids']
#                 edge_logits = decoder_out['edge_logits']
#                 edge_probs = decoder_out['edge_probs']
#                 adj_soft = decoder_out['adj_soft']
#                 adj_hard = decoder_out['adj_hard']
#                 graph_emb = decoder_out['graph_emb']
#
#                 single_adj_soft = adj_soft[0]
#                 if adj_hard is None:
#                     # 如果意外没有硬邻接，就用 soft > 0.5 二值化一个
#                     single_adj_hard = (single_adj_soft > 0.5).float()
#                 else:
#                     single_adj_hard = adj_hard[0]
#                 single_node_features = node_features[0]
#                 single_node_types = node_type_ids[0]
#
#                 # 创建图数据对象
#                 graph_data, node_list = self._create_graph_data_from_generated(
#                     single_node_features, single_adj_hard, single_node_types, prototype_idx=i
#                 )
#
#                 # 现在graph_embedding是通过读出函数计算得到的，维度为[1, proto_dim]
#                 candidate_embeddings.append(graph_emb)
#                 candidate_adj_hard.append(single_adj_hard)
#                 candidate_adj_soft.append(single_adj_soft)
#                 candidate_node_features.append(single_node_features)
#                 candidate_node_types.append(single_node_types)
#                 candidate_edge_logits.append(edge_logits)  # 存储边logits
#                 candidate_edge_probs.append(edge_probs)
#                 candidate_graph_data.append(graph_data)  # 存储图数据
#                 candidate_node_lists.append(node_list)  # 存储节点列表
#
#             # 合并所有候选
#             if candidate_embeddings:
#                 self.generated_candidates = {
#                     'embeddings': torch.cat(candidate_embeddings, dim=0),
#                     'adj_matrices_hard': candidate_adj_hard,
#                     'adj_matrices_soft': candidate_adj_soft,
#                     'node_features': candidate_node_features,
#                     'node_types': candidate_node_types,
#                     'edge_logits': candidate_edge_logits,
#                     'edge_probs': candidate_edge_probs,
#                     'graph_data': candidate_graph_data,
#                     'node_lists': candidate_node_lists
#                 }
#
#         return self.generated_candidates
#
#     # 从生成创建图数据
#     def _create_graph_data_from_generated(self, node_features, adj_hard, node_types, prototype_idx):
#         # 确保节点特征在CPU上
#         node_features = node_features.detach().cpu()
#         adj_matrix = adj_hard.detach().cpu()
#         node_types = node_types.detach().cpu()
#
#         # 获取非零元素的索引（边）
#         edge_index = torch.nonzero(adj_hard, as_tuple=False).t()
#
#         # 创建节点列表
#         num_nodes = node_features.shape[0]
#         node_list = [j for j in range(num_nodes)]
#
#         # 创建PyG Data对象
#         graph_data = Data(
#             x=node_features,
#             edge_index=edge_index,
#             y=torch.tensor([prototype_idx // self.num_prototypes_per_class]),  # 类别标签
#             prototype_idx=torch.tensor([prototype_idx]),  # 原型索引
#             node_types=node_types  # 节点类型
#         )
#
#         return graph_data, node_list
#
#     # 计算重构损失
#     def compute_reconstruction_loss(self):
#         """
#         纯自监督版本的重构损失：
#         - 使用 decoder 对候选原型图生成 edge_logits
#         - 将 edge_logits.sigmoid() 二值化/或用 soft label 得到 target_adj (detach)
#         - 对两个之间做 BCEWithLogitsLoss
#         """
#         # 确保已经有生成结果
#         if self.generated_candidates is None:
#             self.generate_candidate_prototypes()
#
#         if self.generated_candidates is None:
#             return torch.tensor(0.0, device=self.device)
#
#         edge_logits_list = self.generated_candidates['edge_logits']
#         bce_logits = torch.nn.BCEWithLogitsLoss()
#         total_rec_loss = torch.tensor(0.0, device=self.device)
#         num_used = 0
#
#         for p, edge_logits in enumerate(edge_logits_list):
#             if edge_logits is None:
#                 continue
#
#             edge_logits = edge_logits.to(self.device)
#             if edge_logits.dim() == 3:
#                 edge_logits = edge_logits[0]  # [N,N]
#             N = edge_logits.size(0)
#
#             # 使用自身的 sigmoid 作为 target（或者二值化）
#             with torch.no_grad():
#                 # soft label 版本
#                 # target_adj = torch.sigmoid(edge_logits).detach()
#                 # hard label 版本：
#                 target_adj = (torch.sigmoid(edge_logits) > 0.5).float().detach()
#
#             # 只对非对角线元素算 BCE
#             mask = ~torch.eye(N, dtype=torch.bool, device=self.device)
#             logits_flat = edge_logits[mask]
#             target_flat = target_adj[mask]
#
#             rec_loss_p = bce_logits(logits_flat, target_flat)
#             total_rec_loss = total_rec_loss + rec_loss_p
#             num_used += 1
#
#         if num_used == 0:
#             return torch.tensor(0.0, device=self.device)
#
#         reconstruction_loss = total_rec_loss / num_used
#         return reconstruction_loss
#
#     def compute_alignment_loss(self):
#         """
#         对齐约束（生成端 vs 编码端）：
#
#         - 编码端：proto_node_emb_2_graph_emb() 得到每个原型的图级嵌入 p^G ∈ R^d
#         - 生成端：用 GraphDecoder(prototype_node_emb) 得到对应的生成图嵌入 g^G ∈ R^d
#
#         L_align = MSE( g^G , p^G_detach )
#
#         这里将 p^G 视为“目标”，对它做 detach，让梯度主要更新 decoder 和 prototype_node_emb 本身。
#         """
#         device = self.device
#
#         # 编码端：当前原型的图级嵌入 [P, d]
#         proto_graph_emb = self.proto_node_emb_2_graph_emb().detach()  # 作为 target
#
#         dec_graph_emb_list = []
#         for proto_nodes in self.prototype_node_emb:
#             # proto_nodes: [N_p, d_proto]
#             out = self.graph_decoder(
#                 prototype_node_emb=proto_nodes,  # 每个原型单独一张图
#                 tau=None,
#                 hard=False,
#                 readout_layers=self.readout_layers
#             )
#             # graph_emb: [1, d]
#             dec_graph_emb_list.append(out["graph_emb"][0])
#
#         if len(dec_graph_emb_list) == 0:
#             return torch.tensor(0.0, device=device)
#
#         generated_embs = torch.stack(dec_graph_emb_list, dim=0)  # [P, d]
#
#         alignment_loss = F.mse_loss(generated_embs, proto_graph_emb)
#         return alignment_loss
#
#     def compute_proto_loss(self):
#         if not hasattr(self, "init_proto_graph_emb"):
#             return torch.tensor(0.0, device=self.device)
#
#         # 编码端当前的图级原型嵌入
#         proto_graph_emb = self.proto_node_emb_2_graph_emb()
#
#         # buffer 中的初始化图级嵌入，作为 target
#         init_emb = self.init_proto_graph_emb.to(self.device)
#
#         return F.mse_loss(proto_graph_emb, init_emb)
#
#     def compute_diversity_loss(self):
#         diversity_loss = torch.tensor(0.0, device=self.device)
#         prototype_graph_emb = self.proto_node_emb_2_graph_emb()
#         for c in range(self.output_dim):
#             p = prototype_graph_emb[c * self.num_prototypes_per_class:(c + 1) * self.num_prototypes_per_class]
#             p = F.normalize(p, p=2, dim=1)  # L2归一化(余弦相似度等于内积)
#             matrix1 = torch.mm(p, torch.t(p)) - torch.eye(p.shape[0]).to(
#                 self.device) - 0.3  # 内积减单位矩阵(去掉自相似)，再减去阈值\theta 0.3
#             matrix2 = torch.zeros(matrix1.shape).to(self.device)
#             diversity_loss = diversity_loss + torch.sum(torch.where(matrix1 > 0, matrix1, matrix2))
#
#         return diversity_loss
