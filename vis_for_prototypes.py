import os
import torch
import torch.nn as nn
import numpy as np
import argparse
from torch_geometric.utils import subgraph

# 引入项目模块
from models import GnnNets
from load_dataset import get_dataset, get_dataloader
from Configures import data_args, train_args, model_args
from utils import PlotUtils
from visualization import visualize_init_prototypes, visualize_prototypes_from_search, visualize_generated_prototypes
from my_search import differentiable_search_subgraph
import re
from pretrain import GnnClassifier


# -------------------------------------------------------------------
# 1. 辅助加载函数
# -------------------------------------------------------------------
def load_best_model(gnnNets, ckpt_path, device):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    print(f"Loading model from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=device,weights_only=False)

    # 提取 state_dict
    if isinstance(checkpoint, nn.Module):
        # 如果直接保存了对象，那是最好的，直接返回
        print("Checkpoint contained full model object.")
        gnnNets = checkpoint
        gnnNets.to(device)
        gnnNets.eval()
        return gnnNets
    elif isinstance(checkpoint, dict) and 'net' in checkpoint:
        state_dict = checkpoint['net']
    else:
        state_dict = checkpoint

    # [关键修复] 动态重建 prototype_node_emb
    # 扫描 state_dict，找出有多少个原型，以及它们的形状
    # 假设 key 格式为 "model.prototype_node_emb.0" (GnnNets封装) 或 "prototype_node_emb.0" (无封装)

    proto_keys = [k for k in state_dict.keys() if "prototype_node_emb" in k]

    if proto_keys:
        # 1. 找到最大的索引
        max_idx = -1
        shapes = {}
        pattern = re.compile(r"prototype_node_emb\.(\d+)")

        for k in proto_keys:
            match = pattern.search(k)
            if match:
                idx = int(match.group(1))
                max_idx = max(max_idx, idx)
                shapes[idx] = state_dict[k].shape

        if max_idx >= 0:
            print(f"Detected {max_idx + 1} prototypes in state_dict. Reconstructing ParameterList...")

            # 2. 获取正确的 model 引用 (GnnNets -> GINNet)
            if hasattr(gnnNets, 'model'):
                inner_model = gnnNets.model
            else:
                inner_model = gnnNets

            # 3. 重置并填充 ParameterList
            inner_model.prototype_node_emb = nn.ParameterList()

            for i in range(max_idx + 1):
                if i in shapes:
                    # 创建占位参数 (值会被加载覆盖)
                    # 注意：必须放在 device 上，否则 load 可能报错
                    dummy = nn.Parameter(torch.zeros(shapes[i], device=device))
                    inner_model.prototype_node_emb.append(dummy)
                else:
                    # 理论上不该发生，除非索引不连续
                    print(f"Warning: Prototype {i} missing in state_dict keys.")
                    # 补一个默认的 (假设 shape=[1, 128]?) 或者报错
                    # 这里尝试从 0 号获取 shape，或者跳过
                    pass

    # 4. 加载权重
    try:
        gnnNets.load_state_dict(state_dict)
        print("State dict loaded successfully.")
    except RuntimeError as e:
        print(f"Strict loading failed: {e}")
        print("Loading with strict=False...")
        gnnNets.load_state_dict(state_dict, strict=False)

    gnnNets.to(device)
    gnnNets.eval()

    # 5. [额外修复] 尝试恢复 init_proto_node_list 等非参数属性
    # 如果 checkpoint 是字典且包含其他信息，或者 state_dict 里混入了 buffer
    # GIN.py 把 init_info 存在了 buffer 还是普通属性？
    # 如果是普通属性，save_state_dict 是存不下来的。
    # 如果你之前的代码是用 torch.save(gnnNets) 保存的，那就能恢复。
    # 如果是用 state_dict 保存的，这些 list 就会丢失。

    return gnnNets


# -------------------------------------------------------------------
# 2. 格式转换辅助函数
# -------------------------------------------------------------------
def tensor_to_list_int(tensor_data):
    if isinstance(tensor_data, torch.Tensor):
        return tensor_data.detach().cpu().tolist()
    return list(tensor_data)


def edge_index_to_list_tuples(edge_index):
    if edge_index.numel() == 0:
        return []
    edge_index = edge_index.detach().cpu()
    return list(zip(edge_index[0].tolist(), edge_index[1].tolist()))


# -------------------------------------------------------------------
# 3. 主可视化流程
# -------------------------------------------------------------------
def run_visualization():
    # --- [关键修改] 环境设置 ---
    if torch.cuda.is_available():
        model_args.device = 0  # 必须是 int 索引，供 GIN.py 拼接 'cuda:' 使用
        device = torch.device(f'cuda:{model_args.device}')
    else:
        # 注意：你的 GIN.py 硬编码了 'cuda:'，如果跑 CPU 可能会报错，
        # 这里仅作兼容性写法，建议在有 GPU 的环境运行
        model_args.device = 0
        device = torch.device('cpu')

    print(f"Using device: {device}")

    # --- 加载数据 ---
    print(f"Loading dataset {data_args.dataset_name}...")
    dataset = get_dataset(data_args.dataset_dir, data_args.dataset_name, task=data_args.task)

    # --- 加载模型 ---
    input_dim = dataset.num_node_features
    output_dim = int(dataset.num_classes)

    # 此时 model_args.device 已经是 int 了，初始化不会报错
    gnnNets = GnnNets(input_dim, output_dim, model_args)

    ckpt_path = f"./checkpoint/{data_args.dataset_name}/{model_args.model_name}_{model_args.readout}_best.pth"
    gnnNets = load_best_model(gnnNets, ckpt_path, device)

    # 准备绘图工具
    plotter = PlotUtils(dataset_name=data_args.dataset_name)
    img_dir = f"./checkpoint/{data_args.dataset_name}"

    for sub in ["init_prot", "search_prot", "gen_prot"]:
        os.makedirs(os.path.join(img_dir, sub), exist_ok=True)

    # ===============================================================
    # Task 1: 初始化原型可视化
    # ===============================================================
    print("\n[1/3] Visualizing Initialized Prototypes...")
    dataloader = get_dataloader(dataset, 128, 42)
    classifier = GnnClassifier(input_dim, output_dim, model_args)
    classifier.to_device()
    pretrain_dir = f'./pretrain/checkpoint/{data_args.dataset_name}/'
    pretrain_ckpt = torch.load(
        os.path.join(pretrain_dir, f'pre_{model_args.model_name}_{model_args.readout}_best.pth'),
        weights_only=False)
    classifier.update_state_dict(pretrain_ckpt['net'])
    (init_proto_node_emb, init_proto_graph_emb, init_proto_node_list,
     init_proto_edge_list, init_proto_in_which_graph) = gnnNets.model.initialize_prototypes(
        trainloader=dataloader['train'], classifier=classifier)
    if hasattr(gnnNets.model, 'init_proto_node_list') and gnnNets.model.init_proto_node_list:
        node_list_fmt = [tensor_to_list_int(n) for n in gnnNets.model.init_proto_node_list]
        edge_list_fmt = gnnNets.model.init_proto_edge_list
        graph_data_list = init_proto_in_which_graph

        visualize_init_prototypes(
            nodelist=node_list_fmt,
            edgelist=edge_list_fmt,
            num_prototypes_per_class=model_args.num_prototypes_per_class,
            graph_data=graph_data_list,
            plotter=plotter,
            out_dir=os.path.join(img_dir, "init_prot")
        )
    else:
        print("Warning: No 'init_proto_node_list' found. Skipping Init visualization.")

    # ===============================================================
    # Task 2: 搜索原型可视化 (基于 Source Graphs)
    # ===============================================================
    print("\n[2/3] Visualizing Searched Prototypes (on Source Graphs)...")

    if not hasattr(gnnNets.model, "prototype_source_graphs") or not gnnNets.model.prototype_source_graphs:
        print("Restoring source graphs for visualization...")
        dataloader = get_dataloader(dataset, 128, 42)
        classifier = GnnClassifier(input_dim, output_dim, model_args)
        classifier.to_device()
        pretrain_dir = f'./pretrain/checkpoint/{data_args.dataset_name}/'
        pretrain_ckpt = torch.load(
            os.path.join(pretrain_dir, f'pre_{model_args.model_name}_{model_args.readout}_best.pth'),
            weights_only=False)
        classifier.update_state_dict(pretrain_ckpt['net'])
        # 重新运行初始化以获取源图（只为了拿到图，不改变权重）
        gnnNets.model.initialize_prototypes(dataloader['train'], classifier)

    source_graphs = gnnNets.model.prototype_source_graphs

    search_node_list = []
    search_edge_list = []
    search_graph_list = []

    for p_idx in range(gnnNets.model.num_prototypes):
        target_vec = gnnNets.model.proto_node_emb_2_graph_emb()[p_idx]
        src_data = source_graphs[p_idx].to(device)
        if src_data.batch is None:
            src_data.batch = torch.zeros(src_data.num_nodes, dtype=torch.long, device=device)

        nodes_set, _ = differentiable_search_subgraph(
            proto_graph_emb=target_vec,
            source_data=src_data,
            model=gnnNets.model,
            min_nodes=model_args.min_nodes,
            max_nodes=model_args.max_nodes,
            iterations=50
        )

        if nodes_set:
            nodes_list_fmt = sorted(list(nodes_set))
            node_idx_tensor = torch.tensor(nodes_list_fmt, dtype=torch.long, device=device)

            # 提取原图索引的边用于记录
            mask = torch.zeros(src_data.num_nodes, dtype=torch.bool, device=device)
            mask[node_idx_tensor] = True
            row, col = src_data.edge_index
            edge_mask = mask[row] & mask[col]
            src_edges_tensor = src_data.edge_index[:, edge_mask]
            edges_list_fmt = edge_index_to_list_tuples(src_edges_tensor)

            search_node_list.append(nodes_list_fmt)
            search_edge_list.append(edges_list_fmt)
            search_graph_list.append(src_data.cpu())
        else:
            search_node_list.append([])
            search_edge_list.append([])
            search_graph_list.append(src_data.cpu())

    visualize_prototypes_from_search(
        node_list=search_node_list,
        edge_list=search_edge_list,
        graph_data_list=search_graph_list,
        num_prototypes_per_class=model_args.num_prototypes_per_class,
        plotter=plotter,
        out_dir=os.path.join(img_dir, "search_prot")
    )

    # ===============================================================
    # Task 3: 生成原型可视化
    # ===============================================================
    print("\n[3/3] Visualizing Generated Prototypes...")
    # 确保源图在设备上，供 generate 函数使用
    gnnNets.model.prototype_source_graphs = [g.to(device) for g in gnnNets.model.prototype_source_graphs]

    gnnNets.model.generate_candidate_prototypes(threshold=0.9)
    cand = gnnNets.model.generated_candidates

    if cand and "graph_data" in cand:
        gen_graph_data_list = cand["graph_data"]

        visualize_generated_prototypes(
            graph_data_list=gen_graph_data_list,
            num_prototypes_per_class=model_args.num_prototypes_per_class,
            plotter=plotter,
            out_dir=os.path.join(img_dir, "gen_prot")
        )
    else:
        print("Warning: No generated candidates found.")

    print("\nVisualization Done!")


if __name__ == "__main__":
    run_visualization()
