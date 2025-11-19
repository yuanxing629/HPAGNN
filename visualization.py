import os
import torch
from my_search import match_connected_subgraph
from my_search2 import conditional_subgraph_sampling


@torch.no_grad()
def visualize_init_prototypes(nodelist,edgelist,num_prototypes_per_class,graph_data, plotter, out_dir: str):
    # P = int(len(graph_data))  # 原型总数 C*m
    # 每类原型数 m
    m = num_prototypes_per_class

    for p_idx,data in enumerate(graph_data):
        proto_label = p_idx // m

        png_name = f"init_prot_{proto_label}_{p_idx:03d}.png"
        out_png = os.path.join(out_dir, png_name)

        # 取出当前原型对应的节点 / 边（注意索引）
        nodes_for_this_proto = nodelist[p_idx]  #  1D Tensor，原图节点 id
        edges_for_this_proto = edgelist[p_idx]  # List[[u,v], ...]

        print(
            f"[ProtoVis] init_prot: class={proto_label}, proto_idx={p_idx:03d}, "
            f"num_nodes_sub={nodes_for_this_proto.numel()}, "
            f"num_edges_sub={len(edges_for_this_proto)}"
        )

        plotter.save_full_graph_highlight(
            data=data,
            highlight_nodes=nodes_for_this_proto,
            highlight_edges=edges_for_this_proto,
            out_png=out_png,
            title=f"init_prot | label={proto_label} | proto={p_idx:03d}",
        )



@torch.no_grad()
def visualize_prototypes_on_dataset(
        gnnNets,
        dataset,
        plotter,
        out_dir: str,
        filename_prefix: str = "search_prot",
):
    """
    用【当前时刻】的原型嵌入在真实图中搜索最佳连通子图，并保存整图+子图边高亮的可视化。
    - filename_prefix: "search_prot"
    - 图片命名: f"{filename_prefix}_{label}_{proto_idx:03d}.png"
    """
    os.makedirs(out_dir, exist_ok=True)

    device = gnnNets.device if hasattr(gnnNets, "device") else next(gnnNets.parameters()).device
    P = int(gnnNets.model.prototype_node_emb.size(0))  # 原型总数
    # 每类原型数 m
    m = getattr(gnnNets.model, "num_prototypes_per_class", None)
    if m is None:
        args_like = getattr(gnnNets.model, "args", None)
        m = getattr(args_like, "num_prototypes_per_class", 1)
    m = int(m)

    for p_idx in range(P):
        proto_label = p_idx // m
        best = {"sim": float("-inf"), "data_idx": None, "sub_nodes": None, "sub_edges": None}

        # 当前原型节点嵌入（确保在同一设备）
        proto_node_emb = gnnNets.model.prototype_node_emb[p_idx].detach().to(device)

        # 遍历数据集，同类图内寻找最优连通子图
        for data_idx in range(len(dataset)):
            data = dataset[data_idx].to(device)

            # 只在标签匹配的图里搜索
            # data.y 可能是标量张量或 shape=[1]
            label_val = int(data.y.item() if data.y.numel() == 1 else int(data.y))
            if label_val != int(proto_label):
                continue

            # 前向取节点嵌入（不更新梯度）
            _, _, node_emb, _, _ = gnnNets(data)

            # 贪心/连通搜索；要求返回 (sub_nodes, sub_edges, sim)
            sub_nodes, sub_edges, sim = match_connected_subgraph(
                node_emb=node_emb,
                edge_index=data.edge_index,
                prototype_node_emb=proto_node_emb,
            )

            # 记录最优
            if sim is not None and sim > best["sim"]:
                best.update(
                    sim=float(sim),
                    data_idx=int(data_idx),
                    sub_nodes=[int(n) for n in sub_nodes] if sub_nodes is not None else None,
                    sub_edges=[(int(u), int(v)) for (u, v) in sub_edges] if sub_edges is not None else None,
                )

        # 若没有找到匹配，跳过
        if best["data_idx"] is None:
            continue

        # 保存可视化：整图 + 黑色加粗的子图边（utils 已按数据集处理节点配色）
        data = dataset[best["data_idx"]]
        png_name = f"{filename_prefix}_{proto_label}_{p_idx:03d}.png"
        out_png = os.path.join(out_dir, png_name)

        # 打印真实图索引
        print(
            f"[ProtoVis] {filename_prefix}: class={proto_label}, proto_idx={p_idx:03d}, "
            f"matched_graph_idx={best['data_idx']}, sim={best['sim']:.4f}"
        )

        plotter.save_full_graph_highlight(
            data=data,
            highlight_nodes=best["sub_nodes"] or [],
            highlight_edges=best["sub_edges"] or [],
            out_png=out_png,
            title=f"{filename_prefix} | label={proto_label} | proto={p_idx:03d} | graph={best['data_idx']} | sim={best['sim']:.3f}",
        )


@torch.no_grad()
def visualize_generated_prototypes(gnnNets,
                                   plotter,
                                   out_dir: str,
                                   filename_prefix: str = "gen_prot",
                                   *,
                                   thresh: float = 0.30,
                                   topk_edges: int | None = None,
                                   symmetric: bool = True):
    """
    在训练完成后：把“生成的原型”（decoder 的 edge_logits）可视化成独立小图导出。
    - 图片命名: f"{filename_prefix}_{label}_{proto_idx:03d}.png"
    - 这里不依赖真实图；仅画出生成的原型结构本身。
    """
    gnnNets.eval()
    os.makedirs(out_dir, exist_ok=True)

    # 用最终得到的原型进行一次生成
    gnnNets.model.generated_candidates = gnnNets.model.generate_candidate_prototypes()
    cand = gnnNets.model.generated_candidates

    edge_logits_list = cand.get("edge_logits", None) if isinstance(cand, dict) else None
    if edge_logits_list is None:
        print("[GenProtoVis] generated_candidates does not contain 'edge_logits'; skip.")
        return

    P = len(edge_logits_list)
    # 每类原型数
    m = getattr(gnnNets.model, "num_prototypes_per_class", None)
    if m is None:
        args_like = getattr(gnnNets.model, "args", None)
        m = getattr(args_like, "num_prototypes_per_class", 1)
    m = int(m)

    for p_idx in range(P):
        proto_label = p_idx // m
        logits = edge_logits_list[p_idx]
        if logits is None:
            continue

        png_name = f"{filename_prefix}_{proto_label}_{p_idx:03d}.png"
        out_png = os.path.join(out_dir, png_name)

        # 打印简单信息
        shape = list(logits.shape)
        print(f"[GenProtoVis] gen_prot: class={proto_label}, proto_idx={p_idx:03d}, edge_logits_shape={shape}")

        plotter.save_generated_prototype_png(
            edge_logits=logits,
            out_png=out_png,
            title=f"{filename_prefix} | label={proto_label} | proto={p_idx:03d}",
            thresh=thresh,
            topk_edges=topk_edges,
            symmetric=symmetric,
        )
