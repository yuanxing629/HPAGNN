import os
import torch


@torch.no_grad()
def visualize_init_prototypes(nodelist, edgelist, num_prototypes_per_class, graph_data, plotter, out_dir: str):
    # P = int(len(graph_data))  # 原型总数 C*m
    # 每类原型数 m
    m = num_prototypes_per_class

    for p_idx, data in enumerate(graph_data):
        proto_label = p_idx // m

        png_name = f"init_prot_{proto_label}_{p_idx:03d}.png"
        out_png = os.path.join(out_dir, png_name)

        # 取出当前原型对应的节点 / 边（注意索引）
        nodes_for_this_proto = nodelist[p_idx]  # 1D Tensor，原图节点 id
        edges_for_this_proto = edgelist[p_idx]  # List[[u,v], ...]

        print(
            f"[ProtoVis] init_prot: class={proto_label}, proto_idx={p_idx:03d}, "
            f"num_nodes_sub={len(nodes_for_this_proto)}, "
            f"num_edges_sub={len(edges_for_this_proto) // 2}"
        )

        plotter.save_full_graph_highlight(
            data=data,
            highlight_nodes=nodes_for_this_proto,
            highlight_edges=edges_for_this_proto,
            out_png=out_png,
            title=f"init_prot | label={proto_label} | proto={p_idx:03d}",
        )


@torch.no_grad()
def visualize_prototypes_from_search(node_list, edge_list, graph_data_list, num_prototypes_per_class, plotter,
                                     out_dir: str):
    """
    node_list: List[List[int]] (原图节点索引)
    edge_list: List[List[(u,v)]] (原图边索引)
    graph_data_list: List[Data] (源图)
    """
    m = num_prototypes_per_class

    for p_idx, (nodes, edges, data) in enumerate(zip(node_list, edge_list, graph_data_list)):
        proto_label = p_idx // m

        if not nodes:
            continue

        png_name = f"search_prot_{proto_label}_{p_idx:03d}.png"
        out_png = os.path.join(out_dir, png_name)

        # 调用 PlotUtils 的高亮函数
        # 这里传入 nodes (List[int]) 即可，utils 会负责画出这些节点以及它们之间的边
        plotter.save_full_graph_highlight(
            data=data,
            highlight_nodes=nodes,
            highlight_edges=edges,
            out_png=out_png
        )


@torch.no_grad()
def visualize_generated_prototypes(graph_data_list, num_prototypes_per_class, plotter, out_dir: str):
    """
    graph_data_list: List[Data] (直接是生成好的小图)
    """
    m = num_prototypes_per_class
    filename_prefix = "gen_prot"

    for p_idx, data in enumerate(graph_data_list):
        proto_label = p_idx // m

        png_name = f"{filename_prefix}_{proto_label}_{p_idx:03d}.png"
        out_path = os.path.join(out_dir, png_name)

        print(
            f"[GenProtoVis] gen_prot: class={proto_label}, proto_idx={p_idx:03d}, "
            f"num_nodes={data.x.shape[0]}"
        )

        # 调用 utils 中的新函数 (或者复用现有的 save_graph)
        plotter.save_standalone_graph(
            data=data,
            out_path=out_path
        )