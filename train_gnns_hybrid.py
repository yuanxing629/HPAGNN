import os
import random
import torch
import torch.nn as nn

from torch.optim import Adam

import numpy as np
import shutil

from models import GnnNets
from load_dataset import get_dataset, get_dataloader
from Configures import data_args, train_args, model_args
from pretrain import GnnClassifier
from utils import PlotUtils
from visualization import visualize_init_prototypes, visualize_prototypes_on_dataset, visualize_generated_prototypes


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 关键：让 CuDNN 和算子都尽量确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


random_seed = train_args.random_seed
set_seed(random_seed)


# -----------------------------
#   模型阶段控制
# -----------------------------
def warm_only(model):
    # 梯度仍然会通过last_layer 反传到GNN和prototype
    # 只是last_layer 自己的参数不更新。
    for p in model.model.gnn_layers.parameters():
        p.requires_grad = True
    model.model.prototype_node_emb.requires_grad = True
    for p in model.model.last_layer.parameters():
        p.requires_grad = False


def joint(model):
    for p in model.model.gnn_layers.parameters():
        p.requires_grad = True
    model.model.prototype_node_emb.requires_grad = True
    for p in model.model.last_layer.parameters():
        p.requires_grad = True


def append_record(info):
    if not os.path.isdir("./log"):
        os.mkdir("./log")
    f = open("./log/hyper_search", "a")
    f.write(info + "\n")
    f.close()


def save_best(ckpt_dir, epoch, gnnNets, model_name, eval_acc, is_best):
    print("saving....")
    gnnNets.to("cpu")
    state = {
        "net": gnnNets.state_dict(),
        "epoch": epoch,
        "acc": eval_acc,
    }
    pth_name = f"{model_name}_{model_args.readout}_latest.pth"
    # 在早停策略和验证集度量下认为 “泛化最好” 的模型，对应 eval_state["acc"] 达到 历史最高 的那个 epoch 的参数。
    best_pth_name = f"{model_name}_{model_args.readout}_best.pth"
    ckpt_path = os.path.join(ckpt_dir, pth_name)
    torch.save(state, ckpt_path)
    if is_best:
        shutil.copy(ckpt_path, os.path.join(ckpt_dir, best_pth_name))
    gnnNets.to(model_args.device)


def train_GC():
    print(f"Start training HPAGNN")
    print("Loading dataset...")
    dataset = get_dataset(data_args.dataset_dir, data_args.dataset_name, task=data_args.task)
    input_dim = dataset.num_node_features
    output_dim = int(dataset.num_classes)
    dataloader = get_dataloader(dataset, train_args.batch_size, random_seed,
                                data_split_ratio=data_args.data_split_ratio)

    # -------------------------------
    # Step 1: model Initialization
    # -------------------------------
    print("Initializing model...")
    gnnNets = GnnNets(input_dim, output_dim, model_args, data_args)
    gnnNets.to_device()

    # ===================================================
    #  Step 2: Prototype Initialization & Build supervised target adjacency for each prototype
    # ===================================================
    print("Performing Prototype Initialization (based on high-confidence samples)...")
    # 预训练的分类器给出了置信度得分
    classifier = GnnClassifier(input_dim, output_dim, model_args)
    classifier.to_device()
    pretrain_dir = f'./pretrain/checkpoint/{data_args.dataset_name}/'
    pretrain_ckpt = torch.load(os.path.join(pretrain_dir, f'pre_{model_args.model_name}_{model_args.readout}_best.pth'),
                               weights_only=False)
    classifier.update_state_dict(pretrain_ckpt['net'])
    init_proto_graph_emb = None
    try:
        # 初始化原型的图级嵌入init_proto_graph_emb将用于后续约束原型的学习过程
        # 其余信息用于调试或可视化
        init_proto_node_emb, init_proto_graph_emb, init_proto_node_list, init_proto_edge_list, init_proto_in_which_graph, info_dict = gnnNets.model.initialize_prototypes(
            trainloader=dataloader['train'], classifier=classifier)
    except Exception as e:
        print(f"Warning: Prototype initialization failed due to {e}. Using random initialization instead.")
    else:
        # 只有初始化成功才可视化：可视化的异常不会影响初始化结果
        try:
            init_vis_dir = os.path.join("checkpoint", data_args.dataset_name, "init_prot")
            print(init_proto_node_list)
            print(init_proto_edge_list)
            plotter = PlotUtils(dataset_name=data_args.dataset_name)
            visualize_init_prototypes(
                nodelist=init_proto_node_list,
                edgelist=init_proto_edge_list,
                num_prototypes_per_class=model_args.num_prototypes_per_class,
                graph_data=init_proto_in_which_graph,  # 保存的原始 Data 列表
                plotter=plotter,
                out_dir=init_vis_dir
            )
        except Exception as e:
            print(f"Warning: init prototype visualization failed due to {e}.")
    # if init_proto_graph_emb is not None:
    #     init_proto_graph_emb = init_proto_graph_emb.detach()  # 去掉历史计算图
    #     init_proto_graph_emb.requires_grad_(False)  # 明确不参与梯度
    #     init_proto_graph_emb = init_proto_graph_emb.to(model_args.device)

    # -------------------------------
    # Step 3: Training preparation
    # -------------------------------
    ckpt_dir = f"./checkpoint/{data_args.dataset_name}/"
    os.makedirs(ckpt_dir, exist_ok=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(gnnNets.parameters(), lr=train_args.learning_rate,
                     weight_decay=train_args.weight_decay)

    avg_nodes = 0.0
    avg_edge_index = 0.0
    for i in range(len(dataset)):
        avg_nodes += dataset[i].x.shape[0]
        avg_edge_index += dataset[i].edge_index.shape[1]
    avg_nodes /= len(dataset)
    avg_edge_index /= len(dataset)
    print(f"graphs {len(dataset)}, avg_nodes{avg_nodes :.4f}, avg_edge_index_{avg_edge_index / 2 :.4f}")

    best_acc = 0.0
    early_stop_count = 0
    data_size = len(dataset)
    print(f'The total num of dataset is {data_size}')

    # save path for model
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    if not os.path.isdir(os.path.join('checkpoint', data_args.dataset_name)):
        os.makedirs(os.path.join('checkpoint', f"{data_args.dataset_name}"))

    # -------------------------------
    # Step 4: Training loop
    # -------------------------------
    for epoch in range(train_args.max_epochs):
        acc = []
        loss_list = []  # 总损失
        cls_loss_list = []  # 分类损失
        proto_loss_list = []  # 约束原型学习过程
        rec_loss_list = []  # 重构损失
        align_loss_list = []  # 约束生成的原型
        div_loss_list = []  # 防止同类原型过于相似

        # 每个 epoch 开始时让生成候选失效，后续 compute_reconstruction_loss 会重新生成
        gnnNets.model.generated_candidates = None

        gnnNets.train()
        if epoch < train_args.warm_epochs:
            warm_only(gnnNets)
        else:
            joint(gnnNets)

        for batch in dataloader["train"]:
            logits, probs, node_emb, graph_emb, min_distances = gnnNets(batch)
            cls_cost = criterion(logits, batch.y)

            # proto & reconstruction & alignment & diversity losses
            proto_cost = gnnNets.model.compute_proto_loss(init_proto_graph_emb=init_proto_graph_emb)
            rec_cost = gnnNets.model.compute_reconstruction_loss()  # 在计算重构损失的时候会进行生成
            align_cost = gnnNets.model.compute_alignment_loss()
            div_cost = gnnNets.model.compute_diversity_loss()

            loss = cls_cost + train_args.lambda_proto * proto_cost + train_args.lambda_rec * rec_cost + train_args.lambda_align * align_cost + train_args.lambda_div * div_cost

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(gnnNets.parameters(), clip_value=2.0)
            optimizer.step()

            _, prediction = torch.max(logits, -1)

            loss_list.append(loss.item())
            cls_loss_list.append(cls_cost.item())
            proto_loss_list.append(proto_cost.item())
            rec_loss_list.append(rec_cost.item())
            align_loss_list.append(align_cost.item())
            div_loss_list.append(div_cost.item())
            acc.append(prediction.eq(batch.y).cpu().numpy())

        train_acc = np.concatenate(acc, axis=0).mean()
        train_loss = np.average(loss_list)
        l_cls = np.average(cls_loss_list)
        l_proto = np.average(proto_loss_list)
        l_rec = np.average(rec_loss_list)
        l_align = np.average(align_loss_list)
        l_div = np.average(div_loss_list)

        append_record(
            f"Epoch {epoch}, "
            f"Train loss: {train_loss:.3f}, "
            f"cls: {l_cls:.3f}, "
            f"proto: {l_proto:.3f}, "
            f"rec: {l_rec:.3f}, "
            f"align: {l_align:.3f}, "
            f"div: {l_div:.3f}, "
            f"acc: {train_acc:.3f}, "
        )

        print(
            f"Epoch {epoch:03d} | "
            f"Acc: {train_acc:.3f} | Loss: {train_loss:.3f} | "
            f"Lcls: {l_cls:.3f} | Lproto: {l_proto:.3f} | Lrec: {l_rec:.3f} | "
            f"Lalign: {l_align:.3f} | Ldiv: {l_div:.3f} | "
        )

        # -------------------------------
        # Step 5: Validation
        # -------------------------------
        eval_state = evaluate_GC(dataloader["eval"], gnnNets, criterion)
        print(f"Eval | Loss: {eval_state['loss']:.3f} | Acc: {eval_state['acc']:.3f}")
        append_record(f"Eval epoch {epoch}, loss: {eval_state['loss']:.3f}, acc: {eval_state['acc']:.3f}")

        is_best = eval_state["acc"] > best_acc
        if is_best:
            best_acc = eval_state["acc"]
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count > train_args.early_stopping:
            print("Early stopping triggered.")
            break

        if is_best or epoch % train_args.save_epoch == 0:
            save_best(ckpt_dir, epoch, gnnNets, model_args.model_name, eval_state["acc"], is_best)

    # -------------------------------
    # Step 6: Final Test on Best ckpt
    # -------------------------------
    print(f"Best validation accuracy: {best_acc:.3f}")
    checkpoint = torch.load(os.path.join(ckpt_dir, f"{model_args.model_name}_{model_args.readout}_best.pth"),
                            weights_only=False)
    gnnNets.update_state_dict(checkpoint["net"])
    test_state, _, _ = test_GC(dataloader["test"], gnnNets, criterion)
    print(f"Test | Loss: {test_state['loss']:.3f} | Acc: {test_state['acc']:.3f}")
    append_record(f"Test loss: {test_state['loss']:.3f}, acc: {test_state['acc']:.3f}")
    append_record('-' * 100)

    # -------------------------------
    # Step 7: After-training projection for interpretation
    # -------------------------------
    # 这里只是为了让原型对应到真实子图，不再继续训练或改变测试结果
    plotter = PlotUtils(dataset_name=data_args.dataset_name)
    search_vis_dir = os.path.join("checkpoint", data_args.dataset_name, "search_prot")
    visualize_prototypes_on_dataset(
        gnnNets=gnnNets,
        dataset=dataset,
        plotter=plotter,
        out_dir=search_vis_dir,
        filename_prefix="search_prot",
    )

    gen_vis_dir = os.path.join("checkpoint", data_args.dataset_name, "gen_prot")
    plotter = PlotUtils(dataset_name=data_args.dataset_name)
    visualize_generated_prototypes(
        gnnNets=gnnNets,
        plotter=plotter,
        out_dir=gen_vis_dir,
        filename_prefix="gen_prot",
        # 二选一（示例：优先用 topk 显示更清爽）
        topk_edges=6,  # 仅取概率最高的  条边
        thresh=None,  # 若使用 topk_edges，可设为 None 或忽略
        symmetric=True,  # 对称化保证无向
    )


# -----------------------------
#   Evaluation & Test
# -----------------------------
def evaluate_GC(eval_dataloader, gnnNets, criterion):
    acc, loss_list = [], []
    gnnNets.eval()
    with torch.no_grad():
        for batch in eval_dataloader:
            batch = batch.to(model_args.device)
            logits, probs, _, _, _ = gnnNets(batch)
            loss = criterion(logits, batch.y)
            _, prediction = torch.max(logits, -1)
            acc.append(prediction.eq(batch.y).cpu().numpy())
            loss_list.append(loss.item())
    return {"loss": np.average(loss_list), "acc": np.concatenate(acc, axis=0).mean()}


def test_GC(test_dataloader, gnnNets, criterion):
    acc, loss_list, pred_probs, predictions = [], [], [], []
    gnnNets.eval()
    with torch.no_grad():
        for batch in test_dataloader:
            batch = batch.to(model_args.device)
            logits, probs, _, _, _ = gnnNets(batch)
            loss = criterion(logits, batch.y)
            _, prediction = torch.max(logits, -1)
            acc.append(prediction.eq(batch.y).cpu().numpy())
            loss_list.append(loss.item())
            predictions.append(prediction)
            pred_probs.append(probs)
    test_state = {"loss": np.average(loss_list), "acc": np.concatenate(acc, axis=0).mean()}
    pred_probs = torch.cat(pred_probs, dim=0).cpu().detach().numpy()
    predictions = torch.cat(predictions, dim=0).cpu().detach().numpy()
    return test_state, pred_probs, predictions


# -----------------------------
#   入口函数
# -----------------------------
if __name__ == "__main__":
    train_GC()
