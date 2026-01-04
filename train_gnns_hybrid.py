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
from visualization import visualize_init_prototypes
from torch_geometric.loader import DataLoader
import time
from torch.optim.lr_scheduler import CosineAnnealingLR
from metrics_eval import ExplanationEvaluator


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 关键：让 CuDNN 和算子都尽量确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


random_seed = train_args.random_seed
set_seed(random_seed)


def warm_only(model):
    # 梯度仍然会通过last_layer 反传到GNN和prototype
    # 只是last_layer 自己的参数不更新。
    for p in model.model.gnn_layers.parameters():
        p.requires_grad = True
    for p in model.model.prototype_node_emb:
        p.requires_grad = True
    for p in model.model.last_layer.parameters():
        p.requires_grad = False


def joint(model):
    for p in model.model.gnn_layers.parameters():
        p.requires_grad = True
    for p in model.model.prototype_node_emb:
        p.requires_grad = True
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

    gnnNets = GnnNets(input_dim, output_dim, model_args)
    gnnNets.to_device()

    # initialization
    print("Performing Prototype Initialization (based on high-confidence samples)...")
    # 预训练的分类器给出了置信度得分
    classifier = GnnClassifier(input_dim, output_dim, model_args)
    classifier.to_device()
    pretrain_dir = f'./pretrain/checkpoint/{data_args.dataset_name}/'
    pretrain_ckpt = torch.load(os.path.join(pretrain_dir, f'pre_{model_args.model_name}_{model_args.readout}_best.pth'),
                               weights_only=False)
    classifier.update_state_dict(pretrain_ckpt['net'])

    start_time = time.time()

    (init_proto_node_emb, init_proto_graph_emb, init_proto_node_list,
     init_proto_edge_list, init_proto_in_which_graph, info_dict) = gnnNets.model.initialize_prototypes(
        trainloader=dataloader['train'], classifier=classifier)

    # 初始可视化（可选）
    init_vis_dir = os.path.join("checkpoint", data_args.dataset_name, "init_prot")
    # print(init_proto_node_list)
    # print(init_proto_edge_list)
    plotter = PlotUtils(dataset_name=data_args.dataset_name)
    visualize_init_prototypes(
        nodelist=init_proto_node_list,
        edgelist=init_proto_edge_list,
        num_prototypes_per_class=model_args.num_prototypes_per_class,
        graph_data=init_proto_in_which_graph,  # 保存的原始 Data 列表
        plotter=plotter,
        out_dir=init_vis_dir
    )

    ckpt_dir = f"./checkpoint/{data_args.dataset_name}/"
    os.makedirs(ckpt_dir, exist_ok=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(gnnNets.parameters(), lr=train_args.learning_rate,
                     weight_decay=train_args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=train_args.max_epochs, eta_min=1e-5)

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

    for epoch in range(train_args.max_epochs):
        acc = []
        loss_list = []  # 总损失
        cls_loss_list = []  # 分类损失
        proto_loss_list = []  # 约束原型学习过程
        rec_loss_list = []  # 重构损失
        align_loss_list = []  # 约束生成的原型
        div_loss_list = []  # 防止同类原型过于相似

        # [新增] 周期性动态锚点更新 (Hybrid Mechanism Core)
        if epoch >= train_args.proj_epochs and epoch % 20 == 0:
            print(f"Epoch {epoch}: Refreshing source pool (Top-20%) and updating anchors...")

            # 调用新函数：同时完成筛选源图 + Beam Search + 更新 Anchor
            # 传入 dataloader['train'] 用于全量评估
            gnnNets.model.refresh_and_update_anchors(
                trainloader=dataloader['train'],
                top_k_ratio=0.2
            )

        # 每个 Epoch 开始时清空一次，强制 alignment_loss 在本 Epoch 第一个 Batch 时生成一次
        # 随后的 Batch 将直接复用，不再运行 NetworkX
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
            proto_cost = gnnNets.model.proto_loss()  # proto_cost 现在计算的是 MSE(curr, dynamic_anchor)
            rec_cost = gnnNets.model.reconstruction_loss(batch)
            align_cost = gnnNets.model.alignment_loss()
            div_cost = gnnNets.model.diversity_loss()

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

        scheduler.step()

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

        # validation
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

    # test on best ckpt
    end_time = time.time()

    print(f"the trining time is: {end_time - start_time}")
    print(f"Best validation accuracy: {best_acc:.3f}")
    checkpoint = torch.load(os.path.join(ckpt_dir, f"{model_args.model_name}_{model_args.readout}_best.pth"),
                            weights_only=False)
    gnnNets.update_state_dict(checkpoint["net"])
    test_state, _, _ = test_GC(dataloader["test"], gnnNets, criterion)
    print(f"Test | Loss: {test_state['loss']:.3f} | Acc: {test_state['acc']:.3f}")
    append_record(f"Test loss: {test_state['loss']:.3f}, acc: {test_state['acc']:.3f}")
    append_record('-' * 100)

    run_explanation_metrics(gnnNets, dataloader,model_args)



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


def run_explanation_metrics(gnnNets, dataloader, model_args):
    print("\n" + "=" * 30)
    print("Running Explanation Metrics...")
    print("=" * 30)

    evaluator = ExplanationEvaluator(gnnNets.model, model_args)

    # 1. 计算 Silhouette Score (原型质量)
    sil_score = evaluator.evaluate_silhouette()
    print(f"Prototype Silhouette Score: {sil_score:.4f} ↑")

    # 2. 计算 Fidelity (解释忠实度)
    # 使用 test dataloader
    # test_dataset = dataloader["test"].dataset
    # fidelity_metrics = evaluator.evaluate_fidelity(test_dataset, num_samples=len(test_dataset))
    # print(f"Fidelity+ (Occlusion): {fidelity_metrics['fidelity_plus']:.4f} ↑")
    # print(f"Fidelity- (Sparsity) : {fidelity_metrics['fidelity_minus']:.4f} ↓")

    # 3. AUC (仅当数据集有 GT 时)
    # auc_score = evaluator.evaluate_auc(test_dataset)
    # if auc_score:
    #     print(f"Explanation AUC: {auc_score:.4f}")

    print("=" * 30 + "\n")
# -----------------------------
#   入口函数
# -----------------------------
if __name__ == "__main__":
    train_GC()
