import os
import random
import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import shutil

from load_dataset import get_dataset, get_dataloader
from Configures import data_args, train_args, model_args
from pretrain import GnnClassifier


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


def save_best(ckpt_dir, epoch, model, model_name, eval_acc, is_best):
    print("saving....")
    model.to("cpu")
    state = {
        "net": model.state_dict(),
        "epoch": epoch,
        "acc": eval_acc,
    }
    pth_name = f"pre_{model_name}_{model_args.readout}_latest.pth"
    best_pth_name = f"pre_{model_name}_{model_args.readout}_best.pth"
    ckpt_path = os.path.join(ckpt_dir, pth_name)
    torch.save(state, ckpt_path)
    if is_best:
        shutil.copy(ckpt_path, os.path.join(ckpt_dir, best_pth_name))
    model.to(model_args.device)


def append_record(info):
    f = open('./pretrain/hyper_search.txt', 'a')
    f.write(info)
    f.write('\n')
    f.close()


def evaluate(eval_dataloader, classifier, criterion):
    acc, loss_list = [], []
    classifier.eval()
    device = torch.device('cuda:' + str(model_args.device))
    with torch.no_grad():
        for batch in eval_dataloader:
            batch = batch.to(device)
            logits, probs, _, _ = classifier(batch)
            loss = criterion(logits, batch.y)
            _, prediction = torch.max(logits, -1)
            acc.append(prediction.eq(batch.y).cpu().numpy())
            loss_list.append(loss.item())
    return {"loss": np.average(loss_list), "acc": np.concatenate(acc, axis=0).mean()}


def test(test_dataloader, classifier, criterion):
    acc, loss_list, pred_probs, predictions = [], [], [], []
    classifier.eval()
    device = torch.device('cuda:' + str(model_args.device))
    with torch.no_grad():
        for batch in test_dataloader:
            batch = batch.to(device)
            logits, probs, _, _ = classifier(batch)
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


def train():
    print("Loading dataset...")
    dataset = get_dataset(data_args.dataset_dir, data_args.dataset_name, task=data_args.task)
    input_dim = dataset.num_node_features
    output_dim = int(dataset.num_classes)
    print(f'Dataset: {dataset}:')
    print('====================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {input_dim}')
    print(f'Number of classes: {output_dim}')
    print('=============================================================')

    # Split dataset into train and test (e.g., 80% train, 10% eval, 10% test)
    dataloader = get_dataloader(dataset, train_args.batch_size, random_seed,
                                data_split_ratio=data_args.data_split_ratio)

    # train
    classifier = GnnClassifier(input_dim, output_dim, model_args)
    classifier.to_device()
    pretrain_dir = f"./pretrain/checkpoint/{data_args.dataset_name}/"
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(classifier.parameters(), lr=train_args.learning_rate, weight_decay=train_args.weight_decay)

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

    # save path for model
    if not os.path.isdir('pretrain/checkpoint'):
        os.makedirs('pretrain/checkpoint')
    if not os.path.isdir(os.path.join('pretrain/checkpoint', data_args.dataset_name)):
        os.makedirs(os.path.join('pretrain/checkpoint', f"{data_args.dataset_name}"))

    for epoch in range(train_args.max_epochs):
        classifier.train()
        acc = []
        loss_list = []
        for batch in dataloader['train']:
            logits, probs, node_emb, graph_emb = classifier(batch)
            loss = criterion(logits, batch.y)

            # optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(classifier.parameters(), clip_value=2.0)
            optimizer.step()

            # record
            _, prediction = torch.max(logits, -1)
            loss_list.append(loss.item())
            acc.append(prediction.eq(batch.y).cpu().numpy())

        # report train msg
        print(
            f"Train Epoch:{epoch}  |Loss: {np.average(loss_list):.3f} | "f"Acc: {np.concatenate(acc, axis=0).mean():.3f}")
        append_record("Epoch {:2d}, loss: {:.3f}, acc: {:.3f}".format(epoch, np.average(loss_list),
                                                                      np.concatenate(acc, axis=0).mean()))

        # report eval msg
        eval_state = evaluate(dataloader['eval'], classifier, criterion)
        print(f"Eval Epoch: {epoch} | Loss: {eval_state['loss']:.3f} | Acc: {eval_state['acc']:.3f}")
        append_record(
            "Eval epoch {:2d}, loss: {:.3f}, acc: {:.3f}".format(epoch, eval_state['loss'], eval_state['acc']))

        test_state, _, _ = test(dataloader['test'], classifier, criterion)
        print(f"Test Epoch: {epoch} | Loss: {test_state['loss']:.3f} | Acc: {test_state['acc']:.3f}")

        # only save the best model
        is_best = (eval_state['acc'] > best_acc)

        if eval_state['acc'] > best_acc:
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count > train_args.early_stopping:
            break

        if is_best:
            best_acc = eval_state['acc']
            early_stop_count = 0
        if is_best or epoch % train_args.save_epoch == 0:
            save_best(pretrain_dir, epoch, classifier, model_args.model_name, eval_state['acc'], is_best)

    print(f"The best validation accuracy is {best_acc}.")

    # report test msg
    pretrain_ckpt = torch.load(os.path.join(pretrain_dir, f'pre_{model_args.model_name}_{model_args.readout}_best.pth'),
                               weights_only=False)
    # pretrain_ckpt = torch.load(
    #     os.path.join(pretrain_dir, f'pre_{model_args.model_name}_{model_args.readout}_latest.pth'), weights_only=False)
    classifier.update_state_dict(pretrain_ckpt['net'])
    test_state, _, _ = test(dataloader['test'], classifier, criterion)
    print(
        f"Test: | Dataset: {data_args.dataset_name:s} | Loss: {test_state['loss']:.3f} | Acc: {test_state['acc']:.3f}")
    append_record(f"Dataset: {data_args.dataset_name:s} " + "loss: {:.3f}, acc: {:.3f}".format(test_state['loss'],
                                                                                               test_state['acc']))
    append_record('-' * 100)


if __name__ == '__main__':
    train()
