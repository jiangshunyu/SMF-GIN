import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import json

from numpy import linalg as LA
from scipy.stats import mode

from util import load_data, data_load_data, segregate
from util import valtest_load_data
from models.graphcnn import GraphCNN
import matplotlib.pyplot as plt

criterion = nn.CrossEntropyLoss()
from torch.optim.lr_scheduler import MultiStepLR, StepLR, CosineAnnealingLR


def get_scheduler(batches, optimiter, args):
    """
    cosine will change learning rate every iteration, others change learning rate every epoch
    :param batches: the number of iterations in each epochs
    :return: scheduler
    """
    SCHEDULER = {'step': StepLR(optimiter, 30, 0.1),
                 'multi_step': MultiStepLR(optimiter,
                                           milestones=[int(.5 * args.iterations), int(.75 * args.iterations)],
                                           gamma=0.1),
                 'cosine': CosineAnnealingLR(optimiter, batches * args.iterations, eta_min=1e-9)}
    return SCHEDULER[args.scheduler]


def get_optimizer(module, args):
    OPTIMIZER = {'SGD': torch.optim.SGD(module.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay,
                                        nesterov=args.nesterov),
                 'Adam': torch.optim.Adam(module.parameters(), lr=args.lr)}
    return OPTIMIZER[args.optimizer]


def valtest_batch(test_graph, iteration, task, ways, spt_shots, qry_shots, args, retrain=False):
    way = ways[task]
    spt_shot = spt_shots[task]
    qry_shot = qry_shots[task]

    x_spt = []
    x_qry = []
    y_spt = []
    y_qry = []

    for i, j in enumerate(way):
        for m in spt_shot[i]:
            x_spt.append(test_graph[j][m])
        for n in qry_shot[i]:
            x_qry.append(test_graph[j][n])

    for i in range(args.num_ways):
        y_spt.extend([i] * args.spt_shots)
        y_qry.extend([i] * args.qry_shots)

    return x_spt, y_spt, x_qry, y_qry


def train_batch(train_graph, iteration, task, ways, spt_shots, qry_shots, args, retrain=False):
    i = iteration - 1
    way = ways[i][task]
    spt_shot = spt_shots[i][task]
    qry_shot = qry_shots[i][task]

    x_spt = []
    x_qry = []
    y_spt = []
    y_qry = []

    for i, j in enumerate(way):
        for m in spt_shot[i]:
            x_spt.append(train_graph[j][m])
        for n in qry_shot[i]:
            x_qry.append(train_graph[j][n])

    for i in range(args.num_ways):
        y_spt.extend([i] * args.spt_shots)
        y_qry.extend([i] * args.qry_shots)

    return x_spt, y_spt, x_qry, y_qry


def get_metric(metric_type):
    METRICS = {
        'cosine': lambda gallery, query: 1. - F.cosine_similarity(query[:, None, :], gallery[None, :, :], dim=2),
        'euclidean': lambda gallery, query: ((query[:, None, :] - gallery[None, :, :]) ** 2).sum(2),
        'l1': lambda gallery, query: torch.norm((query[:, None, :] - gallery[None, :, :]), p=1, dim=2),
        'l2': lambda gallery, query: torch.norm((query[:, None, :] - gallery[None, :, :]), p=2, dim=2),
    }
    return METRICS[metric_type]


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        # .sum().cpu().item()

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def metric_prediction(i, j, gallery, query, train_label, metric_type):
    gallery = gallery.view(gallery.shape[0], -1)
    query = query.view(query.shape[0], -1)
    distance = get_metric(metric_type)(gallery, query)

    predict = torch.argmin(distance, dim=1)
    predict = torch.take(train_label, predict)

    return predict


def compute_confidence_interval(data):
    """
    Compute 95% confidence interval
    :param data: An array of mean accuracy (or mAP) across a number of sampled episodes.
    :return: the 95% confidence interval for this data.
    """
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m, pm


def train(train_graph, args, iteration, train_way, train_spt_shot, train_qry_shot, model, device, optimizer):
    model.train()
    losses = []  # losses_q[i] is the loss on step i
    correct = []

    for i in range(args.task_num):
        x_spt, y_spt, x_qry, y_qry = train_batch(train_graph, iteration, i, train_way, train_spt_shot, train_qry_shot,
                                                 args, retrain=True)
        y_spt = torch.LongTensor(y_spt).to(device)
        y_qry = torch.LongTensor(y_qry).to(device)


        spt_output1, spt_output2, spt_output3, spt_output4 = model(x_spt)
        qry_output1, qry_output2, qry_output3, qry_output4 = model(x_qry)
        spt_output = model.attention_global(spt_output1, spt_output2, spt_output3, spt_output4)
        qry_output = model.attention_global(qry_output1, qry_output2, qry_output3, qry_output4)

        spt_output = spt_output.reshape(args.num_ways, args.spt_shots, -1).mean(1)
        output = -get_metric(args.meta_train_metric)(spt_output, qry_output)
        spt_loss = criterion(output, y_qry)

        optimizer.zero_grad()
        spt_loss.backward()
        optimizer.step()
        y_spt = y_spt[::args.spt_shots]
        prediction = metric_prediction(1, 1, spt_output, qry_output, y_spt, args.meta_val_metric)
        acc = (prediction == y_qry).float().mean()

        losses.append(spt_loss.cpu().detach().numpy())
        correct.append(acc.cpu().detach().numpy())
    loss = np.array(losses).mean(axis=0)
    acc = np.array(correct).mean(axis=0)

    return loss, acc


def val(val_graph, args, iteration, val_way, val_spt_shot, val_qry_shot, model, device):
    model.eval()
    accs = []

    for i in range(args.val_task_num):
        x_spt, y_spt, x_qry, y_qry = valtest_batch(val_graph, iteration, i, val_way, val_spt_shot, val_qry_shot, args)
        y_spt = torch.LongTensor(y_spt).to(device)
        y_qry = torch.LongTensor(y_qry).to(device)

        spt_output1, spt_output2, spt_output3, spt_output4 = model(x_spt)
        qry_output1, qry_output2, qry_output3, qry_output4 = model(x_qry)
        spt_output = model.attention_global(spt_output1, spt_output2, spt_output3, spt_output4)
        qry_output = model.attention_global(qry_output1, qry_output2, qry_output3, qry_output4)

        spt_output = spt_output.reshape(args.num_ways, args.spt_shots, -1).mean(1)
        y_spt = y_spt[::args.spt_shots]
        prediction = metric_prediction(iteration, i, spt_output, qry_output, y_spt, args.meta_val_metric)
        acc = (prediction == y_qry).float().mean()

        accs.append(acc.cpu().detach().numpy())

    acc_mean, conf = compute_confidence_interval(accs)

    return acc_mean, conf


def test(train_graph, test_graph, args, iteration, test_way, test_spt_shot, test_qry_shot, model, device):
    model.eval()
    accs = []
    out_mean = []

    for j in range(len(train_graph)):
        x = train_graph[j]
        y1, y2, y3, y4 = model(x)
        y = model.adjust_weight(y1, y2, y3, y4)
        out_mean.append(y.cpu().data.numpy())
    train_mean = np.concatenate(out_mean, axis=0).mean(0)

    for i in range(args.test_task_num):

        x_spt, y_spt, x_qry, y_qry = valtest_batch(test_graph, iteration, i, test_way, test_spt_shot, test_qry_shot,
                                                   args)

        spt_output1, spt_output2, spt_output3, spt_output4 = model(x_spt)
        qry_output1, qry_output2, qry_output3, qry_output4 = model(x_qry)
        spt_output = model.attention_global(spt_output1, spt_output2, spt_output3, spt_output4)
        qry_output = model.attention_global(qry_output1, qry_output2, qry_output3, qry_output4)

        spt_output = spt_output.cpu().detach().numpy().astype(np.float32)
        qry_output = qry_output.cpu().detach().numpy().astype(np.float32)

        if args.norm_type == 'CL2N':
            spt_output = spt_output - train_mean
            spt_output = spt_output / LA.norm(spt_output, 2, 1)[:, None]
            qry_output = qry_output - train_mean
            qry_output = qry_output / LA.norm(qry_output, 2, 1)[:, None]
        elif args.norm_type == 'L2N':
            spt_output = spt_output / LA.norm(spt_output, 2, 1)[:, None]
            qry_output = qry_output / LA.norm(qry_output, 2, 1)[:, None]
        spt_output = spt_output.reshape(args.num_ways, args.spt_shots, spt_output.shape[-1]).mean(1)
        y_spt = y_spt[::args.spt_shots]
        subtract = spt_output[:, None, :] - qry_output
        distance = LA.norm(subtract, 2, axis=-1)

        idx = np.argpartition(distance, args.num_NN, axis=0)[:args.num_NN]
        nearest_samples = np.take(y_spt, idx)
        out = mode(nearest_samples, axis=0)[0]
        out = out.astype(int)
        y_qry = np.array(y_qry)
        acc = (out == y_qry).mean()

        accs.append(acc)

    acc_mean, conf = compute_confidence_interval(accs)

    return acc_mean, conf


def main():
    # Training settings
    # Note: Hyper-parameters need to be tuned in order to obtain results reported in the paper.
    parser = argparse.ArgumentParser(
        description='PyTorch graph convolutional neural net for whole-graph classification')
    parser.add_argument('--dataset', type=str, default="TRIANGLES",
                        help='name of dataset (default: TRIANGLES)')
    parser.add_argument('--device', type=int, default=2,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--iterations', type=int, default=700,
                        help='number of epochs to train (default: 350)')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='number of layers INCLUDING the input one (default: 5)')
    parser.add_argument('--meta_train_metric', type=str, choices=('euclidean', 'cosine', 'l1', 'l2'),
                        default='l2',
                        help='meta-train evaluate metric')
    parser.add_argument('--meta_val_metric', type=str, choices=('euclidean', 'cosine', 'l1', 'l2'),
                        default='l2',
                        help='meta-train evaluate metric')
    parser.add_argument('--scheduler', type=str, choices=('step', 'multi_step', 'cosine'),
                        default='multi_step',
                        help='meta-train evaluate metric')
    parser.add_argument('--norm_type', type=str, choices=('L2N', 'CL2N', 'UN'),
                        default='CL2N')
    parser.add_argument('--type', type=str, choices=('local', 'global'),
                        default='local')
    parser.add_argument('--attention_type', type=str, choices=('weight', 'mlp','attention','self-attention','transformer'),
                        default='weight')
    parser.add_argument('--optimizer', default='Adam', choices=('SGD', 'Adam'))
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--num_NN', type=int, default=1,
                        help='number of nearest neighbors, set this number >1 when do kNN')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--nesterov', action='store_true',
                        help='use nesterov for SGD, disable it in default')
    parser.add_argument('--num_ways', type=int, default=3)
    parser.add_argument('--spt_shots', type=int, default=5)
    parser.add_argument('--qry_shots', type=int, default=15)
    parser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=32)
    parser.add_argument('--test_task_num', type=int, help='meta batch size, namely task num', default=500)
    parser.add_argument('--val_task_num', type=int, help='meta batch size, namely task num', default=500)
    parser.add_argument('--num_mlp_layers', type=int, default=2,
                        help='number of layers for MLP EXCLUDING the input one (default: 2). 1 means linear model.')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='number of hidden units (default: 64)')
    parser.add_argument('--final_dropout', type=float, default=0.2,
                        help='final layer dropout (default: 0.5)')
    parser.add_argument('--graph_pooling_type', type=str, default="sum", choices=["sum", "average"],
                        help='Pooling for over nodes in a graph: sum or average')
    parser.add_argument('--neighbor_pooling_type', type=str, default="sum", choices=["sum", "average", "max"],
                        help='Pooling for over neighboring nodes: sum, average or max')
    parser.add_argument('--learn_eps', action="store_true", default=False,
                        help='Whether to learn the epsilon weighting for the center nodes. Does not affect training accuracy though.')
    parser.add_argument('--degree_as_tag', action="store_true",
                        help='let the input node features be the degree of nodes (heuristics for unlabeled graph)')
    parser.add_argument('--input_model_file', type=str, default="/home/jsy/SimpleShot/data_weight_save/",
                        help='filename to read the model (if there is any)')
    args = parser.parse_args()

    # set up seeds and gpu device
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    all_graphs, label_dict = data_load_data(args.dataset, True)
    train_graph, test_graph = segregate(args, all_graphs, label_dict)

    # input_dim = 718
    input_dim = train_graph[0][0].node_features.shape[1]
    num_classes = args.num_ways

    model = GraphCNN(args, args.num_layers, args.num_mlp_layers, input_dim, args.hidden_dim, num_classes, args.final_dropout,
                     args.learn_eps, args.graph_pooling_type, args.neighbor_pooling_type, device).to(device)


    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(args.task_num, optimizer, args)

    with open("./dataset/dataset/TRIANGLES/way_shot/train-way.txt", 'r') as f:
        train_way = f.read()
        train_way = json.loads(train_way)
    with open("./dataset/dataset/TRIANGLES/way_shot/train-spt-shot.txt", 'r') as f:
        train_spt_shot = f.read()
        train_spt_shot = json.loads(train_spt_shot)
    with open("./dataset/dataset/TRIANGLES/way_shot/train-qry-shot.txt", 'r') as f:
        train_qry_shot = f.read()
        train_qry_shot = json.loads(train_qry_shot)
    with open("./dataset/dataset/TRIANGLES/way_shot/test-way.txt", 'r') as f:
        test_way = f.read()
        test_way = json.loads(test_way)
    with open("./dataset/dataset/TRIANGLES/way_shot/test-spt-shot.txt", 'r') as f:
        test_spt_shot = f.read()
        test_spt_shot = json.loads(test_spt_shot)
    with open("/./dataset/dataset/TRIANGLES/way_shot/test-qry-shot.txt", 'r') as f:
        test_qry_shot = f.read()
        test_qry_shot = json.loads(test_qry_shot)
    with open("./dataset/dataset/TRIANGLES/way_shot/val-way.txt", 'r') as f:
        val_way = f.read()
        val_way = json.loads(val_way)
    with open("./dataset/dataset/TRIANGLES/way_shot/val-spt-shot.txt", 'r') as f:
        val_spt_shot = f.read()
        val_spt_shot = json.loads(val_spt_shot)
    with open("./dataset/dataset/TRIANGLES/way_shot/val-qry-shot.txt", 'r') as f:
        val_qry_shot = f.read()
        val_qry_shot = json.loads(val_qry_shot)

    print(args)


    acc1 = []
    acc2 = []
    acc3 = []
    Train_Loss_list = []
    v_conf = []
    t_conf = []

    for iteration in range(1, args.iterations + 1):
        scheduler.step()

        train_loss, train_acc = train(train_graph, args, iteration, train_way, train_spt_shot, train_qry_shot, model,
                                      device,
                                      optimizer)
        acc1.append(train_acc)
        Train_Loss_list.append(train_loss)

        if iteration % 20 == 0:

            # torch.save(model.state_dict(), ("/home/jsy/SimpleShot/save6/{}.pth").format(iteration))
            val_acc, val_conf = val(test_graph, args, iteration, val_way, val_spt_shot, val_qry_shot, model, device)
            test_acc, test_conf = test(train_graph, test_graph, args, iteration, test_way, test_spt_shot, test_qry_shot,
                                       model, device,
                                       )
            acc2.append(val_acc)
            acc3.append(test_acc)
            t_conf.append(test_conf)

            print('iteration', iteration, ':', 'train_loss:', train_loss, 'train_acc:', train_acc,
                  'val_acc:', val_acc, 'val_conf:', val_conf, 'test_acc:', test_acc, 'test_conf:', test_conf)

        else:
            print('iteration', iteration, ':', 'train_loss:', train_loss, 'train_acc:', train_acc)

        if iteration == args.iterations:
            print('train_mean:', np.mean(acc1), 'train_max:', np.max(acc1),
                  'val_mean:', np.mean(acc2), 'val_max:', np.max(acc2),
                  'test_mean:', np.mean(acc3), 'test_max:', np.max(acc3))
            val_max = max(acc2)
            test_max = acc3[acc2.index(val_max)]
            conf = t_conf[acc2.index(val_max)]
            print('iteration', (acc2.index(val_max) + 1) * 20, ':val_max_acc:', val_max, 'test_acc:', test_max,
                  'test_conf:', conf)


if __name__ == '__main__':
    main()
