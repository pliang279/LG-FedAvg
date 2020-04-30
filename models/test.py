#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import copy
import numpy as np
from scipy import stats
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pdb

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

def test_img(net_g, datatest, args, return_probs=False, user_idx=-1):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)

    probs = []

    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.to(args.device), target.to(args.device)
        log_probs = net_g(data)
        probs.append(log_probs)

        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * float(correct) / len(data_loader.dataset)
    if args.verbose:
        if user_idx < 0:
            print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
                test_loss, correct, len(data_loader.dataset), accuracy))
        else:
            print('Local model {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
                user_idx, test_loss, correct, len(data_loader.dataset), accuracy))

    if return_probs:
        return accuracy, test_loss, torch.cat(probs)
    return accuracy, test_loss


def test_img_local(net_g, dataset, args, user_idx=-1, idxs=None):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    # data_loader = DataLoader(dataset, batch_size=args.bs)
    data_loader = DataLoader(DatasetSplit(dataset, idxs), batch_size=args.bs, shuffle=False)
    l = len(data_loader)

    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.to(args.device), target.to(args.device)
        log_probs = net_g(data)

        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * float(correct) / len(data_loader.dataset)
    if args.verbose:
        print('Local model {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            user_idx, test_loss, correct, len(data_loader.dataset), accuracy))

    return accuracy, test_loss

def test_img_local_all(net_local_list, args, dataset_test, dict_users_test, return_all=False):
    acc_test_local = np.zeros(args.num_users)
    loss_test_local = np.zeros(args.num_users)
    for idx in range(args.num_users):
        net_local = net_local_list[idx]
        net_local.eval()
        a, b = test_img_local(net_local, dataset_test, args, user_idx=idx, idxs=dict_users_test[idx])

        acc_test_local[idx] = a
        loss_test_local[idx] = b

    if return_all:
        return acc_test_local, loss_test_local
    return acc_test_local.mean(), loss_test_local.mean()

def test_img_avg_all(net_glob, net_local_list, args, dataset_test, return_net=False):
    net_glob_temp = copy.deepcopy(net_glob)
    w_keys_epoch = net_glob.state_dict().keys()
    w_glob_temp = {}
    for idx in range(args.num_users):
        net_local = net_local_list[idx]
        w_local = net_local.state_dict()

        if len(w_glob_temp) == 0:
            w_glob_temp = copy.deepcopy(w_local)
        else:
            for k in w_keys_epoch:
                w_glob_temp[k] += w_local[k]

    for k in w_keys_epoch:
        w_glob_temp[k] = torch.div(w_glob_temp[k], args.num_users)
    net_glob_temp.load_state_dict(w_glob_temp)
    acc_test_avg, loss_test_avg = test_img(net_glob_temp, dataset_test, args)

    if return_net:
        return acc_test_avg, loss_test_avg, net_glob_temp
    return acc_test_avg, loss_test_avg

criterion = nn.CrossEntropyLoss()

def test_img_ensemble_all(net_local_list, args, dataset_test):
    probs_all = []
    preds_all = []
    for idx in range(args.num_users):
        net_local = net_local_list[idx]
        net_local.eval()
        # _, _, probs = test_img(net_local, dataset_test, args, return_probs=True, user_idx=idx)
        acc, loss, probs = test_img(net_local, dataset_test, args, return_probs=True, user_idx=idx)
        # print('Local model: {}, loss: {}, acc: {}'.format(idx, loss, acc))
        probs_all.append(probs.detach())

        preds = probs.data.max(1, keepdim=True)[1].cpu().numpy().reshape(-1)
        preds_all.append(preds)

    labels = np.array(dataset_test.targets)
    preds_probs = torch.mean(torch.stack(probs_all), dim=0)

    # ensemble (avg) metrics
    preds_avg = preds_probs.data.max(1, keepdim=True)[1].cpu().numpy().reshape(-1)
    loss_test = criterion(preds_probs, torch.tensor(labels).to(args.device)).item()
    acc_test_avg = (preds_avg == labels).mean() * 100

    # ensemble (maj)
    preds_all = np.array(preds_all).T
    preds_maj = stats.mode(preds_all, axis=1)[0].reshape(-1)
    acc_test_maj = (preds_maj == labels).mean() * 100

    return acc_test_avg, loss_test, acc_test_maj