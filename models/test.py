#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

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
            data, target = data.cuda(), target.cuda()
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
    else:
        return accuracy, test_loss


def test_img_local(net_g, dataset, args, user_idx=-1, idxs=None):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    # data_loader = DataLoader(dataset, batch_size=args.bs)
    data_loader = DataLoader(DatasetSplit(dataset, idxs), batch_size=args.bs, shuffle=True)
    l = len(data_loader)

    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.cuda(), target.cuda()
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
