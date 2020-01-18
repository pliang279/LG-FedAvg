#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

class LocalUpdate_noLG(object):
    def __init__(self, args, dataset=None, pretrain=False):
        self.args = args
        self.global_loss_func = nn.BCELoss().to(args.device)
        self.adv_loss_func = nn.BCELoss(reduce=False).to(args.device)
        self.selected_clients = []
        self.ldr_train = dataset
        self.pretrain = pretrain

    def train(self, global_net, adv_model, lambdas, idx=-1, lr=0.1):
        global_net.train()
        adv_model.train()
        # train and update
        global_optimizer = torch.optim.SGD(global_net.parameters(), lr=lr, momentum=0.5)
        adv_optimizer = torch.optim.SGD(adv_model.parameters(), lr=lr, momentum=0.5)

        global_epoch_loss = []
        adv_epoch_loss = []
        if self.pretrain:
            local_eps = self.args.local_ep_pretrain
        else:
            local_eps = self.args.local_ep
        for iter in range(local_eps):
            global_batch_loss = []
            adv_batch_loss = []
            for (batch_idx, data) in enumerate(self.ldr_train):
                (images, labels, protected) = data
                images, labels, protected = images.to(self.args.device), labels.to(self.args.device), protected.to(self.args.device)
                global_net.zero_grad()
                adv_model.zero_grad()

                mid, log_probs = global_net(images)
                pred_proctected = adv_model(mid)

                global_loss = self.global_loss_func(log_probs, labels)
                global_loss.backward(retain_graph=True)
                global_optimizer.step()

                adv_loss = (self.adv_loss_func(pred_proctected, protected) * lambdas.to(self.args.device)).mean()
                adv_loss.backward()
                adv_optimizer.step()

                global_batch_loss.append(global_loss.item())
                adv_batch_loss.append(adv_loss.item())

            global_epoch_loss.append(sum(global_batch_loss)/len(global_batch_loss))
            adv_epoch_loss.append(sum(adv_batch_loss)/len(adv_batch_loss))

        w = global_net.state_dict()
        w_loss = sum(global_epoch_loss) / len(global_epoch_loss)
        adv = adv_model.state_dict()
        adv_loss = sum(adv_epoch_loss) / len(adv_epoch_loss)

        return w, w_loss, adv, adv_loss


class LocalUpdate(object):
    def __init__(self, args, dataset=None, pretrain=False):
        self.args = args
        self.loss_func = nn.BCELoss().to(args.device)
        self.adv_criterion = nn.BCELoss(reduce=False).to(args.device)
        self.selected_clients = []
        self.ldr_train = dataset
        self.pretrain = pretrain
        self.lambdas = torch.Tensor([0.03, 0.03])

    def train(self, local_net, local_opt, local_adv, adv_opt, global_net, global_opt, idx=-1, lr=0.1):
        global_net.train()
        local_net.train()
        local_adv.train()
        # train and update
        # optimizer = torch.optim.SGD(global_net.parameters(), lr=lr, momentum=0.5)

        epoch_loss = []
        if self.pretrain:
            local_eps = self.args.local_ep_pretrain
        else:
            local_eps = self.args.local_ep
        for iter in range(local_eps):
            batch_loss = []
            for (batch_idx, data) in enumerate(self.ldr_train):
                (images, labels, protected) = data
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                local_net.zero_grad()
                global_net.zero_grad()
                mid, _ = local_net(images)
                log_protected = local_adv(mid)
                log_probs = global_net(mid)
                # import pdb
                # pdb.set_trace()
                loss = self.loss_func(log_probs, labels)
                adv_loss = (self.adv_criterion(log_protected.to(self.args.device), protected.to(self.args.device)) * self.lambdas.to(self.args.device)).mean()
                if self.args.adv:
                    loss = loss - adv_loss
                loss.backward(retain_graph=True)
                adv_loss.backward()
                local_opt.step()
                global_opt.step()
                adv_opt.step()

                batch_loss.append(loss.item())

                if not self.pretrain and self.args.verbose and batch_idx % 300 == 0:
                    if idx < 0:
                        print('Update Epoch: {} [{}/{} ({:.0f}%)], Epoch Loss: {:.4f}, Batch Loss: {:.4f}'.format(
                            iter, batch_idx * len(images), len(self.ldr_train.dataset), 100. * batch_idx / len(self.ldr_train),
                            sum(batch_loss)/len(batch_loss), loss.item()))
                    else:
                        print('Local model {}, Update Epoch: {} [{}/{} ({:.0f}%)], Epoch Loss: {:.4f}, Batch Loss: {:.4f}'.format(
                            idx, iter, batch_idx * len(images), len(self.ldr_train.dataset), 100. * batch_idx / len(self.ldr_train),
                            sum(batch_loss)/len(batch_loss), loss.item()))

            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return global_net.state_dict(), sum(epoch_loss) / len(epoch_loss)

