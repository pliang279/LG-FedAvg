#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import os
import pickle
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from utils.options import args_parser
from utils.train_utils import get_data, get_model
from models.Update import DatasetSplit
from models.test import test_img_local, test_img_local_all, test_img_avg_all, test_img_ensemble_all

from torchvision import datasets, transforms
from models.Nets import MLP, CNNMnist, CNNCifar
from utils.sampling import iid, noniid, noniid_replace

import pdb

results = {}

# manually set arguments
args = args_parser()
args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
args.num_classes = 10
args.epochs = 200
if args.dataset == "mnist":
    args.local_bs = 10
else:
    args.local_bs = 50

early_stopping = 100

args.shard_per_user = 2
for num_users in [5, 10, 20, 50, 100, 200, 500, 1000]:
    results[num_users] = []
    for run in range(10):
        args.num_users = num_users

        # dataset
        dataset_train, dataset_test, dict_users_train, dict_users_test = get_data(args)

        # model
        net_glob = get_model(args)
        net_glob.train()

        net_local_list = []
        for user_ix in range(args.num_users):
            net_local_list.append(copy.deepcopy(net_glob))

        criterion = nn.CrossEntropyLoss()
        lr = args.lr

        # run each local model
        for user, net_local in enumerate(net_local_list):
            net_best = None
            best_acc = None

            ldr_train = DataLoader(DatasetSplit(dataset_train, dict_users_train[user]), batch_size=args.local_bs, shuffle=True)
            optimizer = torch.optim.SGD(net_local.parameters(), lr=lr, momentum=0.5)

            epochs_since_improvement = 0
            for iter in range(args.epochs):
                for batch_idx, (images, labels) in enumerate(ldr_train):
                    images, labels = images.to(args.device), labels.to(args.device)
                    net_local.zero_grad()
                    log_probs = net_local(images)

                    loss = criterion(log_probs, labels)
                    loss.backward()
                    optimizer.step()

                acc_test, loss_test = test_img_local(net_local, dataset_test, args, user_idx=user, idxs=dict_users_test[user])

                epochs_since_improvement += 1
                if best_acc is None or acc_test > best_acc:
                    best_acc = acc_test
                    net_best = copy.deepcopy(net_local)
                    epochs_since_improvement = 0

                print('User {}, Epoch {}, Acc {:.2f}'.format(user, iter, acc_test))
                if epochs_since_improvement >= early_stopping:
                    print("Early stopping...")
                    break

            net_local_list[user] = net_best

        acc_test_local, loss_test_local = test_img_local_all(net_local_list, args, dataset_test, dict_users_test)
        results[num_users].append(acc_test_local)

        results_save_path = "save/{}/test_local.pkl".format(args.dataset)
        with open(results_save_path, 'wb') as file:
            pickle.dump(results, file)

import pickle
import numpy as np

# with open("save/mnist/test_local.pkl",'rb') as file: x = pickle.load(file)
with open("save/cifar10/test_local.pkl",'rb') as file: x = pickle.load(file)

for key, value in x.items(): print(key, np.array(value).mean())

for key, value in x.items(): print(key, np.array(value).std())

