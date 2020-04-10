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

import pdb

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    base_dir = './save/{}/{}_iid{}_num{}_C{}_le{}/shard{}/{}/'.format(
        args.dataset, args.model, args.iid, args.num_users, args.frac, args.local_ep, args.shard_per_user, args.results_save)
    if not os.path.exists(os.path.join(base_dir, 'local')):
        os.makedirs(os.path.join(base_dir, 'local'), exist_ok=True)

    dataset_train, dataset_test, dict_users_train, dict_users_test = get_data(args)
    dict_save_path = os.path.join(base_dir, 'dict_users.pkl')
    with open(dict_save_path, 'rb') as handle:
        dict_users_train, dict_users_test = pickle.load(handle)

    # build model
    net_glob = get_model(args)
    net_glob.train()

    net_local_list = []
    for user_ix in range(args.num_users):
        net_local_list.append(copy.deepcopy(net_glob))

    # training
    results_save_path = os.path.join(base_dir, 'local/results.csv')

    loss_train = []
    net_best = None
    best_loss = None
    best_acc = None
    best_epoch = None

    lr = args.lr
    results = []

    criterion = nn.CrossEntropyLoss()

    for user, net_local in enumerate(net_local_list):
        model_save_path = os.path.join(base_dir, 'local/model_user{}.pt'.format(user))
        net_best = None
        best_acc = None

        ldr_train = DataLoader(DatasetSplit(dataset_train, dict_users_train[user]), batch_size=args.local_bs, shuffle=True)
        optimizer = torch.optim.SGD(net_local.parameters(), lr=lr, momentum=0.5)
        for iter in range(args.epochs):
            for batch_idx, (images, labels) in enumerate(ldr_train):
                images, labels = images.to(args.device), labels.to(args.device)
                net_local.zero_grad()
                log_probs = net_local(images)

                loss = criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

            acc_test, loss_test = test_img_local(net_local, dataset_test, args, user_idx=user, idxs=dict_users_test[user])
            if best_acc is None or acc_test > best_acc:
                best_acc = acc_test
                net_best = copy.deepcopy(net_local)
                # torch.save(net_local_list[user].state_dict(), model_save_path)

            print('User {}, Epoch {}, Acc {:.2f}'.format(user, iter, acc_test))

            if iter > 50 and acc_test >= 99:
                break

        net_local_list[user] = net_best

    acc_test_local, loss_test_local = test_img_local_all(net_local_list, args, dataset_test, dict_users_test)
    acc_test_avg, loss_test_avg = test_img_avg_all(net_glob, net_local_list, args, dataset_test)
    acc_test_ens_avg, loss_test, acc_test_ens_maj = test_img_ensemble_all(net_local_list, args, dataset_test)

    print('Final: acc: {:.2f}, acc (avg): {:.2f}, acc (ens,avg): {:.2f}, acc (ens,maj): {:.2f}'.format(acc_test_local, acc_test_avg, acc_test_ens_avg, acc_test_ens_maj))

    final_results = np.array([[acc_test_local, acc_test_avg, acc_test_ens_avg, acc_test_ens_maj]])
    final_results = pd.DataFrame(final_results, columns=['acc_test_local', 'acc_test_avg', 'acc_test_ens_avg', 'acc_test_ens_maj'])
    final_results.to_csv(results_save_path, index=False)
