#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
import copy
import os
import pickle
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import mode
from torchvision import datasets, transforms, models
import torch
from torch import nn

from utils.train_utils import get_model, get_data
from utils.options import args_parser
from models.Update import LocalUpdateMTL
from models.test import test_img, test_img_local, test_img_local_all, test_img_avg_all, test_img_ensemble_all


import pdb

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    base_dir = './save/{}/{}_iid{}_num{}_C{}_le{}/shard{}/{}/'.format(
        args.dataset, args.model, args.iid, args.num_users, args.frac, args.local_ep, args.shard_per_user, args.results_save)

    base_save_dir = os.path.join(base_dir, 'mtl')
    if not os.path.exists(base_save_dir):
        os.makedirs(base_save_dir, exist_ok=True)

    dataset_train, dataset_test, dict_users_train, dict_users_test = get_data(args)
    dict_save_path = os.path.join(base_dir, 'dict_users.pkl')
    with open(dict_save_path, 'rb') as handle:
        dict_users_train, dict_users_test = pickle.load(handle)

    # build model
    net_glob = get_model(args)
    net_glob.train()

    print(net_glob)
    net_glob.train()

    total_num_layers = len(net_glob.weight_keys)
    w_glob_keys = net_glob.weight_keys[total_num_layers - args.num_layers_keep:]
    w_glob_keys = list(itertools.chain.from_iterable(w_glob_keys))

    num_param_glob = 0
    num_param_local = 0
    for key in net_glob.state_dict().keys():
        num_param_local += net_glob.state_dict()[key].numel()
        if key in w_glob_keys:
            num_param_glob += net_glob.state_dict()[key].numel()
    percentage_param = 100 * float(num_param_glob) / num_param_local
    print('# Params: {} (local), {} (global); Percentage {:.2f} ({}/{})'.format(
        num_param_local, num_param_glob, percentage_param, num_param_glob, num_param_local))

    # generate list of local models for each user
    net_local_list = []
    for user_ix in range(args.num_users):
        net_local_list.append(copy.deepcopy(net_glob))

    criterion = nn.CrossEntropyLoss()

    # training
    results_save_path = os.path.join(base_save_dir, 'results.csv')

    loss_train = []
    net_best = None
    best_acc = np.ones(args.num_users) * -1
    best_net_list = copy.deepcopy(net_local_list)

    lr = args.lr
    results = []

    m = max(int(args.frac * args.num_users), 1)
    I = torch.ones((m, m))
    i = torch.ones((m, 1))
    omega = I - 1 / m * i.mm(i.T)
    omega = omega ** 2
    omega = omega.cuda()

    W = [net_local_list[0].state_dict()[key].flatten() for key in w_glob_keys]
    W = torch.cat(W)
    d = len(W)
    del W

    for iter in range(args.epochs):
        w_glob = {}
        loss_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        W = torch.zeros((d, m)).cuda()
        for idx, user in enumerate(idxs_users):
            W_local = [net_local_list[user].state_dict()[key].flatten() for key in w_glob_keys]
            W_local = torch.cat(W_local)
            W[:, idx] = W_local

        for idx, user in enumerate(idxs_users):
            local = LocalUpdateMTL(args=args, dataset=dataset_train, idxs=dict_users_train[user])
            net_local = net_local_list[user]

            w_local, loss = local.train(net=net_local.to(args.device), lr=lr,
                                        omega=omega, W_glob=W.clone(), idx=idx, w_glob_keys=w_glob_keys)
            loss_locals.append(copy.deepcopy(loss))

        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)

        # eval
        acc_test_local, loss_test_local = test_img_local_all(net_local_list, args, dataset_test, dict_users_test, return_all=True)

        for user in range(args.num_users):
            if acc_test_local[user] > best_acc[user]:
                best_acc[user] = acc_test_local[user]
                best_net_list[user] = copy.deepcopy(net_local_list[user])

                model_save_path = os.path.join(base_save_dir, 'model_user{}.pt'.format(user))
                torch.save(best_net_list[user].state_dict(), model_save_path)

        acc_test_local, loss_test_local = test_img_local_all(best_net_list, args, dataset_test, dict_users_test)
        acc_test_avg, loss_test_avg = test_img_avg_all(net_glob, best_net_list, args, dataset_test)
        print('Round {:3d}, Avg Loss {:.3f}, Loss (local): {:.3f}, Acc (local): {:.2f}, Loss (avg): {:.3}, Acc (avg): {:.2f}'.format(
            iter, loss_avg, loss_test_local, acc_test_local, loss_test_avg, acc_test_avg))

        results.append(np.array([iter, acc_test_local, acc_test_avg, best_acc.mean(), None, None]))
        final_results = np.array(results)
        final_results = pd.DataFrame(final_results, columns=['epoch', 'acc_test_local', 'acc_test_avg', 'best_acc_local', 'acc_test_ens_avg', 'acc_test_ens_maj'])
        final_results.to_csv(results_save_path, index=False)

    acc_test_ens_avg, loss_test, acc_test_ens_maj = test_img_ensemble_all(best_net_list, args, dataset_test)
    print('Best model, acc (local): {}, acc (ens,avg): {}, acc (ens,maj): {}'.format(best_acc, acc_test_ens_avg, acc_test_ens_maj))

    results.append(np.array(['Final', None, None, best_acc.mean(), acc_test_ens_avg, acc_test_ens_maj]))
    final_results = np.array(results)
    final_results = pd.DataFrame(final_results,
                                 columns=['epoch', 'acc_test_local', 'acc_test_avg', 'best_acc_local', 'acc_test_ens_avg', 'acc_test_ens_maj'])
    final_results.to_csv(results_save_path, index=False)