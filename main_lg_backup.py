#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import os
import pickle
import itertools
import pandas as pd
import numpy as np
import torch

from utils.options import args_parser
from utils.train_utils import get_data, get_model
from models.Update import LocalUpdate
from models.test import test_img_local_all, test_img_avg_all, test_img_ensemble_all

import pdb

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    base_dir = './save/{}/{}_iid{}_num{}_C{}_le{}/shard{}/{}/'.format(
        args.dataset, args.model, args.iid, args.num_users, args.frac, args.local_ep, args.shard_per_user, args.results_save)

    assert(len(args.load_fed) > 0)
    base_save_dir = os.path.join(base_dir, 'lg/{}'.format(args.load_fed))
    if not os.path.exists(base_save_dir):
        os.makedirs(base_save_dir, exist_ok=True)

    dataset_train, dataset_test, dict_users_train, dict_users_test = get_data(args)
    dict_save_path = os.path.join(base_dir, 'dict_users.pkl')
    with open(dict_save_path, 'rb') as handle:
        dict_users_train, dict_users_test = pickle.load(handle)

    # build model
    net_glob = get_model(args)
    net_glob.train()

    fed_model_path = os.path.join(base_dir, 'fed/{}'.format(args.load_fed))
    net_glob.load_state_dict(torch.load(fed_model_path))

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

    # training
    results_save_path = os.path.join(base_save_dir, 'results.csv')

    loss_train = []
    net_best = None
    best_loss = None
    best_acc = None
    best_epoch = None

    lr = args.lr
    results = []

    for iter in range(args.epochs):
        w_glob = {}
        loss_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        # w_keys_epoch = net_glob.state_dict().keys() if (iter + 1) % 25 == 0 else w_glob_keys
        w_keys_epoch = w_glob_keys

        if args.verbose:
            print("Round {}: lr: {:.6f}, {}".format(iter, lr, idxs_users))
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users_train[idx])
            net_local = net_local_list[idx]

            w_local, loss = local.train(net=net_local.to(args.device), lr=lr)
            loss_locals.append(copy.deepcopy(loss))

            modules_glob = set([x.split('.')[0] for x in w_keys_epoch])
            modules_all = net_local.__dict__['_modules']

            # sum up weights
            if len(w_glob) == 0:
                w_glob = copy.deepcopy(net_glob.state_dict())
            else:
                for k in w_keys_epoch:
                    w_glob[k] += w_local[k]

        if (iter+1) % int(args.num_users * args.frac):
            lr *= args.lr_decay

        # get weighted average for global weights
        for k in w_keys_epoch:
            w_glob[k] = torch.div(w_glob[k], m)

        # copy weight to the global model (not really necessary)
        net_glob.load_state_dict(w_glob)

        # copy weights to each local model
        for idx in range(args.num_users):
            net_local = net_local_list[idx]
            w_local = net_local.state_dict()
            for k in w_keys_epoch:
                w_local[k] = w_glob[k]

            net_local.load_state_dict(w_local)

        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)

        # eval
        acc_test_local, loss_test_local = test_img_local_all(net_local_list, args, dataset_test, dict_users_test)
        acc_test_avg, loss_test_avg = test_img_avg_all(net_glob, net_local_list, args, dataset_test)
        print('Round {:3d}, Avg Loss {:.3f}, Loss (local): {:.3f}, Acc (local): {:.2f}, Loss (avg): {:.3}, Acc (avg): {:.2f}'.format(
            iter, loss_avg, loss_test_local, acc_test_local, loss_test_avg, acc_test_avg))

        if best_acc is None or acc_test_local > best_acc:
            best_acc = acc_test_local
            best_epoch = iter

            for user in range(args.num_users):
                model_save_path = os.path.join(base_save_dir, 'model_user{}.pt'.format(user))
                torch.save(net_local_list[user].state_dict(), model_save_path)

        results.append(np.array([iter, acc_test_local, acc_test_avg, best_acc, None, None]))
        final_results = np.array(results)
        final_results = pd.DataFrame(final_results, columns=['epoch', 'acc_test_local', 'acc_test_avg', 'best_acc_local', 'acc_test_ens_avg', 'acc_test_ens_maj'])
        final_results.to_csv(results_save_path, index=False)

    for user in range(args.num_users):
        model_save_path = os.path.join(base_save_dir, 'model_user{}.pt'.format(user))

        net_local = net_local_list[user]
        net_local.load_state_dict(torch.load(model_save_path))
    acc_test_ens_avg, loss_test, acc_test_ens_maj = test_img_ensemble_all(net_local_list, args, dataset_test)

    print('Best model, iter: {}, acc (local): {}, acc (ens,avg): {}, acc (ens,maj): {}'.format(best_epoch, best_acc, acc_test_ens_avg, acc_test_ens_maj))

    results.append(np.array(['Final', None, None, best_acc, acc_test_ens_avg, acc_test_ens_maj]))
    final_results = np.array(results)
    final_results = pd.DataFrame(final_results,
                                 columns=['epoch', 'acc_test_local', 'acc_test_avg', 'best_acc_local', 'acc_test_ens_avg', 'acc_test_ens_maj'])
    final_results.to_csv(results_save_path, index=False)
