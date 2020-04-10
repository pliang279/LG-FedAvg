#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import os
import itertools
import numpy as np
import torch
from torch import nn

from utils.options import args_parser
from utils.train_utils import get_data, get_model
from models.Update import LocalUpdate
from models.test import test_img, test_img_local

import pdb

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    base_dir = './save/{}/{}_iid{}_num{}_C{}_le{}/{}/'.format(
        args.dataset, args.model, args.iid, args.num_users, args.frac, args.local_ep, args.results_save)
    if not os.path.exists(os.path.join(base_dir, 'lg')):
        os.makedirs(os.path.join(base_dir, 'lg'), exist_ok=True)

    rand_set_all = []
    if len(args.load_fed) > 0:
        rand_save_path = os.path.join(base_dir, 'randset.npy')
        rand_set_all= np.load(rand_save_path)

    dataset_train, dataset_test, dict_users_train, dict_users_test, rand_set_all = get_data(args, rand_set_all=rand_set_all)
    rand_save_path = os.path.join(base_dir, 'randset.npy')
    np.save(rand_save_path, rand_set_all)

    # build model
    net_glob = get_model(args)
    net_glob.train()

    if len(args.load_fed) > 0:
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

    criterion = nn.CrossEntropyLoss()


    def test_img_ensemble_all():
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

        # ensemble metrics
        preds_avg = preds_probs.data.max(1, keepdim=True)[1].cpu().numpy().reshape(-1)
        loss_test = criterion(preds_probs, torch.tensor(labels).to(args.device)).item()
        acc_test = (preds_avg == labels).mean() * 100

        return loss_test, acc_test

    def test_img_local_all():
        acc_test_local = 0
        loss_test_local = 0
        for idx in range(args.num_users):
            net_local = net_local_list[idx]
            net_local.eval()
            a, b = test_img_local(net_local, dataset_test, args, user_idx=idx, idxs=dict_users_test[idx])

            acc_test_local += a
            loss_test_local += b
        acc_test_local /= args.num_users
        loss_test_local /= args.num_users

        return acc_test_local, loss_test_local

    def test_img_avg_all():
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

        return acc_test_avg, loss_test_avg

    # training
    results_save_path = os.path.join(base_dir, 'lg/results.npy')

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
        acc_test_local, loss_test_local = test_img_local_all()
        acc_test_avg, loss_test_avg = test_img_avg_all()

        print('Round {:3d}, Avg Loss {:.3f}, Loss (local): {:.3f}, Acc (local): {:.2f}, Loss (avg): {:.3}, Acc (avg): {:.2f}'.format(
            iter, loss_avg, loss_test_local, acc_test_local, loss_test_avg, acc_test_avg))
        results.append(np.array([iter, loss_avg, loss_test_local, acc_test_local, loss_test_avg, acc_test_avg]))

        final_results = np.array(results)
        np.save(results_save_path, final_results)

        if best_acc is None or acc_test_local > best_acc:
            best_acc = acc_test_local
            best_epoch = iter

            for user in range(args.num_users):
                model_save_path = os.path.join(base_dir, 'lg/model_user{}.pt'.format(user))
                torch.save(net_local_list[user].state_dict(), model_save_path)

    for user in range(args.num_users):
        model_save_path = os.path.join(base_dir, 'lg/model_user{}.pt'.format(user))

        net_local = net_local_list[idx]
        net_local.load_state_dict(torch.load(model_save_path))
    loss_test, acc_test = test_img_ensemble_all()

    print('Best model, iter: {}, acc: {}, acc (ens): {}'.format(best_epoch, best_acc, acc_test))