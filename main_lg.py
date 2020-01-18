#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import os
import itertools
import numpy as np
from scipy.stats import mode
from torchvision import datasets, transforms, models
import torch
from torch import nn

from utils.sampling import mnist_iid, mnist_noniid, cifar10_iid, cifar10_noniid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar, ResnetCifar
from models.Fed import FedAvg
from models.test import test_img, test_img_local

import pdb

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    trans_mnist = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))])
    if args.model == 'resnet':
        trans_cifar_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.Resize([256,256]),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                     std=[0.229, 0.224, 0.225])])
        trans_cifar_val = transforms.Compose([transforms.Resize([256,256]),
                                                transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                   std=[0.229, 0.224, 0.225])])
    else:
        trans_cifar_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                     std=[0.229, 0.224, 0.225])])
        trans_cifar_val = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                   std=[0.229, 0.224, 0.225])])

    # load dataset and split users
    if args.dataset == 'mnist':
        dataset_train = datasets.MNIST('data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users_train = mnist_iid(dataset_train, args.num_users)
            dict_users_test = mnist_iid(dataset_test, args.num_users)
        else:
            dict_users_train, rand_set_all = mnist_noniid(dataset_train, args.num_users, num_shards=200, num_imgs=300, train=True)
            dict_users_test, _ = mnist_noniid(dataset_test, args.num_users, num_shards=200, num_imgs=50, train=False, rand_set_all=rand_set_all)

    elif args.dataset == 'cifar10':
        dataset_train = datasets.CIFAR10('data/cifar10', train=True, download=True, transform=trans_cifar_train)
        dataset_test = datasets.CIFAR10('data/cifar10', train=False, download=True, transform=trans_cifar_val)
        if args.iid:
            dict_users_train = cifar10_iid(dataset_train, args.num_users)
            dict_users_test = cifar10_iid(dataset_test, args.num_users)
        else:
            dict_users_train, rand_set_all = cifar10_noniid(dataset_train, args.num_users, num_shards=200, num_imgs=250, train=True)
            dict_users_test, _ = cifar10_noniid(dataset_test, args.num_users, num_shards=200, num_imgs=50, train=False, rand_set_all=rand_set_all)

    elif args.dataset == 'cifar100':
        dataset_train = datasets.CIFAR100('data/cifar100', train=True, download=True, transform=trans_cifar_train)
        dataset_test = datasets.CIFAR100('data/cifar100', train=False, download=True, transform=trans_cifar_val)
        if args.iid:
            dict_users_train = cifar10_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')

    import pdb; pdb.set_trace()

    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset in ['cifar10', 'cifar100']:
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'resnet' and args.dataset in ['cifar10', 'cifar100']:
        net_glob = ResnetCifar(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')

    print(net_glob)
    net_glob.train()
    if args.load_fed:
        fed_model_path = './save/keep/fed_{}_{}_iid{}_num{}_C{}_le{}_gn{}.npy'.format(
            args.dataset, args.model, args.iid, args.num_users, args.frac, args.local_ep, args.grad_norm)
        if len(args.load_fed_name) > 0:
            fed_model_path = './save/keep/{}'.format(args.load_fed_name)
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
            _, _, probs = test_img(net_local, dataset_test, args, return_probs=True, user_idx=idx)
            probs_all.append(probs.detach())

            preds = probs.data.max(1, keepdim=True)[1].cpu().numpy().reshape(-1)
            preds_all.append(preds)

        labels = np.array(dataset_test.test_labels)
        preds_probs = torch.mean(torch.stack(probs_all), dim=0)

        # ensemble metrics
        preds_avg = preds_probs.data.max(1, keepdim=True)[1].cpu().numpy().reshape(-1)
        loss_test = criterion(preds_probs, torch.tensor(labels).cuda()).item()
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

    if args.local_ep_pretrain > 0:
        # pretrain each local model
        pretrain_save_path = 'pretrain/{}/{}_{}/user_{}/ep_{}/'.format(args.model, args.dataset,
                                                                       'iid' if args.iid else 'noniid', args.num_users,
                                                                       args.local_ep_pretrain)
        if not os.path.exists(pretrain_save_path):
            os.makedirs(pretrain_save_path)

        print("\nPretraining local models...")
        for idx in range(args.num_users):
            net_local = net_local_list[idx]
            net_local_path = os.path.join(pretrain_save_path, '{}.pt'.format(idx))
            if os.path.exists(net_local_path): # check if we have a saved model
                net_local.load_state_dict(torch.load(net_local_path))
            else:
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users_train[idx], pretrain=True)
                w_local, loss = local.train(net=net_local.to(args.device))
                print('Local model {}, Train Epoch Loss: {:.4f}'.format(idx, loss))
                torch.save(net_local.state_dict(), net_local_path)

        print("Getting initial loss and acc...")
        acc_test_local, loss_test_local = test_img_local_all()
        acc_test_avg, loss_test_avg =  test_img_avg_all()
        loss_test, acc_test = test_img_ensemble_all()

        print('Initial Ensemble: Loss (local): {:.3f}, Acc (local): {:.2f}, Loss (avg): {:.3}, Acc (avg): {:.2f}, Loss (ens) {:.3f}, Acc: (ens) {:.2f}, '.format(
            loss_test_local, acc_test_local, loss_test_avg, acc_test_avg, loss_test, acc_test))

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    lr = args.lr
    results = []

    for iter in range(args.epochs):
        w_glob = {}
        loss_locals, grads_local = [], []
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

            # use grads to calculate a weighted average
            if not args.grad_norm:
                grads = 1.0
            else:
                grads = []
                for key in modules_glob:
                    module = modules_all[key]
                    grad = module.weight.grad
                    if grad is not None:
                        grads.append(grad.view(-1))

                    try:
                        grad = module.bias.grad
                        if grad is not None:
                            grads.append(grad.view(-1))
                    except:
                        pass
                grads = torch.cat(grads).norm().item()
            # print(grads)
            grads_local.append(grads)

            # sum up weights
            if len(w_glob) == 0:
                w_glob = copy.deepcopy(net_glob.state_dict())
                for k in w_keys_epoch: # this depends on the layers being named the same (in Nets.py)
                    w_glob[k] = w_local[k] * grads
            else:
                for k in w_keys_epoch:
                    w_glob[k] += w_local[k] * grads

        if (iter+1) % int(args.num_users * args.frac):
            lr *= args.lr_decay

        # get weighted average for global weights
        for k in w_keys_epoch:
            w_glob[k] = torch.div(w_glob[k], sum(grads_local))

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

        if (iter + 1) % args.test_freq == 0: # this takes too much time, so we run it less frequently
            loss_test, acc_test = test_img_ensemble_all()
            print('Round {:3d}, Avg Loss {:.3f}, Loss (local): {:.3f}, Acc (local): {:.2f}, Loss (avg): {:.3}, Acc (avg): {:.2f}, Loss (ens) {:.3f}, Acc: (ens) {:.2f}, '.format(
                iter, loss_avg, loss_test_local, acc_test_local, loss_test_avg, acc_test_avg, loss_test, acc_test))
            results.append(np.array([iter, loss_avg, loss_test_local, acc_test_local, loss_test_avg, acc_test_avg, loss_test, acc_test]))

        else:
            print('Round {:3d}, Avg Loss {:.3f}, Loss (local): {:.3f}, Acc (local): {:.2f}, Loss (avg): {:.3}, Acc (avg): {:.2f}'.format(
                iter, loss_avg, loss_test_local, acc_test_local, loss_test_avg, acc_test_avg))
            results.append(np.array([iter, loss_avg, loss_test_local, acc_test_local, loss_test_avg, acc_test_avg, np.nan, np.nan]))

        final_results = np.array(results)
        results_save_path = './log/lg_{}_{}_keep{}_iid{}_num{}_C{}_le{}_gn{}_pt{}_load{}_tfreq{}.npy'.format(
            args.dataset, args.model, args.num_layers_keep, args.iid, args.num_users, args.frac,
            args.local_ep, args.grad_norm, args.local_ep_pretrain, args.load_fed, args.test_freq)
        np.save(results_save_path, final_results)

    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./log/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))

    # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))