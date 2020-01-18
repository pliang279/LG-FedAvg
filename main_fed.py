#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms, models
import torch

from utils.sampling import mnist_iid, mnist_noniid, cifar10_iid, cifar10_noniid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar, ResnetCifar
from models.Fed import FedAvg
from models.test import test_img

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
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users, _ = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar10':
        dataset_train = datasets.CIFAR10('data/cifar10', train=True, download=True, transform=trans_cifar_train)
        dataset_test = datasets.CIFAR10('data/cifar10', train=False, download=True, transform=trans_cifar_val)
        if args.iid:
            dict_users = cifar10_iid(dataset_train, args.num_users)
        else:
            dict_users, _ = cifar10_noniid(dataset_train, args.num_users)
            # exit('Error: only consider IID setting in CIFAR10')
    elif args.dataset == 'cifar100':
        dataset_train = datasets.CIFAR100('data/cifar100', train=True, download=True, transform=trans_cifar_train)
        dataset_test = datasets.CIFAR100('data/cifar100', train=False, download=True, transform=trans_cifar_val)
        if args.iid:
            dict_users = cifar10_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')
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
        net_glob = MLP(dim_in=len_in, dim_hidden=256, dim_out=args.num_classes).to(args.device)
    elif args.model == 'mlp_orig':
            len_in = 1
            for x in img_size:
                len_in *= x
            net_glob = MLP(dim_in=len_in, dim_hidden=256, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()

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
        w_glob = None
        loss_locals, grads_local = [], []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        print("Round {}, lr: {:.6f}, {}".format(iter, lr, idxs_users))

        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            net_local = copy.deepcopy(net_glob)

            w_local, loss = local.train(net=net_local.to(args.device))
            loss_locals.append(copy.deepcopy(loss))

            if not args.grad_norm:
                grads = 1.0
            else:
                grads = []
                for grad in [param.grad for param in net_local.parameters()]:
                    if grad is not None:
                        grads.append(grad.view(-1))
                grads = torch.cat(grads).norm().item()
            # print(grads)
            grads_local.append(grads)

            if w_glob is None:
                w_glob = copy.deepcopy(w_local)
                for k in w_glob.keys():
                    w_glob[k] *= grads
            else:
                for k in w_glob.keys():
                    w_glob[k] += w_local[k] * grads

        lr *= args.lr_decay

        # update global weights
        for k in w_glob.keys():
            w_glob[k] = torch.div(w_glob[k], sum(grads_local))

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)

        if (iter + 1) % 1 == 0:
            net_glob.eval()
            acc_test, loss_test = test_img(net_glob, dataset_test, args)
            print('Round {:3d}, Average loss {:.3f}, Test loss {:.3f}, Test accuracy: {:.2f}'.format(
                iter, loss_avg, loss_test, acc_test))

            results.append(np.array([iter, loss_avg, loss_test, acc_test]))
            final_results = np.array(results)

            results_save_path = './log/fed_{}_{}_iid{}_num{}_C{}_le{}_gn{}.npy'.format(
                args.dataset, args.model, args.iid, args.num_users, args.frac, args.local_ep, args.grad_norm)
            np.save(results_save_path, final_results)

            model_save_path = './save/fed_{}_{}_iid{}_num{}_C{}_le{}_gn{}.pt'.format(
                args.dataset, args.model, args.iid, args.num_users, args.frac, args.local_ep, args.grad_norm)
            if best_loss is None or loss_test <  best_loss:
                best_loss = loss_test
                torch.save(net_glob.state_dict(), model_save_path)

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