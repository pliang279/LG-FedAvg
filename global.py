#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import os
import itertools
import numpy as np
from scipy.stats import mode
from torchvision import datasets, transforms, models
import torch
from torch import nn
import torch.optim as optim
from utils.sampling import fair_iid, fair_noniid
from utils.options import args_parser
from models.Update import LocalUpdate, LocalUpdate_noLG
from models.Nets import MLP, CNNMnist, CNNCifar, ResnetCifar
from models.Fed import FedAvg
from models.test import test_img, test_img_local

import pandas as pd

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from helpers import load_ICU_data, plot_distributions, _performance_text

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import pdb

def run_all(clf_all1, clf_all2, adv_all1, adv_all2, adv_all3):
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load ICU dataset and split users
    # load ICU data set
    X, y, Z = load_ICU_data('../fairness-in-ml/data/adult.data')

    if not args.iid:
        X = X[:30000]
        y = y[:30000]
        Z = Z[:30000]

    n_points = X.shape[0]
    n_features = X.shape[1]
    n_sensitive = Z.shape[1]

    print (n_features)

    # split into train/test set
    (X_train, X_test, y_train, y_test, Z_train, Z_test) = train_test_split(X, y, Z, test_size=0.5, stratify=y, random_state=7)

    # standardize the data
    scaler = StandardScaler().fit(X_train)
    scale_df = lambda df, scaler: pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)
    X_train = X_train.pipe(scale_df, scaler) 
    X_test = X_test.pipe(scale_df, scaler)

    class PandasDataSet(TensorDataset):
        def __init__(self, *dataframes):
            tensors = (self._df_to_tensor(df) for df in dataframes)
            super(PandasDataSet, self).__init__(*tensors)

        def _df_to_tensor(self, df):
            if isinstance(df, pd.Series):
                df = df.to_frame('dummy')
            return torch.from_numpy(df.values).float()

    def _df_to_tensor(df):
        if isinstance(df, pd.Series):
            df = df.to_frame('dummy')
        return torch.from_numpy(df.values).float()

    train_data = PandasDataSet(X_train, y_train, Z_train)
    test_data = PandasDataSet(X_test, y_test, Z_test)
    
    print('# train samples:', len(train_data))      # 15470
    print('# test samples:', len(test_data))

    batch_size = 32

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=True, drop_last=True)

    # sample users
    if args.iid:
        dict_users_train = fair_iid(train_data, args.num_users)
        dict_users_test = fair_iid(test_data, args.num_users)
    else:
        train_data = [_df_to_tensor(X_train), _df_to_tensor(y_train), _df_to_tensor(Z_train)]
        test_data = [_df_to_tensor(X_test), _df_to_tensor(y_test), _df_to_tensor(Z_test)]
        #import pdb; pdb.set_trace()
        dict_users_train, rand_set_all = fair_noniid(train_data, args.num_users, num_shards=100, num_imgs=150, train=True)
        dict_users_test, _ = fair_noniid(test_data, args.num_users, num_shards=100, num_imgs=150, train=False, rand_set_all=rand_set_all)

    train_data = [_df_to_tensor(X_train), _df_to_tensor(y_train), _df_to_tensor(Z_train)]
    test_data = [_df_to_tensor(X_test), _df_to_tensor(y_test), _df_to_tensor(Z_test)]

    class LocalClassifier(nn.Module):
        def __init__(self, n_features, n_hidden=32, p_dropout=0.2):
            super(LocalClassifier, self).__init__()
            self.network1 = nn.Sequential(
                nn.Linear(n_features, n_hidden),
                nn.ReLU(),
                nn.Dropout(p_dropout),
                nn.Linear(n_hidden, n_hidden),
                nn.ReLU(),
                nn.Dropout(p_dropout),
                nn.Linear(n_hidden, n_hidden)
            )
            self.network2 = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(p_dropout),
                nn.Linear(n_hidden, 1)
            )

        def forward(self, x):
            mid = self.network1(x)
            final = torch.sigmoid(self.network2(mid))
            return mid, final

    def pretrain_classifier(clf, data_loader, optimizer, criterion):
        losses = 0.0
        for x, y, _ in data_loader:
            x = x.to(args.device)
            y = y.to(args.device)
            clf.zero_grad()
            mid, p_y = clf(x)
            loss = criterion(p_y, y)
            loss.backward()
            optimizer.step()
            losses += loss.item()
        print ('loss', losses/len(data_loader))
        return clf

    def test_classifier(clf, data_loader):
        losses = 0
        assert len(data_loader) == 1
        with torch.no_grad():
            for x, y_test, _ in data_loader:
                x = x.to(args.device)
                mid, y_pred = clf(x)
                y_pred = y_pred.cpu()
                clf_accuracy = metrics.accuracy_score(y_test, y_pred > 0.5) * 100
        return clf_accuracy

    class Adversary(nn.Module):

        def __init__(self, n_sensitive, n_hidden=32):
            super(Adversary, self).__init__()
            self.network = nn.Sequential(
                nn.Linear(n_hidden, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_sensitive),
            )

        def forward(self, x):
            return torch.sigmoid(self.network(x))

    def pretrain_adversary(adv, clf, data_loader, optimizer, criterion):
        losses = 0.0
        for x, _, z in data_loader:
            x = x.to(args.device)
            z = z.to(args.device)
            mid, p_y = clf(x)
            mid = mid.detach()
            p_y = p_y.detach()
            adv.zero_grad()
            p_z = adv(mid)
            loss = (criterion(p_z.to(args.device), z.to(args.device)) * lambdas.to(args.device)).mean()
            loss.backward()
            optimizer.step()
            losses += loss.item()
        print ('loss', losses/len(data_loader))
        return adv

    def test_adversary(adv, clf, data_loader):
        losses = 0
        adv_accuracies = []
        assert len(data_loader) == 1
        with torch.no_grad():
            for x, _, z_test in data_loader:
                x = x.to(args.device)
                mid, p_y = clf(x)
                mid = mid.detach()
                p_y = p_y.detach()
                p_z = adv(mid)
                for i in range(p_z.shape[1]):
                    z_test_i = z_test[:,i]
                    z_pred_i = p_z[:,i]
                    z_pred_i = z_pred_i.cpu()
                    adv_accuracy = metrics.accuracy_score(z_test_i, z_pred_i > 0.5) * 100
                    adv_accuracies.append(adv_accuracy)
        return adv_accuracies

    def train_both(clf, adv, data_loader, clf_criterion, adv_criterion, clf_optimizer, adv_optimizer, lambdas):
        # Train adversary
        adv_losses = 0.0
        for x, y, z in data_loader:
            x = x.to(args.device)
            z = z.to(args.device)
            local, p_y = clf(x)
            adv.zero_grad()
            p_z = adv(local)
            loss_adv = (adv_criterion(p_z.to(args.device), z.to(args.device)) * lambdas.to(args.device)).mean()
            loss_adv.backward()
            adv_optimizer.step()
            adv_losses += loss_adv.item()
        print ('adversarial loss', adv_losses/len(data_loader))

        # Train classifier on single batch
        clf_losses = 0.0
        for x, y, z in data_loader:
            x = x.to(args.device)
            y = y.to(args.device)
            z = z.to(args.device)
            local, p_y = clf(x)
            p_z = adv(local)
            clf.zero_grad()
            if args.adv:
                clf_loss = clf_criterion(p_y.to(args.device), y.to(args.device)) - (adv_criterion(p_z.to(args.device), z.to(args.device)) * lambdas.to(args.device)).mean()
            else:
                clf_loss = clf_criterion(p_y.to(args.device), y.to(args.device))
            clf_loss.backward()
            clf_optimizer.step()
            clf_losses += clf_loss.item()
        print ('classifier loss', clf_losses/len(data_loader))
        return clf, adv

    def eval_global_performance_text(test_loader_i, global_model, adv_model):
        with torch.no_grad():
            for test_x, test_y, test_z in test_loader_i:
                test_x = test_x.to(args.device)
                local_pred, clf_pred = global_model(test_x)
                adv_pred = adv_model(local_pred)

            y_post_clf = pd.Series(clf_pred.cpu().numpy().ravel(), index=y_test[list(dict_users_train[idx])].index)
            Z_post_adv = pd.DataFrame(adv_pred.cpu().numpy(), columns=Z_test.columns)

            clf_roc_auc,clf_accuracy,adv_acc1,adv_acc2,adv_roc_auc = _performance_text(test_y, test_z, y_post_clf, Z_post_adv, epoch=None)
        return clf_roc_auc,clf_accuracy,adv_acc1,adv_acc2,adv_roc_auc

    lambdas = torch.Tensor([30.0, 30.0])
    net_local_list = []

    print ('\n\n======================== STARTING LOCAL TRAINING ========================\n\n\n')

    for idx in range(args.num_users):
        train_data_i_raw = [torch.FloatTensor(bb[list(dict_users_train[idx])]) for bb in train_data]
        train_data_i = TensorDataset(train_data_i_raw[0],train_data_i_raw[1],train_data_i_raw[2])
        train_loader_i = torch.utils.data.DataLoader(train_data_i, batch_size=batch_size, shuffle=False, num_workers=4)

        test_data_i_raw = [torch.FloatTensor(bb[list(dict_users_train[idx])]) for bb in test_data]
        test_data_i = TensorDataset(test_data_i_raw[0],test_data_i_raw[1],test_data_i_raw[2])
        test_loader_i = torch.utils.data.DataLoader(test_data_i, batch_size=len(test_data_i), shuffle=False, num_workers=4)

        net_local_list.append([train_loader_i,test_loader_i])

    class GlobalClassifier(nn.Module):
        def __init__(self, n_features, n_hidden=32, p_dropout=0.2):
            super(GlobalClassifier, self).__init__()
            self.network1 = nn.Sequential(
                nn.Linear(n_features, n_hidden),
                nn.ReLU(),
                nn.Dropout(p_dropout),
                nn.Linear(n_hidden, n_hidden),
                nn.ReLU(),
                nn.Dropout(p_dropout),
                nn.Linear(n_hidden, n_hidden)
            )
            self.network2 = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(p_dropout),
                nn.Linear(n_hidden, 1)
            )

        def forward(self, x):
            mid = self.network1(x)
            final = torch.sigmoid(self.network2(mid))
            return mid, final

    # build global model
    global_clf = GlobalClassifier(n_features=n_features).to(args.device)
    global_clf_criterion = nn.BCELoss().to(args.device)
    global_clf_optimizer = optim.Adam(global_clf.parameters(), lr=0.01)

    adv_model = Adversary(Z_train.shape[1]).to(args.device)
    adv_criterion = nn.BCELoss(reduce=False).to(args.device)
    adv_optimizer = optim.Adam(adv_model.parameters(), lr=0.01)

    # copy weights
    w_glob = global_clf.state_dict()
    adv_glob = adv_model.state_dict()

    print ('\n\n======================== STARTING GLOBAL TRAINING ========================\n\n\n')

    global_epochs = 10
    for iter in range(global_epochs):
        w_locals, adv_locals, w_loss_locals, adv_loss_locals = [], [], [], []
        for idx in range(args.num_users):
            print ('\n\n======================== GLOBAL TRAINING, ITERATION %d, USER %d ========================\n\n\n' %(iter,idx))
            train_loader_i,test_loader_i = net_local_list[idx]

            local = LocalUpdate_noLG(args=args, dataset=train_loader_i)
            w, w_loss, adv, adv_loss = local.train(global_net=copy.deepcopy(global_clf).to(args.device), adv_model=copy.deepcopy(adv_model).to(args.device), lambdas=lambdas)

            w_locals.append(copy.deepcopy(w))
            w_loss_locals.append(copy.deepcopy(w_loss))

            adv_locals.append(copy.deepcopy(adv))
            adv_loss_locals.append(copy.deepcopy(adv_loss))

        w_glob = FedAvg(w_locals)
        # copy weight to net_glob
        global_clf.load_state_dict(w_glob)

        adv_glob = FedAvg(adv_locals)
        # copy weight to net_glob
        adv_model.load_state_dict(adv_glob)

        for idx in range(args.num_users):
            train_loader_i,test_loader_i = net_local_list[idx]

            print ('======================== local and global training: evaluating _global_performance_text on device %d ========================' %idx)
            clf_roc_auc,clf_accuracy,adv_acc1,adv_acc2,adv_roc_auc = eval_global_performance_text(test_loader_i, global_clf, adv_model)
            print ('======================== by now the global classifier should work better than local classifier ========================')

        clf_all1.append(clf_roc_auc)
        clf_all2.append(clf_accuracy)
        adv_all1.append(adv_acc1)
        adv_all2.append(adv_acc2)
        adv_all3.append(adv_roc_auc)

    print ('clf_all1', np.mean(np.array(clf_all1)), np.std(np.array(clf_all1)))
    print ('clf_all2', np.mean(np.array(clf_all2)), np.std(np.array(clf_all2)))
    print ('adv_all1', np.mean(np.array(adv_all1)), np.std(np.array(adv_all1)))
    print ('adv_all2', np.mean(np.array(adv_all2)), np.std(np.array(adv_all2)))
    print ('adv_all3', np.mean(np.array(adv_all3)), np.std(np.array(adv_all3)))
    return clf_all1, clf_all2, adv_all1, adv_all2, adv_all3


if __name__ == '__main__':
    clf_all1, clf_all2, adv_all1, adv_all2, adv_all3 = [], [], [], [], []
    for _ in range(10):
        clf_all1, clf_all2, adv_all1, adv_all2, adv_all3 = run_all(clf_all1, clf_all2, adv_all1, adv_all2, adv_all3)
    print ('final')
    print ('clf_all1', np.mean(np.array(clf_all1)), np.std(np.array(clf_all1)))
    print ('clf_all2', np.mean(np.array(clf_all2)), np.std(np.array(clf_all2)))
    print ('adv_all1', np.mean(np.array(adv_all1)), np.std(np.array(adv_all1)))
    print ('adv_all2', np.mean(np.array(adv_all2)), np.std(np.array(adv_all2)))
    print ('adv_all3', np.mean(np.array(adv_all3)), np.std(np.array(adv_all3)))

