# Federated Learning with Local and Global Representations

> Pytorch implementation for federated learning with local and global representations.

Correspondence to: 
  - Paul Liang (pliang@cs.cmu.edu)
  - Terrance Liu (terrancl@cs.cmu.edu)
  
## Paper

[**Think Locally, Act Globally: Federated Learning with Local and Global Representations**](https://arxiv.org/abs/2001.01523)<br>
[Paul Pu Liang*](http://www.cs.cmu.edu/~pliang/), Terrance Liu*, [Liu Ziyin](http://cat.phys.s.u-tokyo.ac.jp/~zliu/), [Louis-Philippe Morency](https://www.cs.cmu.edu/~morency/), and [Ruslan Salakhutdinov](https://www.cs.cmu.edu/~rsalakhu/)<br>
NeurIPS 2019 Workshop on Federated Learning (distinguished student paper award). (*equal contribution)

## Installation

First check that the requirements are satisfied:</br>
Python 3.6</br>
torch 1.2.0</br>
torchvision 0.4.0</br>
numpy 1.18.1</br>
sklearn 0.20.0</br>
matplotlib 3.1.2</br>
Pillow 4.1.1

The next step is to clone the repository:
```bash
git clone https://github.com/pliang279/LG-FedAvg.git
```

## Data

We run FedAvg and LG-FedAvg experiments on MNIST ([link](http://yann.lecun.com/exdb/mnist/)) and CIFAR10 ([link](https://www.cs.toronto.edu/~kriz/cifar.html)). See our paper for a description how we process and partition the data for federated learning experiments.


## LG-FedAvg

Results can be reproduced using the following:

#### MNIST
> python main_lg.py --dataset mnist --model mlp --num_classes 10 --epochs 1500 --lr 0.05 --num_users 100 --frac 0.1 --local_ep 1 --local_bs 10 --num_layers_keep 3

#### CIFAR10 
> python main_lg.py --dataset cifar10 --model cnn --num_classes 10 --epochs 2000 --lr 0.1 --num_users 100 --frac 0.1 --local_ep 1 --local_bs 50 --num_layers_keep 2


## FedAvg

Results can be reproduced using the following:

#### MNIST 
> python main_fed.py --dataset mnist --model mlp --num_classes 10 --epochs 1500 --lr 0.05 --num_users 100 --frac 0.1 --local_ep 1 --local_bs 10

#### CIFAR10 
> python main_fed.py --dataset cifar10 --model cnn --num_classes 10 --epochs 2000 --lr 0.1 --num_users 100 --frac 0.1 --local_ep 1 --local_bs 50


## MTL

Results can be reproduced using the following:

#### MNIST 
> python main_mtl.py --dataset mnist --model mlp --num_classes 10 --epochs 500 --lr 0.05 --num_users 100 --frac 1 --local_ep 1 --local_bs 10 --num_layers_keep 5

#### CIFAR10 
> python main_mtl.py --dataset cifar10 --model cnn --num_classes 10 --epochs 1500 --lr 0.1 --num_users 100 --frac 1 --local_ep 1 --local_bs 50 --num_layers_keep 5


If you use this code, please cite our paper:

```bash
@article{liang2019_federated,
  title={Think Locally, Act Globally: Federated Learning with Local and Global Representations},
  author={Paul Pu Liang and Terrance Liu and Ziyin Liu and Ruslan Salakhutdinov and Louis-Philippe Morency},
  journal={ArXiv},
  year={2019},
  volume={abs/2001.01523}
}
```

# Acknowledgements

This codebase was adapted from https://github.com/shaoxiongji/federated-learning.
