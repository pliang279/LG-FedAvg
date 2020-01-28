# FedAvg

Results can be reproduced using the following:

#### MNIST 
> python3 main_fed.py --dataset mnist --model mlp --num_classes 10 --epochs 1500 --lr 0.05 --num_users 100 --frac 0.1 --local_ep 1 --local_bs 10

#### CIFAR10 
> python3 main_fed.py --dataset cifar10 --model cnn --num_classes 10 --epochs 2000 --lr 0.1--num_users 100 --frac 0.1 --local_ep 1 --local_bs 50

# LG-FedAvg

Results can be reproduced using the following:

#### MNIST
> python3 main_lg.py --dataset mnist --model mlp --num_classes 10 --epochs 1500 --lr 0.05 --num_users 100 --frac 0.1 --local_ep 1 --local_bs 10 --num_layers_keep 3

#### CIFAR10 
> python3 main_lg.py --dataset cifar10 --model cnn --num_classes 10 --epochs 2000 --lr 0.1 --num_users 100 --frac 0.1 --local_ep 1 --local_bs 50 --num_layers_keep 2
