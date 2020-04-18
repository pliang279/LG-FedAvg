#!/bin/bash

NUM_USERS=100
SHARDS=2
while getopts s:n: option; do
  case "${option}" in
    s) SHARDS=${OPTARG};;
    n) NUM_USERS=${OPTARG};;
  esac
done

for RUN in 1 2 3 4 5; do
  python3 main_mtl.py --dataset cifar10 --model cnn --num_classes 10 --epochs 1800 --lr 0.1 \
  --num_users ${NUM_USERS} --shard_per_user ${SHARDS} --frac 0.1 --local_ep 1 --local_bs 50 --num_layers_keep 5 \
  --results_save run${RUN}
done

