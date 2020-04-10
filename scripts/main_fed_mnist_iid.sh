#!/bin/bash

NUM_USERS=100
while getopts s:n: option; do
  case "${option}" in
    n) NUM_USERS=${OPTARG};;
  esac
done

for RUN in 1 2 3 4 5; do
  python3 main_fed.py --dataset mnist --model mlp --num_classes 10 --epochs 1000 --lr 0.05 \
  --num_users ${NUM_USERS} --shard_per_user 10 --frac 0.1 --local_ep 1 --local_bs 10 --results_save run${RUN} --iid
  python3 main_local.py --dataset mnist --model mlp --num_classes 10 --epochs 400 --lr 0.05 \
  --num_users ${NUM_USERS} --shard_per_user 10 --frac 0.1 --local_ep 1 --local_bs 10 --results_save run${RUN} --iid
done

